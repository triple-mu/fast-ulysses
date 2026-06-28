"""Varlen (uneven splits) A2A benchmark: TMA vs non-TMA vs torch all_to_all(list).

mode0: per-rank sequence lengths differ (seq_lens), heads split by head_splits.
Sizes tunable via env: PROF_SCALE (seq multiplier, default 512), PROF_D (head_dim, default 128).
Run: CUSTOM_ULYSSES_USE_TMA=0|1 torchrun --nproc_per_node=8 bench_varlen.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from custom_ulysses_op import UlyssesGroup


def _prefix(xs: list[int]) -> list[int]:
    # Exclusive prefix sum: offsets into the concatenated dimension.
    out = [0]
    for x in xs:
        out.append(out[-1] + x)
    return out


def torch_varlen_mode0(
        x: torch.Tensor,
        me: int,
        ws: int,
        s_off: list[int],
        n_off: list[int],
        group,
) -> torch.Tensor:
    # Variable-length mode0 oracle via torch.distributed.all_to_all (list form).
    b, s_me, N, d = x.shape
    send = [x[:, :, n_off[p]: n_off[p + 1], :].contiguous() for p in range(ws)]
    recv = [
        torch.empty(
            b,
            s_off[p + 1] - s_off[p],
            n_off[me + 1] - n_off[me],
            d,
            dtype=x.dtype,
            device=x.device,
        )
        for p in range(ws)
    ]
    dist.all_to_all(recv, send, group=group)
    return torch.cat(recv, dim=1)


def timed(fn, warmup: int = 5, iters: int = 20) -> list[float]:
    # Per-iter CUDA event timing, sorted ascending (microseconds).
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    st = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    en = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for i in range(iters):
        st[i].record()
        fn()
        en[i].record()
    torch.cuda.synchronize()
    return sorted(st[i].elapsed_time(en[i]) * 1e3 for i in range(iters))  # us


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD
    # CUSTOM_ULYSSES_USE_TMA: unset -> auto (None), "0" -> non-TMA (False), else -> TMA (True).
    _t = os.environ.get("CUSTOM_ULYSSES_USE_TMA")
    ut = None if _t is None else (_t != "0")
    group = UlyssesGroup(process_group=pg, initial_pool_bytes=8 << 30)

    scale = int(os.environ.get("PROF_SCALE", "512"))
    d = int(os.environ.get("PROF_D", "128"))
    # Uneven seqs; head blocks default larger (realistic DiT: uneven seqs, ~8-32 heads), override via PROF_HEADS.
    seq_units = [3, 5, 2, 6, 4, 7, 1, 8][:ws]
    head_splits = [
                      int(x) for x in os.environ.get("PROF_HEADS", "12,20,8,24,16,28,4,32").split(",")
                  ][:ws]
    seq_lens = [u * scale for u in seq_units]
    s_off, n_off = _prefix(seq_lens), _prefix(head_splits)
    S, N = s_off[ws], n_off[ws]
    b = 1
    impl = "auto" if _t is None else ("TMA" if _t != "0" else "non-TMA")
    if rank == 0:
        print(
            f"=== varlen mode0 | ws={ws} d={d} S={S} N={N} | ours={impl} ===",
            flush=True,
        )

    x = torch.randn(b, seq_lens[rank], N, d, dtype=torch.bfloat16, device=dev)
    # Tune the varlen launch config (threads x unroll x blocks) for these splits; verbose prints the winner.
    group.tune_varlen(b, d, seq_lens, head_splits, mode=0, use_tma=ut, verbose=False)
    remote = x.numel() * 2 * (ws - 1) / ws
    ours = timed(
        lambda: group.all_to_all_single_4d(
            x, mode=0, tag="vl", seq_lens=seq_lens, head_splits=head_splits, use_tma=ut
        )
    )
    nccl = timed(lambda: torch_varlen_mode0(x, rank, ws, s_off, n_off, pg))
    if rank == 0:
        og = remote / (ours[len(ours) // 2] * 1e3)
        ng = remote / (nccl[len(nccl) // 2] * 1e3)
        print(
            f"ours({impl}) med={ours[len(ours) // 2]:8.1f}us {og:6.0f} GB/s "
            f"| torch med={nccl[len(nccl) // 2]:8.1f}us {ng:6.0f} GB/s",
            flush=True,
        )

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
