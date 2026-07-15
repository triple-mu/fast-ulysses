"""Uniform A2A throughput, aligned with the ThunderKittens benchmark.

mode0: shape (1, N/ws, H=128, D=128); metric per_rank_comm=(N/ws)*(H/ws)*D*2*(ws-1)/time (matches TK).
mode1: shape (1, N, H/ws, D=128), input (b, s_global, n_local, d); matches bindings mode1 semantics.

Reports our throughput (CUSTOM_ULYSSES_USE_TMA selects non-TMA/TMA) plus an NCCL reference.
Run: PROF_MODE=0|1 CUSTOM_ULYSSES_USE_TMA=0|1 torchrun --nproc_per_node=8 bench_uniform.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def torch_a2a_mode0(x: torch.Tensor, ws: int, group) -> torch.Tensor:
    # mode0 oracle: input (b, s_local, n_global, d) -> output (b, s_global, n_local, d).
    b, s_local, n_global, d = x.shape
    n_local = n_global // ws
    xt = x.view(b, s_local, ws, n_local, d).permute(2, 0, 1, 3, 4).contiguous()
    out = torch.empty_like(xt)
    dist.all_to_all_single(out, xt, group=group)
    return out.permute(1, 0, 2, 3, 4).contiguous().view(b, ws * s_local, n_local, d)


def torch_a2a_mode1(x: torch.Tensor, ws: int, group) -> torch.Tensor:
    # mode1 oracle: input (b, s_global, n_local, d) -> output (b, s_local, n_global, d).
    b, s_global, n_local, d = x.shape
    s_local = s_global // ws
    xt = x.view(b, ws, s_local, n_local, d).permute(1, 0, 2, 3, 4).contiguous()
    out = torch.empty_like(xt)
    dist.all_to_all_single(out, xt, group=group)
    return out.permute(1, 2, 0, 3, 4).contiguous().view(b, s_local, ws * n_local, d)


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
    group = UlyssesGroup(process_group=pg, initial_pool_bytes=12 << 30)

    mode = int(os.environ.get("PROF_MODE", "0"))
    H = int(os.environ.get("PROF_H", "128"))
    D = int(os.environ.get("PROF_D", "128"))
    impl = "auto" if _t is None else ("TMA" if _t != "0" else "non-TMA")
    oracle = torch_a2a_mode0 if mode == 0 else torch_a2a_mode1
    if rank == 0:
        print(
            f"=== mode{mode} | ws={ws} | H={H} D={D} | ours={impl} ===",
            flush=True,
        )
    n_list = [
        int(x) for x in os.environ.get("PROF_N", "16384,32768,65536,131072,262144").split(",")
    ]
    for N in n_list:
        # mode0 input (1, N/ws, H, D); mode1 input (1, N, H/ws, D).
        if mode == 0:
            x = torch.randn(1, N // ws, H, D, dtype=torch.bfloat16, device=dev)
        else:
            x = torch.randn(1, N, H // ws, D, dtype=torch.bfloat16, device=dev)
        # No explicit tune: the timed() warmup loop triggers the first-call lazy autotune, which caches
        # the launch config so the timed iterations all hit the cache.
        remote = x.numel() * 2 * (ws - 1) / ws  # bytes leaving this rank over NVLink
        ours = timed(
            lambda: group.all_to_all_single_4d(x, mode=mode, tag=f"tk{mode}_{N}", use_tma=ut)
        )
        nccl = timed(lambda: oracle(x, ws, pg))
        if rank == 0:
            og = remote / (ours[len(ours) // 2] * 1e3)
            ng = remote / (nccl[len(nccl) // 2] * 1e3)
            print(
                f"N={N:7d} | ours({impl}) med={ours[len(ours) // 2]:8.1f}us {og:6.0f} GB/s "
                f"| NCCL med={nccl[len(nccl) // 2]:8.1f}us {ng:6.0f} GB/s",
                flush=True,
            )

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
