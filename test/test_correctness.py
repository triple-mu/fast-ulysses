"""torchrun correctness check: custom_ulysses_op vs torch permute + all_to_all_single + permute.

Pure data movement, so results must be bit-exact. Run on a multi-GPU host (ws in {2,4,8}):
    torchrun --nproc_per_node=8 custom_ulysses_op/test/test_correctness.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from custom_ulysses_op import UlyssesGroup


def torch_a2a(x: torch.Tensor, mode: int, ws: int, group) -> torch.Tensor:
    """Reference Ulysses all-to-all (permute -> all_to_all_single -> permute)."""
    b, d = x.shape[0], x.shape[-1]
    if mode == 0:
        s_local, n_global = x.shape[1], x.shape[2]
        n_local = n_global // ws
        xt = x.view(b, s_local, ws, n_local, d).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(xt)
        dist.all_to_all_single(out, xt, group=group)
        return out.permute(1, 0, 2, 3, 4).contiguous().view(b, ws * s_local, n_local, d)
    else:
        s_global, n_local = x.shape[1], x.shape[2]
        s_local = s_global // ws
        xt = x.view(b, ws, s_local, n_local, d).permute(1, 0, 2, 3, 4).contiguous()
        out = torch.empty_like(xt)
        dist.all_to_all_single(out, xt, group=group)
        return out.permute(1, 2, 0, 3, 4).contiguous().view(b, s_local, ws * n_local, d)


def torch_a2a_varlen(x, mode, me, ws, s_off, n_off, group):
    """Varlen reference: uneven A2A via torch.distributed.all_to_all (list form)."""
    b, d = x.shape[0], x.shape[-1]
    if mode == 0:
        # send to peer r: my seq x r's head block; recv from r: r's seq x my head block -> concat along seq
        send = [x[:, :, n_off[r] : n_off[r + 1], :].contiguous() for r in range(ws)]
        recv = [
            torch.empty(
                b,
                s_off[r + 1] - s_off[r],
                n_off[me + 1] - n_off[me],
                d,
                dtype=x.dtype,
                device=x.device,
            )
            for r in range(ws)
        ]
        dist.all_to_all(recv, send, group=group)
        return torch.cat(recv, dim=1)
    else:
        send = [x[:, s_off[r] : s_off[r + 1], :, :].contiguous() for r in range(ws)]
        recv = [
            torch.empty(
                b,
                s_off[me + 1] - s_off[me],
                n_off[r + 1] - n_off[r],
                d,
                dtype=x.dtype,
                device=x.device,
            )
            for r in range(ws)
        ]
        dist.all_to_all(recv, send, group=group)
        return torch.cat(recv, dim=2)


def _prefix(xs):
    off = [0]
    for v in xs:
        off.append(off[-1] + v)
    return off


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD

    group = UlyssesGroup(process_group=pg, initial_pool_bytes=1 << 30)

    b = 2
    for dtype in (torch.float16, torch.bfloat16):
        for d in (64, 128, 256):
            for mode in (0, 1):
                shape = (b, 16, 4 * ws, d) if mode == 0 else (b, 16 * ws, 4, d)
                x = torch.randn(shape, dtype=dtype, device=dev)
                tag = f"t_{str(dtype).split('.')[-1]}_d{d}_m{mode}"

                ref = torch_a2a(x, mode, ws, pg)
                # use_tma=None (auto path) for every (dtype,d,mode); explicit True/False only at d==128
                # (avoid blowup), covering both TMA / non-TMA uniform paths. All ranks must pass the
                # same use_tma (hard collective invariant).
                use_tma_list = [None] if d != 128 else [None, True, False]
                for use_tma in use_tma_list:
                    ours = group.all_to_all_single_4d(
                        x, mode=mode, tag=f"{tag}_ut{use_tma}", use_tma=use_tma
                    )
                    if not torch.equal(ours, ref):
                        raise AssertionError(
                            f"MISMATCH rank={rank} ws={ws} dtype={dtype} d={d} "
                            f"mode={mode} use_tma={use_tma}"
                        )
                    if rank == 0:
                        print(
                            f"OK ws={ws} {str(dtype).split('.')[-1]} d={d} mode={mode} "
                            f"use_tma={use_tma} shape={tuple(ours.shape)}",
                            flush=True,
                        )
                    dist.barrier()

    # Varlen (uneven s/n, not divisible by world_size): splits supplied by caller, no runtime gather.
    seq_lens = [4, 6, 3, 7, 5, 2, 8, 1][:ws]
    head_splits = [2, 3, 1, 4, 2, 5, 1, 3][:ws]
    s_off = _prefix(seq_lens)
    n_off = _prefix(head_splits)
    S, N = s_off[-1], n_off[-1]
    for dtype in (torch.float16, torch.bfloat16):
        for d in (64, 128, 256):
            for mode in (0, 1):
                shape = (
                    (b, seq_lens[rank], N, d)
                    if mode == 0
                    else (b, S, head_splits[rank], d)
                )
                x = torch.randn(shape, dtype=dtype, device=dev)
                tag = f"v_{str(dtype).split('.')[-1]}_d{d}_m{mode}"

                ref = torch_a2a_varlen(x, mode, rank, ws, s_off, n_off, pg)
                # use_tma=None (auto -> non-TMA varlen); d==128 also runs True (TMA-varlen debug path) + False.
                use_tma_list = [None] if d != 128 else [None, True, False]
                for use_tma in use_tma_list:
                    ours = group.all_to_all_single_4d(
                        x,
                        mode=mode,
                        tag=f"{tag}_ut{use_tma}",
                        seq_lens=seq_lens,
                        head_splits=head_splits,
                        use_tma=use_tma,
                    )
                    if not torch.equal(ours, ref):
                        raise AssertionError(
                            f"VARLEN MISMATCH rank={rank} ws={ws} dtype={dtype} d={d} "
                            f"mode={mode} use_tma={use_tma}"
                        )
                    if rank == 0:
                        print(
                            f"OK[varlen] ws={ws} {str(dtype).split('.')[-1]} d={d} mode={mode} "
                            f"use_tma={use_tma} shape={tuple(ours.shape)}",
                            flush=True,
                        )
                    dist.barrier()

    if rank == 0:
        print("ALL PASS", flush=True)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
