"""torchrun correctness check: fast_ulysses vs torch permute + all_to_all_single + permute.

Pure data movement, so results must be bit-exact. Run on a multi-GPU host (ws in [2, 8]):
    torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py
(or via pytest: tests/test_multigpu.py)
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


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
                # use_tma=None (auto path) for every (dtype,d,mode); explicit True/False at d in {64,128}
                # to cover both forced TMA / non-TMA uniform paths (d=256 only auto, avoid blowup). All
                # ranks must pass the same use_tma (hard collective invariant). Forced TMA needs sm90+,
                # so drop it on older GPUs (e.g. A100) where use_tma=True is a documented TORCH_CHECK error.
                use_tma_list = [None, True, False] if d in (64, 128) else [None]
                if torch.cuda.get_device_capability()[0] < 9:
                    use_tma_list = [ut for ut in use_tma_list if ut is not True]
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

    # Distinct-tag non-aliasing (replaces the old a2a_frame): two concurrently-live results of the SAME
    # shape must use distinct tags and not clobber each other. Run both a2a, THEN check both -- if out_q
    # aliased out_k's buffer, out_q would now hold k's data.
    xq = torch.randn(b, 16, 4 * ws, 128, dtype=torch.bfloat16, device=dev)
    xk = torch.randn(b, 16, 4 * ws, 128, dtype=torch.bfloat16, device=dev)
    refq, refk = torch_a2a(xq, 0, ws, pg), torch_a2a(xk, 0, ws, pg)
    outq = group.all_to_all_single_4d(xq, mode=0, tag="q")
    outk = group.all_to_all_single_4d(xk, mode=0, tag="k")
    if not (torch.equal(outq, refq) and torch.equal(outk, refk)):
        raise AssertionError(f"TAG ALIAS rank={rank} ws={ws} (distinct tags clobbered each other)")
    if rank == 0:
        print(f"OK[distinct-tag] ws={ws} q/k live together, no alias", flush=True)
    dist.barrier()

    if rank == 0:
        print("ALL PASS", flush=True)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
