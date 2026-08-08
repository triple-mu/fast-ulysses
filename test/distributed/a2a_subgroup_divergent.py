"""torchrun check: two strided subgroups issuing DIFFERENT shapes in DIFFERENT orders.

a2a_subgroup.py keeps both sp groups on identical shapes, so nothing there covers subgroups
running independent work. This does: after ``reserve`` has taken every allocation off the call
path, the two groups get different shapes, different modes and OPPOSITE call orders, and each
still has to match dist.all_to_all_single on its own sp group.

What this checks is CORRECTNESS under divergence, not deadlock avoidance -- the two groups here
have no handshake outstanding when they allocate.

Needs an even world size >= 4:
    torchrun --nproc_per_node=8 test/distributed/a2a_subgroup_divergent.py

NEGATIVE CONTROL, for the divergence: make both groups run the table in the SAME order (drop the
``reversed``). Every rank then issues identical shapes at identical points and the worker passes
on a build that cannot handle divergence at all -- which is the blindness to avoid, not a pass.

The seal this used to check is gone with the static pool: windows are allocated on demand from a
MemPool and freed back to it, so there is nothing to seal and an undeclared tag simply allocates.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from a2a_correctness import torch_a2a  # sibling worker; torchrun puts this dir on sys.path

from fast_ulysses import UlyssesGroup

TP = 2
B, D = 2, 128


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    if ws < 2 * TP or ws % TP:
        raise SystemExit(f"needs an even world size >= {2 * TP}, got {ws}")

    sp_pgs = [dist.new_group(ranks=list(range(t, ws, TP))) for t in range(TP)]
    my_tp = rank % TP
    sp_pg = sp_pgs[my_tp]
    sp = ws // TP
    group = UlyssesGroup(process_group=sp_pg, initial_pool_bytes=1 << 30)

    # The four calls the job will make, by (tag, mode, s_local). Both groups draw from this one
    # table so that every rank reserves the same windows; which of them a group ACTUALLY issues,
    # and in what order, is chosen below and differs per group.
    calls = [
        ("small", 0, 8),
        ("small", 1, 8),
        ("large", 0, 24),
        ("large", 1, 24),
    ]

    def shape(mode: int, s_local: int) -> tuple[int, int, int, int]:
        return (B, s_local, 4 * sp, D) if mode == 0 else (B, s_local * sp, 4, D)

    # COLLECTIVE over the whole job, identical on every rank. Reserving both modes of a tag at
    # its largest sequence covers every smaller call on it, because windows match by capacity.
    group.reserve(
        [
            {"tag": tag, "shape": shape(mode, s), "mode": mode, "dtype": torch.bfloat16}
            for tag, mode, s in calls
        ]
    )

    # Here is the divergence. Group 0 runs the table forwards, group 1 backwards, so the two
    # groups request different sizes on different tags at every step. reserve() above has already
    # made every window, so nothing below allocates.
    mine = calls if my_tp == 0 else list(reversed(calls))

    torch.manual_seed(1234 + rank)
    fails = 0
    for it in range(4):  # repeat: interference between the groups is a race, not a constant
        for tag, mode, s_local in mine:
            x = torch.randn(shape(mode, s_local), dtype=torch.bfloat16, device=dev)
            ref = torch_a2a(x, mode, sp, sp_pg)
            ours = group.all_to_all_single_4d(x, mode=mode, tag=tag)
            ok = torch.equal(ours, ref)
            fails += not ok
            if not ok:
                print(
                    f"FAIL rank={rank} sp_group={my_tp} it={it} tag={tag} mode={mode}", flush=True
                )

    nfail = torch.tensor([fails], device=dev)
    dist.all_reduce(nfail)
    if rank == 0:
        print(
            "divergent subgroups: "
            + ("ALL PASS" if nfail.item() == 0 else f"FAILED {int(nfail.item())}"),
            flush=True,
        )
    group.destroy()
    dist.destroy_process_group()
    if nfail.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
