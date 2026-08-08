"""torchrun worker: what happens if two live groups OVERLAP instead of partitioning?

a2a_subgroup.py covers the shape production uses -- tp2 x sp4 gives {0,2,4,6} and {1,3,5,7},
which PARTITION the job. Nothing in the API stops a caller building groups that share ranks
instead.

ANSWERED, and the answer is a HANG in the constructor. The team split brackets its body in
PARENT-team collectives while the reduce that picks the team index runs over the child triplet
alone, so a rank in two groups enters the parent collective twice and a rank in one enters it
once. Members block, non-members walk past. Nothing detects this: a rank-local check cannot see
a divergence that is already present in the first construction, and an all-gather cannot help
because only members reach the constructor at all. csrc/ulysses_group.cu carries the constraint.

So this file is NOT registered in test/test_multigpu.py and must not be -- running it hangs the
job until the harness timeout. It is kept as the reproducer for the day someone adds the
job-level layout check that would make the case detectable.

Run by hand, expecting a timeout:
    torchrun --nproc_per_node=8 test/distributed/a2a_overlapping_groups.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def log(msg: str) -> None:
    print(f"[rank {dist.get_rank()}] {msg}", flush=True)


def main() -> int:
    rank = int(os.environ["RANK"])
    n_dev = torch.cuda.device_count()
    torch.cuda.set_device(rank % n_dev)
    dev = torch.device("cuda", rank % n_dev)
    dist.init_process_group(backend="nccl", device_id=dev)
    ws = dist.get_world_size()

    if ws < 8:
        if rank == 0:
            print(f"OVERLAPPING_GROUPS SKIP (needs 8 ranks, got {ws})", flush=True)
        dist.destroy_process_group()
        return 0

    # Two groups sharing ranks 2 and 3. dist.new_group is collective over WORLD, so every rank
    # creates both handles; only the members will hand theirs to UlyssesGroup.
    a_ranks, b_ranks = [0, 1, 2, 3], [2, 3, 4, 5]
    pg_a = dist.new_group(ranks=a_ranks)
    pg_b = dist.new_group(ranks=b_ranks)

    if rank == 0:
        log(
            f"building overlapping groups {a_ranks} and {b_ranks} -- "
            "if this is the last line, the constructor HANGS"
        )

    made, err = [], None
    try:
        # Construction order is the same on every rank, which is the most favourable case; if
        # even this hangs, the API must reject overlap outright.
        if rank in a_ranks:
            made.append(("A", UlyssesGroup(process_group=pg_a, initial_pool_bytes=1 << 28)))
        if rank in b_ranks:
            made.append(("B", UlyssesGroup(process_group=pg_b, initial_pool_bytes=1 << 28)))
    except Exception:  # noqa: BLE001 -- classifying, not handling
        err = traceback.format_exc()

    if err is not None:
        if rank in (0, 2, 4):
            log(f"construction RAISED (an error is the ACCEPTABLE outcome):\n{err}")
        outcome = "raised"
    else:
        log(
            f"constructed {[n for n, _ in made]} -- overlap is ACCEPTED, so it must also be correct"
        )
        outcome = "accepted"

    # If they built, do they produce right answers? An accepted-but-wrong overlap is worse than
    # a rejection.
    bad = 0
    for name, g in made:
        pg = pg_a if name == "A" else pg_b
        gws = dist.get_world_size(pg)
        v = float(10 + rank)
        x = torch.full((1, 8, 4 * gws, 64), v, dtype=torch.bfloat16, device=dev)
        try:
            out = g.all_to_all_single_4d(x, mode=0, tag=f"ov_{name}").clone()
        except Exception:  # noqa: BLE001
            log(f"group {name} collective RAISED:\n{traceback.format_exc()}")
            bad += 1
            continue
        # Every member filled with its own constant, so the result must contain exactly the
        # members' constants and nothing else.
        want = {float(10 + r) for r in (a_ranks if name == "A" else b_ranks)}
        saw = set(torch.unique(out.float()).tolist())
        if not saw <= want:
            log(
                f"group {name}: result carries {sorted(saw - want)}, outside its own members {sorted(want)}"
            )
            bad += 1

    for _, g in made:
        try:
            g.destroy()
        except Exception:  # noqa: BLE001
            log(f"destroy RAISED:\n{traceback.format_exc()}")
            bad += 1

    verdict = torch.tensor(bad, dtype=torch.int32, device=dev)
    dist.all_reduce(verdict, op=dist.ReduceOp.SUM)
    dist.barrier()
    if rank == 0:
        # Raising is a pass: the caller is told. Accepting AND being correct is a pass too.
        # Accepting and being wrong is the failure this exists to catch; hanging shows up as
        # no verdict at all.
        ok = outcome == "raised" or verdict.item() == 0
        print(f"OVERLAPPING_GROUPS {'PASS' if ok else 'FAIL'} (outcome={outcome})", flush=True)
    dist.destroy_process_group()
    return 0 if (outcome == "raised" or verdict.item() == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
