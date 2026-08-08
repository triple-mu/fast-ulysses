"""torchrun check: two STRIDED ulysses subgroups, live and transferring at the same time.

The 2-D parallel layout this exists for: tp=2 x ulysses-sp=ws/2, whose sp groups are the two
stride-2 slices of the world -- {0,2,4,6} and {1,3,5,7} at ws=8. Both groups run the same
collective concurrently and each rank checks its result against dist.all_to_all_single on the
MATCHING torch subgroup, so a group that wrote into the other group's symmetric buffers, or
synchronized against the wrong PE set, shows up as a mismatch instead of a plausible answer.

Needs an even world size >= 4:
    torchrun --nproc_per_node=8 test/distributed/a2a_subgroup.py
(or via pytest: test/test_multigpu.py::test_multigpu_subgroup)

Both groups issue the same shapes here on purpose, which keeps this worker about ADDRESSING and
nothing else. a2a_subgroup_divergent.py covers the case where they do not.

NEGATIVE CONTROL (aliasing is the failure this worker exists to catch, so make it alias): in
src/symmetric_pool.cu replace ``nvshmem_ptr(p, peer_global_pes_[i])`` with
``nvshmem_ptr(p, i)`` -- peers addressed by their index INSIDE the group instead of their global
PE -- and rebuild. On a whole-world group that substitution is the identity, so every other
worker still passes; here both sp groups aim at PEs 0..sp-1, and the run must break two ways at
once: the ranks that are addressed get the other group's payload on top of their own (FAIL
lines, whichever write lands last), and the ranks nobody addresses get no flag written, so they
spin in fast_barrier until the 600 s timeout in test/test_multigpu.py kills the job. A build
where this passes is not testing the subgroup path.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from a2a_correctness import torch_a2a  # sibling worker; torchrun puts this dir on sys.path

from fast_ulysses import UlyssesGroup

TP = 2


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    if ws < 2 * TP or ws % TP:
        raise SystemExit(f"needs an even world size >= {2 * TP}, got {ws}")

    # dist.new_group is collective over WORLD, so every rank creates BOTH sp groups and keeps the
    # one it belongs to: sp group t = {t, t+TP, t+2*TP, ...}, stride TP -- the ulysses dimension of
    # a tp-major mesh.
    sp_pgs = [dist.new_group(ranks=list(range(t, ws, TP))) for t in range(TP)]
    my_tp = rank % TP
    sp_pg = sp_pgs[my_tp]
    sp = ws // TP

    group = UlyssesGroup(process_group=sp_pg, initial_pool_bytes=1 << 30)
    if (group.world_size, group.rank) != (sp, rank // TP):
        raise AssertionError(
            f"rank={rank}: group is ({group.world_size}, {group.rank}), expected ({sp}, {rank // TP})"
        )

    b, d = 2, 128
    # Seed per GLOBAL rank: every rank's payload is unique, so a result assembled from the wrong
    # PEs (the other sp group's ranks, or this rank's tp sibling) cannot match by accident.
    torch.manual_seed(1234 + rank)
    fails = 0
    for it in range(4):  # repeat: interference between the two groups is a race, not a constant
        for mode in (0, 1):
            shape = (b, 16, 4 * sp, d) if mode == 0 else (b, 16 * sp, 4, d)
            x = torch.randn(shape, dtype=torch.bfloat16, device=dev)
            ref = torch_a2a(x, mode, sp, sp_pg)
            dist.barrier()  # WORLD: release both sp groups into their a2a together
            ours = group.all_to_all_single_4d(x, mode=mode, tag=f"m{mode}")
            ok = torch.equal(ours, ref)
            fails += not ok
            if not ok:
                print(f"FAIL rank={rank} sp_group={my_tp} it={it} mode={mode}", flush=True)

    nfail = torch.tensor([fails], device=dev)
    dist.all_reduce(nfail)
    if rank == 0:
        print(
            f"sp groups {[list(range(t, ws, TP)) for t in range(TP)]}: "
            + ("ALL PASS" if nfail.item() == 0 else f"FAILED {int(nfail.item())}"),
            flush=True,
        )
    group.destroy()
    dist.destroy_process_group()
    if nfail.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
