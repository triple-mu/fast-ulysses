"""Are two ulysses groups that PARTITION the job correct while both are transferring?

    torchrun --nproc_per_node=8 test/distributed/subgroup.py

The layout is tp=2 x sp=ws/2, whose sp groups are the two stride-2 slices of the world -- {0,2,4,6}
and {1,3,5,7} at ws=8 -- which is the 2-D mesh docs/quickstart.md sells. Both groups run the
collective at the same time, and each rank checks its result against dist.all_to_all_single on the
MATCHING torch subgroup.

Why that is not obvious: every other worker in the suite runs on WORLD, where a group-local index
and a global one are the same number, so a build that confuses them passes all of them. Here `rank`
and `world_size` come from the subgroup (python/fast_ulysses/group.py), the peer buffer and flag
pointers come from a rendezvous keyed on the SUBGROUP's name (src/group.cc), and the barrier
publishes to `peers.p[t] + rank` with a group-local rank (src/barrier.cu). A group that rendezvoused
on the wrong name, addressed the wrong peers, or spun on the wrong flag set is a MISMATCH here
rather than a plausible answer -- and, for the ranks nobody addressed, a hang.

NEW IN v0.2, and why phase 2 is stronger than master's divergent worker: reserve()/seal() are gone,
so the collective allocation is back ON the call path. window() allocates on the first call with a
bigger shape and empty_output() allocates every time, each of them empty_strided_p2p + rendezvous +
barrier over `group_name_` alone. So "every rank must issue the same sequence of shapes" is a
per-GROUP rule, not a per-job one, and phase 2 tests what master could only test once reserve() had
taken allocation off the call path: two groups mid-allocation and mid-transfer at the same time, on
different shapes, in opposite orders.

TWO CHECKS THAT DO NOT DEPEND ON THE TIMING BITING. There is no armable in-process control for the
MEMBERSHIP property itself -- torch's get_buffer_ptrs() is group-indexed by construction, so
master's substitution has no v0.2 analogue, and a build that addresses the wrong ranks can only be
caught by the data. So the two things the data checks silently assume are checked out of band:

  * THE PAYLOAD ARM. Every comparison below is worth nothing unless no two ranks hold the same
    bytes, so that is PROVED over WORLD rather than inferred from the seed line.
  * THE EPOCH, group._handle.epoch_debug(probe, role). Two barriers run per call, so the sync
    window's epoch is exactly 4*ITERS at the end of phase 1 (that phase's first call is what
    allocates it, and allocation zeroes the pad), it advances by exactly 6 per phase-2 iteration
    after the first, and the buffer handed to out= carries exactly 2. Torn data is a SUFFICIENT
    signal that the handshake died, never a necessary one -- a run where no peer happened to be
    late comes back clean -- and on a subgroup run the epoch is additionally what says a barrier
    ran over THIS group's flag set at all. It is in turn blind to membership. Neither check is
    redundant with the other and neither may be deleted for the other's sake.

WHAT MAKES THIS RUN BLIND. Beyond the two above, the worker rests on six arrangements, and deleting
any one of them leaves a file that still prints PASS while testing nothing:

  * torch.manual_seed(1234 + rank), the GLOBAL rank. Seed by group.rank, by my_tp, or by one
    constant and the two sp groups hold IDENTICAL payloads; a result assembled from the other
    group's peers, or from this rank's tp sibling, then compares equal. It looks like boilerplate.
    The payload arm is what refuses to let it be quietly changed.
  * THE STRIDE, ranks=range(t, ws, TP). "Simplified" to contiguous halves {0,1,2,3} / {4,5,6,7},
    group 0's group-rank equals its global rank for every member, so half the job can no longer
    tell a group index from a global one -- the exact bug class being hunted.
  * PHASE 1'S WORLD BARRIER, between the reference and the call. Each rank's reference is itself a
    collective on its own sp group, so nothing else holds the two groups in phase; without it they
    drift apart and the file degenerates into correctness.py at half the world size, with no
    concurrency at all. Interference is a race; without simultaneity it is untested.
  * PHASE 2'S ABSENCE OF ONE, together with list(reversed(calls)). Both are inversions of what a
    maintainer will "fix". A barrier inside that loop, for symmetry with phase 1, re-aligns the
    groups so that neither ever allocates while the other is mid-transfer; dropping `reversed`
    makes both groups issue identical shapes at identical points, which is what phase 1 covers
    already. Either edit deletes the property with no visible change in the output.
  * PHASE 2'S ONE-SIDED SKEW. The two groups run the SAME total work per iteration -- 8, 8, 24, 24
    against 24, 24, 8, 8 -- so with nothing else in the loop the interleaving that iteration 0
    happened to produce simply repeats, and four iterations sample one arrangement rather than
    four. The GEMM chain grows with `it` to walk the phase offset, and it runs on ONE group: run
    on both it re-aligns them, exactly like the barrier that must not be there.
  * empty_output() INSIDE the phase-2 loop. It contradicts docs/api.md ("allocate outside the
    loop") and reads as a mistake; it is deliberate. window() stops allocating once it has grown to
    the largest shape in the table, so from iteration 1 on this is the only thing still putting a
    subgroup-collective rendezvous on the call path while the other group is mid-transfer.

Also blinding, in descending order of likelihood: a reference that is anything but torch's own
collective on the MATCHING sp group (a locally recomputed permutation checks the plan, not the
membership, and membership is the point); torch.equal -> allclose; one iteration instead of four,
since this is a race and not a constant; one mode; dropping the (world_size, rank) assertion; and
aggregating the failures over sp_pg instead of WORLD, which leaves rank 0's verdict covering only
its own group.

NOT COVERED, deliberately. Groups that OVERLAP rather than partition: nothing in the v0.2
constructor can hang on overlap any more (no communication library is left in C++, and its only
collective is the NVLink all_gather, over the subgroup), but the collectives moved onto the call
path, so overlapping groups reduce to the ordinary nested-collective ordering rule -- every shared
rank must enter the two groups' collectives in the same relative order, exactly as two overlapping
NCCL subgroups must. Nothing here detects a violation of it, and a worker for it would hang rather
than fail. Also not covered: the async window, whose rendezvous is a second one on the same group
name and no new membership arithmetic; and destroy()'s barrier SCOPE, since both groups tear down
together after a WORLD all_reduce, so a regression of dist.barrier(group=self.pg) to WORLD would
merely serialise here and still pass.

Below 4 ranks, or at an odd world size, there is nothing to test and this SKIPS: at sp == 1
fast_barrier returns before launching anything (src/barrier.cu) and the transfer is a self-copy, so
an unguarded run would pass while moving no data between ranks.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from correctness import reference_even  # sibling worker; torchrun puts this dir on sys.path

from fast_ulysses import UlyssesGroup

TP = 2
B, D = 2, 128
ITERS = 4  # >= 2: phase 2's epoch check needs iterations after the one that grows the window
BALLAST = 8  # GEMMs per iteration index, on group 1 only; the phase offset between the groups


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())

    # A clean exit 0, not an error: test/test_distributed.py runs every worker at nproc=3 as well
    # and asserts returncode == 0, so a machine where nothing is wrong must not fail the suite.
    if ws < 2 * TP or ws % TP:
        if rank == 0:
            print(f"SUBGROUP SKIP (needs an even world size >= {2 * TP}, got {ws})", flush=True)
        dist.destroy_process_group()
        return

    # dist.new_group is collective over WORLD, so every rank builds BOTH and keeps the one it is in:
    # sp group t = {t, t+TP, t+2*TP, ...}, stride TP -- the ulysses dimension of a tp-major mesh.
    sp_pgs = [dist.new_group(ranks=list(range(t, ws, TP))) for t in range(TP)]
    my_tp = rank % TP
    sp_pg = sp_pgs[my_tp]
    sp = ws // TP

    group = UlyssesGroup(process_group=sp_pg, require_nvlink=False)

    # Named here rather than left to surface as a wrong answer later. Aggregated over WORLD before
    # anyone acts on it, because a build that gets this wrong on SOME ranks would otherwise leave
    # the rest waiting in a barrier for a peer that had already exited -- a hang, not a failure.
    ok = int((group.world_size, group.rank) == (sp, rank // TP))
    if not ok:
        print(
            f"FAIL rank={rank}: group is ({group.world_size}, {group.rank}), expected "
            f"({sp}, {rank // TP}) -- derived from WORLD rather than from process_group",
            flush=True,
        )
    agree = torch.tensor([ok], device=dev)
    dist.all_reduce(agree)
    if int(agree.item()) != ws:
        # Identical on every rank, so every rank leaves here and none is left waiting.
        dist.destroy_process_group()
        raise SystemExit(1)

    # The GLOBAL rank, so every rank's payload is unique. This is the negative control: seeded by
    # anything group-local, the two sp groups hold identical data and a result assembled from the
    # wrong peers compares equal. See the docstring.
    torch.manual_seed(1234 + rank)
    failed = 0

    # ARM IT. Drawn from the same generator as every payload below and gathered over WORLD, since
    # the collision that matters is between the two sp groups: if this passes, no comparison in
    # this file can be satisfied by another rank's bytes. If the seed above is ever made
    # group-local, this fires instead of the whole file going quietly green.
    probe = torch.randn(64, dtype=torch.bfloat16, device=dev)
    drawn = [torch.empty_like(probe) for _ in range(ws)]
    dist.all_gather(drawn, probe)
    collisions = [(a, b) for a in range(ws) for b in range(a) if torch.equal(drawn[a], drawn[b])]
    if collisions:
        failed += 1
        print(
            f"FAIL rank={rank}: ranks {collisions} drew IDENTICAL payloads, so every check in this "
            "file would also be satisfied by the wrong group's data -- the seed is not per-rank",
            flush=True,
        )
    dist.barrier()

    # Not from empty_output(), so epoch() falls through owned_ to windows_[(role, dtype)] and reads
    # the INTERNAL sync window -- the one the copying calls use. bfloat16, which keys it.
    win_probe = torch.empty(1, dtype=torch.bfloat16, device=dev)

    # --- phase 1: identical shapes, maximal simultaneity ------------------------------------
    for it in range(ITERS):  # repeated: interference between the groups is a race, not a constant
        for mode in (0, 1):
            shape = (B, 16, 4 * sp, D) if mode == 0 else (B, 16 * sp, 4, D)
            x = torch.randn(shape, dtype=torch.bfloat16, device=dev)
            ref = reference_even(x, mode, sp, sp_pg)
            # WORLD: releases both sp groups into their a2a together. It is the only thing holding
            # them in phase -- the reference above is a collective on one sp group alone.
            dist.barrier()
            if not torch.equal(group.all_to_all_4d(x, mode=mode), ref):
                failed += 1
                print(f"FAIL rank={rank} sp_group={my_tp} phase1 it={it} mode={mode}", flush=True)
    dist.barrier()

    # Both modes carry the same window_numel (b*sp*16*4*D either way), so phase 1 allocates the
    # sync window on its first call -- which zeroes the pad -- and never grows it again. The epoch
    # is therefore an absolute here, not a delta: 2 barriers x 2 modes x ITERS calls.
    torch.cuda.synchronize()
    e_p1 = group._handle.epoch_debug(win_probe, 0)
    if e_p1 != 4 * ITERS:
        failed += 1
        print(
            f"FAIL rank={rank} sp_group={my_tp}: the sync window's epoch is {e_p1} after phase 1, "
            f"expected {4 * ITERS} -- this group's barrier did not run over its own flag set once "
            "per call, whether or not anything happened to tear",
            flush=True,
        )

    # --- phase 2: different shapes, in opposite orders ---------------------------------------
    calls = [(0, 8), (1, 8), (0, 24), (1, 24)]
    # Group 0 runs the table forwards, group 1 backwards, so the two groups are on different shapes
    # at every step -- and, while the windows are still growing, allocating different ones.
    mine = calls if my_tp == 0 else list(reversed(calls))
    # Allocated on every rank so the two groups stay at the same point in their generators; only
    # group 1 ever multiplies it. Saturates to inf after a few products -- nothing reads it.
    skew = torch.randn(1024, 1024, dtype=torch.bfloat16, device=dev)
    e_p2_start = 0  # sampled at the end of iteration 0, by which point both windows are at max

    for it in range(ITERS):
        # ONE-SIDED, and growing with `it`: the two groups' iterations cost the same, so a fixed
        # offset would replay one interleaving ITERS times. On both groups this would be a barrier
        # by another name.
        if my_tp == 1:
            for _ in range(BALLAST * it):
                skew = skew @ skew.T * 0.001
        for j, (mode, s_local) in enumerate(mine):
            shape = (B, s_local, 4 * sp, D) if mode == 0 else (B, s_local * sp, 4, D)
            x = torch.randn(shape, dtype=torch.bfloat16, device=dev)
            # Collective on sp_pg alone, which is why it does not re-align the two groups. And NO
            # barrier anywhere in this loop: one would re-align them, and their divergence is the
            # property being tested.
            ref = reference_even(x, mode, sp, sp_pg)
            if j == my_tp:
                # Inside the loop ON PURPOSE, against docs/api.md's advice. window() stops
                # allocating once it has grown to the largest shape in the table, so from iteration
                # 1 on empty_output() is the only subgroup-collective rendezvous left on the call
                # path while the other group is mid-transfer. Keying it on j == my_tp puts the two
                # groups' allocations at different steps and on different shapes.
                buf = group.empty_output(x, mode=mode)
                ours = group.all_to_all_4d(x, mode=mode, out=buf)
            else:
                ours = group.all_to_all_4d(x, mode=mode)
            if not torch.equal(ours, ref):
                failed += 1
                print(
                    f"FAIL rank={rank} sp_group={my_tp} phase2 it={it} mode={mode} s={s_local}",
                    flush=True,
                )
        if it == 0:
            # s=24 is the largest shape in the table and both groups reach it in iteration 0, so
            # the sync window is reallocated -- and its epoch reset -- somewhere inside this
            # iteration, at a step that differs per group. Everything after it is a clean delta.
            # A host read costs nothing here: torch.equal above already syncs on every call.
            torch.cuda.synchronize()
            e_p2_start = group._handle.epoch_debug(win_probe, 0)

    torch.cuda.synchronize()
    # Three of the four calls per iteration take the internal window; the fourth is zero-copy into
    # this iteration's own empty_output() buffer and touches that buffer's flags instead.
    e_p2 = group._handle.epoch_debug(win_probe, 0) - e_p2_start
    if e_p2 != 6 * (ITERS - 1):
        failed += 1
        print(
            f"FAIL rank={rank} sp_group={my_tp}: the sync window's epoch moved {e_p2} over the "
            f"last {ITERS - 1} phase-2 iterations, expected {6 * (ITERS - 1)}",
            flush=True,
        )
    # The last iteration's buffer was allocated (and zeroed) inside that iteration and saw exactly
    # one call. Anything but 2 means the peers were not barriering on the buffer they wrote --
    # which is also what a silent fallback to the internal window plus a copy-out looks like.
    e_buf = group._handle.epoch_debug(buf, 0)
    if e_buf != 2:
        failed += 1
        print(
            f"FAIL rank={rank} sp_group={my_tp}: the epoch of the empty_output() buffer handed to "
            f"out= is {e_buf}, expected 2 -- the zero-copy path was not the one taken",
            flush=True,
        )

    # Over WORLD, not sp_pg: rank 0 is in one group, and a verdict aggregated over its own group
    # would say nothing at all about the other.
    nfail = torch.tensor([failed], device=dev)
    dist.all_reduce(nfail)
    if rank == 0:
        slices = [list(range(t, ws, TP)) for t in range(TP)]
        if nfail.item() == 0:
            print(
                f"SUBGROUP PASS (sp groups {slices}, {ITERS} iterations x 2 phases, epoch +2/call)",
                flush=True,
            )
        else:
            print(f"FAILED {int(nfail.item())} checks", flush=True)
    group.destroy()
    dist.destroy_process_group()
    if nfail.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
