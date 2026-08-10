"""Is the copy-engine payload visible at the destination when the flag announcing it arrives?

    torchrun --nproc_per_node=8 test/distributed/ce_ordering.py

This is the one assumption the whole design rests on and the one nothing documents. A call writes
the payload into the peers' windows with ``cudaMemcpy2D/3DAsync`` -- copy engines -- and then
announces itself with a release store from the barrier kernel on the same stream. Stream order
says the copy has COMPLETED before the kernel launches, but "completed" is defined as a host-side
property; the Programming Guide's cross-device ordering guarantee is scoped to the NULL stream and
withdrawn for async copies elsewhere; and PTX scopes ``.release`` to prior operations of the
current THREAD, which a copy-engine transfer is not. So a pass here is evidence for this machine
and this shape, not a proof.

NVIDIA ships the same shape with weaker ordering (NVSHMEM's on-stream SIGNAL_ADD, and
TransformerEngine's userbuffers ring_exchange, both publish from a one-thread kernel with no
fence at all), so this is not a bet only this repo is making. NCCL's kernel-less CE collectives
avoid it instead, by publishing the flag as a device-to-device copy on the payload's own stream.
See docs/design.md.

The worker runs the same loop twice and BOTH results have to hold:

  1. ARMED (``_set_ce_fault``): readers MUST see stale bytes. This is the negative control, and it
     runs on every invocation rather than living in a comment somebody has to remember to apply.
     Phase 1 failing does not mean the operator is fine -- it means this worker has gone blind.
  2. CLEAN: readers must see none. That is the property itself.

What makes the loop able to observe a tear:

  * The ZERO-COPY path (``out=group.empty_output(...)``), so the check reads the window directly.
    Never copy or clone before reading: a device-to-device pass between the barrier and the first
    read is time the writes would use to drain, which hides exactly what is being measured.
  * A LARGE payload, so the copy engines are still busy when the flag is issued.
  * SKEW IN THE TRANSFER, via uneven shards. Arrival skew does not work -- the call OPENS with a
    handshake that re-aligns every rank before any data moves, and an earlier version of this test
    skewed arrival instead and therefore passed while testing nothing. Giving rank 0 most of the
    sequence makes its copies several times longer, so the others reach the closing barrier while
    it is still copying and read the instant its flag lands.
  * A distinct value per iteration, cycling in 1..128 rather than counting up: bfloat16 has 8
    significant bits, so above 256 adjacent iterations collide on one value and a stale byte
    becomes indistinguishable from a fresh one.

THE CONTROL IS SUMMED OVER THE RANKS, which window_race.py names as a blindness mode and refuses
-- there it reduces the number of ranks whose control stayed silent, so no rank's tear can vouch
for another rank's clean phase. The deviation here is forced by the skew: rank 0 is deliberately
the slow WRITER, so its own window is filled by the fast ranks and its reads are the least likely
to tear even with the fault armed. A per-rank requirement would report BLIND on the rank the skew
is built around, on a run that is working. The cost is real and stated rather than hidden: a fault
that reached only some ranks still prints PASS for all of them, so `armed` is evidence that the
fault reaches the path under test, not that it reaches it everywhere.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import _C, UlyssesGroup

ITERS = 120
FAULT_US = 2000


def run(group, x, out, kw) -> tuple[int, int]:
    """(iterations that read a stale value, total stale elements)."""
    torn = stale = 0
    for i in range(1, ITERS + 1):
        v = float(1 + i % 128)
        x.fill_(v)
        y = group.all_to_all_4d(x, mode=0, out=out, **kw)
        # Enqueued directly behind the closing barrier, reading the window itself. The host sync
        # inside .item() happens after the comparison has already read it, so it cannot mask a
        # stale read.
        bad = int((y != v).sum().item())
        if bad:
            torn += 1
            stale += bad
    return torn, stale


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    if ws < 2:
        raise SystemExit(f"needs >= 2 ranks, got {ws}")

    group = UlyssesGroup(process_group=dist.group.WORLD, require_nvlink=False)

    # ~200 MB per rank at ws=4. Rank 0 takes most of the sequence: in mode 0 a rank sends
    # s_me * head_total * d, so this is the axis that makes the transfers unequal.
    b, s_total, d = 1, 8192 * ws, 384
    base = s_total // (2 * ws)
    seq_splits = [base] * ws
    seq_splits[0] = s_total - base * (ws - 1)
    head_splits = [8] * ws
    kw = {"seq_splits": seq_splits, "head_splits": head_splits}
    x = torch.empty((b, seq_splits[rank], 8 * ws, d), dtype=torch.bfloat16, device=dev)
    mb = x.numel() * x.element_size() / 1e6

    # Allocate and rendezvous before the loop: doing it inside would serialise the ranks and hide
    # the skew this worker depends on.
    out = group.empty_output(x, mode=0, **kw)
    x.fill_(0.0)
    group.all_to_all_4d(x, mode=0, out=out, **kw)
    torch.cuda.synchronize()
    dist.barrier()

    _C._set_ce_fault(FAULT_US)
    armed_torn, _ = run(group, x, out, kw)
    _C._set_ce_fault(0)
    torch.cuda.synchronize()
    dist.barrier()

    clean_torn, clean_stale = run(group, x, out, kw)

    counts = torch.tensor([armed_torn, clean_torn, clean_stale], device=dev)
    dist.all_reduce(counts)
    armed, torn, stale = (int(v) for v in counts.tolist())

    ok = armed > 0 and torn == 0
    if rank == 0:
        if armed == 0:
            print(
                "CE_ORDERING BLIND: the injected fault produced no stale reads, so the clean "
                "phase proves nothing. The fault is not reaching the path under test.",
                flush=True,
            )
        elif torn:
            print(f"CE_ORDERING FAIL: {torn} clean iterations read stale bytes ({stale} elements)")
        else:
            print(
                f"CE_ORDERING PASS ({mb:.0f} MB/call; control tore {armed}/{ITERS * ws}, clean 0)",
                flush=True,
            )

    group.destroy()
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
