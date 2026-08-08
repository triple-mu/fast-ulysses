"""torchrun worker: prove the flag-ordering test can still fail, on every run.

    torchrun --nproc_per_node=4 test/distributed/a2a_ce_fault_injection.py

a2a_ce_flag_ordering.py asks whether the copy-engine payload is visible at the destination when
the barrier's release store announcing it arrives. Its negative control is a C++ edit plus a
rebuild -- delete the closing barrier -- which is something a person has to remember to do. It
was forgotten once: an opening barrier was added, the worker's own docstring had named that as
its blinding condition, and for several commits it passed while testing nothing, with
bindings.cpp and docs/API.md both citing it as the evidence for the assumption.

This worker removes the remembering. ``_C._set_ce_fault(delay_us)`` makes launch_a2a_ce hold the
payload on the transfer stream and skip the join back onto the caller's stream, so the closing
barrier publishes while the bytes are still moving. That is the violation, on demand, with no
rebuild -- so the control runs every time the suite does.

Two phases, and BOTH must hold or the result means nothing:

  1. ARMED: readers must see stale bytes. If they do not, the fault is not reaching the path
     under test and phase 2 proves nothing.
  2. DISARMED: readers must see none. This is the property itself.

Phase 1 failing is the more interesting outcome. It does not mean the operator is fine; it means
this worker has gone blind, exactly the way the other one did.

Read the window through the BORROWED form. The copying entry point puts a device-to-device pass
between the barrier and the first read, which is time the delayed writes would use to drain --
it would hide what is being measured, in both phases.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import _C, UlyssesGroup

ITERS = 40
DELAY_US = 2000  # long enough that no plausible copy finishes inside it


def run_phase(group, dev, rank, ws, armed: bool) -> int:
    """Issue ITERS calls and count how many land on a value other than this iteration's."""
    _C._set_ce_fault(DELAY_US if armed else 0)
    # Uneven shards, most of the sequence on rank 0: its copies are several times longer than
    # everyone else's, so it reaches the closing barrier while still copying. The same skew
    # a2a_ce_flag_ordering.py uses, and for the same reason -- the OPENING barrier re-aligns the
    # ranks, so only a skew inside the transfer itself survives it.
    seq_splits = [96 * ws] + [8] * (ws - 1)
    head_splits = [4] * ws
    s_me, d = seq_splits[rank], 128
    torn = 0
    for it in range(ITERS):
        # Cycle in 1..128: bfloat16 carries 8 significant bits, so float(it) is exact only to
        # 256 and adjacent iterations would collide on one bf16 value above that -- which is
        # precisely the comparison a tear has to survive.
        want = float(it % 128 + 1)
        x = torch.full((1, s_me, 4 * ws, d), want, dtype=torch.bfloat16, device=dev)
        y = group.all_to_all_single_4d_borrowed(
            x, mode=0, tag="fault", seq_splits=seq_splits, head_splits=head_splits
        )
        torn += int((y != want).any().item())
        dist.barrier()  # re-align before the next iteration's constant
    _C._set_ce_fault(0)
    torch.cuda.synchronize(dev)
    return torn


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    if ws < 2:
        raise SystemExit(f"needs >= 2 ranks, got {ws}")

    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)

    armed_torn = run_phase(group, dev, rank, ws, armed=True)
    clean_torn = run_phase(group, dev, rank, ws, armed=False)

    # Sum both counts across ranks: any rank observing the fault is enough for phase 1, and any
    # rank tearing without it is a failure for phase 2.
    counts = torch.tensor([armed_torn, clean_torn], device=dev)
    dist.all_reduce(counts)
    armed_total, clean_total = int(counts[0].item()), int(counts[1].item())

    ok = armed_total > 0 and clean_total == 0
    if rank == 0:
        if armed_total == 0:
            print(
                "CE_FAULT BLIND: the injected fault produced no stale reads, so the clean "
                "phase proves nothing. The fault is not reaching the path under test.",
                flush=True,
            )
        elif clean_total:
            print(
                f"CE_FAULT FAIL: {clean_total}/{ITERS * ws} clean iterations read stale bytes",
                flush=True,
            )
        else:
            print(
                f"CE_FAULT PASS (control tore {armed_total}/{ITERS * ws}, clean tore 0)", flush=True
            )

    group.destroy()
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
