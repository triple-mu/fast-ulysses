"""Can a peer's NEXT call overwrite this rank's window while this rank is still reading the result?

    torchrun --nproc_per_node=8 test/distributed/window_race.py

The closing barrier of call i proves every peer's call-i WRITES have landed. It proves nothing
about anyone's READS, and on the zero-copy path the caller reads the window at a time the operator
never sees -- so the guarantee has to sit at the START of call i+1, not at the end of call i. That
is the first ``fast_barrier`` in ``transfer_on_stream`` (src/bindings.cc:155-160, "WRITERS WAIT FOR
READERS"), and it only delivers anything because a peer's copy engines are chained behind that
peer's OWN opening barrier: the ``ready`` event is recorded on the caller's stream after the
barrier kernel (src/transfer.cu:87) and the transfer stream waits on it (:97). Neither of those had
to be true. docs/api.md:66-68 sells the result ("it is simply overwritten by the next call that
uses it, like any output buffer") and docs/design.md:53-55 states the model.

Only ``out=`` from ``empty_output()`` can observe this. That buffer IS a window and the peers write
it directly (src/bindings.cc:103-108); the copying form hands back a private tensor the peers
cannot reach, so the race would still exist and this worker would report nothing.

TWO PHASES, and BOTH results have to hold:

  1. ARMED: the same delay and the same read, issued on a side stream that the call following it is
     not ordered against, and the caller's stream let exactly ONE call ahead of the pending read --
     the same skew the clean phase builds, so what the control measures is one ballast against one
     next call. It MUST tear, ON EVERY RANK. This is the negative control, and it runs on every
     invocation rather than living in a comment somebody has to remember to apply.
  2. CLEAN: the same delay and the same read, ordered on the caller's stream. It must not tear.
     That is the property itself, and it is the only reader the library promises anything about.

An armed phase that tears nothing means the ballast is no longer than a peer's next call, so the
clean phase would pass with no barrier at all: a BLIND run, not a pass.

WHAT MAKES THIS WORKER BLIND -- every one of these makes it pass on a build with the opening
barrier deleted:

  * SYMMETRIC BALLAST. Dropping the ``if rank == i % ws`` guard so every rank runs the delay makes
    the clean phase vacuous: every rank arrives at call i+1 together, nobody is ahead, nobody would
    overwrite anybody. This is the worst one, because the armed phase does not depend on rank
    asymmetry and still prints a green control -- so it is the one blindness with a mechanical
    detector, the ``skewed`` counter, whose world total must be exactly ITERS.
  * DELETING OR SHRINKING THE BALLAST. The read then lands within microseconds of the closing
    barrier and beats the peers' next-call copies with no barrier at all. Caught only because both
    phases go through one ``ballast_chain`` with one ``BALLAST``: two tuned copies and the armed
    phase stops measuring the clean phase's delay.
  * LETTING THE ARMED LAG ACCUMULATE. The bound is ``current_stream().wait_stream(side)`` at the
    top of the armed loop, which waits for the PREVIOUS iteration's read. Drop it and the side
    stream falls a whole further ballast behind on every iteration: by iteration 3 the read is
    overtaken by dozens of calls, so the control tears on accumulated lag and stays green with a
    ``BALLAST`` an order of magnitude too short for the clean phase. It would then be a control
    that cannot fail, which is the same thing as no control. It disarms nothing, because it waits
    on work already enqueued -- the read of iteration i is still unordered against call i+1.
  * COUNTING THIS RANK'S OWN SHARD. In mode 0 the gathered sequence axis is ordered by source rank
    (correctness.py's ``reference_even``), so ``out[:, rank * S_LOCAL : (rank + 1) * S_LOCAL]`` is
    written by this rank's OWN copy, on the caller's stream (src/transfer.cu:116). A read on that
    stream is ordered after it by construction, so it can never tear in the clean phase and is the
    first thing to land in the armed one -- comparing the whole tensor lets the control tear on a
    purely local write while the peer path, the only path the opening barrier guards, goes
    unmeasured. Both phases compare the peer shards and nothing else.
  * AGGREGATING THE CONTROL WITH A SUM. ``armed > 0`` after an ``all_reduce`` lets one rank's tear
    vouch for every other rank's clean phase. A control is evidence about the rank that ran it, so
    what is reduced is the number of ranks whose control stayed silent, and one of those is a blind
    run.
  * READING A COPY. ``snap = out.clone()`` (or ``.cpu()``, or comparing against a reference tensor)
    right after the call moves the read off the window and makes it prompt. Both phases go silent
    permanently.
  * NOT USING THE WINDOW. ``group.all_to_all_4d(x, mode=0)`` or ``out=torch.empty_like(...)`` reads
    a private buffer. So does an ``out`` from a smaller shape: src/bindings.cc:105 requires
    ``owned->numel >= plan->window_numel`` and falls back to the internal window plus a copy-out
    otherwise, with no error. The ``epoch_debug(plain probe, 0) == -1`` check below is the detector
    for all of these; delete it and the substitution is invisible.
  * A HOST READ INSIDE THE ARMED LOOP. ``int(bad.item())``, ``torch.cuda.synchronize()``, an ``if``
    on a device value: the host then waits for the side stream before submitting call i+1, the
    caller's stream never overtakes it, and the control stops tearing. That one self-reports (the
    run prints BLIND) as long as the "armed must tear" verdict is still here.
  * ANY COLLECTIVE INSIDE EITHER LOOP -- ``dist.barrier()``, an ``all_reduce`` of the counter, a
    reference computed with ``dist.all_to_all``, a rank-0 print gated on a reduction. Before the
    read it blocks the peers directly; after it, it re-aligns them before call i+1. This is why
    this worker does not follow correctness.py's "check, then dist.barrier()" cadence: everything
    is aggregated after the loops.
  * A CONSTANT FILL. Hoisting ``x.fill_()`` out of the loop makes a torn window hold the same bytes
    as a correct one: the per-iteration value is the whole detector. It is distinct over the WHOLE
    phase rather than merely between neighbours, so it keeps detecting if the armed loop's one-call
    lag bound is ever loosened and the caller's stream gets several calls ahead again.

THE EPOCH PAIR IS THE NECESSARY CHECK, AND IT IS NOT THE CHECK. Torn data is sufficient evidence
that the handshake died, not necessary: a machine whose timing happens not to bite comes back
clean. Two barriers run per call, so the epoch of the buffer handed to ``out=`` must advance by
exactly 2 per call, and a deleted or dead opening barrier shows up as +1 whether or not anything
tore. But the epoch is blind to PLACEMENT: move the opening ``fast_barrier`` to after
``launch_a2a_ce`` (src/bindings.cc:160 -> after :162) and it still advances by 2 per call while the
property is gone. Only the clean phase sees that, so this check must not be promoted to "the" one,
and neither may be deleted as redundant with the other.

NEGATIVE CONTROL, out of process: delete the FIRST of the two ``fast_barrier(stream,
call.win->flag_ptrs, rank)`` calls in ``transfer_on_stream`` -- the one at src/bindings.cc:160,
before ``launch_a2a_ce`` -- and rebuild. The clean phase must then report torn elements AND the
epoch delta must halve. ``all_to_all_4d_timed`` at :472-476 carries the same pair, so a build being
bisected must not have only one of them patched. If that build still passes, the timing here has
stopped being adversarial on this machine and the worker is worthless as written: fix the worker
before trusting a pass.

SCOPE. The sync window only, and readers ordered on the call's stream only -- which is exactly what
the armed phase documents by tearing. The uniform fill is a tear detector, not a correctness check;
correctness.py owns the permutation.

Deviation from the usual cadence: both loops run to the end on failure. A rank that leaves early
stops issuing collectives and its peers hang in ``fast_barrier`` -- a 600 s timeout in
test/test_distributed.py, not a reported failure.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup

ITERS = 64  # <= 128, so every fill value in a phase is distinct and exact in bfloat16
BALLAST = 32  # dependent GEMMs between the call and the read; the knob the whole worker rests on
S_LOCAL, HEADS, D = 2048, 8, 128


def ballast_chain(t: torch.Tensor) -> torch.Tensor:
    """A dependent chain, so it cannot be overlapped away. The SAME chain runs in both phases: the
    armed phase is what measures whether it is longer than a peer's next call. The values underflow
    to zero after a few products in bfloat16, which changes nothing -- a dense GEMM takes the same
    time on zeros, and only the time is read."""
    for _ in range(BALLAST):
        t = t @ t.T * 0.001
    return t


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    # fast_barrier launches nothing at ws=1 (src/barrier.cu:56), so the epoch never advances and
    # there is no peer to race: every check below would be vacuous.
    if ws < 2:
        raise SystemExit(f"needs >= 2 ranks, got {ws}")
    torch.manual_seed(7 + rank)

    group = UlyssesGroup(process_group=dist.group.WORLD, require_nvlink=False)
    x = torch.empty(1, S_LOCAL, HEADS * ws, D, dtype=torch.bfloat16, device=dev)
    mb = x.numel() * x.element_size() / 1e6

    # COLLECTIVE, and hoisted above the loops: allocating inside one would rendezvous every
    # iteration and re-align every rank, which is the one thing both phases must not do.
    out = group.empty_output(x, mode=0)

    # The only write the opening barrier can hold back is a PEER's. In mode 0 the gathered sequence
    # axis is ordered by source rank, so slab `rank` of it is this rank's own share -- copied
    # on the caller's stream (src/transfer.cu:116), which a read on that stream is ordered after by
    # construction and the side stream's read is not. Counting it would let the armed control tear
    # on a local write and stop calibrating the path under test. Both phases read these two views.
    peer_lo = out.narrow(1, 0, rank * S_LOCAL)
    peer_hi = out.narrow(1, (rank + 1) * S_LOCAL, (ws - 1 - rank) * S_LOCAL)

    # One ballast per stream. The chain reassigns its argument every step, and a tensor allocated
    # on one stream and freed while another is still reading it is a caching-allocator hazard.
    side = torch.cuda.Stream()
    main_ballast = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
    with torch.cuda.stream(side):
        side_ballast = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)

    # Warm-up. The first call builds the plan, and the first GEMM on a stream allocates cuBLAS's
    # workspace, which synchronises -- landing either inside a phase would serialise its first
    # iterations and cost the control its tear. Both streams need it.
    x.fill_(0.0)
    group.all_to_all_4d(x, mode=0, out=out)
    main_ballast = ballast_chain(main_ballast)
    with torch.cuda.stream(side):
        side_ballast = ballast_chain(side_ballast)
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    dist.barrier()

    # A blocking cudaMemcpy, so it is only ever sampled after a full sync. It resolves `out`
    # through owned_ first (src/group.cc:281), which is why it reads the epoch of the buffer the
    # peers actually wrote rather than an internal window's.
    e0 = group._handle.epoch_debug(out, 0)

    # --- ARMED: nothing orders the read before the next call, so it MUST be overtaken ----------
    # `torn` is allocated on the caller's stream and incremented on `side`. Safe because it is held
    # for the whole phase and read only after the join below -- no free, so no allocator hazard.
    # Do not "fix" it with a per-iteration .item(): a host read here disarms the phase.
    torn = torch.zeros((), dtype=torch.int64, device=dev)
    for i in range(1, ITERS + 1):
        v = float(i)
        x.fill_(v)
        group.all_to_all_4d(x, mode=0, out=out)
        # Waits for the PREVIOUS iteration's read, which bounds the lag to exactly one call: read
        # i-1 was overtaken by call i and by nothing after it, so what the control measures is one
        # ballast against one next call -- the clean phase's skew. Without it the side stream falls
        # a further ballast behind every iteration and the control tears on a lag no reader here
        # has. It waits on work already enqueued, so this iteration's read stays unordered -- but
        # ABOVE the call it would order read i-1 before call i, and nothing would overtake it.
        torch.cuda.current_stream().wait_stream(side)
        side.wait_stream(torch.cuda.current_stream())  # the read starts after THIS call's barrier
        with torch.cuda.stream(side):
            side_ballast = ballast_chain(side_ballast)
            torn += (((peer_lo != v).sum() + (peer_hi != v).sum()) > 0).to(torch.int64)
        # The caller's stream deliberately does NOT wait for THIS iteration's read: that is the
        # arming. It runs into call i+1, whose copies land here while the read is still queued.
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    armed_torn = int(torn.item())
    dist.barrier()
    e1 = group._handle.epoch_debug(out, 0)

    # --- CLEAN: the same delay and the same read, ordered on the caller's stream ---------------
    torn = torch.zeros((), dtype=torch.int64, device=dev)
    stale = torch.zeros((), dtype=torch.int64, device=dev)
    skewed = 0
    for i in range(1, ITERS + 1):
        v = float(i)
        x.fill_(v)
        group.all_to_all_4d(x, mode=0, out=out)
        # ONE rank at a time is the slow reader, rotating so no rank settles into a steady lag and
        # every rank's window gets a turn. If every rank ran the ballast, nobody would be ahead of
        # anybody and this phase would pass with the opening barrier deleted. `skewed` is counted
        # INSIDE the branch and reduced below, so widening or deleting this condition shows up as a
        # world total that is not ITERS instead of as a silently vacuous phase.
        if rank == i % ws:
            main_ballast = ballast_chain(main_ballast)
            skewed += 1
        # On the caller's stream, so call i+1's opening barrier is ordered after it -- which is the
        # whole property. Nothing between the call and this read may touch the host or the peers.
        bad = (peer_lo != v).sum() + (peer_hi != v).sum()
        torn += (bad > 0).to(torch.int64)
        stale += bad
    torch.cuda.synchronize()
    clean_torn, clean_stale = int(torn.item()), int(stale.item())
    dist.barrier()
    e2 = group._handle.epoch_debug(out, 0)

    # --- the two out-of-band checks, on every rank ---------------------------------------------
    failed = 0
    if e1 - e0 != 2 * ITERS or e2 - e1 != 2 * ITERS:
        failed += 1
        print(
            f"FAIL rank={rank}: the epoch moved {e1 - e0} then {e2 - e1} over {ITERS} calls each, "
            f"expected {2 * ITERS} -- a build with two live barriers per call cannot do that, so "
            f"the opening fast_barrier (src/bindings.cc:160) did not run",
            flush=True,
        )
    # A probe that is not an empty_output() buffer falls through to windows_[(role, dtype)] and
    # returns -1 when that window was never allocated -- and only a call that did NOT take the
    # zero-copy path allocates it. This is what proves the peers were writing the buffer this
    # worker read.
    if group._handle.epoch_debug(torch.empty(1, dtype=torch.bfloat16, device=dev), 0) != -1:
        failed += 1
        print(
            f"FAIL rank={rank}: an internal sync window exists, so some call did not take the "
            "zero-copy path and this worker was reading a buffer no peer can reach",
            flush=True,
        )
    dist.barrier()

    # `armed_torn == 0` is reduced per rank, not as a sum: a control says something about the rank
    # that ran it, and this rank's clean phase is the one it vouches for.
    counts = torch.tensor(
        [int(armed_torn == 0), armed_torn, clean_torn, clean_stale, failed, skewed], device=dev
    )
    dist.all_reduce(counts)
    blind_ranks, armed, torn_total, stale_total, failures, skew_total = (
        int(c) for c in counts.tolist()
    )

    ok = blind_ranks == 0 and torn_total == 0 and failures == 0 and skew_total == ITERS
    if rank == 0:
        if blind_ranks:
            print(
                f"WINDOW_RACE BLIND on {blind_ranks}/{ws} ranks: the unordered read was never "
                "overtaken there, so the ballast is no longer than a peer's next call and the "
                "clean phase proves nothing.",
                flush=True,
            )
        elif skew_total != ITERS:
            print(
                f"WINDOW_RACE BLIND: {skew_total} skewed reads over {ITERS} clean iterations, "
                "expected exactly one rank behind per iteration. With every rank or no rank "
                "running the ballast nobody is ahead of anybody and the clean phase is vacuous.",
                flush=True,
            )
        elif torn_total:
            print(
                f"WINDOW_RACE FAIL: {torn_total} skewed reads carried a neighbouring call's value "
                f"({stale_total} elements)",
                flush=True,
            )
        elif failures:
            print(f"WINDOW_RACE FAIL: {failures} out-of-band checks failed", flush=True)
        else:
            print(
                f"WINDOW_RACE PASS ({mb:.0f} MB/call; control tore {armed}/{ITERS * ws} unordered "
                f"reads, clean 0 over {ITERS} skewed reads, epoch +2/call)",
                flush=True,
            )

    group.destroy()
    dist.destroy_process_group()
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
