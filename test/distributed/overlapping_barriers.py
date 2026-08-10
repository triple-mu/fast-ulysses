"""Two barrier spin kernels resident at once, on streams nothing orders -- do they stay apart?

    torchrun --nproc_per_node=8 test/distributed/overlapping_barriers.py

docs/api.md says "the sync and async calls do not share a window, so mixing them is safe". This
worker is the evidence for that sentence, under the one condition that could make it false: an
async call still in flight on the group's high-priority comm stream while sync calls are issued on
the caller's stream, with NOTHING ordering the two streams. python/fast_ulysses/group.py states
that non-ordering, where it builds the comm stream, and names it as the reason the C++ side splits
``kSyncWindow`` from ``kAsyncWindow``; windows are keyed ``(role, dtype)``. In that shape two
one-block ``barrier_kernel``s (src/barrier.cu) are resident at the same time, each spinning on its
own ``flags[ws]`` + ``epoch`` region inside its own allocation's signal pad.

It is not obvious, because the reference implementation reproduced two distinct bugs in exactly
this shape:

  * ONE epoch counter for the group with a plain read-modify-write: both kernels can claim the same
    epoch and one handshake becomes a no-op. That HANGS -- a rank that collides falls permanently
    one epoch behind, and the first peer to reach a barrier it can no longer satisfy spins forever.
  * making the counter atomic fixed the hang and not the race. Epochs were then unique but not
    consistently ORDERED across the two collectives: rank X gave its async barrier epoch 3 and its
    sync barrier 4 while rank Y did the reverse, so X waited on a flag Y had published for the
    other call, and the results tore.

That history is why the handshake state is per window by construction (docs/design.md, "a window
is single-buffered") and why v0.2 gives the two roles separate windows. master's worker ran two
``tag``s; ``tag`` is gone (docs/design.md, "Removed, and why") and the axis it left behind is
sync-vs-async.

THE ASYNC CALL IS ISSUED FIRST, and swapping the two lines makes this file vacuous. ``stage()``
copies the input on the CALLER's stream and makes the comm stream wait on that copy
(src/group.cc:285-332, called from src/bindings.cc:300), so an async call issued SECOND is ordered
behind the entire sync call in front of it, copy-out included, and nothing is ever co-resident.
master had no staging and either order overlapped; here only async-first does.

BOTH CALLS TAKE THE COPYING PATH (``out=None``), which INVERTS master's deliberate
borrowed-on-both-sides decision. A buffer from ``empty_output()`` is its OWN window in ``owned_``
(src/group.cc:275, src/bindings.cc:104), so a call handed one stops touching the role windows
altogether -- and ``epoch()`` resolves its probe through ``owned_`` first (src/group.cc:344-356),
so the probes below would read the wrong counter. What that costs is the ability to see a few
undrained bytes at the end of a transfer, which is ce_ordering.py's question, not this one: the
failure guarded against here is a whole payload from the wrong call plus an epoch collision. The
permutation itself is correctness.py's -- the payloads here are constants.

THREE CHECKS, and none of them is redundant:

  1. DATA, sufficient but not necessary. Over ITERS interleaved rounds every result must be exactly
     its own call's constant, as a bit-exact ``!=`` count, never a tolerance. A shared window shows
     up as one call's payload in the other's result, or as an adjacent round's constant. It needs
     the timing to bite.
  2. EPOCH, necessary and immune to timing. Each call runs exactly two barriers on exactly one
     window (src/bindings.cc:160,167), so over the loop the sync probe must advance by EXACTLY
     2 x (sync calls) and the async probe by EXACTLY 2 x (async calls). Two sync calls per async
     round makes those two numbers deliberately different -- 4N and 2N -- and neither is what a
     merge reports. Merging the roles onto one KEY leaves ``windows_[(kAsyncWindow, bf16)]``
     unallocated, so the sync probe reads 6N while the async probe returns the -1 sentinel
     (src/group.cc:353) for a delta of 0; a merge that keeps both keys over one allocation reads
     6N on both.
  3. OVERLAP, the run's own liveness proof, in two halves, because neither half alone is enough.
     Device-side: event timestamps on sampled rounds must show the sync call issued before the
     MIDPOINT of the async call's device life. Not merely before its END -- the async call cannot
     start before the staging copy that the sync mark follows by microseconds, so "ended after the
     sync call started" is true of ANY run, serialized ones included, and as a check it is vacuous.
     Host-side and free of timing: the async wrapper must still be unwaited when the sync calls
     have been issued. Without check 3 the first two can be structurally correct and still be
     measuring a fully serialized pair.

WHAT MAKES THIS RUN BLIND, in the order a future maintainer is likely to reach for it:

  1. Reading the async result before the sync calls -- moving ``handle.wait()`` above them, or
     touching the wrapper with any aten op, which waits implicitly. The wait puts a stream
     dependency on the caller's stream, so the sync barrier launches only after the whole async
     call has retired. Every data check passes, both epoch deltas are still 4N/2N, and nothing is
     ever co-resident. The implicit form is caught outright by ``handle.completed``. An explicit
     ``.wait()`` does NOT set that flag (it calls ``wait_tensor`` directly, only ``trigger_wait``
     sets it), so it is caught by the event pair -- and ONLY while it sits above the sync mark,
     which is what stands in for the sync call's start. A stream dependency inserted BETWEEN that
     mark and the sync call is invisible to everything in this file; that is why the mark is the
     last statement before the call and why the comment there says so.
  2. Swapping the two calls, "because order does not matter". See the staging note above.
  3. "Use the zero-copy path, ce_ordering.py does." See the copying-path note above.
  4. Relaxing the epoch assertion to ``>=``, ``> 0``, or "advanced at all". A merged window gives 6N
     on the sync probe, which passes any such form. The equality is the whole fingerprint.
  5. Deleting the skew. Barriers that all arrive together publish and exit in a couple of
     microseconds and the two kernels then barely coexist. Same class of mistake as ce_ordering.py's
     "skewed arrival instead of the transfer" predecessor.
  6. Adding a ``dist.barrier()``, ``torch.cuda.synchronize()``, ``.item()`` or a ``print`` inside
     the loop. It re-aligns the ranks and drains the pipeline every round.
  7. ``break`` on the first mismatch, which master's reference case did. A rank that leaves early
     stops issuing collectives and its peers spin forever: a watchdog kill instead of a report.
  8. Making the two calls "more distinguishable" with a different DTYPE. Windows are keyed
     ``(role, dtype)``, so that gives a different window even with the roles merged and the negative
     control stops failing. (A different shape would not: a window is matched by capacity, so one
     dtype is still one allocation. The shapes are equal for the cheaper reasons -- one plan, one
     staging buffer, and no window reallocation mid-loop.)
  9. ``v = float(i)`` instead of the 1..128 cycle. bfloat16 has 8 significant bits, so above 256
     adjacent rounds collide on one value and a tear becomes indistinguishable from a clean read --
     which silently disables the data check for most of a loop this long.
 10. Dropping the ``CompletedHandle`` guard. On a libtorch without ``c10d::register_work``,
     ``register_stream_completion`` makes the caller's stream wait on the comm event AT CALL TIME
     (src/work.cc:73-79) and returns false; the async call is then not async at all. Every check
     passes with zero overlap, and the only visible difference is the returned type.
 11. Running at world_size 1. ``fast_barrier`` returns before launching anything
     (src/barrier.cu:56), so there is no barrier, no epoch advance, and a vacuous pass. Refused.
 12. Deleting the watchdog "because pytest already has a 600 s timeout". It does -- but the
     documented debugging path is this file under torchrun, which would then hang forever with spin
     kernels pinned on eight GPUs.
 13. "More coverage: alternate which call goes first by rank parity." That deadlocks a HEALTHY
     build and would be misread as a library bug. Remote copies for both calls serialise on one
     ``xfer_`` stream (src/transfer.cu), so with sync-first on one rank its async barrier sits
     behind its own sync call; rank A then waits on rank B's async barrier while rank B waits on
     rank A's sync barrier. Host submission order must be identical on every rank.
 14. Rounding ``SAMPLE_EVERY`` back to 100. It is prime so that the sampled rounds walk through
     every rank's turn to be the late one; a stride that shares a factor with the world size samples
     one fixed phase of the rotation, and check 3 then describes a subset it chose in advance.

NEGATIVE CONTROL, one identifier and a rebuild: change ``kAsyncWindow`` to ``kSyncWindow`` at
src/bindings.cc:302. The sync probe must then report 6N and the async probe the -1 sentinel, and
the results must tear -- or the run must hit the watchdog, since colliding epochs on one window is
also how the reference implementation HUNG. If none of the three happens, this loop is blind and a
pass here means nothing.

DEADLOCK IS THE OTHER FAILURE MODE. A spin kernel has no timeout, ``cudaStreamSynchronize`` never
returns an error, and nothing in the stack raises (CLAUDE.md: "Violating it hangs; nothing raises
and nothing times out"), so the worker bounds its own wall clock. A TIMEOUT line is a failure, not
an infrastructure hiccup.

DEVIATION from the house rhythm, deliberately: no ``dist.barrier()`` inside the loop, because it
would re-align the ranks and destroy the co-residency being measured, and no early ``break``. What
makes the deviation safe is that every rank runs every round regardless of what it sees, and the
per-round results stay on the device until one read after the loop.
"""

from __future__ import annotations

import os
import statistics
import threading
import time

import torch
import torch.distributed as dist

from fast_ulysses import CompletedHandle, UlyssesGroup

ITERS = 2000
# Prime, and larger than any legal world size, so the sampled rounds cycle through every rank's
# turn to be the late one. A stride of 100 divides evenly into ws 2 and 4 and hits only ranks 0
# and 4 at ws 8: one fixed phase of the rotation, reported as if it were the whole loop.
SAMPLE_EVERY = 97
SAMPLES = -(-ITERS // SAMPLE_EVERY)  # ceil: the last partial block samples too, and needs a slot
# The late rank has to be late by more than the host takes to submit the next call, or the skew is
# absorbed before it reaches the device and no peer ever waits in two spin kernels at once.
SKEW_CYCLES = 200_000
TIMEOUT_S = 120  # a healthy run is a few seconds -- this is a deadlock bound, not a perf assertion


def watchdog(rank: int) -> None:
    """Turn a hung barrier into exit 1.

    A daemon thread rather than a signal: the main thread is inside a CUDA call that will never
    return, and torch releases the GIL there, so this one still runs. ``os._exit`` rather than an
    exception, because normal interpreter teardown would itself block -- in NCCL's destructor, on
    the same stream that is stuck.
    """
    time.sleep(TIMEOUT_S)
    print(
        f"[rank {rank}] OVERLAPPING_BARRIERS TIMEOUT after {TIMEOUT_S} s: "
        "a barrier spin kernel never exited",
        flush=True,
    )
    os._exit(1)


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    threading.Thread(target=watchdog, args=(rank,), daemon=True).start()
    if ws < 2:
        # fast_barrier returns before launching anything at ws == 1 (src/barrier.cu:56): no
        # barrier, no epoch, nothing to overlap.
        raise SystemExit(f"needs >= 2 ranks, got {ws}")

    group = UlyssesGroup(process_group=dist.group.WORLD, require_nvlink=False)

    # ONE shape and ONE dtype for both calls -- that is what makes them collide under a merged
    # window, since windows are keyed (role, dtype). ~4 MB per rank per call at ws=4, ~8 at ws=8.
    #
    # The size is set by the LIVENESS check, not by the data check. `stage()` copies the input on
    # the caller's stream before any async device work, so that copy sits inside `issued` in every
    # round; the async call's device life has to be long enough that a sync call issued right after
    # it still lands in the first half. Sized for that: at a payload small enough for a fast
    # machine to transfer in microseconds the staging copy is most of the span, most sampled rounds
    # fall in the second half, and the worker reports BLIND. Raising the payload is the fix;
    # lowering the 2.0 bar below would only make it stop noticing that it had been serialized.
    shape = (1, 2048, 4 * ws, 128)
    q = torch.empty(shape, dtype=torch.bfloat16, device=dev)
    k1 = torch.empty_like(q)
    k2 = torch.empty_like(q)

    # WARM UP SERIALLY, one call at a time on one stream. The first use of a role allocates its
    # window -- empty_strided_p2p + rendezvous + zero_() + sym->barrier (src/group.cc:207-253), all
    # collective and host-blocking -- and doing that inside the loop would make this also a test of
    # two windows registering concurrently on two streams. Same shape as the loop, so the plan cache
    # and the staging buffer are warm too.
    blind_reason = ""
    q.fill_(0.0)
    k1.fill_(0.0)
    handle = group.all_to_all_4d_async(q, mode=0)
    is_act = not isinstance(handle, CompletedHandle)
    if not is_act:
        blind_reason = (
            "this libtorch has no c10d::register_work, so register_stream_completion made the "
            "caller's stream wait on the comm event at call time (src/work.cc:73-79). The 'async' "
            "call is fully ordered before the sync ones and nothing can overlap"
        )
    handle.wait()
    group.all_to_all_4d(k1, mode=0)
    torch.cuda.synchronize()
    dist.barrier()

    # epoch() is a blocking cudaMemcpy and is not ordered against the comm stream, hence the sync
    # above. The probe is an ordinary tensor, so it resolves through windows_[(role, dtype)] rather
    # than through owned_ -- which is the other reason neither call may use empty_output().
    base_sync = group._handle.epoch_debug(q, 0)
    base_async = group._handle.epoch_debug(q, 1)

    # Per-round results, written on the device and read once after the loop: async, sync k1, sync k2.
    # The min and max are the forensics -- they name which constant leaked.
    count = torch.zeros(3, ITERS, dtype=torch.int64, device=dev)
    lo = torch.zeros(3, ITERS, dtype=torch.float32, device=dev)
    hi = torch.zeros(3, ITERS, dtype=torch.float32, device=dev)
    marks = [tuple(torch.cuda.Event(enable_timing=True) for _ in range(3)) for _ in range(SAMPLES)]

    early_wait = 0
    t0 = time.perf_counter()
    for i in range(ITERS):
        # Three constants, all exact in bfloat16 (integers up to 256 are), all distinct from each
        # other and from their neighbours' -- so a tear carries a recognisable value rather than
        # plausible noise. Refilled rather than reallocated: stage() reads q on the CALLER's stream
        # and launch_a2a_ce joins the transfer back onto it, so the next round's fill_ is already
        # ordered behind the previous round's reads.
        v = float(1 + i % 128)
        q.fill_(v)
        k1.fill_(-v)
        k2.fill_(-(v + 128.0))
        # One late rank per round, rotating, on the caller's stream so both of this rank's calls
        # arrive late and its peers sit in both spin kernels at once.
        if rank == i % ws:
            torch.cuda._sleep(SKEW_CYCLES)

        sample = marks[i // SAMPLE_EVERY] if i % SAMPLE_EVERY == 0 else None
        if sample is not None:
            sample[0].record()  # caller stream, before anything this round submits
        handle = group.all_to_all_4d_async(q, mode=0)  # FIRST -- see the docstring
        if sample is not None:
            sample[1].record(group._comm_stream)  # after the whole async call, on its own stream
            # Caller stream, and it stands in for the sync call's start: NOTHING may be inserted
            # between this record and the call below, since anything that orders the caller's
            # stream against the comm stream there would be invisible to the gap check.
            sample[2].record()
        k1_out = group.all_to_all_4d(k1, mode=0)
        k2_out = group.all_to_all_4d(k2, mode=0)  # the sync window reused while the async is live
        # The timing-free half of the liveness proof, read BEFORE this round's wait: an aten op on
        # the wrapper waits implicitly, so a handle already marked completed here means the caller's
        # stream was ordered behind the comm stream before the sync calls were ever issued. An
        # explicit .wait() does not set the flag, which is why the event pair is still needed.
        if is_act and handle.completed:
            early_wait += 1
        q_out = handle.wait()  # LAST, and a GPU-side wait rather than a host one

        for j, (out, want) in enumerate(((q_out, v), (k1_out, -v), (k2_out, -(v + 128.0)))):
            count[j, i] = (out != want).sum()
            lo[j, i] = out.amin()
            hi[j, i] = out.amax()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    dist.barrier()

    # --- the necessary check: exact epoch deltas, per window ---------------------------------
    sync_delta = group._handle.epoch_debug(q, 0) - base_sync
    async_delta = group._handle.epoch_debug(q, 1) - base_async
    epoch_bad = int(sync_delta != 4 * ITERS) + int(async_delta != 2 * ITERS)
    if epoch_bad:
        print(
            f"FAIL rank={rank}: epochs advanced sync={sync_delta} async={async_delta}, want "
            f"sync={4 * ITERS} async={2 * ITERS} -- two barriers per call, two sync calls and one "
            f"async call per round. Merging the roles reports {6 * ITERS} on the sync probe, which "
            "passes any >= form, so the equality is the fingerprint: if the number of barriers per "
            "call changed, update these constants rather than loosening the check.",
            flush=True,
        )

    # --- the sufficient check: bit-exact, one host read for the whole loop ---------------------
    counts, los, his = count.cpu(), lo.cpu(), hi.cpu()
    data_bad = 0
    for j, name in enumerate(("async", "sync k1", "sync k2")):
        rounds = torch.nonzero(counts[j]).flatten()
        if rounds.numel() == 0:
            continue
        data_bad += int(rounds.numel())
        i = int(rounds[0])
        v = float(1 + i % 128)
        want = (v, -v, -(v + 128.0))[j]
        print(
            f"FAIL rank={rank} {name}: {rounds.numel()} of {ITERS} rounds carried another call's "
            f"value; first at round {i}, which saw [{los[j][i]:.0f}, {his[j][i]:.0f}] "
            f"want {want:.0f}",
            flush=True,
        )

    # --- the liveness check: was the pair ever actually concurrent? ---------------------------
    # Both samples are measured FORWARD from one baseline on the caller's stream, so nothing here
    # relies on cudaEventElapsedTime being signed across streams. That baseline precedes both: it
    # is on the same stream as sample[2], and the comm stream's work is gated by the staging copy
    # that follows it. `span` is the async call's whole device life measured from there; `issued`
    # is how far into it the caller's stream got before submitting the sync call.
    #
    # The bar is the MIDPOINT, not `span > issued`. The async call cannot begin before the staging
    # copy that sample[2] follows by microseconds, so `span > issued` holds in every run that ever
    # completes -- including one where a wait was inserted and `issued` sits at the async call's
    # very end. Half the span is a bar only a genuinely co-resident pair clears: the skewed rank
    # stretches the span far past the one staging copy that `issued` contains.
    spans = [(b.elapsed_time(a1), b.elapsed_time(s0)) for b, a1, s0 in marks]
    live = [span - issued for span, issued in spans if span > 2.0 * issued]
    if not blind_reason and early_wait:
        blind_reason = (
            f"the async wrapper was already waited on {early_wait} of {ITERS} rounds by the time "
            "the sync calls had been issued, so the caller's stream was ordered behind the comm "
            "stream and the two barrier kernels were never co-resident"
        )
    if not blind_reason and len(live) * 2 < len(spans):
        blind_reason = (
            f"only {len(live)} of {len(spans)} sampled rounds had the sync call issued in the "
            "first half of the async call's device life: the two ran serialized, so the data check "
            "was never given the chance to observe a shared window"
        )
    if blind_reason:
        print(f"[rank {rank}] BLIND: {blind_reason}", flush=True)

    verdict = torch.tensor([data_bad, epoch_bad, int(bool(blind_reason)), len(live)], device=dev)
    dist.all_reduce(verdict)
    data_bad, epoch_bad, blind, overlapped = (int(x) for x in verdict.tolist())
    if rank == 0:
        # FAIL outranks BLIND: a torn round or a wrong epoch delta says the windows are shared no
        # matter how concurrent the pair was, and it is what the negative control has to print.
        if data_bad or epoch_bad:
            print(
                f"OVERLAPPING_BARRIERS FAIL: {data_bad} rounds carried another call's value and "
                f"{epoch_bad} epoch probes were off. The sync and async windows are not separate.",
                flush=True,
            )
        elif blind:
            print(
                "OVERLAPPING_BARRIERS BLIND: the two calls were not concurrent, or not async at "
                "all -- this run checked NOTHING. See the per-rank BLIND lines above.",
                flush=True,
            )
        else:
            median = statistics.median(live) * 1000.0 if live else 0.0
            print(
                f"OVERLAPPING_BARRIERS PASS ({ITERS} rounds, 1 async + 2 sync per round on "
                f"unordered streams; epochs +{4 * ITERS}/+{2 * ITERS}; both calls concurrent on "
                f"{overlapped}/{len(spans) * ws} sampled rounds, median {median:.0f} us on rank 0; "
                f"{elapsed:.1f} s)",
                flush=True,
            )

    group.destroy()
    dist.destroy_process_group()
    if data_bad or epoch_bad or blind:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
