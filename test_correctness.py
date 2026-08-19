"""Correctness worker. Run under torchrun with 1, 2, 4 or 8 ranks.

- representative shapes and both supported dtypes/modes against the process group's own
  all-to-all;
- the rejection paths and the context guards;
- back-to-back calls with selected ranks deliberately skewed, which stresses missing or
  mis-ordered barriers.

The raw P2P no-barrier loop is only an informational scheduler smoke check. It does not exercise
verbs, CQ completion, or NIC flush, so observing (or not observing) a tear cannot arm or validate
mlx5 correctness.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

import fast_ulysses as fu
from fast_ulysses import (
    SUPPORTED_WORLD_SIZES,
    UlyssesGroup,
    supports_dtype,
    supports_world_size,
)
from fast_ulysses._C import segment_releases_seen
from fast_ulysses._fallback import all_to_all_4d as reference

# About 130 us at a 1.5 GHz clock, against a host gap between two calls of about 10 us.
SKEW_CYCLES = 200_000
ROUNDS = 400
SKEWED_RANKS = (0, 5)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def raises(fn, what: str, expected: str | None = None) -> None:
    try:
        fn()
    except (RuntimeError, ValueError, TypeError) as error:
        if expected is not None and expected not in str(error):
            FAILURES.append(
                f"{what} raised the wrong error: expected {expected!r}, got {error!r}"
            )
        return
    FAILURES.append(f"{what} was accepted")


def expected_backend(ws: int) -> str:
    configured = os.environ.get("FAST_ULYSSES_EXPECT_BACKEND")
    if configured is not None:
        if configured not in ("p2p", "mlx5"):
            raise ValueError("FAST_ULYSSES_EXPECT_BACKEND must be 'p2p' or 'mlx5'")
        return configured
    if os.environ.get("FAST_ULYSSES_DISABLE_RDMA") or ws != 8:
        return "p2p"
    return "mlx5"


def shapes(ws: int, backend: str) -> list[tuple[int, int, int, int]]:
    """(batch, seq_local, heads_global, dim). The mlx5 backend takes batch 1 only."""
    # The first case is deliberately not page-sized: torch's VMM allocation is page-mapped even
    # when the tensor ends part-way through the page, and the dma-buf export must cover it.
    cases = [(1, 3, ws, 16), (1, 8, 2 * ws, 16), (1, 128, ws, 64), (1, 592, 7 * ws, 128)]
    if backend == "p2p":
        cases += [(2, 16, 2 * ws, 32), (3, 64, ws, 128)]
    return cases


def input_shape(case: tuple[int, int, int, int], mode: int, ws: int) -> tuple:
    batch, seq, heads, dim = case
    return (batch, seq * ws, heads // ws, dim) if mode else (batch, seq, heads, dim)


def check_shapes(device: int, ws: int, backend: str) -> None:
    for dtype in (torch.bfloat16, torch.float16):
        for case in shapes(ws, backend):
            for mode in (0, 1):
                shape = input_shape(case, mode, ws)
                if backend == "mlx5" and case == (1, 3, ws, 16):
                    check(
                        fu.unsupported_reason(shape, dtype, mode) is None,
                        f"mlx5 declined the non-page-sized {shape} registration",
                    )
                x = torch.randn(shape, dtype=dtype, device=device)
                want = reference(x, mode)
                out = fu.all_to_all_4d(x, mode)
                torch.cuda.synchronize()
                check(
                    torch.equal(out, want),
                    f"{dtype} {tuple(shape)} mode={mode}: "
                    f"{int((out != want).sum())} of {out.numel()} elements differ",
                )


def check_semantic_mapping(device: int, ws: int, rank: int) -> None:
    """Check the rank/sequence/head mapping without using the fallback as the oracle."""
    s_local, h_local, dim = 3, 2, 2
    lane = torch.arange(dim, device=device).view(1, 1, 1, dim)

    seq = torch.arange(s_local, device=device).view(1, s_local, 1, 1)
    head = torch.arange(h_local * ws, device=device).view(1, 1, h_local * ws, 1)
    mode0_input = (rank * 128 + seq * 32 + head * 2 + lane).to(torch.float16)
    mode0_output = fu.all_to_all_4d(mode0_input, 0)
    global_seq = torch.arange(s_local * ws, device=device).view(1, s_local * ws, 1, 1)
    local_head = torch.arange(h_local, device=device).view(1, 1, h_local, 1)
    mode0_expected = (
        (global_seq // s_local) * 128
        + (global_seq % s_local) * 32
        + (rank * h_local + local_head) * 2
        + lane
    ).to(torch.float16)
    check(torch.equal(mode0_output, mode0_expected), "mode=0 semantic mapping is incorrect")

    global_seq = torch.arange(s_local * ws, device=device).view(1, s_local * ws, 1, 1)
    local_head = torch.arange(h_local, device=device).view(1, 1, h_local, 1)
    mode1_input = (rank * 128 + global_seq * 4 + local_head * 2 + lane).to(torch.float16)
    mode1_output = fu.all_to_all_4d(mode1_input, 1)
    seq = torch.arange(s_local, device=device).view(1, s_local, 1, 1)
    global_head = torch.arange(h_local * ws, device=device).view(1, 1, h_local * ws, 1)
    mode1_expected = (
        (global_head // h_local) * 128
        + (rank * s_local + seq) * 4
        + (global_head % h_local) * 2
        + lane
    ).to(torch.float16)
    check(torch.equal(mode1_output, mode1_expected), "mode=1 semantic mapping is incorrect")


def check_results_are_owned(device: int, ws: int) -> None:
    """Two results must be two tensors, the way any other torch op behaves.

    The old API handed back a workspace that the next call with the same geometry overwrote, and
    a caller holding two of them silently held one. Results come from a caching allocator pool
    now, so this is the contract that replaced it -- and the recycling that makes it affordable
    is the same recycling check_repeated stresses.
    """
    shape = (1, 32, 2 * ws, 32)
    x = torch.full(shape, 11, dtype=torch.bfloat16, device=device)
    first = fu.all_to_all_4d(x, 0)
    x.fill_(23)
    second = fu.all_to_all_4d(x, 0)
    check(first.data_ptr() != second.data_ptr(), "two live results shared storage")
    torch.cuda.synchronize()
    check(bool((first == 11).all()), "the second call overwrote the first result")
    check(bool((second == 23).all()), "the second result is incorrect")

    # And when the first is dropped, the block has to come back: an exchange that leaked one
    # allocation per call would run a serving stack out of memory rather than fail a test.
    pointer = first.data_ptr()
    del first
    third = fu.all_to_all_4d(x, 0)
    check(third.data_ptr() == pointer, "a released result did not return to the pool")
    del second, third


def check_input_allocations(device: int, ws: int) -> None:
    """The answer must not depend on whether the input's address moves.

    On mlx5 the NIC reads through a registration made against one address: an input that stays
    put is read in place, and one that moves is staged into it. Both have to give the same
    answer. The second arm only exists because it is *made* to happen -- with nothing holding the
    previous input, the caching allocator hands the same block back, which is why the first arm
    is what a real loop does.
    """
    shape = (1, 32, 2 * ws, 32)
    addresses = set()
    for value in (7, 11, 13, 17):
        x = torch.full(shape, value, dtype=torch.bfloat16, device=device)
        addresses.add(x.data_ptr())
        check(
            torch.equal(fu.all_to_all_4d(x, 0), reference(x, 0)),
            f"a reused input allocation gave the wrong result at value {value}",
        )
        del x
    # Not a detail: the transport only avoids a copy per exchange because this holds, so if it
    # ever stops the suite should say which property was lost rather than only that things got
    # slower somewhere.
    check(
        len(addresses) == 1,
        f"a released input block did not come back: {len(addresses)} addresses over 4 calls, "
        "so every exchange after the first pays a staging copy",
    )

    for mode, moving_shape in ((0, shape), (1, (1, 32 * ws, 2, 32))):
        held = []
        for value in (19, 23, 29):
            x = torch.full(moving_shape, value, dtype=torch.bfloat16, device=device)
            held.append((value, x))
            check(
                torch.equal(fu.all_to_all_4d(x, mode), reference(x, mode)),
                f"a moving mode={mode} input allocation gave the wrong result at value {value}",
            )
        check(
            len({tensor.data_ptr() for _, tensor in held}) == len(held),
            f"the moving-input mode={mode} arm did not actually move the input",
        )
        # The results being right is not enough. If staging writes through a stale registration,
        # it can corrupt one of these still-live inputs without any completion reporting it.
        torch.cuda.synchronize()
        for value, x in held:
            check(
                bool((x == value).all()),
                f"a mode={mode} input held across later exchanges was overwritten: expected "
                f"{value}, got {float(x.flatten()[0])}",
            )
        del held


def check_segment_release(device: int, ws: int) -> None:
    """A segment handed back to the driver must invalidate the NIC's input registration.

    The registration is made against an address nothing holds -- deliberately, because holding it
    is what would stop the allocator ever offering that block again and so force a staging copy
    on every call. The price of not holding it is that the block can be released, after which the
    memory region would point at pages the process no longer owns and the NIC would read them
    with no completion saying anything was wrong. torch.cuda.empty_cache() is that event.
    """
    before = segment_releases_seen()
    for mode, shape in ((0, (1, 32, 2 * ws, 32)), (1, (1, 32 * ws, 2, 32))):
        for value in (29, 31, 37):
            x = torch.full(shape, value, dtype=torch.bfloat16, device=device)
            want = reference(x, mode)
            got = fu.all_to_all_4d(x, mode)
            torch.cuda.synchronize()
            check(
                torch.equal(got, want),
                f"a released mode={mode} segment gave the wrong result at value {value}",
            )
            del x, want, got
            torch.cuda.empty_cache()
    # Without this the check above passes whether the guard works or is wired to an event that
    # never fires -- which is exactly what watching only SEGMENT_FREE does once the allocator is
    # configured with expandable_segments, since that path records SEGMENT_UNMAP instead.
    check(
        segment_releases_seen() > before,
        "empty_cache() did not reach the segment-release counter, so the guard that drops a "
        "stale input registration is watching an event this allocator never emits",
    )


def check_context_guards(device: int, ws: int) -> None:
    """What must be refused, and -- just as load-bearing -- what must not be."""
    good = torch.randn((1, 32, 2 * ws, 32), dtype=torch.bfloat16, device=device)

    # An exchange runs on the caller's stream, so another stream is not an error any more. It is
    # still one group and one issuing thread, because two threads in one process group hang.
    changed = good + 1
    want = reference(good, 0)
    want_changed = reference(changed, 0)
    reusable = fu.all_to_all_4d(good, 0)
    transport = fu._transport()
    check(transport is not None, "the stream-ordering test has no transport")
    if transport is None:
        return

    # Force both calls to use one registered output. Relying on allocator reuse would make this
    # test pass whenever its per-stream cache happened to choose a fresh block, even if the
    # ordering edge were missing.
    transport._new_output = lambda shape, dtype: reusable
    other = torch.cuda.Stream(device=device)
    try:
        with torch.cuda.stream(other):
            first = fu.all_to_all_4d(good, 0)
            check(first.data_ptr() == reusable.data_ptr(), "stream test changed output blocks")
            preserved = first.clone()
        # No explicit wait here. Switching streams must order the next exchange after the
        # previous stream's exchange and already-submitted consumer.
        second = fu.all_to_all_4d(changed, 0)
    finally:
        del transport._new_output
    torch.cuda.synchronize()
    check(torch.equal(preserved, want), "a stream switch overwrote a prior result")
    check(torch.equal(second, want_changed), "an exchange after a stream switch is incorrect")

    thread_errors: list[BaseException] = []

    def run_from_non_owner() -> None:
        try:
            fu.all_to_all_4d(good, 0)
        except (RuntimeError, ValueError, TypeError) as error:
            # The worker records the exception and checks it on the owner thread.
            thread_errors.append(error)

    thread = threading.Thread(target=run_from_non_owner)
    thread.start()
    thread.join()
    check(bool(thread_errors), "an exchange on a non-owner thread was accepted")
    if thread_errors:
        check(
            "thread that built it" in str(thread_errors[0]),
            f"a non-owner thread raised the wrong error: {thread_errors[0]!r}",
        )

    # Results are the caller's now, not a workspace the next call overwrites, so autograd being
    # enabled is no longer a hazard in itself. What stays refused is an input that would need a
    # gradient, because nothing here produces one.
    with torch.inference_mode(False), torch.enable_grad():
        plain = torch.randn((1, 32, 2 * ws, 32), dtype=torch.bfloat16, device=device)
        check(
            torch.equal(fu.all_to_all_4d(plain, 0), reference(plain, 0)),
            "an exchange with autograd enabled did not match the reference",
        )
        raises(
            lambda: fu.all_to_all_4d(plain.clone().requires_grad_(), 0),
            "an input that requires grad",
            "inference only",
        )

    # A real graph capture is unsafe to start while a distributed test has live work. Patch the
    # CUDA query so this remains a deterministic Python guard test with no native enqueue.
    is_capturing = torch.cuda.is_current_stream_capturing
    try:
        torch.cuda.is_current_stream_capturing = lambda: True
        raises(
            lambda: fu.all_to_all_4d(good, 0),
            "an exchange during CUDA Graph capture",
            "CUDA Graph capture",
        )
    finally:
        torch.cuda.is_current_stream_capturing = is_capturing

    is_compiling = torch.compiler.is_compiling
    try:
        torch.compiler.is_compiling = lambda: True
        raises(
            lambda: fu.all_to_all_4d(good, 0),
            "an exchange under torch.compile",
            "torch.compile",
        )
    finally:
        torch.compiler.is_compiling = is_compiling


def check_repeated(device: int, ws: int, mode: int, rank: int) -> int:
    """Back-to-back calls with one rank per quad skewed. Returns the number of torn rounds.

    Results are freshly allocated, but the pool hands the same block back as soon as the previous
    one is dropped, so this is exactly the race the opening barrier exists for: a peer reaching
    the next call writes into a block this rank may not have finished reading.

    The payload cycles through 1..128 rather than counting up: a round that lands a neighbouring
    iteration's bytes is invisible when consecutive values are close.
    """
    shape = (1, 256 * ws, 4, 128) if mode else (1, 256, 4 * ws, 128)
    x = torch.empty(shape, dtype=torch.bfloat16, device=device)
    torn = 0
    for i in range(ROUNDS):
        value = float(i % 128 + 1)
        x.fill_(value)
        if rank in SKEWED_RANKS:
            torch.cuda._sleep(SKEW_CYCLES)
        out = fu.all_to_all_4d(x, mode)
        if not bool((out == value).all()):
            torn += 1
        del out
    torch.cuda.synchronize()
    return torn


def scheduler_smoke(device: int, ws: int, rank: int, pg) -> int:
    """Observe raw P2P scheduling without barriers; this is not a correctness oracle."""
    slice_numel = 256 * 4 * 128
    x = torch.empty(slice_numel * ws, dtype=torch.bfloat16, device=device)
    window = symm_mem.empty(slice_numel * ws, dtype=torch.bfloat16, device=device)
    handle = symm_mem.rendezvous(window, pg.group_name)
    peers = [
        handle.get_buffer(p, (slice_numel * ws,), torch.bfloat16) for p in range(ws)
    ]
    window.zero_()
    torch.cuda.synchronize()
    dist.barrier(device_ids=[device])
    torn = 0
    for i in range(ROUNDS):
        value = float(i % 128 + 1)
        x.fill_(value)
        if rank in SKEWED_RANKS:
            torch.cuda._sleep(SKEW_CYCLES)
        for step in range(ws):
            peer = rank ^ step
            peers[peer][rank * slice_numel : (rank + 1) * slice_numel].copy_(
                x[peer * slice_numel : (peer + 1) * slice_numel], non_blocking=True
            )
        if not bool((window == value).all()):
            torn += 1
    torch.cuda.synchronize()
    dist.barrier(device_ids=[device])
    return torn


def check_asymmetric_construction(device: int, ws: int, rank: int) -> None:
    """One rank failing to build must return None on all of them, not strand the rest.

    This is the shape of the real hazard: mlx5 setup fails per rank, because select_nic hands
    each rank a different NIC, so a failure only one rank sees still has to become a decision
    all of them make. A bad device on rank 1 stands in for it -- it fails at the same point in
    the sequence, locally and before the first collective. If the agreement were missing, the
    other seven would block here rather than fail.
    """
    if ws < 2:
        return
    for bad_device in ("cpu", "cuda:9999"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            group = UlyssesGroup.create(device=bad_device if rank == 1 else device)
        check(group is None, f"an asymmetric {bad_device} failure still produced a group")
        if group is not None:
            group.destroy()


def check_capability_query(device: int, ws: int, backend: str) -> None:
    """The query has to agree with what happens, or acting on it is worse than not having it.

    A caller branches on the query to skip a collective. If it ever says yes where the exchange
    falls back -- or no where it would have worked -- that branch is a lie, and on the hang side
    a dangerous one.
    """
    for dtype in (torch.bfloat16, torch.float16):
        for case in shapes(ws, backend):
            for mode in (0, 1):
                shape = input_shape(case, mode, ws)
                check(
                    fu.unsupported_reason(shape, dtype, mode) is None,
                    f"the query refused {shape} {dtype} mode={mode}, which the suite exchanges",
                )
                produced = fu.all_to_all_4d(
                    torch.empty(shape, dtype=dtype, device=device), mode
                ).shape
                check(
                    tuple(fu.output_shape(shape, mode)) == tuple(produced),
                    f"output_shape disagrees with the exchange for {shape} mode={mode}",
                )

    refused = [
        ((1, 64, 4 * ws, 128), torch.float32, 0, "a float32 shape"),
        ((1, 64, 4 * ws, 128, 1), torch.bfloat16, 0, "a 5-D shape"),
        ((1, 0, 4 * ws, 128), torch.bfloat16, 0, "an empty dimension"),
    ]
    if ws > 1:
        refused.append(
            ((1, 64, 4 * ws + 1, 128), torch.bfloat16, 0, "an indivisible head count")
        )
    for shape, dtype, mode, what in refused:
        check(
            fu.unsupported_reason(shape, dtype, mode) is not None,
            f"the query accepted {what}",
        )

    # The MKey stride is mlx5's limit and only mlx5's. A caller that transcribes it instead of
    # asking refuses this shape on p2p too, where it is perfectly legal -- which is the whole
    # reason the query exists.
    wide = ((1, 64, 256, 128), torch.bfloat16, 0)
    if backend == "mlx5":
        reason = fu.unsupported_reason(*wide)
        check(reason is not None, "mlx5 accepted a head stride over the MKey limit")
        check(
            reason is None or "DISABLE_RDMA" in reason,
            f"the MKey refusal does not say what to do about it: {reason!r}",
        )
    else:
        check(
            fu.unsupported_reason(*wide) is None,
            "p2p refused a head stride that only the MKey has a limit on",
        )

    check(ws in tuple(SUPPORTED_WORLD_SIZES), f"world size {ws} is not advertised")
    check(supports_dtype(torch.bfloat16), "bfloat16 is not advertised")
    check(not supports_dtype(torch.float32), "float32 is advertised")
    check(supports_world_size(ws), f"supports_world_size({ws}) is False")
    check(not supports_world_size(3), "world size 3 is advertised")

    invalid_queries = [
        ((1, 8, 2 * ws, 16), torch.bfloat16, 2, "mode must be 0 or 1"),
        ((1, 8, 2 * ws), torch.bfloat16, 0, "4-D"),
    ]
    if ws > 1:
        invalid_queries.append(
            ((1, 8, 2 * ws + 1, 16), torch.bfloat16, 0, "divisible")
        )
    for shape, dtype, mode, expected in invalid_queries:
        reason = fu.unsupported_reason(shape, dtype, mode)
        check(reason is not None and expected in reason, f"invalid query returned {reason!r}")
        raises(lambda s=shape, m=mode: fu.output_shape(s, m), "an invalid output query", expected)


def check_fallback(device: int, ws: int, backend: str) -> None:
    """A shape no transport carries still has to come back, correct, from the same call."""
    wide = torch.randn((1, 64, 256, 128), dtype=torch.bfloat16, device=device)
    if 256 % ws == 0:
        out = fu.all_to_all_4d(wide, 0)
        check(
            tuple(out.shape) == (1, 64 * ws, 256 // ws, 128),
            f"the fallback returned {tuple(out.shape)} for a declined shape",
        )
        check(
            torch.equal(out, reference(wide, 0)),
            "the fallback disagreed with itself, which means the call did not take it",
        )
    # float32 is refused by every transport, so this only ever reaches the fallback.
    single = torch.randn((1, 32, 2 * ws, 32), dtype=torch.float32, device=device)
    check(
        torch.equal(fu.all_to_all_4d(single, 0), reference(single, 0)),
        "the fallback did not carry an unsupported dtype",
    )


def check_rejections(device: int, ws: int) -> None:
    """Every rank runs these, so no rank is left alone in a collective."""
    good = torch.randn((1, 64, 4 * ws, 128), dtype=torch.bfloat16, device=device)
    raises(lambda: fu.all_to_all_4d(good, 2), "an invalid mode", "mode must be 0 or 1")
    raises(
        lambda: fu.all_to_all_4d(good.transpose(1, 2), 0),
        "a non-contiguous input",
        "contiguous",
    )
    fallback_dtype = good.float()
    raises(
        lambda: fu.all_to_all_4d(fallback_dtype.transpose(1, 2), 0),
        "a non-contiguous fallback input",
        "contiguous",
    )
    raises(
        lambda: fu.all_to_all_4d(fallback_dtype.requires_grad_(), 0),
        "a fallback input that requires grad",
        "inference only",
    )

    transport = fu._transport()
    if transport is None:
        return
    # The overlap guard is on the native surface, which the functional API cannot reach: it
    # allocates the output itself, so it can never hand the transfer its own source. Reaching
    # past it is the only way to keep the guard tested rather than merely present.
    native = transport._group
    out = fu.all_to_all_4d(good, 0)
    alias = torch.from_dlpack(out).view(1, 64, 4 * ws, 128)
    check(
        alias.untyped_storage()._cdata != out.untyped_storage()._cdata,
        "DLPack did not create the independent storage needed by the overlap test",
    )
    raises(
        lambda: native.all_to_all_4d(alias, out, 0),
        "an input overlapping the output",
        "overlap",
    )
    raises(
        lambda: native.all_to_all_4d(good, good, 0),
        "an unregistered output",
        "not registered",
    )

    if ws > 1:
        rank = dist.get_rank(transport.pg)
        # The sequence length has to be one no rank has exchanged yet. The check gathers only
        # when a geometry is new, so a divergence in which one rank's shape is already familiar
        # reaches no collective on that rank -- it deadlocks instead, which is exactly the
        # documented consequence of ranks issuing different shapes and not something to assert.
        mismatched = torch.empty(
            (1, 40 + rank, 2 * ws, 32), dtype=torch.bfloat16, device=device
        )
        raises(
            lambda: fu.all_to_all_4d(mismatched, 0),
            "a rank-inconsistent geometry",
            "rank-inconsistent exchange geometry",
        )


@torch.inference_mode()
def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(1234 + rank)

    expected = expected_backend(ws)
    check_asymmetric_construction(local_rank, ws, rank)

    backend = fu.backend()
    check(
        backend == expected,
        f"expected backend={expected}, got {backend}; the fallback is not a pass",
    )
    if rank == 0:
        skewed = tuple(r for r in SKEWED_RANKS if r < ws)
        print(
            f"# world_size={ws} backend={backend} expected_backend={expected} "
            f"rounds={ROUNDS} skew={SKEW_CYCLES} skewed_ranks={skewed}"
        )

    check_capability_query(local_rank, ws, backend)
    check_semantic_mapping(local_rank, ws, rank)
    check_shapes(local_rank, ws, backend)
    check_results_are_owned(local_rank, ws)
    check_input_allocations(local_rank, ws)
    check_segment_release(local_rank, ws)
    check_context_guards(local_rank, ws)
    check_fallback(local_rank, ws, backend)
    check_rejections(local_rank, ws)
    for mode in (0, 1):
        torn = check_repeated(local_rank, ws, mode, rank)
        check(torn == 0, f"mode={mode}: {torn} of {ROUNDS} skewed back-to-back rounds tore")

    smoke_tears = (
        0 if ws == 1 else scheduler_smoke(local_rank, ws, rank, dist.group.WORLD)
    )

    transport = fu._transport()
    fu.shutdown()
    if transport is not None:
        # shutdown() is intentionally idempotent, but the group it released is closed.
        transport.destroy()
        dead = torch.empty((1, 8, 2 * ws, 16), dtype=torch.bfloat16, device=local_rank)
        raises(
            lambda: transport.exchange(dead, 0),
            "an exchange after destroy",
            "group is destroyed",
        )

    smoke_all = [None] * ws
    failures = [None] * ws
    dist.all_gather_object(smoke_all, smoke_tears)
    dist.all_gather_object(failures, FAILURES)
    dist.destroy_process_group()

    if rank == 0:
        for r, messages in enumerate(failures):
            for message in messages:
                print(f"FAIL rank {r}: {message}")
        print(
            "# scheduler_smoke_no_barrier_torn_rounds="
            f"{smoke_all} (informational; not an mlx5 correctness oracle)"
        )
        print("FAILED" if any(failures) else "PASSED")
    sys.exit(1 if any(failures) else 0)


if __name__ == "__main__":
    main()
