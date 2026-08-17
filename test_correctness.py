"""Correctness worker. Run under torchrun with 1, 2, 4 or 8 ranks.

- representative shapes and both supported dtypes/modes against the NCCL reference;
- the rejection paths;
- back-to-back calls with selected ranks deliberately skewed, which stresses missing or
  mis-ordered barriers.

The raw P2P no-barrier loop is a control: it runs the same skewed pattern with no barrier and
must tear. A run whose control stays clean saw nothing, and is a blind run rather than a passing
one.
"""

from __future__ import annotations

import os
import sys
import threading
import warnings

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

from benchmark import nccl_mode0, nccl_mode1
from fast_ulysses import (
    SUPPORTED_WORLD_SIZES,
    UlyssesGroup,
    supports_dtype,
    supports_world_size,
)

# About 130 us at a 1.5 GHz clock, against a host gap between two calls of about 10 us.
SKEW_CYCLES = 200_000
ROUNDS = 400
SKEWED_RANKS = (0, 5)

FAILURES: list[str] = []


def check(condition: bool, message: str) -> None:
    if not condition:
        FAILURES.append(message)


def reference(x: torch.Tensor, mode: int, ws: int) -> torch.Tensor:
    """The NCCL path the benchmark already validates against, allocated fresh."""
    send = torch.empty(x.numel(), dtype=x.dtype, device=x.device)
    recv = torch.empty_like(send)
    b, s, h, d = x.shape
    if mode == 0:
        out = torch.empty((b, s * ws, h // ws, d), dtype=x.dtype, device=x.device)
        return nccl_mode0(x, send, recv, out, ws)
    out = torch.empty((b, s // ws, h * ws, d), dtype=x.dtype, device=x.device)
    return nccl_mode1(x, send, recv, out, ws)


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


def shapes(ws: int) -> list[tuple[int, int, int, int]]:
    """(batch, seq_local, heads_global, dim)."""
    return [
        (1, 8, 2 * ws, 16),
        (1, 128, ws, 64),
        (1, 592, 7 * ws, 128),
        (2, 16, 2 * ws, 32),
        (3, 64, ws, 128),
    ]


def check_shapes(group: UlyssesGroup, device: int, ws: int) -> None:
    for dtype in (torch.bfloat16, torch.float16):
        for batch, seq, heads, dim in shapes(ws):
            for mode in (0, 1):
                shape = (
                    (batch, seq * ws, heads // ws, dim)
                    if mode
                    else (batch, seq, heads, dim)
                )
                x = torch.randn(shape, dtype=dtype, device=device)
                want = reference(x, mode, ws)
                out = group.allocate_output(x, mode)
                group.all_to_all_4d(x, mode, out=out)
                torch.cuda.synchronize()
                check(
                    torch.equal(out, want),
                    f"{dtype} {tuple(shape)} mode={mode}: "
                    f"{int((out != want).sum())} of {out.numel()} elements differ",
                )


def check_output_workspaces(group: UlyssesGroup, device: int, ws: int) -> None:
    """Cover automatic pool identity, key isolation, and explicit live outputs."""
    shape = (1, 32, 2 * ws, 32)
    x = torch.full(shape, 11, dtype=torch.bfloat16, device=device)

    pooled_first = group.all_to_all_4d(x, mode=0)
    pooled_second = group.all_to_all_4d(x, mode=0)
    check(pooled_first is pooled_second, "the automatic pool did not reuse its key")
    check(bool((pooled_second == 11).all()), "the reused automatic output is incorrect")

    x_fp16 = x.to(torch.float16)
    other_key = group.all_to_all_4d(x_fp16, mode=0)
    check(other_key is not pooled_first, "different dtype keys shared an output object")
    check(
        other_key.data_ptr() != pooled_first.data_ptr(),
        "different dtype keys shared output storage",
    )

    first = group.allocate_output(x, mode=0)
    second = group.allocate_output(x, mode=0)
    check(first.data_ptr() != second.data_ptr(), "explicit outputs shared storage")
    x.fill_(17)
    group.all_to_all_4d(x, mode=0, out=first)
    x.fill_(23)
    group.all_to_all_4d(x, mode=0, out=second)
    check(bool((first == 17).all()), "a second explicit output overwrote the first")
    check(bool((second == 23).all()), "the second explicit output is incorrect")


def check_fixed_storage(group: UlyssesGroup, device: int, ws: int) -> None:
    """An output workspace must never silently follow another input allocation."""
    shape = (1, 32, 2 * ws, 32)
    x = torch.full(shape, 7, dtype=torch.bfloat16, device=device)
    out = group.allocate_output(x, mode=0)
    group.all_to_all_4d(x, mode=0, out=out)
    replacement = x.clone()
    raises(
        lambda: group.all_to_all_4d(replacement, mode=0, out=out),
        "a second input storage for one output workspace",
        "fixed input storage",
    )


def check_python_context_guards(group: UlyssesGroup, device: int, ws: int) -> None:
    """Context errors must be rejected before native collective work is enqueued."""
    good = torch.randn((1, 32, 2 * ws, 32), dtype=torch.bfloat16, device=device)
    out = group.allocate_output(good, 0)
    wrong_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(wrong_stream):
        raises(
            lambda: group.all_to_all_4d(good, 0, out=out),
            "an exchange on a stream other than the bound stream",
            "stream bound at construction",
        )
        raises(
            lambda: group.allocate_output(good, 0),
            "an allocation on a stream other than the bound stream",
            "stream bound at construction",
        )

    thread_errors: list[BaseException] = []

    def run_from_non_owner() -> None:
        try:
            group.all_to_all_4d(good, 0, out=out)
        except (RuntimeError, ValueError, TypeError) as error:
            # The worker records the exception and checks it on the owner thread.
            thread_errors.append(error)

    thread = threading.Thread(target=run_from_non_owner)
    thread.start()
    thread.join()
    check(bool(thread_errors), "an exchange on a non-owner thread was accepted")
    if thread_errors:
        check(
            "owner thread" in str(thread_errors[0]),
            f"a non-owner thread raised the wrong error: {thread_errors[0]!r}",
        )

    with torch.inference_mode(False), torch.enable_grad():
        raises(
            lambda: group.all_to_all_4d(good, 0, out=out),
            "an exchange with autograd enabled",
            "requires autograd to be off",
        )

    # What the guard is really about is recording, not inference mode as such, so no_grad has to
    # be accepted: it is what a serving stack that predates inference_mode still runs under. The
    # tensors are allocated inside the block too, since that is how such a stack produces them.
    with torch.inference_mode(False), torch.no_grad():
        plain = torch.randn((1, 32, 2 * ws, 32), dtype=torch.bfloat16, device=device)
        plain_out = group.allocate_output(plain, 0)
        check(
            torch.equal(
                group.all_to_all_4d(plain, 0, out=plain_out), reference(plain, 0, ws)
            ),
            "an exchange under torch.no_grad() did not match the reference",
        )

    # A real graph capture is unsafe to start while a distributed test has live work. Patch the
    # CUDA query so this remains a deterministic Python guard test with no native enqueue.
    is_capturing = torch.cuda.is_current_stream_capturing
    try:
        torch.cuda.is_current_stream_capturing = lambda: True
        raises(
            lambda: group.all_to_all_4d(good, 0, out=out),
            "an exchange during CUDA Graph capture",
            "CUDA Graph capture",
        )
    finally:
        torch.cuda.is_current_stream_capturing = is_capturing


def check_repeated(
    group: UlyssesGroup, device: int, ws: int, mode: int, rank: int
) -> int:
    """Back-to-back calls with one rank per quad skewed. Returns the number of torn rounds.

    The payload cycles through 1..128 rather than counting up: a round that lands a neighbouring
    iteration's bytes is invisible when consecutive values are close.
    """
    shape = (1, 256 * ws, 4, 128) if mode else (1, 256, 4 * ws, 128)
    x = torch.empty(shape, dtype=torch.bfloat16, device=device)
    out = group.allocate_output(x, mode)
    torn = 0
    for i in range(ROUNDS):
        value = float(i % 128 + 1)
        x.fill_(value)
        if rank in SKEWED_RANKS:
            torch.cuda._sleep(SKEW_CYCLES)
        group.all_to_all_4d(x, mode, out=out)
        if not bool((out == value).all()):
            torn += 1
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

    A failure only one rank sees still has to become a decision all of them make. A bad device
    on rank 1 fails locally, before the first collective, which is where a rank-asymmetric
    failure does the damage: without the agreement the other seven would block here rather
    than fail.
    """
    if ws < 2:
        return
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        group = UlyssesGroup.create(device="cpu" if rank == 1 else device)
    check(group is None, "an asymmetric construction failure still produced a group")
    if group is not None:
        group.destroy()


def check_capability_query(group: UlyssesGroup, device: int, ws: int) -> None:
    """The query has to agree with the refusal, or acting on it is worse than not having it.

    A caller branches on `supports` to skip a collective. If the query ever says yes where
    allocate_output says no, that branch turns a loud rank-local failure into a hang, so
    every shape the suite exercises is checked both ways.
    """
    for dtype in (torch.bfloat16, torch.float16):
        for batch, seq, heads, dim in shapes(ws):
            for mode in (0, 1):
                shape = (
                    (batch, seq * ws, heads // ws, dim)
                    if mode
                    else (batch, seq, heads, dim)
                )
                check(
                    group.supports(shape, dtype, mode),
                    f"the query refused {shape} {dtype} mode={mode}, which the suite exchanges",
                )
                check(
                    tuple(group.output_shape(shape, mode))
                    == tuple(
                        group.allocate_output(
                            torch.empty(shape, dtype=dtype, device=device), mode
                        ).shape
                    ),
                    f"output_shape disagrees with allocate_output for {shape} mode={mode}",
                )

    refused = [
        ((1, 64, 4 * ws, 128), torch.float32, 0, "a float32 shape"),
        ((1, 64, 4 * ws, 128), torch.bfloat16, 2, "an invalid mode"),
        ((1, 64, 4 * ws, 128, 1), torch.bfloat16, 0, "a 5-D shape"),
        ((1, 0, 4 * ws, 128), torch.bfloat16, 0, "an empty dimension"),
    ]
    if ws > 1:
        refused.append(
            ((1, 64, 4 * ws + 1, 128), torch.bfloat16, 0, "an indivisible head count")
        )
    for shape, dtype, mode, what in refused:
        check(
            group.unsupported_reason(shape, dtype, mode) is not None,
            f"the query accepted {what}",
        )

    # No head stride is out of range here: the copies are pitched, and a pitch is a pitch.
    check(
        group.supports((1, 64, 256, 128), torch.bfloat16, 0),
        "a wide head stride was refused, and nothing on this path has a limit on one",
    )

    check(ws in tuple(SUPPORTED_WORLD_SIZES), f"world size {ws} is not advertised")
    check(supports_dtype(torch.bfloat16), "bfloat16 is not advertised")
    check(not supports_dtype(torch.float32), "float32 is advertised")
    check(supports_world_size(ws), f"supports_world_size({ws}) is False")
    check(not supports_world_size(3), "world size 3 is advertised")


def check_rejections(group: UlyssesGroup, device: int, ws: int) -> None:
    """Every rank runs these, so no rank is left alone in a collective."""
    good = torch.randn((1, 64, 4 * ws, 128), dtype=torch.bfloat16, device=device)
    out = group.allocate_output(good, 0)

    if ws > 1:
        bad = torch.randn((1, 64, 4 * ws + 1, 128), dtype=torch.bfloat16, device=device)
        raises(
            lambda: group.all_to_all_4d(bad, 0, out=out),
            "a head count not divisible by world_size",
        )
    raises(lambda: group.all_to_all_4d(good.float(), 0, out=out), "a float32 input")
    raises(
        lambda: group.all_to_all_4d(good.transpose(1, 2), 0, out=out),
        "a non-contiguous input",
    )
    raises(
        lambda: group.all_to_all_4d(good.clone().requires_grad_(), 0, out=out),
        "an input that requires grad",
    )
    raises(lambda: group.all_to_all_4d(good, 0, out=good), "an unallocated output")
    raises(
        lambda: group.all_to_all_4d(out.view(1, 64, 4 * ws, 128), 0, out=out),
        "an input overlapping the output",
    )
    raises(lambda: group.all_to_all_4d(good, 2, out=out), "an invalid mode")

    if ws > 1:
        rank = dist.get_rank(group.pg)
        mismatched = torch.empty(
            (1, 32 + rank, 2 * ws, 32),
            dtype=torch.bfloat16,
            device=device,
        )
        raises(
            lambda: group.allocate_output(mismatched, 0),
            "a rank-inconsistent cold allocation",
            "rank-inconsistent output allocation",
        )

    rebound = group.allocate_output(good, 0)
    rebound.set_(torch.empty_like(rebound))
    raises(
        lambda: group.all_to_all_4d(good, 0, out=rebound),
        "an output rebound with Tensor.set_()",
        "storage",
    )


@torch.inference_mode()
def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.manual_seed(1234 + rank)

    check_asymmetric_construction(local_rank, ws, rank)
    created = UlyssesGroup.create(device=local_rank)
    check(created is not None, "create() refused a group the constructor accepts")
    if created is not None:
        created.destroy()

    with UlyssesGroup(device=local_rank) as group:
        if rank == 0:
            skewed = tuple(r for r in SKEWED_RANKS if r < ws)
            print(
                f"# world_size={ws} backend={group.backend} "
                f"rounds={ROUNDS} skew={SKEW_CYCLES} skewed_ranks={skewed}"
            )

        check_capability_query(group, local_rank, ws)
        check_shapes(group, local_rank, ws)
        check_output_workspaces(group, local_rank, ws)
        check_fixed_storage(group, local_rank, ws)
        check_python_context_guards(group, local_rank, ws)
        check_rejections(group, local_rank, ws)
        for mode in (0, 1):
            torn = check_repeated(group, local_rank, ws, mode, rank)
            check(
                torn == 0,
                f"mode={mode}: {torn} of {ROUNDS} skewed back-to-back rounds tore",
            )

        smoke_tears = (
            0 if ws == 1 else scheduler_smoke(local_rank, ws, rank, dist.group.WORLD)
        )

    # destroy() is intentionally idempotent, but every operational API is closed afterwards.
    group.destroy()
    dead_input = torch.empty(
        (1, 8, 2 * ws, 16), dtype=torch.bfloat16, device=local_rank
    )
    raises(
        lambda: group.allocate_output(dead_input, 0),
        "allocate_output after destroy",
        "group is destroyed",
    )
    raises(
        lambda: group.all_to_all_4d(dead_input, 0),
        "all_to_all_4d after destroy",
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
            f"{smoke_all} (the control: a run whose control stays clean is blind)"
        )
        print("FAILED" if any(failures) else "PASSED")
    sys.exit(1 if any(failures) else 0)


if __name__ == "__main__":
    main()
