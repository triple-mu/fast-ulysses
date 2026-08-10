"""Is the result bit-exact against torch.distributed, on every path?

    torchrun --nproc_per_node=8 test/distributed/correctness.py

Pure data movement, so "close enough" is not a passing result anywhere in this file. Covers both
modes, every supported dtype, even and uneven shards, the three things ``out=`` can be, the async
form including a non-contiguous input, window reuse across repeated calls, the autograd backward,
the meta kernel's shape propagation, and the benchmark-only timed variant.
"""

from __future__ import annotations

import math
import os
from itertools import accumulate

import torch
import torch.autograd.forward_ad as fwd_ad
import torch.distributed as dist
from torch._subclasses.fake_tensor import FakeTensorMode

from fast_ulysses import UlyssesGroup

B, D_SWEEP = 2, (64, 128, 256, 512)


def reference_even(x: torch.Tensor, mode: int, ws: int, pg) -> torch.Tensor:
    """permute -> all_to_all_single -> permute, the path sglang's usp.py takes."""
    b, d = x.shape[0], x.shape[-1]
    if mode == 0:
        s_local, n_global = x.shape[1], x.shape[2]
        xt = x.view(b, s_local, ws, n_global // ws, d).permute(2, 0, 1, 3, 4).contiguous()
        out = torch.empty_like(xt)
        dist.all_to_all_single(out, xt, group=pg)
        return out.permute(1, 0, 2, 3, 4).contiguous().view(b, ws * s_local, n_global // ws, d)
    s_global, n_local = x.shape[1], x.shape[2]
    xt = x.view(b, ws, s_global // ws, n_local, d).permute(1, 0, 2, 3, 4).contiguous()
    out = torch.empty_like(xt)
    dist.all_to_all_single(out, xt, group=pg)
    return out.permute(1, 2, 0, 3, 4).contiguous().view(b, s_global // ws, ws * n_local, d)


def reference_uneven(x, mode, seq_splits, head_splits, rank, pg) -> torch.Tensor:
    """dist.all_to_all over a list, the only shape torch takes when the shards differ."""
    off = lambda s: [0] + list(accumulate(s))  # noqa: E731
    ws = len(seq_splits)
    b, d = x.shape[0], x.shape[-1]
    axis, cuts = (2, off(head_splits)) if mode == 0 else (1, off(seq_splits))
    send = [x.narrow(axis, cuts[p], cuts[p + 1] - cuts[p]).contiguous() for p in range(ws)]
    recv = [
        torch.empty(
            b,
            seq_splits[r if mode == 0 else rank],
            head_splits[rank if mode == 0 else r],
            d,
            dtype=x.dtype,
            device=x.device,
        )
        for r in range(ws)
    ]
    dist.all_to_all(recv, send, group=pg)
    return torch.cat(recv, dim=1 if mode == 0 else 2)


class Checks:
    def __init__(self, rank: int, ws: int) -> None:
        self.rank, self.ws, self.failed = rank, ws, 0

    def equal(self, name: str, got: torch.Tensor, want: torch.Tensor) -> None:
        if got.shape != want.shape:
            self.failed += 1
            print(f"FAIL rank={self.rank} {name}: shape {tuple(got.shape)} != {tuple(want.shape)}")
        elif not torch.equal(got, want):
            n = int((got != want).sum().item())
            self.failed += 1
            print(f"FAIL rank={self.rank} {name}: {n} elements differ", flush=True)
        elif self.rank == 0:
            print(f"OK ws={self.ws} {name}", flush=True)
        dist.barrier()


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    pg = dist.group.WORLD
    torch.manual_seed(1234 + rank)

    group = UlyssesGroup(process_group=pg, require_nvlink=False)
    check = Checks(rank, ws)

    # --- even shards, both modes, both dtypes, the head dims that matter -------------------
    for dtype in (torch.float16, torch.bfloat16):
        for d in D_SWEEP:
            for mode in (0, 1):
                shape = (B, 16, 4 * ws, d) if mode == 0 else (B, 16 * ws, 4, d)
                x = torch.randn(shape, dtype=dtype, device=dev)
                check.equal(
                    f"even {str(dtype).split('.')[-1]} d={d} mode={mode}",
                    group.all_to_all_4d(x, mode=mode),
                    reference_even(x, mode, ws, pg),
                )

    # --- uneven shards: shards differing by one token, and a badly skewed split ------------
    s_total = 97 * ws + 13  # does not divide, which is the point
    even = s_total // ws
    for name, seq_splits in (
        ("uneven near-even", [even + (1 if p < s_total - even * ws else 0) for p in range(ws)]),
        ("uneven skewed", [s_total - (ws - 1)] + [1] * (ws - 1)),
    ):
        head_splits = [4] * ws
        for mode in (0, 1):
            shape = (
                (B, seq_splits[rank], sum(head_splits), 128)
                if mode == 0
                else (B, sum(seq_splits), head_splits[rank], 128)
            )
            x = torch.randn(shape, dtype=torch.bfloat16, device=dev)
            kw = {"seq_splits": seq_splits, "head_splits": head_splits}
            check.equal(
                f"{name} mode={mode}",
                group.all_to_all_4d(x, mode=mode, **kw),
                reference_uneven(x, mode, seq_splits, head_splits, rank, pg),
            )

    # --- what `out=` can be: absent, a plain tensor, and the window itself -----------------
    x = torch.randn(B, 16, 4 * ws, 128, dtype=torch.bfloat16, device=dev)
    want = reference_even(x, 0, ws, pg)

    plain = torch.empty_like(want)
    check.equal("out=plain tensor", group.all_to_all_4d(x, mode=0, out=plain), want)

    window = group.empty_output(x, mode=0)
    check.equal("out=empty_output (zero copy)", group.all_to_all_4d(x, mode=0, out=window), want)
    # The method returns the very object handed in, so a pointer comparison here would be true by
    # construction. What can silently differ is the PATH: a buffer not recognised as a window falls
    # back to an internal window plus a copy-out (src/bindings.cc), with the right answer and no
    # error. The buffer's own epoch is the detector -- allocation zeroes its signal pad, and only a
    # barrier over that buffer advances it, twice for the one call above.
    if ws > 1:
        e = group._handle.epoch_debug(window, 0)
        if e != 2:
            check.failed += 1
            print(
                f"FAIL rank={rank} out=empty_output: the buffer's epoch is {e}, expected 2 -- the "
                "zero-copy path was not the one taken",
                flush=True,
            )
        elif rank == 0:
            print(f"OK ws={ws} out=empty_output took the zero-copy path", flush=True)
        dist.barrier()

    # A copied result must survive the next call; the zero-copy one is overwritten by design, so
    # only the copying form can be checked for it.
    kept = group.all_to_all_4d(x, mode=0)
    group.all_to_all_4d(torch.randn_like(x), mode=0)
    check.equal("copied result outlives the next call", kept, want)

    # --- round trip: mode 1 undoes mode 0 --------------------------------------------------
    check.equal("round trip", group.all_to_all_4d(group.all_to_all_4d(x, mode=0), mode=1), x)

    # --- async matches sync, and takes the same arguments ----------------------------------
    check.equal("async", group.all_to_all_4d_async(x, mode=0).wait(), want)

    # A NON-CONTIGUOUS async input. The staging copy has to absorb the strided read on the
    # CALLER's stream; materialising it on the comm stream instead is a cross-stream read of a
    # tensor nothing has ordered, and it is invisible whenever the input happens to be contiguous.
    wide = torch.empty(B, 16, 4 * ws, 256, dtype=torch.bfloat16, device=dev)
    strided = wide[..., :128]
    strided.copy_(x)
    if strided.is_contiguous():
        raise AssertionError("the non-contiguous case built a contiguous tensor; it tests nothing")
    check.equal(
        "async, non-contiguous input", group.all_to_all_4d_async(strided, mode=0).wait(), want
    )
    async_window = group.empty_output(x, mode=0)
    check.equal(
        "async out=empty_output",
        group.all_to_all_4d_async(x, mode=0, out=async_window).wait(),
        want,
    )

    # --- steady state: the window is allocated once and reused ------------------------------
    # Correct results do not show this: a window reallocated every round would produce them too.
    # The epoch does, because allocating a window zeroes its signal pad. Two barriers per call, so
    # ROUNDS calls on ONE window move it by exactly 2 * ROUNDS, and any reallocation partway
    # through leaves it short. The probe is a plain tensor, so it resolves to the internal sync
    # window rather than to an empty_output() buffer, and that window is already at its high-water
    # mark here -- the uneven shapes above are larger than this one, so it cannot grow either.
    rounds = 20
    probe = torch.empty(1, dtype=torch.bfloat16, device=dev)
    e_before = group._handle.epoch_debug(probe, 0)
    for _ in range(rounds):
        check_out = group.all_to_all_4d(x, mode=0)
    check.equal(f"{rounds} rounds on one window", check_out, want)
    e_after = group._handle.epoch_debug(probe, 0)
    if ws > 1 and e_after - e_before != 2 * rounds:
        check.failed += 1
        print(
            f"FAIL rank={rank} {rounds} rounds on one window: the sync window's epoch moved "
            f"{e_after - e_before}, expected {2 * rounds} -- the window was reallocated mid-loop",
            flush=True,
        )
    dist.barrier()

    # --- the dtypes that are not float16/bfloat16 -------------------------------------------
    # The transport is byte-oriented all the way down, so these differ from the sweep above only
    # in element size. Compared as bytes: all_to_all_single carries no float8, and the reference
    # for a permutation does not need to know what the bytes mean. A round trip cannot see an
    # addressing error that is symmetric between the two modes, so the absolute reference for the
    # one-byte element sizes is test_plan.py's replay, which is parametrised over them.
    for dtype in (
        torch.float32,
        torch.float8_e4m3fn,
        torch.float8_e5m2,
        torch.int8,
        torch.uint8,
    ):
        seed = torch.randn(B, 16, 4 * ws, 128, dtype=torch.float32, device=dev)
        xd = seed.to(dtype) if dtype.is_floating_point else (seed.abs() * 50).to(dtype)
        back = group.all_to_all_4d(group.all_to_all_4d(xd, mode=0), mode=1)
        check.equal(
            f"round trip {str(dtype).split('.')[-1]}",
            back.view(torch.uint8),
            xd.view(torch.uint8),
        )

    # --- autograd: the vjp of a permutation is the inverse permutation ------------------------
    # Bit-exact, like everything else here -- the backward moves the same bytes, it does not
    # recompute anything, so "close" would hide a wrong plan.
    for mode in (0, 1):
        shape = (B, 16, 4 * ws, 128) if mode == 0 else (B, 16 * ws, 4, 128)
        xg = torch.randn(shape, dtype=torch.bfloat16, device=dev, requires_grad=True)
        upstream = torch.randn_like(group.all_to_all_4d(xg.detach(), mode=mode))
        group.all_to_all_4d(xg, mode=mode).backward(upstream)
        check.equal(
            f"backward mode={mode} is forward mode={1 - mode}",
            xg.grad,
            group.all_to_all_4d(upstream, mode=1 - mode),
        )

    # The splits pass through the reversal UNCHANGED. This is the check that fails if someone
    # "fixes" the backward by swapping them the way all_to_all_single's backward does -- and with
    # a skewed split it fails on the shape, not just the values.
    seq_splits = [s_total - (ws - 1)] + [1] * (ws - 1)
    head_splits = [4] * ws
    xu = torch.randn(
        B, seq_splits[rank], sum(head_splits), 128, dtype=torch.bfloat16, device=dev
    ).requires_grad_(True)
    kw = {"seq_splits": seq_splits, "head_splits": head_splits}
    up = torch.randn_like(group.all_to_all_4d(xu.detach(), mode=0, **kw))
    group.all_to_all_4d(xu, mode=0, **kw).backward(up)
    check.equal(
        "backward with skewed splits, passed through",
        xu.grad,
        group.all_to_all_4d(up, mode=1, **kw),
    )

    # `out=` on a grad-requiring input is REFUSED, not silently detached; validation.py owns that
    # rejection, since this file only ever passes arguments the operator accepts.

    # --- forward-mode AD -----------------------------------------------------------------------
    # torch::autograd::Function has no C++ jvp hook, so the Autograd kernel carries the tangent by
    # hand. Without that it is dropped silently: the dual comes back with no tangent and nothing
    # raises. The jvp of a permutation is the permutation, so the tangent goes through the same
    # collective -- which means a dual input issues TWO, and every rank must agree on carrying one.
    with fwd_ad.dual_level():
        tangent = torch.randn_like(x)
        primal, jvp = fwd_ad.unpack_dual(group.all_to_all_4d(fwd_ad.make_dual(x, tangent), mode=0))
    if jvp is None:
        check.failed += 1
        print(f"FAIL rank={rank} forward-mode AD: the tangent was dropped", flush=True)
        dist.barrier()
    else:
        check.equal(
            "forward-mode AD carries the tangent", jvp, group.all_to_all_4d(tangent, mode=0)
        )
    check.equal("forward-mode AD leaves the primal alone", primal, want)

    # --- double backward -----------------------------------------------------------------------
    # backward() routes through apply() rather than the plain call so the graph continues. A plain
    # call there loses grad_fn on the first-order gradient and the second raises.
    xd = torch.randn(B, 16, 4 * ws, 128, dtype=torch.bfloat16, device=dev, requires_grad=True)
    (first,) = torch.autograd.grad(
        (group.all_to_all_4d(xd, mode=0) ** 2).sum(), xd, create_graph=True
    )
    (second,) = torch.autograd.grad(first.sum(), xd)
    check.equal(
        "double backward of sum(y**2) is 2", second.float(), torch.full_like(second.float(), 2.0)
    )

    # --- async off the default stream -----------------------------------------------------------
    # The staging copy runs on the CALLER's stream, which the op is told about explicitly. Reading
    # the default stream instead is invisible whenever the caller is already on it -- which every
    # other async check here is. The ballast is what makes the input's producer late enough that an
    # unordered staging read sees the wrong bytes.
    side = torch.cuda.Stream(device=dev)
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        ballast = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)
        for _ in range(24):
            ballast = ballast @ ballast.T * 0.001
        produced = torch.empty_like(x).copy_(x)
        off_stream = group.all_to_all_4d_async(produced, mode=0).wait()
    torch.cuda.current_stream().wait_stream(side)
    check.equal("async issued off the default stream", off_stream, want)

    # --- empty_output hands back a plain tensor -------------------------------------------------
    # It inherits the Autograd key from the alias registration unless a fallthrough says otherwise,
    # and would then return a non-leaf that cannot even be detached in place -- which then fails as
    # `out=`. docs/api.md promises an ordinary tensor.
    grad_in = x.clone().requires_grad_(True)
    plain_buf = group.empty_output(grad_in, mode=0)
    if plain_buf.requires_grad or plain_buf.grad_fn is not None or not plain_buf.is_leaf:
        check.failed += 1
        print(f"FAIL rank={rank} empty_output claims a gradient it has no formula for", flush=True)
    elif rank == 0:
        print(f"OK ws={ws} empty_output hands back a plain tensor", flush=True)
    dist.barrier()

    # --- meta: shape propagation without touching the device ---------------------------------
    with FakeTensorMode() as fake:
        shaped = group.all_to_all_4d(fake.from_tensor(x), mode=0)
    if tuple(shaped.shape) != tuple(want.shape) or shaped.device.type != "cuda":
        check.failed += 1
        print(f"FAIL rank={rank} meta: {tuple(shaped.shape)} on {shaped.device}")
    elif rank == 0:
        print(f"OK ws={ws} meta propagates {tuple(shaped.shape)}", flush=True)
    dist.barrier()

    # --- the timed variant, which is a SECOND copy of the call ---------------------------------
    # all_to_all_4d_timed re-issues barrier -> transfer -> barrier -> copy_out inline instead of
    # calling transfer_on_stream() and copy_out(), so the real path can change under it. THAT
    # duplication is what is checked here, not the timings: every number in docs/benchmark.md is
    # read off this op, and a benchmark that measures a different collective from the one shipped
    # measures nothing. The result is compared bit-exact; the durations only have to be finite,
    # non-negative and bounded by the call that produced them.
    #
    # The limit: this sees a stage that moves the wrong bytes, not one that drops an ordering the
    # bytes here never depend on. Dropping the OPENING barrier is invisible -- it guards the
    # previous call's readers against this call's writes, and nothing below reads a window twice.
    for mode in (0, 1):
        shape = (B, 16, 4 * ws, 128) if mode == 0 else (B, 16 * ws, 4, 128)
        xt = torch.randn(shape, dtype=torch.bfloat16, device=dev)
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        timed_out, stages = group._timed(xt, mode=mode)
        stop.record()
        torch.cuda.synchronize()
        whole = start.elapsed_time(stop)
        check.equal(
            f"timed mode={mode} moves the same bytes", timed_out, group.all_to_all_4d(xt, mode=mode)
        )
        # The four marks are recorded on one stream, between two events this loop brackets it with,
        # so their sum cannot exceed the whole call; 0.01 ms covers the event clock's resolution.
        # Nothing here requires a stage to be POSITIVE: at ws == 1 fast_barrier launches nothing
        # (src/barrier.cu), so both barrier stages are legitimately zero.
        problems = []
        if list(stages) != ["barrier_in", "transfer", "barrier_out", "copy_out"]:
            problems.append(f"stages are {list(stages)}")
        problems += [f"{k}={v}" for k, v in stages.items() if not math.isfinite(v) or v < 0.0]
        # `transfer` has to be POSITIVE wherever there is a transfer. Without this the whole stage
        # half passes on an implementation that reports four zeros, which is what a timed op that
        # stopped issuing anything would look like.
        if stages["transfer"] <= 0.0:
            problems.append(f"transfer is {stages['transfer']}, but this call moves bytes")
        if sum(stages.values()) > whole + 0.01:
            problems.append(f"stages sum to {sum(stages.values()):.3f} ms of a {whole:.3f} ms call")
        if problems:
            check.failed += 1
            print(f"FAIL rank={rank} timed mode={mode}: {', '.join(problems)}", flush=True)
        elif rank == 0:
            print(f"OK ws={ws} timed mode={mode} reports four stages inside the call", flush=True)
        dist.barrier()

    verdict = torch.tensor([check.failed], device=dev)
    dist.all_reduce(verdict)
    if rank == 0:
        print("ALL PASS" if verdict.item() == 0 else f"FAILED {int(verdict.item())} checks")
    group.destroy()
    dist.destroy_process_group()
    if verdict.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
