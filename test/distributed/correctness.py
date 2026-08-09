"""Is the result bit-exact against torch.distributed, on every path?

    torchrun --nproc_per_node=8 test/distributed/correctness.py

Pure data movement, so "close enough" is not a passing result anywhere in this file. Covers both
modes, both dtypes, even and uneven shards, the three things ``out=`` can be, the async form, and
window reuse across repeated calls.
"""

from __future__ import annotations

import os
from itertools import accumulate

import torch
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
    got = group.all_to_all_4d(x, mode=0, out=plain)
    check.equal("out=plain tensor", got, want)
    if got.data_ptr() != plain.data_ptr():
        check.failed += 1
        print(f"FAIL rank={rank} out=plain tensor: the result is not the tensor handed in")

    window = group.empty_output(x, mode=0)
    got = group.all_to_all_4d(x, mode=0, out=window)
    check.equal("out=empty_output (zero copy)", got, want)
    if got.data_ptr() != window.data_ptr():
        check.failed += 1
        print(f"FAIL rank={rank} out=empty_output: the result is not the buffer handed in")

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
    # Every round after the first must hit the cache. A round that missed would call rendezvous,
    # which is collective -- if the ranks ever disagreed about when that happens, this hangs.
    for _ in range(20):
        check_out = group.all_to_all_4d(x, mode=0)
    check.equal("20 rounds on one window", check_out, want)

    # --- the dtypes that are not float16/bfloat16 -------------------------------------------
    # The transport is byte-oriented all the way down, so these differ from the sweep above only
    # in element size. Compared as bytes: all_to_all_single carries no float8, and the reference
    # for a permutation does not need to know what the bytes mean.
    for dtype in (torch.float32, torch.float8_e4m3fn, torch.float8_e5m2, torch.int8):
        seed = torch.randn(B, 16, 4 * ws, 128, dtype=torch.float32, device=dev)
        xd = seed.to(dtype) if dtype != torch.int8 else (seed * 50).to(torch.int8)
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

    # `out=` must not silently detach: the out-variant is not differentiable, so a grad-requiring
    # input has to come back with grad_fn absent rather than with a gradient that goes nowhere.
    xo = torch.randn(B, 16, 4 * ws, 128, dtype=torch.bfloat16, device=dev, requires_grad=True)
    res = group.all_to_all_4d(xo, mode=0, out=torch.empty_like(want))
    if res.grad_fn is not None or res.requires_grad:
        check.failed += 1
        print(f"FAIL rank={rank} out= claims to be differentiable but the op is not")
    elif rank == 0:
        print(f"OK ws={ws} out= does not claim a gradient it cannot deliver", flush=True)
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
