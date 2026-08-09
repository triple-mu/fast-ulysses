"""Does the operator reject what it says it rejects, and does it reject it BEFORE any collective?

    torchrun --nproc_per_node=8 test/distributed/validation.py

docs/api.md lists what raises. Nothing tested any of it: test_plan.py reaches build_plan through
a2a_plan_debug, which builds A2ADims by hand and never runs make_dims or prepare, and
correctness.py only ever passes valid arguments.

Every rejection here is checked on EVERY rank and the ranks then meet at a barrier. A check that
raised on some ranks and not others, or that raised after the call's first handshake, hangs this
worker instead of failing it, which is the failure mode worth catching.

The aliasing guard compares against the window's CAPACITY rather than this call's requirement.
That is the conservative form, but only the case below is reachable through the public API: the
one window a caller can hold is an empty_output() buffer, and passing it as `out` takes the
zero-copy path, which is exempt by construction. A regression to the requirement form would
therefore not be caught here -- or anywhere -- so leave the capacity form alone.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


class Rejects:
    def __init__(self, rank: int, ws: int) -> None:
        self.rank, self.ws, self.failed = rank, ws, 0

    def raises(self, name: str, fragment: str, fn) -> None:
        """`fn` must raise, the message must contain `fragment`, and every rank must agree."""
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 -- the type is torch's, the message is the contract
            if fragment not in str(exc):
                self.failed += 1
                print(f"FAIL rank={self.rank} {name}: raised, but not for the stated reason: {exc}")
            elif self.rank == 0:
                print(f"OK ws={self.ws} rejects {name}", flush=True)
        else:
            self.failed += 1
            print(f"FAIL rank={self.rank} {name}: accepted", flush=True)
        # If a rejection were not rank-uniform, or happened after the opening handshake, the ranks
        # would already be out of step and this is where it shows.
        dist.barrier()


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())

    group = UlyssesGroup(require_nvlink=False)
    check = Rejects(rank, ws)
    kw16 = {"dtype": torch.bfloat16, "device": dev}

    good = torch.randn(2, 16, 4 * ws, 128, **kw16)

    # --- shape, dtype and mode -----------------------------------------------------------
    check.raises(
        "a 3D input", "must be 4D", lambda: group.all_to_all_4d(torch.randn(2, 16, 128, **kw16))
    )
    # float32 is SUPPORTED now; float64 stands in as a dtype that is still out.
    check.raises("float64", "dtype must be", lambda: group.all_to_all_4d(good.double()))
    check.raises("int32", "dtype must be", lambda: group.all_to_all_4d(good.to(torch.int32)))
    check.raises(
        "a head dim that is not 16B-aligned",
        "16-byte aligned",
        lambda: group.all_to_all_4d(torch.randn(2, 16, 4 * ws, 4, **kw16)),
    )
    # The alignment rule tightens as the element shrinks: 8 halves are 16 B, 8 fp8 are 8 B.
    check.raises(
        "a head dim that is aligned at bfloat16 but not at float8",
        "16-byte aligned",
        lambda: group.all_to_all_4d(torch.randn(2, 16, 4 * ws, 8, **kw16).to(torch.float8_e4m3fn)),
    )
    check.raises("mode 2", "mode must be 0 or 1", lambda: group.all_to_all_4d(good, mode=2))
    check.raises("a CPU input", "must be a CUDA tensor", lambda: group.all_to_all_4d(good.cpu()))

    # --- splits ---------------------------------------------------------------------------
    check.raises(
        "one split list without the other",
        "both seq_splits and head_splits",
        lambda: group.all_to_all_4d(good, seq_splits=[16] * ws),
    )
    check.raises(
        "splits that contradict the shape",
        "but the splits imply",
        lambda: group.all_to_all_4d(good, seq_splits=[99] * ws, head_splits=[4] * ws),
    )
    if (4 * ws) % ws == 0 and ws > 1:
        odd = torch.randn(2, 16, 4 * ws + 1, 128, **kw16)
        check.raises(
            "a head axis that does not divide",
            "must divide world_size",
            lambda: group.all_to_all_4d(odd),
        )

    # --- an empty collective: window_numel would be 0, which is not an allocation ----------
    check.raises(
        "a shape with nothing to exchange",
        "moves no data",
        lambda: group.all_to_all_4d(torch.empty(2, 16, 0, 128, **kw16)),
    )
    check.raises(
        "empty_output for a shape with nothing to exchange",
        "moves no data",
        lambda: group.empty_output(torch.empty(2, 0, 4 * ws, 128, **kw16)),
    )

    # --- out= ------------------------------------------------------------------------------
    want_shape = (2, 16 * ws, 4, 128)
    check.raises(
        "out with the wrong shape",
        "out has shape",
        lambda: group.all_to_all_4d(good, out=torch.empty(2, 3, 4, 128, **kw16)),
    )
    check.raises(
        "out with the wrong dtype",
        "out has dtype",
        lambda: group.all_to_all_4d(
            good, out=torch.empty(*want_shape, dtype=torch.float16, device=dev)
        ),
    )
    check.raises(
        "a non-contiguous out",
        "contiguous CUDA tensor",
        lambda: group.all_to_all_4d(good, out=torch.empty(2, 16 * ws, 4, 256, **kw16)[..., ::2]),
    )

    # --- the aliasing guard -----------------------------------------------------------------
    # The one shape a caller can actually reach: `out` is a window from empty_output(), so the
    # peers write it directly, and the input is a view of that SAME window. Reshaping the window
    # to the input whose output shape is exactly the window's shape works at any world size,
    # unlike feeding it back in as itself, whose two shapes only coincide at ws == 1.
    window = group.empty_output(good, mode=0)  # (2, 16*ws, 4, 128)
    check.raises(
        "an input that is a view of the window it fills",
        "input overlaps the window",
        lambda: group.all_to_all_4d(window.view(2, 16, 4 * ws, 128), mode=0, out=window),
    )

    # --- autograd -----------------------------------------------------------------------------
    # The async result is an AsyncCollectiveTensor, which is a leaf: backward() through it would
    # run to completion and leave x.grad as None. Refused rather than silently wrong.
    check.raises(
        "an async call on an input that requires grad",
        "does not support autograd",
        lambda: group.all_to_all_4d_async(good.detach().requires_grad_(True)),
    )

    # --- the plan cache must not answer for a call it never saw ------------------------------
    # An even-split call and an EMPTY-split one differ only in whether the lists are present. Held
    # as bare vectors the two keys were identical, so the second inherited the first's plan instead
    # of its own rejection. Warming the cache first is what makes this check mean anything.
    group.all_to_all_4d(good, mode=0)
    check.raises(
        "empty split lists, after an even-split call cached its plan",
        "seq_splits has 0 entries",
        lambda: group.all_to_all_4d(good, mode=0, seq_splits=[], head_splits=[]),
    )

    # --- a destroyed group --------------------------------------------------------------------
    # `out` from empty_output() takes the zero-copy path, which reaches neither window() nor
    # make_output() -- the only two places that used to check. Without a check in prepare() this
    # ran the entire collective on a destroyed group and rebuilt the transfer stream it had just
    # torn down, leaking it. Held from before destroy(), since empty_output() checks too.
    dead_out = group.empty_output(good, mode=0)
    group.destroy()
    check.raises(
        "a zero-copy call on a destroyed group",
        "has been destroyed",
        lambda: group.all_to_all_4d(good, mode=0, out=dead_out),
    )
    check.raises(
        "a copying call on a destroyed group",
        "has been destroyed",
        lambda: group.all_to_all_4d(good, mode=0),
    )

    verdict = torch.tensor([check.failed], device=dev)
    dist.all_reduce(verdict)
    if rank == 0:
        print("ALL PASS" if verdict.item() == 0 else f"FAILED {int(verdict.item())} checks")
    dist.destroy_process_group()
    if verdict.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
