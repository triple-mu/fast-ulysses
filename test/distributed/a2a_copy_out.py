"""torchrun worker: the copying form owns its result, the borrowed form does not.

    torchrun --nproc_per_node=8 test/distributed/a2a_copy_out.py

The two entry points return the same numbers and differ in exactly one property: who owns the
memory. That property is invisible in any single call -- both look like a correct tensor -- and
only shows up on the SECOND call with the same tag, which overwrites the window. So that is what
this worker builds, twice:

  * ``all_to_all_single_4d`` twice on one tag: the first result must still hold its own value.
  * ``all_to_all_single_4d_borrowed`` twice on one tag: the first result must now hold the
    SECOND call's value, at the same address.

The borrowed half DELIBERATELY BREAKS the contract in that method's docstring (consume the result
before the next call with its tag). That is the point: it pins down what the rule protects
against, so nobody deletes the copy-out believing the two forms are interchangeable. Do not
"fix" it by cloning.

Constants are 1.0 and 2.0 because bfloat16 represents both exactly, so a mixed buffer is
unmistakable rather than plausible rounding.

Also covers ``out=``: a caller-supplied destination is filled and handed back, and the four
things that make one invalid (not CUDA, not contiguous, wrong dtype, wrong shape) are rejected.
Every rejection is raised while validating arguments, before the call issues anything to the
stream, so all ranks raise together and none is left waiting in a barrier.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)

    fails = 0

    def check(name, ok):
        nonlocal fails
        fails += not ok
        if rank == 0:
            print(f"{'OK  ' if ok else 'FAIL'} ws={ws} {name}", flush=True)
        dist.barrier()

    b, s_local, d = 1, 64, 128
    n_global = 4 * ws
    dtype = torch.bfloat16
    out_shape = (b, s_local * ws, n_global // ws, d)  # mode 0
    x1 = torch.full((b, s_local, n_global, d), 1.0, dtype=dtype, device=dev)
    x2 = torch.full((b, s_local, n_global, d), 2.0, dtype=dtype, device=dev)

    # Warm both tags: the first use of a tag allocates its symmetric buffer and barrier state
    # collectively, which has nothing to do with what is being checked below.
    group.all_to_all_single_4d(x1, mode=0, tag="own")
    group.all_to_all_single_4d_borrowed(x1, mode=0, tag="lent")
    torch.cuda.synchronize()
    dist.barrier()

    # A copied result is the caller's memory, so the second call cannot reach it.
    y1 = group.all_to_all_single_4d(x1, mode=0, tag="own")
    y2 = group.all_to_all_single_4d(x2, mode=0, tag="own")
    check("copied result survives the next call on its tag", bool((y1 == 1.0).all()))
    check("copied result of the second call is its own", bool((y2 == 2.0).all()))

    # A borrowed result IS the window, so the second call overwrites it -- same address, and
    # the first handle now reads the second call's payload.
    z1 = group.all_to_all_single_4d_borrowed(x1, mode=0, tag="lent")
    z2 = group.all_to_all_single_4d_borrowed(x2, mode=0, tag="lent")
    check("borrowed results share one window", z1.data_ptr() == z2.data_ptr())
    check("borrowed result IS clobbered by the next call on its tag", bool((z1 == 2.0).all()))

    # Same answer either way, on tags that stay live together.
    yc = group.all_to_all_single_4d(x1, mode=0, tag="cmp_copy")
    zb = group.all_to_all_single_4d_borrowed(x1, mode=0, tag="cmp_lent")
    check("copying and borrowing agree bitwise", torch.equal(yc, zb))

    # out=: filled in place and returned.
    dst = torch.empty(out_shape, dtype=dtype, device=dev)
    got = group.all_to_all_single_4d(x2, mode=0, tag="dst", out=dst)
    check(
        "out= is filled and returned",
        got.data_ptr() == dst.data_ptr() and bool((dst == 2.0).all()),
    )

    # out= rejections. Each is a single defect, so a passing case cannot mask a missing check.
    bad = {
        "wrong dtype": (torch.empty(out_shape, dtype=torch.float16, device=dev), "out has dtype"),
        "wrong shape": (
            torch.empty((b, s_local * ws + 1, n_global // ws, d), dtype=dtype, device=dev),
            "out has shape",
        ),
        "non-contiguous": (
            torch.empty((b, n_global // ws, s_local * ws, d), dtype=dtype, device=dev).transpose(
                1, 2
            ),
            "contiguous CUDA tensor",
        ),
        "on the cpu": (torch.empty(out_shape, dtype=dtype), "contiguous CUDA tensor"),
    }
    for name, (dst_bad, want) in bad.items():
        rejected = False
        try:
            group.all_to_all_single_4d(x1, mode=0, tag="rej", out=dst_bad)
        except RuntimeError as e:
            rejected = want in str(e)
        check(f"out= rejected: {name}", rejected)

    nfail = torch.tensor([fails], device=dev)
    dist.all_reduce(nfail)
    if rank == 0:
        verdict = "PASS" if nfail.item() == 0 else f"FAIL {int(nfail.item())}"
        print(f"COPY_OUT {verdict}", flush=True)
    group.destroy()
    dist.destroy_process_group()
    if nfail.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
