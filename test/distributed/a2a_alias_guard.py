"""torchrun worker: does the operator refuse a call that reads and writes the same window?

    torchrun --nproc_per_node=2 test/distributed/a2a_alias_guard.py

A borrowed result IS the tag's symmetric window. Feeding it back under the same tag hands the
transport the very buffer every peer is about to write, which is silent corruption rather than a
crash. bindings.cpp refuses it -- and the interesting question is what "it" covers.

The check used to be ``x.data_ptr() != buf.sym_base``, pointer equality against the base. Two
things slipped through:

  * a borrowed result SLICED on its batch axis. ``y[1:2]`` is contiguous, so nothing copies it
    out of the window, and its data_ptr is past the base -- inside the window, unequal to it.
  * a borrowed result passed as ``out=``. It has exactly the right shape by construction, so
    validation accepted it, and the copy-out then read and wrote the same bytes. There was no
    check on the output side at all.

Both are byte-interval overlaps, so both are now rejected by an interval test.

Every case must raise BEFORE anything reaches the stream, so all ranks raise together and none
is left in a barrier -- these run inside the loop, and a rank that hung instead of raising would
take the whole job to the harness timeout rather than failing here.

NEGATIVE CONTROL: in bindings.cpp, replace the body of ``check_window_aliasing`` with the old
pointer test (``TORCH_CHECK(prepared.x.data_ptr() != buf.sym_base, ...)``) and rebuild. Case
"whole borrowed result" must still be refused -- it is the one the old check caught -- while
"sliced borrowed result" and "borrowed result as out" must both stop raising. A build where all
three still raise is not running the code this worker is about.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup

TAG = "alias"


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    if ws < 2:
        raise SystemExit(f"needs >= 2 ranks, got {ws}")

    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)
    b, s_me, d = 2, 8, 128
    x = torch.randn((b, s_me, 4 * ws, d), dtype=torch.bfloat16, device=dev)

    # One borrowed call to populate the window and hand back a view of it. Everything below
    # aliases THIS.
    y = group.all_to_all_single_4d_borrowed(x, mode=0, tag=TAG)

    cases = {
        # The case the pointer test already caught: y starts exactly at the window base.
        "whole borrowed result": lambda: group.all_to_all_single_4d_borrowed(y, mode=1, tag=TAG),
        # Slicing the outermost axis of a contiguous tensor stays contiguous, so .contiguous()
        # inside prepare() is a no-op and this really does reach the transport as window memory.
        "sliced borrowed result": lambda: group.all_to_all_single_4d_borrowed(
            y[1:2], mode=1, tag=TAG
        ),
        # The output side, which had no check at all: copy-out would read and write the same bytes.
        "borrowed result as out": lambda: group.all_to_all_single_4d(x, mode=0, tag=TAG, out=y),
    }

    fails = 0
    for name, call in cases.items():
        try:
            call()
        except Exception as exc:  # noqa: BLE001 -- the type is torch's, the message is the contract
            if "overlaps the window" not in str(exc):
                fails += 1
                print(
                    f"FAIL rank={rank} [{name}]: raised, but not the aliasing check: {exc}",
                    flush=True,
                )
            continue
        fails += 1
        print(f"FAIL rank={rank} [{name}]: accepted a call that aliases the window", flush=True)

    # ...and a call that does NOT alias must still work, or the guard is just rejecting everything.
    ok = group.all_to_all_single_4d(x, mode=0, tag="clean")
    if ok.shape != (b, s_me * ws, 4, d):
        fails += 1
        print(f"FAIL rank={rank}: non-aliasing call returned {tuple(ok.shape)}", flush=True)

    nfail = torch.tensor([fails], device=dev)
    dist.all_reduce(nfail)
    if rank == 0:
        print(
            "ALIAS_GUARD " + ("PASS" if nfail.item() == 0 else f"FAILED {int(nfail.item())}"),
            flush=True,
        )
    group.destroy()
    dist.destroy_process_group()
    if nfail.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
