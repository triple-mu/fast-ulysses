"""torchrun worker: is the copy-engine payload visible when the flag announcing it arrives?

    torchrun --nproc_per_node=8 test/distributed/a2a_ce_flag_ordering.py

A call writes the payload into the peers' windows with ``cudaMemcpy2DAsync`` -- copy engines --
and then announces itself with a ``st.release.sys`` store from the one-block barrier kernel on the
same stream. The reader spins on that flag with ``ld.acquire.sys`` and then reads the payload.

Two different engines write to the same remote memory. Stream order says the copy has COMPLETED
before the kernel launches, but "completed" is a statement about the source; whether it means the
bytes are visible at the DESTINATION is the question. Carried over from the reference
implementation, which loaded the same question: the CUDA API reference defines a memcpy's
completion purely as a host-side property; the Programming Guide's one cross-device ordering
guarantee (3.4.2.1) is scoped to the NULL stream and to when commands START, and the next sentence
withdraws it for an async copy in a non-default stream; PTX 8.5's ``.release`` covers "prior
operations from the current thread" and ``.sys`` scope is a set of THREADS -- a copy-engine
transfer is neither, so the barrier kernel's release does not cover the payload at all. Neither
NVSHMEM's nor NCCL's CE paths use this shape: all of them keep the peer-visible flag on the data's
path (``cuStreamWriteValue64``, or a ``cudaMemcpyAsync`` of the flag on the transfer stream).
So a pass is EVIDENCE for this machine and this shape, not a proof. The documented-safe change is
to write the flag with a ``cudaMemcpyAsync`` on the transfer stream instead of from the kernel.

If the ordering does not hold, a reader that is ALREADY WAITING when the flag lands sees the
previous call's payload in part of the buffer. The worker maximises exactly that:

  * BORROWED results -- ``all_to_all_single_4d_borrowed``, whose result is the symmetric-heap view
    itself, and the check reads it directly. Never ``.clone()`` it here: a device-to-device copy
    between the barrier and the first read is time the writes could use to drain, which would hide
    the very thing being tested. That is also why the default ``all_to_all_single_4d`` is wrong for
    this worker -- its copy-out is exactly that copy, sitting exactly there.
  * A LARGE payload, so the copy engine is still busy when the flag is issued.
  * SKEW IN THE TRANSFER, via UNEVEN SHARDS. In mode 0 a rank sends `s_me * head_total * d`, so
    giving rank 0 most of the sequence makes its copies several times longer than everyone
    else's. They reach the CLOSING barrier while it is still copying and read the instant its
    flag arrives.
  * A distinct constant per iteration, so a stale byte is unmistakable rather than plausible. It
    CYCLES in 1..128 rather than being the iteration number, because bfloat16 has 8 significant
    bits: ``float(i)`` is exact only up to 256 and above that adjacent iterations collide on one
    bf16 value, which is precisely the comparison a tear has to survive. (The reference worker
    used ``float(i)`` over 300 iterations and is blind from 257 on; it failed at iteration 2, so
    it never showed.)

WHY THE SKEW IS ON THE TRANSFER AND NOT BEFORE THE CALL. An earlier version of this worker held
rank 0 back with a ballast GEMM chain BEFORE the call, and said so in a docstring that also named
its own expiry condition: that it would go blind if an opening barrier were ever added. One was
(bindings.cpp, writers-wait-for-readers), and this file was not updated with it -- so for several
commits this worker passed while testing nothing, and bindings.cpp and docs/API.md went on citing
it as the evidence for an undocumented assumption. An audit caught it.

The mechanism is simple once seen: the call now OPENS with a handshake, which re-aligns every rank
before any data moves, so an arrival skew is absorbed and equal shards then copy for equal times.
The reference hit exactly this and recorded it (custom_nccl_op/test/distributed/
run_ce_flag_ordering.py) -- its first version "passed even with the closing barrier deleted --
proof that it was testing nothing". Only a skew the opening barrier CANNOT absorb works, and that
means making the transfers themselves unequal.

Deviation from the reference worker: it breaks out of the loop on the first tear. Here the loop
runs to the end, because a rank that leaves early stops issuing collectives and its peers hang in
``fast_barrier`` -- a 600 s timeout instead of a reported failure.

NEGATIVE CONTROL: delete the CLOSING ``group->fast_barrier(stream, tag)`` in
src/bindings.cpp -- the one under ``if (barrier)``, NOT the opening one -- and
rebuild. The reference's equivalent control failed at iteration 2 with 62.9M stale elements
reading [i-1, i]. THIS CONTROL MUST BE RE-RUN whenever the barriers change; the previous version
of this worker was left passing a control it could no longer fail.

The one-line variant that used to be offered here (``barrier=False``) is no longer a control:
``barrier=False`` defers only the closing handshake, and the opening one still runs, so the ranks
stay aligned. Deleting the closing barrier in C++ and rebuilding is the only valid control now.

This worker uses the BORROWED form deliberately: the copying entry point would put a
device-to-device pass between the barrier and the first read, which is time the peers'
writes could use to drain -- it would hide the very thing being tested.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def log(msg: str) -> None:
    print(f"[rank {dist.get_rank()}] {msg}", flush=True)


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=2 << 30)

    # ~200 MB per rank at ws=4 (~400 MB at ws=8): the copy engines are still draining when the
    # barrier kernel publishes the flag.
    # Rank 0 takes most of the sequence, so its transfer runs several times longer than the
    # others' and they wait for it at the closing barrier. In mode 0 a rank sends
    # s_me * head_total * d, so this is the axis that makes the transfers unequal.
    b, s_total, d = 1, 8192 * ws, 384
    base = s_total // (2 * ws)
    seq_splits = [base] * ws
    seq_splits[0] = s_total - base * (ws - 1)
    head_splits = [8] * ws
    kw = {"seq_splits": seq_splits, "head_splits": head_splits}
    x = torch.empty((b, seq_splits[rank], 8 * ws, d), dtype=torch.bfloat16, device=dev)
    mb = x.numel() * x.element_size() / 1e6

    # Warm the tag: the first use allocates and collectively registers the symmetric buffer, which
    # serialises the ranks and would hide the skew this worker depends on.
    x.fill_(0.0)
    group.all_to_all_single_4d_borrowed(x, mode=0, tag="ord", **kw)
    torch.cuda.synchronize()
    dist.barrier()

    iters, stale, first_bad = 300, 0, 0
    for i in range(1, iters + 1):
        v = float(1 + i % 128)  # bf16-exact and distinct from every neighbouring call; see above
        x.fill_(v)

        # No arrival skew: the opening barrier would absorb it. The skew is in the shards --
        # rank 0's transfer is several times longer, so the others reach the closing barrier
        # while it is still copying.

        y = group.all_to_all_single_4d_borrowed(x, mode=0, tag="ord", **kw)
        # Enqueued directly behind the barrier on this stream, reading the window itself. The host
        # sync inside .item() happens after this comparison has already read it, so it cannot mask
        # a stale read.
        bad = int((y != v).sum().item())
        if bad:
            stale += bad
            if not first_bad:
                first_bad = i
                seen = torch.unique(y.float()).tolist()[:6]
                log(f"iteration {i}: {bad} elements are not {v}; saw {seen}")

    if stale:
        log(
            f"FAILURE: copy-engine payload was not visible when the flag arrived -- {stale} "
            f"stale elements, first at iteration {first_bad}"
        )
    else:
        log(f"{iters} skewed calls, {mb:.0f} MB per call on this rank, stayed coherent")

    verdict = torch.tensor([int(stale > 0)], device=dev)
    dist.all_reduce(verdict)
    if rank == 0:
        print("CE_FLAG_ORDER " + ("PASS" if verdict.item() == 0 else "FAIL"), flush=True)
    group.destroy()
    dist.destroy_process_group()
    if verdict.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
