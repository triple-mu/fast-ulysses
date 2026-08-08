"""torchrun worker: an outstanding async call plus a sync call on another tag, streams unordered.

    torchrun --nproc_per_node=8 test/distributed/a2a_overlapping_barriers.py

The two calls run on different streams -- the group's comm stream and the caller's -- and nothing
orders them, so both barrier kernels are resident at once. The reference implementation reproduced
two distinct bugs in exactly this shape, which is why its barrier state ended up PER TAG:

  * with ONE epoch counter for the group and a plain read-modify-write, both kernels can claim the
    same epoch and one handshake becomes a no-op. 4 ranks x 2000 iterations HANGS -- a rank that
    collides falls permanently one epoch behind, and the first peer to reach a barrier it can no
    longer satisfy spins forever.
  * making the counter atomic fixed the hang and not the race. Epochs were then unique but not
    consistently ORDERED: rank X could give its async barrier epoch 3 and its sync barrier 4 while
    rank Y did the reverse, so X waited on a flag Y had published for the other collective. Torn
    results at 4 ranks on a PCIe box, iteration 126.

fast_ulysses now keeps the barrier state per tag as well -- flags and epoch counter both live in
``UlyssesGroup::BarrierState``, keyed by tag (ulysses_group.cu). Per tag there is nothing to
interleave, because a tag's calls are ordered by the contract; across tags nothing is shared. This
worker is what keeps that true.

BORROWED RESULTS ON BOTH SIDES, deliberately. A swallowed handshake shows up as a read of the
window before the peers' writes for that call have landed, and the borrowed forms put that read
where the check is -- directly on the window. The copying forms would put it in the copy-out
instead: still a read at the wrong time, but one whose result the check only sees second-hand,
through a copy that has itself given the writes more time to drain. The controls below also need
``barrier=False``, which only the borrowed form has.

THIS WORKER IS THE EVIDENCE FOR THE DOCUMENTED CONTRACT. comm.py's
``all_to_all_single_4d_async`` docstring and docs/API.md used to say "barrier kernels must execute
in submission order (one per-group epoch), so wait() every outstanding async handle before the
next sync collective" -- written when the epoch WAS per group. They now state the contract the
code implements: ordered within a tag, unordered across tags. Reading a result: a pass is what
that contract rests on; torn results mean it is wrong and the per-tag split is incomplete; a HANG
(600 s timeout) means a rank fell behind an epoch it can never satisfy. Do not relax the
documented contract on anything weaker than a run of this worker.

Deviation from the reference case: it breaks out of the loop on the first tear. Here the loop runs
to the end, because a rank that leaves early stops issuing collectives and its peers hang in
``fast_barrier`` -- a timeout instead of a reported failure.

NEGATIVE CONTROL, one line, and it is the reference's own history: make the barrier state shared
again by passing a constant instead of the tag at the ``group->fast_barrier(stream, tag)`` call
site in src/bindings.cpp (e.g. ``group->fast_barrier(stream, "")``), and rebuild.
Both tags then share one flag array and one epoch counter, which is the configuration the
reference measured: torn results with an atomic counter, and an outright HANG with a plain
read-modify-write. Cheaper control, no rebuild: replace the sync
``group.all_to_all_single_4d_borrowed(k, ...)`` call below with
``group.all_to_all_single_4d_borrowed_async(k, mode=0, tag="ob_k", barrier=False).wait()`` -- that
handle's wait orders this rank's own work only, so k is read with no handshake at all and ``k_out`` must
report a neighbouring iteration's constant. If neither control produces a failure, the loop is
blind and a pass means nothing.
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
    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)

    b, s_local, d = 1, 32, 32
    n_global = 4 * ws

    # Warm both tags: a tag's first use allocates its symmetric buffer AND its barrier state, whose
    # init runs an on-stream nvshmem barrier. Doing that here, serially on one stream, keeps the
    # loop below from also being a test of two tags registering concurrently on two streams.
    warm = torch.zeros((b, s_local, n_global, d), dtype=torch.bfloat16, device=dev)
    group.all_to_all_single_4d_borrowed(warm, mode=0, tag="ob_q")
    group.all_to_all_single_4d_borrowed(warm, mode=0, tag="ob_k")
    torch.cuda.synchronize()
    dist.barrier()

    iters, bad_q, bad_k, first_bad = 2000, 0, 0, 0

    # Distinct constants per iteration AND per tag, so a swallowed barrier shows up as a value
    # from the wrong call rather than as plausible-looking noise. The constant CYCLES in 1..128
    # instead of being the iteration number: bfloat16 has 8 significant bits, so it represents
    # integers exactly only up to 256 and `float(i)` for i>256 collides with its own neighbours:
    # spacing is 2 there, so 259, 260 and 261 all land on 260, and above 1024 spacing is 8 and up
    # to eight consecutive i share one value.
    # The reference worker used float(i) over 2000 iterations and is blind for the last 87% of its
    # run for exactly that reason -- it reproduced its bugs at iteration 126, so it never showed.
    # A tear carries a value from an ADJACENT call, and 128 distinct constants separate those.
    for i in range(1, iters + 1):
        v = float(1 + i % 128)
        q = torch.full((b, s_local, n_global, d), v, dtype=torch.bfloat16, device=dev)
        k = torch.full((b, s_local, n_global, d), -v, dtype=torch.bfloat16, device=dev)

        handle = group.all_to_all_single_4d_borrowed_async(q, mode=0, tag="ob_q")
        # No wait here: that is the whole point. This runs on the caller's stream while the call
        # above is still going on the comm stream.
        k_out = group.all_to_all_single_4d_borrowed(k, mode=0, tag="ob_k")
        q_out = handle.wait()

        if not (q_out == v).all():
            bad_q += 1
            first_bad = first_bad or i
        if not (k_out == -v).all():
            bad_k += 1
            first_bad = first_bad or i
        if first_bad == i:
            log(
                f"iteration {i}: async saw {torch.unique(q_out.float()).tolist()[:4]} "
                f"(want {v}), sync saw {torch.unique(k_out.float()).tolist()[:4]} (want {-v})"
            )

    if bad_q or bad_k:
        log(
            f"FAILURE: {bad_q} async and {bad_k} sync results carried another call's value, "
            f"first at iteration {first_bad}"
        )
    else:
        log(f"{iters} interleaved async/sync pairs on unordered streams stayed coherent")

    verdict = torch.tensor([bad_q + bad_k], device=dev)
    dist.all_reduce(verdict)
    if rank == 0:
        print("OVERLAPPING_BARRIERS " + ("PASS" if verdict.item() == 0 else "FAIL"), flush=True)
    group.destroy()
    dist.destroy_process_group()
    if verdict.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
