"""torchrun worker: is a CAPTURED steady-state call correct on REPLAY?

    torchrun --nproc_per_node=8 test/distributed/a2a_cudagraph.py

Capture succeeding proves nothing, which is what this worker exists to pin down.

The hazard is the barrier epoch. A HOST-computed epoch is a constant baked into the graph at
capture: every replay announces the same E, the peer flags still hold E from the previous replay,
the ``while (v < epoch)`` spin is satisfied by stale state, and the handshake silently becomes a
no-op -- peers are then free to overwrite our window while we are reading it. The reference
implementation shipped that version and its first replay came back with 260k corrupted elements.
``ulysses_barrier_kernel`` advances the epoch ON THE DEVICE (``atomicAdd`` on a per-tag counter,
ulysses_group.cu) precisely so that a replay gets a fresh one. This worker is what keeps that
true.

That failure is invisible when all ranks happen to arrive together -- which is exactly what a
tight benchmark loop does. So the ranks are skewed deliberately: rank 0 runs a long GEMM chain
before replaying, giving the others time to race ahead. The NCCL reference all-to-all that
computes ``want`` is a collective and re-aligns everyone, so it is issued BEFORE the ballast.

Capture failing is REPORTED, not failed: graph capture is not a documented feature of this
extension. A green run with ``captured=False`` on the last line checked NOTHING -- read the line.

BORROWED, because the graph needs a static output tensor and the borrowed form has one for free:
the pool hands back the same window for a given (tag, capacity, dtype) on every call, so the view
taken during the warm-up call is the address every replay writes. The default
``all_to_all_single_4d`` allocates its output inside the call, which under capture comes from the
graph's private pool and is not the tensor the warm-up call returned -- a different worker, not
this one.

NEGATIVE CONTROL, one line, aimed at exactly the bug above: in ``ulysses_barrier_kernel``
(src/ulysses_group.cu) replace

    epoch = atomicAdd(reinterpret_cast<unsigned long long*>(epoch_counter), 1ULL) + 1;
with
    epoch = *epoch_counter + 1;

and rebuild. The counter then never advances, so every barrier announces a value the flags already
hold from the warm-up call and the handshake is dead from replay 1 -- the same end state a baked-in
host epoch reaches (that one survives exactly one replay, then goes stale). Replays should come
back with elements from a neighbouring iteration. Deleting ``group->fast_barrier(stream, tag)`` in
bindings.cpp is the blunter version of the same control. In the OTHER direction, delete the
``if rank == 0:`` ballast loop below: the replays re-align, and the worker then passes even with a
dead barrier -- that is the blindness the skew exists to prevent.

Two checks per replay, and they are not redundant. Torn data is a SUFFICIENT signal that the
handshake died, not a necessary one: it needs a peer to actually be late, so a run where the skew
happens not to bite comes back clean. The epoch counter is the necessary one -- if it does not
advance, the handshake is dead whether or not anything tore this time. The ballast is still what
makes the data check meaningful; the counter is what makes a quiet run trustworthy.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from a2a_correctness import torch_a2a  # sibling worker; torchrun puts this dir on sys.path

from fast_ulysses import UlyssesGroup


def log(msg: str) -> None:
    print(f"[rank {dist.get_rank()}] {msg}", flush=True)


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD
    group = UlyssesGroup(process_group=pg, initial_pool_bytes=1 << 30)

    b, s_local, n_global, d = 1, 256, 4 * ws, 128
    static_in = torch.zeros((b, s_local, n_global, d), dtype=torch.bfloat16, device=dev)
    # Slow enough to skew arrival times by far more than the transfer takes.
    ballast = torch.randn((4096, 4096), dtype=torch.bfloat16, device=dev)

    # Warm the tag: its first use allocates and collectively registers the symmetric buffer and
    # this tag's barrier state (cudaMalloc + an nvshmem barrier), none of which is capturable --
    # capture would fail for an unrelated reason. The returned view is the graph's static output:
    # the pool hands back the same buffer for this (tag, shape, dtype) on every call.
    static_in.copy_(torch.randn_like(static_in))
    static_out = group.all_to_all_single_4d_borrowed(static_in, mode=0, tag="cg")
    torch.cuda.synchronize()
    dist.barrier()

    graph = torch.cuda.CUDAGraph()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    try:
        with torch.cuda.stream(side):
            with torch.cuda.graph(graph):
                group.all_to_all_single_4d_borrowed(static_in, mode=0, tag="cg")
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        captured = True
        log("capture OK")
    except Exception as exc:  # noqa: BLE001
        captured = False
        log(f"capture failed: {type(exc).__name__}: {str(exc)[:160]}")

    bad_replays = 0
    stale_epochs = 0
    if captured:
        dist.barrier()
        prev_epoch = group.barrier_epoch("cg")
        for i in range(8):
            # Fresh values every replay -- same layout, different data, so a stale read shows up
            # as a mismatch instead of as a coincidentally equal buffer.
            fresh = torch.randn(
                (b, s_local, n_global, d),
                dtype=torch.bfloat16,
                device=dev,
                generator=torch.Generator(device=dev).manual_seed(1000 * i + rank),
            )
            static_in.copy_(fresh)
            want = torch_a2a(static_in, 0, ws, pg)  # collective: re-aligns, so it goes first

            if rank == 0:  # skew: rank 0 arrives late, everyone else early
                for _ in range(30):
                    ballast = ballast @ ballast * 1e-4

            graph.replay()
            torch.cuda.synchronize()

            # The epoch directly, not inferred from whether the data tore. A dead handshake only
            # SOMETIMES produces a mismatch -- it needs a peer to actually be late -- so torn data
            # is a sufficient signal and not a necessary one. A frozen counter is necessary.
            epoch = group.barrier_epoch("cg")
            if epoch <= prev_epoch:
                stale_epochs += 1
                if stale_epochs == 1:
                    log(f"replay {i}: epoch did not advance ({prev_epoch} -> {epoch})")
            prev_epoch = epoch

            if not torch.equal(static_out, want):
                bad_replays += 1
                if bad_replays == 1:
                    log(f"replay {i}: {int((static_out != want).sum().item())} elements differ")
        if not bad_replays and not stale_epochs:
            log(f"8 skewed replays matched the reference; epoch advanced to {prev_epoch}")

    verdict = torch.tensor([int(bad_replays > 0 or stale_epochs > 0)], device=dev)
    dist.all_reduce(verdict)
    if rank == 0:
        status = "PASS" if verdict.item() == 0 else "FAIL"
        print(f"CUDAGRAPH_REPLAY {status} (captured={captured})", flush=True)
    group.destroy()
    dist.destroy_process_group()
    if verdict.item():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
