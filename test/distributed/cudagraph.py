"""Is a CAPTURED steady-state call still correct on REPLAY?

    torchrun --nproc_per_node=8 test/distributed/cudagraph.py

Capture succeeding proves nothing, which is what this worker exists to pin down.

The hazard is the barrier epoch. A HOST-computed epoch would be a constant baked into the graph at
capture: every replay announces the same E, the peer flags still hold E from the previous replay,
the ``while (v < epoch)`` spin in barrier_kernel is satisfied by stale state, and the handshake
silently becomes a no-op -- peers are then free to overwrite our window while we are reading it.
That is not a hypothetical: it is the version this replaced, and its replays came back corrupt.
``barrier_kernel`` advances the epoch ON THE DEVICE (src/barrier.cu: ``atomicAdd`` on the slot past
the ws flags) precisely so a replay gets a fresh one, and docs/design.md rests a claim on that.
This worker is what keeps the claim honest.

SCOPE, and it is narrow. Only the SYNC call, and only on a shape that has ALREADY been warmed:

  - a window is allocated by ``empty_strided_p2p`` + ``rendezvous`` + ``zero_()``, none of which is
    legal during capture, so a shape whose window does not exist yet kills the capture outright;
  - the async path is out regardless: ``stage()`` waits on an event recorded by the PREVIOUS,
    uncaptured call (src/group.cc), and a cross-graph event dependency invalidates the capture.

Neither is a bug being hidden -- they are the boundary, and docs/design.md states it.

TWO CHECKS PER REPLAY, and they are not redundant:

  - Torn data is a SUFFICIENT signal that the handshake died, not a necessary one: it needs a peer
    to actually be late. A replay where the skew happens not to bite comes back clean.
  - The epoch is the NECESSARY one, and it is an EQUALITY. Two barriers run per call, so a replay
    moves it by exactly 2; a deleted opening barrier moves it by 1, which passes any "advanced at
    all" form while the window is unguarded for the whole of every call.

THE SKEW IS WHAT MAKES THE DATA CHECK MEAN ANYTHING. Replays that all arrive together re-align, and
a dead barrier then passes -- exactly the blindness a tight benchmark loop produces. So rank 0 runs
a GEMM chain before each replay while the others race ahead. The reference all-to-all that computes
``want`` is itself collective and re-aligns everyone, so it is issued BEFORE the ballast.

Delete the ``if rank == 0`` ballast and this worker passes with the barrier removed: that is the
negative control, and it is why the ballast is not an optimisation to be tidied away.

CAPTURE FAILING IS A FAILURE. Graph capture is a boundary this library states rather than a
feature it sells, but a run that could not capture checked NOTHING, so it exits 1 with the reason
on its last line rather than passing quietly.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from correctness import reference_even  # sibling worker; torchrun puts this dir on sys.path

from fast_ulysses import UlyssesGroup

REPLAYS = 8
BALLAST = 24  # GEMMs on rank 0 per replay; enough to let the others get ahead


def main() -> None:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    dev = torch.device("cuda", torch.cuda.current_device())
    pg = dist.group.WORLD
    if ws < 2:
        raise SystemExit(f"needs >= 2 ranks, got {ws}")
    torch.manual_seed(99 + rank)

    group = UlyssesGroup(process_group=pg, require_nvlink=False)
    x = torch.randn(2, 16, 4 * ws, 128, dtype=torch.bfloat16, device=dev)

    # WARM UP FIRST, outside capture: this is what allocates the window (collective) and builds the
    # plan, so the captured region contains only barrier / copies / barrier.
    out = group.empty_output(x, mode=0)  # collective, and the static address every replay writes
    group.all_to_all_4d(x, mode=0, out=out)
    torch.cuda.synchronize()

    ballast = torch.randn(2048, 2048, dtype=torch.bfloat16, device=dev)

    graph = torch.cuda.CUDAGraph()
    captured, why = True, ""
    try:
        # A side stream, as torch requires for capture, and one more warm-up on it.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            group.all_to_all_4d(x, mode=0, out=out)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        with torch.cuda.graph(graph):
            group.all_to_all_4d(x, mode=0, out=out)
    except Exception as exc:  # noqa: BLE001 -- the point is to report it, whatever it is
        captured, why = False, str(exc).splitlines()[0]

    failed = 0
    if captured:
        epochs = [group._handle.epoch_debug(out, 0)]
        for i in range(REPLAYS):
            # New input every replay, so a stale window is a WRONG answer rather than the same one.
            x.normal_()
            # Collective, and it re-aligns the ranks -- so it goes BEFORE the ballast, never after.
            want = reference_even(x, 0, ws, pg)
            if rank == 0:
                for _ in range(BALLAST):
                    ballast = ballast @ ballast.T * 0.001
            graph.replay()
            torch.cuda.synchronize()
            if not torch.equal(out, want):
                failed += 1
                n = int((out != want).sum().item())
                print(f"FAIL rank={rank} replay {i}: {n} elements differ", flush=True)
            epochs.append(group._handle.epoch_debug(out, 0))

        # The necessary check, and it is an EQUALITY: two barriers per call, so the epoch moves by
        # exactly 2 every replay, and nothing else touches this buffer's window between two samples
        # (the reference is a NCCL collective, the ballast is GEMMs). A constant epoch is a dead
        # handshake even on a replay where nothing happened to tear, and a delta of 1 is one dead
        # barrier -- which "> 0" would accept. Update the constant if the number of barriers per
        # call ever changes; do not loosen the form.
        steps = [b - a for a, b in zip(epochs[:-1], epochs[1:], strict=True)]
        if any(s != 2 for s in steps):
            failed += 1
            print(
                f"FAIL rank={rank}: the epoch moved {steps} over {REPLAYS} replays, expected 2 per "
                f"replay: {epochs}",
                flush=True,
            )
        elif rank == 0:
            print(f"OK ws={ws} epoch advanced {steps} over {REPLAYS} replays", flush=True)

    verdict = torch.tensor([failed], device=dev)
    dist.all_reduce(verdict)
    if rank == 0:
        if not captured:
            print(f"CUDAGRAPH NOT CAPTURED -- this run checked NOTHING: {why}")
        elif verdict.item() == 0:
            print(f"CUDAGRAPH PASS ({REPLAYS} replays, rank 0 skewed by {BALLAST} GEMMs)")
        else:
            print(f"FAILED {int(verdict.item())} checks")
    group.destroy()
    dist.destroy_process_group()
    # An uncaptured run is not a pass, but it is not this library's failure either: report and exit
    # 0 only when something was actually checked.
    if verdict.item() or not captured:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
