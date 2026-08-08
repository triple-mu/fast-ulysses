"""torchrun worker: can this extension share a process with torch's own NVSHMEM?

NVSHMEM is a process-global singleton. This extension initialises it itself
(`nvshmemx_hostlib_init_attr` with a broadcast unique id) and finalises it when the last group
is destroyed. torch 2.13 ALSO ships NVSHMEM -- `nvidia-nvshmem-cu13`, the very library we link
-- and `torch.distributed._symmetric_memory.is_nvshmem_available()` returns True, so a real
sglang process may well have brought it up before we get there.

Nobody has tested what happens then. The failure modes worth distinguishing:

  * torch's init is picked up and ours is a no-op          -- fine
  * ours re-initialises over torch's                        -- torch's symmetric memory breaks
  * the second init errors                                  -- fine if the message says so
  * the second init hangs                                   -- the worst case, and plausible,
    because NVSHMEM's bootstrap is collective
  * ours FINALISES on destroy() while torch still needs it  -- breaks torch later, silently

So this does not assert a particular outcome. It establishes WHICH outcome, in order, and
fails only on the ones that are actually broken. Read the printed transcript.

Run:  torchrun --nproc_per_node=4 test/distributed/a2a_torch_nvshmem_coexist.py
"""

from __future__ import annotations

import os
import sys
import traceback

import torch
import torch.distributed as dist


def log(msg: str) -> None:
    print(f"[rank {dist.get_rank()}] {msg}", flush=True)


def main() -> int:
    rank = int(os.environ["RANK"])
    n_dev = torch.cuda.device_count()
    torch.cuda.set_device(rank % n_dev)
    dev = torch.device("cuda", rank % n_dev)
    dist.init_process_group(backend="nccl", device_id=dev)
    failures = []

    import torch.distributed._symmetric_memory as sm

    if rank == 0:
        log(f"torch reports is_nvshmem_available() = {sm.is_nvshmem_available()}")

    # 1. Bring torch's NVSHMEM up FIRST, the way a real process would: allocate a symmetric
    #    tensor through torch's own API and rendezvous it.
    torch_sym = None
    try:
        group_name = dist.group.WORLD.group_name
        torch_sym = sm.empty(1024, dtype=torch.float32, device=dev)
        sm.rendezvous(torch_sym, group_name)
        torch_sym.fill_(float(rank))
        if rank == 0:
            log("torch symmetric memory: up")
    except Exception as exc:  # noqa: BLE001 -- we are classifying, not handling
        if rank == 0:
            log(
                f"torch symmetric memory unavailable ({type(exc).__name__}: {exc}); "
                "the coexistence question is moot on this build"
            )
        dist.destroy_process_group()
        print("TORCH_NVSHMEM_COEXIST SKIP", flush=True) if rank == 0 else None
        return 0
    dist.barrier()

    # 2. Now build ours on top of it. A hang here is the failure this test exists for -- the
    #    harness timeout is what catches it, so keep the window between the barriers tight.
    from fast_ulysses import UlyssesGroup

    group = None
    try:
        group = UlyssesGroup(device=dev, initial_pool_bytes=1 << 28)
        if rank == 0:
            log("UlyssesGroup constructed alongside torch's NVSHMEM")
    except Exception:  # noqa: BLE001
        if rank == 0:
            log("UlyssesGroup construction RAISED:\n" + traceback.format_exc())
        failures.append("construction failed alongside torch's NVSHMEM")
    dist.barrier()

    # 3. Ours works.
    if group is not None:
        ws = dist.get_world_size()
        x = torch.randn((1, 16, 4 * ws, 64), dtype=torch.bfloat16, device=dev)
        try:
            group.all_to_all_single_4d(x, mode=0, tag="coexist")
            if rank == 0:
                log("our collective ran")
        except Exception:  # noqa: BLE001
            if rank == 0:
                log("our collective RAISED:\n" + traceback.format_exc())
            failures.append("collective failed alongside torch's NVSHMEM")
        dist.barrier()

    # 4. torch's symmetric memory still works AFTER ours has been up. If our init reset the
    #    heap under it, this is where it shows.
    try:
        torch_sym.fill_(float(rank) + 100.0)
        torch.cuda.synchronize()
        if not (torch_sym == float(rank) + 100.0).all():
            failures.append("torch's symmetric tensor is not writable after our init")
        elif rank == 0:
            log("torch's symmetric memory still usable after our init")
    except Exception:  # noqa: BLE001
        if rank == 0:
            log("torch's symmetric memory BROKE after our init:\n" + traceback.format_exc())
        failures.append("torch's symmetric memory broke after our init")
    dist.barrier()

    # 5. Our destroy() finalises NVSHMEM when the last group goes. Does torch survive it?
    if group is not None:
        group.destroy()
        dist.barrier()
        try:
            torch_sym.fill_(float(rank) + 200.0)
            torch.cuda.synchronize()
            if not (torch_sym == float(rank) + 200.0).all():
                failures.append("torch's symmetric tensor is not writable after our destroy()")
            elif rank == 0:
                log("torch's symmetric memory still usable after our destroy()")
        except Exception:  # noqa: BLE001
            if rank == 0:
                log(
                    "torch's symmetric memory BROKE after our destroy():\n" + traceback.format_exc()
                )
            failures.append("torch's symmetric memory broke after our destroy()")

    verdict = torch.tensor(len(failures), dtype=torch.int32, device=dev)
    dist.all_reduce(verdict, op=dist.ReduceOp.SUM)
    if failures:
        log("FAILURES: " + "; ".join(failures))
    dist.barrier()
    if rank == 0:
        print("TORCH_NVSHMEM_COEXIST " + ("PASS" if verdict.item() == 0 else "FAIL"), flush=True)
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


if __name__ == "__main__":
    sys.exit(main())
