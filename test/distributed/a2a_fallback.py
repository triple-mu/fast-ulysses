"""torchrun worker: TorchUlyssesGroup must be interchangeable with UlyssesGroup.

The fallback exists so a caller on a multi-socket PCIe box keeps one code path, which is only
true if the two agree bit for bit on every entry point and every shape family. Both are pure data
movement, so bit-exact is the right bar -- there is no reduction to reorder.

Also checks that make_group's choice follows spans_sockets(), since a silent wrong choice would
look like a performance mystery rather than a bug.

Run:  torchrun --nproc_per_node=8 test/distributed/a2a_fallback.py

NEGATIVE CONTROL: change `_even`'s final permute in fast_ulysses/fallback.py from
(1, 0, 2, 3, 4) to (0, 1, 2, 3, 4) and this fails on the mode-0 even case with a shape or value
mismatch. A pass with that edit applied means the comparison is not running.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist

from fast_ulysses import TorchUlyssesGroup, UlyssesGroup, make_group, spans_sockets


def check(name: str, ours: torch.Tensor, ref: torch.Tensor, failures: list[str]) -> None:
    if ours.shape != ref.shape:
        failures.append(f"{name}: shape {tuple(ours.shape)} != {tuple(ref.shape)}")
    elif not torch.equal(ours, ref):
        bad = (ours != ref).sum().item()
        failures.append(f"{name}: {bad}/{ours.numel()} elements differ")


def main() -> int:
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    torch.manual_seed(1234)  # same input on every rank's own shard, drawn independently below

    failures: list[str] = []
    fast = UlyssesGroup(initial_pool_bytes=1 << 28)
    slow = TorchUlyssesGroup()

    b, d = 2, 64
    n_global, s_local = 4 * ws, 8

    # mode 0 and mode 1, even shards, all four entry points.
    for mode in (0, 1):
        if mode == 0:
            x = torch.randn(b, s_local, n_global, d, dtype=torch.bfloat16, device=dev)
        else:
            x = torch.randn(b, s_local * ws, n_global // ws, d, dtype=torch.bfloat16, device=dev)
        ref = slow.all_to_all_single_4d(x, mode=mode)
        check(
            f"mode{mode} sync",
            fast.all_to_all_single_4d(x, mode=mode, tag=f"s{mode}"),
            ref,
            failures,
        )
        check(
            f"mode{mode} borrowed",
            fast.all_to_all_single_4d_borrowed(x, mode=mode, tag=f"b{mode}").clone(),
            ref,
            failures,
        )
        check(
            f"mode{mode} async",
            fast.all_to_all_single_4d_async(x, mode=mode, tag=f"a{mode}").wait(),
            ref,
            failures,
        )
        check(
            f"mode{mode} slow async",
            slow.all_to_all_single_4d_async(x, mode=mode).wait(),
            ref,
            failures,
        )
        # out= must be honoured by both
        got = torch.empty_like(ref)
        slow.all_to_all_single_4d(x, mode=mode, out=got)
        check(f"mode{mode} slow out=", got, ref, failures)

    # Uneven shards: one extra token on rank 0, which is the case padding-free callers hit.
    seq_splits = [s_local + 1 if r == 0 else s_local for r in range(ws)]
    head_splits = [n_global // ws] * ws
    for mode in (0, 1):
        if mode == 0:
            shape = (b, seq_splits[rank], sum(head_splits), d)
        else:
            shape = (b, sum(seq_splits), head_splits[rank], d)
        x = torch.randn(*shape, dtype=torch.bfloat16, device=dev)
        ref = slow.all_to_all_single_4d(
            x, mode=mode, seq_splits=seq_splits, head_splits=head_splits
        )
        check(
            f"mode{mode} uneven",
            fast.all_to_all_single_4d(
                x, mode=mode, tag=f"u{mode}", seq_splits=seq_splits, head_splits=head_splits
            ),
            ref,
            failures,
        )

    # Only one UlyssesGroup may be live at a time here: live groups have to PARTITION the job, and
    # two groups over the whole world do not. So the comparisons end before make_group is asked
    # for anything that might build a second one.
    fast.destroy()

    # make_group's choice must follow the topology, not a coin flip.
    spans = spans_sockets()
    if make_group(prefer="torch").fallback is not True:
        failures.append("prefer='torch' did not return TorchUlyssesGroup")
    chosen = make_group(initial_pool_bytes=1 << 28)
    if chosen.fallback != bool(spans):
        failures.append(
            f"make_group(auto) returned fallback={chosen.fallback} with spans_sockets()={spans}"
        )
    chosen.destroy()
    forced = make_group(initial_pool_bytes=1 << 28, prefer="fast")
    if forced.fallback is not False:
        failures.append("prefer='fast' did not return UlyssesGroup")
    forced.destroy()

    verdict = torch.tensor(len(failures), dtype=torch.int32, device=dev)
    dist.all_reduce(verdict, op=dist.ReduceOp.SUM)
    if failures:
        print(f"[rank {rank}] FAILURES: " + "; ".join(failures), flush=True)
    dist.barrier()
    if rank == 0:
        print(
            f"FALLBACK {'PASS' if verdict.item() == 0 else 'FAIL'} "
            f"(ws={ws}, spans_sockets={spans})",
            flush=True,
        )
    dist.destroy_process_group()
    return 1 if verdict.item() else 0


if __name__ == "__main__":
    sys.exit(main())
