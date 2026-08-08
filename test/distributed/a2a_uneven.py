"""torchrun correctness check for CALLER-SUPPLIED shards (seq_splits / head_splits).

The reference is ``dist.all_to_all`` over a LIST of differently-shaped tensors -- the one
collective that has no divisibility assumption in it at all -- so nothing here is compared
against our own restatement of the layout. Pure data movement, so results must be bit-exact.

Two shapes matter:
  * the production one: an unpadded sequence, i.e. shards differing by at most one token. This
    is what sglang's build_shard_plan produces once its pad-to-a-multiple-of-sp_size is dropped.
  * a deliberately lopsided one (one rank holds almost everything), in both directions, because
    every offset in the plan is a prefix sum and the window has to be sized for the LARGEST
    rank's output rather than this rank's.
Heads are split evenly in both: sglang's usp.py asserts ``h_global % world_size == 0``, so
nothing in production can produce a head-uneven call, and this test does not invent one.

Run on a multi-GPU host (ws in [2, 8]):
    torchrun --nproc_per_node=8 test/distributed/a2a_uneven.py
(or via pytest: test/test_multigpu.py)
"""

from __future__ import annotations

import os
from itertools import accumulate

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def offsets(splits: list[int]) -> list[int]:
    return [0] + list(accumulate(splits))


def torch_a2a_uneven(x, mode, seq_splits, head_splits, rank, pg):
    """Reference Ulysses all-to-all over uneven shards, via dist.all_to_all.

    Rank r sends send[p] to rank p, which receives it as recv[r]; concatenating recv along the
    axis that was gathered is the whole operation.
    """
    ws = len(seq_splits)
    b, d = x.shape[0], x.shape[-1]
    if mode == 0:
        head_off = offsets(head_splits)
        send = [x[:, :, head_off[p] : head_off[p + 1], :].contiguous() for p in range(ws)]
        recv = [
            torch.empty(b, seq_splits[r], head_splits[rank], d, dtype=x.dtype, device=x.device)
            for r in range(ws)
        ]
        dist.all_to_all(recv, send, group=pg)
        return torch.cat(recv, dim=1)
    seq_off = offsets(seq_splits)
    send = [x[:, seq_off[p] : seq_off[p + 1], :, :].contiguous() for p in range(ws)]
    recv = [
        torch.empty(b, seq_splits[rank], head_splits[r], d, dtype=x.dtype, device=x.device)
        for r in range(ws)
    ]
    dist.all_to_all(recv, send, group=pg)
    return torch.cat(recv, dim=2)


def make_input(b, d, seq_splits, head_splits, mode, rank, dev, dtype):
    if mode == 0:
        shape = (b, seq_splits[rank], sum(head_splits), d)
    else:
        shape = (b, sum(seq_splits), head_splits[rank], d)
    return torch.randn(shape, dtype=dtype, device=dev)


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD

    group = UlyssesGroup(process_group=pg, initial_pool_bytes=1 << 30)

    b, d, dtype = 2, 128, torch.bfloat16
    head_splits = [2] * ws  # heads are always even; see the module docstring

    # Unpadded sequence: 7*ws + 1 tokens over ws ranks -> shards differing by exactly one.
    s = 7 * ws + 1
    unpadded = [s // ws + (1 if r < s % ws else 0) for r in range(ws)]
    # Lopsided, both directions: the big shard first, then last. A window sized from
    # seq_splits[0] (or from this rank's own share) passes one of these and fails the other.
    front = [8 * ws] + [1] * (ws - 1)
    back = [1] * (ws - 1) + [8 * ws]

    for name, seq_splits in (("unpadded", unpadded), ("front", front), ("back", back)):
        for mode in (0, 1):
            x = make_input(b, d, seq_splits, head_splits, mode, rank, dev, dtype)
            ref = torch_a2a_uneven(x, mode, seq_splits, head_splits, rank, pg)
            ours = group.all_to_all_single_4d(
                x,
                mode=mode,
                tag=f"{name}_m{mode}",
                seq_splits=seq_splits,
                head_splits=head_splits,
            )
            if ours.shape != ref.shape:
                raise AssertionError(
                    f"SHAPE rank={rank} ws={ws} {name} mode={mode}: "
                    f"{tuple(ours.shape)} != {tuple(ref.shape)}"
                )
            if not torch.equal(ours, ref):
                raise AssertionError(f"MISMATCH rank={rank} ws={ws} {name} mode={mode}")
            if rank == 0:
                print(
                    f"OK {name} ws={ws} mode={mode} splits={seq_splits} shape={tuple(ours.shape)}",
                    flush=True,
                )
            dist.barrier()

    # Round trip: gather(scatter(x)) == x, which is how the two modes wrap attention. `mid` is
    # a tensor of its own, so feeding it straight back in is safe on any tag; the borrowed form
    # would need the two distinct tags used here (transfer_on_stream in bindings.cpp rejects a
    # borrowed result fed back under its own tag).
    x = make_input(b, d, unpadded, head_splits, 0, rank, dev, dtype)
    mid = group.all_to_all_single_4d(
        x, mode=0, tag="rt0", seq_splits=unpadded, head_splits=head_splits
    )
    back_out = group.all_to_all_single_4d(
        mid, mode=1, tag="rt1", seq_splits=unpadded, head_splits=head_splits
    )
    if not torch.equal(back_out, x):
        raise AssertionError(f"ROUND TRIP rank={rank} ws={ws}")
    if rank == 0:
        print(f"OK[round-trip] ws={ws}", flush=True)
    dist.barrier()

    # Async form takes the same splits.
    x = make_input(b, d, unpadded, head_splits, 0, rank, dev, dtype)
    ref = torch_a2a_uneven(x, 0, unpadded, head_splits, rank, pg)
    h = group.all_to_all_single_4d_async(
        x, mode=0, tag="uneven_async", seq_splits=unpadded, head_splits=head_splits
    )
    if not torch.equal(h.wait(), ref):
        raise AssertionError(f"ASYNC MISMATCH rank={rank} ws={ws}")
    if rank == 0:
        print(f"OK[async] ws={ws}", flush=True)
    dist.barrier()

    # Rejections. Both are raised while validating arguments, before anything is issued to the
    # stream, so every rank raises and none is left waiting in a barrier.
    x = make_input(b, d, unpadded, head_splits, 0, rank, dev, dtype)
    try:
        group.all_to_all_single_4d(x, mode=0, tag="bad_pair", seq_splits=unpadded)
        raise AssertionError(f"seq_splits without head_splits was accepted (rank={rank})")
    except RuntimeError as e:
        if "pass both" not in str(e):
            raise
    wrong = list(unpadded)
    wrong[rank] += 1  # claims a shard this rank does not hold
    try:
        group.all_to_all_single_4d(
            x, mode=0, tag="bad_shape", seq_splits=wrong, head_splits=head_splits
        )
        raise AssertionError(f"mis-sharded input was accepted (rank={rank})")
    except RuntimeError as e:
        if "splits imply" not in str(e):
            raise
    if rank == 0:
        print(f"OK[rejects] ws={ws} half-splits and mis-sharded input", flush=True)
    dist.barrier()

    if rank == 0:
        print("ALL PASS", flush=True)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
