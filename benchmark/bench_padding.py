"""Is dropping the sequence padding free, as far as the collective is concerned?

sglang shards a sequence by rounding it up to a multiple of the group size and padding the
last rank's tail (`build_shard_plan`, runtime/distributed/sp_shard_utils.py). Every rank then
holds the same length, and the padded tokens ride through attention and through this
collective on every layer of every step. Dropping the pad means shards differing by at most
one token -- which this operator now accepts.

The pad is at most P-1 tokens, so the two move nearly the same bytes and the byte count is not
the interesting number. What matters is whether our UNEVEN path costs the same as our EVEN
one. If it does, removing the padding is free here and the saving elsewhere in the model is
pure profit. If it does not, the collective claws back part of what attention saves.

The baseline is measured for contrast, on both shards, because it is the one that changes
character: sglang's fast path is one permute + a flat `all_to_all_single` + one permute, and
it cannot stay on that path once the shards differ at all -- it needs split sizes, a per-peer
reshape and a `cat`.

Both of our columns use the DEFAULT copying entry point, so the copy-out is in both and cancels
out of the even-vs-uneven ratio this benchmark is about.

Run under tools/exclusive.sh; the numbers are meaningless on a shared GPU.

    ./tools/exclusive.sh 4,5,6,7 -- torchrun --nproc_per_node=4 benchmark/bench_padding.py
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup

# bench_ce.py's image budgets plus a text tail: the image counts all divide by 8, so without
# the tail there would be nothing to pad. 227 is chosen only for being awkward -- a real
# prompt length is whatever the attention mask sums to.
#                img    heads  3*head_dim  txt
SHAPES = [
    ("wan-720p", 75600, 40, 384, 227),
    ("wan-480p", 32760, 40, 384, 227),
    ("h3-t2va-5s", 37824, 56, 384, 227),
]


def padded_shard(seq_len: int, ws: int) -> tuple[int, int]:
    """build_shard_plan: (per-rank length, pad tokens). Every rank gets the same length."""
    local = (seq_len + ws - 1) // ws
    return local, local * ws - seq_len


def unpadded_splits(seq_len: int, ws: int) -> list[int]:
    """The real sequence split as evenly as it goes. Shards differ by at most one."""
    base, extra = divmod(seq_len, ws)
    return [base + (1 if p < extra else 0) for p in range(ws)]


def median_submit_us(fn, iters: int, warmup: int) -> float:
    """Host wall clock of the CALL, not of the transfer.

    The batch fusion folds b copies per peer into one cudaMemcpy3DAsync, which changes nothing
    on the device and removes (b-1)*P launches from the host. Device timing cannot see that;
    this can. Measured before any synchronise, so it is submission only.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1e6)
    torch.cuda.synchronize()
    return statistics.median(samples)


def median_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, stop = torch.cuda.Event(True), torch.cuda.Event(True)
    samples = []
    for _ in range(iters):
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def baseline_padded(x, pg, ws):
    """usp.py's fast path for mode 0: permute, flat all_to_all_single, permute."""
    b, s_local, h_global, d = x.shape
    h_local = h_global // ws
    y = x.permute(2, 0, 1, 3).contiguous().flatten()
    recv = torch.empty_like(y)
    dist.all_to_all_single(recv, y, group=pg)
    z = recv.reshape(ws, h_local, b, s_local, d)
    return z.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_local * ws, h_local, d)


def baseline_unpadded(x, pg, ws, rank, seq_splits):
    """The varlen path: permute, split-size all_to_all_single, per-peer reshape, cat."""
    b, _, h_global, d = x.shape
    h_local = h_global // ws
    s_me = seq_splits[rank]
    y = x.permute(2, 0, 1, 3).contiguous().flatten()
    in_sz = [h_local * b * s_me * d] * ws
    out_sz = [h_local * b * sp * d for sp in seq_splits]
    recv = torch.empty(sum(out_sz), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(recv, y, out_sz, in_sz, group=pg)
    chunks, off = [], 0
    for sp, size in zip(seq_splits, out_sz):
        chunks.append(recv[off : off + size].reshape(h_local, b, sp, d))
        off += size
    return torch.cat(chunks, dim=2).permute(1, 2, 0, 3).contiguous()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--pool-gb", type=float, default=6.0)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    n_dev = torch.cuda.device_count()
    torch.cuda.set_device(rank % n_dev)
    dev = torch.device("cuda", rank % n_dev)
    dist.init_process_group(backend="nccl", device_id=dev)
    pg = dist.group.WORLD
    ws = dist.get_world_size(pg)
    b, dtype = args.batch, torch.bfloat16

    group = UlyssesGroup(device=dev, initial_pool_bytes=int(args.pool_gb * (1 << 30)))

    if rank == 0:
        print(f"# world_size={ws} b={b} iters={args.iters} dtype={dtype}")
        print("# padded = build_shard_plan: s rounded up to a multiple of P, pad on the last")
        print("#          rank's tail. unpad = the real s, shards differing by <=1.")
        print("# ours-unpad / ours-pad is the number this benchmark exists for: if it is 1.00")
        print("#          the collective does not care whether the padding is there.")
        print()
        hdr = (
            f"{'shape':<12} {'pad':>4} {'MB':>7} | {'base-pad':>9} {'base-unpad':>11} "
            f"{'b-unp/pad':>10} | {'ours-pad':>9} {'ours-unpad':>11} {'o-unp/pad':>10} | "
            f"{'ours/base':>10} {'submit us':>9}"
        )
        print(hdr)
        print("-" * len(hdr))

    for label, img, n_global, d, txt in SHAPES:
        if n_global % ws:
            continue  # heads always divide; usp.py asserts it
        s_real = img + txt
        n_me = n_global // ws

        s_pad_me, n_pad = padded_shard(s_real, ws)
        xp = torch.randn((b, s_pad_me, n_global, d), dtype=dtype, device=dev)
        t_bp = median_ms(lambda: baseline_padded(xp, pg, ws), args.iters, args.warmup)
        t_op = median_ms(
            lambda: group.all_to_all_single_4d(xp, mode=0, tag="pad"), args.iters, args.warmup
        )
        mb = xp.numel() * xp.element_size() / 1e6
        xp = None
        torch.cuda.empty_cache()

        seq_splits = unpadded_splits(s_real, ws)
        head_splits = [n_me] * ws
        x = torch.randn((b, seq_splits[rank], n_global, d), dtype=dtype, device=dev)
        t_bu = median_ms(
            lambda: baseline_unpadded(x, pg, ws, rank, seq_splits), args.iters, args.warmup
        )
        t_ou = median_ms(
            lambda: group.all_to_all_single_4d(
                x, mode=0, tag="unpad", seq_splits=seq_splits, head_splits=head_splits
            ),
            args.iters,
            args.warmup,
        )
        sub_us = median_submit_us(
            lambda: group.all_to_all_single_4d(
                x, mode=0, tag="unpad", seq_splits=seq_splits, head_splits=head_splits
            ),
            args.iters,
            args.warmup,
        )
        x = None
        torch.cuda.empty_cache()

        if rank == 0:
            print(
                f"{label:<12} {n_pad:>4} {mb:>7.0f} | {t_bp:>9.3f} {t_bu:>11.3f} "
                f"{t_bu / t_bp:>9.2f}x | {t_op:>9.3f} {t_ou:>11.3f} {t_ou / t_op:>9.2f}x | "
                f"{t_bp / t_op:>9.2f}x {sub_us:>9.1f}",
                flush=True,
            )

    group.destroy()
    dist.barrier()
    if rank == 0:
        print("PADDING_BENCH_DONE", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
