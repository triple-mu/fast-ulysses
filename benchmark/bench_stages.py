"""Where does the time actually go, on both sides?

Totals say which is faster; they do not say what to attack next. This times every stage of
both paths with CUDA events on one stream, so the parts sum to the whole and the bottleneck is
visible rather than inferred.

    sglang's path (usp.py)              ours
      permute_in   relayout in            barrier_in   writers wait for readers
      a2a          the collective         transfer     peer copies + our own share
      permute_out  relayout out           barrier_out  readers wait for writers
                                          copy_out     window -> the caller's tensor

`a2a` and `transfer` are the like-for-like comparison: same bytes, same all-to-all pattern,
neither doing any relayout. Everything else is what each path has to do around it -- and that
is the whole argument, because the baseline's relayout is SM work that competes with the
compute this is supposed to hide behind, while ours is folded into the copies' addressing.

`copy_out` is what the DEFAULT entry point pays to hand back a tensor the caller owns; it is a
flat device-to-device copy of the result, and it is the only stage
``all_to_all_single_4d_borrowed`` does not run. Its share of the call is the number to look at
before reaching for the borrowed form -- custom_nccl_op measured 15-25% for its own copy-out at
model shapes, which is a different operator on a different plan and is NOT a prediction for this
column.

Also reported per call:
  * `raw`      -- all_to_all_single with nothing around it, the transport floor.
  * `submit`   -- host wall clock of our call before any synchronise. The only thing the batch
                  fusion changes: it folds b copies per peer into one, removing (b-1)*P
                  launches that the device timing cannot see. It also carries the default
                  entry point's output allocation, which the borrowed form does not make.

Run under tools/exclusive.sh; the numbers are meaningless on a shared GPU.
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

#                img    heads  3*head_dim  txt
SHAPES = [
    ("wan-720p", 75600, 40, 384, 227),
    ("wan-480p", 32760, 40, 384, 227),
    ("h3-t2va-5s", 37824, 56, 384, 227),
]


class Stopwatch:
    """CUDA events between stages, read once at the end.

    Recording an event costs ~1 us on the host and nothing on the device, so a subdivided run
    measures the same total as an undivided one -- which is what makes the stage numbers add up
    rather than merely correlate.
    """

    def __init__(self, n_marks: int):
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(n_marks)]
        self.i = 0

    def mark(self) -> None:
        self.events[self.i].record()
        self.i += 1

    def read(self) -> list[float]:
        self.events[-1].synchronize()
        return [
            self.events[k].elapsed_time(self.events[k + 1]) for k in range(len(self.events) - 1)
        ]


def baseline_stages(x, pg, ws, recv) -> list[float]:
    """usp.py mode 0, split at its three stages: permute in, collective, permute out."""
    b, s_local, h_global, d = x.shape
    h_local = h_global // ws
    w = Stopwatch(4)

    w.mark()
    y = x.permute(2, 0, 1, 3).contiguous().flatten()
    w.mark()
    dist.all_to_all_single(recv, y, group=pg)
    w.mark()
    z = recv.reshape(ws, h_local, b, s_local, d)
    z.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_local * ws, h_local, d)
    w.mark()
    return w.read()


def raw_only(send, recv, pg) -> float:
    w = Stopwatch(2)
    w.mark()
    dist.all_to_all_single(recv, send, group=pg)
    w.mark()
    return w.read()[0]


def repeat(fn, iters: int, warmup: int) -> list[float]:
    """Median of each element of fn()'s list, over `iters` runs."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    runs = [fn() for _ in range(iters)]
    torch.cuda.synchronize()
    return [statistics.median(r[k] for r in runs) for k in range(len(runs[0]))]


def submit_us(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t0) * 1e6)
    torch.cuda.synchronize()
    return statistics.median(s)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--pool-gb", type=float, default=24.0)
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
        print(f"# world_size={ws} b={b} iters={args.iters} mode=0 dtype={dtype}")
        print("# BASE = usp.py: permute + all_to_all_single + permute")
        print("# OURS = barrier_in + transfer + barrier_out + copy_out; transfer carries self")
        print("# copy_out = the default entry point's window -> caller's tensor; borrowed skips it")
        print("# raw  = all_to_all_single alone, no relayout: the transport floor")
        print()
        hdr = (
            f"{'shape':<12} {'MB':>6} | {'perm_in':>8} {'a2a':>8} {'perm_out':>8} {'BASE':>8} | "
            f"{'barr_in':>8} {'transfer':>8} {'barr_out':>8} {'copy_out':>8} {'OURS':>8} | "
            f"{'raw':>7} {'CE/raw':>7} {'relayout%':>9} {'copyout%':>8} {'submit us':>9}"
        )
        print(hdr)
        print("-" * len(hdr))

    for label, img, n_global, d, txt in SHAPES:
        if n_global % ws:
            continue
        s_real = img + txt
        s_me = (s_real + ws - 1) // ws  # padded shard, so both paths see the same shape
        x = torch.randn((b, s_me, n_global, d), dtype=dtype, device=dev)
        flat = torch.empty(x.numel(), dtype=dtype, device=dev)
        mb = x.numel() * x.element_size() / 1e6

        base = repeat(lambda: baseline_stages(x, pg, ws, flat), args.iters, args.warmup)
        raw = repeat(lambda: [raw_only(x.flatten(), flat, pg)], args.iters, args.warmup)[0]
        ours = repeat(
            lambda: list(group.all_to_all_single_4d_timed(x, mode=0, tag="st")[1].values()),
            args.iters,
            args.warmup,
        )
        sub = submit_us(
            lambda: group.all_to_all_single_4d(x, mode=0, tag="st"), args.iters, args.warmup
        )

        if rank == 0:
            bp, op = sum(base), sum(ours)
            print(
                f"{label:<12} {mb:>6.0f} | {base[0]:>8.3f} {base[1]:>8.3f} {base[2]:>8.3f} "
                f"{bp:>8.3f} | {ours[0]:>8.3f} {ours[1]:>8.3f} {ours[2]:>8.3f} {ours[3]:>8.3f} "
                f"{op:>8.3f} | {raw:>7.3f} {raw / ours[1]:>6.2f}x "
                f"{(base[0] + base[2]) / bp * 100:>8.1f}% {ours[3] / op * 100:>7.1f}% "
                f"{sub:>9.1f}",
                flush=True,
            )

        x = flat = None
        torch.cuda.empty_cache()

    group.destroy()
    dist.barrier()
    if rank == 0:
        print("STAGES_DONE", flush=True)
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
