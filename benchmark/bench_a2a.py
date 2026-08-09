"""Where the time goes, how much of it hides under compute, and what the padding costs.

Run under tools/exclusive.sh -- a number from a shared GPU is not a number:

    ./tools/exclusive.sh 0,1,2,3 -- torchrun --nproc_per_node=4 benchmark/bench_a2a.py
    ./tools/exclusive.sh 0,1,2,3 -- torchrun --nproc_per_node=4 benchmark/bench_a2a.py --overlap
    ./tools/exclusive.sh 0,1,2,3 -- torchrun --nproc_per_node=4 benchmark/bench_a2a.py --padding

stages (default)
    Both paths broken into their parts on one stream, so they sum to the whole call rather than
    merely correlate. BASE is sglang's usp.py: permute + all_to_all_single + permute. `raw` is
    all_to_all_single alone -- same bytes, no relayout, result in the wrong layout, so it is a
    transport floor and not an alternative. `a2a` vs `transfer` is the like-for-like pair;
    everything else is what each path has to do around it, which is the whole argument, because
    the baseline's relayout is SM work competing with the compute this is meant to hide behind.

overlap
    How much of the async call disappears under a concurrent 3-GEMM chain shaped like to_q/k/v:
    hidden% = (serial - concurrent) / a2a_alone. This is the claim the zero-SM design exists to
    support. Serial and concurrent alternate and are compared by median, because the GEMM window
    drifts run to run by about as much as the a2a itself.

padding
    Rounding a sequence up to a multiple of the group size is what lets the baseline stay on its
    flat path; the padded tokens then ride through attention and the collective on every layer of
    every step. Per-rank seq_splits accepts shards differing by one token instead. The question is
    whether our uneven path costs the same as our even one -- if it does, dropping the pad is free
    here and whatever it saves elsewhere is profit.
"""

from __future__ import annotations

import argparse
import os
import statistics
import time

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup

#                  label      img    heads  3*head_dim  txt
SHAPES = [
    ("wan-720p", 75600, 40, 384, 227),
    ("wan-480p", 32760, 40, 384, 227),
    ("h3-t2va-5s", 37824, 56, 384, 227),
]


class Stopwatch:
    """CUDA events between stages, read once at the end. Recording costs ~1 us on the host and
    nothing on the device, so a subdivided run measures the same total as an undivided one."""

    def __init__(self, n_marks: int) -> None:
        self.events = [torch.cuda.Event(enable_timing=True) for _ in range(n_marks)]
        self.i = 0

    def mark(self) -> None:
        self.events[self.i].record()
        self.i += 1

    def read(self) -> list[float]:
        self.events[-1].synchronize()
        return [self.events[k].elapsed_time(self.events[k + 1]) for k in range(self.i - 1)]


def median_of(fn, iters: int, warmup: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    runs = [fn() for _ in range(iters)]
    torch.cuda.synchronize()
    return [statistics.median(r[k] for r in runs) for k in range(len(runs[0]))]


def median_ms(fn, iters: int, warmup: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    out = []
    for _ in range(iters):
        w = Stopwatch(2)
        w.mark()
        fn()
        w.mark()
        out.append(w.read()[0])
    torch.cuda.synchronize()
    return statistics.median(out)


def submit_us(fn, iters: int, warmup: int) -> float:
    """Host time to submit one call, which is what the batch folding is meant to hold flat."""
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


def baseline_padded(x, pg, ws, recv) -> None:
    """usp.py mode 0: permute, flat all_to_all_single, permute. No instrumentation, so it can be
    timed from the outside like everything else it is compared against."""
    b, s_local, h_global, d = x.shape
    h_local = h_global // ws
    y = x.permute(2, 0, 1, 3).contiguous().flatten()
    dist.all_to_all_single(recv, y, group=pg)
    z = recv.reshape(ws, h_local, b, s_local, d)
    z.permute(2, 0, 3, 1, 4).contiguous().reshape(b, s_local * ws, h_local, d)


def baseline_stages(x, pg, ws, recv) -> list[float]:
    """The same path with CUDA events between its three stages. Only run_stages uses this: its
    read() host-syncs, which would land inside any enclosing timed region."""
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


def run_stages(group, pg, rank, ws, args) -> None:
    if rank == 0:
        print(f"# world_size={ws} b={args.batch} iters={args.iters} mode=0 dtype=bfloat16")
        print("# BASE = usp.py: permute + all_to_all_single + permute")
        print("# OURS = barrier_in + transfer + barrier_out + copy_out; transfer carries self")
        print("# copy_out = the copying call's window -> caller's tensor; out= a window skips it")
        print("# raw  = all_to_all_single alone, no relayout: the transport floor\n")
        head = (
            f"{'shape':<14} {'MB':>5} | {'perm_in':>8} {'a2a':>8} {'perm_out':>8} {'BASE':>8} | "
            f"{'barr_in':>8} {'transfer':>8} {'barr_out':>8} {'copy_out':>8} {'OURS':>8} | "
            f"{'raw':>7} {'CE/raw':>7} {'relayout%':>9} {'copyout%':>8} {'submit us':>9}"
        )
        print(head)
        print("-" * len(head))

    dev = torch.device("cuda", torch.cuda.current_device())
    for label, img, heads, d, txt in SHAPES:
        s_total = img + txt
        s_total -= s_total % ws  # the padded shard the baseline needs
        x = torch.randn((args.batch, s_total // ws, heads, d), dtype=torch.bfloat16, device=dev)
        mb = x.numel() * x.element_size() / 1e6
        flat = x.numel()
        send = torch.empty(flat, dtype=torch.bfloat16, device=dev)
        recv = torch.empty(flat, dtype=torch.bfloat16, device=dev)

        base = median_of(lambda: baseline_stages(x, pg, ws, recv), args.iters, args.warmup)
        ours = median_of(lambda: list(group._timed(x, mode=0)[1].values()), args.iters, args.warmup)
        raw = median_ms(
            lambda: dist.all_to_all_single(recv, send, group=pg), args.iters, args.warmup
        )
        submit = submit_us(lambda: group.all_to_all_4d(x, mode=0), args.iters, args.warmup)

        base_total, ours_total = sum(base), sum(ours)
        if rank == 0:
            print(
                f"{label:<14} {mb:5.0f} | {base[0]:8.3f} {base[1]:8.3f} {base[2]:8.3f} "
                f"{base_total:8.3f} | {ours[0]:8.3f} {ours[1]:8.3f} {ours[2]:8.3f} {ours[3]:8.3f} "
                f"{ours_total:8.3f} | {raw:7.3f} {raw / ours[1]:6.2f}x "
                f"{(base[0] + base[2]) / base_total * 100:8.1f}% "
                f"{ours[3] / ours_total * 100:7.1f}% {submit:9.1f}",
                flush=True,
            )
        dist.barrier()


def run_overlap(group, pg, rank, ws, args) -> None:
    dev = torch.device("cuda", torch.cuda.current_device())
    label, img, heads, d, txt = SHAPES[0]
    s_total = img + txt
    s_total -= s_total % ws
    k = n = 5120
    gen = torch.Generator(device=dev).manual_seed(1 + rank)
    x = torch.randn((1, s_total // ws, heads, d), generator=gen, device=dev, dtype=torch.bfloat16)
    a = torch.randn((s_total // ws, k), generator=gen, device=dev, dtype=torch.bfloat16)
    w = torch.randn((k, n), generator=gen, device=dev, dtype=torch.bfloat16)

    def gemms():
        for _ in range(3):
            _ = a @ w

    t_gemm = median_ms(gemms, args.iters, args.warmup)
    t_a2a = median_ms(lambda: group.all_to_all_4d(x, mode=0), args.iters, args.warmup)

    def serial():
        group.all_to_all_4d(x, mode=0)
        gemms()

    def concurrent():
        h = group.all_to_all_4d_async(x, mode=0)
        gemms()
        h.wait()

    # Alternate the two arrangements and compare medians: the GEMM window drifts run to run by
    # about as much as the a2a itself, and alternating cancels the drift.
    ts, tc = [], []
    for _ in range(8):
        ts.append(median_ms(serial, 8, 1))
        tc.append(median_ms(concurrent, 8, 1))
    ts.sort()
    tc.sort()
    t_serial, t_conc = ts[len(ts) // 2], tc[len(tc) // 2]
    if rank == 0:
        print(f"# world_size={ws} shape={label}")
        print(f"gemm_alone   {t_gemm:.3f} ms")
        print(f"a2a_alone    {t_a2a:.3f} ms")
        print(f"serial       {t_serial:.3f} ms  (spread {ts[0]:.3f}-{ts[-1]:.3f})")
        print(f"concurrent   {t_conc:.3f} ms  (spread {tc[0]:.3f}-{tc[-1]:.3f})")
        print(f"hidden       {(t_serial - t_conc) / t_a2a * 100:.0f}%")


def run_padding(group, pg, rank, ws, args) -> None:
    dev = torch.device("cuda", torch.cuda.current_device())
    if rank == 0:
        print(f"# world_size={ws} b={args.batch} iters={args.iters} mode=0 dtype=bfloat16")
        print("# padded = every rank the same length; unpadded = shards differing by one token\n")
        head = (
            f"{'shape':<14} {'base pad':>9} {'base unpad':>11} {'base cost':>10} | "
            f"{'ours pad':>9} {'ours unpad':>11} {'ours cost':>10}"
        )
        print(head)
        print("-" * len(head))

    for label, img, heads, d, txt in SHAPES:
        s_true = img + txt
        s_pad = s_true + (-s_true % ws)
        even = [s_pad // ws] * ws
        uneven = [s_true // ws + (1 if p < s_true % ws else 0) for p in range(ws)]
        head_splits = [heads // ws] * ws

        xp = torch.randn((args.batch, even[rank], heads, d), dtype=torch.bfloat16, device=dev)
        xu = torch.randn((args.batch, uneven[rank], heads, d), dtype=torch.bfloat16, device=dev)
        recv = torch.empty(xp.numel(), dtype=torch.bfloat16, device=dev)

        base_pad = median_ms(lambda: baseline_padded(xp, pg, ws, recv), args.iters, args.warmup)
        base_unpad = median_ms(
            lambda: _baseline_unpadded(xu, pg, ws, rank, uneven), args.iters, args.warmup
        )
        ours_pad = median_ms(lambda: group.all_to_all_4d(xp, mode=0), args.iters, args.warmup)
        ours_unpad = median_ms(
            lambda: group.all_to_all_4d(xu, mode=0, seq_splits=uneven, head_splits=head_splits),
            args.iters,
            args.warmup,
        )
        if rank == 0:
            print(
                f"{label:<14} {base_pad:9.3f} {base_unpad:11.3f} "
                f"{base_unpad / base_pad:9.2f}x | {ours_pad:9.3f} {ours_unpad:11.3f} "
                f"{ours_unpad / ours_pad:9.2f}x",
                flush=True,
            )
        dist.barrier()


def _baseline_unpadded(x, pg, ws, rank, seq_splits):
    """What usp.py's fast path has to become once the shards differ: the same permutes, but with
    split sizes, a per-peer reshape and a cat. Comparing against dist.all_to_all over a list would
    be comparing two different baselines, not measuring what the pad costs."""
    b, _, h_global, d = x.shape
    h_local = h_global // ws
    y = x.permute(2, 0, 1, 3).contiguous().flatten()
    in_sz = [h_local * b * seq_splits[rank] * d] * ws
    out_sz = [h_local * b * sp * d for sp in seq_splits]
    recv = torch.empty(sum(out_sz), dtype=x.dtype, device=x.device)
    dist.all_to_all_single(recv, y, out_sz, in_sz, group=pg)
    chunks, off = [], 0
    for sp, size in zip(seq_splits, out_sz, strict=True):
        chunks.append(recv[off : off + size].reshape(h_local, b, sp, d))
        off += size
    return torch.cat(chunks, dim=2).permute(1, 2, 0, 3).contiguous()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--overlap", action="store_true", help="GEMM overlap instead of stages")
    parser.add_argument("--padding", action="store_true", help="padding cost instead of stages")
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    pg = dist.group.WORLD
    group = UlyssesGroup(process_group=pg)

    if args.overlap:
        run_overlap(group, pg, rank, ws, args)
    elif args.padding:
        run_padding(group, pg, rank, ws, args)
    else:
        run_stages(group, pg, rank, ws, args)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
