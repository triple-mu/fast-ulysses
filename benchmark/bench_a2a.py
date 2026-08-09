"""Where the time goes, how much of it hides under compute, and what each path costs.

Run under tools/exclusive.sh -- a number from a shared GPU is not a number:

    ./tools/exclusive.sh 0,1,2,3 -- torchrun --nproc_per_node=4 benchmark/bench_a2a.py --mode stages

Modes: stages (default), overlap, padding, zerocopy, sweep, link, zerosm. benchmark/collect.sh runs
all of them in order and records the environment they ran in.

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

zerocopy
    What `out=` from empty_output() is worth. The peers then write the caller's buffer directly and
    there is no copy-out, which matters more than it looks: a same-device copy competes with
    compute for SMs, while the peer copies do not.

sweep
    The four stages against message size. The barriers cost what they cost regardless of payload,
    so their share is what says at which size this operator stops being the right tool. Read the
    barrier column as a floor, not as something to optimise away -- most of it is rank arrival
    skew, which any synchronisation would pay somewhere.

link
    Flat peer copies, one pair and all pairs at once, to establish what the fabric can actually
    do. It is what makes `transfer` interpretable: at 80% of this ceiling there is nothing left to
    schedule, and at 30% there is.

zerosm
    Does a copy take SMs? The same bytes and the same `dst.copy_(src)` twice -- once into a peer's
    symmetric-memory window, once into this rank's own memory -- the same copy count for both, each
    under a GEMM chain matched to its own length, and each also run alone. Read the GEMM's own
    slowdown -- its time with the copy underneath it over its time alone -- together with the pair's
    wall clock: 1.00x at a 1.00x pair ratio is "that copy used no SMs", while 1.00x at full
    competition is a copy the GEMM starved. Neither column decides it alone.
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

# --mode zerosm. The GEMM is square fp16 at this size on every machine, which is the shape the
# measurement quoted in docs/design.md used. LONG_MS is the floor the stretched payload is
# calibrated to, for the shorter of the two arms: at 20 ms a launch gap of tens of microseconds is
# a tenth of a percent. It is a chosen floor, not a measured noise level.
GEMM_N = 8192
LONG_MS = 20.0
ROUNDS, INNER = 5, 5


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
        if heads % ws:
            if rank == 0:
                print(f"{label:<14} skipped: {heads} heads do not divide world_size {ws}")
            continue
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
        if heads % ws:
            if rank == 0:
                print(f"{label:<14} skipped: {heads} heads do not divide world_size {ws}")
            continue
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


def run_zerocopy(group, pg, rank, ws, args) -> None:
    """What `out=` from empty_output() removes, and what that is worth end to end."""
    dev = torch.device("cuda", torch.cuda.current_device())
    if rank == 0:
        print(f"# world_size={ws} b={args.batch} iters={args.iters} mode=0 dtype=bfloat16")
        print("# copying  = the call allocates, transfers into its own window, copies out")
        print("# zerocopy = out= from empty_output(): the peers write that buffer directly\n")
        head = (
            f"{'shape':<14} {'MB':>6} | {'copying':>9} {'zerocopy':>9} {'saved':>8} {'speedup':>8} | "
            f"{'copy_out':>9} {'of stages':>11}"
        )
        print(head)
        print("-" * len(head))

    for label, img, heads, d, txt in SHAPES:
        if heads % ws:
            if rank == 0:
                print(f"{label:<14} skipped: {heads} heads do not divide world_size {ws}")
            continue
        s_total = img + txt
        s_total -= s_total % ws
        x = torch.randn((args.batch, s_total // ws, heads, d), dtype=torch.bfloat16, device=dev)
        mb = x.numel() * x.element_size() / 1e6
        buf = group.empty_output(x, mode=0)

        copying = median_ms(lambda: group.all_to_all_4d(x, mode=0), args.iters, args.warmup)
        zero = median_ms(lambda: group.all_to_all_4d(x, mode=0, out=buf), args.iters, args.warmup)
        # The stage the zero-copy path skips, measured on the copying path for attribution.
        stages = median_of(
            lambda: list(group._timed(x, mode=0)[1].values()), args.iters, args.warmup
        )
        if rank == 0:
            print(
                f"{label:<14} {mb:6.0f} | {copying:9.3f} {zero:9.3f} {copying - zero:8.3f} "
                f"{copying / zero:7.2f}x | {stages[3]:9.3f} {stages[3] / sum(stages) * 100:10.1f}%",
                flush=True,
            )
        dist.barrier()


def run_sweep(group, pg, rank, ws, args) -> None:
    """Every stage against message size, so the barrier share is a curve rather than one point."""
    dev = torch.device("cuda", torch.cuda.current_device())
    heads = ((40 + ws - 1) // ws) * ws  # mode 0 scatters the head axis, so it must divide
    d = 384
    if rank == 0:
        print(f"# world_size={ws} heads={heads} d={d} bf16, medians over {args.iters} iters")
        head = (
            f"{'s_local':>8} {'MB/rank':>8} | {'barr_in':>8} {'transfer':>9} {'barr_out':>9} "
            f"{'copy_out':>9} {'total':>9} | {'barriers':>9} {'GB/s':>7}"
        )
        print(head)
        print("-" * len(head))

    biggest = (75600 + 227) // ws
    for s_local in (16, 64, 256, 1024, 4096, 16384, biggest):
        x = torch.randn((1, s_local, heads, d), dtype=torch.bfloat16, device=dev)
        mb = x.numel() * x.element_size() / 1e6
        stages = median_of(
            lambda: list(group._timed(x, mode=0)[1].values()), args.iters, args.warmup
        )
        total = sum(stages)
        if rank == 0:
            bar = stages[0] + stages[2]
            # Bytes that actually cross a link: everything except this rank's own share.
            crossed = mb * (ws - 1) / ws / 1e3
            print(
                f"{s_local:>8} {mb:>8.1f} | {stages[0] * 1e3:>8.1f} {stages[1] * 1e3:>9.1f} "
                f"{stages[2] * 1e3:>9.1f} {stages[3] * 1e3:>9.1f} {total * 1e3:>9.1f} | "
                f"{bar / total * 100:>8.1f}% {crossed / (stages[1] / 1e3):>7.1f}",
                flush=True,
            )
        dist.barrier()


def run_link(group, pg, rank, ws, args) -> None:
    """What the fabric does on flat peer copies, which is the ceiling `transfer` is measured against.

    Uses torch symmetric memory directly rather than the operator, so the ceiling is established
    independently of the thing being judged against it.
    """
    import torch.distributed._symmetric_memory as symm_mem

    dev = torch.device("cuda", torch.cuda.current_device())
    n = (64 << 20) // 2  # 64 MiB of bfloat16
    src = symm_mem.empty(n, dtype=torch.bfloat16, device=dev)
    src.fill_(1.0)
    handle = symm_mem.rendezvous(src, pg)
    peer = (rank + 1) % ws
    dst = handle.get_buffer(peer, (n,), torch.bfloat16)
    gb = n * 2 / 1e9  # bytes moved per copy, in GB

    def one_pair():
        # Only rank 0 writes, so the fabric carries a single flow.
        if rank == 0:
            dst.copy_(src)

    def all_pairs():
        dst.copy_(src)

    if ws == 1:
        if rank == 0:
            print("# world_size=1: no link to measure")
        return

    t1 = median_ms(one_pair, args.iters, args.warmup)
    dist.barrier()
    tn = median_ms(all_pairs, args.iters, args.warmup)
    dist.barrier()
    if rank == 0:
        print(f"# world_size={ws}, flat 64 MiB peer copy, rank r -> rank (r+1)%ws\n")
        print(f"{'flows':<10} {'ms':>8} {'GB/s per flow':>15} {'GB/s aggregate':>16}")
        print("-" * 52)
        print(f"{'1':<10} {t1:8.3f} {gb / (t1 / 1e3):>15.1f} {gb / (t1 / 1e3):>16.1f}")
        print(f"{ws:<10} {tn:8.3f} {gb / (tn / 1e3):>15.1f} {gb * ws / (tn / 1e3):>16.1f}")
        print("\n# `transfer` in --mode stages, over the bytes that cross a link, against the")
        print("# per-flow number above, is how much of the fabric the collective is using.")


def run_zerosm(group, pg, rank, ws, args) -> None:
    """Whether a copy takes SMs, as an A/B with a control: peer destination against same-device.

    Goes through torch symmetric memory rather than through the operator, for the reason run_link
    does and one more: the operator's path also contains two barrier kernels and a copy-out, which
    are SM work by construction, so a measurement of the whole call cannot attribute anything to
    the transfer. Here the two arms differ in one thing -- where the destination lives. At equal
    bytes they do not last equally long, so each gets its own chain matched to its own copy; a
    chain longer than the copy under it pulls the slowdown towards 1.00x whatever the copy does.
    Read each arm's copy column against its gemm column before reading its slowdown.

    The GEMM chain's own slowdown is a more direct statement than the pair's wall clock -- it is
    the compute's own cost, not bounded by how much compute happens to be available, which is what
    --mode overlap's hidden% is. It is not sufficient on its own: a copy the chain starved never
    ran under it and leaves the slowdown at 1.00x too, which is why the pair ratio and the
    full-competition reference are printed next to it and why the header says to read both. They
    are also what docs/design.md quotes.
    """
    import torch.distributed._symmetric_memory as symm_mem

    dev = torch.device("cuda", torch.cuda.current_device())
    cur = torch.cuda.current_stream()
    cps = torch.cuda.Stream()
    lhs = torch.randn((GEMM_N, GEMM_N), dtype=torch.float16, device=dev)
    rhs = torch.randn((GEMM_N, GEMM_N), dtype=torch.float16, device=dev)

    def gemm_chain(reps):
        def go():
            for _ in range(reps):
                _ = lhs @ rhs

        return go

    def enqueue(src, dst, reps):
        """The copy under test, on its own stream. Nothing joins it back: pair() needs it left in
        flight, and alone() adds the join."""

        def go():
            with torch.cuda.stream(cps):
                for _ in range(reps):
                    dst.copy_(src)

        return go

    def alone(copy_go):
        def go():
            cps.wait_stream(cur)
            copy_go()
            cur.wait_stream(cps)

        return go

    def pair(copy_go, gemm_go):
        """One concurrent region: the copy queued first on its own stream, the chain on the
        caller's, with an inner pair of marks around the chain by itself. Returns the region's wall
        clock and the chain's own time inside it."""
        w = Stopwatch(4)
        w.mark()
        cps.wait_stream(cur)
        copy_go()
        w.mark()
        gemm_go()
        w.mark()
        cur.wait_stream(cps)
        w.mark()
        parts = w.read()
        return [sum(parts), parts[1]]

    label0, img, heads, d, txt = SHAPES[0]
    s_total = img + txt
    s_total -= s_total % ws
    design = args.batch * (s_total // ws) * heads * d * 2  # bytes one rank hands the collective

    t_one_gemm = median_ms(gemm_chain(1), args.iters, args.warmup)
    if rank == 0:
        print(f"# world_size={ws}, gemm is m=n=k={GEMM_N} fp16. The copy and gemm columns are")
        print(f"#        medians over {args.iters} iters; the pair columns are medians over")
        print(f"#        {ROUNDS}x{INNER} = {ROUNDS * INNER} samples with the arms alternating")
        print(
            f"# one gemm is {t_one_gemm:.3f} ms; `chain` is how many of them ran under that arm's"
        )
        print("#        copy, chosen so the chain lasts about as long as the copy does, and `xN`")
        print("#        on the payload is how many copies both arms were stretched to. The bytes")
        print("#        are shared between the arms; the chain is per arm, because the same bytes")
        print("#        do not take the same time to reach a peer and to reach local memory")
        print("# peer = dst is rank (r+1)%ws's symmetric-memory window, so the copy crosses a link")
        print("# same = dst is this rank's own memory. Same bytes, same call, local destination:")
        print("#        the only difference between the two arms is where the destination is")
        print("# gemm slowdown = the chain's own time with the copy underneath it, over its time")
        print("#        alone. 1.00x is 'that copy cost the chain nothing'")
        print("# pair ratio = concurrent wall clock / max(copy alone, gemm alone); full comp is")
        print("#        (copy + gemm) over the same max, i.e. what perfect competition would cost")
        print("# NEITHER COLUMN IS THE RESULT ON ITS OWN. A copy the chain starved also leaves the")
        print("#        slowdown at 1.00x: overlapped is 1.00x with a 1.00x pair ratio, starved is")
        print("#        1.00x with a pair ratio at full comp. Only the first is the zero-SM one,")
        print("#        and it is the pair of columns, read together, that separates them")
        print("# Both are diluted whenever copy < gemm, since the uncovered part of the chain runs")
        print("#        at its alone speed -- read the copy and gemm columns first")
        if ws == 1:
            print("#")
            print("# world_size=1: there is no peer, so THE PEER ARM IS NOT MEASURED and this run")
            print("# does not test the claim. The same-device arm and the gemm baseline are real.")
            print(
                "# Two NVLink-joined GPUs are the minimum for the comparison this mode exists for."
            )
        head = (
            f"\n{'payload':<16} {'MB/iter':>8} {'flows':>5} {'arm':<5} {'copy':>8} {'gemm':>8} "
            f"{'chain':>5} {'pair':>8} {'pair ratio':>11} {'full comp':>10} {'gemm slowdown':>14}"
        )
        print(head)
        print("-" * (len(head) - 1))

    for label, nbytes, stretch in ((f"{label0}/rank", design, False), ("64 MiB", 64 << 20, True)):
        n = nbytes // 2
        src = symm_mem.empty(n, dtype=torch.bfloat16, device=dev)
        src.fill_(1.0)
        handle = symm_mem.rendezvous(src, pg)
        arms = [("same", torch.empty(n, dtype=torch.bfloat16, device=dev))]
        if ws > 1:
            arms.insert(0, ("peer", handle.get_buffer((rank + 1) % ws, (n,), torch.bfloat16)))

        # creps sets the bytes, and both arms move the same bytes. Calibrated on the *fastest* arm
        # so the shortest of them still lasts LONG_MS: the same bytes take several times longer to
        # reach a peer than this rank's own memory (docs/benchmark.md, H200 wan-720p: 689 us of
        # transfer for 255 MB crossed against 143 us of copy_out for 291 MB local), so one copy
        # count cannot make both arms last the same. Rank 0's answer is broadcast, because every
        # rank has to issue the same work.
        t_one = [median_ms(alone(enqueue(src, dst, 1)), args.iters, args.warmup) for _, dst in arms]
        creps = max(max(1, round(max(LONG_MS, t_one_gemm) / t)) for t in t_one) if stretch else 1
        plan = torch.tensor([creps], device=dev, dtype=torch.int64)
        dist.broadcast(plan, 0)
        creps = int(plan[0])
        moved = creps * nbytes / 1e6

        # One flow is rank 0 writing rank 1 with every other rank idle, as in run_link; all pairs
        # is every rank writing its neighbour and running its own chain. Reported separately: they
        # are different questions, and at ws=1 they would be the same row.
        rows = [(1, rank == 0)] + ([(ws, True)] if ws > 1 else [])
        for flows, mine in rows:
            copies = {name: enqueue(src, dst, creps if mine else 0) for name, dst in arms}
            t_copy = {a: median_ms(alone(c), args.iters, args.warmup) for a, c in copies.items()}
            # One chain per arm, matched to that arm's own copy, which is the experiment
            # docs/design.md describes: a chain "of the same duration". A chain longer than the
            # copy under it pulls the slowdown towards 1.00x whatever the copy does, and with one
            # shared chain that would hit the same-device arm -- the one that is supposed to
            # move -- hardest, since it is the shorter of the two at equal bytes.
            plan = torch.tensor(
                [max(1, round(t_copy[a] / t_one_gemm)) for a in copies],
                device=dev,
                dtype=torch.int64,
            )
            dist.broadcast(plan, 0)
            greps = dict(zip(copies, plan.tolist()))
            chains = {a: gemm_chain(greps[a] if mine else 0) for a in copies}
            t_gemm = {a: median_ms(c, args.iters, args.warmup) for a, c in chains.items()}
            # Alternate the arms and compare medians, so that a drift over the run cannot land on
            # one arm. The barrier keeps every rank in the same arm at the same time, which is
            # what makes the all-pairs row an all-pairs measurement.
            samples = {name: [] for name in copies}
            for _ in range(ROUNDS):
                for name in copies:
                    dist.barrier()
                    samples[name].append(
                        median_of(lambda: pair(copies[name], chains[name]), INNER, 1)
                    )
            dist.barrier()
            if rank != 0:
                continue
            for name in copies:
                t_pair = statistics.median(r[0] for r in samples[name])
                t_under = statistics.median(r[1] for r in samples[name])
                longer = max(t_copy[name], t_gemm[name])
                full = (t_copy[name] + t_gemm[name]) / longer
                shown = f"{label} x{creps}" if creps > 1 else label
                print(
                    f"{shown:<16} {moved:8.0f} {flows:>5} {name:<5} "
                    f"{t_copy[name]:8.3f} {t_gemm[name]:8.3f} {greps[name]:>5} "
                    f"{t_pair:8.3f} {t_pair / longer:10.2f}x "
                    f"{full:9.2f}x {t_under / t_gemm[name]:13.2f}x",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="stages",
        choices=["stages", "overlap", "padding", "zerocopy", "sweep", "link", "zerosm"],
        help="which measurement to run; see the module docstring",
    )
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--allow-non-nvlink",
        action="store_true",
        help="measure a group the constructor would refuse. Only for establishing what the "
        "PCIe / cross-socket case actually costs -- see docs/design.md, 'Why NVLink only'",
    )
    args = parser.parse_args()

    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank)))
    pg = dist.group.WORLD
    group = UlyssesGroup(process_group=pg, require_nvlink=not args.allow_non_nvlink)

    {
        "stages": run_stages,
        "overlap": run_overlap,
        "padding": run_padding,
        "zerocopy": run_zerocopy,
        "sweep": run_sweep,
        "link": run_link,
        "zerosm": run_zerosm,
    }[args.mode](group, pg, rank, ws, args)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
