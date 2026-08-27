"""Where the time goes, how much of it hides under compute, and what each path costs.

Run under tools/exclusive.sh -- a number from a shared GPU is not a number:

    ./tools/exclusive.sh 0,1,2,3 -- torchrun --nproc_per_node=4 benchmark/bench_a2a.py --mode stages

Modes: stages (default), overlap, padding, zerocopy, sweep, link, pcie-pretest, h3-block,
zerosm.
benchmark/collect.sh runs the established modes in order and records the environment they ran in.

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

pcie-pretest
    Feasibility gate for a future PCIe backend. Mode 0 is locally packed by destination, then each
    destination receives one flat copy directly into its final sequence slice. On an NVLink B300
    this validates the layout, scheduling, and pack cost but not PCIe peer bandwidth; pinned D2H
    and H2D rows separately exercise the machine's real host PCIe path. On a PCIe-only machine,
    pass --allow-non-nvlink and the flat-peer row exercises real GPU P2P. This is a benchmark-only
    prototype: its peer-copy timing deliberately has no cross-rank GPU barrier, and the report adds
    the current operator's measured barrier cost as an estimate.

h3-block
    A production-path A/B/C for MiniMax H3. Unlike the historical shape rows, this sends Q, K,
    and V as three independent head_dim=128 mode-0 calls, then one mode-1 output call, exactly as
    vLLM-Omni's Ulysses wrapper does today. It reports slowest-rank p50/p95 for NCCL, pitched, and
    packed backends, checks exact round trips, and projects the communication-only denoise total.

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
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup

#                  label      img    heads  3*head_dim  txt
SHAPES = [
    ("wan-720p", 75600, 40, 384, 227),
    ("wan-480p", 32760, 40, 384, 227),
    ("h3-t2va-5s", 37824, 56, 384, 227),
]

# --mode zerosm. The GEMM is square fp16 at this size on every machine, so the two arms of the A/B
# are compared against the same shape everywhere. LONG_MS is the floor the stretched payload is
# calibrated to, for the shorter of the two arms: at 20 ms a launch gap of tens of microseconds is
# a tenth of a percent. It is a chosen floor, not a measured noise level.
GEMM_N = 8192
LONG_MS = 20.0
ROUNDS, INNER = 5, 5


class Stopwatch:
    """CUDA events between stages, read once at the end. A mark costs host time and no device
    time, so a subdivided run measures the same total as an undivided one."""

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
            f"{'raw':>7} {'raw/CE':>7} {'relayout%':>9} {'copyout%':>8} {'submit us':>9}"
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


def run_pcie_pretest(group, pg, rank, ws, args) -> None:
    """Test the packed-flat design before it becomes a production transport.

    Batch one makes each destination's sequence slice contiguous, so mode 0 needs a local pack
    but no receive-side unpack. The peer-copy region intentionally measures outgoing work only;
    repeated writes are stream ordered and every source owns disjoint destination slices. A real
    backend still needs a completion barrier, whose measured current cost is reported separately.
    """
    import torch.distributed._symmetric_memory as symm_mem

    if args.batch != 1:
        raise ValueError("pcie-pretest currently requires --batch 1")
    if ws < 2:
        raise ValueError("pcie-pretest requires at least two ranks")

    dev = torch.device("cuda", torch.cuda.current_device())
    label, img, heads, d, txt = next(shape for shape in SHAPES if shape[0] == args.shape)
    if heads % ws:
        raise ValueError(f"{heads} heads do not divide world_size {ws}")
    s_total = img + txt
    s_total -= s_total % ws
    s_local, h_local = s_total // ws, heads // ws

    # Small exact integers make a full layout check cheap to construct and bit-exact in bf16.
    seq_index = torch.arange(s_local, device=dev, dtype=torch.int64)[:, None]
    head_index = torch.arange(heads, device=dev, dtype=torch.int64)[None, :]
    pattern = ((rank * 97 + seq_index * 7 + head_index) % 251).to(torch.bfloat16)
    x = torch.empty((1, s_local, heads, d), dtype=torch.bfloat16, device=dev)
    x.copy_(pattern[None, :, :, None].expand_as(x))

    packed = torch.empty((ws, 1, s_local, h_local, d), dtype=x.dtype, device=dev)
    recv = torch.empty_like(x).flatten()

    out_numel = s_total * h_local * d
    out_storage = symm_mem.empty(out_numel, dtype=x.dtype, device=dev)
    out_storage.zero_()
    handle = symm_mem.rendezvous(out_storage, pg)
    out_shape = (1, s_total, h_local, d)
    out = out_storage.view(out_shape)
    peer_outputs = [
        handle.get_buffer(peer, (out_numel,), x.dtype).view(out_shape) for peer in range(ws)
    ]

    current = torch.cuda.current_stream()
    transfer = torch.cuda.Stream()
    ready = torch.cuda.Event()
    done = torch.cuda.Event()

    def pack_input() -> None:
        source = x.view(1, s_local, ws, h_local, d).permute(2, 0, 1, 3, 4)
        packed.copy_(source)

    # Match the production path's XOR order when possible: it spreads the first destination of
    # every rank instead of making all ranks target the same peer first.
    if ws & (ws - 1) == 0:
        remote_order = [rank ^ step for step in range(1, ws)]
    else:
        remote_order = [(rank + step) % ws for step in range(1, ws)]

    def flat_peer_copy() -> None:
        ready.record(current)
        transfer.wait_event(ready)
        with torch.cuda.stream(transfer):
            for peer in remote_order:
                dst = peer_outputs[peer].narrow(1, rank * s_local, s_local)
                dst.copy_(packed[peer], non_blocking=True)
            done.record(transfer)
        own = peer_outputs[rank].narrow(1, rank * s_local, s_local)
        own.copy_(packed[rank], non_blocking=True)
        current.wait_event(done)

    def pack_and_copy() -> None:
        pack_input()
        flat_peer_copy()

    pack_input()
    torch.cuda.synchronize()
    raw_send = packed.flatten()

    def raw_a2a() -> None:
        dist.all_to_all_single(recv, raw_send, group=pg)

    base_ms = median_ms(lambda: baseline_padded(x, pg, ws, recv), args.iters, args.warmup)
    raw_ms = median_ms(raw_a2a, args.iters, args.warmup)
    pack_ms = median_ms(pack_input, args.iters, args.warmup)
    peer_ms = median_ms(flat_peer_copy, args.iters, args.warmup)
    packed_ms = median_ms(pack_and_copy, args.iters, args.warmup)

    # Reuse the implemented synchronization protocol to avoid inventing an optimistic constant.
    current_stages = median_of(
        lambda: list(group._timed(x, mode=0)[1].values()), args.iters, args.warmup
    )
    barrier_ms = current_stages[0] + current_stages[2]
    copy_out_ms = current_stages[3]

    def slowest(value: float) -> float:
        result = torch.tensor(value, dtype=torch.float64, device=dev)
        dist.all_reduce(result, op=dist.ReduceOp.MAX, group=pg)
        return float(result.item())

    base_ms, raw_ms, pack_ms, peer_ms, packed_ms, barrier_ms, copy_out_ms = (
        slowest(value)
        for value in (
            base_ms,
            raw_ms,
            pack_ms,
            peer_ms,
            packed_ms,
            barrier_ms,
            copy_out_ms,
        )
    )

    # All outgoing streams are complete before this host barrier, so every local output is now
    # safe to inspect. Check every element without allocating a second full-size output tensor.
    torch.cuda.synchronize()
    dist.barrier(group=pg)
    for source_rank in range(ws):
        source_seq = torch.arange(s_local, device=dev, dtype=torch.int64)[:, None]
        source_heads = torch.arange(
            rank * h_local, (rank + 1) * h_local, device=dev, dtype=torch.int64
        )[None, :]
        expected = ((source_rank * 97 + source_seq * 7 + source_heads) % 251).to(x.dtype)
        actual = out[:, source_rank * s_local : (source_rank + 1) * s_local]
        if not torch.equal(actual, expected[None, :, :, None].expand_as(actual)):
            raise AssertionError(
                f"packed mode-0 layout mismatch on rank {rank}, source {source_rank}"
            )

    # The integrated backend is measured separately from the prototype above. Keeping both in the
    # report shows exactly what production validation/copy-out costs beyond the flat-copy kernel.
    production = UlyssesGroup(process_group=pg, require_nvlink=False, backend="packed")
    production_out = production.empty_output(x, mode=0)
    production_owned_ms = median_ms(
        lambda: production.all_to_all_4d(x, mode=0), args.iters, args.warmup
    )
    production_zero_ms = median_ms(
        lambda: production.all_to_all_4d(x, mode=0, out=production_out),
        args.iters,
        args.warmup,
    )
    mode1_input = production.all_to_all_4d(x, mode=0)
    production_mode1_ms = median_ms(
        lambda: production.all_to_all_4d(mode1_input, mode=1), args.iters, args.warmup
    )
    round_trip = production.all_to_all_4d(mode1_input, mode=1)
    if not torch.equal(round_trip, x):
        raise AssertionError(f"production packed round trip mismatch on rank {rank}")
    production_owned_ms = slowest(production_owned_ms)
    production_zero_ms = slowest(production_zero_ms)
    production_mode1_ms = slowest(production_mode1_ms)
    production.destroy()

    # These are local host-path probes, not a multi-rank host-staged collective. They remain useful
    # on B300 because D2H/H2D uses PCIe even when GPU peer traffic is routed over NVLink/NVSwitch.
    host_bytes = min(x.numel() * x.element_size(), args.host_mib << 20)
    gpu_src = torch.empty(host_bytes, dtype=torch.uint8, device=dev)
    gpu_dst = torch.empty_like(gpu_src)
    host = torch.empty(host_bytes, dtype=torch.uint8, pin_memory=True)

    def d2h() -> None:
        host.copy_(gpu_src, non_blocking=True)

    def h2d() -> None:
        gpu_dst.copy_(host, non_blocking=True)

    def host_roundtrip() -> None:
        host.copy_(gpu_src, non_blocking=True)
        gpu_dst.copy_(host, non_blocking=True)

    d2h_ms = slowest(median_ms(d2h, args.iters, args.warmup))
    h2d_ms = slowest(median_ms(h2d, args.iters, args.warmup))
    host_serial_ms = slowest(median_ms(host_roundtrip, args.iters, args.warmup))

    tensor_bytes = x.numel() * x.element_size()
    crossed_bytes = tensor_bytes * (ws - 1) / ws
    estimated_zero_copy_ms = packed_ms + barrier_ms
    estimated_owned_ms = estimated_zero_copy_ms + copy_out_ms

    # Exercise an actual GPU -> pinned host -> different GPU route. Each rank owns the peer-side
    # scratch allocation in its process; this measures the data path and all-rank contention, not
    # a usable collective buffer. Two host slots pipeline D2H(i+1) with H2D(i).
    staged_bytes = int(crossed_bytes)
    chunk_bytes = min(staged_bytes, args.host_mib << 20)
    staged_src = torch.empty(staged_bytes, dtype=torch.uint8, device=dev)
    staged_src.fill_(rank % 251)
    peer_device = torch.device("cuda", (torch.cuda.current_device() + 1) % ws)
    with torch.cuda.device(peer_device):
        staged_dst = torch.empty(staged_bytes, dtype=torch.uint8, device=peer_device)
        peer_stream = torch.cuda.Stream(device=peer_device)
    host_slots = [
        torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True),
        torch.empty(chunk_bytes, dtype=torch.uint8, pin_memory=True),
    ]
    d2h_done = [torch.cuda.Event() for _ in range(2)]
    with torch.cuda.device(peer_device):
        h2d_done = [torch.cuda.Event() for _ in range(2)]

    def host_staged_peer() -> None:
        for chunk_index, offset in enumerate(range(0, staged_bytes, chunk_bytes)):
            slot = chunk_index % 2
            size = min(chunk_bytes, staged_bytes - offset)
            if chunk_index >= 2:
                current.wait_event(h2d_done[slot])
            host_chunk = host_slots[slot].narrow(0, 0, size)
            host_chunk.copy_(staged_src.narrow(0, offset, size), non_blocking=True)
            d2h_done[slot].record(current)
            with torch.cuda.stream(peer_stream):
                peer_stream.wait_event(d2h_done[slot])
                staged_dst.narrow(0, offset, size).copy_(host_chunk, non_blocking=True)
                h2d_done[slot].record(peer_stream)
        for event in h2d_done:
            current.wait_event(event)

    staged_peer_ms = slowest(median_ms(host_staged_peer, args.iters, args.warmup))
    torch.cuda.synchronize()
    with torch.cuda.device(peer_device):
        if not bool(torch.all(staged_dst == rank % 251)):
            raise AssertionError(f"host-staged peer copy mismatch on source rank {rank}")

    def gbps(nbytes: float, elapsed_ms: float) -> float:
        return nbytes / (elapsed_ms * 1e6)

    if rank == 0:
        print(f"# {label} mode=0 world_size={ws} b=1 bf16; slowest-rank medians")
        print("# packed-flat is benchmark-only and has no production cross-rank GPU barrier")
        print("# estimates add the current operator's measured barriers and optional copy-out")
        print(
            "# On NVLink hardware, peer rows validate layout/scheduling only; host rows use PCIe."
        )
        print("# On PCIe-only hardware, rerun with --allow-non-nvlink for the real P2P result.\n")
        print(f"{'path':<30} {'ms':>9} {'effective GB/s':>15} {'vs BASE':>10}")
        print("-" * 68)
        rows = (
            ("BASE permute+NCCL+permute", base_ms, crossed_bytes),
            ("raw NCCL (prepared layout)", raw_ms, crossed_bytes),
            ("local pack", pack_ms, 2 * tensor_bytes),
            ("flat peer copies (no barrier)", peer_ms, crossed_bytes),
            ("pack+flat peer (no barrier)", packed_ms, crossed_bytes),
            ("estimated packed (zero-copy)", estimated_zero_copy_ms, crossed_bytes),
            ("estimated packed (owned out)", estimated_owned_ms, crossed_bytes),
            ("PRODUCTION packed mode0 zero", production_zero_ms, crossed_bytes),
            ("PRODUCTION packed mode0 owned", production_owned_ms, crossed_bytes),
            ("PRODUCTION packed mode1 owned", production_mode1_ms, crossed_bytes),
        )
        for name, elapsed, moved in rows:
            print(
                f"{name:<30} {elapsed:9.3f} {gbps(moved, elapsed):15.1f} {base_ms / elapsed:9.2f}x"
            )
        print(
            f"\n# current overhead estimates: barriers={barrier_ms:.3f} ms, "
            f"owned-output copy={copy_out_ms:.3f} ms"
        )
        print(f"# local pinned-host probe: {host_bytes / (1 << 20):.0f} MiB")
        print(f"{'D2H':<30} {d2h_ms:9.3f} {gbps(host_bytes, d2h_ms):15.1f}")
        print(f"{'H2D':<30} {h2d_ms:9.3f} {gbps(host_bytes, h2d_ms):15.1f}")
        print(
            f"{'D2H then H2D (serial)':<30} {host_serial_ms:9.3f} "
            f"{gbps(2 * host_bytes, host_serial_ms):15.1f}"
        )
        print(
            f"{'host-staged all pairs':<30} {staged_peer_ms:9.3f} "
            f"{gbps(crossed_bytes, staged_peer_ms):15.1f} {base_ms / staged_peer_ms:9.2f}x BASE"
        )
        print(
            f"# ideal double-buffer host pipeline floor for this chunk: "
            f"{max(d2h_ms, h2d_ms):.3f} ms (before synchronization and routing)"
        )
        host_floor_ms = (
            crossed_bytes / min(gbps(host_bytes, d2h_ms), gbps(host_bytes, h2d_ms)) / 1e6
        )
        host_serial_projected_ms = crossed_bytes / gbps(host_bytes, d2h_ms) / 1e6
        host_serial_projected_ms += crossed_bytes / gbps(host_bytes, h2d_ms) / 1e6
        print(
            f"# projected full-message host floors per rank: ideal duplex={host_floor_ms:.3f} ms, "
            f"serial={host_serial_projected_ms:.3f} ms; excludes contention/sync/NUMA"
        )


def _baseline_mode0(x: torch.Tensor, pg, ws: int) -> torch.Tensor:
    """vLLM-Omni strict Ulysses: scatter heads, gather sequence."""
    b, s_local, h_global, d = x.shape
    h_local = h_global // ws
    send = x.reshape(b, s_local, ws, h_local, d).transpose(0, 2).contiguous()
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=pg)
    return recv.reshape(s_local * ws, b, h_local, d).transpose(0, 1).contiguous()


def _baseline_mode1(x: torch.Tensor, pg, ws: int) -> torch.Tensor:
    """vLLM-Omni strict Ulysses reverse: scatter sequence, gather heads."""
    b, s_global, h_local, d = x.shape
    s_local = s_global // ws
    send = (
        x.reshape(b, ws, s_local, h_local, d)
        .transpose(0, 3)
        .transpose(0, 1)
        .contiguous()
        .reshape(ws, h_local, s_local, b, d)
    )
    recv = torch.empty_like(send)
    dist.all_to_all_single(recv, send, group=pg)
    return recv.reshape(h_local * ws, s_local, b, d).transpose(0, 2).contiguous()


def _slowest_rank_distribution(fn, iters: int, warmup: int, timing_pg) -> dict[str, float]:
    """CUDA-event latency distribution, reduced to the slowest rank per iteration."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group=timing_pg)

    samples = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))

    values = torch.tensor(samples, dtype=torch.float64, device=torch.cuda.current_device())
    dist.all_reduce(values, op=dist.ReduceOp.MAX, group=timing_pg)
    ordered = sorted(float(value) for value in values.cpu().tolist())

    def percentile(fraction: float) -> float:
        index = min(len(ordered) - 1, max(0, int(len(ordered) * fraction + 0.999999) - 1))
        return ordered[index]

    return {
        "min_ms": ordered[0],
        "p50_ms": statistics.median(ordered),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
        "mean_ms": statistics.mean(ordered),
    }


def run_h3_block(group, pg, rank, ws, args) -> None:
    """Measure the four collectives issued by one MiniMax H3 transformer block."""
    if args.batch != 1:
        raise ValueError("h3-block currently requires --batch 1")
    if ws < 2:
        raise ValueError("h3-block requires at least two ranks")

    label, img, model_heads, _, txt = next(shape for shape in SHAPES if shape[0] == args.shape)
    if label != "h3-t2va-5s":
        raise ValueError("h3-block currently requires --shape h3-t2va-5s")
    if model_heads % args.tensor_parallel_size:
        raise ValueError(
            f"{model_heads} model heads do not divide tensor parallel size "
            f"{args.tensor_parallel_size}"
        )
    heads = model_heads // args.tensor_parallel_size
    if heads % ws:
        raise ValueError(f"{heads} TP-local heads do not divide Ulysses world_size {ws}")

    sequence_length = args.sequence_length or img + txt
    sequence_length += (-sequence_length) % ws
    s_local = sequence_length // ws
    head_dim = 128
    h_local = heads // ws
    dev = torch.device("cuda", torch.cuda.current_device())

    qkv = [
        torch.randn((1, s_local, heads, head_dim), dtype=torch.bfloat16, device=dev)
        for _ in range(3)
    ]
    attn_output = torch.randn(
        (1, sequence_length, h_local, head_dim), dtype=torch.bfloat16, device=dev
    )

    packed = UlyssesGroup(process_group=pg, require_nvlink=False, backend="packed")
    try:
        baseline_round_trip = _baseline_mode1(_baseline_mode0(qkv[0], pg, ws), pg, ws)
        pitched_round_trip = group.all_to_all_4d(group.all_to_all_4d(qkv[0], mode=0), mode=1)
        packed_round_trip = packed.all_to_all_4d(packed.all_to_all_4d(qkv[0], mode=0), mode=1)
        for name, actual in (
            ("nccl", baseline_round_trip),
            ("pitched", pitched_round_trip),
            ("packed", packed_round_trip),
        ):
            if not torch.equal(actual, qkv[0]):
                raise AssertionError(f"{name} H3 round trip mismatch on rank {rank}")

        pitched_qkv_out = [group.empty_output(x, mode=0) for x in qkv]
        pitched_o_out = group.empty_output(attn_output, mode=1)
        packed_qkv_out = [packed.empty_output(x, mode=0) for x in qkv]
        packed_o_out = packed.empty_output(attn_output, mode=1)

        def nccl_block() -> None:
            for x in qkv:
                _baseline_mode0(x, pg, ws)
            _baseline_mode1(attn_output, pg, ws)

        def pitched_owned_block() -> None:
            for x in qkv:
                group.all_to_all_4d(x, mode=0)
            group.all_to_all_4d(attn_output, mode=1)

        def pitched_zero_block() -> None:
            for x, out in zip(qkv, pitched_qkv_out):
                group.all_to_all_4d(x, mode=0, out=out)
            group.all_to_all_4d(attn_output, mode=1, out=pitched_o_out)

        def packed_owned_block() -> None:
            for x in qkv:
                packed.all_to_all_4d(x, mode=0)
            packed.all_to_all_4d(attn_output, mode=1)

        def packed_zero_block() -> None:
            for x, out in zip(qkv, packed_qkv_out):
                packed.all_to_all_4d(x, mode=0, out=out)
            packed.all_to_all_4d(attn_output, mode=1, out=packed_o_out)

        measurements = {
            "nccl_mode0": lambda: _baseline_mode0(qkv[0], pg, ws),
            "nccl_mode1": lambda: _baseline_mode1(attn_output, pg, ws),
            "pitched_owned_mode0": lambda: group.all_to_all_4d(qkv[0], mode=0),
            "pitched_owned_mode1": lambda: group.all_to_all_4d(attn_output, mode=1),
            "packed_owned_mode0": lambda: packed.all_to_all_4d(qkv[0], mode=0),
            "packed_owned_mode1": lambda: packed.all_to_all_4d(attn_output, mode=1),
            "nccl_block": nccl_block,
            "pitched_owned_block": pitched_owned_block,
            "pitched_zero_block": pitched_zero_block,
            "packed_owned_block": packed_owned_block,
            "packed_zero_block": packed_zero_block,
        }
        results = {
            name: _slowest_rank_distribution(fn, args.iters, args.warmup, dist.group.WORLD)
            for name, fn in measurements.items()
        }
    finally:
        packed.destroy()

    baseline_block = results["nccl_block"]["p50_ms"]
    predictions = {}
    for name in (
        "nccl_block",
        "pitched_owned_block",
        "pitched_zero_block",
        "packed_owned_block",
        "packed_zero_block",
    ):
        block_ms = results[name]["p50_ms"]
        predictions[name] = {
            "block_p50_ms": block_ms,
            "versus_nccl": baseline_block / block_ms,
            "denoise_communication_s": block_ms * args.blocks * args.steps / 1000,
        }

    report = {
        "shape": {
            "label": label,
            "batch": 1,
            "sequence_length": sequence_length,
            "local_sequence_length": s_local,
            "model_heads": model_heads,
            "tensor_parallel_size": args.tensor_parallel_size,
            "tp_local_heads": heads,
            "local_heads": h_local,
            "head_dim": head_dim,
            "dtype": "bfloat16",
            "ulysses_world_size": ws,
            "process_world_size": dist.get_world_size(),
        },
        "iterations": args.iters,
        "warmup": args.warmup,
        "blocks": args.blocks,
        "steps": args.steps,
        "measurements": results,
        "predictions": predictions,
    }

    if rank == 0:
        print(
            f"# MiniMax H3 separate Q/K/V block; s={sequence_length} model_heads={model_heads} "
            f"TP={args.tensor_parallel_size} Ulysses={ws} local_heads={heads} head_dim={head_dim}"
        )
        print("# one block = 3 x mode0(Q/K/V) + 1 x mode1(O); slowest-rank samples")
        print(f"{'path':<28} {'p50 ms':>10} {'p95 ms':>10} {'p99 ms':>10}")
        print("-" * 62)
        for name, stats in results.items():
            print(
                f"{name:<28} {stats['p50_ms']:10.3f} {stats['p95_ms']:10.3f} "
                f"{stats['p99_ms']:10.3f}"
            )
        print(
            f"\n# projected communication-only denoise: {args.blocks} blocks x {args.steps} steps"
        )
        print(f"{'path':<28} {'seconds':>10} {'vs NCCL':>10}")
        print("-" * 50)
        for name, prediction in predictions.items():
            print(
                f"{name:<28} {prediction['denoise_communication_s']:10.3f} "
                f"{prediction['versus_nccl']:9.2f}x"
            )

        if args.json_out:
            output_path = Path(args.json_out)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(f"\n# JSON: {output_path}")


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
        """One concurrent region: the chain on the caller's stream, the copy on its own, with a
        mark after the chain so its own time inside the region is read separately. Returns the
        region's wall clock and that time.

        The chain is queued first because whichever is queued second starts as late as the first
        one took the host to submit, and the arms are matched in length, so that much of the chain
        then runs uncovered at its alone speed and pulls the slowdown towards 1.00x. A chain is a
        few launches where the copy is hundreds, so queueing the chain first makes that lag the
        smaller of the two."""
        w = Stopwatch(3)
        w.mark()
        cps.wait_stream(cur)
        gemm_go()
        w.mark()
        copy_go()
        cur.wait_stream(cps)
        w.mark()
        parts = w.read()
        return [sum(parts), parts[0]]

    label0, img, heads, d, txt = SHAPES[0]
    s_total = img + txt
    s_total -= s_total % ws
    design = args.batch * (s_total // ws) * heads * d * 2  # bytes one rank hands the collective

    t_one_gemm = median_ms(gemm_chain(1), args.iters, args.warmup)
    if rank == 0:
        print(f"# world_size={ws}, gemm is m=n=k={GEMM_N} fp16. Every column is a median over")
        print(f"#        {ROUNDS}x{INNER} = {ROUNDS * INNER} samples. The arms alternate, and in")
        print("#        an arm the alone measurements alternate with the concurrent region")
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

    for label, nbytes in ((f"{label0}/rank", design), ("64 MiB", 64 << 20)):
        n = nbytes // 2
        src = symm_mem.empty(n, dtype=torch.bfloat16, device=dev)
        src.fill_(1.0)
        handle = symm_mem.rendezvous(src, pg)
        arms = [("same", torch.empty(n, dtype=torch.bfloat16, device=dev))]
        if ws > 1:
            arms.insert(0, ("peer", handle.get_buffer((rank + 1) % ws, (n,), torch.bfloat16)))

        # creps sets the bytes, and both arms move the same bytes. Calibrated on the *fastest* arm
        # so the shortest of them still lasts LONG_MS: the same bytes take several times longer to
        # reach a peer than this rank's own memory, so one copy count cannot make both arms last
        # the same. Both payloads are stretched, the design one included -- a single copy of it is
        # shorter than one GEMM, and a chain cannot go below one, so both arms would overshoot
        # their copy and neither of the two states this mode exists to separate would be readable.
        # Rank 0's answer is broadcast, because every rank has to issue the same work.
        t_one = [median_ms(alone(enqueue(src, dst, 1)), args.iters, args.warmup) for _, dst in arms]
        creps = max(max(1, round(max(LONG_MS, t_one_gemm) / t)) for t in t_one)
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
            # One chain per arm, matched to that arm's own copy, which is the experiment
            # docs/design.md describes: a chain "of the same duration". A chain longer than the
            # copy under it pulls the slowdown towards 1.00x whatever the copy does, and with one
            # shared chain that would hit the same-device arm -- the one that is supposed to
            # move -- hardest, since it is the shorter of the two at equal bytes. The calibration
            # only has to survive rounding to an integer chain length, so it is a short one.
            calib = {a: median_ms(alone(c), INNER, 1) for a, c in copies.items()}
            plan = torch.tensor(
                [max(1, round(calib[a] / t_one_gemm)) for a in copies],
                device=dev,
                dtype=torch.int64,
            )
            dist.broadcast(plan, 0)
            greps = dict(zip(copies, plan.tolist()))
            chains = {a: gemm_chain(greps[a] if mine else 0) for a in copies}
            # Every reported number is sampled inside this loop: the arms alternate, and within an
            # arm the concurrent region alternates with the two alone measurements it is divided
            # by. A baseline taken once, before the loop, would put the whole run's drift straight
            # into the slowdown column. The barrier keeps every rank in the same arm at the same
            # time, which is what makes the all-pairs row an all-pairs measurement.
            samples = {name: [] for name in copies}
            for _ in range(ROUNDS):
                for name in copies:
                    dist.barrier()
                    c = median_ms(alone(copies[name]), INNER, 1)
                    g = median_ms(chains[name], INNER, 1)
                    samples[name].append(
                        [c, g] + median_of(lambda: pair(copies[name], chains[name]), INNER, 1)
                    )
            dist.barrier()
            if rank != 0:
                continue
            for name in copies:
                t_copy, t_gemm, t_pair, t_under = (
                    statistics.median(r[k] for r in samples[name]) for k in range(4)
                )
                longer = max(t_copy, t_gemm)
                full = (t_copy + t_gemm) / longer
                shown = f"{label} x{creps}" if creps > 1 else label
                print(
                    f"{shown:<16} {moved:8.0f} {flows:>5} {name:<5} "
                    f"{t_copy:8.3f} {t_gemm:8.3f} {greps[name]:>5} "
                    f"{t_pair:8.3f} {t_pair / longer:10.2f}x "
                    f"{full:9.2f}x {t_under / t_gemm:13.2f}x",
                    flush=True,
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        default="stages",
        choices=[
            "stages",
            "overlap",
            "padding",
            "zerocopy",
            "sweep",
            "link",
            "pcie-pretest",
            "h3-block",
            "zerosm",
        ],
        help="which measurement to run; see the module docstring",
    )
    parser.add_argument("--iters", type=int, default=25)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument(
        "--shape",
        choices=[shape[0] for shape in SHAPES],
        default=SHAPES[0][0],
        help="shape for --mode pcie-pretest (default: wan-720p)",
    )
    parser.add_argument(
        "--host-mib",
        type=int,
        default=64,
        help="pinned-host probe size for --mode pcie-pretest (default: 64 MiB)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        help="override the total sequence length for --mode h3-block",
    )
    parser.add_argument(
        "--blocks",
        type=int,
        default=50,
        help="transformer blocks used by the h3-block denoise projection (default: 50)",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="TP degree surrounding each Ulysses group in --mode h3-block (default: 1)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="denoise steps used by the h3-block projection (default: 50)",
    )
    parser.add_argument(
        "--json-out",
        help="rank-0 JSON output path for --mode h3-block",
    )
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
    if args.mode == "h3-block" and args.tensor_parallel_size > 1:
        if ws % args.tensor_parallel_size:
            raise ValueError(
                f"world_size {ws} is not divisible by tensor parallel size "
                f"{args.tensor_parallel_size}"
            )
        ulysses_degree = ws // args.tensor_parallel_size
        local_pg = None
        for tp_rank in range(args.tensor_parallel_size):
            ranks = [
                tp_rank + sp_rank * args.tensor_parallel_size for sp_rank in range(ulysses_degree)
            ]
            candidate = dist.new_group(ranks)
            if rank in ranks:
                local_pg = candidate
        if local_pg is None:
            raise RuntimeError(f"rank {rank} was not assigned to an H3 Ulysses group")
        pg = local_pg
    group = UlyssesGroup(process_group=pg, require_nvlink=not args.allow_non_nvlink)

    {
        "stages": run_stages,
        "overlap": run_overlap,
        "padding": run_padding,
        "zerocopy": run_zerocopy,
        "sweep": run_sweep,
        "link": run_link,
        "pcie-pretest": run_pcie_pretest,
        "h3-block": run_h3_block,
        "zerosm": run_zerosm,
    }[args.mode](group, pg, rank, dist.get_world_size(pg), args)

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
