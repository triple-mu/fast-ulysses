"""Microbenchmark: QK RMSNorm+RoPE fused into the mode0 a2a vs done as separate ops.

    torchrun --nproc_per_node=8 fast-ulysses/benchmark/bench_qk_fused.py

At Wan-14B shapes (n_global=40, d=128), per world size, compares paths that all end in the same
custom NVLink a2a (so the delta isolates the fusion):
  a2a     : plain all_to_all_single_4d (mode0), no norm/rope   -- lower-bound reference
  unfused : rms_norm(cross_head) + rope(interleaved) + a2a     -- 3 kernels (what Wan does today)
  fused   : all_to_all_single_4d_qk                            -- norm+rope fused into the scatter
  qk2     : all_to_all_single_4d_qk2(q, k)                     -- q+k fused scatters, ONE shared barrier
Reports ms/iter, the fused-vs-unfused speedup, the norm+rope overhead each path adds over a2a, and
qk2 vs 2x fused (the shared-barrier saving).
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

import fast_ulysses
from fast_ulysses import UlyssesGroup


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(iters):
        fn()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters  # ms/iter


def main():
    dist.init_process_group("nccl")
    rank, ws = dist.get_rank(), dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    pg = dist.group.WORLD
    group = UlyssesGroup(process_group=pg, initial_pool_bytes=6 << 30)

    b, n_global, d, dtype, eps = 1, 40, 128, torch.bfloat16, 1e-6
    w = torch.randn(n_global * d, device=dev, dtype=torch.float32)  # cross-head weight [n*d]
    wk = torch.randn(n_global * d, device=dev, dtype=torch.float32)
    if rank == 0:
        print(f"# ws={ws} dtype=bf16 n_global={n_global} d={d}", flush=True)
        print(f"{'seq_global':>10} {'a2a':>9} {'unfused':>9} {'fused':>9} {'fused/unfused':>14} {'nr_unfused':>11} {'nr_fused':>10} {'qk2':>9} {'qk2/2fused':>11}", flush=True)

    for s_global in (20480, 46080):  # ~480p and ~720p Wan latent token counts
        if s_global % ws:
            continue
        s_local = s_global // ws
        x = torch.randn(b, s_local, n_global, d, device=dev, dtype=dtype)
        xk = torch.randn(b, s_local, n_global, d, device=dev, dtype=dtype)
        theta = torch.randn(s_local, d // 2, device=dev, dtype=torch.float32)
        cos, sin = theta.cos().contiguous(), theta.sin().contiguous()

        def run_a2a():
            return group.all_to_all_single_4d(x, mode=0, tag="a")

        def run_unfused():
            xn = fast_ulysses.rms_norm(x, w, mode="cross_head", eps=eps)
            xr = fast_ulysses.rope(xn, cos, sin, interleaved=True)
            return group.all_to_all_single_4d(xr, mode=0, tag="u")

        def run_fused():
            return group.all_to_all_single_4d_qk(x, w, cos, sin, mode="cross_head", interleaved=True, eps=eps, tag="f")

        def run_qk2():
            return group.all_to_all_single_4d_qk2(x, xk, w, wk, cos, sin, mode="cross_head", interleaved=True, eps=eps, tag="f2")

        t_a = bench(run_a2a)
        t_u = bench(run_unfused)
        t_f = bench(run_fused)
        t_2 = bench(run_qk2)
        if rank == 0:
            sp = t_u / t_f if t_f else 0.0
            r2 = t_2 / (2 * t_f) if t_f else 0.0
            print(f"{s_global:>10} {t_a:>9.3f} {t_u:>9.3f} {t_f:>9.3f} {sp:>13.3f}x {t_u - t_a:>11.3f} {t_f - t_a:>10.3f} {t_2:>9.3f} {r2:>10.3f}x", flush=True)
        dist.barrier()

    if rank == 0:
        print("# ms/iter; nr_* = norm+rope overhead over plain a2a; qk2 = q+k in one call (1 barrier)", flush=True)
    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
