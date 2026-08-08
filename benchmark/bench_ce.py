"""a2a + GEMM-overlap bench.

Times the sync collective, then how much of the async a2a hides under a concurrent 3-GEMM
chain (to_q/k/v-shaped): hidden% = (serial - concurrent) / a2a_alone, which the copy engines
should drive toward 100%.

    ./tools/exclusive.sh <gpus> -- torchrun --nproc_per_node=4 benchmark/bench_ce.py

The DEFAULT copying entry point, so the copy-out is part of both the a2a being timed and the
work that has to hide under the GEMMs -- which is the question a caller actually has.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup

S_GLOBAL, H, D = 75600, 40, 128  # Wan2.1 14B 720p/81f
K, N = 5120, 5120  # to_q GEMM: [S_GLOBAL/ws, K] x [K, N]
GEMMS = 3
ITERS = 30


def bench(fn, iters=ITERS, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    dist.barrier()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    t0.record()
    for _ in range(iters):
        fn()
    t1.record()
    torch.cuda.synchronize()
    return t0.elapsed_time(t1) / iters


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)
    group = UlyssesGroup(initial_pool_bytes=2 << 30)

    gen = torch.Generator(device=dev).manual_seed(1 + rank)
    x = torch.randn((1, S_GLOBAL // ws, H, D), generator=gen, device=dev, dtype=torch.bfloat16)
    a = torch.randn((S_GLOBAL // ws, K), generator=gen, device=dev, dtype=torch.bfloat16)
    w = torch.randn((K, N), generator=gen, device=dev, dtype=torch.bfloat16)

    def gemms():
        for _ in range(GEMMS):
            _ = a @ w

    t_gemm = bench(gemms)
    if rank == 0:
        print(f"gemm_only({GEMMS}x): {t_gemm:.3f} ms", flush=True)

    def sync_fn(tag):
        return group.all_to_all_single_4d(x, mode=0, tag=tag)

    def async_fn(tag):
        return group.all_to_all_single_4d_async(x, mode=0, tag=tag)

    tag = "b_ce"
    t_a2a = bench(lambda: sync_fn(tag))

    def serial():
        sync_fn(tag)
        gemms()

    def concurrent():
        h = async_fn(tag)
        gemms()
        h.wait()

    # The GEMM window drifts a few percent run to run (shared machine, clocks), which
    # is the same magnitude as the a2a itself -- so serial and concurrent are measured
    # ALTERNATELY and compared by median, cancelling slow drift.
    ts, tc = [], []
    for _ in range(8):
        ts.append(bench(serial, iters=8, warmup=1))
        tc.append(bench(concurrent, iters=8, warmup=1))
    ts.sort()
    tc.sort()
    t_serial, t_conc = ts[len(ts) // 2], tc[len(tc) // 2]
    hidden = (t_serial - t_conc) / t_a2a * 100 if t_a2a > 0 else 0.0
    if rank == 0:
        print(
            f"a2a={t_a2a:.3f}ms serial={t_serial:.3f}ms "
            f"(spread {ts[0]:.3f}-{ts[-1]:.3f}) concurrent={t_conc:.3f}ms "
            f"(spread {tc[0]:.3f}-{tc[-1]:.3f}) -> hidden {hidden:.0f}%",
            flush=True,
        )

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
