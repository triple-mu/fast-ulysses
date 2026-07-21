# Benchmarks

## Reference shape: Wan2.2 5s 720p

The benchmark shape is the attention all-to-all of a **Wan2.2 (14B-class) 5s 720p video**:

- 720p (1280×720) × 81 frames; VAE 4×8×8 + patchify (1,2,2) → 21 latent frames × 45×80 patches =
  **sequence N = 75600**.
- **H = 40 heads, head_dim D = 128**, `bf16`, `b = 1`.
- Ulysses parallelism = `world_size`: each rank holds `N/ws` sequence, and after the a2a
  `n_local = 40/ws` heads.

## Methodology

- Throughput = **per-rank remote bytes / time** = `numel * 2 * (ws-1) / ws ÷ elapsed` — the same
  accounting as the ThunderKittens benchmark (see `benchmark/bench_uniform.py`).
- Latency is the median of 20 iterations timed with CUDA events; the warmup loop absorbs the
  first-call lazy autotune so timed iterations all hit the config cache.
- `ours` = this op with `use_tma=None` (runtime path selection); `NCCL` = `torch.distributed`
  permute + `all_to_all_single` reference; speedup = NCCL latency / ours latency.

## Uniform all-to-all — 8×H200 (no NVSwitch fabric, idle machine)

| ws | n_local | ours mode0 (GB/s) | ours mode1 (GB/s) | NCCL (GB/s) | speedup |
| --- | --- | --- | --- | --- | --- |
| 2 | 20 | 355 | 355 | 171 | **2.1×** |
| 4 | 10 | 301 | 306 | 202 | **1.5×** |
| 8 | 5 | 301 | 301 | 203 | **1.5×** |

Observations:

- **Stable 1.5–2.1× over NCCL across all world sizes**; the smaller the `ws` (bigger per-peer head
  blocks, fewer peers), the bigger the lead.
- **Runtime path selection lets mode0 and mode1 each use their faster path**: on H200 at this shape
  both directions auto-select non-TMA (clearly faster than forced TMA) — no offline table or manual
  pick needed.
- Different nodes / fabric topologies differ significantly; the table is a single idle 8×H200 node,
  order-of-magnitude guidance only.

Reproduce:

```bash
PROF_N=75600 PROF_H=40 PROF_D=128 PROF_MODE=0 torchrun --nproc_per_node=8 benchmark/bench_uniform.py
# PROF_MODE=0|1 selects direction; FAST_ULYSSES_USE_TMA (unset=auto, 0=non-TMA, 1=TMA) forces the path
```

## CE path: overlap under compute (`benchmark/bench_ce.py`)

The CE path trades standalone latency for overlap: its per-peer `cudaMemcpy2DAsync` fan-out rides
the DMA engines, so it keeps moving data while compute kernels hold every SM block slot — the
regime where both kernel paths stall.

- **Workload**: Wan 720p/81f shapes (`S=75600, H=40, D=128`, bf16) with a concurrent
  to_q/k/v-shaped 3-GEMM chain (`K=N=5120`).
- **Metric**: `hidden% = (serial − concurrent) / a2a_alone` — how much of the standalone a2a time
  disappears when it runs under the GEMM chain. Serial and concurrent are measured **alternately**
  and compared by median: on shared machines the GEMM window drifts a few percent run to run,
  which would otherwise swamp the sub-millisecond a2a effect.
- **Exclusive 4×H100/4×H200 runs (Wan ws=4)**: standalone CE 0.68–0.70 ms vs 0.48 ms for the SM
  scatter — but **93–98% of the CE a2a hides** under the GEMM chain vs 25–39% for the kernel
  paths, i.e. net exposed time ~0.05 ms/call vs ~0.37 ms. Per-peer CE throughput is ~385 GB/s and
  is unaffected by a full-SM spin kernel.

Reproduce:

```bash
torchrun --nproc_per_node=4 benchmark/bench_ce.py
```

## Profiling

`benchmark/profile.py` is a minimal nsys/ncu driver (NVTX ranges, fixed shape).
