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

## Fused QK ops — 8×H200, bf16, Wan `n_global=40` / `d=128`, cross-head + interleaved

`benchmark/bench_qk_fused.py` (ws=8, ms/iter). `a2a` is the pure-transfer lower bound; `unfused` =
standalone rms_norm + rope + a2a (Wan status quo); `fused` = `all_to_all_single_4d_qk`; `qk2` =
q and k in one call (shared barrier).

| seq_global | a2a | unfused | fused | fused/unfused | qk2 | qk2 vs 2×fused |
| --- | --- | --- | --- | --- | --- | --- |
| 20480 | 0.109 | 0.216 | 0.150 | **1.44×** | 0.281 | 0.94× |
| 46080 | 0.199 | 0.420 | 0.288 | **1.46×** | 0.523 | 0.91× |

- Fusion cuts the norm+rope overhead on top of the pure a2a from ~0.11/0.22 ms (unfused) down to
  ~0.04/0.09 ms.
- For Wan's real q+k pattern: `qk2` at 0.523 ms vs unfused 2×0.420 = 0.840 ms → **1.61×**.

Reproduce:

```bash
torchrun --nproc_per_node=8 benchmark/bench_qk_fused.py
```

## Profiling

`benchmark/profile.py` is a minimal nsys/ncu driver (NVTX ranges, fixed shape).
