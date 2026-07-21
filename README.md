<div align="center">

# fast-ulysses

**Ulysses sequence-parallel all-to-all as a torch custom op — NVSHMEM symmetric heap + NVLink P2P, no NCCL on the hot path.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

## Why fast-ulysses?

Ulysses sequence parallelism (DeepSpeed-Ulysses) shards very long sequences across GPUs for attention: one all-to-all before attention swaps the "sequence-sharded" layout for "head-sharded" (every rank gets the full sequence for its own subset of heads), and a second all-to-all swaps back afterwards. For long-sequence / video DiT workloads (Wan, HunyuanVideo, ...) this all-to-all is the critical communication — the longer the sequence and the more GPUs, the harder it bottlenecks.

`fast_ulysses` implements this 4D all-to-all as a standalone, distributable **torch custom op** (namespace `fast_ulysses`, `torch.ops.fast_ulysses.all_to_all_single_4d`) that bypasses NCCL inside a node: output buffers are allocated on the NVSHMEM symmetric heap, data is written directly into peer GPU memory over NVLink P2P, and a lightweight custom NVLink flag barrier synchronizes ranks — the whole path never touches the host and never issues an NCCL collective.

## Features

- **Three transfer paths**:
  - **non-TMA path**: SM-resident vectorized direct writes with a per-shape autotuned launch config; the fallback for sm80 (A100) and anything without TMA.
  - **TMA path** (sm90+, Hopper/Blackwell): `cp.async.bulk` (the TMA unit) moves the data with a software pipeline, occupying almost no SM.
  - With `use_tma=None` (auto), **both kernel paths are micro-benchmarked on the actual hardware at first call and the faster one is cached** (replacing any offline static table); it can also be forced per call (see the `use_tma` tri-state in [docs/API.md](docs/API.md)).
  - **CE path** (`all_to_all_single_4d_ce`, chosen explicitly): a per-peer `cudaMemcpy2DAsync` fan-out on the DMA engines — **zero SM usage**, so the transfer keeps running at full NVLink bandwidth while compute kernels (e.g. cuBLAS nvjet GEMMs) hold every SM slot. The overlap path: 93–98% of the CE a2a hides under a concurrent GEMM chain vs 25–39% for the kernel paths (4×H100/H200, Wan shapes).
- **Grouped handshakes for async pipelines**: `barrier=False` defers the completion handshake so several async a2as (e.g. one layer's q/k/v) share a single barrier — see [docs/API.md](docs/API.md).
- **Compute-communication fusion examples** (fused QK RMSNorm + RoPE into the scatter kernel, standalone `rms_norm` / `rope` / `norm_rope` ops) live on the `examples/qk-norm-rope-fusion` branch — this branch keeps the pure all-to-all core.
- **Single-node NVLink P2P**, `world_size ∈ [1, 8]` (odd sizes such as 3/5/6/7 included).
- **Uniform splits**: sequence length `s` and head count `n` divisible by `world_size`.
- **Both directions**: `mode=0` scatters heads / gathers sequence (entering attention); `mode=1` is its inverse (leaving attention).
- `float16` / `bfloat16`; requires `d * elem_size` to be 16-byte aligned.

## Installation

Requirements: **NVSHMEM 3.7.0**, **PyTorch** (CUDA 12 build), **CUDA 12**, and a GPU from sm80 (A100) / sm90 (H100/H200) / sm100 (B200) / sm120.

```bash
NVSHMEM_HOME=<nvshmem install root> \
FAST_ULYSSES_CUDA_ARCH=90 \
pip install -e . --no-build-isolation
```

- `NVSHMEM_HOME` (required): NVSHMEM install root (must contain `include/nvshmem.h` and `lib/cmake/nvshmem`).
- `FAST_ULYSSES_CUDA_ARCH`: target compute capabilities, `;`-separated, default `80;90;100;120`.
- `--no-build-isolation`: build against the PyTorch already installed in the host environment.

Docker setup, nodes without an NVSwitch fabric, and troubleshooting: [docs/INSTALL.md](docs/INSTALL.md).

## Quick Start

Save as `example.py` and run with `torchrun --nproc_per_node=2 example.py`:

```python
import os

import torch
import torch.distributed as dist

from fast_ulysses import UlyssesGroup


def main() -> None:
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    ws = dist.get_world_size()
    lr = int(os.environ.get("LOCAL_RANK", rank))
    torch.cuda.set_device(lr)
    dev = torch.device("cuda", lr)

    group = UlyssesGroup(process_group=dist.group.WORLD, initial_pool_bytes=1 << 30)

    # mode0: input (b, s_local, n_global, d) -> output (b, s_global, n_local, d)
    b, s_local, d = 2, 16, 128
    n_global = 4 * ws  # must be divisible by world_size
    x = torch.randn(b, s_local, n_global, d, dtype=torch.bfloat16, device=dev)

    # First call per shape micro-benchmarks and caches the best launch config;
    # all ranks must issue the same (shape, mode, use_tma) call sequence.
    out = group.all_to_all_single_4d(x, mode=0, tag="demo", use_tma=None)
    assert out.shape == (b, s_local * ws, n_global // ws, d)
    if rank == 0:
        print(f"ws={ws} in={tuple(x.shape)} out={tuple(out.shape)}", flush=True)

    # Concurrently-live results (e.g. q/k/v) must use distinct tags,
    # otherwise they alias the same symmetric-heap buffer.
    q = torch.randn(b, s_local, n_global, d, dtype=torch.bfloat16, device=dev)
    k = torch.randn(b, s_local, n_global, d, dtype=torch.bfloat16, device=dev)
    oq = group.all_to_all_single_4d(q, mode=0, tag="q")
    ok = group.all_to_all_single_4d(k, mode=0, tag="k")

    group.destroy()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

## API at a Glance

| API | Summary |
| --- | --- |
| `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)` | Collective group construction: NVSHMEM init + symmetric-heap pool. |
| `group.all_to_all_single_4d(x, *, mode=0, tag="", use_tma=None)` | Uniform 4D all-to-all (mode0 / mode1), kernel paths. |
| `group.all_to_all_single_4d_async(..., barrier=True) -> AsyncA2AHandle` | Same op on a high-priority comm stream; overlap until `handle.wait()`; `barrier=False` groups several calls under one handshake. |
| `group.all_to_all_single_4d_ce(x, *, mode=0, tag="")` | Same collective on the DMA engines (zero SM) — the overlap path. |
| `group.all_to_all_single_4d_ce_async(..., barrier=True) -> AsyncA2AHandle` | Async CE variant; its in-flight window genuinely overlaps concurrent compute. |
| `group.destroy()` | Release symmetric-heap resources (collective). |

Shapes, the `use_tma` tri-state, tag semantics, and the **collective hard constraints** (call-sequence uniformity across ranks — violating them hangs the whole group) are documented in [docs/API.md](docs/API.md).

## Benchmarks

Wan2.2 (14B-class) 5s 720p attention shape (N=75600, H=40, D=128, bf16), single node 8×H200, `use_tma=None`, vs. `torch.distributed` permute + `all_to_all_single` (NCCL):

| ws | ours mode0 (GB/s) | ours mode1 (GB/s) | NCCL (GB/s) | speedup |
| --- | --- | --- | --- | --- |
| 2 | 355 | 355 | 171 | **2.1×** |
| 4 | 301 | 306 | 202 | **1.5×** |
| 8 | 301 | 301 | 203 | **1.5×** |

Full methodology, shape derivation, the CE-path overlap study, and reproduction commands: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Testing

```bash
pytest                     # torchrun-wrapped multi-GPU suites; auto-skip below 2 GPUs
torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py   # direct worker invocation
```

Development setup (pre-commit, formatting, layout): [docs/DEVELOP.md](docs/DEVELOP.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
