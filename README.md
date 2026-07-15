<div align="center">

# fast-ulysses

**Ulysses sequence-parallel all-to-all as a torch custom op — NVSHMEM symmetric heap + NVLink P2P, no NCCL on the hot path.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

## Why fast-ulysses?

Ulysses sequence parallelism (DeepSpeed-Ulysses) shards very long sequences across GPUs for attention: one all-to-all before attention swaps the "sequence-sharded" layout for "head-sharded" (every rank gets the full sequence for its own subset of heads), and a second all-to-all swaps back afterwards. For long-sequence / video DiT workloads (Wan, HunyuanVideo, ...) this all-to-all is the critical communication — the longer the sequence and the more GPUs, the harder it bottlenecks.

`fast_ulysses` implements this 4D all-to-all as a standalone, distributable **torch custom op** (namespace `fast_ulysses`, `torch.ops.fast_ulysses.all_to_all_single_4d`) that bypasses NCCL inside a node: output buffers are allocated on the NVSHMEM symmetric heap, data is written directly into peer GPU memory over NVLink P2P, and a lightweight custom NVLink flag barrier synchronizes ranks — the whole path never touches the host and never issues an NCCL collective.

## Features

- **Two kernel paths, picked at runtime**:
  - **TMA path** (sm90+, Hopper/Blackwell): `cp.async.bulk` (the TMA copy engine) moves the data with a software pipeline. TMA copies run on the copy engine rather than the SMs, **occupying almost no SM**, leaving compute capacity for communication/computation overlap.
  - **non-TMA path**: SM-resident vectorized direct writes (512 threads keeping remote writes in flight); the fallback for sm80 (A100) and anything without TMA.
  - With `use_tma=None` (auto), **both paths are micro-benchmarked on the actual hardware at first call and the faster one is cached** (replacing any offline static table); it can also be forced per call (see the `use_tma` tri-state in [docs/API.md](docs/API.md)).
- **Fused QK RMSNorm + RoPE**: standalone single-GPU ops (`rms_norm` / `rope` / `norm_rope`) plus an all-to-all variant that fuses the q/k norm+rope into the scatter kernel itself — see [docs/API.md](docs/API.md).
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
| `group.all_to_all_single_4d(x, *, mode=0, tag="", use_tma=None)` | Uniform 4D all-to-all (mode0 / mode1). |
| `group.all_to_all_single_4d_async(...) -> AsyncA2AHandle` | Same op on a high-priority comm stream; overlap until `handle.wait()`. |
| `group.all_to_all_single_4d_qk(x, weight, cos, sin, ...)` | mode0 a2a with source-side fused QK RMSNorm + RoPE. |
| `group.all_to_all_single_4d_qk2(q, k, ...)` | q + k in one collective call (shared barrier). |
| `group.destroy()` | Release symmetric-heap resources (collective). |
| `rms_norm` / `rope` / `norm_rope` | Standalone single-GPU fused elementwise ops. |

Shapes, the `use_tma` tri-state, tag semantics, and the **collective hard constraints** (call-sequence uniformity across ranks — violating them hangs the whole group) are documented in [docs/API.md](docs/API.md).

## Benchmarks

Wan2.2 (14B-class) 5s 720p attention shape (N=75600, H=40, D=128, bf16), single node 8×H200, `use_tma=None`, vs. `torch.distributed` permute + `all_to_all_single` (NCCL):

| ws | ours mode0 (GB/s) | ours mode1 (GB/s) | NCCL (GB/s) | speedup |
| --- | --- | --- | --- | --- |
| 2 | 355 | 355 | 171 | **2.1×** |
| 4 | 301 | 306 | 202 | **1.5×** |
| 8 | 301 | 301 | 203 | **1.5×** |

Fusing QK RMSNorm+RoPE into the a2a gives **1.44–1.61×** over the unfused sequence on the Wan q+k workload. Full methodology, shape derivation, and reproduction commands: [docs/BENCHMARK.md](docs/BENCHMARK.md).

## Testing

```bash
pytest                     # single-GPU op tests; multi-GPU suites auto-skip below 2 GPUs
pytest -m multigpu         # torchrun-wrapped multi-GPU correctness suites
torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py   # direct worker invocation
```

Development setup (pre-commit, formatting, layout): [docs/DEVELOP.md](docs/DEVELOP.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
