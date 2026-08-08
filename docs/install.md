# Installation

## Requirements

- **PyTorch 2.10+**, Linux x86_64, CPython 3.10–3.13
- **CUDA 12.8+ or 13**
- sm80 / sm90 / sm100 / sm120, NVLink-joined
- For a source build only: CMake ≥ 3.18 and `nvcc`. `ccache` is used when present.

torch is the only runtime dependency. Windows come from `torch.distributed._symmetric_memory`,
whose `get_mem_pool` / `get_mempool_allocator` arrive in torch 2.10 — that is where the floor comes
from. CUDA 12.6 and below cannot emit `sm_100` / `sm_120`.

## Install

```bash
pip install fast-ulysses
```

That wheel is built against the newest stable torch. For any other supported torch, take the
matching wheel from the release page — the torch minor must match exactly and the CUDA major must
match, since the extension embeds `c10d::Work` vtables, a pybind11 module and a `TORCH_LIBRARY`
registration, none of which survive a minor bump.

| your torch | torch's CUDA | wheel tag |
|---|---|---|
| 2.10.x | 12.x | `torch210cu128` |
| 2.10.x | 13.x | `torch210cu130` |
| 2.11.x | 12.x | `torch211cu128` |
| 2.11.x | 13.x | `torch211cu130` |
| 2.12.x | 12.x | `torch212cu129` |
| 2.12.x | 13.x | `torch212cu130` |
| 2.13.x | 12.x | `torch213cu129` |
| 2.13.x | 13.x | PyPI, above |

```bash
python -c "import sys,torch; print(torch.__version__, torch.version.cuda, sys.version_info[:2])"
```

## Build from source

```bash
pip install -e . --no-build-isolation                              # all four architectures
FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation    # one, much faster
```

`--no-build-isolation` is required: CMake locates libtorch through the installed torch, so torch
has to be importable at build time, and it must be the one you intend to run against.

| Variable | Meaning |
| --- | --- |
| `FAST_ULYSSES_CUDA_ARCH` | Target compute capabilities, `;`-separated. Default `80;90;100;120`. |
| `CUDACXX` | CUDA compiler; defaults to `/usr/local/cuda/bin/nvcc`. |
| `FAST_ULYSSES_BUILD_DIR` | CMake build tree. Default `./build`, kept between builds so rebuilds are incremental. |
| `FAST_ULYSSES_CMAKE_ARGS` | Extra flags passed through to CMake. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | Overrides the job count, which is otherwise bounded by available memory rather than core count. |

## When the import fails

`import fast_ulysses` catches the loader error and reports what the extension was built against
next to what is installed. `fast-ulysses doctor` prints the same block plus the devices and the
NVLink matrix. The two common causes:

- **`undefined symbol: _ZN3c10...`** — built for a different torch minor. Install the wheel for
  your torch, from the table above.
- **`libcudart.so.12: cannot open shared object file`** — a CUDA-12 wheel in a CUDA-13 environment
  or the reverse. No `LD_LIBRARY_PATH` fixes this; install the right wheel.

## When the group will not build

- **`cuda:i and cuda:j are not joined by NVLink`** — the constructor's check. Use
  `torch.distributed` on that machine; over PCIe, and especially across a CPU socket, it is faster
  than this transport. `require_nvlink=False` builds the group anyway, for measuring that case.
- **`NVML could not report the link topology`** (from `doctor`) — the check cannot answer, so it
  does not refuse. Nothing is claimed either way.

## Other problems

- **NCCL dies at init with `unhandled cuda error` / "Failed to bind NVLink SHARP (NVLS) Multicast
  memory"** — a Fabric Manager or NVSwitch configuration problem on that machine. Run with
  `NCCL_NVLS_ENABLE=0`. This affects the `torch.distributed` bootstrap, not this operator.
- **`CUDASymmetricMemory.cu: ... init_multicast_for_block` warnings, then "Gracefully skipping
  multicast initialization"** — the same machines, the same cause, printed by torch rather than
  NCCL. Harmless here: the multicast path is NVLS, and this transport writes peer addresses
  directly and never uses it.
- **CMake `CMakeCache.txt directory ... is different than ...`** — the persistent `build/` was
  configured from another path (the repo moved). `rm -rf build` and rebuild.
