# Installation

## Requirements

- **NVSHMEM 3.7+**, latest recommended (host library + headers; only the install root is needed)
- **PyTorch** (any recent CUDA build)
- **CUDA 12 or 13** toolkit (`nvcc`; CUDA 13 needs one extra include flag — see Troubleshooting)
- CMake ≥ 3.18 (plus `ccache` optionally — picked up automatically for faster rebuilds)
- Target GPUs: sm80 (A100) / sm90 (H100/H200) / sm100 (B200) / sm120

Tested combinations: NVSHMEM 3.7.0/cu12 + nvcc 12.8 + torch cu130; NVSHMEM 3.7.2/cu13 + nvcc 13.3 + torch cu126/cu132.

## Build

```bash
NVSHMEM_HOME=<nvshmem install root> \
FAST_ULYSSES_CUDA_ARCH=90 \
pip install -e . --no-build-isolation
```

| Variable | Required | Meaning |
| --- | --- | --- |
| `NVSHMEM_HOME` | yes | NVSHMEM install root; must contain `include/nvshmem.h` and `lib/cmake/nvshmem`. |
| `FAST_ULYSSES_CUDA_ARCH` | no | Target compute capabilities, `;`-separated (e.g. `90` for H100/H200, `100` for B200, `80;90;100;120` multi-target). Default `80;90;100;120`. Building only your actual arch is much faster. |
| `CUDACXX` | no | Override the CUDA compiler; defaults to `/usr/local/cuda/bin/nvcc` when present. |

Notes:

- `--no-build-isolation` is required: the build links against the PyTorch already installed in your environment (CMake locates libtorch through it).
- The build keeps a persistent `build/` directory so rebuilds are incremental.

## Docker

Any recent NGC PyTorch image works as a base:

```bash
docker run --rm -it --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD":/workspace/fast-ulysses -w /workspace/fast-ulysses \
  nvcr.io/nvidia/pytorch:25.03-py3 bash

# inside the container (with NVSHMEM unpacked somewhere, e.g. /opt/nvshmem):
NVSHMEM_HOME=/opt/nvshmem FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation
```

## Nodes without an NVSwitch fabric

On nodes without an NVSwitch fabric (or with IB NICs), NVSHMEM's default init may attempt the NVLS multicast mapping or the IB remote transport and **segfault**. This op is single-node NVLink P2P only, so `UlyssesGroup` applies safe defaults at construction (`os.environ.setdefault` — override them **before** constructing the group if needed):

```text
NVSHMEM_DISABLE_NVLS=1
NVSHMEM_REMOTE_TRANSPORT=none
```

NCCL (the `torch.distributed` bootstrap and test/benchmark references) probes NVLS on its own: on a broken Fabric Manager it dies at init with `unhandled cuda error` ("Failed to bind NVLink SHARP (NVLS) Multicast memory ... CUDA error 401"). Run with `NCCL_NVLS_ENABLE=0` there.

## Troubleshooting

- **`NVSHMEM_HOME must point to ...`**: the variable is unset or does not contain `include/nvshmem.h`. Point it at the unpacked NVSHMEM archive root.
- **CMake error `CMakeCache.txt directory ... is different than ...`**: the persistent `build/` directory was configured from another path (e.g. the repo was moved or renamed). `rm -rf build` and rebuild.
- **Import error for `fast_ulysses._C` after a torch upgrade**: the extension links libtorch; rebuild after switching PyTorch versions (`rm -rf build` first if CMake gets confused).
- **Init segfault inside NVSHMEM**: see the fabric section above; also make sure all ranks construct `UlyssesGroup` together (construction is collective).
- **`fatal error: cuda/std/array: No such file or directory` (CUDA 13 toolkit)**: CUDA 13 moved the CCCL headers into `include/cccl/`, where the NVSHMEM 3.7 headers no longer find them. Add the path for **both** compilers (the host compiler includes `nvshmem.h` too):

  ```bash
  CUDAFLAGS=-I/usr/local/cuda/include/cccl CXXFLAGS=-I/usr/local/cuda/include/cccl \
  NVSHMEM_HOME=<nvshmem install root> pip install -e . --no-build-isolation
  ```
- **CMake error `The link interface of target "nvshmem::nvshmem_host" contains: CCCL::CCCL`** (CUDA-13 NVSHMEM build on a CUDA-12 toolkit): CUDA 12 has no `CCCL::CCCL` CMake target, but CCCL is header-only here, so a stub satisfies it:

  ```bash
  printf 'if(NOT TARGET CCCL::CCCL)\n  add_library(CCCL::CCCL INTERFACE IMPORTED)\nendif()\n' > /tmp/cccl_stub.cmake
  FAST_ULYSSES_CMAKE_ARGS="-DCMAKE_PROJECT_INCLUDE=/tmp/cccl_stub.cmake" \
  NVSHMEM_HOME=<nvshmem install root> pip install -e . --no-build-isolation
  ```
