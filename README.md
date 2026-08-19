# fast-ulysses

Minimal equal-split Ulysses all-to-all that writes directly into the final layout. The
steady-address fast path has no pack, unpack, or staging tensor; an mlx5 input whose address
keeps moving uses one persistent staging block and a device-to-device copy.

Supported:

- one rank per GPU, with 1, 2, 4, or 8 GPUs;
- contiguous `[B, S, H, D]` FP16/BF16 tensors;
- equal splits, and eager execution -- no `torch.compile`, no CUDA Graph capture;
- any stream, and one owning host thread per group;
- batch size 1 on the 8-GPU mlx5 path, where `heads * head_dim * itemsize` summed over all
  ranks must also fit 65535 bytes -- the MKey stride field. Larger shapes fall back to the
  process group's own all-to-all;
- mode 0 `[B, S_local, H_global, D] -> [B, S_global, H_local, D]`;
- mode 1 `[B, S_global, H_local, D] -> [B, S_local, H_global, D]`.

There is no varlen, uneven split, autograd, async work wrapper, plan cache, CUDA Graph,
`torch.compile`/export, fault recovery, or release-wheel machinery. This is a controlled
single-node operator prototype, not a general distributed runtime.

## Install

Only Python 3.11+ and an editable build from a complete source checkout are supported. Wheels,
sdists, and installs from an exported Python-only tree are not supported.

```bash
FAST_ULYSSES_CUDA_ARCH=100 python -m pip install -e .
```

Both checkout-local editable entry points are supported in the active environment:

```bash
python -m pip install -e .
python setup.py develop
```

The architecture is detected with `nvidia-smi` when `FAST_ULYSSES_CUDA_ARCH` is not set. The build
locates Torch CMake files directly in the active virtual environment, so PEP 517 build isolation
does not install or import a second copy of PyTorch.
It uses up to 32 parallel jobs by default; override that with `CMAKE_BUILD_PARALLEL_LEVEL` or
`FAST_ULYSSES_BUILD_JOBS`. If `ccache` is on `PATH`, it is enabled automatically for both C++ and
CUDA. `FAST_ULYSSES_CCACHE=/path/to/ccache` selects it explicitly, while
`FAST_ULYSSES_CCACHE=0` disables it.
The build also links the system `libibverbs` and `libmlx5` libraries.
The checkout-local `build/` directory is persistent. Remove it before rebuilding after changing
the Python ABI, PyTorch/CUDA installation, or CUDA architecture so CMake does not reuse stale
discovery results from the previous environment.

For example:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=32 FAST_ULYSSES_CUDA_ARCH=100 python -m pip install -e .
```

## Use

```python
import torch
import fast_ulysses as fu

with torch.inference_mode():
    q, k, v = split(fu.scatter_heads(qkv))   # [T_local, H, ...] -> [T_global, H/ws, ...]
    out = fu.gather_heads(attention(q, k, v))
```

A tensor goes in and a tensor comes out, on whatever stream the caller is on, like any other
torch operation. `fu.all_to_all_4d(x, mode=0)` is the same call spelled with the mode; `mode=0`
splits heads and gathers sequence, `mode=1` is its inverse. Pass `group=` for a process group
other than the default.

**There is nothing to set up and nothing to release.** The transport for a process group is
built on first use and cached; the result is an ordinary tensor the caller owns and frees.
`fu.shutdown()` releases every transport and is collective, but a process that is exiting does
not have to call it.

**Any shape a transport cannot carry falls back to the process group's own all-to-all**, with
the relayout the library would otherwise fold away. So does a group that could not build a
transport at all. The decision depends only on the mode, shape, dtype, world size and transport
-- every term identical on every rank -- so the ranks never take different paths, and a caller
never needs `if transport is not None`. `fu.backend()` reports which of `mlx5`, `p2p` and
`fallback` is carrying a group, and `fu.unsupported_reason(shape, dtype, mode)` says why a shape
would fall back, before issuing it.

On mlx5 the NIC reads an input through a memory region registered once against its address, and
nothing is registered on p2p. The library holds no reference to the input, so a producer that
allocates, uses and frees the same buffer every iteration gets the same block back and the
region keeps matching -- which is what a loop does, and it costs nothing. A producer whose
address genuinely moves has its input copied into the registered block instead
(`fu::stage_moving_input` in an NVTX trace), because re-registering is milliseconds against a
copy that is tens of microseconds.

If the caching allocator hands that segment back to the driver -- `torch.cuda.empty_cache()`,
or recovering from an OOM -- the registration is dropped and remade on the next call rather than
read through.

The thread that first uses a process group owns its transport, and every later call for that
group must come from the same thread -- two threads issuing collectives into one process group
hang however the streams are arranged.

Results come from a symmetric-memory pool the group owns, through torch's caching allocator, so
a result that is dropped is reused by the next call and a result that is held is not. Two live
results are two tensors. Nothing is overwritten behind the caller's back.

**Every rank must issue the same sequence of shapes and retain or drop returned tensors in the
same pattern.** Allocation reuse determines which call has to register and rendezvous a peer
buffer. If one rank retains a result while another drops the corresponding result, their pools
can choose different blocks and the next exchange may hang or address the wrong peer block. The
first time a geometry is seen, the ranks compare it and say so loudly; already-familiar geometry
or lifetime divergence has no hot-path collective to diagnose it. This is a deliberate prototype
constraint: checking it completely would add a collective to every call.

A rank-local error, timeout, or process exit poisons the run: terminate every torchrun process
with an external watchdog, and never catch an error and continue. Construction is the one
exception, because it agrees its outcome across the ranks explicitly before anything commits.
The barrier is the other half of that: it gives up on a peer that has not arrived within a
minute and traps with the rank and epoch it was waiting for, rather than waiting forever.

On the supported 8-GPU PCIe host, same-socket transfers use CUDA IPC pointers. Cross-socket
transfers use mlx5 interleaved MKeys: the NIC gathers or scatters the strided `[S,H,D]` slices
directly, so the application tensor layout never changes. The closest NIC is selected from sysfs.

Set `FAST_ULYSSES_DISABLE_RDMA=1` to use CUDA P2P only. To override NIC discovery, set all eight
rank-local devices explicitly, for example:

```bash
export FAST_ULYSSES_NICS=mlx5_2,mlx5_3,mlx5_0,mlx5_1,mlx5_6,mlx5_7,mlx5_4,mlx5_5
```

## Test

`test_correctness.py` runs under torchrun and inference mode. It covers representative shapes,
including a non-page-sized RDMA registration, and both supported dtypes/modes against the process
group's own all-to-all plus a small independent rank/sequence/head oracle. It checks that two live
results are two tensors and a dropped one is reused, that the answer does not depend on where the
input was allocated, cross-stream ordering, the fallback, rejection and context paths, and
back-to-back calls with selected ranks deliberately skewed. Its raw P2P no-barrier loop is only an informational scheduler
smoke check: it does not exercise verbs, CQ completion, or NIC flush and is not an mlx5 correctness
oracle. An 8-rank run with RDMA enabled expects `mlx5`, so an unintended P2P fallback fails; set
`FAST_ULYSSES_EXPECT_BACKEND=p2p|mlx5` only when deliberately testing another admitted setup.

```bash
torchrun --standalone --nproc_per_node=8 test_correctness.py
FAST_ULYSSES_DISABLE_RDMA=1 torchrun --standalone --nproc_per_node=8 test_correctness.py
```

Run both: the two backends synchronise differently, and only the second one exercises batch > 1.

## Benchmark

`benchmark.py` runs in inference mode and checks results against NCCL before timing. It records the
source revision, package and software versions, host, GPU, backend, and run parameters, then
reports both every slowest-rank trial sample and its median. In a source export without `.git`, set
`FAST_ULYSSES_SOURCE_REV` to the exact snapshot commit or content ID. The report includes the
configured `FAST_ULYSSES_NICS` mapping (or marks automatic discovery).

It compares:

- `raw`: pre-packed NCCL `all_to_all_single`, communication only;
- `layout`: preallocated NCCL pack + communication + unpack;
- `fast`: the selected transport writing into the final layout, allocating the result per call
  as a caller does;
- `GB/s`: per-rank remote-payload throughput, equivalent to NCCL bus bandwidth for all-to-all;
- `vs raw` and `vs layout`: baseline latency divided by fast latency.

For `N` ranks, NCCL algorithm bandwidth is `bus GB/s * N / (N - 1)`, and aggregate remote
throughput is `bus GB/s * N`. The Markdown report includes both values for raw NCCL.

Every case runs untimed warmup iterations first. Ranks are aligned outside the timed region before
each iteration. Each iteration records the slowest rank; the table is the median across trials.

```bash
torchrun --standalone --nproc_per_node=8 benchmark.py \
  --seq-len 37824 --num-heads 56 --head-dim 128 \
  --report benchmark_report.md
```

`seq-len` is the global sequence length, not the per-rank length. The defaults are 10 warmup calls,
one measured call per trial, and the median of 20 trials.
