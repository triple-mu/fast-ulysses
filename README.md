# fast-ulysses

Minimal equal-split Ulysses all-to-all with no layout, pack, unpack, or staging tensors.

Supported:

- one rank per GPU, with 1, 2, 4, or 8 GPUs;
- contiguous `[B, S, H, D]` FP16/BF16 tensors;
- equal splits, and eager execution with autograd off -- `torch.inference_mode()` or
  `torch.no_grad()`;
- one owning host thread and one CUDA stream, bound for the group's entire lifetime;
- mode 0 `[B, S_local, H_global, D] -> [B, S_global, H_local, D]`;
- mode 1 `[B, S_global, H_local, D] -> [B, S_local, H_global, D]`.

There is no varlen, uneven split, autograd, async work wrapper, plan cache, CUDA Graph,
`torch.compile`/export, fault recovery, or release-wheel machinery. This is a controlled
single-node operator prototype, not a general distributed runtime.

## Install

Only an editable build from a complete source checkout is supported. Wheels, sdists, and installs
from an exported Python-only tree are not supported.

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

For example:

```bash
CMAKE_BUILD_PARALLEL_LEVEL=32 FAST_ULYSSES_CUDA_ARCH=100 python -m pip install -e .
```

## Use

```python
import torch

from fast_ulysses import UlyssesGroup

stream = torch.cuda.current_stream()
with torch.inference_mode():
    with UlyssesGroup(stream=stream) as group:
        output = group.all_to_all_4d(x, mode=0)
        consume(output)  # on the same bound stream
```

A caller with a fallback wants `create()` instead, which returns `None` on **every** rank if any
rank could not build a group, and `unsupported_reason()` instead of catching a refusal:

```python
group = UlyssesGroup.create(stream=stream)   # None everywhere, or a group everywhere
if group is not None and group.supports(x.shape, x.dtype, mode=0):
    output = group.all_to_all_4d(x, mode=0)
else:
    output = fallback(x)
```

Both exist because the obvious spellings are wrong. `except: fall back` around a constructor is
only safe when the failure is the same on every rank, which a device or stream a caller got wrong
on one rank is not; and a caller that transcribes the shape limits instead of asking will get them
wrong somewhere the library cannot see. `unsupported_reason` is pure, collective-free and
rank-invariant -- it depends only on the mode, shape, dtype and world size -- so skipping a call on
the strength of it skips it on every rank. `supports_world_size()`, `supports_dtype()` and
`SUPPORTED_WORLD_SIZES` answer the same way before a group exists.

The thread that constructs the group owns it. `allocate_output()`, `all_to_all_4d()`, and
`destroy()` must run on that thread with the group's device and bound stream current. Passing a
different stream per call is intentionally unsupported. A producer on another stream must record
an event and make the bound stream wait before the call; a consumer on another stream must wait
for the bound stream before reading the result, then make the bound stream wait for that consumer
before the workspace is overwritten or the group is destroyed. The input is recorded on the bound
stream so the caching allocator cannot recycle it while a copy is in flight. Nothing on this path
blocks the host: the call returns as soon as the copies and both barriers are enqueued.

The first call for each `(mode, shape, dtype)` collectively creates a registered output workspace;
later calls reuse it automatically. The returned workspace is overwritten by the next call with
the same key. `allocate_output` and an explicit `out` remain available only when two results of
the same geometry must stay live at once. `destroy()` releases all registered workspaces; there is
no per-output release step. Always call it collectively, preferably through the context manager.
Dropping a live group only emits `ResourceWarning`: its destructor deliberately performs no
collective cleanup.

The first accepted exchange binds each output workspace to the input Storage, address, offset,
and byte range used for that exchange. Every later use of that output must use the same input
Storage. Keep a fixed input buffer and copy producer results into it when an application would
otherwise allocate a fresh tensor each iteration.

Every rank must execute construction, cold output allocations, exchanges, and destruction in
exactly the same order. Cold allocations compare their
ordinal, mode, shape, dtype, and byte count across ranks. Exchanges deliberately add no hot-path
host collective for argument validation. A rank-local error, timeout, or process exit therefore
poisons the run: terminate every torchrun process with an external watchdog, and never catch an
error and continue using the group. Construction is the one exception, and only because `create()`
agrees its outcome explicitly; nothing after it does.

Every peer is reached through torch symmetric-memory windows and written with pitched copies, so
the sequence/head relayout is folded into the copy strides and the application tensor layout never
changes. On a two-socket host the peers do not all run at the same rate -- a copy that crosses the
socket boundary is roughly half as fast as one that does not, under the concurrent all-to-all
pattern -- and each copy carries an NVTX range naming its class so a profile can tell them apart.

## Test

`test_correctness.py` runs under torchrun and inference mode. It covers representative shapes and
both supported dtypes/modes against the NCCL reference, automatic and explicit workspaces,
fixed-storage and execution-context rejection paths, destruction, and back-to-back calls with
selected ranks deliberately skewed. Its raw no-barrier loop is an armed control: it runs the same
skewed pattern without a barrier and must tear. A run whose control stays clean saw nothing, and is
a blind run rather than a passing one -- raise `SKEW_CYCLES` until it tears again before believing
anything else.

```bash
torchrun --standalone --nproc_per_node=8 test_correctness.py
torchrun --standalone --nproc_per_node=4 test_correctness.py
```

## Benchmark

`benchmark.py` runs in inference mode and checks results against NCCL before timing. It records the
source revision, package and software versions, host, GPU, backend, and run parameters, then
reports both every slowest-rank trial sample and its median. In a source export without `.git`, set
`FAST_ULYSSES_SOURCE_REV` to the exact snapshot commit or content ID.

It compares:

- `raw`: pre-packed NCCL `all_to_all_single`, communication only;
- `layout`: preallocated NCCL pack + communication + unpack;
- `fast`: direct P2P into the final layout through the automatic output pool;
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
