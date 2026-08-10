<div align="center">

# fast-ulysses

**Ulysses sequence-parallel all-to-all as a torch custom op, moved by the GPU copy engines.**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

</div>

Ulysses sequence parallelism shards a long sequence across GPUs: one all-to-all before attention
trades the sequence shard for a head shard, a second trades it back. For long-sequence video DiT
workloads those two collectives are the critical communication.

This ships that 4D all-to-all as `torch.ops.fast_ulysses.all_to_all_4d`, over NVLink, on one node.

## Key features

- **No SMs.** The peer transfers are pitched `cudaMemcpy2D/3DAsync` straight into peers'
  symmetric-memory addresses, so they run on the copy engines while compute kernels hold every SM.
  (Measured: a peer copy overlaps a concurrent GEMM completely; a same-device copy does not, which
  is why `out=` from `empty_output()` — which removes the copy-out — is the faster path.)
- **Relayout for free.** The sequence/head permutation is expressed as source and destination
  strides on those copies, not as two permute kernels. That is roughly half of what the
  `torch.distributed` path spends.
- **Uneven shards.** Per-rank `seq_splits` / `head_splits`, which is what lets a caller drop
  sequence padding. Even splits are the special case, not a separate path.

## Performance

8 GPUs, one exclusively allocated node per row, the same wheel and the same `.so` on all of them.
`base` is `torch.distributed` permute + `all_to_all_single` + permute. Wan 720p, bf16, ms.

| GPU | fabric | base | ours | vs base | transfer | of the fabric |
|---|---|---|---|---|---|---|
| H200 | NVLink | 1.569 | **0.860** | **1.82×** | 0.680 | 99% |
| B200 | NVLink | 1.194 | **0.550** | **2.17×** | 0.410 | 90% |
| B300 | NVLink | 1.184 | **0.548** | **2.16×** | 0.411 | 90% |
| H100 | NVLink | 1.616 | **0.971** | **1.66×** | 0.715 | 94% |

47–60% of the baseline is relayout that costs nothing here. "Of the fabric" is the transfer against
what a flat peer copy achieves on the same machine — at 90–99% there is nothing left to schedule.
`out=` from `empty_output()` removes the copy-out for a further 1.14–1.28×.

Every number was taken twice, on two different nodes except H100, whose two samples are the same
node. Full tables, per-stage timings and the method: [docs/benchmark.md](docs/benchmark.md).

## Quick start

Save this as `example.py` and run `torchrun --nproc_per_node=8 example.py`:

```python
import os

import torch
import torch.distributed as dist
from fast_ulysses import UlyssesGroup

dist.init_process_group("nccl")
torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
group = UlyssesGroup()

b, s_local, heads, d = 1, 4096, 8, 128
ws = dist.get_world_size()
x = torch.randn(b, s_local, heads * ws, d, dtype=torch.bfloat16, device="cuda")

y = group.all_to_all_4d(x, mode=0)        # (b, s_local * ws, heads, d)
z = group.all_to_all_4d(y, mode=1)        # back to x's shape

group.destroy()
dist.destroy_process_group()
```

Every rank must issue the same sequence of shapes. More: [docs/quickstart.md](docs/quickstart.md).

A real one — a DiT attention block, sequence-parallel, forward and backward, checked against the
same block without sequence parallelism:

```bash
torchrun --nproc_per_node=8 examples/dit_attention.py
```

The collective is differentiable, so that block is an ordinary `nn.Module`: no `autograd.Function`
wrapper, no manual backward.

## API

| Entry point | Purpose |
| --- | --- |
| `UlyssesGroup(process_group=None, device=None, *, require_nvlink=True)` | Collective. Refuses a group whose GPUs are not NVLink-joined. |
| `group.all_to_all_4d(x, *, mode=0, out=None, seq_splits=None, head_splits=None)` | The collective. Returns a tensor the caller owns. |
| `group.all_to_all_4d_async(x, *, ...)` | The same, on a comm stream, returning an `AsyncCollectiveTensor`. |
| `group.empty_output(x, *, mode=0, ...)` | A symmetric buffer to pass as `out=`, which removes the copy-out. `out=` gives up the gradient. |
| `group.destroy()` | Release the windows. Collective. |
| `fast-ulysses doctor` | Build, devices, NVLink matrix. |

Shapes, splits and the collective contract: [docs/api.md](docs/api.md). Why the code is shaped
this way, and what it rests on that is not guaranteed: [docs/design.md](docs/design.md). Coming from
DeepSpeed-Ulysses, sglang's `usp.py` or yunchang — what each call becomes, and what this does not
have: [docs/migration.md](docs/migration.md).

## Limits

- **NVLink, one node, `world_size` in [1, 8]**, including odd sizes. Over PCIe — especially across
  a CPU socket — `torch.distributed` is faster, because the pitched copies this transport is built
  on degrade badly there. The constructor refuses such a group.
- `float16` / `bfloat16` / `float32` / `float8_e4m3fn` / `float8_e5m2` / `int8` / `uint8`, and
  `d * elem_size` must be 16-byte aligned — so `d % 8` at bfloat16, `d % 16` at float8.
- Differentiable, and it propagates shapes under `FakeTensor`. Not traceable by `torch.compile`:
  the group is a torchbind object with no registered fake class, so Dynamo graph-breaks on it.
- The **async** form is not differentiable and says so: its `AsyncCollectiveTensor` is a leaf, so a
  gradient would be dropped silently. Use `all_to_all_4d` when you need one.
- **`out=` is not differentiable either**, and cannot say so: it reaches a mutating op with no
  autograd formula, so the result is the caller's buffer, detached. It is a speed knob for
  inference and no-grad regions, not for a training graph.
- CUDA-graph capture covers the sync call on an already-warmed shape; see
  [docs/design.md](docs/design.md).

## Install

Requires **PyTorch 2.10+**, **CUDA 12.8+ or 13**, and sm80 / sm90 / sm100 / sm120. torch is the
only runtime dependency.

A wheel is built against one (torch minor, CUDA major) and embeds that torch's C++ ABI, so the pair
has to match. Install torch first, then pick the index by the CUDA major it was built against:

```bash
pip install fast-ulysses --index-url https://triple-mu.github.io/fast-ulysses/whl/cu13/   # or cu12
```

That index carries fast-ulysses and nothing else, so pip cannot move your torch to satisfy it: it
backtracks to the wheel pinned to the torch minor you already have. `fast-ulysses wheel-url` names
the row for the environment running now and prints the same install as a fully pinned
`--find-links` command against the release assets, when you want the file itself.

```bash
pip install fast-ulysses                                          # PyPI: torch 2.13.x + CUDA 13
pip install -e . --no-build-isolation                             # from source, all four arches
FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation   # one arch, much faster
```

PyPI carries that one pair, pinned `torch==2.13.*`. On any other torch the first command does not
fail — pip upgrades torch to 2.13, a different CUDA build than the one you chose.

The one thing to know before reaching for `--find-links`: PEP 440 orders local versions, so an
install that does not write `+torch210cu128` out in full takes `torch213cu130` whatever torch you
have, silently. That, the per-row indexes and what to do when the import fails:
[docs/install.md](docs/install.md).

## Testing

```bash
pytest                  # the host-only plan tests always run; the GPU workers skip below 2 GPUs
pytest -m "not multigpu"                                   # host-only, no GPU needed
torchrun --nproc_per_node=8 test/distributed/correctness.py
```

Development setup: [docs/develop.md](docs/develop.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
