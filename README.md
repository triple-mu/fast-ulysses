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

`pending` — to be measured in one pass across machines. Method and reproduction commands are in
[docs/benchmark.md](docs/benchmark.md).

## Quick start

`torchrun --nproc_per_node=8 example.py`:

```python
import torch
import torch.distributed as dist
from fast_ulysses import UlyssesGroup

dist.init_process_group("nccl")
torch.cuda.set_device(dist.get_rank())
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

## API

| Entry point | Purpose |
| --- | --- |
| `UlyssesGroup(process_group=None, device=None, require_nvlink=True)` | Collective. Refuses a group whose GPUs are not NVLink-joined. |
| `group.all_to_all_4d(x, mode=0, out=None, seq_splits=None, head_splits=None)` | The collective. Returns a tensor the caller owns. |
| `group.all_to_all_4d_async(...)` | The same, on a comm stream, returning an `AsyncCollectiveTensor`. |
| `group.empty_output(x, mode=0, ...)` | A symmetric buffer to pass as `out=`, which removes the copy-out. |
| `group.destroy()` | Release the windows. Collective. |
| `fast-ulysses doctor` | Build, devices, NVLink matrix. |

Shapes, splits and the collective contract: [docs/api.md](docs/api.md). Why the code is shaped
this way, and what it rests on that is not guaranteed: [docs/design.md](docs/design.md).

## Limits

- **NVLink, one node, `world_size` in [1, 8]**, including odd sizes. Over PCIe — especially across
  a CPU socket — `torch.distributed` is faster, because it routes around the link and this
  transport always writes peer memory directly. The constructor refuses such a group.
- `float16` / `bfloat16`, and `d * elem_size` must be 16-byte aligned.
- Forward only: no backward and no meta implementation, so no autograd and no `torch.compile`
  tracing.

## Install

Requires **PyTorch 2.10+**, **CUDA 12.8+ or 13**, and sm80 / sm90 / sm100 / sm120. torch is the
only runtime dependency.

```bash
pip install fast-ulysses                                          # newest torch, from PyPI
pip install -e . --no-build-isolation                             # from source, all four arches
FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation   # one arch, much faster
```

Wheels for other torch versions, and what to do when the import fails:
[docs/install.md](docs/install.md).

## Testing

```bash
pytest                                                     # auto-skips below 2 GPUs
torchrun --nproc_per_node=8 test/distributed/correctness.py
```

Development setup: [docs/develop.md](docs/develop.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
