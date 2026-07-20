# API Reference

Everything is exposed from the top-level package:

```python
from fast_ulysses import UlyssesGroup, AsyncA2AHandle
```

Shape conventions used throughout: `b` batch, `d` head dim, `ws = world_size`,
`s_local = s_global / ws`, `n_local = n_global / ws`.

## `UlyssesGroup(process_group=None, device=None, initial_pool_bytes=2<<30)`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `process_group` | `torch.distributed.ProcessGroup` or `None` | Bootstrap process group; `None` uses `dist.group.WORLD`. **Must span all ranks** — the NVSHMEM bootstrap is world-collective, so a subgroup raises `NotImplementedError`. |
| `device` | `torch.device` or `None` | This rank's CUDA device; `None` uses the current device. |
| `initial_pool_bytes` | `int` | NVSHMEM symmetric-heap reservation, default `2<<30` (2 GiB). Every collective's output buffer is carved from this pool (reused per `tag`). NVSHMEM sizes the heap when it (re)initializes — i.e. from the first **live** group; while any group is alive, a later group's larger request may not be backed (it warns). Destroying all groups finalizes NVSHMEM, so the next group re-initializes with its own size. |

Construction broadcasts the NVSHMEM unique id with `dist.broadcast`, runs `init_world`, and wraps
the sequence in `dist.barrier`s — **all ranks must construct the group together** (construction is
itself collective).

## `all_to_all_single_4d(x, *, mode=0, tag="", use_tma=None) -> Tensor`

| Parameter | Type | Meaning |
| --- | --- | --- |
| `x` | `Tensor` | 4D CUDA tensor, `float16`/`bfloat16`; `.contiguous()` is applied internally. |
| `mode` | `int` | `0` (scatter heads / gather sequence) or `1` (its inverse). |
| `tag` | `str` | Labels the symmetric-heap output buffer (reused per `tag+shape+dtype`). **Concurrently-live results (e.g. q/k/v) must use distinct tags**, otherwise they alias one buffer and clobber each other. |
| `use_tma` | `bool` or `None` | Kernel-path tri-state, see below. |

**Input / output shapes**

| mode | input `x` | output |
| --- | --- | --- |
| 0 | `(b, s_local, n_global, d)` | `(b, s_global, n_local, d)` |
| 1 | `(b, s_global, n_local, d)` | `(b, s_local, n_global, d)` |

**The `use_tma` tri-state**

- `None` (auto): sm<9 → non-TMA; **sm90+ → on first sight of a shape, both paths are
  micro-benchmarked at runtime and the faster one is cached**; later calls hit the cache directly.
  This replaces any offline static table and adapts to the actual hardware.
- `True`: force TMA. `TORCH_CHECK` error when TMA is unavailable or infeasible: sm<9, `d > 256`
  (tensormap boxDim cap), or no tile config fits the device's dynamic-smem cap (e.g. sm_120's
  ~99KB). The auto path treats these cases as "non-TMA only" instead of erroring.
- `False`: force non-TMA.

**Collective hard constraints (violating them hangs the whole group)**

- All ranks must call this method with the **same `(shape, mode, use_tma)` sequence**. `use_tma` is
  as strict as `shape`/`mode` — a mismatch sends ranks down different kernels/barriers and forks the
  internal cache key, which hangs.
- The first call for a given `(shape, mode, use_tma)` runs a **lazy micro-benchmark** to pick the
  best launch config (the auto path additionally compares both kernels) and caches it (later hits
  add zero collective overhead). Under strict SPMD all ranks miss the same entry on the first call
  together, hence hang-free.
- Sync and async calls **both count** in the rank-uniform call sequence (both advance the same
  per-group barrier epoch; sync calls run on the caller's stream, async on the comm stream).

## `all_to_all_single_4d_async(x, *, mode=0, tag="", use_tma=None) -> AsyncA2AHandle`

Async variant: the collective is submitted to the group's dedicated high-priority comm stream and
the call returns immediately; kernels submitted to the caller's stream afterwards overlap with the
a2a. `handle.wait()` makes the **caller's** current stream wait (GPU-side event wait — the host
does not block) and returns the output view. Collective constraints are identical to the sync call.

**Ordering constraint when mixing with sync calls**: the `fast_barrier` epoch is one per-group
monotonic counter, so barrier kernels must execute in submission order. `wait()` every outstanding
async handle of the group **before** issuing the next sync collective on the main stream — the data
dependency forces the comm-stream barriers to complete first.

**Overlap in practice (measured on 8×H200)**: the direct-write scatter is an SM-resident large
grid; cooperative-launch GEMMs (e.g. cuBLAS nvjet) release no SM slots while running, so the a2a
can only wait for them to drain (nsys shows zero overlap). The async API pays off in
non-cooperative compute windows — or use the CE path below, which overlaps by construction.

## `all_to_all_single_4d_ce(x, *, mode=0, tag="") -> Tensor`

CE (**copy-engine**) transfer path — the third path next to the SM scatter (`use_tma=False`) and
TMA (`use_tma=True`). Identical collective semantics, layouts, tag-scoped output buffers and
barrier epochs, but the transfer is a per-peer `cudaMemcpy2DAsync` fan-out on the GPU's DMA
engines (one pitched 2D copy per `(peer, b)`; internal per-peer streams joined back with events,
then the usual flag barrier).

Why it exists: the DMA engines use **no SMs at all**, so the transfer keeps running at full NVLink
bandwidth while compute kernels hold every SM block slot. Measured on 4×H200: 385 GB/s per peer,
pitched rows of `n_local*d*2B` at zero throughput loss, **unaffected by a full-SM spin kernel** —
whereas both kernel paths serialize behind nvjet GEMMs. Pair it with
`all_to_all_single_4d_ce_async` to actually hide the a2a behind concurrent compute.

Notes:
- Path choice is explicit; the `use_tma=None` auto-tune does **not** consider CE.
- No autotune micro-benchmark, no launch config — first calls are collective-safe by construction.
- Per call it issues ~`world_size` memcpy launches (a few µs each): prefer the kernel paths for
  tiny shapes or latency-bound regimes.
- Same rank-uniform call-sequence constraint as every other collective (sync and async advance the
  same barrier epoch).

## `all_to_all_single_4d_ce_async(x, *, mode=0, tag="") -> AsyncA2AHandle`

Async CE variant (same comm-stream launch and ordering constraint as
`all_to_all_single_4d_async`). Because the transfer rides the DMA engines, the in-flight window
overlaps concurrent GEMMs/attention instead of time-slicing with them.

## `destroy() -> None`

Releases the symmetric-heap resources (internally: drain the comm stream, `dist.barrier`, then
destroy). All ranks must call it together. Dropping a group without calling `destroy()` leaks the
symmetric heap (with a warning) — the teardown is collective, so it cannot run from GC.

---

# Environment variables

Set by `UlyssesGroup.__init__` (before NVSHMEM init):

| Variable | Value | Why |
| --- | --- | --- |
| `NVSHMEM_SYMMETRIC_SIZE` | `initial_pool_bytes` | Heap reservation must be set via env before NVSHMEM init. |
| `NVSHMEM_DISABLE_NVLS` | `1` (setdefault) | P2P direct writes don't need NVLS; its multicast heap mapping segfaults on some nodes. |
| `NVSHMEM_REMOTE_TRANSPORT` | `none` (setdefault) | Single-node op; the IB remote transport segfaults NVSHMEM init on IB-equipped nodes. |

Read by the library / build / tests:

| Variable | Where | Meaning |
| --- | --- | --- |
| `FAST_ULYSSES_CUDA_ARCH` | build (`setup.py`) | Target compute capabilities, `;`-separated. Default `80;90;100;120`. |
| `FAST_ULYSSES_USE_TMA` | `benchmark/bench_uniform.py` | Unset → auto, `0` → non-TMA, else → TMA. |
| `FAST_ULYSSES_TEST_NPROC` | `tests/test_multigpu.py` | Overrides the torchrun process count (e.g. odd world sizes). |
