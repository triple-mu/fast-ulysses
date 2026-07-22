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
| `initial_pool_bytes` | `int` | NVSHMEM symmetric-heap reservation, default `2<<30` (2 GiB); every collective's output buffer comes from this pool (reused per `tag`). The heap is sized by the **first live** group — a later, larger request only warns; destroying all groups lets the next one re-size. |

Construction broadcasts the NVSHMEM unique id over `torch.distributed` and is itself collective:
**all ranks must construct the group together**.

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

- `None` (auto): sm<9 → non-TMA; sm90+ → on first sight of a shape, **both paths are
  micro-benchmarked and the faster one is cached**.
- `True`: force TMA. `TORCH_CHECK` error when TMA is unavailable or infeasible: sm<9, `d > 256`
  (tensormap boxDim cap), or no tile config fits the device's dynamic-smem cap (e.g. sm_120's
  ~99KB). Auto treats these cases as "non-TMA only" instead of erroring.
- `False`: force non-TMA.

**Collective hard constraints (violating them hangs the whole group)**

- All ranks must call with the **same `(shape, mode, use_tma)` sequence** — a mismatch sends ranks
  down different kernels/barriers and forks the internal cache key.
- The first call per `(shape, mode, use_tma)` runs a **lazy micro-benchmark** and caches the launch
  config; under SPMD all ranks miss the same entry together, hence hang-free.
- Sync and async calls **both count** in the sequence (both advance the same per-group barrier
  epoch; sync runs on the caller's stream, async on the comm stream).

## `all_to_all_single_4d_async(x, *, mode=0, tag="", use_tma=None, barrier=True) -> AsyncA2AHandle`

Submits the collective to the group's high-priority comm stream and returns immediately;
`handle.wait()` makes the **caller's** current stream wait (GPU-side — the host does not block)
and returns the output view. Constraints identical to the sync call.

**Mixing with sync calls**: barrier kernels must execute in submission order (one per-group
epoch), so `wait()` every outstanding async handle **before** the next sync collective.

**Grouped handshake (`barrier=False`)**: several async calls share ONE handshake — pass
`barrier=False` on all but the last call (e.g. q, k, v of one layer). Only the barrier-carrying
handle's `wait()` guarantees peers' writes have landed; a `barrier=False` handle's `wait()` orders
this rank's own work only. All ranks must use the identical barrier pattern. Saves N-1 barrier
kernels and N-1 cross-rank skew couplings per group.

**Overlap in practice (8×H200)**: cooperative-launch GEMMs (e.g. cuBLAS nvjet) release no SM
slots, so the SM-resident scatter just waits for them to drain — nsys shows zero overlap. Use the
CE path below for those windows.

## `all_to_all_single_4d_ce(x, *, mode=0, tag="") -> Tensor`

CE (**copy-engine**) transfer path: identical collective semantics, layouts, tags and barrier
epochs, but the transfer is a per-peer pitched `cudaMemcpy2DAsync` fan-out on the DMA engines
(per-peer streams joined back with events, then the flag barrier).

The DMA engines use **no SMs**, so the transfer keeps full NVLink bandwidth while compute holds
every SM slot. Measured on 4×H200: 385 GB/s per peer,
pitched rows of `n_local*d*2B` at zero throughput loss, **unaffected by a full-SM spin kernel** —
whereas both kernel paths serialize behind nvjet GEMMs.

Notes:
- Path choice is explicit — the `use_tma=None` auto-tune does not consider CE.
- No autotune, no launch config; first calls are collective-safe by construction.
- ~`world_size` memcpy launches per call (a few µs each): prefer the kernel paths for tiny shapes.
- Same rank-uniform call-sequence constraint as every other collective.

## `all_to_all_single_4d_ce_async(x, *, mode=0, tag="", barrier=True) -> AsyncA2AHandle`

Async CE variant (same comm-stream launch and ordering constraint as
`all_to_all_single_4d_async`); its in-flight window genuinely overlaps concurrent GEMMs/attention.
`barrier=False` grouping works exactly as on the kernel path. (The sync CE call deliberately has
no `barrier` parameter: a deferred sync result would be an unreadable view with nothing left to
publish it.)

## `destroy() -> None`

Releases the symmetric-heap resources (drain comm stream, `dist.barrier`, destroy). All ranks must
call it together. Dropping a group without `destroy()` leaks the heap with a warning — the
teardown is collective, so it cannot run from GC.

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
| `FAST_ULYSSES_CMAKE_ARGS` | build (`setup.py`) | Extra CMake `-D` flags (see docs/INSTALL.md Troubleshooting). |
| `FAST_ULYSSES_USE_TMA` | `benchmark/bench_uniform.py` | Unset → auto, `0` → non-TMA, else → TMA. |
| `FAST_ULYSSES_TEST_NPROC` | `tests/test_multigpu.py` | Overrides the torchrun process count (e.g. odd world sizes). |
