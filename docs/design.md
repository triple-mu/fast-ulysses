# Design

Why the code is shaped this way, and what it rests on that is not guaranteed. Contracts are in
[api.md](api.md); numbers are in [benchmark.md](benchmark.md).

## The transfer

Peer windows are written with plain `cudaMemcpy2D/3DAsync` into addresses obtained from torch
symmetric memory. That uses the copy engines and **no SMs**, which is the point: an SM-resident
collective cannot get a block slot while GEMMs hold every SM, and this one does not need one.

The sequence/head relayout is expressed as source and destination strides on those copies, so it
costs nothing beyond the transfer that had to happen anyway. That is why the baseline's two permute
kernels do not appear on this side at all.

All the addressing lives in `src/a2a_plan.cc`, which has no CUDA, no torch and no communication
library in it. The layout contract can therefore be tested on a machine with no GPU
(`test/test_plan.py`), which is also the only correctness check CI can run.

Uneven shards are the general case; even splits are `seq_splits = [s/P] * P`. There is one code
path, so there is one thing to get right.

Remote copies are serialised on one stream because they all leave through the same egress and
separate streams only make them contend. This rank's own share crosses no link, so it runs on the
caller's stream alongside them.

## Where the memory comes from

Windows are torch symmetric-memory tensors, allocated from a `MemPool` the group owns and
registered with `symm_mem.rendezvous()`, which returns each peer's address for that allocation.

The pool has to be **ours** rather than torch's implicit one. `rendezvous` is collective, so every
rank must reach it for the same allocation at the same point in its own program. A pool only this
group allocates from is what keeps the allocation sequence identical across ranks; sharing torch's
pool would let an unrelated allocation reorder it, and one rank would then rendezvous while another
did not.

The consequence is that memory is no longer committed up front and no longer grows monotonically:
dropping a window returns it to the caching allocator. A window is matched by capacity and grows if
a later call needs more, so a call site costs one window at its high-water mark.

Two internal windows exist per dtype, one per stream the collectives run on. A window is
single-buffered, so two calls may share one only when the stream orders them — and the sync call
runs on the caller's stream while the async one runs on the comm stream.

## The barrier

A one-block spin kernel, publishing with a release store and waiting with an acquire load, over a
device-resident epoch counter. Both live in the allocation's own signal pad, so the handshake state
is per window by construction.

The epoch is on the device rather than computed on the host so that a CUDA-graph capture replays
correctly — a host-computed epoch would bake a constant into the graph and every replay would be
satisfied by stale state.

`cuStreamWriteValue64` / `cuStreamWaitValue64` would remove the last kernel launch from the path
and were tried. Two things against them: they measured worse under concurrent compute, which is the
regression this operator exists to avoid, and the waiting form needs a remote-write-flush device
attribute that much of the target hardware does not have. The spin kernel's inline PTX needs only
`sm_70`.

## What this rests on that is not documented

**A completed copy-engine write is visible at the destination by the time a later kernel's release
store announcing it arrives.** No vendor document says so:

- the CUDA API reference defines memcpy completion as a *host-side* property;
- the Programming Guide's cross-device ordering guarantee is scoped to the NULL stream and is
  withdrawn for async copies on a non-default stream;
- PTX scopes `.release` to "prior operations from the current thread", which a copy-engine transfer
  is not;
- neither NVSHMEM nor NCCL pairs a host-issued CE transfer with an SM release store; both keep the
  flag on the data's own path.

It holds in testing, which is evidence and not a guarantee.
`test/distributed/ce_ordering.py` tests it and arms its own negative control on every run, so the
test cannot silently stop testing.

## Why NVLink only

Over PCIe the operator is correct and, within one CPU socket, as fast relative to the baseline as
it is on NVLink. Across a socket boundary it is about 0.62× of `torch.distributed` — not because
the transfer is slow, but because `all_to_all_single` does not use direct GPU P2P there. It routes
around the boundary through the InfiniBand NICs or through host shared memory; this transport always
writes peer memory directly. Deny NCCL that bypass and we are 3.8–4.9× faster on the same path.

Measured on two 8-GPU PCIe machines, 4/4 across NUMA nodes, both CPU vendors. Cross-socket P2P on
the Intel machine did not scale with concurrency at all: four concurrent pairs moved no more than
one. Pitched copies and additional streams were both tested and refuted as explanations.

Matching NCCL there means a shared-host-memory transport with a second handshake — a new transport,
not a scheduling change, worth nothing on NVLink. Not built, so the constructor refuses the
topology instead of pretending.

## The async result

`all_to_all_4d_async` returns an `AsyncCollectiveTensor` registered against torch's work registry,
so the first aten op on the result waits by itself. The registry keys on the output's **storage**,
so a registry entry belongs to that call.

When the registry is unavailable in the linked libtorch, the same function returns a handle with an
explicit `.wait()`. A result that is simply dropped leaves an entry behind; torch prints a count of
the survivors at process exit.

The sync collective stays on the caller's stream rather than the comm stream. Routing it through
would cost two event hops per call, which is comparable to the collective itself.

## Removed, and why

Earlier versions carried more surface than they could defend:

- **NVSHMEM.** It supplied the same thing torch symmetric memory does — VMM allocations with peer
  addresses — at the cost of a bootstrap, a strided-team split, an entire pool allocator, two
  undocumented `hostlib_` entry points chosen to avoid a symbol clash with torch's own bundled
  NVSHMEM, and a rule that live groups must partition the job (unenforceable, because the check
  would have to run on ranks that never reach the constructor). None of that survives the swap.
- **A borrowed entry point** whose result was the window itself, under a lifetime contract nothing
  enforced. `out=` from `empty_output()` gives the same saving with the buffer's lifetime in the
  caller's hands.
- **`tag`.** It existed to keep concurrently-live results in separate windows. The copying path
  gives each call its own tensor and the zero-copy path uses the caller's own buffer, so there is
  nothing left to disambiguate.
- **`reserve()` / `seal()`.** They existed because the old pool's allocation was collective and
  stream-synchronising, which deadlocked if it happened mid-call. Allocation is still collective,
  but the deadlock it fed on — a spin barrier deliberately left in flight — went with the deferred
  closing handshake.
- **A `torch.distributed` fallback group** for the topology above. The answer there is to use
  `torch.distributed` directly.
