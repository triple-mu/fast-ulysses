# Design

Why the code is shaped this way, and what it rests on that is not guaranteed. Contracts are in
[api.md](api.md); numbers are in [benchmark.md](benchmark.md).

## The transfer

Peer windows are written with plain `cudaMemcpy2D/3DAsync` into addresses obtained from torch
symmetric memory. That uses the copy engines and **no SMs**, which is the point: an SM-resident
collective cannot get a block slot while GEMMs hold every SM, and this one does not need one.

Measured on H200, against a concurrent 8192³ fp16 GEMM chain of the same duration: a **peer**
copy overlaps it completely (67.9 ms alone, 67.9 ms concurrent). A **same-device** copy does not
— it runs at 1.79× the longer of the two, where full competition would be 1.99×. So the zero-SM
property is a property of the *remote* copies. Two same-device copies remain: this rank's own
share of the transfer, and the copy-out on the copying path. That is the strongest argument for
`out=` from `empty_output()`, which removes the second one.

`benchmark/bench_a2a.py --mode zerosm` is that A/B, and is how to check it on a machine here:

```bash
./tools/exclusive.sh 0,1,2,3,4,5,6,7 -- torchrun --nproc_per_node=8 \
    benchmark/bench_a2a.py --mode zerosm
```

The same bytes and the same `dst.copy_(src)`, once into a peer's window and once into this rank's
own memory, each under its own GEMM chain matched to that copy's length — the same bytes take
several times longer to reach a peer than to reach local memory (H200 wan-720p in
[benchmark.md](benchmark.md): 255 MB crossing a link in 689 µs of `transfer`, against 291 MB copied
locally in 143 µs of `copy_out`), so one chain cannot be the same duration as both. It goes through
torch symmetric memory rather than through the operator, whose path also contains two barrier
kernels and a copy-out and so cannot attribute anything to the transfer. Besides the pair ratio and
the full-competition reference above, it reports the GEMM chain's **own** slowdown with the copy
underneath it: 1.00× is "that copy cost the chain nothing".
That column is not sufficient on its own — a copy the chain starved never ran under it and reads
1.00× too, at a pair ratio equal to full competition — so the two are read together, and the run
says so in its header. Two GPUs are the minimum: at `world_size = 1` there is no peer arm and the
run says that too. The numbers in the paragraph above predate the mode and were not produced by it;
no table in [benchmark.md](benchmark.md) carries them yet.

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

Windows are torch symmetric-memory tensors, allocated with
`c10d::symmetric_memory::empty_strided_p2p` and registered with `rendezvous()`, which returns each
peer's address for that allocation. Both are `TORCH_API` and unchanged in signature from torch 2.10
through 2.13, so all of this lives in C++ (`src/group.cc`).

Going through the allocator entry point directly, rather than through a `MemPool`, is what makes
the ordering argument simple. `rendezvous` is collective: every rank must reach it for the same
allocation at the same point in its own program. Nothing unrelated can be served from that entry
point, so the sequence of allocations this group makes is the sequence every rank makes, and the
SPMD call contract is the only thing it rests on.

Memory is not committed up front and does not grow monotonically: dropping a window returns it to
the symmetric allocator. A window is matched by capacity and grows if a later call needs more, so a
call site costs one window at its high-water mark. Buffers from `empty_output()` are released when
the caller drops them -- the group holds a record of each, and prunes the records whose buffer no
longer has another holder.

Two internal windows exist per dtype, one per stream the collectives run on. A window is
single-buffered, so two calls may share one only when the stream orders them — and the sync call
runs on the caller's stream while the async one runs on the comm stream.

## The barrier

A one-block spin kernel, publishing with a release store and waiting with an acquire load, over a
device-resident epoch counter. Both live in the allocation's own signal pad, so the handshake state
is per window by construction.

The epoch is on the device rather than computed on the host so that a CUDA-graph capture replays
correctly — a host-computed epoch would bake a constant into the graph and every replay would be
satisfied by stale state. `test/distributed/cudagraph.py` holds this: over eight replays with one
rank deliberately skewed, the epoch advances by two per replay (one per barrier) and every replay
is bit-exact.

What is capturable is narrower than that argument alone suggests, and the boundary is a property of
the allocation rather than of the handshake:

- the **sync** call, on a shape whose window **already exists**. Allocating one is
  `empty_strided_p2p` + `rendezvous` + `zero_()`, none of which is legal inside a capture, so a
  first call for a new shape ends the capture rather than being recorded.
- the **async** call is out entirely: `stage()` waits on an event recorded by the previous,
  uncaptured call, and a cross-graph event dependency invalidates the capture.

Warm every shape eagerly before capturing, and pass `out=` from `empty_output()` so the graph writes
a fixed address rather than one from its private pool.

`cuStreamWriteValue64` / `cuStreamWaitValue64` would remove the kernel launches from the path and
were tried. They measured worse under concurrent compute — overlap fell from +34% to −28% — which
is the regression this operator exists to avoid, and that remains the reason not to use them. The
CUDA memop documentation offers a mechanism: ordering established through those APIs "is not
visible to CUDA", so a blocked memop is invisible to the work scheduler and concurrent kernels can
be queued behind it.

An earlier version of this note also blamed `CU_DEVICE_ATTRIBUTE_CAN_FLUSH_REMOTE_WRITES`. That
attribute gates only the `_FLUSH` variants, which exist for writes arriving over PCIe from a NIC.
NVSHMEM's on-stream wait does require it and does use `FLUSH`; NCCL's CE wait uses neither. The
attribute is therefore not by itself a reason the waiting form cannot be used over NVLink.

The spin kernel's inline PTX needs only `sm_70`.

## What this rests on that is not documented

**A completed copy-engine write is visible at the destination by the time a later kernel's release
store announcing it arrives.** No vendor document says so:

- the CUDA API reference defines memcpy completion as a *host-side* property;
- the Programming Guide's cross-device ordering guarantee is scoped to the NULL stream and is
  withdrawn for async copies on a non-default stream;
- PTX scopes `.release` to "prior operations from the current thread", which a copy-engine transfer
  is not;
- no vendor documents the ordering either way.

We are not alone in relying on it, which is worth knowing before deciding how much to worry.
NVIDIA ships this exact shape in at least two places, both with *weaker* ordering than ours:
NVSHMEM's on-stream `signal_op` takes a CE payload and then publishes the flag from a
`<<<1,1>>>` kernel doing a bare `atomicAdd_system` with no fence (`sync.cpp` → the
`signal_op_kernel` it falls back to whenever the op is `SIGNAL_ADD`), and TransformerEngine's
userbuffers does the same in `ring_exchange`, which is its shipping default. Ours at least
publishes with `red.release.sys`.

NCCL is the one that avoids the shape rather than accepting it. Its kernel-less CE collectives
(`ce_coll.cc`, `NCCL_CTA_POLICY_ZERO`, NCCL 2.28+) publish the flag as an 8-byte device-to-device
`cudaMemcpyAsync` on the same stream as the payload and wait with `cuStreamWaitValue32`, so the
flag rides the engine the data rode. That is the shape to move to if this assumption ever has to
go; the cost, and why it is not obviously better, is in the barrier section below.

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
