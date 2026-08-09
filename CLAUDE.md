# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

One torch custom op — the Ulysses 4D all-to-all — where the transfer is pitched
`cudaMemcpy2D/3DAsync` straight into peers' torch symmetric-memory addresses. It uses **zero SMs**
(copy engines only) and folds the sequence/head relayout into copy strides instead of two permute
kernels. NVLink, one node, `world_size ∈ [1, 8]`, `float16`/`bfloat16`.

## Commands

```bash
# Build. --no-build-isolation is REQUIRED: setup.py imports the target torch.
FAST_ULYSSES_CUDA_ARCH=90 pip install -e ".[dev]" --no-build-isolation   # one arch, much faster
pip install -e . --no-build-isolation                                    # all of 80;90;100;120

pre-commit run --all-files        # the single lint entry point (ruff + clang-format v15.0.7)

pytest                            # everything runnable here
pytest -m "not multigpu"          # host-only, no GPU needed (test_plan.py)
pytest -m multigpu                # the torchrun-wrapped workers
FAST_ULYSSES_TEST_NPROC=3 pytest -m multigpu     # force an odd world size

# Workers run directly — this is the debugging path:
torchrun --nproc_per_node=8 test/distributed/correctness.py
torchrun --nproc_per_node=8 test/distributed/validation.py
torchrun --nproc_per_node=8 test/distributed/ce_ordering.py
torchrun --nproc_per_node=8 test/distributed/cudagraph.py

fast-ulysses doctor               # build facts, devices, NVLink matrix

# Benchmarks MUST go through exclusive.sh; a CONTENDED number is not a number.
./tools/exclusive.sh 0,1,2,3 -- torchrun --nproc_per_node=4 benchmark/bench_a2a.py \
    --mode {stages,overlap,padding,zerocopy,sweep,link,zerosm}
```

`build/` is a persistent CMake tree, so rebuilds are incremental. `rm -rf build` only when the repo
has moved.

## Architecture

| layer | files | what it owns |
|---|---|---|
| addressing | `include/fast_ulysses/a2a_plan.hpp`, `src/a2a_plan.cc` | **Pure host arithmetic. No CUDA, no torch, no communication library.** Dims → a list of `CopyOp` (pitched copies, byte offsets). `src/plan_bindings.cc` exposes it as `torch.ops.fast_ulysses.a2a_plan_debug` so `test/test_plan.py` can replay it over numpy with no GPU. |
| transport | `src/transfer.cu` | Issues `plan.ops`. Remote peers serialised on ONE stream, this rank's own share on the caller's stream, joined with fresh per-call events. XOR-shift peer order. |
| sync | `src/barrier.cu` | One-block spin kernel over `uint64 flags[ws]` + `uint64 epoch`, all inside the allocation's signal pad. |
| op surface | `src/bindings.cc` | Validation, aliasing guard, barrier→transfer→barrier→(copy-out). The group object is **one CUDA stream** plus rank/world_size. |
| memory | `src/group.cc` | Everything that survives a call: the symmetric windows and their handshake state, the plan cache, the staging buffers. |
| topology | `src/nvlink.cc` | NVML link-type probing, through `dlopen`. |
| API | `python/fast_ulysses/group.py` | What has no C++ equivalent: the process group's name, the comm stream, `AsyncCollectiveTensor`. 205 lines. |

**No communication library appears in C++ at all.** Windows are torch symmetric-memory tensors
(`c10d::symmetric_memory`), and the only thing Python hands down is the process group's name.

### Things that are easy to get wrong

- **Allocate through `empty_strided_p2p` directly, never through a `MemPool`.** `rendezvous` is
  collective, so the allocation sequence has to be identical on every rank. Nothing unrelated is
  served from that entry point, so this group's sequence is every rank's sequence; a shared pool
  would let an unrelated allocation reorder it and one rank would rendezvous while another did not.
- **Handshake state is per window**, by construction — it lives in that allocation's signal pad.
  Two calls may share a window only when a stream orders them, which is why sync and async have
  separate internal windows.
- **The aliasing guard takes the window's capacity**, not the current call's requirement. A result
  sliced on its batch axis starts past what one call needs and would otherwise slip through.
- **Every rank must issue the same sequence of shapes.** A new shape allocates collectively.
  Violating it hangs; nothing raises and nothing times out.
- **No `getenv` in compiled code.** Environment is read in Python if at all.
- **Undocumented assumption the design rests on:** a completed copy-engine write is visible at the
  destination by the time a later kernel's release store announcing it arrives. No vendor doc says
  so. `ce_ordering.py` tests it and arms its own negative control every run. See `docs/design.md`.
- **torch 2.10 is the floor** and the compatible API set is the intersection across 2.10–2.13. Do
  not reach for `is_symm_mem_tensor` (2.12+) or rely on `empty()`'s implicit pool (2.11+).

### Tests

Four workers. `correctness.py` is bit-exact-or-fail on every path including the backward;
`validation.py` covers the rejection paths and that they happen before the first handshake;
`cudagraph.py` checks a captured replay and reports an uncaptured run as having checked NOTHING.
`ce_ordering.py` is the adversarial one and is worth exactly as much as the timing it builds — its predecessor went blind for several commits when an
opening barrier was added. Re-read its docstring after any barrier or ordering change; a run whose
armed control tears nothing is a **blind** run, not a passing one.

### Documentation

`docs/*.md` are English only, lowercase filenames. Style: state the function, the number and the
limit. No rhetorical build-ups, no personification. `docs/benchmark.md` carries the v0.2 measurements; only the
H100 row is still `pending`.

## Known limits

- `world_size ≤ 8` is structural: `BarPeers::p[8]` in `src/barrier.cu`.
- Single node, NVLink only. The constructor refuses a non-NVLink group; over PCIe across a socket
  `torch.distributed` is genuinely faster and the reason is in `docs/design.md`.
- No `torch.compile` tracing: the group is a torchbind object with no registered fake class, so
  Dynamo graph-breaks on it. Backward and `FakeTensor` shape propagation DO work.
- The async form is not differentiable, by construction — see `docs/api.md`.

## Test machines

The **ComputeLab Slurm cluster** (`ssh tailscale-computelab-sc`), through
`/home/sonlin/scratch/workspace/nvidia/scripts/clab.py`, which wraps salloc + pyxis/enroot:

```bash
./clab.py -p b200x4 alloc      # or h200x8 / pro6000x8; b200 is capped at 4h by its partitions
./clab.py -p b200x4 exec -- bash /workspace/<script>.sh
./clab.py -p b200x4 cancel     # NOT optional: an allocation bills until it is released
```

`tools/sync_to_cluster.sh <host> /home/sonlin/scratch/workspace/nvidia/fu-v02` puts the tree where
the container sees it as `/workspace/fu-v02`. Four things bite:

- the login shell is **csh**, so `2>&1` in a remote command gives `Ambiguous output redirect` — wrap
  remote commands in `bash -lc`;
- the ssh host sets `RemoteCommand`, so any non-interactive use needs `-o RemoteCommand=none`
  (`sync_to_cluster.sh` already passes it);
- the container's python is externally managed: `pip install -e .` needs `--break-system-packages`;
- `sync_to_cluster.sh` runs `rsync --delete`, so scratch scripts belong OUTSIDE the synced tree.

**`NCCL_NVLS_ENABLE=0` everywhere** — Fabric Manager cannot bind NVLink SHARP. A benchmark still
has to go through `tools/exclusive.sh`; a 4-GPU allocation on an 8-GPU node has neighbours.
