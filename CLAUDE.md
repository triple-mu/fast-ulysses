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
torchrun --nproc_per_node=8 test/distributed/ce_ordering.py

fast-ulysses doctor               # build facts, devices, NVLink matrix

# Benchmarks MUST go through exclusive.sh; a CONTENDED number is not a number.
./tools/exclusive.sh 0,1,2,3 -- torchrun --nproc_per_node=4 benchmark/bench_a2a.py [--overlap|--padding]
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
| memory + API | `python/fast_ulysses/group.py` | Windows, the handshake state, the async path. |
| topology | `python/fast_ulysses/nvlink.py` | NVML link-type probing. |

**Every address the C++ side touches arrives as an argument.** Windows are torch symmetric-memory
tensors owned by Python; no communication library appears in C++ at all.

### Things that are easy to get wrong

- **The MemPool must be the group's own**, not torch's implicit one. `rendezvous` is collective, so
  the allocation sequence has to be identical on every rank; an unrelated allocation from a shared
  pool reorders it and one rank rendezvous-es while another does not. `use_on_oom=False,
  no_split=True` match torch's own symmetric pool and are load-bearing (no_split keeps two windows
  from sharing a signal pad).
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

Two workers, both bit-exact-or-fail. `ce_ordering.py` is the only adversarial one and is worth
exactly as much as the timing it builds — its predecessor went blind for several commits when an
opening barrier was added. Re-read its docstring after any barrier or ordering change; a run whose
armed control tears nothing is a **blind** run, not a passing one.

### Documentation

`docs/*.md` are English only, lowercase filenames. Style: state the function, the number and the
limit. No rhetorical build-ups, no personification. `docs/benchmark.md` is deliberately a skeleton
with `pending` in the number cells — the v0.2 measurements are taken in one pass when the maintainer
schedules the machines.

## Known limits

- `world_size ≤ 8` is structural: `BarPeers::p[8]` in `src/barrier.cu`.
- Single node, NVLink only. The constructor refuses a non-NVLink group; over PCIe across a socket
  `torch.distributed` is genuinely faster and the reason is in `docs/design.md`.
- No backward, no meta impl — so no autograd and no `torch.compile` tracing.

## Test machines

`hyper00` / `hyper01` (8×H200) and `novita-h100` (8×H100), all in containers named
`sglang-diffusion-triplemu*`. Sync with rsync to `/tmp/fu-v02` then `docker cp` into
`/workspace/fu-v02` (the container has its own `/tmp`). **All of them need `NCCL_NVLS_ENABLE=0`** —
Fabric Manager cannot bind NVLink SHARP. Check `nvidia-smi` before benchmarking: these boxes are
shared and occupancy changes within minutes.
