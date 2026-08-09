# Development

## Setup

```bash
FAST_ULYSSES_CUDA_ARCH=<your arch, e.g. 90> pip install -e ".[dev]" --no-build-isolation
pre-commit install
```

The build reuses the persistent `build/` directory, so edit-rebuild cycles only recompile changed
translation units (plus `ccache` when installed).

## Layout

```
include/fast_ulysses/   public headers, Doxygen comments
src/                    a2a_plan.cc (pure host addressing), transfer.cu, barrier.cu,
                        group.cc (windows, plans, staging), nvlink.cc, bindings.cc
python/fast_ulysses/    group.py (the API surface), cli.py, _diagnose.py
test/                   test_plan.py (host-only); distributed/ holds the torchrun workers
benchmark/bench_a2a.py  stages, GEMM overlap, padding cost
tools/                  GPU-exclusivity wrapper, wheel build and gate, release preflight
docs/
```

`src/a2a_plan.cc` has no CUDA, no torch and no communication library in it. Keep it that way: it is
what makes the layout contract testable without a GPU.

## Linting

pre-commit is the single entry point:

```bash
pre-commit run --all-files
```

Python is **ruff** (check + format, line length 100, py310+). C++/CUDA under `include/` and `src/`
is **clang-format**, pinned to **v15.0.7** in `.pre-commit-config.yaml` — the version the code is
formatted with. Keep the pin and any local binary in sync; another version reformats the whole tree.

## Tests

```bash
pytest                      # everything runnable here
pytest -m "not multigpu"    # host-only, no GPU needed
pytest -m multigpu          # the torchrun-wrapped workers
```

`test/test_plan.py` replays the addressing (`src/a2a_plan.cc`) over numpy buffers against an
`all_to_all_single` + permute reference. It needs no GPU and no process group, only the built
extension — which is why it is the one correctness check CI can run.

`test/test_distributed.py` launches each worker under `test/distributed/` as a
`torch.distributed.run` subprocess and skips below 2 GPUs. Each worker runs at `min(ngpu, 8)`
processes and, with ≥ 3 GPUs, also at 3 — an odd world size exercises the non-power-of-two peer
sweep. `FAST_ULYSSES_TEST_NPROC` overrides the list.

Workers stay directly runnable, which is the debugging path:

```bash
torchrun --nproc_per_node=8 test/distributed/correctness.py
```

| worker | what it asserts |
|---|---|
| `correctness` | bit-exact against `torch.distributed`: both modes, both dtypes, even and uneven shards, the three things `out=` can be, async, round trip, and 20 rounds on one window |
| `validation` | that every documented rejection raises, with the right message, on every rank, and before the call's first handshake — including the aliasing guard |
| `ce_ordering` | that a copy-engine payload is visible when the flag announcing it arrives — **and** that the test can still fail, by arming the fault itself on every run |

`ce_ordering` is the one adversarial worker, and it is worth exactly as much as the timing it
builds. Its predecessor went blind for several commits when an opening barrier was added and the
worker still skewed arrival instead of the transfer, so re-read its docstring after any change to
the barrier or the ordering. It reports both phases: a run where the armed control tears nothing is
a blind run, not a passing one.

## Releasing

CI has no GPU runner, so nothing under `test/distributed/` ever runs there. What CI proves is that
each configuration compiles for four architectures, links against exactly the expected libraries
with a relocatable RUNPATH, loads under the target torch, and passes `test_plan.py`.

```bash
tools/build_wheels.sh          # one (torch, CUDA) row, inside a manylinux builder
tools/check_wheel.py <whl>     # the ELF/metadata gate; also runs inside build_wheels.sh
tools/preflight_gpu.sh <whl>   # MANDATORY before a tag: the built wheel on a real multi-GPU box
```

`preflight_gpu.sh` prints a block for the release notes. Run it for at least the newest torch row
and one CUDA-12 row; the oldest rows ship on compile-and-load evidence only, and the release notes
should say so.

Bump `VERSION` before tagging: the release job refuses a tag that disagrees with it, because every
artifact takes its version from that file and PyPI never lets a version be replaced.

Benchmarks must run under `tools/exclusive.sh`, which refuses to start until the requested GPUs are
free and prints `EXCLUSIVE` or `CONTENDED`. A `CONTENDED` number is not a number.
