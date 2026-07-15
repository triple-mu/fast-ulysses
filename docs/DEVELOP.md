# Development

## Setup

```bash
NVSHMEM_HOME=<nvshmem install root> \
FAST_ULYSSES_CUDA_ARCH=<your arch, e.g. 90> \
pip install -e . --no-build-isolation

pip install -e ".[dev]" --no-build-isolation   # adds pre-commit, pytest, ruff
pre-commit install                             # run hooks on every commit
```

The build reuses the persistent `build/` directory, so edit-rebuild cycles only recompile changed
translation units (plus `ccache` when installed).

## Linting & formatting

pre-commit is the single lint entry point:

```bash
pre-commit run --all-files
```

- Python: **ruff** (check + format, line length 100, py310+).
- C++/CUDA (`fast_ulysses/csrc/`): **clang-format**, pinned to **v15.0.7** in
  `.pre-commit-config.yaml` — the version the current code is formatted with. Keep the pin and the
  local binary in sync if you format manually.

## Tests

```bash
pytest                      # everything runnable on this machine
pytest -m "not multigpu"    # only the single-GPU op tests (fast)
pytest -m multigpu          # only the torchrun-wrapped multi-GPU suites
```

- `tests/test_ops.py` — single-GPU correctness of `rms_norm` / `rope` / `norm_rope` against an
  fp32 torch reference (skips when the extension is not built or CUDA is unavailable).
- `tests/test_multigpu.py` — launches each worker under `tests/distributed/` as a
  `torch.distributed.run` subprocess; skips below 2 GPUs. `FAST_ULYSSES_TEST_NPROC` overrides the
  process count (e.g. `=3` to exercise odd world sizes).
- Workers stay directly runnable for debugging (full output, single suite):

```bash
torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py
torchrun --nproc_per_node=8 tests/distributed/a2a_async.py
torchrun --nproc_per_node=8 tests/distributed/a2a_qk.py
```

All a2a checks are bit-exact comparisons against `torch.distributed` references (pure data
movement); the fused qk paths compare against an fp32 reference at ~1 dtype ULP.

## Layout

```
fast_ulysses/          Python package (comm.py: UlyssesGroup; ops.py: standalone ops)
fast_ulysses/csrc/     C++/CUDA sources (bindings.cpp registers the torch library)
tests/                 pytest suites; tests/distributed/ holds the torchrun workers
benchmark/             throughput / fusion benchmarks and a minimal nsys/ncu driver
docs/                  this documentation
```
