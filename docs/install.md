# Installation

## Requirements

- **PyTorch 2.10+**, Linux x86_64, CPython 3.10–3.13
- **CUDA 12.8+ or 13**
- sm80 / sm90 / sm100 / sm120, NVLink-joined
- For a source build only: CMake ≥ 3.18 and `nvcc`. `ccache` is used when present.

torch is the only runtime dependency. Windows come from `c10d::symmetric_memory`, and the floor is
`get_signal_pad_size()`, which becomes a free `TORCH_API` function in torch 2.10 — that is what the
handshake's placement in the pad is computed from. `empty_strided_p2p` and `rendezvous` themselves
go back further. CUDA 12.6 and below cannot emit `sm_100` / `sm_120`.

## Install

A release is 36 wheels: 8 (torch minor, CUDA major) rows × 4 CPythons, plus 4 untagged ones for
PyPI. The torch minor must match exactly and the CUDA major must match, because the extension
embeds `c10d::Work` vtables, a pybind11 module and a `TORCH_LIBRARY` registration, none of which
survive a minor bump. Which row you are:

```bash
python -c "import sys,torch; print(torch.__version__, torch.version.cuda, sys.version_info[:2])"
```

| your torch | torch's CUDA | wheel tag |
|---|---|---|
| 2.10.x | 12.x | `torch210cu128` |
| 2.10.x | 13.x | `torch210cu130` |
| 2.11.x | 12.x | `torch211cu128` |
| 2.11.x | 13.x | `torch211cu130` |
| 2.12.x | 12.x | `torch212cu129` |
| 2.12.x | 13.x | `torch212cu130` |
| 2.13.x | 12.x | `torch213cu129` |
| 2.13.x | 13.x | `torch213cu130` |

The tag is the wheel's PEP 440 local version — `fast_ulysses-0.2.0+torch210cu128-cp312-…` — which
is what you write after `==` when you name one. The index picks it for you.

### From the index

```bash
pip install fast-ulysses --index-url https://triple-mu.github.io/fast-ulysses/whl/cu13/
```

Two indexes, `cu12` and `cu13`, picked by the major of `torch.version.cuda`; `cu128`, `cu129` and
`cu130` are aliases holding the same wheels, since that suffix is what you read off the URL your
torch came from. The CUDA major is chosen by URL because no wheel metadata records it, the same
reason `download.pytorch.org/whl/cu128` is laid out that way.

The torch minor is not chosen, it is resolved. An index carries fast-ulysses and nothing else, so
pip cannot install a torch to satisfy its first candidate — `0.2.0+torch213cu130`, the highest
local version there — and backtracks until it reaches the one whose `Requires-Dist: torch==2.11.*`
the installed torch already satisfies.

Install torch first. With no torch installed nothing satisfies any candidate, and the install ends
on `ResolutionImpossible` after listing every row it tried:

```
ERROR: Cannot install fast-ulysses==0.2.0+torch210cu130, ... because these package versions have
conflicting dependencies.
The conflict is caused by:
    fast-ulysses 0.2.0+torch213cu130 depends on torch==2.13.*
    ...
```

The rows do not conflict with each other; each of them conflicts with the torch that is not there.
That is the intended failure — nothing here resolves torch for you — and `pip install torch` first
is the whole of the fix.

Do not add `--extra-index-url https://pypi.org/simple`. It gives pip a torch to install, so the
first candidate resolves after all, by moving your torch to 2.13. Install whatever else needs PyPI
in its own command — or use a per-row index, one per tag in the table above:

```bash
pip install fast-ulysses --index-url https://triple-mu.github.io/fast-ulysses/whl/torch211cu128/
```

That one holds a single row, four wheels, one per CPython, so there is nothing for pip to choose
between and no other index can outbid it. It is the form to use when something else in the same
command must come from PyPI. It follows its row across releases: when a row's CUDA *minor* moves —
2.12 went `cu128` → `cu129` — the index named after the old one keeps receiving that row's wheels,
so a URL written down once keeps resolving. The CUDA major never moves that way; it is what the
name promises.

The indexes cover 0.2.0 and later. For 0.1.0, use `--find-links` on that release.

### From the release, by hand or by `wheel-url`

```bash
fast-ulysses wheel-url
```

reports the row for the torch, CUDA, python and machine running now, and prints the command for
it, with the version written out in full:

```bash
pip install "fast-ulysses==0.2.0+torch210cu128" \
  --find-links https://github.com/triple-mu/fast-ulysses/releases/expanded_assets/v0.2.0
```

`--find-links` on `expanded_assets` rather than a constructed asset URL: the file name carries
whatever platform tag auditwheel compressed to — v0.1.0 shipped
`manylinux_2_24_x86_64.manylinux_2_28_x86_64` — so writing the name out is a guess. That endpoint
is the fragment the release page loads its asset list from, and it lists all 36. The page itself,
<https://github.com/triple-mu/fast-ulysses/releases/tag/v0.2.0>, carries no `.whl` link and answers
`Could not find a version that satisfies the requirement` for every version. GitHub documents
neither endpoint. `SHA256SUMS` on the release covers every file.

`wheel-url` needs an extension that loads: it is a subcommand of the `fast-ulysses` script, which
imports the package, which imports `_C`. When `_C` is what is broken, the import error itself
already names both sides — see [When the import fails](#when-the-import-fails).

### From PyPI — torch 2.13.x and CUDA 13 only

```bash
pip install fast-ulysses
```

PyPI rejects PEP 440 local versions, so exactly one row goes there: `torch213cu130`, with the tag
stripped. Its metadata is `Requires-Dist: torch==2.13.*`. **On any other torch this command does
not fail.** pip satisfies that pin by upgrading torch to 2.13 from PyPI — a different torch and a
different CUDA build than the one you installed — and reports it as an ordinary dependency
resolution. Use the index, or the command `wheel-url` prints, for every other row.

### The version must be written out in full

Under PEP 440 a local label is compared segment by segment, and a specifier without one ignores
the label entirely, so `==0.2.0` matches all nine artifacts of 0.2.0 and the highest wins:

```
0.2.0 < +torch210cu128 < +torch210cu130 < +torch211cu128 < +torch211cu130
      < +torch212cu129 < +torch212cu130 < +torch213cu129 < +torch213cu130
```

Over any file set holding more than one row — `--find-links`, `--extra-index-url`, a directory of
downloaded wheels — an unqualified requirement therefore takes `torch213cu130` whatever torch is
installed, silently:

```bash
# ./wheelhouse holding all 36
pip install fast-ulysses --find-links ./wheelhouse                       # -> +torch213cu130
pip install "fast-ulysses==0.2.0" --find-links ./wheelhouse              # -> +torch213cu130
pip install "fast-ulysses==0.2.0+torch210cu128" --find-links ./wheelhouse   # -> only that one
```

So with `--find-links` or `--extra-index-url`, write the label out; `--index-url` at `whl/cu12` or
`whl/cu13` needs no label because the torch pin resolves the row there, and `whl/<tag>` needs none
because there is only the one row in it — that is the one that also holds up beside PyPI. The
written-out label is also the form that does not depend on index priority: pip unions every index
and takes the highest version, `uv pip install` defaults to `first-index` and stops at the first
index carrying the project, and an exact version survives both.

One more thing pip does not do here: replace an installation it already has. `pip install
fast-ulysses` with any version present prints `Requirement already satisfied` and leaves the wrong
row in place — `--upgrade` does not help either, since the row it would upgrade to is the highest
one. Uninstall first, or name the version in full.

### For a torch the table does not list

```bash
pip install fast-ulysses --no-binary fast-ulysses --no-build-isolation
```

The sdist compiles against the torch already installed. `--no-build-isolation` is required for the
reason in the next section; without it the build stops with that instruction rather than a
traceback.

## Build from source

```bash
pip install -e . --no-build-isolation                              # all four architectures
FAST_ULYSSES_CUDA_ARCH=90 pip install -e . --no-build-isolation    # one, much faster
```

`--no-build-isolation` is required: CMake locates libtorch through the installed torch, so torch
has to be importable at build time, and it must be the one you intend to run against.

| Variable | Meaning |
| --- | --- |
| `FAST_ULYSSES_CUDA_ARCH` | Target compute capabilities, `;`-separated. Default `80;90;100;120`. |
| `CUDACXX` | CUDA compiler; defaults to `/usr/local/cuda/bin/nvcc`. |
| `FAST_ULYSSES_BUILD_DIR` | CMake build tree. Default `./build`, kept between builds so rebuilds are incremental. |
| `FAST_ULYSSES_CMAKE_ARGS` | Extra flags passed through to CMake. |
| `CMAKE_BUILD_PARALLEL_LEVEL` | Overrides the job count, which is otherwise bounded by available memory rather than core count. |

## When the import fails

`import fast_ulysses` catches the loader error and reports what the extension was built against
next to what is installed. The `fast-ulysses` script imports the same package, so it raises the
same report instead of running — `doctor`'s devices and NVLink matrix need a load that works. The
two common causes:

- **`undefined symbol: _ZN3c10...`** — the wheel's torch minor is not the installed torch's. Either
  the wrong row was installed, or torch moved after it: the plain `pip install fast-ulysses` moves
  torch to 2.13 itself, and reinstalling the torch you wanted afterwards leaves exactly this.
- **`libcudart.so.12: cannot open shared object file`** — a CUDA-12 wheel in a CUDA-13 environment
  or the reverse. No `LD_LIBRARY_PATH` fixes this.

Both are the same repair, and the import error has already printed the two sides of it, so it needs
neither the table nor a working extension:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda)"   # the torch you chose?
pip uninstall -y fast-ulysses
pip install fast-ulysses --index-url https://triple-mu.github.io/fast-ulysses/whl/cu13/
```

Check torch first, in that order. If it is not the torch you installed, an earlier
`pip install fast-ulysses` moved it (see PyPI, above); reinstall the torch you want before this, or
you pin the row you were moved to. The uninstall is not optional — without it pip reports
`Requirement already satisfied` and the broken wheel stays. Naming the version in full,
`pip install "fast-ulysses==0.2.0+torch210cu128" --find-links …`, replaces it in one command.

## When the group will not build

- **`cuda:i and cuda:j are not joined by NVLink`** — the constructor's check. Use
  `torch.distributed` on that machine; over PCIe, and especially across a CPU socket, it is faster
  than this transport. `require_nvlink=False` builds the group anyway, for measuring that case.
- **`NVML could not report the link topology`** (from `doctor`) — the check cannot answer, so it
  does not refuse. Nothing is claimed either way.

## Other problems

- **NCCL dies at init with `unhandled cuda error` / "Failed to bind NVLink SHARP (NVLS) Multicast
  memory"** — a Fabric Manager or NVSwitch configuration problem on that machine. Run with
  `NCCL_NVLS_ENABLE=0`. This affects the `torch.distributed` bootstrap, not this operator.
- **`CUDASymmetricMemory.cu: ... init_multicast_for_block` warnings, then "Gracefully skipping
  multicast initialization"** — the same machines, the same cause, printed by torch rather than
  NCCL. Harmless here: the multicast path is NVLS, and this transport writes peer addresses
  directly and never uses it.
- **CMake `CMakeCache.txt directory ... is different than ...`** — the persistent `build/` was
  configured from another path (the repo moved). `rm -rf build` and rebuild.
