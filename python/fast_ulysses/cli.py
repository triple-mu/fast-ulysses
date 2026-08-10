"""The ``fast-ulysses`` command: ``doctor`` and ``wheel-url``.

``doctor`` answers whether this machine can run the operator. What decides it is whether every
pair of devices is joined by NVLink, since the transport writes peer memory directly. This reports
what can be established without a process group -- the build, the devices, the NVLink matrix -- so
a green matrix is a necessary condition, not a substitute for the check UlyssesGroup's constructor
makes over the actual group.

``wheel-url`` answers which published wheel matches the torch, CUDA, python and architecture
running right now, and prints the version written out in full, because PEP 440 orders local
versions and a bare requirement therefore resolves to the highest tag rather than to this torch.
docs/install.md has that argument in full.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import sys
from pathlib import Path


def _load():
    """(module, error). NOT the path a failed import takes when this runs as the console script:
    that imports the package first, so ``_diagnose.explain``'s report is raised before ``main``
    runs -- docs/install.md says so under "When the import fails". What is left for this to catch
    is cli.py run as a file, from a tree where nothing is installed."""
    try:
        import fast_ulysses

        return fast_ulysses, None
    except Exception as exc:  # noqa: BLE001 -- the failure IS the report
        return None, exc


def _doctor() -> int:
    pkg, err = _load()
    if err is not None:
        # Already rendered: fast_ulysses/__init__ routes a load failure through _diagnose.explain
        # before re-raising, and importing _diagnose here would re-run the __init__ that failed.
        print(f"extension: FAILED TO LOAD\n{type(err).__name__}: {err}")
        return 1
    print(f"extension: {pkg._C.__file__}")
    from fast_ulysses._diagnose import build_meta

    for key, value in sorted(build_meta().items()):
        print(f"  {key.lower()}: {value}")

    import torch

    if not torch.cuda.is_available():
        print("devices: none visible to CUDA")
        return 1

    n = torch.cuda.device_count()
    # Leading digits only: the arch list carries whatever was asked for, and 90a and 120-real name
    # the same GPUs as 90 and 120. CMakeLists normalises with the same match.
    built = {
        m.group()
        for a in pkg._C.build_info()["cuda_arch_list"].split(",")
        if (m := re.match(r"[0-9]+", a.strip()))
    }
    print(f"devices: {n}")
    for i in range(n):
        p = torch.cuda.get_device_properties(i)
        arch = f"{p.major}{p.minor}"
        mark = "" if arch in built else "  <-- NOT IN THIS BUILD"
        print(f"  [{i}] {p.name}  sm_{arch}  {p.total_memory >> 30} GiB{mark}")

    matrix = pkg._C.nvlink_matrix(list(range(n)))
    if matrix is None:
        print("\nNVLink: NVML could not report the link topology on this machine.")
        print("UlyssesGroup will build anyway; nothing here can say whether it should.")
        return 0

    print("\nnvlink, row=from col=to:")
    print("     " + " ".join(f"{j:>3}" for j in range(n)))
    missing = []
    for i in range(n):
        row = []
        for j in range(n):
            ok = matrix[(i, j)]
            row.append(" . " if i == j else (" y " if ok else " N "))
            if i != j and not ok:
                missing.append((i, j))
        print(f"  {i:>2} " + " ".join(row))

    if missing:
        print(
            f"\nNOT NVLink-joined: {len(missing)} pair(s), e.g. {missing[:4]}.\n"
            "A group spanning any of those is refused at construction. Use torch.distributed\n"
            "there: it routes around the link, and this transport does not."
        )
        return 1
    if n < 2:
        # The pair loop above is empty here, and "all 1 devices are joined" is the one sentence a
        # reader takes away from the subcommand.
        print("\none visible device: no pair to check, and nothing above world_size 1 to run")
        return 0
    print(f"\nall {n} devices are mutually NVLink-joined")
    return 0


_MATRIX = Path(__file__).with_name("wheel_matrix.json")

# What to do when no wheel matches. The sdist builds against the torch already installed, which
# is the whole reason it cannot run under pip's build isolation. --no-binary is not optional:
# without it pip takes the untagged PyPI wheel -- which matches any x86_64 cp310-cp313 and pins
# torch==2.13.* -- and moves this torch to 2.13 to satisfy it. That is the failure this whole
# subcommand exists to prevent, and every message below ends on this line.
_FROM_SOURCE = (
    "  pip install fast-ulysses --no-binary fast-ulysses --no-build-isolation   # needs nvcc"
)


def _abi(version: str) -> tuple[int, int]:
    """(major, minor) of a torch version. Ordered as integers, so 2.9 sorts below 2.10."""
    major, minor = version.split(".")[:2]
    return int(major), int(minor)


def _tag(row: dict) -> str:
    """A row's local version, e.g. torch210cu128 -- check_wheel.py parses the inverse form."""
    major, minor = _abi(row["torch"])
    return f"torch{major}{minor}cu{row['cuda'].replace('.', '')}"


def _match(matrix: dict, live: dict[str, str], machine: str) -> dict | str:
    """The published row for this environment, or the text naming the axis that has no wheel.

    The axes are checked in the order they are decided in: architecture and python are the wheel's
    own tags, torch minor and CUDA major are what the .so is pinned to. Matching on the CUDA MAJOR
    and not the minor is deliberate -- the extension links libcudart.so.<major> and nothing else
    from the toolkit, and check_wheel.py asserts exactly that much.
    """
    rows = matrix["rows"]
    if machine != matrix["machine"]:
        return (
            f"no wheel for {machine}: every published wheel is {matrix['platform_tag']}.\n"
            f"options:\n{_FROM_SOURCE}"
        )
    if live["python"] not in matrix["pythons"]:
        return (
            f"no wheel for {live['python']}: the published pythons are "
            f"{', '.join(matrix['pythons'])}.\n"
            f"options:\n  run one of those interpreters, or\n{_FROM_SOURCE}"
        )
    if live["torch"] == "unimportable":
        return (
            "torch does not import here, and its minor and CUDA major are what pick the wheel.\n"
            "Install the torch you intend to run, then run this again."
        )

    abis = sorted({".".join(r["torch"].split(".")[:2]) for r in rows}, key=_abi)
    if live["torch_abi"] not in abis:
        near = abis[0] if _abi(live["torch_abi"]) < _abi(abis[0]) else abis[-1]
        return (
            f"no wheel for torch {live['torch_abi']}: published for torch {', '.join(abis)}.\n"
            "The .so embeds c10d::Work vtables, a pybind11 module and a TORCH_LIBRARY\n"
            "registration, none of which survive a torch minor bump.\n"
            f"options:\n"
            f'  pip install "torch=={near}.*"   # the nearest published torch, then run this '
            f"again\n{_FROM_SOURCE}"
        )

    same = [r for r in rows if ".".join(r["torch"].split(".")[:2]) == live["torch_abi"]]
    majors = sorted({r["cuda"].split(".")[0] for r in same})
    if live["cuda"] in ("None", "unknown"):
        return (
            "no wheel: torch.version.cuda is None, so this is a CPU (or ROCm) build of\n"
            "torch, and every wheel links libcudart.\n"
            f"Install a CUDA {' or '.join(majors)} build of torch {live['torch_abi']}, then run "
            "this again."
        )
    if live["cuda"].split(".")[0] not in majors:
        return (
            f"no wheel for CUDA {live['cuda']}: torch {live['torch_abi']} is published for CUDA "
            f"{' and '.join(majors)}.\n"
            "A CUDA major mismatch fails at import on libcudart, and no LD_LIBRARY_PATH fixes "
            f"it.\noptions:\n{_FROM_SOURCE}"
        )
    return next(r for r in same if r["cuda"].split(".")[0] == live["cuda"].split(".")[0])


def _wheel_url() -> int:
    # Relative, with the flat name as the fallback: `import fast_ulysses` imports torch, and this
    # subcommand has to answer for a torch that is not installed or will not import. Running
    # cli.py as a file is the only way to reach here in that state, and there is no package to be
    # relative to then.
    try:
        from ._diagnose import _live, build_meta
    except ImportError:
        from _diagnose import _live, build_meta

    matrix, live, machine = json.loads(_MATRIX.read_text()), _live(), platform.machine()
    print(f"running: torch {live['torch']}, cuda {live['cuda']}, {live['python']}, {machine}")

    row = _match(matrix, live, machine)
    if isinstance(row, str):
        print(row)
        return 1

    # The base version of THIS package: a release publishes one wheel per row, so the row picked
    # above only names a file within the release this package came from.
    version = str(build_meta().get("VERSION", "")).partition("+")[0]
    if not version:
        print("_build_meta.py is missing, so the release this package came from cannot be named.")
        return 1

    torch_abi = ".".join(row["torch"].split(".")[:2])
    print(
        f"\nrow: torch {torch_abi}, CUDA {row['cuda']}, {live['python']}, {matrix['platform_tag']}"
    )
    if row.get("pypi"):
        print(f"\n  pip install fast-ulysses=={version}\n")
        print(
            "This is the row PyPI carries, and the only one published without a local\n"
            "version: its torch is what a plain `pip install torch` resolves to."
        )
        return 0

    # --find-links, not a constructed asset URL: the filename carries whatever compressed platform
    # tag auditwheel emitted, which nothing here can predict, so naming the file would be a guess.
    # The release PAGE is useless to pip -- its assets are lazy-loaded and its HTML holds no .whl
    # link -- and expanded_assets is the fragment it loads them from, which lists all of them.
    print(
        f'\n  pip install "fast-ulysses=={version}+{_tag(row)}" \\\n'
        f"    --find-links {matrix['find_links'].format(version=version)}\n"
    )
    pypi_row = next(r for r in matrix["rows"] if r.get("pypi"))
    pypi_abi = ".".join(pypi_row["torch"].split(".")[:2])
    print(
        f"The version has to be written out in full. PEP 440 orders local versions, and\n"
        f"{version}+{_tag(pypi_row)} is above every other tag of this release, so both\n"
        f"`pip install fast-ulysses` and `fast-ulysses=={version}` resolve to the torch\n"
        f"{pypi_abi} wheel -- which declares torch=={pypi_abi}.*, so pip would then move this\n"
        f"torch to {pypi_abi} as well. Nothing warns."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fast-ulysses")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report the build, the devices, and the NVLink matrix")
    sub.add_parser("wheel-url", help="the published wheel for this torch, CUDA, python and arch")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    if args.command == "wheel-url":
        return _wheel_url()
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
