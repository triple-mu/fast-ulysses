"""``fast-ulysses doctor``: can this machine run the operator?

What decides it is whether every pair of devices is joined by NVLink, since the transport writes
peer memory directly. This reports what can be established without a process group -- the build,
the devices, the NVLink matrix -- so a green matrix is a necessary condition, not a substitute for
the check UlyssesGroup's constructor makes over the actual group.
"""

from __future__ import annotations

import argparse
import sys


def _load():
    """(module, error). Import is deferred so `doctor` can report a load failure as its output."""
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
    built = {a.strip() for a in pkg._C.build_info()["cuda_arch_list"].split(",") if a.strip()}
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
    print(f"\nall {n} devices are mutually NVLink-joined")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fast-ulysses")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report the build, the devices, and the NVLink matrix")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
