"""``fast-ulysses doctor``: can this machine run the operator?

What decides whether a group can be formed is whether every pair of devices is P2P-mappable, since
the transport writes peer windows with plain ``cudaMemcpy*Async``. This reports what can be
established WITHOUT a process group -- the build, the devices, the pairwise P2P matrix -- so a green
matrix is a necessary condition, not a substitute for the authoritative ``nvshmem_ptr`` check
UlyssesGroup's constructor makes.
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
    for key, value in sorted(pkg._C.build_info().items()):
        print(f"  {key}: {value}")
    from fast_ulysses._diagnose import build_meta

    meta = build_meta()
    print("build metadata:" if meta else "build metadata: none (_build_meta.py not generated)")
    for key, value in sorted(meta.items()):
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

    # Every ORDERED pair: asymmetry is possible, so print the full matrix, not the upper triangle.
    print("p2p (cudaDeviceCanAccessPeer), row=from col=to:")
    print("     " + " ".join(f"{j:>3}" for j in range(n)))
    unreachable = []
    for i in range(n):
        row = []
        for j in range(n):
            ok = i == j or torch.cuda.can_device_access_peer(i, j)
            row.append(" . " if i == j else (" y " if ok else " N "))
            if not ok:
                unreachable.append((i, j))
        print(f"  {i:>2} " + " ".join(row))

    if unreachable:
        print(
            f"\nNOT P2P-mappable: {len(unreachable)} pair(s), e.g. {unreachable[:4]}.\n"
            "A UlyssesGroup spanning any of those pairs cannot be formed -- nvshmem_ptr returns\n"
            "NULL and the constructor refuses. Groups within a reachable subset still work."
        )
    else:
        print(f"\nall {n} devices are mutually P2P-mappable")

    from fast_ulysses.fallback import numa_nodes

    nodes = numa_nodes(range(n))
    if len(set(nodes.values())) > 1:
        by_node: dict[int, list[int]] = {}
        for dev, node in sorted(nodes.items()):
            by_node.setdefault(node, []).append(dev)
        layout = ", ".join(f"node {k}: {v}" for k, v in sorted(by_node.items()))
        print(
            f"\nNUMA: these devices span more than one CPU socket ({layout}).\n"
            "A group confined to one node keeps the full advantage. A group spanning nodes stays\n"
            "CORRECT but is slower than torch.distributed, which routes around the socket boundary\n"
            "over a NIC or through host memory while this transport always writes peer memory\n"
            "directly. See docs/BENCHMARK.md, 'Two-socket PCIe'."
        )

    return 1 if unreachable else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fast-ulysses")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="report the build, the devices, and pairwise P2P reachability")
    args = parser.parse_args(argv)
    if args.command == "doctor":
        return _doctor()
    parser.error(f"unknown command {args.command}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
