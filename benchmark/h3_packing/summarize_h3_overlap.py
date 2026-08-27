#!/usr/bin/env python3
"""Summarize repeated MiniMax H3 tiled-overlap reports."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PATHS = (
    "full_serial",
    "tiled_serial",
    "fused_full_serial",
    "fused_full_with_pack",
    "fused_tiled_serial",
    "fused_tiled_overlap",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.reports]
    shape = reports[0]["shape"]
    if any(report["shape"] != shape for report in reports[1:]):
        raise ValueError("overlap reports do not use the same shape and parallel configuration")

    baseline = statistics.median(
        report["measurements"]["full_serial"]["p50_ms"] for report in reports
    )
    rows = []
    for name in PATHS:
        p50 = statistics.median(
            report["measurements"][name]["p50_ms"] for report in reports
        )
        p95 = statistics.median(
            report["measurements"][name]["p95_ms"] for report in reports
        )
        rows.append((name, len(reports), p50, p95, baseline / p50))

    header = "path\truns\tmedian_p50_ms\tmedian_p95_ms\tversus_full_serial"
    lines = [header]
    lines.extend(
        f"{name}\t{runs}\t{p50:.3f}\t{p95:.3f}\t{speedup:.3f}"
        for name, runs, p50, p95, speedup in rows
    )
    wins = sum(
        report["measurements"]["fused_tiled_overlap"]["p50_ms"]
        < min(
            report["measurements"]["full_serial"]["p50_ms"],
            report["measurements"]["fused_tiled_serial"]["p50_ms"],
        )
        for report in reports
    )
    lines.append(f"# overlap_gate={'PASS' if wins >= 2 else 'FAIL'} wins={wins}/{len(reports)}")
    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.output:
        args.output.write_text(output)


if __name__ == "__main__":
    main()
