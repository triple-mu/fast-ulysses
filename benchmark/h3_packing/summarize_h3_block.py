#!/usr/bin/env python3
"""Summarize repeated h3-block JSON reports and evaluate the packed PCIe gate."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

BLOCK_PATHS = (
    "nccl_block",
    "pitched_owned_block",
    "pitched_zero_block",
    "packed_owned_block",
    "packed_zero_block",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    reports = [json.loads(path.read_text()) for path in args.reports]
    shapes = [report["shape"] for report in reports]
    if any(shape != shapes[0] for shape in shapes[1:]):
        raise ValueError("h3-block reports do not use the same shape and parallel configuration")

    rows = []
    baseline_runs = [report["measurements"]["nccl_block"]["p50_ms"] for report in reports]
    baseline_median = statistics.median(baseline_runs)
    for name in BLOCK_PATHS:
        p50_runs = [report["measurements"][name]["p50_ms"] for report in reports]
        p95_runs = [report["measurements"][name]["p95_ms"] for report in reports]
        median_p50 = statistics.median(p50_runs)
        rows.append(
            {
                "path": name,
                "runs": len(reports),
                "median_p50_ms": median_p50,
                "min_p50_ms": min(p50_runs),
                "max_p50_ms": max(p50_runs),
                "median_p95_ms": statistics.median(p95_runs),
                "versus_nccl": baseline_median / median_p50,
                "denoise_communication_s": (
                    median_p50 * reports[0]["blocks"] * reports[0]["steps"] / 1000
                ),
            }
        )

    packed_wins = sum(
        baseline / packed >= 1.10
        for baseline, packed in zip(
            baseline_runs,
            [report["measurements"]["packed_owned_block"]["p50_ms"] for report in reports],
        )
    )
    packed_tail_ok = all(
        report["measurements"]["packed_owned_block"]["p95_ms"]
        <= 1.10 * report["measurements"]["packed_owned_block"]["p50_ms"]
        for report in reports
    )
    gate_pass = packed_wins >= min(4, len(reports)) and packed_tail_ok

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        output.write(
            "path\truns\tmedian_p50_ms\tmin_p50_ms\tmax_p50_ms\tmedian_p95_ms"
            "\tversus_nccl\tdenoise_communication_s\n"
        )
        for row in rows:
            output.write(
                f"{row['path']}\t{row['runs']}\t{row['median_p50_ms']:.3f}"
                f"\t{row['min_p50_ms']:.3f}\t{row['max_p50_ms']:.3f}"
                f"\t{row['median_p95_ms']:.3f}\t{row['versus_nccl']:.3f}"
                f"\t{row['denoise_communication_s']:.3f}\n"
            )
        output.write(
            f"# packed_gate={'PASS' if gate_pass else 'FAIL'} wins={packed_wins}/{len(reports)} "
            f"p95_within_10pct={packed_tail_ok}\n"
        )

    print(args.output.read_text(), end="")
    if not gate_pass:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
