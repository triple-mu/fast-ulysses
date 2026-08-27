#!/usr/bin/env python3
"""Summarize the MiniMax H3 U8 DLO AllGather A/B experiment."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


DENOISE_RE = re.compile(r"denoise_step_latency_ms\s*\|\s*([\d,.]+)")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, int((len(ordered) - 1) * fraction + 0.999999))
    return ordered[index]


def read_seconds(path: Path) -> list[float]:
    return [float(p.read_text().strip()) for p in sorted(path.glob("run-*.seconds"))]


def read_denoise_ms(path: Path, measured_runs: int) -> list[float]:
    text = (path / "server.log").read_text(errors="replace")
    values = [float(match.replace(",", "")) for match in DENOISE_RE.findall(text)]
    return values[-measured_runs:]


def read_peak_gpu_mib(path: Path) -> float:
    sample_file = path / "gpu-samples.csv"
    if not sample_file.exists():
        return float("nan")
    peak = 0.0
    with sample_file.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or row[1].strip().lower() == "index":
                continue
            match = re.search(r"([\d.]+)", row[2])
            if match:
                peak = max(peak, float(match.group(1)))
    return peak


def read_peak_cpu_rss_gib(path: Path) -> float:
    sample_file = path / "process-rss-samples.csv"
    if not sample_file.exists():
        return float("nan")
    peak_kib = 0.0
    with sample_file.open(newline="") as handle:
        for row in csv.reader(handle):
            if len(row) != 2 or row[0] == "timestamp":
                continue
            peak_kib = max(peak_kib, float(row[1]))
    return peak_kib / (1024 * 1024)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--measured-runs", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for mode in ("dlo-use-allgather", "dlo-no-allgather"):
        path = args.result_root / "e2e" / mode
        status_file = path / "status.txt"
        status = status_file.read_text().strip() if status_file.exists() else "MISSING"
        seconds = read_seconds(path) if path.exists() else []
        denoise = read_denoise_ms(path, args.measured_runs) if (path / "server.log").exists() else []
        startup_file = path / "startup.seconds"
        rows.append(
            {
                "mode": mode,
                "status": status,
                "runs": len(seconds),
                "mean_e2e_s": statistics.fmean(seconds) if seconds else float("nan"),
                "median_e2e_s": statistics.median(seconds) if seconds else float("nan"),
                "p95_e2e_s": percentile(seconds, 0.95),
                "mean_denoise_step_ms": statistics.fmean(denoise) if denoise else float("nan"),
                "peak_gpu_memory_mib": read_peak_gpu_mib(path),
                "peak_process_rss_gib": read_peak_cpu_rss_gib(path),
                "startup_s": float(startup_file.read_text()) if startup_file.exists() else float("nan"),
            }
        )

    no_ag = next(row["mean_e2e_s"] for row in rows if row["mode"] == "dlo-no-allgather")
    for row in rows:
        mean = row["mean_e2e_s"]
        row["speedup_vs_no_allgather"] = no_ag / mean if mean == mean and no_ag == no_ag else float("nan")

    columns = list(rows[0])
    lines = ["\t".join(columns)]
    for row in rows:
        fields = []
        for column in columns:
            value = row[column]
            fields.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        lines.append("\t".join(fields))
    output = "\n".join(lines) + "\n"
    print(output, end="")
    if args.output:
        args.output.write_text(output)


if __name__ == "__main__":
    main()
