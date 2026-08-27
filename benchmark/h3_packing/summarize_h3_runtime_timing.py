#!/usr/bin/env python3
"""Aggregate env-gated diffusion runtime timing from vLLM-Omni worker logs."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

PREFIX = "[DiffusionRuntimeTiming] "
METRICS = (
    "dlo.cpu_pack",
    "dlo.staging_reuse_wait",
    "dlo.mmap_submit",
    "dlo.h2d",
    "dlo.allgather",
    "dlo.prefetch_wait",
    "dlo.resident_wait",
    "dit.forward",
    "dit.streaming_block_compute",
    "ulysses.mode0.pack",
    "ulysses.mode0.a2a",
    "ulysses.mode0.unpack",
    "ulysses.mode1.pack",
    "ulysses.mode1.a2a",
    "ulysses.mode1.unpack",
    "ulysses.metadata_allgather",
    "ulysses.joint_allgather",
)


def _read_payloads(path: Path, warmups: int) -> list[dict[str, Any]]:
    payloads = []
    for line in path.read_text(errors="replace").splitlines():
        if PREFIX not in line:
            continue
        payload = json.loads(line.split(PREFIX, 1)[1])
        if int(payload["request"]) > warmups:
            payloads.append(payload)
    return payloads


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_root", type=Path)
    parser.add_argument("--modes", default="dlo-use-allgather,dlo-no-allgather")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--expected-ranks", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--detail-output", type=Path)
    args = parser.parse_args()

    summary_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for mode in args.modes.split(","):
        log_path = args.result_root / "e2e" / mode / "server.log"
        payloads = _read_payloads(log_path, args.warmups)
        by_request: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for payload in payloads:
            by_request[int(payload["request"])].append(payload)

        request_rows: list[dict[str, float]] = []
        request_bytes: list[dict[str, int]] = []
        rank_counts: list[int] = []
        for request, rank_payloads in sorted(by_request.items()):
            rank_counts.append(len(rank_payloads))
            metric_values: dict[str, float] = {}
            byte_values: dict[str, int] = {}
            for metric in METRICS:
                per_rank = [
                    float(payload.get("metrics", {}).get(metric, {}).get("total_ms", 0.0))
                    for payload in rank_payloads
                ]
                per_rank_bytes = [
                    int(payload.get("metrics", {}).get(metric, {}).get("bytes", 0))
                    for payload in rank_payloads
                ]
                metric_values[metric] = max(per_rank, default=0.0)
                byte_values[metric] = max(per_rank_bytes, default=0)
                detail_rows.append(
                    {
                        "mode": mode,
                        "request": request,
                        "metric": metric,
                        "rank_max_ms": max(per_rank, default=0.0),
                        "rank_mean_ms": _mean(per_rank),
                        "rank_max_bytes": max(per_rank_bytes, default=0),
                    }
                )
            request_rows.append(metric_values)
            request_bytes.append(byte_values)

        row: dict[str, Any] = {
            "mode": mode,
            "requests": len(request_rows),
            "min_ranks": min(rank_counts, default=0),
        }
        if args.expected_ranks and row["min_ranks"] != args.expected_ranks:
            raise SystemExit(
                f"{mode}: expected {args.expected_ranks} timing records per request, "
                f"found as few as {row['min_ranks']}"
            )
        for metric in METRICS:
            column = metric.replace(".", "_") + "_ms"
            row[column] = _mean([values[metric] for values in request_rows])
        transfer_ms = row["dlo_h2d_ms"] + row["dlo_allgather_ms"]
        exposed_ms = row["dlo_prefetch_wait_ms"] + row["dlo_resident_wait_ms"]
        hidden_ms = max(0.0, transfer_ms - exposed_ms)
        row["dlo_hidden_ms"] = hidden_ms
        row["dlo_overlap_pct"] = 100.0 * hidden_ms / transfer_ms if transfer_ms else 0.0
        h2d_bytes = _mean([float(values["dlo.h2d"]) for values in request_bytes])
        ag_bytes = _mean([float(values["dlo.allgather"]) for values in request_bytes])
        row["dlo_h2d_gbps"] = h2d_bytes / (row["dlo_h2d_ms"] * 1e6) if row["dlo_h2d_ms"] else 0.0
        row["dlo_allgather_gbps"] = (
            ag_bytes / (row["dlo_allgather_ms"] * 1e6) if row["dlo_allgather_ms"] else 0.0
        )
        summary_rows.append(row)

    if not summary_rows:
        raise SystemExit("no timing records found")
    columns = list(summary_rows[0])
    summary = "\t".join(columns) + "\n"
    summary += "\n".join(
        "\t".join(
            _fmt(value) if isinstance(value, float) else str(value)
            for value in (row[c] for c in columns)
        )
        for row in summary_rows
    )
    summary += "\n"
    print(summary, end="")
    if args.output:
        args.output.write_text(summary)

    if args.detail_output:
        detail_columns = list(detail_rows[0]) if detail_rows else []
        detail = "\t".join(detail_columns) + "\n"
        detail += "\n".join(
            "\t".join(
                _fmt(value) if isinstance(value, float) else str(value)
                for value in (row[column] for column in detail_columns)
            )
            for row in detail_rows
        )
        detail += "\n"
        args.detail_output.write_text(detail)


if __name__ == "__main__":
    main()
