import json
import subprocess
import sys
from pathlib import Path


def _payload(rank: int, request: int, h2d: float, gather: float, wait: float) -> dict:
    return {
        "rank": rank,
        "request": request,
        "metrics": {
            "dlo.h2d": {"total_ms": h2d, "count": 2, "bytes": 200_000_000},
            "dlo.allgather": {"total_ms": gather, "count": 2, "bytes": 700_000_000},
            "dlo.prefetch_wait": {"total_ms": wait, "count": 2, "bytes": 0},
        },
        "derived": {},
    }


def test_runtime_summary_uses_slowest_rank_and_skips_warmup(tmp_path: Path) -> None:
    root = tmp_path / "result"
    for mode in ("dlo-use-allgather", "dlo-no-allgather"):
        mode_dir = root / "e2e" / mode
        mode_dir.mkdir(parents=True)
        payloads = [
            _payload(0, 1, 100.0, 100.0, 100.0),
            _payload(1, 1, 100.0, 100.0, 100.0),
            _payload(0, 2, 4.0, 6.0, 2.0),
            _payload(1, 2, 5.0, 7.0, 3.0),
        ]
        lines = [f"INFO [DiffusionRuntimeTiming] {json.dumps(payload)}" for payload in payloads]
        (mode_dir / "server.log").write_text("\n".join(lines) + "\n")

    script = (
        Path(__file__).parents[1] / "benchmark" / "h3_packing" / "summarize_h3_runtime_timing.py"
    )
    output = root / "summary.tsv"
    subprocess.run(
        [sys.executable, str(script), str(root), "--warmups", "1", "--output", str(output)],
        check=True,
        capture_output=True,
        text=True,
    )

    columns, first_row, second_row = [line.split("\t") for line in output.read_text().splitlines()]
    use_ag = dict(zip(columns, first_row, strict=True))
    no_ag = dict(zip(columns, second_row, strict=True))
    assert use_ag["requests"] == "1"
    assert use_ag["min_ranks"] == "2"
    assert use_ag["dlo_h2d_ms"] == "5.000"
    assert use_ag["dlo_allgather_ms"] == "7.000"
    assert use_ag["dlo_prefetch_wait_ms"] == "3.000"
    assert use_ag["dlo_hidden_ms"] == "9.000"
    assert use_ag["dlo_overlap_pct"] == "75.000"
    assert no_ag["requests"] == "1"
