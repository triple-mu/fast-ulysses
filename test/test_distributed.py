"""Runs the torchrun workers under test/distributed/ as subprocesses.

Each worker is standalone and can be run directly, which is the debugging path:
    torchrun --nproc_per_node=8 test/distributed/correctness.py

Needs >= 2 GPUs; skipped otherwise. Each worker runs at min(ngpu, 8) processes and, when >= 3 GPUs
are present, also at 3 -- an odd world size exercises the non-power-of-two peer sweep in
launch_a2a_ce. FAST_ULYSSES_TEST_NPROC overrides the list with a single value.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fast_ulysses", reason="fast_ulysses._C extension not built")

_DISTRIBUTED = Path(__file__).parent / "distributed"

pytestmark = pytest.mark.multigpu


def _nprocs() -> list[int]:
    env = os.environ.get("FAST_ULYSSES_TEST_NPROC")
    if env:
        return [int(env)]
    ngpu = torch.cuda.device_count()
    out = {min(max(ngpu, 2), 8)}
    if ngpu >= 3:
        out.add(3)
    return sorted(out)


@pytest.mark.parametrize("nproc", _nprocs())
@pytest.mark.parametrize(
    "worker",
    [
        "correctness.py",
        "validation.py",
        "ce_ordering.py",
        "cudagraph.py",
        "window_race.py",
        "overlapping_barriers.py",
        "subgroup.py",
    ],
)
def test_worker(worker: str, nproc: int) -> None:
    ngpu = torch.cuda.device_count()
    if ngpu < max(nproc, 2):
        pytest.skip(f"needs >={max(nproc, 2)} GPUs, found {ngpu}")
    _run(worker, nproc)


def _run(worker: str, nproc: int) -> None:
    # Timeout teardown: a plain subprocess.run timeout SIGKILLs only the torchrun launcher and
    # orphans the rank workers (they live in their own sessions) -- a hung barrier spin kernel then
    # keeps them pinned on the GPUs indefinitely, and inherited stdout pipes would block the reader
    # forever. So: log to files, SIGTERM torchrun first (its elastic agent tears the workers down),
    # and only then killpg.
    with tempfile.TemporaryFile(mode="w+") as out_f, tempfile.TemporaryFile(mode="w+") as err_f:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                f"--nproc_per_node={nproc}",
                str(_DISTRIBUTED / worker),
            ],
            stdout=out_f,
            stderr=err_f,
            text=True,
            start_new_session=True,
        )
        timed_out = False
        try:
            proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            timed_out = True
            proc.terminate()  # elastic agent SIGTERMs and reaps its workers
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
        out_f.seek(0)
        err_f.seek(0)
        stdout, stderr = out_f.read(), err_f.read()
    report = f"\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    if timed_out:
        pytest.fail(f"{worker} timed out (nproc={nproc}){report}")
    assert proc.returncode == 0, f"{worker} failed (nproc={nproc}){report}"
