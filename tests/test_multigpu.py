"""Multi-GPU correctness suites, launched as torchrun subprocesses.

Each worker under tests/distributed/ is a standalone torchrun script and can also be run directly:
    torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py
Needs >=2 GPUs (skipped otherwise). FAST_ULYSSES_TEST_NPROC overrides the process count, e.g. to
exercise odd world sizes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("fast_ulysses", reason="fast_ulysses._C extension not built")

_DISTRIBUTED = Path(__file__).parent / "distributed"

pytestmark = pytest.mark.multigpu


@pytest.mark.parametrize("worker", ["a2a_correctness.py", "a2a_async.py", "a2a_qk.py"])
def test_multigpu_worker(worker):
    ngpu = torch.cuda.device_count()
    if ngpu < 2:
        pytest.skip(f"needs >=2 GPUs, found {ngpu}")
    nproc = int(os.environ.get("FAST_ULYSSES_TEST_NPROC", min(ngpu, 8)))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            f"--nproc_per_node={nproc}",
            str(_DISTRIBUTED / worker),
        ],
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, (
        f"{worker} failed (nproc={nproc})\n"
        f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
    )
