"""Multi-GPU correctness suites, launched as torchrun subprocesses.

Each worker under tests/distributed/ is a standalone torchrun script and can also be run directly:
    torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py
Needs >=2 GPUs (skipped otherwise). By default each worker runs at min(ngpu, 8) processes plus, when
>=3 GPUs are present, an odd world size (different kernel template instantiations).
FAST_ULYSSES_TEST_NPROC overrides the process count list with a single value.
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


def _nprocs() -> list[int]:
    env = os.environ.get("FAST_ULYSSES_TEST_NPROC")
    if env:
        return [int(env)]
    ngpu = torch.cuda.device_count()
    out = {min(max(ngpu, 2), 8)}
    if ngpu >= 3:
        out.add(3)  # odd world size: exercises the odd-WS launch_ws template instantiations
    return sorted(out)


@pytest.mark.parametrize("nproc", _nprocs())
@pytest.mark.parametrize("worker", ["a2a_correctness.py", "a2a_async.py"])
def test_multigpu_worker(worker, nproc):
    ngpu = torch.cuda.device_count()
    if ngpu < max(nproc, 2):
        pytest.skip(f"needs >={max(nproc, 2)} GPUs, found {ngpu}")
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
