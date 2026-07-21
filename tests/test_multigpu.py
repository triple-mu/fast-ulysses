"""Multi-GPU correctness suites, launched as torchrun subprocesses.

Each worker under tests/distributed/ is a standalone torchrun script and can also be run directly:
    torchrun --nproc_per_node=8 tests/distributed/a2a_correctness.py
Needs >=2 GPUs (skipped otherwise). By default each worker runs at min(ngpu, 8) processes plus, when
>=3 GPUs are present, an odd world size (different kernel template instantiations).
FAST_ULYSSES_TEST_NPROC overrides the process count list with a single value.
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
        out.add(3)  # odd world size: exercises the odd-WS launch_ws template instantiations
    return sorted(out)


@pytest.mark.parametrize("nproc", _nprocs())
@pytest.mark.parametrize("worker", ["a2a_correctness.py", "a2a_async.py"])
def test_multigpu_worker(worker, nproc):
    ngpu = torch.cuda.device_count()
    if ngpu < max(nproc, 2):
        pytest.skip(f"needs >={max(nproc, 2)} GPUs, found {ngpu}")
    # Timeout teardown: a plain subprocess.run timeout SIGKILLs only the torchrun launcher
    # and orphans the rank workers (they live in their own sessions) -- a hung fast_barrier
    # spin kernel then keeps them pinned on the GPUs indefinitely, and inherited stdout
    # pipes would block the reader forever. So: log to files (no pipe to block on), SIGTERM
    # torchrun first (its elastic agent tears the workers down), and only then killpg as a
    # last resort.
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
    if timed_out:
        pytest.fail(
            f"{worker} timed out (nproc={nproc})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
        )
    assert proc.returncode == 0, (
        f"{worker} failed (nproc={nproc})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}"
    )
