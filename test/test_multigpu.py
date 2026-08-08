"""Multi-GPU correctness suites, launched as torchrun subprocesses.

Each worker under test/distributed/ is a standalone torchrun script and can also be run directly:
    torchrun --nproc_per_node=8 test/distributed/a2a_correctness.py
Needs >=2 GPUs (skipped otherwise). By default each worker runs at min(ngpu, 8) processes plus, when
>=3 GPUs are present, an odd world size (exercises the non-power-of-two peer sweep).
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
        out.add(3)  # odd world size: exercises launch_a2a_ce's non-power-of-two peer sweep
    return sorted(out)


@pytest.mark.parametrize("nproc", _nprocs())
@pytest.mark.parametrize(
    "worker",
    [
        "a2a_correctness.py",
        "a2a_fallback.py",
        "a2a_async.py",
        "a2a_uneven.py",
        "a2a_copy_out.py",
        # Adversarial workers. Each builds one specific unsafe timing and is worth only as much as
        # that timing: each module docstring names the NEGATIVE CONTROL (the line to delete to make
        # it fail) and what its failure looks like. Re-run those controls after any barrier change.
        "a2a_window_race.py",
        "a2a_cudagraph.py",
        "a2a_ce_flag_ordering.py",
        "a2a_overlapping_barriers.py",
        # The one adversarial worker whose control does NOT need a rebuild: it arms the fault
        # itself, so it re-proves it can fail on every run rather than in a comment.
        "a2a_ce_fault_injection.py",
        "a2a_alias_guard.py",
    ],
)
def test_multigpu_worker(worker, nproc):
    ngpu = torch.cuda.device_count()
    if ngpu < max(nproc, 2):
        pytest.skip(f"needs >={max(nproc, 2)} GPUs, found {ngpu}")
    _run_worker(worker, nproc)


def test_multigpu_torch_nvshmem_coexist():
    """NVSHMEM is a process-global singleton and torch ships its own.

    Runs at 4 ranks regardless of how many GPUs are present: the question is whether the two
    initialisations coexist, which two pairs already answer. Skips itself when torch's symmetric
    memory is unavailable, so it is inert on a build where the question does not arise.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip(f"needs >=4 GPUs, found {torch.cuda.device_count()}")
    _run_worker("a2a_torch_nvshmem_coexist.py", 4)


@pytest.mark.parametrize("worker", ["a2a_subgroup.py", "a2a_subgroup_divergent.py"])
def test_multigpu_subgroup(worker):
    """tp=2 x ulysses-sp: two stride-2 subgroups of the same job, live together.

    Identical shapes in both groups (a2a_subgroup) and, after reserve() has taken the collective
    allocation off the call path, deliberately divergent ones (a2a_subgroup_divergent).
    """
    env = os.environ.get("FAST_ULYSSES_TEST_NPROC")
    nproc = int(env) if env else min(torch.cuda.device_count(), 8)
    nproc -= nproc % 2  # tp=2 needs an even world
    if nproc < 4:
        pytest.skip(f"needs >=4 GPUs for tp=2 x sp>=2, found {torch.cuda.device_count()}")
    _run_worker(worker, nproc)


def _run_worker(worker: str, nproc: int) -> None:
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
