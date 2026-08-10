#!/usr/bin/env bash
# The manual gate before pushing a v* tag: run the multi-GPU suite against a real wheel.
#
# CI has no GPU runner, so nothing in .github/ ever executes test/test_distributed.py -- the
# release job proves the wheel is well-formed and imports, and stops there. Everything the
# operator actually does (the symmetric windows, the transfers, the barriers) is only ever
# exercised here. Nothing else covers it, so a tag pushed without running this is untested.
#
# It installs the wheel into a clean venv rather than testing the working tree, because the
# working tree has a developer .so in it whose RPATH points at this machine: that build can
# pass while the wheel is broken. GPU idleness is tools/exclusive.sh's job, not this one's --
# numbers and pass/fail from a contended box are both worthless.
#
# Usage:
#   ./tools/preflight_gpu.sh dist/fast_ulysses-0.1.0+torch213cu130-cp312-*.whl
#   GPUS=0,1,2,3 TORCH_INDEX=https://download.pytorch.org/whl/cu128 ./tools/preflight_gpu.sh <whl>
#
# Environment:
#   GPUS          default: every index nvidia-smi reports
#   PYTHON        default python3; must match the wheel's cp3XX tag
#   TORCH_INDEX   index to take torch from before the wheel is installed. PyPI carries only
#                 one CUDA build per torch release, so a +cuNN wheel needs the matching index
#                 or the pin resolves to a torch linked against the other CUDA major.
#   WAIT_SECS     passed through to exclusive.sh: how long to wait for the GPUs to go idle
#   KEEP_VENV     1 to leave the venv behind for poking at
set -euo pipefail

WHEEL="${1:?usage: $0 <wheel> ; see the header for GPUS/PYTHON/TORCH_INDEX}"
[[ -f "${WHEEL}" ]] || { echo "no such wheel: ${WHEEL}" >&2; exit 2; }
WHEEL="$(cd "$(dirname "${WHEEL}")" && pwd)/$(basename "${WHEEL}")"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
TORCH_INDEX="${TORCH_INDEX-}"
GPUS="${GPUS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"

WORK="$(mktemp -d /tmp/fast_ulysses_preflight.XXXXXX)"
VENV="${WORK}/venv"
cleanup() {
    if [[ "${KEEP_VENV:-0}" == "1" ]]; then
        echo "# venv kept at ${VENV}"
    else
        rm -rf "${WORK}"
    fi
}
trap cleanup EXIT

echo "# preflight: ${WHEEL##*/} on GPUs ${GPUS}"
"${PYTHON}" -m venv "${VENV}"
"${VENV}/bin/pip" install --upgrade pip >/dev/null
# torch first when an index is given, so the wheel's torch==X.Y.* pin is already satisfied by
# the right CUDA build and pip has no reason to resolve it from PyPI.
if [[ -n "${TORCH_INDEX}" ]]; then
    "${VENV}/bin/pip" install torch --index-url "${TORCH_INDEX}"
fi
"${VENV}/bin/pip" install "${WHEEL}"
"${VENV}/bin/pip" install pytest numpy

# Every python below runs from a scratch dir, whatever the caller's cwd was. ${REPO}/python holds
# this same package with a developer build in it, and a cwd anywhere inside it would be imported
# in place of the wheel under test.
meta="$(cd "${WORK}" && "${VENV}/bin/python" -c '
from fast_ulysses import _build_meta as m
print(f"version={m.VERSION} torch={m.TORCH_VERSION} cuda={m.CUDA_VERSION} "
      f"arch={m.CUDA_ARCH_LIST} work_registry={m.HAS_WORK_REGISTRY}")
')"
py_version="$(cd "${WORK}" && "${VENV}/bin/python" -c \
    'import platform; print(platform.python_version())')"
torch_version="$(cd "${WORK}" && "${VENV}/bin/python" -c 'import torch; print(torch.__version__)')"

doctor_log="${WORK}/doctor.log"
set +e
(cd "${WORK}" && "${VENV}/bin/fast-ulysses" doctor) 2>&1 | tee "${doctor_log}"
doctor_status="${PIPESTATUS[0]}"  # $? here is tee's, which always succeeds
set -e

# Run from a scratch dir with an absolute test path, for the same shadowing reason: each worker
# is re-launched as `python -m torch.distributed.run`, and -m puts the cwd at the front of
# sys.path. The pytest console script (not `python -m pytest`) keeps the cwd off sys.path here.
suite_log="${WORK}/multigpu.log"
set +e
(cd "${WORK}" && "${REPO}/tools/exclusive.sh" "${GPUS}" -- \
    "${VENV}/bin/pytest" "${REPO}/test/test_distributed.py" -q) 2>&1 | tee "${suite_log}"
suite_status="${PIPESTATUS[0]}"
set -e

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)"
driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
# || true on every grep: pipefail is on, and a run that died early has none of these lines --
# which is exactly when the block below is worth printing.
# The verdicts `fast-ulysses doctor` can reach about the link topology. The release notes have to
# record which one this machine gave: the transport writes peer memory directly, so a run from a
# box that is not fully NVLink-joined says nothing about the wheel.
nvlink="$(grep -E "NVLink-joined|no pair to check|could not report the link topology" \
    "${doctor_log}" | tail -1 || true)"
# Lowercase on purpose: pytest's per-test lines say SKIPPED, only its final tally is lowercase.
suite_summary="$(grep -E "passed|failed|error|skipped|no tests ran" "${suite_log}" | tail -1 || true)"
verdict="$(grep -E "^# VERDICT" "${suite_log}" | tail -1 || true)"

doctor_line="ok"
if [[ "${doctor_status}" -ne 0 ]]; then
    doctor_line="FAILED (exit ${doctor_status})"
fi
suite_line="ok -- ${suite_summary}"
if [[ "${suite_status}" -ne 0 ]]; then
    suite_line="FAILED (exit ${suite_status}) -- ${suite_summary}"
fi

cat <<EOF

--- paste into the release notes ---
Preflight (manual; the multi-GPU suite has no CI runner)
  host        : $(hostname), ${gpu_count} x ${gpu_name}, driver ${driver}
  gpus used   : ${GPUS}
  wheel       : ${WHEEL##*/}
  python      : ${py_version}
  torch       : ${torch_version}
  build meta  : ${meta}
  doctor      : ${doctor_line}
  nvlink      : ${nvlink:-unreported}
  multigpu    : ${suite_line}
  exclusivity : ${verdict:-unreported}
------------------------------------
EOF

# exclusive.sh exits 3 (refused) or 4 (contended) without the suite having proven anything, so
# a non-zero status here is never "the wheel is fine, the box was busy" -- rerun it.
[[ "${doctor_status}" -eq 0 && "${suite_status}" -eq 0 ]] || exit 1
