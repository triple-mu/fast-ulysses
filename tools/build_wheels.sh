#!/usr/bin/env bash
# Build every CPython wheel for ONE (torch, CUDA) row, inside a manylinux builder container.
#
# One invocation per row rather than per (row, python) because the builder image is 8-10 GB
# compressed and pulling it dominates the job, while the CPython versions all compile the same
# .cu translation units -- ccache turns every python after the first into a link step. Splitting
# the matrix finer would pay the pull three more times to save nothing.
#
# The container's CUDA major must match the row's torch CUDA major. Build a cu13 torch in a
# CUDA 12 image and the extension links libcudart.so.12 against a CUDA 13 runtime:
# it installs cleanly and dies on import. tools/check_wheel.py is what refuses to let that
# out, which is why every wheel goes through it here and not only in the release job.
#
# Usage (repo bind-mounted at /io, which is also where the wheels land):
#   docker run --rm -v "$PWD:/io" -w /io \
#     -e TORCH_VERSION=2.13.0 -e TORCH_INDEX=pypi -e CUDA_MAJOR=13 -e LOCAL_VERSION=torch213cu130 \
#     pytorch/manylinux2_28-builder:cuda13.0 bash tools/build_wheels.sh
#
# Environment:
#   TORCH_VERSION   required, e.g. 2.13.0
#   TORCH_INDEX     required, an index URL or the literal "pypi"
#   CUDA_MAJOR      required, the container's CUDA major (12 or 13)
#   PYTHONS         default "cp310 cp311 cp312 cp313"
#   OUTDIR          default /io/wheelhouse
#   LOCAL_VERSION   default empty; empty builds the bare version, i.e. the wheel PyPI accepts
#   CUDA_ARCH       default "80;90;100;120", semicolon-separated
#   CCACHE_DIR      default /io/.ccache, shared by every python in the row
set -euo pipefail
# nullglob so a glob that matches nothing yields an empty array rather than the pattern
# itself -- the "expected one wheel" checks below only mean something with it on.
shopt -s nullglob

TORCH_VERSION="${TORCH_VERSION:?set TORCH_VERSION, e.g. 2.13.0}"
TORCH_INDEX="${TORCH_INDEX:?set TORCH_INDEX to an index URL or the literal pypi}"
CUDA_MAJOR="${CUDA_MAJOR:?set CUDA_MAJOR to the CUDA major of this container, 12 or 13}"
PYTHONS="${PYTHONS:-cp310 cp311 cp312 cp313}"
OUTDIR="${OUTDIR:-/io/wheelhouse}"
LOCAL_VERSION="${LOCAL_VERSION-}"
CUDA_ARCH="${CUDA_ARCH:-80;90;100;120}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO}"
mkdir -p "${OUTDIR}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT

export FAST_ULYSSES_LOCAL_VERSION="${LOCAL_VERSION}"
export FAST_ULYSSES_CUDA_ARCH="${CUDA_ARCH}"
# One cache for the whole row: the device objects do not depend on the CPython ABI.
export CCACHE_DIR="${CCACHE_DIR:-/io/.ccache}"
# Counters only -- the cached objects survive. Without this the summary at the end reports the
# lifetime of a cache restored from a previous run, which says nothing about this row.
command -v ccache >/dev/null && ccache --zero-stats >/dev/null

for PY in ${PYTHONS}; do
    PYBIN="/opt/python/${PY}-${PY}/bin"
    # Loud, not skipped: a typo in PYTHONS would otherwise ship a matrix row missing a python
    # and nothing downstream counts wheels per row.
    if [[ ! -x "${PYBIN}/python" ]]; then
        echo "FATAL: no interpreter at ${PYBIN}/python -- '${PY}' is not in this image." >&2
        echo "       present: $(compgen -G '/opt/python/*/' | tr '\n' ' ')" >&2
        exit 2
    fi

    echo "=== ${PY}: torch ${TORCH_VERSION} cu${CUDA_MAJOR}${LOCAL_VERSION:+ +${LOCAL_VERSION}} ==="
    "${PYBIN}/pip" install --upgrade "setuptools>=77" wheel "auditwheel>=6"
    if [[ "${TORCH_INDEX}" == "pypi" ]]; then
        "${PYBIN}/pip" install "torch==${TORCH_VERSION}"
    else
        "${PYBIN}/pip" install "torch==${TORCH_VERSION}" --index-url "${TORCH_INDEX}"
    fi

    # Its own CMake tree per config: a tree configured for another python or another torch has
    # those paths cached and would relink against them.
    export FAST_ULYSSES_BUILD_DIR="${WORK}/build-${PY}"
    rm -rf "${FAST_ULYSSES_BUILD_DIR}"

    raw_dir="${WORK}/raw-${PY}"
    repaired_dir="${WORK}/repaired-${PY}"
    rm -rf "${raw_dir}" "${repaired_dir}"
    mkdir -p "${raw_dir}" "${repaired_dir}"

    # Deliberately not piped anywhere. Past a pipe the status read back is the last stage's,
    # so a build that never ran reports success and the row silently ships nothing.
    "${PYBIN}/pip" wheel . --no-build-isolation --no-deps -w "${raw_dir}"
    raw=("${raw_dir}"/*.whl)
    if [[ ${#raw[@]} -ne 1 ]]; then
        echo "FATAL: expected one wheel from ${PY}, got: ${raw[*]}" >&2
        exit 1
    fi

    # The exclude list lives in check_wheel.py so the repair and the gate cannot disagree
    # about what counts as an external dependency.
    mapfile -t excludes < <("${PYBIN}/python" tools/check_wheel.py --print-excludes)
    if [[ ${#excludes[@]} -eq 0 ]]; then
        echo "FATAL: check_wheel.py --print-excludes produced nothing" >&2
        exit 1
    fi
    "${PYBIN}/python" -m auditwheel repair "${excludes[@]}" \
        --plat manylinux_2_28_x86_64 -w "${repaired_dir}" "${raw[0]}"
    repaired=("${repaired_dir}"/*.whl)
    if [[ ${#repaired[@]} -ne 1 ]]; then
        echo "FATAL: expected one repaired wheel from ${PY}, got: ${repaired[*]}" >&2
        exit 1
    fi

    # On the repaired wheel: repair rewrites DT_RUNPATH even when it vendors nothing.
    "${PYBIN}/python" tools/check_wheel.py \
        --cuda-major "${CUDA_MAJOR}" --arch "${CUDA_ARCH//;/,}" "${repaired[0]}"

    # Smoke test in a throwaway venv. --system-site-packages so the 2 GB torch is not
    # downloaded again; a venv all the same, so what gets imported is the installed package
    # and not the source tree in the working directory.
    venv="${WORK}/venv-${PY}"
    rm -rf "${venv}"
    "${PYBIN}/python" -m venv --system-site-packages "${venv}"
    "${venv}/bin/pip" install "${repaired[0]}"
    (
        cd "${venv}"
        "${venv}/bin/python" - <<'PY'
import torch

import fast_ulysses

assert torch.ops.fast_ulysses.a2a_plan_debug is not None, "a2a_plan_debug is not registered"
print("build_info:", fast_ulysses._C.build_info())
PY
    )

    mv "${repaired[0]}" "${OUTDIR}/"
    rm -rf "${raw_dir}" "${repaired_dir}" "${venv}"
done

# The claim in this file's header -- that ccache turns every python after the first into a link
# step -- is worth being able to check. Each python installs its torch under a different
# site-packages, so the -I paths differ and a miss here would be silent otherwise.
if command -v ccache >/dev/null; then
    echo "=== ccache, this row ==="
    ccache -s
fi

echo "=== ${OUTDIR} ==="
ls -1 "${OUTDIR}"
