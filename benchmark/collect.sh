#!/usr/bin/env bash
# One machine's full measurement, with the environment it ran in recorded next to the numbers.
#
# A benchmark is only worth as much as its attribution: two nodes of the same GPU model can
# disagree by more than anything being measured, and that is only catchable if every number says
# which machine, build and commit it came from. So this writes a fingerprint header before any
# number, and every measurement goes through tools/exclusive.sh -- which refuses to start on a
# busy GPU, samples throughout, and reports the lowest SM clock it saw.
#
# Usage:
#   benchmark/collect.sh <label> [gpu-list]
#     label     goes in the output filename, e.g. b200-node1
#     gpu-list  default: every GPU nvidia-smi reports
#
# Environment:
#   OUTDIR      where the log lands. Default ./benchmark-results
#   ITERS       passed through to bench_a2a.py. Default 25
#   SKIP_TESTS  1 to skip the correctness gate. Do not use for a published number.
#   ALLOW_NON_NVLINK
#               1 to measure a machine the constructor refuses (PCIe / cross-socket). The
#               refusal is itself a result, so this is never the default.
set -uo pipefail

LABEL="${1:?usage: $0 <label> [gpu-list]}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GPUS="${2:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd, -)}"
NPROC="$(tr ',' '\n' <<< "${GPUS}" | grep -c .)"
OUTDIR="${OUTDIR:-${PWD}/benchmark-results}"
ITERS="${ITERS:-25}"
mkdir -p "${OUTDIR}"
LOG="${OUTDIR}/${LABEL}.log"

exec > >(tee "${LOG}") 2>&1

echo "================================================================================"
echo "fast-ulysses measurement: ${LABEL}"
echo "================================================================================"
echo "date          $(date -Is)"
echo "host          $(hostname)"
echo "slurm job     ${SLURM_JOB_ID:-<none>}  node=${SLURMD_NODENAME:-<none>}"
echo "gpus          ${GPUS}  (nproc=${NPROC})"
echo "repo          ${REPO}"
# On a cluster the tree is usually rsync'd without .git, so fall back to the COMMIT file the
# sync writes. A number whose commit is unknown cannot be reproduced or retracted.
if git -C "${REPO}" rev-parse --short HEAD >/dev/null 2>&1; then
    echo "commit        $(git -C "${REPO}" rev-parse --short HEAD)  ($(git -C "${REPO}" status --porcelain | wc -l) modified file(s))"
elif [[ -f "${REPO}/COMMIT" ]]; then
    echo "commit        $(cat "${REPO}/COMMIT")  (from COMMIT file; tree synced without .git)"
else
    echo "commit        UNKNOWN -- this run is not attributable, do not publish from it"
fi
echo "driver        $(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
python3 - <<'PY'
import torch
print(f"torch         {torch.__version__}  cuda={torch.version.cuda}")
print(f"gpu           {torch.cuda.get_device_name(0)}  x{torch.cuda.device_count()}")
try:
    import fast_ulysses
    print(f"fast_ulysses  {fast_ulysses.__version__}  {fast_ulysses._C.build_info()}")
except Exception as exc:  # noqa: BLE001
    print(f"fast_ulysses  IMPORT FAILED: {exc}")
PY
echo
echo "--- clocks (before) ---"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu,power.limit --format=csv
echo
echo "--- nvidia-smi topo -m ---"
nvidia-smi topo -m 2>/dev/null || echo "(unavailable)"
echo
echo "--- fast-ulysses doctor ---"
fast-ulysses doctor 2>&1 || true
echo

NVLINK_FLAG=""
if [[ "${ALLOW_NON_NVLINK:-0}" == "1" ]]; then
    NVLINK_FLAG="--allow-non-nvlink"
    echo "NOTE: measuring a group the constructor refuses. The refusal is the headline result for"
    echo "      this machine; these numbers say what it costs to ignore it."
fi

FAILED=()

run() {  # run <mode> <nproc>
    local mode="$1" nproc="$2"
    echo
    echo "================================================================================"
    echo "== mode=${mode}  nproc=${nproc}"
    echo "================================================================================"
    "${REPO}/tools/exclusive.sh" "${GPUS}" -- \
        torchrun --nproc_per_node="${nproc}" "${REPO}/benchmark/bench_a2a.py" \
        --mode "${mode}" --iters "${ITERS}" ${NVLINK_FLAG} 2>&1 |
        grep -vE '^\[rank[1-9]|Setting OMP_NUM_THREADS|^\*\*\*\*|torch/distributed/run.py'
    # The run's own status, not the pipeline's: under pipefail a grep that selects no line exits 1
    # and would mark a mode that succeeded as failed. A mode that died is easy to miss in a log
    # this long, so collect the names and say so at the end.
    local status="${PIPESTATUS[0]}"
    if ((status != 0)); then
        FAILED+=("${mode}/nproc=${nproc}")
    fi
}

# The gate. A number from a build that fails its own tests is worse than no number, because it
# looks like data. Everything below is skipped if this fails.
if [[ "${SKIP_TESTS:-0}" != "1" ]]; then
    echo "================================================================================"
    echo "== correctness gate: pytest test/"
    echo "================================================================================"
    if ! (cd "${REPO}" && python3 -m pytest test/ -q 2>&1 | tail -25); then
        echo
        echo "GATE FAILED: tests did not pass on this machine. No measurements taken."
        exit 1
    fi
fi

for mode in stages zerocopy sweep link zerosm overlap padding; do
    run "${mode}" "${NPROC}"
done

# The stage table at half the world size too: the split shape changes with the world size.
#
# No odd-world-size pass here. The shapes are real models' -- 40 and 56 heads -- and mode 0
# scatters the head axis, so neither divides 3. Correctness at odd world sizes is pytest's job
# (test_distributed.py runs every worker at 3), and a performance number on a head count invented
# to divide 3 would not be comparable to anything else in the table.
if ((NPROC >= 4)); then
    run stages 4
fi

echo
echo "--- clocks (after) ---"
nvidia-smi --query-gpu=index,clocks.sm,clocks.max.sm,temperature.gpu --format=csv
echo
if ((${#FAILED[@]})); then
    echo "INCOMPLETE ${LABEL}: ${#FAILED[@]} mode(s) failed -> ${FAILED[*]}"
    echo "Publish nothing from this run until they are explained."
    echo "log: ${LOG}"
    exit 1
fi
echo "DONE ${LABEL} -> ${LOG}"
