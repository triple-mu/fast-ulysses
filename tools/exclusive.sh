#!/usr/bin/env bash
# Run a benchmark only when the requested GPUs are exclusively ours, and prove it stayed
# that way.
#
# These boxes are shared and jobs arrive without warning. A run that starts on idle GPUs
# can be sharing them thirty seconds later, and the result looks like a real measurement --
# just a wrong one. Worse, partial contention (2 of 4 ranks slowed) skews a collective
# benchmark in a way that is invisible in the output.
#
# Usage:
#   ./tools/exclusive.sh 4,5,6,7 -- torchrun --nproc_per_node=4 benchmark/bench_a2a.py --mode stages
#   WAIT_SECS=1800 ./tools/exclusive.sh 4,5,6,7 -- <cmd>     # wait up to 30 min for idle
set -uo pipefail

DEVICES="${1:?usage: $0 <gpu list, e.g. 4,5,6,7> -- <command...>}"
shift
[[ "${1:-}" == "--" ]] && shift

WAIT_SECS="${WAIT_SECS:-0}"          # how long to wait for the GPUs to become free
SAMPLE_SECS="${SAMPLE_SECS:-5}"
# One process per GPU is what our own run contributes; anything beyond that is a neighbour.
EXPECT_PER_GPU="${EXPECT_PER_GPU:-1}"
EXCLUSIVE_DIAGNOSTICS="${EXCLUSIVE_DIAGNOSTICS:-}"

IFS=',' read -ra GPU_LIST <<< "${DEVICES}"
expected=$((${#GPU_LIST[@]} * EXPECT_PER_GPU))

if [[ -n "${EXCLUSIVE_DIAGNOSTICS}" ]]; then
    mkdir -p "$(dirname -- "${EXCLUSIVE_DIAGNOSTICS}")"
    : > "${EXCLUSIVE_DIAGNOSTICS}"
fi

uuid_of() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v i="$1" '$1==i{print $2}'; }

# Processes currently on the requested GPUs.
foreign_count() {
    local total=0
    for idx in "${GPU_LIST[@]}"; do
        local uuid n
        uuid=$(uuid_of "${idx}")
        n=$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null |
            grep -c "${uuid}" || true)
        total=$((total + n))
    done
    echo "${total}"
}

# --- 1. wait for idle ---
waited=0
while :; do
    busy=$(foreign_count)
    [[ "${busy}" -eq 0 ]] && break
    if [[ "${waited}" -ge "${WAIT_SECS}" ]]; then
        echo "REFUSED: GPUs ${DEVICES} have ${busy} process(es) on them and did not free up" \
             "within ${WAIT_SECS}s. Not measuring on shared GPUs." >&2
        exit 3
    fi
    sleep 10
    waited=$((waited + 10))
done
echo "# exclusive: GPUs ${DEVICES} idle after ${waited}s wait"

# --- 2. sample while the command runs ---
max_seen=0
min_clock=99999
samples=0
sample_file=$(mktemp /tmp/cno_exclusive_samples.XXXXXX)
# Trap the signals an interrupted ssh actually delivers, not just EXIT: a dropped
# connection leaves the sampler and its temp file behind otherwise, and stale samplers
# then show up as "foreign" processes to the next run.
cleanup() {
    [[ -n "${sampler:-}" ]] && kill "${sampler}" 2>/dev/null
    rm -f "${sample_file}"
}
trap cleanup EXIT INT TERM HUP

(
    # Stop if the parent went away, so an orphaned sampler cannot outlive the run.
    parent=$$
    while kill -0 "${parent}" 2>/dev/null; do
        n=$(foreign_count)
        if [[ -n "${EXCLUSIVE_DIAGNOSTICS}" && "${n}" -gt "${expected}" ]]; then
            {
                printf 'timestamp=%s process_count=%s expected=%s\n' \
                    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${n}" "${expected}"
                nvidia-smi --query-compute-apps=pid,process_name,gpu_uuid,used_gpu_memory \
                    --format=csv,noheader 2>&1
            } >> "${EXCLUSIVE_DIAGNOSTICS}"
        fi
        clk=$(nvidia-smi --query-gpu=index,clocks.sm --format=csv,noheader,nounits |
              awk -F', ' -v list="${DEVICES}" 'BEGIN{split(list,a,",");for(i in a)want[a[i]]=1}
                                               want[$1]{print $2}' | sort -n | head -1)
        echo "${n} ${clk:-0}"
        sleep "${SAMPLE_SECS}"
    done
) > "${sample_file}" &
sampler=$!

# Bind the command to the GPUs we just verified. Without this the script polices one set of
# devices and the run lands on another -- torch defaults to 0..N-1 -- so a VERDICT of
# EXCLUSIVE would say nothing about where the numbers came from. DEVICES is always physical
# indices, matching what nvidia-smi reports; the child sees them renumbered from 0.
export CUDA_VISIBLE_DEVICES="${DEVICES}"

"$@"
status=$?

kill "${sampler}" 2>/dev/null
wait "${sampler}" 2>/dev/null

# --- 3. verdict ---
while read -r n clk; do
    samples=$((samples + 1))
    [[ "${n}" -gt "${max_seen}" ]] && max_seen="${n}"
    if [[ -n "${clk}" && "${clk}" -gt 0 && "${clk}" -lt "${min_clock}" ]]; then
        min_clock="${clk}"
    fi
done < "${sample_file}"

if [[ "${max_seen}" -gt "${expected}" ]]; then
    echo "# VERDICT: CONTENDED -- saw up to ${max_seen} processes on ${DEVICES}," \
         "expected at most ${expected} (ours). Discard these numbers." >&2
    exit 4
fi
echo "# VERDICT: EXCLUSIVE (${samples} samples, max ${max_seen}/${expected} procs," \
     "min SM clock $([[ ${min_clock} -eq 99999 ]] && echo n/a || echo "${min_clock} MHz"))"
exit "${status}"
