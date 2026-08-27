#!/usr/bin/env bash
# Run one MiniMax H3 E2E backend. Called under tools/exclusive.sh.
set -Eeuo pipefail

BACKEND="${BACKEND:?set BACKEND to nccl, pitched-owned, packed-owned, auto-owned, or an explicit zero-copy diagnostic}"
WORK_ROOT="${WORK_ROOT:?set WORK_ROOT}"
MODEL_ROOT="${MODEL_ROOT:?set MODEL_ROOT}"
VLLM_OMNI_DIR="${VLLM_OMNI_DIR:?set VLLM_OMNI_DIR}"
RESULT_ROOT="${RESULT_ROOT:?set RESULT_ROOT}"
NUMA_NODE="${NUMA_NODE:-0}"
NUMA_POLICY="${NUMA_POLICY:-bind}"
NUM_GPUS="${NUM_GPUS:-4}"
TP_SIZE="${TP_SIZE:-2}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-2}"
TEXT_ENCODER_TP_SIZE="${TEXT_ENCODER_TP_SIZE:-${NUM_GPUS}}"
VAE_PATCH_PARALLEL_SIZE="${VAE_PATCH_PARALLEL_SIZE:-${NUM_GPUS}}"
DLO_MODE="${DLO_MODE:-off}"
DLO_RESIDENT_LAYERS="${DLO_RESIDENT_LAYERS:-0}"
OUTPUT_LABEL="${OUTPUT_LABEL:-${BACKEND}}"
NUM_INFERENCE_STEPS="${NUM_INFERENCE_STEPS:-5}"
WARMUPS="${WARMUPS:-2}"
MEASURED_RUNS="${MEASURED_RUNS:-3}"
RUNTIME_TIMING="${RUNTIME_TIMING:-0}"
PORT="${PORT:-8091}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
if [[ -z "${REQUEST_TIMEOUT:-}" ]]; then
  if (( NUM_INFERENCE_STEPS <= 5 )); then
    REQUEST_TIMEOUT=180
  else
    REQUEST_TIMEOUT=1800
  fi
fi

case "${BACKEND}" in
  nccl)
    transport="nccl"
    zero_copy=0
    ;;
  pitched-owned)
    transport="pitched"
    zero_copy=0
    ;;
  pitched-zero)
    transport="pitched"
    zero_copy=1
    ;;
  packed-owned)
    transport="packed"
    zero_copy=0
    ;;
  auto-owned)
    transport="auto"
    zero_copy=0
    ;;
  auto-zero)
    transport="auto"
    zero_copy=1
    ;;
  *) echo "invalid BACKEND=${BACKEND}" >&2; exit 2 ;;
esac

export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_ROOT}/xdg-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORK_ROOT}/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${WORK_ROOT}/torchinductor-cache}"
export PATH="${WORK_ROOT}/bin:${WORK_ROOT}/ffmpeg-tools:${WORK_ROOT}/ffmpeg-tools/bin:${WORK_ROOT}/ffmpeg-shared/bin:${VLLM_OMNI_DIR}/.venv/bin:${PATH}"
if [[ "${transport}" == "nccl" ]]; then
  unset VLLM_OMNI_ULYSSES_TRANSPORT
  unset VLLM_OMNI_FAST_ULYSSES_ALLOW_NON_NVLINK
  unset VLLM_OMNI_FAST_ULYSSES_ZERO_COPY
else
  export VLLM_OMNI_ULYSSES_TRANSPORT="${transport}"
  export VLLM_OMNI_FAST_ULYSSES_ALLOW_NON_NVLINK=1
  export VLLM_OMNI_FAST_ULYSSES_ZERO_COPY="${zero_copy}"
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_OMNI_VIDEO_SYNC_TIMEOUT=1800
export VLLM_OMNI_DIFFUSION_TIMING="${RUNTIME_TIMING}"

case "${DLO_MODE}" in
  off)
    dlo_args=()
    ;;
  use-allgather)
    dlo_args=(
      --enable-distributed-layerwise-offload
      --dlo-use-allgather
      --dlo-resident-layers "${DLO_RESIDENT_LAYERS}"
      --enforce-eager
    )
    ;;
  no-allgather)
    dlo_args=(
      --enable-distributed-layerwise-offload
      --dlo-no-use-allgather
      --dlo-resident-layers "${DLO_RESIDENT_LAYERS}"
      --enforce-eager
    )
    ;;
  *) echo "invalid DLO_MODE=${DLO_MODE}" >&2; exit 2 ;;
esac

case "${NUMA_POLICY}" in
  bind)
    numa_args=(--cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}")
    ;;
  interleave)
    numa_args=(--interleave=all)
    ;;
  none)
    numa_args=()
    ;;
  *) echo "invalid NUMA_POLICY=${NUMA_POLICY}" >&2; exit 2 ;;
esac

OUTPUT_DIR="${RESULT_ROOT}/e2e/${OUTPUT_LABEL}"
mkdir -p "${OUTPUT_DIR}"
printf '%s\n' "${BACKEND}" >"${OUTPUT_DIR}/backend.txt"
{
  printf 'BACKEND=%s\n' "${BACKEND}"
  printf 'TRANSPORT=%s\n' "${transport}"
  printf 'ZERO_COPY=%s\n' "${zero_copy}"
  printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-}"
  printf 'NUMA_NODE=%s\n' "${NUMA_NODE}"
  printf 'NUMA_POLICY=%s\n' "${NUMA_POLICY}"
  printf 'NUM_GPUS=%s\n' "${NUM_GPUS}"
  printf 'TP_SIZE=%s\n' "${TP_SIZE}"
  printf 'ULYSSES_DEGREE=%s\n' "${ULYSSES_DEGREE}"
  printf 'TEXT_ENCODER_TP_SIZE=%s\n' "${TEXT_ENCODER_TP_SIZE}"
  printf 'VAE_PATCH_PARALLEL_SIZE=%s\n' "${VAE_PATCH_PARALLEL_SIZE}"
  printf 'DLO_MODE=%s\n' "${DLO_MODE}"
  printf 'DLO_RESIDENT_LAYERS=%s\n' "${DLO_RESIDENT_LAYERS}"
  printf 'NUM_INFERENCE_STEPS=%s\n' "${NUM_INFERENCE_STEPS}"
  printf 'WARMUPS=%s\n' "${WARMUPS}"
  printf 'MEASURED_RUNS=%s\n' "${MEASURED_RUNS}"
  printf 'VLLM_OMNI_DIFFUSION_TIMING=%s\n' "${RUNTIME_TIMING}"
  printf 'REQUEST_TIMEOUT=%s\n' "${REQUEST_TIMEOUT}"
} >"${OUTPUT_DIR}/environment.txt"

server_pid=""
sampler_pid=""
rss_sampler_pid=""
cleanup() {
  if [[ -n "${rss_sampler_pid}" ]]; then
    kill "${rss_sampler_pid}" 2>/dev/null || true
    wait "${rss_sampler_pid}" 2>/dev/null || true
  fi
  if [[ -n "${sampler_pid}" ]]; then
    kill "${sampler_pid}" 2>/dev/null || true
    wait "${sampler_pid}" 2>/dev/null || true
  fi
  if [[ -n "${server_pid}" ]]; then
    kill -TERM -- "-${server_pid}" 2>/dev/null || true
    for _ in $(seq 1 60); do
      kill -0 "${server_pid}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL -- "-${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM HUP

show_server_failure() {
  echo "# server process group ${server_pid}" >&2
  ps -eo user,pid,ppid,pgid,etime,cmd | \
    awk -v pgid="${server_pid}" 'NR == 1 || $4 == pgid || $2 == pgid' >&2 || true
  echo "# first relevant server errors" >&2
  grep -nEi \
    'out of memory|cuda error|nvshmem|segmentation|signal|killed|traceback|exception|error' \
    "${OUTPUT_DIR}/server.log" | head -n 120 >&2 || true
  echo "# last 300 server log lines" >&2
  tail -n 300 "${OUTPUT_DIR}/server.log" >&2 || true
}

startup_start="$(date +%s)"
setsid numactl "${numa_args[@]}" \
  vllm serve "${MODEL_ROOT}/FL2VA" \
  --omni \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --trust-remote-code \
  --task-type fl2va \
  --num-gpus "${NUM_GPUS}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --usp "${ULYSSES_DEGREE}" \
  --ring 1 \
  --text-encoder-tp-size "${TEXT_ENCODER_TP_SIZE}" \
  --vae-patch-parallel-size "${VAE_PATCH_PARALLEL_SIZE}" \
  --vae-parallel-mode tile \
  --vae-use-tiling \
  --diffusion-attention-backend CUDNN_ATTN \
  --enable-diffusion-pipeline-profiler \
  --stage-init-timeout "${STARTUP_TIMEOUT}" \
  --init-timeout "${STARTUP_TIMEOUT}" \
  "${dlo_args[@]}" \
  >"${OUTPUT_DIR}/server.log" 2>&1 &
server_pid=$!
printf '%s\n' "${server_pid}" >"${OUTPUT_DIR}/server.pid"

nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu,power.draw \
  --format=csv -l 1 >"${OUTPUT_DIR}/gpu-samples.csv" 2>&1 &
sampler_pid=$!

(
  printf 'timestamp,total_process_group_rss_kib\n'
  while kill -0 "${server_pid}" 2>/dev/null; do
    printf '%s,' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    ps -eo pgid=,rss= | awk -v pgid="${server_pid}" '$1 == pgid {sum += $2} END {print sum + 0}'
    sleep 1
  done
) >"${OUTPUT_DIR}/process-rss-samples.csv" &
rss_sampler_pid=$!

deadline=$((SECONDS + STARTUP_TIMEOUT))
until curl -fsS "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; do
  kill -0 "${server_pid}" 2>/dev/null || {
    show_server_failure
    exit 1
  }
  (( SECONDS < deadline )) || {
    echo "server startup timed out after ${STARTUP_TIMEOUT}s" >&2
    show_server_failure
    exit 1
  }
  sleep 10
done
printf '%s\n' "$(( $(date +%s) - startup_start ))" >"${OUTPUT_DIR}/startup.seconds"

API_URL="http://127.0.0.1:${PORT}/v1/videos/sync"
request() {
  local label="$1"
  /usr/bin/time -f '%e' -o "${OUTPUT_DIR}/${label}.seconds" \
    curl --fail-with-body -sS --max-time "${REQUEST_TIMEOUT}" -D "${OUTPUT_DIR}/${label}.headers" \
    -X POST "${API_URL}" \
    -F 'prompt=At night, three cats march into a bedroom playing tiny brass instruments, then abruptly file out, with synchronized room ambience.' \
    -F 'width=1344' \
    -F 'height=768' \
    -F 'aspect_ratio=16:9' \
    -F 'fps=24' \
    -F "num_inference_steps=${NUM_INFERENCE_STEPS}" \
    -F 'flow_shift=12' \
    -F 'seed=1101' \
    -F 'extra_params={"task":"t2va","duration":5.0,"audio_flow_shift":3.0}' \
    -o "${OUTPUT_DIR}/${label}.mp4"
}

for warmup in $(seq 1 "${WARMUPS}"); do
  request "warmup-${warmup}"
done

if [[ "${transport}" == "auto" ]]; then
  grep -q "Selected fast-ulysses auto backend=" "${OUTPUT_DIR}/server.log" || {
    echo "server did not confirm a fast-ulysses auto selection; refusing to record fallback data" >&2
    exit 1
  }
elif [[ "${transport}" != "nccl" ]]; then
  grep -q "Initialized fast-ulysses transport backend=${transport}" "${OUTPUT_DIR}/server.log" || {
    echo "server did not confirm fast-ulysses backend=${transport}; refusing to record fallback data" >&2
    exit 1
  }
fi

if [[ "${zero_copy}" == "1" ]]; then
  grep -q "Allocated fast-ulysses zero-copy output" "${OUTPUT_DIR}/server.log" || {
    echo "server did not confirm zero-copy output allocation" >&2
    exit 1
  }
fi

if [[ "${DLO_MODE}" == "use-allgather" ]]; then
  grep -q "using SP group (world_size=${ULYSSES_DEGREE})" "${OUTPUT_DIR}/server.log" || {
    echo "server did not confirm DLO AllGather over SP${ULYSSES_DEGREE}" >&2
    exit 1
  }
elif [[ "${DLO_MODE}" == "no-allgather" ]]; then
  grep -q "dlo_use_allgather=False" "${OUTPUT_DIR}/server.log" || {
    echo "server did not confirm DLO no-AllGather mode" >&2
    exit 1
  }
fi

if [[ "${DLO_MODE}" != "off" ]]; then
  grep -q "unified shared_buffers=2" "${OUTPUT_DIR}/server.log" || {
    echo "server did not confirm the fixed two-buffer DLO residency" >&2
    exit 1
  }
fi

for run in $(seq 1 "${MEASURED_RUNS}"); do
  request "run-${run}"
done

ffprobe -v error -show_entries stream=index,codec_name,width,height,r_frame_rate,channels,sample_rate \
  -of json "${OUTPUT_DIR}/run-1.mp4" >"${OUTPUT_DIR}/run-1.ffprobe.json"
ffmpeg -v error -i "${OUTPUT_DIR}/run-1.mp4" -map 0:v -f framemd5 \
  "${OUTPUT_DIR}/run-1.video.framemd5"
ffmpeg -v error -i "${OUTPUT_DIR}/run-1.mp4" -map 0:a -f framemd5 \
  "${OUTPUT_DIR}/run-1.audio.framemd5"

awk '{sum += $1; count += 1} END {printf "mean_seconds=%.3f\n", sum / count}' \
  "${OUTPUT_DIR}"/run-*.seconds | tee "${OUTPUT_DIR}/summary.txt"
