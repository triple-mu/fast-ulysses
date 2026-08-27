#!/usr/bin/env bash
# Build an isolated environment and run the MiniMax H3 packing A/B/C on RTX PRO 5000.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
FAST_ULYSSES_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
WORK_ROOT="${WORK_ROOT:-$(cd -- "${FAST_ULYSSES_ROOT}/.." && pwd)}"
ACTION="${1:-all}"

if [[ "${ACTION}" == "dlo-ab" || "${ACTION}" == "dlo-profile" ]]; then
  # Keep NCCL DLO experiments isolated from the older custom fast-ulysses
  # integration. The A/B uses official main; the profiler uses its dedicated
  # instrumentation branch. Both resolve vLLM from the checkout's CI image.
  if [[ "${ACTION}" == "dlo-profile" ]]; then
    VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-${WORK_ROOT}/vllm-omni-h3-dlo-profile}"
    VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/lishunyang12/vllm-omni.git}"
    VLLM_OMNI_BRANCH="${VLLM_OMNI_BRANCH:-bench/dlo-runtime-timing}"
  else
    VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-${WORK_ROOT}/vllm-omni-h3-dlo-latest}"
    VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/vllm-project/vllm-omni.git}"
    VLLM_OMNI_BRANCH="${VLLM_OMNI_BRANCH:-main}"
  fi
  VLLM_VERSION="${VLLM_VERSION:-auto}"
  INSTALL_FAST_ULYSSES="${INSTALL_FAST_ULYSSES:-0}"
else
  VLLM_OMNI_DIR="${VLLM_OMNI_DIR:-${WORK_ROOT}/vllm-omni-fast-ulysses}"
  VLLM_OMNI_REPO="${VLLM_OMNI_REPO:-https://github.com/lishunyang12/vllm-omni.git}"
  VLLM_OMNI_BRANCH="${VLLM_OMNI_BRANCH:-feat/fast-ulysses-transport-v026}"
  VLLM_VERSION="${VLLM_VERSION:-0.26.0}"
  INSTALL_FAST_ULYSSES="${INSTALL_FAST_ULYSSES:-1}"
fi
MODEL_ROOT="${MODEL_ROOT:-${WORK_ROOT}/MiniMax-H3}"
GPU_IDS="${GPU_IDS:-0,2,1,3}"
NUMA_NODE="${NUMA_NODE:-0}"
TP_SIZE="${TP_SIZE:-2}"
ULYSSES_DEGREE="${ULYSSES_DEGREE:-2}"
MICRO_RUNS="${MICRO_RUNS:-5}"
MICRO_ITERS="${MICRO_ITERS:-200}"
MICRO_WARMUP="${MICRO_WARMUP:-50}"
OVERLAP_RUNS="${OVERLAP_RUNS:-3}"
OVERLAP_ITERS="${OVERLAP_ITERS:-12}"
OVERLAP_WARMUP="${OVERLAP_WARMUP:-3}"
H3_SEQUENCE_LENGTH="${H3_SEQUENCE_LENGTH:-37760}"
H3_ATTENTION_BACKEND="${H3_ATTENTION_BACKEND:-cudnn}"
DLO_GPU_IDS="${DLO_GPU_IDS:-0,1,2,3,4,5,6,7}"
DLO_SP_BACKEND="${DLO_SP_BACKEND:-nccl}"
DLO_NUMA_POLICY="${DLO_NUMA_POLICY:-interleave}"
DLO_PROFILE_STEPS="${DLO_PROFILE_STEPS:-2}"
DLO_PROFILE_RESIDENT_LAYERS="${DLO_PROFILE_RESIDENT_LAYERS:-0}"
DLO_PROFILE_MODES="${DLO_PROFILE_MODES:-use-allgather,no-allgather}"
RUN_LEVEL="${RUN_LEVEL:-screen}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_ROOT="${RESULT_ROOT:-${WORK_ROOT}/results/h3-packing-${STAMP}}"

export HF_HOME="${HF_HOME:-${WORK_ROOT}/hf-cache}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${WORK_ROOT}/uv-cache}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${WORK_ROOT}/xdg-cache}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${WORK_ROOT}/triton-cache}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${WORK_ROOT}/torchinductor-cache}"
export PATH="${WORK_ROOT}/bin:${WORK_ROOT}/ffmpeg-tools:${WORK_ROOT}/ffmpeg-tools/bin:${WORK_ROOT}/ffmpeg-shared/bin:${PATH}"

if [[ -x "${WORK_ROOT}/bin/uv" ]]; then
  UV="${UV:-${WORK_ROOT}/bin/uv}"
else
  UV="${UV:-$(command -v uv || true)}"
fi

usage() {
  cat <<'EOF'
Usage: run_pro5000_suite.sh [setup|microbench|overlap|e2e|dlo-ab|dlo-profile|all]

Defaults match the validated socket-0 RTX PRO 5000 layout:
  GPU_IDS=0,2,1,3  -> Ulysses pairs are physical (0,1) and (2,3)
  TP_SIZE=2, ULYSSES_DEGREE=2, NUMA_NODE=0

RUN_LEVEL=screen uses 5 denoise steps. RUN_LEVEL=full uses 50 steps.
H3_E2E_BACKENDS is a comma-separated override. Zero-copy variants are diagnostics only.
Override RESULT_ROOT to append later phases to an existing result directory.

dlo-ab runs one isolated 8-GPU experiment:
  TP1 x Ulysses8, DLO with zero resident layers and two shared streaming buffers
  --dlo-use-allgather versus --dlo-no-use-allgather
  official vLLM-Omni main in a fresh worktree, with its CI-targeted vLLM release
Override DLO_GPU_IDS, DLO_SP_BACKEND, or DLO_NUMA_POLICY if needed.

dlo-profile is a short, env-gated run that records deferred CUDA-event and CPU
timings without Nsight Systems. It uses one warmup and one measured request.
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 2
}

acquire_e2e_lock() {
  require_command flock
  local devices="${1:-${GPU_IDS}}"
  local gpu_lock_key="${devices//,/-}"
  local lock_file="/tmp/fast-ulysses-h3-e2e-${USER:-unknown}-${gpu_lock_key}.lock"
  exec {E2E_LOCK_FD}<>"${lock_file}"
  flock -n "${E2E_LOCK_FD}" || \
    die "another H3 E2E suite is already using GPUs=${devices} (lock ${lock_file})"
  printf 'pid=%s\nresult_root=%s\n' "$$" "${RESULT_ROOT}" >"${lock_file}"
}

require_command() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

capture_machine() {
  local metadata_dir="${RESULT_ROOT}/metadata"
  local recorded_gpu_ids="${GPU_IDS}"
  local recorded_parallelism
  if [[ "${ACTION}" == "dlo-ab" || "${ACTION}" == "dlo-profile" ]]; then
    recorded_gpu_ids="${DLO_GPU_IDS}"
    recorded_parallelism=$'TP_SIZE=1\nULYSSES_DEGREE=8\nNUMA_POLICY='"${DLO_NUMA_POLICY}"
  else
    recorded_parallelism=$'TP_SIZE='"${TP_SIZE}"$'\nULYSSES_DEGREE='"${ULYSSES_DEGREE}"$'\nNUMA_NODE='"${NUMA_NODE}"
  fi
  mkdir -p "${metadata_dir}"
  nvidia-smi >"${metadata_dir}/nvidia-smi.txt"
  nvidia-smi topo -m >"${metadata_dir}/topology.txt"
  nvidia-smi --query-gpu=index,name,pci.bus_id,memory.total,driver_version \
    --format=csv >"${metadata_dir}/gpus.csv"
  lscpu >"${metadata_dir}/lscpu.txt"
  numactl --hardware >"${metadata_dir}/numa.txt"
  git -C "${FAST_ULYSSES_ROOT}" rev-parse HEAD >"${metadata_dir}/fast-ulysses.commit"
  printf '%s\n' "${recorded_gpu_ids}" >"${metadata_dir}/gpu-order.txt"
  printf '%s\n' "${recorded_parallelism}" >"${metadata_dir}/parallelism.env"
}

setup_env() {
  local resolved_vllm_version="${VLLM_VERSION}"
  [[ -n "${UV}" ]] || die "uv was not found; expected ${WORK_ROOT}/bin/uv or uv on PATH"
  require_command git
  require_command nvidia-smi
  require_command numactl
  require_command curl
  require_command ffmpeg
  require_command ffprobe

  mkdir -p "${RESULT_ROOT}" "${HF_HOME}" "${UV_CACHE_DIR}" "${XDG_CACHE_HOME}" \
    "${TRITON_CACHE_DIR}" "${TORCHINDUCTOR_CACHE_DIR}"
  capture_machine

  if [[ ! -d "${VLLM_OMNI_DIR}/.git" ]]; then
    git clone --branch "${VLLM_OMNI_BRANCH}" --single-branch \
      "${VLLM_OMNI_REPO}" "${VLLM_OMNI_DIR}"
  else
    [[ -z "$(git -C "${VLLM_OMNI_DIR}" status --short)" ]] || \
      die "${VLLM_OMNI_DIR} has local changes; refusing to update it"
    git -C "${VLLM_OMNI_DIR}" fetch origin \
      "refs/heads/${VLLM_OMNI_BRANCH}:refs/remotes/origin/${VLLM_OMNI_BRANCH}"
    if git -C "${VLLM_OMNI_DIR}" show-ref --verify --quiet \
      "refs/heads/${VLLM_OMNI_BRANCH}"; then
      git -C "${VLLM_OMNI_DIR}" checkout "${VLLM_OMNI_BRANCH}"
    else
      git -C "${VLLM_OMNI_DIR}" checkout -b "${VLLM_OMNI_BRANCH}" \
        "refs/remotes/origin/${VLLM_OMNI_BRANCH}"
    fi
    git -C "${VLLM_OMNI_DIR}" merge --ff-only "origin/${VLLM_OMNI_BRANCH}"
  fi

  if [[ "${resolved_vllm_version}" == "auto" ]]; then
    local ci_dockerfile="${VLLM_OMNI_DIR}/docker/Dockerfile.ci"
    [[ -f "${ci_dockerfile}" ]] || \
      die "cannot resolve vLLM compatibility: ${ci_dockerfile} is missing"
    resolved_vllm_version="$(sed -n 's/^ARG VLLM_BASE_TAG=v\{0,1\}\([^[:space:]]\+\)$/\1/p' \
      "${ci_dockerfile}" | head -n 1)"
    [[ -n "${resolved_vllm_version}" ]] || \
      die "cannot resolve vLLM version from ${ci_dockerfile}"
  fi

  if [[ ! -x "${VLLM_OMNI_DIR}/.venv/bin/python" ]]; then
    "${UV}" venv --python 3.12 --seed "${VLLM_OMNI_DIR}/.venv"
  fi
  export PATH="${VLLM_OMNI_DIR}/.venv/bin:${PATH}"

  "${UV}" pip install --python "${VLLM_OMNI_DIR}/.venv/bin/python" \
    "vllm==${resolved_vllm_version}" --torch-backend=auto
  "${UV}" pip install --python "${VLLM_OMNI_DIR}/.venv/bin/python" \
    -e "${VLLM_OMNI_DIR}"
  if [[ "${INSTALL_FAST_ULYSSES}" == "1" ]]; then
    "${UV}" pip install --python "${VLLM_OMNI_DIR}/.venv/bin/python" \
      cmake ninja
    FAST_ULYSSES_CUDA_ARCH=120 "${UV}" pip install \
      --python "${VLLM_OMNI_DIR}/.venv/bin/python" --no-build-isolation \
      -e "${FAST_ULYSSES_ROOT}"
  elif [[ "${INSTALL_FAST_ULYSSES}" != "0" ]]; then
    die "INSTALL_FAST_ULYSSES must be 0 or 1"
  fi

  git -C "${VLLM_OMNI_DIR}" rev-parse HEAD >"${RESULT_ROOT}/metadata/vllm-omni.commit"
  {
    printf 'VLLM_OMNI_REPO=%s\n' "${VLLM_OMNI_REPO}"
    printf 'VLLM_OMNI_BRANCH=%s\n' "${VLLM_OMNI_BRANCH}"
    printf 'VLLM_VERSION=%s\n' "${resolved_vllm_version}"
  } >"${RESULT_ROOT}/metadata/software-refs.env"
  "${VLLM_OMNI_DIR}/.venv/bin/python" - \
    >"${RESULT_ROOT}/metadata/python-environment.txt" <<'PY'
import torch
import vllm
import vllm_omni

print("torch", torch.__version__)
print("torch_cuda", torch.version.cuda)
print("vllm", vllm.__version__)
print("vllm_omni_version", vllm_omni.__version__)
print("vllm_omni", vllm_omni.__file__)
try:
    import fast_ulysses
except ModuleNotFoundError:
    print("fast_ulysses", "not installed (NCCL-only experiment)")
else:
    print("fast_ulysses", fast_ulysses.__version__, fast_ulysses.__file__)
print("capability", torch.cuda.get_device_capability())
PY
  cat "${RESULT_ROOT}/metadata/python-environment.txt"
}

run_microbench() {
  local torchrun="${VLLM_OMNI_DIR}/.venv/bin/torchrun"
  [[ -x "${torchrun}" ]] || die "environment missing; run '$0 setup' first"
  local output_dir="${RESULT_ROOT}/microbench"
  mkdir -p "${output_dir}"

  IFS=',' read -r -a ordered_gpus <<<"${GPU_IDS}"
  [[ "${#ordered_gpus[@]}" -eq 4 ]] || die "GPU_IDS must contain four physical GPU IDs"
  local pair0="${ordered_gpus[0]},${ordered_gpus[2]}"
  local pair1="${ordered_gpus[1]},${ordered_gpus[3]}"

  "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${GPU_IDS}" -- \
    numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
    "${torchrun}" --standalone --nproc_per_node=4 \
    "${FAST_ULYSSES_ROOT}/benchmark/bench_a2a.py" \
    --mode link --allow-non-nvlink --iters "${MICRO_ITERS}" --warmup "${MICRO_WARMUP}" \
    2>&1 | tee "${output_dir}/link.log"

  for pair in "${pair0}" "${pair1}"; do
    local pair_label="${pair//,/-}"
    "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${pair}" -- \
      numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
      "${torchrun}" --standalone --nproc_per_node=2 \
      "${FAST_ULYSSES_ROOT}/benchmark/bench_a2a.py" \
      --mode pcie-pretest --shape h3-t2va-5s --allow-non-nvlink \
      --iters "${MICRO_ITERS}" --warmup "${MICRO_WARMUP}" --host-mib 64 \
      2>&1 | tee "${output_dir}/decomposition-pair-${pair_label}.log"
  done

  for run in $(seq 1 "${MICRO_RUNS}"); do
    "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${GPU_IDS}" -- \
      numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
      "${torchrun}" --standalone --nproc_per_node=4 \
      "${FAST_ULYSSES_ROOT}/benchmark/bench_a2a.py" \
      --mode h3-block --shape h3-t2va-5s --tensor-parallel-size "${TP_SIZE}" \
      --sequence-length "${H3_SEQUENCE_LENGTH}" \
      --allow-non-nvlink --iters "${MICRO_ITERS}" --warmup "${MICRO_WARMUP}" \
      --blocks 50 --steps 50 --json-out "${output_dir}/h3-block-${run}.json" \
      2>&1 | tee "${output_dir}/h3-block-${run}.log"
  done

  "${VLLM_OMNI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/summarize_h3_block.py" \
    "${output_dir}"/h3-block-*.json --output "${output_dir}/h3-block-summary.tsv"
}

run_overlap() {
  local torchrun="${VLLM_OMNI_DIR}/.venv/bin/torchrun"
  [[ -x "${torchrun}" ]] || die "environment missing; run '$0 setup' first"
  local output_dir="${RESULT_ROOT}/overlap"
  mkdir -p "${output_dir}"

  for run in $(seq 1 "${OVERLAP_RUNS}"); do
    "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${GPU_IDS}" -- \
      numactl --cpunodebind="${NUMA_NODE}" --membind="${NUMA_NODE}" \
      "${torchrun}" --standalone --nproc_per_node=4 \
      "${SCRIPT_DIR}/bench_h3_tiled_overlap.py" \
      --sequence-length "${H3_SEQUENCE_LENGTH}" \
      --tensor-parallel-size "${TP_SIZE}" \
      --attention-backend "${H3_ATTENTION_BACKEND}" \
      --allow-non-nvlink --iters "${OVERLAP_ITERS}" --warmup "${OVERLAP_WARMUP}" \
      --json-out "${output_dir}/h3-tiled-overlap-${run}.json" \
      2>&1 | tee "${output_dir}/h3-tiled-overlap-${run}.log"
  done

  "${VLLM_OMNI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/summarize_h3_overlap.py" \
    "${output_dir}"/h3-tiled-overlap-*.json \
    --output "${output_dir}/h3-tiled-overlap-summary.tsv"
}

run_e2e() {
  [[ -f "${MODEL_ROOT}/FL2VA/model_index.json" ]] || \
    die "MiniMax H3 FL2VA checkpoint not found under ${MODEL_ROOT}/FL2VA"
  local backend_script="${SCRIPT_DIR}/run_h3_e2e_backend.sh"
  local steps=5 warmups=2 runs=3
  if [[ "${RUN_LEVEL}" == "full" ]]; then
    steps=50
  elif [[ "${RUN_LEVEL}" != "screen" ]]; then
    die "RUN_LEVEL must be 'screen' or 'full'"
  fi

  local backend_csv="${H3_E2E_BACKENDS:-nccl,pitched-owned,packed-owned,auto-owned}"
  local backends
  IFS=',' read -r -a backends <<<"${backend_csv}"
  [[ "${#backends[@]}" -gt 0 ]] || die "H3_E2E_BACKENDS selected no backends"
  for backend in "${backends[@]}"; do
    "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${GPU_IDS}" -- env \
      BACKEND="${backend}" WORK_ROOT="${WORK_ROOT}" MODEL_ROOT="${MODEL_ROOT}" \
      VLLM_OMNI_DIR="${VLLM_OMNI_DIR}" RESULT_ROOT="${RESULT_ROOT}" \
      NUMA_NODE="${NUMA_NODE}" TP_SIZE="${TP_SIZE}" ULYSSES_DEGREE="${ULYSSES_DEGREE}" \
      NUM_INFERENCE_STEPS="${steps}" WARMUPS="${warmups}" MEASURED_RUNS="${runs}" \
      bash "${backend_script}"
  done

  {
    printf 'backend\truns\tmean_seconds\n'
    for backend in "${backends[@]}"; do
      awk -v backend="${backend}" '
        {sum += $1; count += 1}
        END {printf "%s\t%d\t%.3f\n", backend, count, sum / count}
      ' "${RESULT_ROOT}/e2e/${backend}"/run-*.seconds
    done
  } | tee "${RESULT_ROOT}/e2e/summary.tsv"

  local video_checks=() audio_checks=()
  for backend in "${backends[@]}"; do
    video_checks+=("${RESULT_ROOT}/e2e/${backend}/run-1.video.framemd5")
    audio_checks+=("${RESULT_ROOT}/e2e/${backend}/run-1.audio.framemd5")
  done
  sha256sum "${video_checks[@]}" | tee "${RESULT_ROOT}/e2e/video-correctness.sha256"
  sha256sum "${audio_checks[@]}" | tee "${RESULT_ROOT}/e2e/audio-correctness.sha256"

  [[ "$(awk '{print $1}' "${RESULT_ROOT}/e2e/video-correctness.sha256" | sort -u | wc -l)" -eq 1 ]] || \
    die "decoded video FrameMD5 differs across backends"
  [[ "$(awk '{print $1}' "${RESULT_ROOT}/e2e/audio-correctness.sha256" | sort -u | wc -l)" -eq 1 ]] || \
    die "decoded audio FrameMD5 differs across backends"
}

run_dlo_ab() {
  [[ -f "${MODEL_ROOT}/FL2VA/model_index.json" ]] || \
    die "MiniMax H3 FL2VA checkpoint not found under ${MODEL_ROOT}/FL2VA"
  IFS=',' read -r -a dlo_gpus <<<"${DLO_GPU_IDS}"
  [[ "${#dlo_gpus[@]}" -eq 8 ]] || die "DLO_GPU_IDS must contain eight physical GPU IDs"

  local backend_script="${SCRIPT_DIR}/run_h3_e2e_backend.sh"
  local steps=5 warmups=2 runs=3
  if [[ "${RUN_LEVEL}" == "full" ]]; then
    steps=50
  elif [[ "${RUN_LEVEL}" != "screen" ]]; then
    die "RUN_LEVEL must be 'screen' or 'full'"
  fi

  mkdir -p "${RESULT_ROOT}/e2e"
  {
    printf 'DLO_GPU_IDS=%s\n' "${DLO_GPU_IDS}"
    printf 'DLO_SP_BACKEND=%s\n' "${DLO_SP_BACKEND}"
    printf 'DLO_NUMA_POLICY=%s\n' "${DLO_NUMA_POLICY}"
    printf 'DLO_RESIDENT_LAYERS=0\n'
    printf 'DLO_SHARED_STREAMING_BUFFERS=2\n'
    printf 'TP_SIZE=1\nULYSSES_DEGREE=8\n'
  } >"${RESULT_ROOT}/e2e/dlo-ab.env"

  local failed=0
  for mode in use-allgather no-allgather; do
    local label="dlo-${mode}"
    mkdir -p "${RESULT_ROOT}/e2e/${label}"
    if "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${DLO_GPU_IDS}" -- env \
      BACKEND="${DLO_SP_BACKEND}" OUTPUT_LABEL="${label}" DLO_MODE="${mode}" \
      WORK_ROOT="${WORK_ROOT}" MODEL_ROOT="${MODEL_ROOT}" \
      VLLM_OMNI_DIR="${VLLM_OMNI_DIR}" RESULT_ROOT="${RESULT_ROOT}" \
      NUMA_POLICY="${DLO_NUMA_POLICY}" NUM_GPUS=8 TP_SIZE=1 ULYSSES_DEGREE=8 \
      DLO_RESIDENT_LAYERS=0 \
      TEXT_ENCODER_TP_SIZE=8 VAE_PATCH_PARALLEL_SIZE=8 \
      NUM_INFERENCE_STEPS="${steps}" WARMUPS="${warmups}" MEASURED_RUNS="${runs}" \
      STARTUP_TIMEOUT=3600 REQUEST_TIMEOUT=1800 bash "${backend_script}"; then
      printf 'PASS\n' >"${RESULT_ROOT}/e2e/${label}/status.txt"
    else
      local rc=$?
      printf 'FAIL exit=%s\n' "${rc}" >"${RESULT_ROOT}/e2e/${label}/status.txt"
      failed=1
    fi
  done

  "${VLLM_OMNI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/summarize_h3_dlo.py" \
    "${RESULT_ROOT}" --measured-runs "${runs}" \
    --output "${RESULT_ROOT}/e2e/dlo-ab-summary.tsv"

  if [[ -f "${RESULT_ROOT}/e2e/dlo-use-allgather/run-1.video.framemd5" && \
        -f "${RESULT_ROOT}/e2e/dlo-no-allgather/run-1.video.framemd5" ]]; then
    sha256sum \
      "${RESULT_ROOT}/e2e/dlo-use-allgather/run-1.video.framemd5" \
      "${RESULT_ROOT}/e2e/dlo-no-allgather/run-1.video.framemd5" \
      | tee "${RESULT_ROOT}/e2e/dlo-video-correctness.sha256"
    [[ "$(awk '{print $1}' "${RESULT_ROOT}/e2e/dlo-video-correctness.sha256" | sort -u | wc -l)" -eq 1 ]] || \
      die "decoded video FrameMD5 differs between DLO modes"
  fi

  (( failed == 0 )) || die "one or more DLO modes failed; inspect the summary and server logs"
}

run_dlo_profile() {
  [[ -f "${MODEL_ROOT}/FL2VA/model_index.json" ]] || \
    die "MiniMax H3 FL2VA checkpoint not found under ${MODEL_ROOT}/FL2VA"
  IFS=',' read -r -a dlo_gpus <<<"${DLO_GPU_IDS}"
  [[ "${#dlo_gpus[@]}" -eq 8 ]] || die "DLO_GPU_IDS must contain eight physical GPU IDs"
  [[ "${DLO_PROFILE_STEPS}" -ge 2 ]] || die "DLO_PROFILE_STEPS must be at least 2"

  local backend_script="${SCRIPT_DIR}/run_h3_e2e_backend.sh"
  local warmups=1 runs=1 failed=0
  local profile_modes
  local profile_labels=()
  IFS=',' read -r -a profile_modes <<<"${DLO_PROFILE_MODES}"
  [[ "${#profile_modes[@]}" -gt 0 ]] || die "DLO_PROFILE_MODES selected no modes"
  mkdir -p "${RESULT_ROOT}/e2e"
  for mode in "${profile_modes[@]}"; do
    [[ "${mode}" == "use-allgather" || "${mode}" == "no-allgather" ]] || \
      die "invalid DLO profile mode: ${mode}"
    local label="dlo-${mode}"
    profile_labels+=("${label}")
    mkdir -p "${RESULT_ROOT}/e2e/${label}"
    if EXCLUSIVE_DIAGNOSTICS="${RESULT_ROOT}/e2e/${label}/exclusive-processes.log" \
      "${FAST_ULYSSES_ROOT}/tools/exclusive.sh" "${DLO_GPU_IDS}" -- env \
      BACKEND="${DLO_SP_BACKEND}" OUTPUT_LABEL="${label}" DLO_MODE="${mode}" \
      WORK_ROOT="${WORK_ROOT}" MODEL_ROOT="${MODEL_ROOT}" \
      VLLM_OMNI_DIR="${VLLM_OMNI_DIR}" RESULT_ROOT="${RESULT_ROOT}" \
      NUMA_POLICY="${DLO_NUMA_POLICY}" NUM_GPUS=8 TP_SIZE=1 ULYSSES_DEGREE=8 \
      DLO_RESIDENT_LAYERS="${DLO_PROFILE_RESIDENT_LAYERS}" \
      TEXT_ENCODER_TP_SIZE=8 VAE_PATCH_PARALLEL_SIZE=8 \
      NUM_INFERENCE_STEPS="${DLO_PROFILE_STEPS}" WARMUPS="${warmups}" \
      MEASURED_RUNS="${runs}" RUNTIME_TIMING=1 REQUEST_TIMEOUT=1800 \
      STARTUP_TIMEOUT=3600 bash "${backend_script}"; then
      printf 'PASS\n' >"${RESULT_ROOT}/e2e/${label}/status.txt"
    else
      local rc=$?
      printf 'FAIL exit=%s\n' "${rc}" >"${RESULT_ROOT}/e2e/${label}/status.txt"
      failed=1
    fi
  done

  local parser_modes
  parser_modes="$(IFS=,; echo "${profile_labels[*]}")"
  "${VLLM_OMNI_DIR}/.venv/bin/python" "${SCRIPT_DIR}/summarize_h3_runtime_timing.py" \
    "${RESULT_ROOT}" --modes "${parser_modes}" \
    --warmups "${warmups}" --expected-ranks 8 \
    --output "${RESULT_ROOT}/e2e/dlo-runtime-summary.tsv" \
    --detail-output "${RESULT_ROOT}/e2e/dlo-runtime-detail.tsv"
  (( failed == 0 )) || die "one or more DLO profile modes failed; inspect server logs"
}

case "${ACTION}" in
  setup)
    setup_env
    ;;
  microbench)
    setup_env
    run_microbench
    ;;
  overlap)
    setup_env
    run_overlap
    ;;
  e2e)
    acquire_e2e_lock
    setup_env
    run_e2e
    ;;
  dlo-ab)
    acquire_e2e_lock "${DLO_GPU_IDS}"
    setup_env
    run_dlo_ab
    ;;
  dlo-profile)
    acquire_e2e_lock "${DLO_GPU_IDS}"
    setup_env
    run_dlo_profile
    ;;
  all)
    acquire_e2e_lock
    setup_env
    run_microbench
    run_overlap
    run_e2e
    ;;
  -h|--help|help)
    usage
    exit 0
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac

echo "RESULT_ROOT=${RESULT_ROOT}"
