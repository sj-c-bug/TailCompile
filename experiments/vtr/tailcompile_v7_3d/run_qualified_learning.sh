#!/usr/bin/env bash
# Qualified-design learning; the completed 004 cohort is a read-only dependency.
# Execute with bash. Do not source this launcher into a running 2D environment.
set -eEuo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

usage() {
  printf '%s\n' \
    'Usage: bash run_qualified_learning.sh STAGE [--background | --foreground]' \
    'Groups: cpu-teachers, gpu-pretrain, cpu-feedback, gpu-finetune, cpu-evaluate.' \
    'Stages: plan, teachers, pretrain, feedback, finetune, validate, compare, report.' \
    'Default is foreground. The current Python/Conda environment is inherited.' \
    'Source: development_004_capacity; output: development_005_qualified_learning.' \
    'CPU/GPU stages must finish in order. This is development, not formal 480-run acceptance.'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
stage="$1"
mode="${2:---foreground}"
case "$mode" in --background|--foreground) ;; *) usage >&2; exit 2 ;; esac
case "$stage" in
  cpu-teachers) stages=(plan teachers) ;;
  gpu-pretrain) stages=(pretrain) ;;
  cpu-feedback) stages=(feedback) ;;
  gpu-finetune) stages=(finetune) ;;
  cpu-evaluate) stages=(validate compare report) ;;
  plan|teachers|pretrain|feedback|finetune|validate|compare|report) stages=("$stage") ;;
  *) printf 'Unknown learning stage: %s\n' "$stage" >&2; usage >&2; exit 2 ;;
esac

module="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module/../../.." && pwd -P)"
[[ "$module" == "$repo/experiments/vtr/tailcompile_v7_3d" && -f "$repo/REPOSITORY_WORKFLOW.md" ]] || {
  echo 'Launcher must reside in the canonical TailCompile repository.' >&2; exit 2;
}
runs="$module/runs"
source="$(realpath -m "${TAILCOMPILE_V7_LEARNING_SOURCE:-$runs/development_004_capacity}")"
out="$(realpath -m "${TAILCOMPILE_V7_LEARNING_OUT:-$runs/development_005_qualified_learning}")"
scratch="$repo/vtr_work/tailcompile_v7_3d/qualified_learning"
logs="$repo/logs"
# Compare resolved destinations with lexical owned namespaces before any mkdir,
# lock file, log redirection or background launch. An escaping symlink is invalid.
for owned in "$runs" "$scratch" "$logs"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || {
    printf 'Owned directory resolves outside its repository namespace: %s\n' "$owned" >&2; exit 2;
  }
done
for cohort in "$source" "$out"; do
  case "$cohort/" in "$runs/"?*) ;; *) echo 'Learning source and output must be cohorts inside v7 runs/.' >&2; exit 2 ;; esac
done
case "$out/" in "$source/"*) echo 'Source and output must be disjoint; old results are read-only.' >&2; exit 2 ;; esac
case "$source/" in "$out/"*) echo 'Source and output must be disjoint; old results are read-only.' >&2; exit 2 ;; esac
for required in protocol.json campaign.json prepared.json conditions.json conditions_manifest.json binary_identity.json; do
  [[ -f "$source/$required" ]] || { printf 'Completed source cohort is missing: %s\n' "$source/$required" >&2; exit 2; }
done
[[ -f "$module/qualified_learning.py" ]] || { echo 'qualified_learning.py is missing; synchronize this commit first.' >&2; exit 2; }

python="${TAILCOMPILE_3D_PYTHON:-python3}"
command -v -- "$python" >/dev/null || { printf 'Python executable not found: %s\n' "$python" >&2; exit 2; }
jobs="${TAILCOMPILE_V7_JOBS:-8}"
job_mib="${TAILCOMPILE_V7_JOB_MIB:-4096}"
timeout="${TAILCOMPILE_V7_TIMEOUT:-3600}"
for value in "$jobs" "$job_mib" "$timeout"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Jobs, per-job memory and timeout must be positive integers.' >&2; exit 2; }
done
case "$stage" in gpu-*|pretrain|finetune) default_device=cuda ;; *) default_device=cpu ;; esac
device="${TAILCOMPILE_3D_TRAIN_DEVICE:-$default_device}"
case "$device" in cpu|cuda) ;; *) echo 'Use cpu or cuda for the training device; select GPU IDs with CUDA_VISIBLE_DEVICES.' >&2; exit 2 ;; esac

# Never reuse generic TAILCOMPILE_V7_OUT/CONFIG/SCRATCH or a stale env_3d_v7.sh.
export TAILCOMPILE_V7_LEARNING_SOURCE="$source" TAILCOMPILE_V7_LEARNING_OUT="$out"
export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/cache" TORCH_HOME="$scratch/cache/torch"
export TORCH_EXTENSIONS_DIR="$scratch/cache/torch_extensions" CUDA_CACHE_PATH="$scratch/cache/cuda"
export TRITON_CACHE_DIR="$scratch/cache/triton" MPLCONFIGDIR="$scratch/cache/matplotlib"
lock="$out/.qualified-learning-launch.lock"
for owned in "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
             "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$lock"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || {
    printf 'Runtime path is redirected by a symlink: %s\n' "$owned" >&2; exit 2;
  }
done

if [[ "$mode" == --background ]]; then
  mkdir -p -- "$logs"
  log="$(mktemp "$logs/v7_qualified_${stage}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  nohup bash "$module/run_qualified_learning.sh" "$stage" --foreground >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$log.pid"
  printf 'PID=%s\nLOG=%s\nPIDFILE=%s\nOUT=%s\n' "$pid" "$log" "$log.pid" "$out"
  exit 0
fi

mkdir -p -- "$out" "$TMPDIR" "$XDG_CACHE_HOME"
exec 9>"$lock"
flock -n 9 || { echo 'Another qualified-learning stage group is active for this output.' >&2; exit 2; }
trap 'code=$?; printf "V7_LEARNING_FAILED stage=%s exit=%s; inspect this log before advancing.\n" "$current" "$code" >&2; exit "$code"' ERR
common=(--source "$source" --out "$out" --jobs "$jobs" --per-job-mib "$job_mib"
        --timeout "$timeout" --device "$device" --threads 1)
for current in "${stages[@]}"; do
  printf 'V7_LEARNING_STAGE_START stage=%s source=%s out=%s device=%s\n' "$current" "$source" "$out" "$device"
  "$python" "$module/qualified_learning.py" "$current" "${common[@]}"
  printf 'V7_LEARNING_STAGE_COMPLETE stage=%s\n' "$current"
done
case "$current" in
  plan) next='cpu-teachers (collect the registered teachers)' ;;
  teachers) next='gpu-pretrain (train the teacher anchor model)' ;;
  pretrain) next='cpu-feedback (collect real P&R candidate feedback)' ;;
  feedback) next='gpu-finetune (train on successes and negative feedback)' ;;
  finetune) next='cpu-evaluate (select on validation, compare three methods, report)' ;;
  validate) next='compare (run the frozen five-design development panel)' ;;
  compare) next='report (summarize development results)' ;;
  report) next='inspect the development report; formal 480-run acceptance remains separate' ;;
esac
printf 'V7_LEARNING_GROUP_COMPLETE group=%s\nNEXT=%s\n' "$stage" "$next"
