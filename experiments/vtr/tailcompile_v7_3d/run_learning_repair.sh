#!/usr/bin/env bash
# 006 repair: 004 inputs and 005 teachers/checkpoints remain read-only.
# Execute with bash. Do not source this launcher into a running 2D environment.
set -eEuo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

usage() {
  printf '%s\n' 'Usage: bash run_learning_repair.sh all|cpu-prepare|cpu-feedback|gpu-train|cpu-evaluate [--background|--foreground]' \
    'New cohort006; source004 and completed005 are read-only. No calibration or teacher recollection.'
}
if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
stage="$1"
mode="${2:---foreground}"
case "$mode" in --background|--foreground) ;; *) usage >&2; exit 2 ;; esac
case "$stage" in
  all) stages=(prepare feedback train validate compare report) ;;
  cpu-prepare) stages=(prepare) ;;
  cpu-feedback) stages=(feedback) ;;
  gpu-train) stages=(train) ;;
  cpu-evaluate) stages=(validate compare report) ;;
  prepare|feedback|train|validate|compare|report) stages=("$stage") ;;
  *) usage >&2; exit 2 ;;
esac

module="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module/../../.." && pwd -P)"
[[ "$module" == "$repo/experiments/vtr/tailcompile_v7_3d" && -f "$repo/REPOSITORY_WORKFLOW.md" ]] || {
  echo 'Launcher must reside in the canonical TailCompile repository.' >&2; exit 2;
}
runs="$module/runs"
source="$(realpath -m "${TAILCOMPILE_V7_REPAIR_SOURCE:-$runs/development_004_capacity}")"
out="$(realpath -m "${TAILCOMPILE_V7_REPAIR_OUT:-$runs/development_006_learning_repair}")"
prior="$(realpath -m "${TAILCOMPILE_V7_REPAIR_PRIOR:-$runs/development_005_qualified_learning}")"
scratch="$repo/vtr_work/tailcompile_v7_3d/learning_repair"
logs="$repo/logs"
# Compare resolved destinations with lexical owned namespaces before any mkdir,
# lock file, log redirection or background launch. An escaping symlink is invalid.
for owned in "$runs" "$scratch" "$logs"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || {
    printf 'Owned directory resolves outside its repository namespace: %s\n' "$owned" >&2; exit 2;
  }
done
for cohort in "$source" "$prior" "$out"; do
  case "$cohort/" in "$runs/"?*) ;; *) echo 'Learning source and output must be cohorts inside v7 runs/.' >&2; exit 2 ;; esac
done
case "$out/" in "$source/"*) echo 'Source and output must be disjoint; old results are read-only.' >&2; exit 2 ;; esac
case "$source/" in "$out/"*) echo 'Source and output must be disjoint; old results are read-only.' >&2; exit 2 ;; esac
case "$out/" in "$prior/"*) echo 'Prior and output must be disjoint.' >&2; exit 2 ;; esac
case "$prior/" in "$out/"*) echo 'Prior and output must be disjoint.' >&2; exit 2 ;; esac
for required in selection.json teacher_records.json models/pretrain/completed.json models/pretrain/input_signature.json; do
  [[ -f "$prior/$required" ]] || { printf 'Completed005 input is missing: %s\n' "$prior/$required" >&2; exit 2; }
done
for required in protocol.json campaign.json prepared.json conditions.json conditions_manifest.json binary_identity.json; do
  [[ -f "$source/$required" ]] || { printf 'Completed source cohort is missing: %s\n' "$source/$required" >&2; exit 2; }
done
[[ -f "$module/repair_learning.py" ]] || { echo 'repair_learning.py is missing; synchronize this commit first.' >&2; exit 2; }

python="${TAILCOMPILE_3D_PYTHON:-python3}"
command -v -- "$python" >/dev/null || { printf 'Python executable not found: %s\n' "$python" >&2; exit 2; }
jobs="${TAILCOMPILE_V7_JOBS:-4}"
job_mib="${TAILCOMPILE_V7_JOB_MIB:-4096}"
timeout="${TAILCOMPILE_V7_TIMEOUT:-3600}"
for value in "$jobs" "$job_mib" "$timeout"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Jobs, per-job memory and timeout must be positive integers.' >&2; exit 2; }
done
case "$stage" in all|gpu-*|train) default_device=cuda ;; *) default_device=cpu ;; esac
device="${TAILCOMPILE_3D_TRAIN_DEVICE:-$default_device}"
case "$device" in cpu|cuda) ;; *) echo 'Use cpu or cuda for the training device; select GPU IDs with CUDA_VISIBLE_DEVICES.' >&2; exit 2 ;; esac

# Never reuse generic TAILCOMPILE_V7_OUT/CONFIG/SCRATCH or a stale env_3d_v7.sh.
export TAILCOMPILE_V7_REPAIR_SOURCE="$source" TAILCOMPILE_V7_REPAIR_OUT="$out" TAILCOMPILE_V7_REPAIR_PRIOR="$prior"
export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/cache" TORCH_HOME="$scratch/cache/torch"
export TORCH_EXTENSIONS_DIR="$scratch/cache/torch_extensions" CUDA_CACHE_PATH="$scratch/cache/cuda"
export TRITON_CACHE_DIR="$scratch/cache/triton" MPLCONFIGDIR="$scratch/cache/matplotlib"
lock="$out/.learning-repair-launch.lock"
for owned in "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
             "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$lock"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || {
    printf 'Runtime path is redirected by a symlink: %s\n' "$owned" >&2; exit 2;
  }
done

if [[ "$mode" == --background ]]; then
  mkdir -p -- "$logs"
  log="$(mktemp "$logs/v7_repair_${stage}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  nohup bash "$module/run_learning_repair.sh" "$stage" --foreground >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$log.pid"
  printf 'PID=%s\nLOG=%s\nPIDFILE=%s\nOUT=%s\n' "$pid" "$log" "$log.pid" "$out"
  exit 0
fi

mkdir -p -- "$out" "$TMPDIR" "$XDG_CACHE_HOME"
exec 9>"$lock"
flock -n 9 || { echo 'Another qualified-learning stage group is active for this output.' >&2; exit 2; }
trap 'code=$?; printf "V7_REPAIR_FAILED stage=%s exit=%s; inspect this log before advancing.\n" "$current" "$code" >&2; exit "$code"' ERR
common=(--source "$source" --prior "$prior" --out "$out" --jobs "$jobs" --per-job-mib "$job_mib"
        --timeout "$timeout" --device "$device")
for current in "${stages[@]}"; do
  printf 'V7_REPAIR_STAGE_START stage=%s source=%s out=%s device=%s\n' "$current" "$source" "$out" "$device"
  "$python" "$module/repair_learning.py" "$current" "${common[@]}"
  printf 'V7_REPAIR_STAGE_COMPLETE stage=%s\n' "$current"
done
case "$current" in
  prepare) next='cpu-feedback (collect updated measured candidate costs)' ;;
  feedback) next='gpu-train (matched three-epoch clipped/guarded arms)' ;;
  train) next='cpu-evaluate (validation selection and controlled comparison)' ;;
  validate) next='compare' ;;
  compare) next='report' ;;
  report) next='inspect repair_report.json; no formal acceptance claim' ;;
esac
printf 'V7_REPAIR_GROUP_COMPLETE group=%s\nNEXT=%s\n' "$stage" "$next"
