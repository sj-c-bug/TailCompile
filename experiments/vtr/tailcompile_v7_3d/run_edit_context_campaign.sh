#!/usr/bin/env bash
# 012 offline scorer-label holdouts; no VPR or new candidate collection.
set -eEuo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

usage() {
  printf '%s\n' \
    'Usage: bash run_edit_context_campaign.sh plan|check|prepare|train|report|all [--foreground|--background]' \
    '012 reuses unchanged011 candidates/labels; 28 offline G/E fits, no P&R.' \
    'all: check -> prepare -> train -> report.'
}
if [[ "${1:---help}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
stage="$1"; mode="${2:---foreground}"
case "$stage" in
  plan|check|prepare|train|report) stages=("$stage") ;;
  all) stages=(check prepare train report) ;;
  *) printf 'Unknown 012 stage: %s\n' "$stage" >&2; usage >&2; exit 2 ;;
esac
case "$mode" in --foreground|--background) ;; *) usage >&2; exit 2 ;; esac
case "$stage:$mode" in plan:--background|check:--background)
  echo 'plan/check are read-only foreground checks.' >&2; exit 2 ;;
esac
module="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module/../../.." && pwd -P)"
[[ "$module" == "$repo/experiments/vtr/tailcompile_v7_3d" && -f "$repo/REPOSITORY_WORKFLOW.md" ]] || {
  echo 'Launcher must reside in the canonical TailCompile repository.' >&2; exit 2;
}
python="${TAILCOMPILE_3D_PYTHON:-python3}"
command -v -- "$python" >/dev/null || { echo 'Active Python unavailable.' >&2; exit 2; }
source="$(realpath -m "${TAILCOMPILE_V7_EDIT_CONTEXT_SOURCE:-$module/runs/development_011_residual}")"
out="${TAILCOMPILE_V7_EDIT_CONTEXT_OUT:-$module/runs/development_012_edit_context_parallel}"
scratch="$repo/vtr_work/tailcompile_v7_3d/edit_context"; logs="$repo/logs"
for owned in "$out" "$scratch" "$logs"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || {
    printf '012 path redirected outside namespace: %s\n' "$owned" >&2; exit 2;
  }
done
case "$source/" in "$module/runs/"?*) ;; *) echo 'Source must be a v7 runs cohort.' >&2; exit 2 ;; esac
case "$out/" in "$module/runs/development_012"?*) ;; *) echo 'Output requires development_012 namespace.' >&2; exit 2 ;; esac
case "$out/" in "$source/"*) echo 'Source011 and output012 must be disjoint.' >&2; exit 2 ;; esac
case "$source/" in "$out/"*) echo 'Source011 and output012 must be disjoint.' >&2; exit 2 ;; esac
[[ -f "$module/edit_context_campaign.py" ]] || { echo 'Synchronize edit_context_campaign.py first.' >&2; exit 2; }
device="${TAILCOMPILE_3D_TRAIN_DEVICE:-cpu}"
[[ "$device" =~ ^(cpu|cuda(:[0-9]+)?)$ ]] || { echo 'Use cpu, cuda or cuda:N.' >&2; exit 2; }
default_jobs=4
[[ "$device" == cpu ]] || default_jobs=1
jobs="${TAILCOMPILE_V7_EDIT_CONTEXT_JOBS-$default_jobs}"
fit_timeout="${TAILCOMPILE_V7_EDIT_CONTEXT_FIT_TIMEOUT-1800}"
for value in "$jobs" "$fit_timeout"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Jobs and fit timeout must be positive integers.' >&2; exit 2; }
done
[[ "$device" == cpu || "$jobs" == 1 ]] || { echo 'CUDA uses one fit process; choose CPU for parallel fits.' >&2; exit 2; }
common=(--source "$source" --out "$out" --device "$device" --jobs "$jobs" --fit-timeout "$fit_timeout")
if [[ -n "${TAILCOMPILE_V7_EDIT_CONTEXT_CACHE_FROM:-}" ]]; then
  cache_from="$(realpath -m "$TAILCOMPILE_V7_EDIT_CONTEXT_CACHE_FROM")"
  case "$cache_from/" in "$module/runs/development_012"?*) ;; *) echo 'Cache import requires a v7 012 cohort.' >&2; exit 2 ;; esac
  case "$cache_from/" in "$out/"*) echo 'Imported cache and output must be disjoint.' >&2; exit 2 ;; esac
  case "$out/" in "$cache_from/"*) echo 'Imported cache and output must be disjoint.' >&2; exit 2 ;; esac
  common+=(--cache-from "$cache_from")
fi
if [[ "$stage" == plan || "$stage" == check ]]; then
  exec "$python" "$module/edit_context_campaign.py" "$stage" "${common[@]}"
fi
export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/cache" TORCH_HOME="$scratch/cache/torch"
export TORCH_EXTENSIONS_DIR="$scratch/cache/torch_extensions" CUDA_CACHE_PATH="$scratch/cache/cuda"
export TRITON_CACHE_DIR="$scratch/cache/triton" MPLCONFIGDIR="$scratch/cache/matplotlib"
for owned in "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
             "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$out/.edit-context-launch.lock"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || {
    printf '012 cache/lock path redirected by symlink: %s\n' "$owned" >&2; exit 2;
  }
done
if [[ "$mode" == --background ]]; then
  mkdir -p -- "$logs"
  log="$(mktemp "$logs/v7_edit_context_${stage}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  nohup bash "$module/run_edit_context_campaign.sh" "$stage" --foreground >"$log" 2>&1 </dev/null &
  pid=$!; printf '%s\n' "$pid" >"$log.pid"
  printf 'PID=%s\nLOG=%s\nPIDFILE=%s\nSOURCE=%s\nOUT=%s\n' "$pid" "$log" "$log.pid" "$source" "$out"
  exit 0
fi
mkdir -p -- "$out" "$TMPDIR" "$XDG_CACHE_HOME"
exec 9>"$out/.edit-context-launch.lock"
flock -n 9 || { echo 'Another012 stage group is active for this output.' >&2; exit 2; }
current="startup"
trap 'code=$?; printf "V7_EDIT_CONTEXT_FAILED stage=%s exit=%s; later stages were not started.\n" "$current" "$code" >&2; exit "$code"' ERR
for current in "${stages[@]}"; do
  printf 'V7_EDIT_CONTEXT_STAGE_START stage=%s source=%s out=%s device=%s\n' "$current" "$source" "$out" "$device"
  "$python" "$module/edit_context_campaign.py" "$current" "${common[@]}"
  printf 'V7_EDIT_CONTEXT_STAGE_COMPLETE stage=%s\n' "$current"
done
printf 'V7_EDIT_CONTEXT_GROUP_COMPLETE group=%s\n' "$stage"
