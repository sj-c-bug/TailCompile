#!/usr/bin/env bash
# 010: Resource/density decoding, netlist/geometry GNN and paired QoR experiments.
# Uses the active Python environment. Execute with bash; do not source it.
set -eEuo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

usage() {
  printf '%s\n' \
    'Usage: bash run_structured_campaign.sh plan|check|prepare|feedback|train|validate|compare|report|evaluate|all [--foreground|--background]' \
    '010 uses completed009X, strict resource decoding and new deployable feedback; no Tail/backend rebuild.' \
    'all: check -> prepare -> feedback -> train -> validate -> compare -> report.'
}
if [[ "${1:---help}" == --help || "${1:-}" == -h ]]; then usage; exit 0; fi
[[ $# -ge 1 && $# -le 2 ]] || { usage >&2; exit 2; }
stage="$1"
mode="${2:---foreground}"
case "$stage" in
  plan|check|prepare|feedback|train|validate|compare|report) stages=("$stage") ;;
  evaluate) stages=(validate compare report) ;;
  all) stages=(check prepare feedback train validate compare report) ;;
  *) printf 'Unknown 010 stage: %s\n' "$stage" >&2; usage >&2; exit 2 ;;
esac
case "$mode" in --foreground|--background) ;; *) usage >&2; exit 2 ;; esac
case "$stage:$mode" in plan:--background|check:--background)
  echo 'plan/check are read-only foreground checks; use --foreground.' >&2; exit 2 ;;
esac

module="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module/../../.." && pwd -P)"
[[ "$module" == "$repo/experiments/vtr/tailcompile_v7_3d" && -f "$repo/REPOSITORY_WORKFLOW.md" ]] || {
  echo 'Launcher must reside in the canonical TailCompile repository.' >&2; exit 2;
}
python="${TAILCOMPILE_3D_PYTHON:-python3}"
command -v -- "$python" >/dev/null || { echo 'Active Python executable is unavailable.' >&2; exit 2; }
source="$(realpath -m "${TAILCOMPILE_V7_STRUCTURED_SOURCE:-$module/runs/development_009_qor}")"
out="${TAILCOMPILE_V7_STRUCTURED_OUT:-$module/runs/development_010_structured}"
scratch="$repo/vtr_work/tailcompile_v7_3d/structured"
logs="$repo/logs"

# Resolve all future writes before mkdir, lock creation, or log redirection.
for owned in "$out" "$scratch" "$logs"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || {
    printf '010 write path is redirected outside its namespace: %s\n' "$owned" >&2; exit 2;
  }
done
case "$source/" in "$module/runs/"?*) ;; *) echo 'Source must be a 3D cohort inside v7 runs.' >&2; exit 2 ;; esac
case "$out/" in "$module/runs/development_010"?*) ;; *) echo 'Output must be a new development_010 cohort inside v7 runs.' >&2; exit 2 ;; esac
case "$out/" in "$source/"*) echo 'Source009 and output010 must be disjoint.' >&2; exit 2 ;; esac
case "$source/" in "$out/"*) echo 'Source009 and output010 must be disjoint.' >&2; exit 2 ;; esac
[[ -f "$module/structured_campaign.py" ]] || { echo 'structured_campaign.py is missing; synchronize the implementation commit.' >&2; exit 2; }

jobs="${TAILCOMPILE_V7_STRUCTURED_JOBS:-4}"
job_mib="${TAILCOMPILE_V7_STRUCTURED_JOB_MIB:-8192}"
timeout="${TAILCOMPILE_V7_STRUCTURED_TIMEOUT:-3600}"
for value in "$jobs" "$job_mib" "$timeout"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo 'Jobs, per-job MiB and timeout must be positive integers.' >&2; exit 2; }
done
device="${TAILCOMPILE_3D_TRAIN_DEVICE:-cpu}"
[[ "$device" =~ ^(cpu|cuda(:[0-9]+)?)$ ]] || { echo 'Use cpu, cuda or cuda:N as training device.' >&2; exit 2; }
common=(--source "$source" --out "$out" --jobs "$jobs" --per-job-mib "$job_mib" --timeout "$timeout" --device "$device")

if [[ "$stage" == plan || "$stage" == check ]]; then
  exec "$python" "$module/structured_campaign.py" "$stage" "${common[@]}"
fi

export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/cache" TORCH_HOME="$scratch/cache/torch"
export TORCH_EXTENSIONS_DIR="$scratch/cache/torch_extensions" CUDA_CACHE_PATH="$scratch/cache/cuda"
export TRITON_CACHE_DIR="$scratch/cache/triton" MPLCONFIGDIR="$scratch/cache/matplotlib"
for owned in "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$TORCH_EXTENSIONS_DIR" \
             "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR" "$MPLCONFIGDIR" "$out/.structured-launch.lock"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || {
    printf '010 runtime path is redirected by a symlink: %s\n' "$owned" >&2; exit 2;
  }
done

if [[ "$mode" == --background ]]; then
  mkdir -p -- "$logs"
  log="$(mktemp "$logs/v7_structured_${stage}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  nohup bash "$module/run_structured_campaign.sh" "$stage" --foreground >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$log.pid"
  printf 'PID=%s\nLOG=%s\nPIDFILE=%s\nSOURCE=%s\nOUT=%s\n' "$pid" "$log" "$log.pid" "$source" "$out"
  exit 0
fi

mkdir -p -- "$out" "$TMPDIR" "$XDG_CACHE_HOME"
exec 9>"$out/.structured-launch.lock"
flock -n 9 || { echo 'Another 010 stage group is active for this output.' >&2; exit 2; }
current="startup"
trap 'code=$?; printf "V7_STRUCTURED_FAILED stage=%s exit=%s; later stages were not started.\n" "$current" "$code" >&2; exit "$code"' ERR
for current in "${stages[@]}"; do
  printf 'V7_STRUCTURED_STAGE_START stage=%s source=%s out=%s device=%s\n' "$current" "$source" "$out" "$device"
  "$python" "$module/structured_campaign.py" "$current" "${common[@]}"
  printf 'V7_STRUCTURED_STAGE_COMPLETE stage=%s\n' "$current"
done
printf 'V7_STRUCTURED_GROUP_COMPLETE group=%s\n' "$stage"
