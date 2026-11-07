#!/usr/bin/env bash
# v8_015 uses the existing qualified 3D backend and the active Python environment.
set -eEuo pipefail
ulimit -c 0
stage="${1:---help}"
mode="${2:---foreground}"
if [[ "$stage" == --help || "$stage" == -h ]]; then
  echo 'Usage: bash run_attribution_campaign.sh plan|check|prepare|predict|compare|report|all [--foreground|--background]'
  exit 0
fi
case "$stage" in plan|check|prepare|predict|compare|report|all) ;; *) echo "Unknown v8_015 stage: $stage" >&2; exit 2 ;; esac
case "$mode" in --foreground|--background) ;; *) echo "Unknown mode: $mode" >&2; exit 2 ;; esac
[[ $# -le 2 ]] || { echo 'Too many arguments.' >&2; exit 2; }
case "$stage:$mode" in plan:--background|check:--background)
  echo 'Use a foreground process for read-only checks.' >&2; exit 2 ;;
esac
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 GIT_OPTIONAL_LOCKS=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module/../../.." && pwd -P)"
[[ "$module" == "$repo/experiments/vtr/tailcompile_v7_3d" && -f "$repo/REPOSITORY_WORKFLOW.md" ]] || {
  echo 'Use the canonical TailCompile checkout.' >&2; exit 2;
}
python="${TAILCOMPILE_3D_PYTHON:-python3}"
command -v -- "$python" >/dev/null || { echo 'Active Python is unavailable.' >&2; exit 2; }
out="${TAILCOMPILE_V8_015_OUT:-$module/runs/development_015_attribution}"
alias_root="${TAILCOMPILE_REPO:-$repo}"
alias_root="${alias_root%/}"
if [[ "$alias_root" != "$repo" && "$out" == "$alias_root/"* && "$(realpath -m -- "$alias_root")" == "$repo" ]]; then
  out="$repo/${out#"$alias_root/"}"
fi
[[ "$(dirname -- "$out")" == "$module/runs" && "${out##*/}" == development_015_?* ]] || {
  echo 'Output must be a dedicated development_015_ cohort.' >&2; exit 2;
}
logs="$repo/logs/v8_015"
scratch="$repo/vtr_work/tailcompile_v7_3d/v8_015/${out##*/}"
export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/cache" TORCH_HOME="$scratch/cache/torch" MPLCONFIGDIR="$scratch/cache/matplotlib"
export TORCH_EXTENSIONS_DIR="$scratch/cache/torch_extensions" CUDA_CACHE_PATH="$scratch/cache/cuda" TRITON_CACHE_DIR="$scratch/cache/triton"
for owned in "$logs" "$out" "$out/.campaign-stage.lock" "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$MPLCONFIGDIR" \
             "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"; do
  [[ "$(realpath -m -- "$owned")" == "$owned" ]] || { echo "Redirected v8_015 write path: $owned" >&2; exit 2; }
done
jobs="${TAILCOMPILE_V8_015_JOBS:-16}"
job_mib="${TAILCOMPILE_V8_015_JOB_MIB:-8192}"
rrg_jobs="${TAILCOMPILE_V8_015_RRG_JOBS:-2}"
timeout="${TAILCOMPILE_V8_015_TIMEOUT:-3600}"
for n in "$jobs" "$job_mib" "$rrg_jobs" "$timeout"; do
  [[ "$n" =~ ^[1-9][0-9]*$ ]] || { echo 'Invalid v8_015 count/budget.' >&2; exit 2; }
done
opts=(--out "$out" --jobs "$jobs" --per-job-mib "$job_mib" --rrg-jobs "$rrg_jobs" --timeout "$timeout")
[[ -z "${TAILCOMPILE_V8_015_MANIFEST:-}" ]] || opts+=(--manifest "$TAILCOMPILE_V8_015_MANIFEST")
[[ -z "${TAILCOMPILE_V8_015_BACKEND_MANIFEST:-}" ]] || opts+=(--backend-manifest "$TAILCOMPILE_V8_015_BACKEND_MANIFEST")
if [[ "$mode" == --background ]]; then
  mkdir -p -- "$logs"
  log="$(mktemp "$logs/v8_015_${stage}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  nohup bash "$module/run_attribution_campaign.sh" "$stage" --foreground >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$log.pid"
  printf 'PID=%s\nLOG=%s\nPIDFILE=%s\nOUT=%s\n' "$pid" "$log" "$log.pid" "$out"
  exit 0
fi
case "$stage" in plan|check) ;; *) mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" ;; esac
exec "$python" "$module/attribution_campaign.py" "$stage" "${opts[@]}"
