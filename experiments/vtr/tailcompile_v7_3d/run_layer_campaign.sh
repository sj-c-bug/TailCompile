#!/usr/bin/env bash
# 007: explicit layers. Uses the active Python environment; never activates conda.
set -eEuo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
module="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module/../../.." && pwd -P)"
stage="${1:---help}"
mode="${2:---foreground}"
case "$stage" in
  --help|-h) printf '%s\n' 'Usage: bash run_layer_campaign.sh bootstrap|plan|check|prepare|pretrain|feedback|train|validate|compare|report|evaluate|all [--foreground|--background]' 'all builds/qualifies the independent backend, then runs the registered research stages. Uses the active environment.'; exit 0 ;;
  bootstrap|plan|check|prepare|pretrain|feedback|train|validate|compare|report) stages=("$stage") ;;
  evaluate) stages=(validate compare report) ;;
  all) stages=(bootstrap check prepare pretrain feedback train validate compare report) ;;
  *) printf 'Unknown stage: %s\n' "$stage" >&2; exit 2 ;;
esac
case "$mode" in --foreground|--background) ;; *) echo 'Unknown launch mode' >&2; exit 2 ;; esac
python="${TAILCOMPILE_3D_PYTHON:-python3}"
command -v -- "$python" >/dev/null
out="${TAILCOMPILE_V7_LAYER_OUT:-$module/runs/development_007_layer}"
backend="${TAILCOMPILE_V7_LAYER_BACKEND:-$repo/.deps/tailcompile_v7_3d_layer/vtr}"
qualification="${TAILCOMPILE_V7_LAYER_QUALIFICATION:-$out/qualification/qualification_report.json}"
# Resolve before log/mkdir/redirection. A symlink may not redirect new writes.
for owned in "$out" "$repo/logs" "$repo/vtr_work/tailcompile_v7_3d/layers"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || { echo "Redirected write root: $owned" >&2; exit 2; }
done
case "$out/" in "$module/runs/"?*) ;; *) echo 'New output must be inside the 3D runs namespace' >&2; exit 2 ;; esac
if [[ "$mode" == --background ]]; then
  mkdir -p -- "$repo/logs"
  log="$(mktemp "$repo/logs/v7_layer_${stage}_$(date +%Y%m%d_%H%M%S)_XXXXXX.log")"
  nohup bash "$module/run_layer_campaign.sh" "$stage" --foreground >"$log" 2>&1 </dev/null &
  pid=$!
  printf '%s\n' "$pid" >"$log.pid"
  printf 'PID=%s\nLOG=%s\nOUT=%s\n' "$pid" "$log" "$out"
  exit 0
fi
scratch="$repo/vtr_work/tailcompile_v7_3d/layers"
export TMPDIR="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache"
export TORCH_HOME="$scratch/cache/torch" MPLCONFIGDIR="$scratch/cache/matplotlib"
mkdir -p -- "$TMPDIR" "$XDG_CACHE_HOME" "$out"
exec 9>"$out/.layer-launch.lock"
flock -n 9 || { echo 'Another 007 group is running' >&2; exit 2; }
for current in "${stages[@]}"; do
  printf 'V7_LAYER_STAGE_START stage=%s\n' "$current"
  if [[ "$current" == bootstrap ]]; then
    bash "$module/prepare_layer_backend.sh"
    printf 'V7_LAYER_STAGE_COMPLETE stage=%s\n' "$current"
    continue
  fi
  "$python" "$module/layer_campaign.py" "$current" --out "$out" --backend "$backend" \
    --qualification "$qualification" --jobs "${TAILCOMPILE_V7_JOBS:-4}" \
    --per-job-mib "${TAILCOMPILE_V7_JOB_MIB:-4096}" --timeout "${TAILCOMPILE_V7_TIMEOUT:-3600}" \
    --device "${TAILCOMPILE_3D_TRAIN_DEVICE:-cpu}"
  printf 'V7_LAYER_STAGE_COMPLETE stage=%s\n' "$current"
done
