#!/usr/bin/env bash
# Resume the capacity-qualified Default calibration; existing backends are read-only.
set -eEuo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1 GIT_OPTIONAL_LOCKS=0 PYTHONUNBUFFERED=1
module="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module/../../.." && pwd -P)"
workspace="$(dirname -- "$repo")"
export TAILCOMPILE_REPO="$repo"
export TAILCOMPILE_3D_PYTHON="${TAILCOMPILE_3D_PYTHON:-$(command -v python3)}"
export TAILCOMPILE_3D_TRAIN_DEVICE=cpu
export TAILCOMPILE_3D_VTR_ROOT="${TAILCOMPILE_3D_VTR_ROOT:-$workspace/vtr_tailcompile_3d_v7/vtr-verilog-to-routing}"
export TAILCOMPILE_3D_BACKEND_READ_ONLY=1
previous="${TAILCOMPILE_V7_CAPACITY_PREVIOUS:-$module/runs/development_003_paths}"
# Deliberately replace an inherited development_003 OUT/CONFIG/QUALIFICATION.
export TAILCOMPILE_V7_OUT="${TAILCOMPILE_V7_CAPACITY_OUT:-$module/runs/development_004_capacity}"
export TAILCOMPILE_V7_CONFIG="$TAILCOMPILE_V7_OUT/campaign.json"
export TAILCOMPILE_3D_QUALIFICATION="$TAILCOMPILE_V7_OUT/qualification"
export TAILCOMPILE_V7_TIMEOUT="${TAILCOMPILE_V7_TIMEOUT:-3600}"
export TAILCOMPILE_V7_JOBS="${TAILCOMPILE_V7_JOBS:-1}"
export TAILCOMPILE_V7_RRG_JOBS="${TAILCOMPILE_V7_RRG_JOBS:-1}"
entry="$module/prepare_server.sh"
mode="${1:---background}"
case "$mode" in --background|--foreground) ;; *) echo 'Usage: bash recover_capacity.sh [--background | --foreground]' >&2; exit 2 ;; esac
# Validates all resolved output, log, backend and temporary paths before mkdir/redirection.
bash "$entry" --check
if [[ "$mode" == --background ]]; then
    mkdir -p "$repo/logs"
    log="$repo/logs/v7_capacity_$(date +%Y%m%d_%H%M%S).log"
    nohup bash "$module/recover_capacity.sh" --foreground >"$log" 2>&1 </dev/null &
    printf 'PID=%s\nLOG=%s\nOUT=%s\n' "$!" "$log" "$TAILCOMPILE_V7_OUT"
    exit 0
fi
scratch="$repo/vtr_work/tailcompile_v7_3d"
mkdir -p "$scratch/tmp" "$scratch/cache"
export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache"
exec 9>"$scratch/.capacity-recovery.lock"
flock -n 9 || { echo 'Another capacity recovery is already running; no duplicate workflow started.' >&2; exit 2; }
"$TAILCOMPILE_3D_PYTHON" "$module/recover_capacity.py" \
    --previous-out "$previous" --out "$TAILCOMPILE_V7_OUT" --backend "$TAILCOMPILE_3D_VTR_ROOT"
bash "$entry" --stage plan
bash "$entry" --stage prepare
bash "$entry" --stage calibrate
printf '%s\n' 'V7_CAPACITY_BASELINE_FINISHED_REVIEW_BOUNDARIES' \
    "Inspect $TAILCOMPILE_V7_OUT/boundaries.json before collecting teachers."
