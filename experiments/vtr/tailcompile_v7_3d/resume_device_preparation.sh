#!/usr/bin/env bash
# Complete large RRG feature preparation without changing the frozen campaign.
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
export TAILCOMPILE_V7_OUT="${TAILCOMPILE_V7_PREPARATION_OUT:-${TAILCOMPILE_V7_CAPACITY_OUT:-$module/runs/development_004_capacity}}"
export TAILCOMPILE_V7_CONFIG="$TAILCOMPILE_V7_OUT/campaign.json"
export TAILCOMPILE_3D_QUALIFICATION="$TAILCOMPILE_V7_OUT/qualification"
export TAILCOMPILE_V7_TIMEOUT="${TAILCOMPILE_V7_TIMEOUT:-3600}"
export TAILCOMPILE_V7_JOBS="${TAILCOMPILE_V7_JOBS:-1}"
# XML parsing is Python work; independently limit it even when P&R uses many cores.
export TAILCOMPILE_V7_RRG_JOBS=1
budget="${TAILCOMPILE_V7_DEVICE_PREP_SECONDS:-3600}"
mode="${1:---background}"
case "$mode" in --background|--foreground) ;; *) echo 'Usage: bash resume_device_preparation.sh [--background | --foreground]' >&2; exit 2 ;; esac
entry="$module/prepare_server.sh"
bash "$entry" --check
for required in protocol.json campaign.json prepared.json qualification/qualification_report.json; do
    [[ -f "$TAILCOMPILE_V7_OUT/$required" ]] || { echo "Existing capacity cohort is not ready: $TAILCOMPILE_V7_OUT/$required" >&2; exit 2; }
done
if [[ "$mode" == --background ]]; then
    mkdir -p "$repo/logs"
    log="$repo/logs/v7_device_prepare_$(date +%Y%m%d_%H%M%S).log"
    nohup bash "$module/resume_device_preparation.sh" --foreground >"$log" 2>&1 </dev/null &
    printf 'PID=%s\nLOG=%s\nOUT=%s\n' "$!" "$log" "$TAILCOMPILE_V7_OUT"
    exit 0
fi
scratch="$repo/vtr_work/tailcompile_v7_3d"
mkdir -p "$scratch/tmp" "$scratch/cache"
export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp" XDG_CACHE_HOME="$scratch/cache"
[[ "$(realpath -m "$scratch/.capacity-recovery.lock")" == "$scratch/.capacity-recovery.lock" ]] || { echo 'Recovery lock resolves outside its owned path.' >&2; exit 2; }
exec 9>"$scratch/.capacity-recovery.lock"
flock -n 9 || { echo 'Another capacity/preparation workflow is running; no duplicate started.' >&2; exit 2; }
"$TAILCOMPILE_3D_PYTHON" "$module/resume_device_preparation.py" \
    --out "$TAILCOMPILE_V7_OUT" --entry "$entry" --max-seconds "$budget"
