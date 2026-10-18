#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"

phase="${1:-status}"
case "$phase" in plan|run|summarize|status) ;; *) echo "usage: $0 plan|run|summarize|status"; exit 2 ;; esac
out="${TAILCOMPILE_V64_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v64/runs/wmin_001}"
selection="${TAILCOMPILE_V63_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v63/runs/expanded_001}/selection/selection_report.json"
jobs="${TAILCOMPILE_V64_JOBS:-4}"
timeout="${TAILCOMPILE_V64_TIMEOUT:-1800}"
export PYTHONUNBUFFERED=1
stage=initialization
trap 'code=$?; echo "ERROR phase=$phase stage=$stage line=$LINENO code=$code"; exit "$code"' ERR

if [[ "$phase" == plan ]]; then
  stage=freeze_width_sweep
  python3 experiments/vtr/run_tailcompile_v64_wmin.py plan \
    --selection "$selection" --out "$out"
elif [[ "$phase" == run ]]; then
  : "${VTR_ROOT:?set VTR_ROOT for the pinned server VTR checkout}"
  stage=paired_vpr_sweep
  mkdir -p "$out"
  exec 9>"$out/.vpr.lock"
  flock -n 9 || { echo "another v6.4 sweep is active: $out"; exit 1; }
  python3 experiments/vtr/run_tailcompile_v64_wmin.py run \
    --out "$out" --jobs "$jobs" --timeout "$timeout"
else
  stage="$phase"
  python3 experiments/vtr/run_tailcompile_v64_wmin.py "$phase" --out "$out"
fi
