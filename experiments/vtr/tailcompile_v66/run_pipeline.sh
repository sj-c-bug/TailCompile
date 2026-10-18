#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"
phase="${1:-status}"
out="${TAILCOMPILE_V66_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v66/runs/gp_001}"
base=experiments/vtr/tailcompile_v66
mkdir -p "$out"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${TAILCOMPILE_V66_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
stage=initialization
trap 'code=$?; echo "ERROR phase=$phase stage=$stage line=$LINENO code=$code"; exit "$code"' ERR
if [[ "$phase" != status ]]; then
  exec 9>"$out/.campaign.lock"
  flock -n 9 || { echo "Another phase holds the campaign lock: $out"; exit 1; }
fi
cpu() {
  stage="$1"
  python3 "$base/campaign.py" "$1" --out "$out" \
    --jobs "${TAILCOMPILE_V66_JOBS:-4}" \
    --prepare-jobs "${TAILCOMPILE_V66_PREPARE_JOBS:-2}" \
    --vpr-threads "$OMP_NUM_THREADS" \
    --timeout "${TAILCOMPILE_V66_TIMEOUT:-7200}" "${@:2}"
}
case "$phase" in
  prepare)
    cpu prepare
    cpu screen
    cpu pool
    cpu label
    ;;
  train)
    stage=gpu_train
    python3 "$base/learn.py" train --out "$out" --device cuda --epochs "${TAILCOMPILE_V66_EPOCHS:-80}"
    ;;
  evaluate)
    stage=freeze_cpu_selection
    python3 "$base/learn.py" select --out "$out" --device cpu
    cpu evaluate
    cpu deploy
    cpu timing
    cpu report
    ;;
  screen|pool|label|deploy|timing|report|status) cpu "$phase" ;;
  recover) cpu recover --workers-stopped ;;
  *) echo "usage: $0 prepare|train|evaluate|status|recover (also screen/pool/label/deploy/timing/report)"; exit 2 ;;
esac
echo "TailCompile v6.6 phase=$phase complete"
