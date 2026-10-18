#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"
phase="${1:-status}"
out="${TAILCOMPILE_V67_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v67/runs/direct_001}"
mkdir -p "$out"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "$phase" != status ]]; then
  exec 9>"$out/.lock"
  flock -n 9 || { echo "Another v6.7 phase is active"; exit 1; }
fi
stage=init
trap 'code=$?; echo "ERROR phase=$phase stage=$stage line=$LINENO code=$code"; exit "$code"' ERR
run() {
  stage="$1"
  python3 experiments/vtr/tailcompile_v67/direct.py "$1" --out "$out" \
    --jobs "${TAILCOMPILE_V67_JOBS:-8}" --timeout "${TAILCOMPILE_V67_TIMEOUT:-7200}" \
    --device "${TAILCOMPILE_V67_DEVICE:-cpu}" \
    --checkpoint "${TAILCOMPILE_GENERATOR_CHECKPOINT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v3_training/model_width_conditioned/ranker.pt}" \
    --inputs "${TAILCOMPILE_V67_INPUTS:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001/inputs/holdout_inputs.json}" \
    --screen "${TAILCOMPILE_V67_SCREEN:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001/screen/runs}"
}
case "$phase" in
  generate|run|timing|report|status) run "$phase" ;;
  evaluate) run run; run timing; run report ;;
  all) run generate; run run; run timing; run report ;;
  *) echo "usage: $0 generate|evaluate|all|status"; exit 2 ;;
esac
echo "v6.7 $phase complete"
