#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"
phase="${1:-status}"
out="${TAILCOMPILE_V67_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v67/runs/abcd_001}"
mkdir -p "$out"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
if [[ "$phase" != status && "$phase" != parallel-status && "$phase" != stop && "$phase" != scan-report ]]; then
  exec 9>"$out/.lock"
  flock -n 9 || { echo "Another v6.7 phase is active"; exit 1; }
fi
stage=init
trap 'code=$?; echo "ERROR phase=$phase stage=$stage line=$LINENO code=$code"; exit "$code"' ERR
run() {
  stage="$1"
  extra=()
  [[ "${TAILCOMPILE_V67_RETRY_TIMEOUTS:-0}" == 1 ]] && extra+=(--retry-timeouts)
  [[ "${TAILCOMPILE_V67_RETRY_FAILED:-0}" == 1 ]] && extra+=(--retry-failed)
  [[ "${TAILCOMPILE_V67_TIME_SCAN_TIMEOUTS:-0}" == 1 ]] && extra+=(--time-scan-timeouts)
  [[ "${TAILCOMPILE_V67_REPLAN_TIMING:-0}" == 1 ]] && extra+=(--replan-timing)
  python3 -u experiments/vtr/tailcompile_v67/resume.py "$1" --out "$out" "${extra[@]}" \
    --heartbeat "${TAILCOMPILE_V67_HEARTBEAT:-120}" \
    --timing-seed-count "${TAILCOMPILE_V67_TIMING_SEEDS:-3}" \
    --timing-width-count "${TAILCOMPILE_V67_TIMING_WIDTHS:-1}" \
    --jobs "${TAILCOMPILE_V67_JOBS:-8}" --timeout "${TAILCOMPILE_V67_TIMEOUT:-7200}" \
    --device "${TAILCOMPILE_V67_DEVICE:-cpu}" \
    --checkpoint "${TAILCOMPILE_GENERATOR_CHECKPOINT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v3_training/model_width_conditioned/ranker.pt}" \
    --inputs "${TAILCOMPILE_V67_INPUTS:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001/inputs/holdout_inputs.json}" \
    --screen "${TAILCOMPILE_V67_SCREEN:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001/screen/runs}"
}
run_parallel() {
  stage="$1"
  parallel_extra=()
  [[ -n "${TAILCOMPILE_V67_TIMING_CPUS:-}" ]] && parallel_extra+=(--cpu-list "$TAILCOMPILE_V67_TIMING_CPUS")
  python3 -u experiments/vtr/tailcompile_v67/parallel_runtime.py "$1" --out "$out" \
    --run-name "${TAILCOMPILE_V67_PARALLEL_RUN:-c${TAILCOMPILE_V67_TIMING_JOBS:-4}_001}" \
    --concurrency "${TAILCOMPILE_V67_TIMING_JOBS:-4}" \
    --timing-seed-count "${TAILCOMPILE_V67_TIMING_SEEDS:-3}" \
    --timing-width-count "${TAILCOMPILE_V67_TIMING_WIDTHS:-1}" \
    --timeout "${TAILCOMPILE_V67_TIMEOUT:-7200}" --heartbeat "${TAILCOMPILE_V67_HEARTBEAT:-120}" \
    "${parallel_extra[@]}"
}
case "$phase" in
  generate|run|scan-report|timing-plan|timing|report|status|stop) run "$phase" ;;
  evaluate) run run; run scan-report; run timing-plan; run timing; run report ;;
  all) run generate; run run; run scan-report; run timing-plan; run timing; run report ;;
  parallel-plan) run_parallel plan ;;
  parallel-timing) run_parallel run ;;
  parallel-report) run_parallel report ;;
  parallel-status) run_parallel status ;;
  parallel-evaluate) run_parallel plan; run_parallel run; run_parallel report ;;
  *) echo "usage: $0 generate|evaluate|scan-report|timing-plan|timing|report|all|status|stop|parallel-plan|parallel-evaluate|parallel-status|parallel-report"; exit 2 ;;
esac
echo "v6.7 $phase complete"
