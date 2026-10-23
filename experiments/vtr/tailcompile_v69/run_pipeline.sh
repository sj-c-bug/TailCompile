#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"
phase="${1:-status}"
out="${TAILCOMPILE_V69_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v69/runs/runtime_001}"
source_out="${TAILCOMPILE_V68_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v68/runs/hier_init_001}"
mkdir -p "$out" "$out/tmp"
export TMPDIR="$out/tmp" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TAILCOMPILE_V68_MIN_FREE_GIB="${TAILCOMPILE_V69_MIN_FREE_GIB:-10}"
trap 'code=$?; echo "ERROR v69 phase=$phase line=$LINENO code=$code"; exit "$code"' ERR
if [[ "$phase" != status ]]; then
  exec 9>"$out/.lock"
  flock -n 9 || { echo "Another v69 task owns $out/.lock"; exit 1; }
fi
echo "TailCompile v6.9 phase=$phase host=$(hostname) out=$out"
if [[ "$phase" == finetune ]]; then
  python3 -u experiments/vtr/tailcompile_v69/train_runtime.py \
    --source "$source_out" --out "$out" \
    --device "${TAILCOMPILE_V69_TRAIN_DEVICE:-cuda}" \
    --epochs "${TAILCOMPILE_V69_EPOCHS:-50}"
else
  extra=()
  [[ -n "${TAILCOMPILE_V69_INITIAL_CHECKPOINT:-}" ]] && extra+=(--initial-checkpoint "$TAILCOMPILE_V69_INITIAL_CHECKPOINT")
  [[ -n "${TAILCOMPILE_V69_CHECKPOINT:-}" ]] && extra+=(--checkpoint "$TAILCOMPILE_V69_CHECKPOINT")
  [[ -n "${TAILCOMPILE_V69_TIMING_CPUS:-}" ]] && extra+=(--cpu-list "$TAILCOMPILE_V69_TIMING_CPUS")
  [[ "${TAILCOMPILE_V69_RETRY_TIMEOUTS:-0}" == 1 ]] && extra+=(--retry-timeouts)
  [[ "${TAILCOMPILE_V69_RETRY_FAILED:-0}" == 1 ]] && extra+=(--retry-failed)
  [[ "${TAILCOMPILE_V69_CONFIRM_REMOTE_STOPPED:-0}" == 1 ]] && extra+=(--confirm-remote-stopped)
  jobs="${TAILCOMPILE_V69_JOBS:-16}"
  python3 -u experiments/vtr/tailcompile_v69/run.py "$phase" \
    --out "$out" --source "$source_out" --jobs "$jobs" \
    --generation-jobs "${TAILCOMPILE_V69_GENERATION_JOBS:-$jobs}" \
    --timing-jobs "${TAILCOMPILE_V69_TIMING_JOBS:-$jobs}" \
    --run-name "${TAILCOMPILE_V69_TIMING_RUN:-paired_001}" \
    --timeout "${TAILCOMPILE_V69_TIMEOUT:-7200}" \
    --heartbeat "${TAILCOMPILE_V69_HEARTBEAT:-60}" \
    --candidates "${TAILCOMPILE_V69_CANDIDATES:-4}" \
    --soft-density-passes "${TAILCOMPILE_V69_DENSITY_PASSES:-2}" \
    --density-price-weight "${TAILCOMPILE_V69_DENSITY_WEIGHT:-1}" \
    --width-budget "${TAILCOMPILE_V69_WIDTH_BUDGET:-0.03}" \
    --wire-budget "${TAILCOMPILE_V69_WIRE_BUDGET:-0.03}" \
    --cpd-budget "${TAILCOMPILE_V69_CPD_BUDGET:-0.03}" \
    --time-gain "${TAILCOMPILE_V69_TIME_GAIN:-0.05}" "${extra[@]}"
fi
echo "v69 phase=$phase finished; completion is not a scientific pass (inspect reports)"
