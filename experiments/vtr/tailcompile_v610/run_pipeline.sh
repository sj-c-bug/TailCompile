#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"
phase="${1:-status}"
out="${TAILCOMPILE_V610_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v610/runs/koios_sweep_001}"
source_out="${TAILCOMPILE_V68_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v68/runs/hier_init_001}"
model_out="${TAILCOMPILE_V69_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v69/runs/runtime_001}"
checkpoint="${TAILCOMPILE_V610_CHECKPOINT:-$model_out/models/runtime/generator.pt}"
mkdir -p "$out/tmp"
export TMPDIR="$out/tmp" PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TAILCOMPILE_V68_MIN_FREE_GIB="${TAILCOMPILE_V610_MIN_FREE_GIB:-30}"
export TAILCOMPILE_V610_PRUNE_COMPLETED="${TAILCOMPILE_V610_PRUNE_COMPLETED:-1}"
trap 'code=$?; echo "ERROR v610 phase=$phase line=$LINENO code=$code"; exit "$code"' ERR
if [[ "$phase" != status ]]; then
  exec 9>"$out/.lock"
  flock -n 9 || { echo "Another task owns $out/.lock"; exit 1; }
fi
extra=()
[[ -n "${TAILCOMPILE_V610_TIMING_CPUS:-}" ]] && extra+=(--cpu-list "$TAILCOMPILE_V610_TIMING_CPUS")
if [[ -n "${TAILCOMPILE_V610_DESIGNS:-}" ]]; then
  read -r -a designs <<< "$TAILCOMPILE_V610_DESIGNS"
  extra+=(--designs "${designs[@]}")
fi
[[ "${TAILCOMPILE_V610_RETRY_FAILED:-0}" == 1 ]] && extra+=(--retry-failed)
[[ "${TAILCOMPILE_V610_RETRY_TIMEOUTS:-0}" == 1 ]] && extra+=(--retry-timeouts)
[[ "${TAILCOMPILE_V610_CONFIRM_REMOTE_STOPPED:-0}" == 1 ]] && extra+=(--confirm-remote-stopped)
jobs="${TAILCOMPILE_V610_JOBS:-32}"
echo "TailCompile v6.10 phase=$phase host=$(hostname) out=$out frozen_checkpoint=$checkpoint"
python3 -u experiments/vtr/tailcompile_v610/run.py "$phase" \
  --out "$out" --source "$source_out" --checkpoint "$checkpoint" \
  --scope "${TAILCOMPILE_V610_SCOPE:-all}" \
  --jobs "$jobs" --prepare-jobs "${TAILCOMPILE_V610_PREPARE_JOBS:-$jobs}" \
  --rr-jobs "${TAILCOMPILE_V610_RR_JOBS:-$jobs}" \
  --generation-jobs "${TAILCOMPILE_V610_GENERATION_JOBS:-$jobs}" \
  --timing-jobs "${TAILCOMPILE_V610_TIMING_JOBS:-$jobs}" \
  --run-name "${TAILCOMPILE_V610_TIMING_RUN:-preloaded_001}" \
  --timeout "${TAILCOMPILE_V610_TIMEOUT:-7200}" \
  --setup-timeout "${TAILCOMPILE_V610_SETUP_TIMEOUT:-1800}" \
  --heartbeat "${TAILCOMPILE_V610_HEARTBEAT:-60}" \
  --tile-size "${TAILCOMPILE_V610_TILE_SIZE:-4}" \
  --width-start "${TAILCOMPILE_V610_WIDTH_START:-48}" \
  --width-max "${TAILCOMPILE_V610_WIDTH_MAX:-512}" \
  --width-step "${TAILCOMPILE_V610_WIDTH_STEP:-4}" \
  --half-band "${TAILCOMPILE_V610_HALF_BAND:-3}" \
  --decoder "${TAILCOMPILE_V610_DECODER:-balanced}" \
  --repair-passes "${TAILCOMPILE_V610_REPAIR_PASSES:-2}" \
  --repair-topk "${TAILCOMPILE_V610_REPAIR_TOPK:-32}" "${extra[@]}"
echo "v610 phase=$phase finished; inspect status/reports for coverage and scientific results"
