#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"
phase="${1:-status}"
out="${TAILCOMPILE_V68_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v68/runs/hier_init_001}"
mkdir -p "$out"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
stage=init
trap 'code=$?; echo "ERROR v68 phase=$phase stage=$stage line=$LINENO code=$code"; exit "$code"' ERR
if [[ "$phase" != status && "$phase" != parallel-status ]]; then
  exec 9>"$out/.lock"
  flock -n 9 || { echo "Another v6.8 stage is active in $out"; exit 1; }
fi
cpu() {
  stage="$1"
  local jobs="${TAILCOMPILE_V68_JOBS:-16}"
  local prepare_jobs="${TAILCOMPILE_V68_PREPARE_JOBS:-$jobs}"
  local rr_jobs="${TAILCOMPILE_V68_RR_JOBS:-$jobs}"
  local generation_jobs="${TAILCOMPILE_V68_GENERATION_JOBS:-$jobs}"
  extra=()
  [[ -n "${TAILCOMPILE_V68_DESIGNS:-}" ]] && read -r -a designs <<< "$TAILCOMPILE_V68_DESIGNS" && extra+=(--designs "${designs[@]}")
  [[ "${TAILCOMPILE_V68_RETRY_TIMEOUTS:-0}" == 1 ]] && extra+=(--retry-timeouts)
  [[ "${TAILCOMPILE_V68_RETRY_FAILED:-0}" == 1 ]] && extra+=(--retry-failed)
  [[ "${TAILCOMPILE_V68_CONFIRM_REMOTE_STOPPED:-0}" == 1 ]] && extra+=(--confirm-remote-stopped)
  [[ -n "${TAILCOMPILE_V68_CHECKPOINT:-}" ]] && extra+=(--checkpoint "$TAILCOMPILE_V68_CHECKPOINT")
  python3 -u experiments/vtr/tailcompile_v68/experiment.py "$1" --out "$out" \
    --cohort "${TAILCOMPILE_V68_COHORT:-experiments/vtr/tailcompile_v68/cohort.json}" \
    --jobs "$jobs" --prepare-jobs "$prepare_jobs" \
    --rr-jobs "$rr_jobs" --generation-jobs "$generation_jobs" \
    --tile-size "${TAILCOMPILE_V68_TILE_SIZE:-4}" \
    --timeout "${TAILCOMPILE_V68_TIMEOUT:-7200}" --heartbeat "${TAILCOMPILE_V68_HEARTBEAT:-60}" \
    --candidates "${TAILCOMPILE_V68_CANDIDATES:-8}" \
    --timing-name "${TAILCOMPILE_V68_TIMING_RUN:-serial_001}" --device cpu "${extra[@]}"
}
gpu() {
  stage="$1"
  extra=()
  target=pretrain
  if [[ "$1" == finetune ]]; then
    target=finetune
    extra+=(--initialize "$out/models/pretrain/generator.pt" --feedback "$out/feedback/feedback_dataset.json")
  fi
  python3 -u experiments/vtr/tailcompile_v68/train.py \
    --dataset "$out/dataset.json" --out "$out/models/$target" \
    --device "${TAILCOMPILE_V68_TRAIN_DEVICE:-cuda}" \
    --epochs "${TAILCOMPILE_V68_EPOCHS:-150}" \
    --tile-size "${TAILCOMPILE_V68_TILE_SIZE:-4}" \
    --hidden "${TAILCOMPILE_V68_HIDDEN:-64}" --chunk-size "${TAILCOMPILE_V68_CHUNK_SIZE:-512}" \
    --graph-cache "${TAILCOMPILE_V68_GRAPH_CACHE:-2}" "${extra[@]}"
}
parallel_timing() {
  stage="$1"
  extra=()
  [[ "${TAILCOMPILE_V68_RETRY_TIMEOUTS:-0}" == 1 ]] && extra+=(--retry-timeouts)
  [[ "${TAILCOMPILE_V68_RETRY_FAILED:-0}" == 1 ]] && extra+=(--retry-failed)
  [[ "${TAILCOMPILE_V68_CONFIRM_REMOTE_STOPPED:-0}" == 1 ]] && extra+=(--confirm-remote-stopped)
  [[ -n "${TAILCOMPILE_V68_TIMING_CPUS:-}" ]] && extra+=(--cpu-list "$TAILCOMPILE_V68_TIMING_CPUS")
  python3 -u experiments/vtr/tailcompile_v68/parallel_timing.py "$1" --out "$out" \
    --run-name "${TAILCOMPILE_V68_PARALLEL_TIMING_RUN:-paired_001}" \
    --timing-jobs "${TAILCOMPILE_V68_TIMING_JOBS:-16}" \
    --timeout "${TAILCOMPILE_V68_TIMEOUT:-7200}" \
    --heartbeat "${TAILCOMPILE_V68_HEARTBEAT:-60}" "${extra[@]}"
}
echo "TailCompile v6.8 phase=$phase host=$(hostname) out=$out"
case "$phase" in
  inventory|prepare|feedback|evaluate|timing|report|status|recover) cpu "$phase" ;;
  parallel-timing) parallel_timing run ;;
  parallel-status) parallel_timing status ;;
  parallel-report) parallel_timing report ;;
  train|finetune) gpu "$phase" ;;
  *) echo "CPU: inventory prepare feedback evaluate timing parallel-timing parallel-status parallel-report report status recover; GPU: train finetune"; exit 2 ;;
esac
echo "v6.8 phase=$phase finished (inspect status/report for scientific outcome)"
