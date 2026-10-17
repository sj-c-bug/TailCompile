#!/usr/bin/env bash
set -eEuo pipefail

: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"

phase="${1:-all}"
case "$phase" in
  prepare|train|evaluate|all) ;;
  *) echo "usage: $0 [prepare|train|evaluate|all]" >&2; exit 2 ;;
esac

source_root="${TAILCOMPILE_V61_SOURCE:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001/confirm}"
out="${TAILCOMPILE_V61_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v61/runs/tile8_001}"
export PYTHONUNBUFFERED=1
stage=initialization
trap 'code=$?; echo "ERROR phase=$phase stage=$stage line=$LINENO code=$code"; exit "$code"' ERR

vtr_preflight() {
  : "${VTR_ROOT:?set VTR_ROOT to the server VTR checkout for CPU phases}"
  python3 -c 'import os, sys; sys.path.insert(0, "experiments/vtr"); from run_tailcompile_v4 import Backend; Backend(os.environ["VTR_ROOT"])'
}

prepare() {
  echo "TailCompile v6.1 phase=prepare host=$(hostname) out=$out"
  stage=prepare_preflight
  vtr_preflight
  stage=prepare
  python3 experiments/vtr/prepare_tailcompile_v61.py --source "$source_root" \
    --out "$out/data" --tile-size "${TAILCOMPILE_V61_TILE_SIZE:-8}"
  test -s "$out/data/bundle.pt"
  test -s "$out/data/prepare_report.json"
  echo "TailCompile v6.1 CPU prepare complete: $out/data/bundle.pt"
}

train() {
  echo "TailCompile v6.1 phase=train host=$(hostname) out=$out"
  stage=train_preflight
  test -s "$out/data/bundle.pt"
  train_device="${TAILCOMPILE_V61_TRAIN_DEVICE:-cuda}"
  if [[ "$train_device" == cuda* ]]; then
    python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA requested but unavailable"; print({"stage":"gpu_preflight","device":torch.cuda.get_device_name(0)})'
  fi
  stage=train
  python3 experiments/vtr/train_tailcompile_v61.py --bundle "$out/data/bundle.pt" \
    --out "$out/models" --epochs "${TAILCOMPILE_V61_EPOCHS:-60}" \
    --device "$train_device"
  test -s "$out/models/report.json"
  echo "TailCompile v6.1 GPU train complete: $out/models/report.json"
}

evaluate() {
  echo "TailCompile v6.1 phase=evaluate host=$(hostname) out=$out"
  stage=evaluate_preflight
  vtr_preflight
  test -s "$out/data/bundle.pt"
  test -s "$out/models/report.json"
  stage=generate
  python3 experiments/vtr/run_tailcompile_v61.py --bundle "$out/data/bundle.pt" \
    --models "$out/models" --out "$out/candidates" \
    --budget "${TAILCOMPILE_V61_BUDGET:-64}" --device cpu
  stage=vpr
  python3 experiments/vtr/run_tailcompile_v61.py --out "$out/candidates" \
    --stage vpr --seeds 41 42 --jobs "${TAILCOMPILE_V61_JOBS:-1}" \
    --recover-interrupted
  stage=acceptance
  python3 experiments/vtr/check_tailcompile_v61.py --out "$out" --require-complete
  echo "TailCompile v6.1 CPU evaluation complete: $out/candidates/vpr_results.json"
}

case "$phase" in
  prepare) prepare ;;
  train) train ;;
  evaluate) evaluate ;;
  all) prepare; train; evaluate ;;
esac
