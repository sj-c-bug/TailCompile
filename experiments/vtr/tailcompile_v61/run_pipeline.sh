#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
: "${VTR_ROOT:?set VTR_ROOT to server VTR checkout}"
cd "$TAILCOMPILE_REPO"
source_root="${TAILCOMPILE_V61_SOURCE:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001/confirm}"
out="${TAILCOMPILE_V61_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v61/runs/tile8_001}"
export PYTHONUNBUFFERED=1
trap 'code=$?; echo "ERROR stage=$stage line=$LINENO code=$code"; exit "$code"' ERR
stage=preflight
python3 -c 'import os, sys; sys.path.insert(0, "experiments/vtr"); from run_tailcompile_v4 import Backend; Backend(os.environ["VTR_ROOT"])'
stage=prepare
python3 experiments/vtr/prepare_tailcompile_v61.py --source "$source_root" \
  --out "$out/data" --tile-size "${TAILCOMPILE_V61_TILE_SIZE:-8}"
stage=train
python3 experiments/vtr/train_tailcompile_v61.py --bundle "$out/data/bundle.pt" \
  --out "$out/models" --epochs "${TAILCOMPILE_V61_EPOCHS:-60}"
stage=generate
python3 experiments/vtr/run_tailcompile_v61.py --bundle "$out/data/bundle.pt" \
  --models "$out/models" --out "$out/candidates" --budget "${TAILCOMPILE_V61_BUDGET:-64}"
stage=vpr
python3 experiments/vtr/run_tailcompile_v61.py --out "$out/candidates" --stage vpr --seeds 41 42
test -s "$out/candidates/vpr_results.json"
echo "TailCompile v6.1 complete: $out/candidates/vpr_results.json"
