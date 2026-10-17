#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"
phase="${1:-status}"
case "$phase" in probe|prepare|train|evaluate|status) ;; *) echo "usage: $0 probe|prepare|train|evaluate|status"; exit 2;; esac
out="${TAILCOMPILE_V63_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v63/runs/expanded_001}"
v61="${TAILCOMPILE_V61_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v61/runs/tile8_001}"
v62="${TAILCOMPILE_V62_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v62/runs/intervention_001}"
jobs="${TAILCOMPILE_V63_JOBS:-4}"
export PYTHONUNBUFFERED=1
stage=initialization
trap 'code=$?; echo "ERROR phase=$phase stage=$stage line=$LINENO code=$code"; exit "$code"' ERR

if [[ "$phase" == status ]]; then
  python3 experiments/vtr/check_tailcompile_v63.py --out "$out"
  exit
fi

if [[ "$phase" == probe ]]; then
  : "${VTR_ROOT:?set VTR_ROOT for VPR runs}"
  stage=probe
  python3 experiments/vtr/run_tailcompile_v63.py --stage probe \
    --v62-pool "$v62/pool/pool_report.json" \
    --v62-pilot "$v62/pilot/vpr_results.json" \
    --out "$out/probe" --jobs "$jobs"
  python3 experiments/vtr/check_tailcompile_v63.py --out "$out" --require probe
fi

if [[ "$phase" == prepare ]]; then
  : "${VTR_ROOT:?set VTR_ROOT for VPR labeling}"
  stage=pool
  python3 experiments/vtr/run_tailcompile_v63.py --stage pool \
    --v61-bundle "$v61/data/bundle.pt" --out "$out/pool"
  stage=labels
  python3 experiments/vtr/run_tailcompile_v63.py --stage label \
    --pool "$out/pool" --v62-pool "$v62/pool/pool_report.json" \
    --v62-labels "$v62/labels/label_results.json" \
    --out "$out/labels" --jobs "$jobs"
  stage=dataset
  python3 experiments/vtr/run_tailcompile_v63.py --stage dataset \
    --v61-bundle "$v61/data/bundle.pt" --pool "$out/pool" \
    --labels "$out/labels" --out "$out/data"
  python3 experiments/vtr/check_tailcompile_v63.py --out "$out" --require prepare
fi

if [[ "$phase" == train ]]; then
  stage=gpu_preflight
  python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
  stage=training
  python3 experiments/vtr/run_tailcompile_v63.py --stage train \
    --bundle "$out/data/bundle.pt" --out "$out/models" --device cuda \
    --epochs "${TAILCOMPILE_V63_EPOCHS:-100}" --members 3
  python3 experiments/vtr/check_tailcompile_v63.py --out "$out" --require train
fi

if [[ "$phase" == evaluate ]]; then
  : "${VTR_ROOT:?set VTR_ROOT for VPR evaluation}"
  stage=selection
  python3 experiments/vtr/run_tailcompile_v63.py --stage select \
    --bundle "$out/data/bundle.pt" --pool "$out/pool" \
    --models "$out/models" --out "$out/selection" --members 3 --device cpu
  stage=fresh_seed_vpr
  python3 experiments/vtr/run_tailcompile_v63.py --stage evaluate \
    --selection "$out/selection" --out "$out/evaluation" --jobs "$jobs"
  python3 experiments/vtr/check_tailcompile_v63.py --out "$out" --require evaluate
fi
