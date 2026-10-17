#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"
phase="${1:-status}"
case "$phase" in prepare|train|pilot|full|status) ;; *) echo "usage: $0 prepare|train|pilot|full|status"; exit 2;; esac
out="${TAILCOMPILE_V62_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v62/runs/intervention_001}"
v61="${TAILCOMPILE_V61_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v61/runs/tile8_001}"
jobs="${TAILCOMPILE_V62_JOBS:-2}"
export TAILCOMPILE_V62_OUT="$out"
export PYTHONUNBUFFERED=1
stage=initialization
trap 'code=$?; echo "ERROR phase=$phase stage=$stage line=$LINENO code=$code"; exit "$code"' ERR

if [[ "$phase" == status ]]; then
  python3 experiments/vtr/check_tailcompile_v62.py --out "$out"
  exit
fi

if [[ "$phase" == prepare ]]; then
  : "${VTR_ROOT:?set VTR_ROOT for CPU VPR labeling}"
  stage=pool
  python3 experiments/vtr/run_tailcompile_v62.py --stage pool \
    --v61-bundle "$v61/data/bundle.pt" --out "$out/pool"
  stage=label
  python3 experiments/vtr/run_tailcompile_v62.py --stage label \
    --pool "$out/pool" --out "$out/labels" --jobs "$jobs"
  stage=dataset
  python3 experiments/vtr/run_tailcompile_v62.py --stage dataset \
    --v61-bundle "$v61/data/bundle.pt" --pool "$out/pool" \
    --labels "$out/labels" --out "$out/data"
  python3 experiments/vtr/check_tailcompile_v62.py --out "$out" --require prepare
fi

if [[ "$phase" == train ]]; then
  stage=gpu_preflight
  python3 -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print(torch.cuda.get_device_name(0))'
  stage=train
  python3 experiments/vtr/run_tailcompile_v62.py --stage train \
    --bundle "$out/data/bundle.pt" --out "$out/models" --device cuda \
    --epochs "${TAILCOMPILE_V62_EPOCHS:-80}" --members 3
  python3 experiments/vtr/check_tailcompile_v62.py --out "$out" --require train
fi

if [[ "$phase" == pilot ]]; then
  : "${VTR_ROOT:?set VTR_ROOT for CPU pilot evaluation}"
  stage=select
  python3 experiments/vtr/run_tailcompile_v62.py --stage select \
    --bundle "$out/data/bundle.pt" --pool "$out/pool" --models "$out/models" \
    --out "$out/selection" --device cpu --members 3
  stage=pilot_vpr
  python3 experiments/vtr/run_tailcompile_v62.py --stage evaluate \
    --selection "$out/selection" --out "$out/pilot" --jobs "$jobs" \
    --designs conv_layer eltwise_layer lenet lstm
  python3 experiments/vtr/check_tailcompile_v62.py --out "$out" --require pilot
fi

if [[ "$phase" == full ]]; then
  : "${VTR_ROOT:?set VTR_ROOT for CPU full evaluation}"
  stage=pilot_gate
  python3 -c 'import json,os; from pathlib import Path; p=Path(os.environ["TAILCOMPILE_V62_OUT"])/"pilot/vpr_results.json"; assert json.loads(p.read_text())["pilot_gate_passed"] is True, "pilot gate did not pass"'
  stage=full_vpr
  python3 experiments/vtr/run_tailcompile_v62.py --stage evaluate \
    --selection "$out/selection" --out "$out/full" --jobs "$jobs" --designs all \
    --prior-results "$out/pilot/vpr_results.json"
  python3 experiments/vtr/check_tailcompile_v62.py --out "$out" --require full
fi
