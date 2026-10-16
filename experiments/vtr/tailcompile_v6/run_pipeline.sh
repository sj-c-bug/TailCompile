#!/usr/bin/env bash
set -eEuo pipefail

: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO to the TailCompile repository root}"

cd "$TAILCOMPILE_REPO"

v5_root="${TAILCOMPILE_V6_SOURCE:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001/confirm}"
v6_root="${TAILCOMPILE_V6_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v6/runs/information_ablation_001}"
epochs="${TAILCOMPILE_V6_EPOCHS:-60}"
device="${TAILCOMPILE_V6_DEVICE:-auto}"

trap 'code=$?; echo "[$(date --iso-8601=seconds)] ERROR line=$LINENO code=$code"; exit $code' ERR

echo "[$(date --iso-8601=seconds)] stage=prepare_v6_dataset source=$v5_root"
python3 experiments/vtr/prepare_tailcompile_v6.py \
  --v5-root "$v5_root" \
  --out "$v6_root/dataset"

echo "[$(date --iso-8601=seconds)] stage=train_v6 device=$device epochs=$epochs"
python3 experiments/vtr/train_tailcompile_v6.py \
  --dataset "$v6_root/dataset/dataset.json" \
  --out "$v6_root/models" \
  --epochs "$epochs" \
  --device "$device"

test -s "$v6_root/models/evaluation_report.json"
echo "[$(date --iso-8601=seconds)] TailCompile v6 information ablation complete"
echo "report=$v6_root/models/evaluation_report.json"
