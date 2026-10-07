#!/usr/bin/env bash
set -euo pipefail

repo_root="${1:-$(pwd)}"
cd "$repo_root"
export KMP_DUPLICATE_LIB_OK=TRUE

# Rebuild after the VPR campaign so newly collected trials enter the bundle.
python3 experiments/vtr/prepare_tailcompile_v3_training.py \
  --out experiments/vtr/tailcompile_v3_training/bundle

# This bootstrap command is intentionally marked as a smoke run because only
# two design families currently exist.  Replace --smoke-train-all with a
# design-family holdout once at least three independently generated families
# have controlled candidate pairs.
python3 experiments/vtr/train_tailcompile_v3_ranker.py \
  --dataset experiments/vtr/tailcompile_v3_training/bundle/dataset.json \
  --out experiments/vtr/tailcompile_v3_training/model_bootstrap \
  --epochs 200 \
  --hidden 32 \
  --learning-rate 1e-3 \
  --coordinate-weight 0.2 \
  --smoke-train-all
