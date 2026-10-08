#!/usr/bin/env bash
set -eEuo pipefail

export PYTHONUNBUFFERED=1

on_error() {
  status=$?
  echo "[$(date -Is)] ERROR stage=${stage:-startup} line=${BASH_LINENO[0]} exit_code=${status}" >&2
  exit "$status"
}
trap on_error ERR

repo_root="${1:-$(pwd)}"
cd "$repo_root"
export KMP_DUPLICATE_LIB_OK=TRUE

stage="environment"
echo "[$(date -Is)] TailCompile training started"
echo "repo_root=$repo_root"
echo "python=$(command -v python3)"
python3 -u -c 'import os, platform, torch; print({"stage":"environment", "pid":os.getpid(), "python":platform.python_version(), "torch":torch.__version__, "cuda_available":torch.cuda.is_available(), "cuda_device":torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}, flush=True)'

# Rebuild after the VPR campaign so newly collected trials enter the bundle.
stage="prepare_dataset"
echo "[$(date -Is)] stage=$stage"
read -r -a design_array <<< "${TAILCOMPILE_DESIGNS:-sha attention_layer}"
echo "training_design_bundle=${design_array[*]}"
python3 -u experiments/vtr/prepare_tailcompile_v3_training.py \
  --designs "${design_array[@]}" \
  --out experiments/vtr/tailcompile_v3_training/bundle

# This bootstrap command is intentionally marked as a smoke run because only
# two design families currently exist.  Replace --smoke-train-all with a
# design-family holdout once at least three independently generated families
# have controlled candidate pairs.
stage="train_ranker"
echo "[$(date -Is)] stage=$stage"
validation_args=(--smoke-train-all)
if [[ -n "${TAILCOMPILE_VALIDATION_DESIGNS:-}" ]]; then
  read -r -a validation_design_array <<< "$TAILCOMPILE_VALIDATION_DESIGNS"
  validation_args=(--validation-designs "${validation_design_array[@]}")
fi
python3 -u experiments/vtr/train_tailcompile_v3_ranker.py \
  --dataset experiments/vtr/tailcompile_v3_training/bundle/dataset.json \
  --out experiments/vtr/tailcompile_v3_training/model_width_conditioned \
  --epochs 200 \
  --hidden 32 \
  --learning-rate 1e-3 \
  --coordinate-weight 0.2 \
  --device auto \
  --log-every 10 \
  --checkpoint-every 20 \
  "${validation_args[@]}"

stage="verify_outputs"
model_dir="experiments/vtr/tailcompile_v3_training/model_width_conditioned"
test -s "$model_dir/ranker.pt"
test -s "$model_dir/training_report.json"
echo "[$(date -Is)] SUCCESS ranker=$model_dir/ranker.pt report=$model_dir/training_report.json"
