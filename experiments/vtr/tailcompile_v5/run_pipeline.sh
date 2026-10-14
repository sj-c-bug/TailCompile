#!/usr/bin/env bash
set -eEuo pipefail

: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO to the TailCompile repository root}"
: "${VTR_ROOT:?set VTR_ROOT to the pinned VTR checkout}"
: "${TAILCOMPILE_SCRATCH:?set TAILCOMPILE_SCRATCH to fast local scratch storage}"

cd "$TAILCOMPILE_REPO"

campaign_root="${TAILCOMPILE_V5_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001}"
input_root="$campaign_root/inputs"
screen_root="$campaign_root/screen"
confirm_root="$campaign_root/confirm"
checkpoint="${TAILCOMPILE_V5_CHECKPOINT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v3_training/model_width_conditioned/ranker.pt}"

prepare_args=()
if [[ -n "${TAILCOMPILE_HOLDOUT_DESIGNS:-}" ]]; then
    read -r -a holdout_designs <<<"$TAILCOMPILE_HOLDOUT_DESIGNS"
    prepare_args+=(--designs "${holdout_designs[@]}")
fi

echo "[$(date --iso-8601=seconds)] stage=prepare_holdout_inputs out=$input_root"
python3 experiments/vtr/prepare_tailcompile_v5_inputs.py \
    --vtr-root "$VTR_ROOT" \
    --scratch "$TAILCOMPILE_SCRATCH" \
    --out "$input_root" \
    "${prepare_args[@]}"

echo "[$(date --iso-8601=seconds)] stage=default_only_screen out=$screen_root"
python3 experiments/vtr/screen_tailcompile_v5.py \
    --inputs "$input_root/holdout_inputs.json" \
    --checkpoint "$checkpoint" \
    --vtr-root "$VTR_ROOT" \
    --out "$screen_root"

echo "[$(date --iso-8601=seconds)] stage=lock_heterogeneity out=$confirm_root"
python3 experiments/vtr/run_tailcompile_v5.py \
    --config "$screen_root/confirmatory_config.json" \
    --vtr-root "$VTR_ROOT" \
    --out "$confirm_root" \
    --stage prepare

echo "[$(date --iso-8601=seconds)] stage=confirmatory_vpr"
python3 experiments/vtr/run_tailcompile_v5.py \
    --config "$screen_root/confirmatory_config.json" \
    --vtr-root "$VTR_ROOT" \
    --out "$confirm_root" \
    --stage run

echo "[$(date --iso-8601=seconds)] stage=conditional_analysis"
python3 experiments/vtr/analyze_tailcompile_v5.py --root "$confirm_root"

echo "[$(date --iso-8601=seconds)] TailCompile v5 campaign complete"
echo "analysis=$confirm_root/confirmatory_analysis.json"
