#!/usr/bin/env bash
set -eEuo pipefail

repo_root="${TAILCOMPILE_REPO:-$(pwd)}"
cd "$repo_root"

bash experiments/vtr/tailcompile_v3_training/prepare_spmv_pilot_inputs.sh

spmv_run="experiments/vtr/runs/pilot_v1/spmv"
python3 experiments/vtr/tailcompile_v2.py \
  --design spmv \
  --blif "$spmv_run/full-seed1-w300/spmv.pre-vpr.blif" \
  --packed-net "$spmv_run/full-seed1-w300/spmv.net" \
  --rr-graph "$spmv_run/route-seed1-w20-rr/spmv.rr_graph.xml.gz" \
  --out experiments/vtr/tailcompile_v2/spmv/4x4 \
  --region-grid 4x4

python3 experiments/vtr/tailcompile_v3_campaign.py \
  --config experiments/vtr/tailcompile_v3_training/campaign_seed_aggregated_v2.json \
  --stage all \
  --script experiments/vtr/tailcompile_v3_training/run_labels_seed_aggregated_v2.sh

echo "Prepared seed-aggregated v2 campaign."
echo "Run: bash experiments/vtr/tailcompile_v3_training/run_labels_seed_aggregated_v2.sh"
