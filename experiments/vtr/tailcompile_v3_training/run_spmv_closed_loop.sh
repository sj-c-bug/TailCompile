#!/usr/bin/env bash
set -u

repo_root="${TAILCOMPILE_REPO:-$(pwd)}"
export TAILCOMPILE_SHARED="${TAILCOMPILE_SHARED:-${repo_root}/experiments/vtr}"
cd "$repo_root"

packed_net="${repo_root}/experiments/vtr/runs/pilot_v1/spmv/full-seed1-w300/spmv.net"
seeds=(11 12 13 14)
widths=(80 90 100 110)

if [[ ! -s "$packed_net" ]]; then
  echo "ERROR: missing packed netlist: $packed_net" >&2
  exit 2
fi

# rrg_only, mean_rrg, and tail_rrg have the same fplace hash for spmv.  Run the
# shared deterministic placement once under tail_rrg and alias it in analysis.
for method in default_ap tail_rrg; do
  for seed in "${seeds[@]}"; do
    for width in "${widths[@]}"; do
      python3 experiments/vtr/run_tailcompile_v0.py \
        --variant v2 --design spmv --region-grid 4x4 \
        --method "$method" --packed-net "$packed_net" \
        --seed "$seed" --width "$width" \
        --legalizer flat-recon --detailed-placer annealer \
        --device-width 84 --timeout 1200 --skip-existing || true
    done
  done
done

for width in "${widths[@]}"; do
  pool="${repo_root}/experiments/vtr/tailcompile_v3/spmv/4x4/pools/trained_w${width}_24"
  shopt -s nullglob
  candidates=("$pool"/candidate_*.fplace)
  shopt -u nullglob
  if [[ "${#candidates[@]}" -ne 24 ]]; then
    echo "ERROR: expected 24 trained candidates in $pool, found ${#candidates[@]}" >&2
    exit 3
  fi
  for fplace in "${candidates[@]}"; do
    for seed in "${seeds[@]}"; do
      python3 experiments/vtr/run_tailcompile_v0.py \
        --variant v3 --design spmv --region-grid 4x4 \
        --method gnn_candidate --fplace "$fplace" --packed-net "$packed_net" \
        --seed "$seed" --width "$width" \
        --legalizer flat-recon --detailed-placer annealer \
        --device-width 84 --timeout 1200 --skip-existing || true
    done
  done
done

echo "spmv closed-loop campaign completed."
python3 experiments/vtr/summarize_spmv_closed_loop.py
