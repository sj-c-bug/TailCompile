#!/usr/bin/env bash
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
cd "$TAILCOMPILE_REPO"

phase="${1:-status}"
case "$phase" in generate|plan|run|summarize|runtime|status) ;; *)
  echo "usage: $0 generate|plan|run|summarize|runtime|status"; exit 2 ;; esac

out="${TAILCOMPILE_V65_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v65/runs/guided_001}"
v61="${TAILCOMPILE_V61_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v61/runs/tile8_001}"
bundle="$v61/data/bundle.pt"
generation="$out/generation"
sweep="$out/sweep"
jobs="${TAILCOMPILE_V65_JOBS:-8}"
timeout="${TAILCOMPILE_V65_TIMEOUT:-1800}"
export PYTHONUNBUFFERED=1
stage=initialization
trap 'code=$?; echo "ERROR phase=$phase stage=$stage line=$LINENO code=$code"; exit "$code"' ERR

case "$phase" in
  generate)
    stage=proxy_guided_candidate_generation
    python3 experiments/vtr/build_tailcompile_v65_guided.py \
      --bundle "$bundle" --out "$generation"
    ;;
  plan)
    stage=freeze_candidate_and_width_grid
    python3 experiments/vtr/run_tailcompile_v64_wmin.py plan \
      --selection "$generation/selection_report.json" --out "$sweep" \
      --candidate-method guided_tail --seeds 101 102 103 104 105 106 107 108 109 110 \
      --widths conv_layer:120:80,90,100,110,120,130,140,150,160 \
      --widths lenet:180:140,150,160,170,180,190,200,210,220
    ;;
  run)
    : "${VTR_ROOT:?set VTR_ROOT for the pinned server VTR checkout}"
    stage=paired_vpr_sweep
    mkdir -p "$sweep"
    exec 9>"$sweep/.vpr.lock"
    flock -n 9 || { echo "another v6.5 sweep is active: $sweep"; exit 1; }
    python3 experiments/vtr/run_tailcompile_v64_wmin.py run \
      --out "$sweep" --jobs "$jobs" --timeout "$timeout"
    ;;
  summarize|status)
    stage="$phase"
    python3 experiments/vtr/run_tailcompile_v64_wmin.py "$phase" --out "$sweep"
    ;;
  runtime)
    stage=runtime_and_iteration_audit
    python3 experiments/vtr/analyze_tailcompile_runtime.py \
      --root "$sweep" --require-logs
    ;;
esac
