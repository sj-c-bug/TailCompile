#!/usr/bin/env bash
set -eEuo pipefail
trap 'echo "ERROR: v4 2D campaign failed at line $LINENO, exit=$?" >&2' ERR
: "${TAILCOMPILE_REPO:?Set TAILCOMPILE_REPO}"
: "${VTR_ROOT:?Set VTR_ROOT to the server commit checkout}"
export PYTHONUNBUFFERED=1
cd "$TAILCOMPILE_REPO"
out="${TAILCOMPILE_V4_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v4/runs/pilot_2d_001}"
date -Is
python3 experiments/vtr/run_tailcompile_v4.py --stage all --out "$out" --vtr-root "$VTR_ROOT" "$@"
date -Is
echo "2D v4 paired campaign complete: $out/summary.json"
