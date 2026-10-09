#!/usr/bin/env bash
set -eEuo pipefail

# Materialize the three large VTR artifacts needed by the spmv holdout.  They
# are intentionally not stored in Git, so a fresh checkout must recreate them.
repo_root="${TAILCOMPILE_REPO:-$(pwd)}"
vtr_root="${VTR_ROOT:?VTR_ROOT must point to the vtr-verilog-to-routing checkout}"
scratch_base="${TAILCOMPILE_SCRATCH:-${repo_root}/vtr_work}"
vtr_python="${VTR_PYTHON:-python3}"
expected_vtr_commit="${VTR_COMMIT:-e422b08861dfc8500874f04105ba2a7eb2f11ccd}"

cd "$repo_root"

archive_root="experiments/vtr/runs/pilot_v1/spmv"
full_archive="$archive_root/full-seed1-w300"
rr_archive="$archive_root/route-seed1-w20-rr"
blif="$full_archive/spmv.pre-vpr.blif"
packed_net="$full_archive/spmv.net"
rr_graph="$rr_archive/spmv.rr_graph.xml.gz"

if [[ -s "$blif" && -s "$packed_net" && -s "$rr_graph" ]]; then
  echo "spmv pilot inputs already exist; skipping VTR materialization."
  exit 0
fi

flow="$vtr_root/vtr_flow/scripts/run_vtr_flow.py"
source_v="$vtr_root/vtr_flow/benchmarks/verilog/koios/spmv.v"
include_v="$vtr_root/vtr_flow/benchmarks/verilog/koios/hard_block_include.v"
arch="$vtr_root/vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml"
vpr="$vtr_root/vpr/vpr"

for required in "$flow" "$source_v" "$include_v" "$arch" "$vpr"; do
  if [[ ! -f "$required" ]]; then
    echo "ERROR: missing VTR input: $required" >&2
    exit 2
  fi
done

actual_vtr_commit="$(git -C "$vtr_root" rev-parse HEAD 2>/dev/null || true)"
if [[ "$actual_vtr_commit" != "$expected_vtr_commit" ]]; then
  echo "ERROR: VTR commit mismatch." >&2
  echo "  expected: $expected_vtr_commit" >&2
  echo "  actual:   ${actual_vtr_commit:-not a Git checkout}" >&2
  echo "Set VTR_COMMIT only if intentionally starting a separate experiment cohort." >&2
  exit 2
fi

mkdir -p "$scratch_base" "$full_archive" "$rr_archive"
work_root="$(mktemp -d "$scratch_base/spmv-pilot-inputs.XXXXXX")"
full_work="$work_root/full-seed1-w300"
rr_work="$work_root/route-seed1-w20-rr"
mkdir -p "$full_work" "$rr_work"

if [[ ! -s "$blif" || ! -s "$packed_net" || ! -s "$full_archive/spmv.place" ]]; then
  echo "Generating packed spmv netlist with the pinned VTR flow..."
  "$vtr_python" "$flow" "$source_v" "$arch" \
    -include "$include_v" \
    -temp_dir "$full_work" \
    -timeout 1800 \
    --seed 1 \
    --route_chan_width 300

  for generated in spmv.pre-vpr.blif spmv.net spmv.place; do
    if [[ ! -s "$full_work/$generated" ]]; then
      echo "ERROR: VTR full flow did not create $full_work/$generated" >&2
      exit 3
    fi
  done
  cp "$full_work/spmv.pre-vpr.blif" "$blif"
  cp "$full_work/spmv.net" "$packed_net"
  cp "$full_work/spmv.place" "$full_archive/spmv.place"
fi

if [[ ! -s "$rr_graph" ]]; then
  place="$full_archive/spmv.place"
  if [[ ! -s "$place" ]]; then
    echo "ERROR: missing placement needed to create RRG: $place" >&2
    echo "Remove the two partial spmv inputs or copy the matching spmv.place, then retry." >&2
    exit 4
  fi

  echo "Generating width-20 spmv RR graph (an unroutable VPR exit is expected)..."
  set +e
  "$vpr" "$arch" "$blif" \
    --route --analysis \
    --net_file "$packed_net" \
    --place_file "$place" \
    --route_file "$rr_work/spmv.route" \
    --seed 1 \
    --route_chan_width 20 \
    --write_rr_graph "$rr_work/spmv.rr_graph.xml" \
    >"$rr_work/vpr.out" 2>&1
  vpr_code=$?
  set -e
  if [[ ! -s "$rr_work/spmv.rr_graph.xml" ]]; then
    echo "ERROR: VPR exited with code $vpr_code and did not create the RR graph." >&2
    echo "Inspect: $rr_work/vpr.out" >&2
    exit 5
  fi
  gzip -c "$rr_work/spmv.rr_graph.xml" > "$rr_graph"
  cp "$rr_work/vpr.out" "$rr_archive/vpr.out"
  echo "RR graph generated (VPR exit code $vpr_code; nonzero is normal at width 20)."
fi

for required in "$blif" "$packed_net" "$rr_graph"; do
  if [[ ! -s "$required" ]]; then
    echo "ERROR: spmv pilot input is empty or missing: $required" >&2
    exit 6
  fi
done

echo "spmv pilot inputs are ready:"
echo "  $blif"
echo "  $packed_net"
echo "  $rr_graph"
echo "VTR scratch retained for audit: $work_root"
