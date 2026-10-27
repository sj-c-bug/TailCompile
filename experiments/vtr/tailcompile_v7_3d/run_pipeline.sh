#!/usr/bin/env bash
# CPU/GPU stages use explicit artifact hand-offs and the same frozen campaign.
set -eEuo pipefail
ulimit -c 0
stage="${1:?stage required: plan/cpu-prepare/pretrain/feedback/finetune/validate/pilot/scan/timing/resident/report}"
shift
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
: "${TAILCOMPILE_REPO:?set repository root containing experiments/vtr}"
: "${TAILCOMPILE_3D_VTR_ROOT:?set isolated v7 VTR checkout, not the frozen 2D installation}"
: "${TAILCOMPILE_3D_QUALIFICATION:?set directory containing passed qualification_report.json and protocol.json}"
: "${TAILCOMPILE_MAPPING_VTR_ROOT:?set read-only full pinned VTR with ABC/Yosys for RTL mapping}"
: "${TAILCOMPILE_SCRATCH:?set fast scratch directory inside the registered DAC2027 workspace}"
workspace="${TAILCOMPILE_WORKSPACE:-/inspire/hdd/project/project-public/changshunjie-253114010247/changshunjie-253114010247/workspace/DAC2027}"
out="${TAILCOMPILE_V7_OUT:-$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_001}"
config="${TAILCOMPILE_V7_CONFIG:-$script_dir/campaign.example.json}"
binary="${TAILCOMPILE_3D_BINARY:-$TAILCOMPILE_3D_VTR_ROOT/vpr/vpr}"
manifest="${TAILCOMPILE_3D_BACKEND_MANIFEST:-$TAILCOMPILE_3D_VTR_ROOT/backend_manifest.json}"
python="${TAILCOMPILE_PYTHON:-python3}"
common=(--config "$config" --workspace "$workspace" --vtr-root "$TAILCOMPILE_3D_VTR_ROOT"
  --mapping-vtr-root "$TAILCOMPILE_MAPPING_VTR_ROOT"
  --binary "$binary" --backend-manifest "$manifest"
  --qualification-report "$TAILCOMPILE_3D_QUALIFICATION/qualification_report.json"
  --out "$out" --scratch "$TAILCOMPILE_SCRATCH/tailcompile_v7_3d"
  --jobs "${TAILCOMPILE_V7_JOBS:-32}" --rrg-jobs "${TAILCOMPILE_V7_RRG_JOBS:-4}"
  --per-job-mib "${TAILCOMPILE_V7_JOB_MIB:-4096}" --rrg-mib "${TAILCOMPILE_V7_RRG_MIB:-4096}"
  --timeout "${TAILCOMPILE_V7_TIMEOUT:-600}" --threads 1
  --device "${TAILCOMPILE_TRAIN_DEVICE:-cuda}")
run() { "$python" "$script_dir/campaign.py" "$1" "${common[@]}" "${@:2}"; }
case "$stage" in
  cpu-prepare)
    run plan "$@"
    run prepare "$@"
    run calibrate "$@"
    run teachers "$@"
    ;;
  plan|prepare|calibrate|teachers|pretrain|feedback|finetune|validate|pilot|scan|timing|resident|report)
    run "$stage" "$@"
    ;;
  *) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
