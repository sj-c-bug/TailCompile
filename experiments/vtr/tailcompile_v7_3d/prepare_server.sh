#!/usr/bin/env bash
# Git-main-repository CPU bootstrap. Run with bash, never source this file.
set -eEuo pipefail
ulimit -c 0
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
action=bootstrap
stage=
case "${1:-}" in
  --help|-h)
    printf '%s\n' 'Usage: bash prepare_server.sh [--check | --stage STAGE [stage arguments]]' \
      'Default: isolated 3D build -> checks -> SHA mapping -> qualification -> plan -> prepare; then stop.' \
      'Source repository is located from this script; no ZIP extraction or conda activation.' \
      'Writable 3D backend defaults to repo/.deps/tailcompile_v7_3d/; original 2D VTR is read-only.' \
      'Existing external backend requires TAILCOMPILE_3D_BACKEND_READ_ONLY=1; no installation there.'
    exit 0 ;;
  --check) action=check; shift ;;
  --stage) action=stage; stage="${2:?--stage needs a stage name}"; shift 2 ;;
  '') ;;
  *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
esac
trap 'printf "V7_PREPARE_FAILED line=%s exit=%s; inspect this log before continuing.\n" "$LINENO" "$?" >&2' ERR
module="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module/../../.." && pwd -P)"
git_root="$(git -C "$repo" rev-parse --show-toplevel)"
[[ "$(realpath "$git_root")" == "$repo" ]] || { echo 'Script must be inside the TailCompile Git root.' >&2; exit 2; }
workspace="$(dirname -- "$repo")"
expected=40f8d19708b0b60e2aa04452deb1cfed68d96b0b
# This is the fixed dependency documented by REPOSITORY_WORKFLOW.md.
# Do not use inherited VTR_ROOT/TAILCOMPILE_SCRATCH from a 2D shell.
mapping="$workspace/vtr_tailcompile/vtr-verilog-to-routing"
mapping="$(realpath "$mapping")"
deps="$repo/.deps/tailcompile_v7_3d"
export TAILCOMPILE_3D_VTR_ROOT="$(realpath -m "${TAILCOMPILE_3D_VTR_ROOT:-$deps/vtr-verilog-to-routing}")"
export TAILCOMPILE_3D_BACKEND_READ_ONLY="${TAILCOMPILE_3D_BACKEND_READ_ONLY:-0}"
case "$TAILCOMPILE_3D_BACKEND_READ_ONLY" in 0|1) ;; *) echo 'TAILCOMPILE_3D_BACKEND_READ_ONLY must be 0 or 1.' >&2; exit 2 ;; esac
case "$TAILCOMPILE_3D_VTR_ROOT/" in "$mapping/"*) echo '3D build must not modify the original 2D VTR.' >&2; exit 2 ;; esac
case "$mapping/" in "$TAILCOMPILE_3D_VTR_ROOT/"*) echo '3D root must be disjoint from the original 2D VTR.' >&2; exit 2 ;; esac
case "$repo/" in "$TAILCOMPILE_3D_VTR_ROOT/"*) echo '3D dependency must not contain the TailCompile source repository.' >&2; exit 2 ;; esac
if [[ "$TAILCOMPILE_3D_BACKEND_READ_ONLY" == 1 ]]; then
  [[ -f "$TAILCOMPILE_3D_VTR_ROOT/backend_manifest.json" && -x "$TAILCOMPILE_3D_VTR_ROOT/vpr/vpr" ]] || {
    echo 'Read-only 3D backend needs an existing manifest and executable; no files will be installed there.' >&2; exit 2;
  }
else
  # Compare against the lexical owned namespace as well as resolved paths: a
  # .deps symlink must not redirect a clone/build outside the allowed repository.
  case "$(realpath -m "$deps")/" in "$repo/.deps/tailcompile_v7_3d/"*) ;; *) echo '3D dependency namespace resolves outside its owned repository path.' >&2; exit 2 ;; esac
  case "$TAILCOMPILE_3D_VTR_ROOT/" in "$deps/"*) ;; *) echo 'Writable 3D backend must stay under repo/.deps/tailcompile_v7_3d; use explicit read-only mode for an existing external backend.' >&2; exit 2 ;; esac
fi
export TAILCOMPILE_3D_PYTHON="${TAILCOMPILE_3D_PYTHON:-python3}"
export TAILCOMPILE_3D_TRAIN_DEVICE="${TAILCOMPILE_3D_TRAIN_DEVICE:-cpu}"
export TAILCOMPILE_3D_BINARY="$TAILCOMPILE_3D_VTR_ROOT/vpr/vpr"
export TAILCOMPILE_3D_BACKEND_MANIFEST="$TAILCOMPILE_3D_VTR_ROOT/backend_manifest.json"
export TAILCOMPILE_V7_OUT="$(realpath -m "${TAILCOMPILE_V7_OUT:-$module/runs/development_004_capacity}")"
case "$TAILCOMPILE_V7_OUT/" in "$module/runs/"*) ;; *) echo 'v7 results must stay inside this module runs directory.' >&2; exit 2 ;; esac
export TAILCOMPILE_V7_CONFIG="$(realpath -m "${TAILCOMPILE_V7_CONFIG:-$TAILCOMPILE_V7_OUT/campaign.json}")"
export TAILCOMPILE_3D_QUALIFICATION="$(realpath -m "${TAILCOMPILE_3D_QUALIFICATION:-$TAILCOMPILE_V7_OUT/qualification}")"
case "$TAILCOMPILE_V7_CONFIG/" in "$TAILCOMPILE_V7_OUT/"*) ;; *) echo 'v7 config must stay inside this cohort output.' >&2; exit 2 ;; esac
case "$TAILCOMPILE_3D_QUALIFICATION/" in "$TAILCOMPILE_V7_OUT/"*) ;; *) echo '3D qualification must stay inside this cohort output.' >&2; exit 2 ;; esac
export TAILCOMPILE_V7_JOBS="${TAILCOMPILE_V7_JOBS:-32}"
export TAILCOMPILE_V7_RRG_JOBS="${TAILCOMPILE_V7_RRG_JOBS:-4}"
export TAILCOMPILE_V7_BUILD_JOBS="${TAILCOMPILE_V7_BUILD_JOBS:-8}"
export TAILCOMPILE_V7_TIMEOUT="${TAILCOMPILE_V7_TIMEOUT:-3600}"
export TAILCOMPILE_3D_QUALIFICATION_TIMEOUT="${TAILCOMPILE_3D_QUALIFICATION_TIMEOUT:-600}"
export TAILCOMPILE_3D_QUALIFICATION_JOBS="${TAILCOMPILE_3D_QUALIFICATION_JOBS:-4}"
scratch_base="$repo/vtr_work"
scratch="$(realpath -m "$scratch_base/tailcompile_v7_3d")"
logs="$(realpath -m "$repo/logs")"
envfile="$(realpath -m "$repo/env_3d_v7.sh")"
case "$scratch/" in "$repo/vtr_work/tailcompile_v7_3d/"*) ;; *) echo '3D scratch resolves outside its owned repository path.' >&2; exit 2 ;; esac
case "$logs/" in "$repo/logs/"*) ;; *) echo 'Log directory resolves outside its owned repository path.' >&2; exit 2 ;; esac
[[ "$envfile" == "$repo/env_3d_v7.sh" ]] || { echo '3D environment file resolves outside its owned repository path.' >&2; exit 2; }
# Configure owned temporary/cache paths before Python imports and stage dispatch.
# --check does not materialize them; actual stages create them before use.
export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/cache" TORCH_HOME="$scratch/cache/torch"
export CCACHE_DIR="$scratch/cache/ccache" TORCH_EXTENSIONS_DIR="$scratch/cache/torch_extensions"
export CUDA_CACHE_PATH="$scratch/cache/cuda" TRITON_CACHE_DIR="$scratch/cache/triton"
for runtime_path in "$TMPDIR" "$XDG_CACHE_HOME" "$TORCH_HOME" "$CCACHE_DIR" \
  "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"; do
  case "$(realpath -m "$runtime_path")/" in "$scratch/"*) ;; *) echo '3D temporary/cache path resolves outside its owned scratch directory.' >&2; exit 2 ;; esac
done
printf 'TAILCOMPILE_GIT_ROOT=%s\nGIT_HEAD=%s\nREAD_ONLY_MAPPING_ROOT=%s\n3D_ROOT=%s\nSCRATCH=%s\n' \
  "$repo" "$(git -C "$repo" rev-parse HEAD)" "$mapping" "$TAILCOMPILE_3D_VTR_ROOT" "$scratch"
[[ "$(git -C "$mapping" rev-parse HEAD)" == "$expected" ]] || { echo 'Original VTR dependency is not the fixed 40f8 commit.' >&2; exit 2; }
for required in "$mapping/abc/abc" "$mapping/parmys/yosys/yosys" "$mapping/vpr/vpr"; do
  [[ -x "$required" ]] || { printf 'Complete fixed mapping environment is missing executable: %s\n' "$required" >&2; exit 2; }
done
[[ -f "$mapping/vtr_flow/scripts/run_vtr_flow.py" ]] || { echo 'Fixed VTR mapping runner is missing.' >&2; exit 2; }
[[ -f "$repo/REPOSITORY_WORKFLOW.md" ]] || { echo 'Canonical repository workflow document is missing.' >&2; exit 2; }
if [[ -n "${CONDA_PREFIX:-}" ]]; then
  export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
if [[ "$action" == check ]]; then
  "$TAILCOMPILE_3D_PYTHON" -c 'import importlib.util,sys; missing=[n for n in ("torch","numpy") if importlib.util.find_spec(n) is None]; assert not missing, "Missing packages: "+str(missing); print("PYTHON",sys.executable,"TORCH_NUMPY_DISCOVERABLE",flush=True)'
  printf 'V7_BOOTSTRAP_PREREQUISITES_OK; no build or experiment started.\n'
  exit 0
fi
# Full imports can initialize runtime caches; create owned paths before them.
mkdir -p "$scratch/tmp" "$scratch/cache"
"$TAILCOMPILE_3D_PYTHON" -c 'import sys,torch,numpy; print("PYTHON",sys.executable,"TORCH",torch.__version__,flush=True)'
run_stage() {
  # Legacy generic names exist only in this child process adapter.
  env TAILCOMPILE_REPO="$repo" TAILCOMPILE_WORKSPACE="$workspace" \
    TAILCOMPILE_MAPPING_VTR_ROOT="$mapping" TAILCOMPILE_SCRATCH="$scratch_base" \
    TAILCOMPILE_PYTHON="$TAILCOMPILE_3D_PYTHON" TAILCOMPILE_TRAIN_DEVICE="$TAILCOMPILE_3D_TRAIN_DEVICE" \
    bash "$module/run_pipeline.sh" "$@"
}
if [[ "$action" == stage ]]; then
  "$TAILCOMPILE_3D_PYTHON" "$module/install_hook.py" "$TAILCOMPILE_3D_VTR_ROOT" --verify
  run_stage "$stage" "$@"
  exit 0
fi
mkdir -p "$TAILCOMPILE_V7_OUT" "$logs" "$scratch/tmp" "$scratch/cache"
# Bootstrap runs before Campaign configures its own environment. Never inherit
# a 2D TMPDIR/cache when compiling, mapping SHA or running the interface tests.
export TMPDIR="$scratch/tmp" TMP="$scratch/tmp" TEMP="$scratch/tmp"
export XDG_CACHE_HOME="$scratch/cache"
exec 9>"$scratch/.bootstrap.lock"
flock -n 9 || { echo 'Another v7 CPU bootstrap is running; no duplicate build is started.' >&2; exit 2; }
if [[ ! -f "$TAILCOMPILE_V7_CONFIG" ]]; then
  cp -- "$module/campaign.example.json" "$TAILCOMPILE_V7_CONFIG"
fi
{
  printf '# Source after activating the intended Torch environment; only 3D/v7 variables.\n'
  for name in TAILCOMPILE_3D_VTR_ROOT TAILCOMPILE_3D_BINARY TAILCOMPILE_3D_BACKEND_MANIFEST \
    TAILCOMPILE_3D_BACKEND_READ_ONLY \
    TAILCOMPILE_3D_PYTHON TAILCOMPILE_3D_TRAIN_DEVICE \
    TAILCOMPILE_3D_QUALIFICATION TAILCOMPILE_3D_QUALIFICATION_TIMEOUT TAILCOMPILE_3D_QUALIFICATION_JOBS \
    TAILCOMPILE_V7_OUT TAILCOMPILE_V7_CONFIG TAILCOMPILE_V7_JOBS TAILCOMPILE_V7_RRG_JOBS \
    TAILCOMPILE_V7_BUILD_JOBS TAILCOMPILE_V7_TIMEOUT; do
    printf 'export %s=%q\n' "$name" "${!name}"
  done
} >"$envfile"
readonly_evidence="$TAILCOMPILE_V7_OUT/mapping_dependency.sha256"
if [[ ! -f "$readonly_evidence" ]]; then
  sha256sum "$mapping/abc/abc" "$mapping/parmys/yosys/yosys" "$mapping/vpr/vpr" >"$readonly_evidence"
fi
sha256sum --check "$readonly_evidence"
if [[ "$TAILCOMPILE_3D_BACKEND_READ_ONLY" == 0 && ! -f "$TAILCOMPILE_3D_VTR_ROOT/backend_manifest.json" ]]; then
  if [[ ! -d "$TAILCOMPILE_3D_VTR_ROOT/.git" ]]; then
    [[ ! -e "$TAILCOMPILE_3D_VTR_ROOT" ]] || { echo '3D target exists without a Git checkout; inspect it without deleting.' >&2; exit 2; }
    mkdir -p "$(dirname -- "$TAILCOMPILE_3D_VTR_ROOT")"
    git clone --no-hardlinks "$mapping" "$TAILCOMPILE_3D_VTR_ROOT"
    git -C "$TAILCOMPILE_3D_VTR_ROOT" checkout --detach "$expected"
  fi
  [[ "$(git -C "$TAILCOMPILE_3D_VTR_ROOT" rev-parse HEAD)" == "$expected" ]]
  git -C "$TAILCOMPILE_3D_VTR_ROOT" submodule update --init --recursive
  patch="$repo/experiments/vtr3d/patches/vtr40f8_layer_heterogeneity_v2.patch"
  if ! git -C "$TAILCOMPILE_3D_VTR_ROOT" apply --reverse --check "$patch"; then
    git -C "$TAILCOMPILE_3D_VTR_ROOT" apply --check "$patch"
    git -C "$TAILCOMPILE_3D_VTR_ROOT" apply "$patch"
  fi
  env TAILCOMPILE_REPO="$repo" VTR_ROOT="$TAILCOMPILE_3D_VTR_ROOT" \
    TAILCOMPILE_HETERO_PATCH="$patch" TAILCOMPILE_V7_BUILD_DIR="$TAILCOMPILE_3D_VTR_ROOT/build_tailcompile_v7" \
    bash "$module/install_vtr.sh"
fi
"$TAILCOMPILE_3D_PYTHON" "$module/install_hook.py" "$TAILCOMPILE_3D_VTR_ROOT" --verify
"$TAILCOMPILE_3D_PYTHON" "$module/verify.py" --out "$TAILCOMPILE_V7_OUT/server_contract_tests.json"
sha_flow="$scratch/qualification_sha_mapping"
if [[ ! -f "$sha_flow/sha.abc.blif" ]]; then
  mkdir -p "$sha_flow"
  "$TAILCOMPILE_3D_PYTHON" "$mapping/vtr_flow/scripts/run_vtr_flow.py" \
    "$mapping/vtr_flow/benchmarks/verilog/sha.v" \
    "$TAILCOMPILE_3D_VTR_ROOT/vtr_flow/arch/multi_die/hecate/3d/hecate_3d_sb_10um_bidir_fanin_16_fanout_16_with_esd_7nm.xml" \
    -temp_dir "$sha_flow" -ending_stage abc -timeout "$TAILCOMPILE_V7_TIMEOUT"
fi
"$TAILCOMPILE_3D_PYTHON" "$module/qualify.py" \
  --binary "$TAILCOMPILE_3D_VTR_ROOT/vpr/vpr" --vtr-root "$TAILCOMPILE_3D_VTR_ROOT" \
  --sha-blif "$sha_flow/sha.abc.blif" --out "$TAILCOMPILE_3D_QUALIFICATION" \
  --jobs "$TAILCOMPILE_3D_QUALIFICATION_JOBS" --timeout "$TAILCOMPILE_3D_QUALIFICATION_TIMEOUT"
run_stage plan
run_stage prepare
sha256sum --check "$readonly_evidence"
printf 'CPU_PREPARE_OK\nENV=%s\nRESULT=%s/prepared.json\nStopped before calibration, training and 480 pilot.\n' \
  "$envfile" "$TAILCOMPILE_V7_OUT"
