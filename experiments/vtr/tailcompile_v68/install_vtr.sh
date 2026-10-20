#!/usr/bin/env bash
# Compile the audited native-GP initializer, preserving the old executable.
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set TAILCOMPILE_REPO}"
: "${VTR_ROOT:?set VTR_ROOT to the fixed server commit}"
: "${TAILCOMPILE_V68_BUILD_DIR:?set the existing CMake build directory explicitly, e.g. $VTR_ROOT/build_tailcompile_conda}"
cd "$TAILCOMPILE_REPO"
root="$(realpath "$VTR_ROOT")"
build="$(realpath "$TAILCOMPILE_V68_BUILD_DIR")"
case "$build" in "$root"/*) ;; *) echo "Build directory must be inside VTR_ROOT"; exit 2 ;; esac
test -f "$build/CMakeCache.txt"
configured_source="$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "$build/CMakeCache.txt")"
test -n "$configured_source"
if [[ "$(realpath "$configured_source")" != "$root" ]]; then
  echo "CMake cache belongs to another source tree: $configured_source"; exit 2
fi
test -f "$root/vpr/vpr"
old_sha="$(sha256sum "$root/vpr/vpr" | cut -d' ' -f1)"
mkdir -p "$root/tailcompile_frozen_binaries"
backup="$root/tailcompile_frozen_binaries/vpr_${old_sha}"
if [[ ! -f "$backup" ]]; then cp -L "$root/vpr/vpr" "$backup"; fi
python3 experiments/vtr/tailcompile_v68/install_hook.py "$root"
cmake --build "$build" --target vpr -j "${TAILCOMPILE_V68_BUILD_JOBS:-8}"
test -x "$build/vpr/vpr"
ln -sfn "$build/vpr/vpr" "$root/vpr/vpr"
python3 experiments/vtr/tailcompile_v68/install_hook.py "$root" --verify
smoke="$TAILCOMPILE_REPO/vtr_work/v68_hook_smoke_$(date +%Y%m%d_%H%M%S)"
python3 experiments/vtr/tailcompile_v68/test_hook.py --real-vtr "$root" \
  --smoke-out "$smoke" --baseline-vpr "$backup"
echo "Old VPR preserved: $backup"
echo "New native-GP hook verified: $smoke/smoke_report.json"
