#!/usr/bin/env bash
# Run in an explicitly isolated 40f8 checkout already carrying heterogeneity v2.
set -eEuo pipefail
: "${TAILCOMPILE_REPO:?set repository containing experiments/vtr}"
: "${VTR_ROOT:?set isolated 3D VTR checkout, never the frozen 2D installation}"
: "${TAILCOMPILE_V7_BUILD_DIR:?set CMake build directory inside isolated VTR_ROOT}"
: "${TAILCOMPILE_HETERO_PATCH:?set reviewed vtr40f8_layer_heterogeneity_v2.patch path}"
root="$(realpath "$VTR_ROOT")"
build="$(realpath -m "$TAILCOMPILE_V7_BUILD_DIR")"
case "$build" in "$root"/*) ;; *) echo 'Build must be inside isolated VTR_ROOT'; exit 2 ;; esac
if [[ -e "$root/tailcompile_heterogeneity_v2.patch" ]]; then
  cmp -- "$TAILCOMPILE_HETERO_PATCH" "$root/tailcompile_heterogeneity_v2.patch"
else
  cp -- "$TAILCOMPILE_HETERO_PATCH" "$root/tailcompile_heterogeneity_v2.patch"
fi
python3 "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/install_hook.py" "$root"
cmake -S "$root" -B "$build" -DCMAKE_BUILD_TYPE=Release -DVTR_ENABLE_GRAPHICS=OFF \
  -DWITH_ABC=OFF -DWITH_ODIN=OFF -DWITH_PARMYS=OFF
cmake --build "$build" --target vpr -j "${TAILCOMPILE_V7_BUILD_JOBS:-4}"
ln -sfn "$build/vpr/vpr" "$root/vpr/vpr"
python3 "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/install_hook.py" "$root" --verify
python3 "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/write_backend_manifest.py" \
  --root "$root" --build "$build"
