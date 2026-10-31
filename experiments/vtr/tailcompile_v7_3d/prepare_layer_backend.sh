#!/usr/bin/env bash
# New layer backend; old 2D/3D source and build are read-only dependencies.
set -eEuo pipefail
export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
module="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
repo="$(cd -- "$module/../../.." && pwd -P)"
python="${TAILCOMPILE_3D_PYTHON:-python3}"
root="${TAILCOMPILE_V7_LAYER_BACKEND:-$repo/.deps/tailcompile_v7_3d_layer/vtr}"
out="${TAILCOMPILE_V7_LAYER_OUT:-$module/runs/development_007_layer}"
qualification="${TAILCOMPILE_V7_LAYER_QUALIFICATION:-$out/qualification/qualification_report.json}"
for owned in "$root" "$qualification"; do
  [[ "$(realpath -m "$owned")" == "$owned" ]] || { echo "Redirected write path: $owned" >&2; exit 2; }
done
case "$root/" in "$repo/.deps/"?*) ;; *) echo 'New backend must be inside repository .deps' >&2; exit 2 ;; esac
case "$qualification" in "$module/runs/development_007"*) ;; *) echo 'Qualification must be inside a new 007 cohort' >&2; exit 2 ;; esac
if [[ -n "${TAILCOMPILE_V7_LAYER_BASE:-}" ]]; then
  base="$TAILCOMPILE_V7_LAYER_BASE"
else
  base="$("$python" - "$module/runs/development_004_capacity/binary_identity.json" <<'PY'
import json,sys
from pathlib import Path
binary=Path(json.loads(Path(sys.argv[1]).read_text())['binary']).resolve()
matches=[p for p in binary.parents if (p/'backend_manifest.json').is_file()]
if not matches: raise SystemExit('Cannot find the existing qualified 3D backend; set TAILCOMPILE_V7_LAYER_BASE.')
print(matches[0])
PY
)"
fi
if [[ ! -f "$root/backend_manifest.json" ]]; then
  "$python" "$module/build_layer_backend.py" --base "$base" --root "$root" --jobs "${TAILCOMPILE_V7_BUILD_JOBS:-2}"
fi
"$python" "$module/install_layer_hook.py" "$root" --verify
binary="$("$python" - "$root/backend_manifest.json" <<'PY'
import hashlib,json,sys
from pathlib import Path
m=json.loads(Path(sys.argv[1]).read_text()); p=Path(m['binary'])
if hashlib.sha256(p.read_bytes()).hexdigest()!=m['binary_sha256']: raise SystemExit('Backend binary changed')
print(p)
PY
)"
"$python" "$module/layer_qualification.py" --binary "$binary" --vtr-root "$base" --out "$(dirname -- "$qualification")"
printf 'V7_LAYER_BACKEND_READY root=%s qualification=%s\n' "$root" "$qualification"
