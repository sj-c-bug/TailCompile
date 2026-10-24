"""Audited backend reuse for the frozen-weight benchmark sweep."""
from pathlib import Path
import importlib.util
import os
import sys
import threading
import time

HERE = Path(__file__).resolve().parent
V68 = HERE.with_name('tailcompile_v68')
V69 = HERE.with_name('tailcompile_v69')
REPO = HERE.parents[2]
# Append dependencies: local v610 generate/bridge modules must win imports.
for path in (V68, V69, V68.parent):
    if str(path) not in sys.path:
        sys.path.append(str(path))

from common import emit, execute, fingerprint, freeze, read, recover, sha, write
import experiment as legacy_experiment
import parallel_timing as legacy_timing
from lifecycle import (check_free_space, log_storage_failure, StoragePressureError,
                       parallel, storage_error)
e, pt = legacy_experiment, legacy_timing
# experiment.py prepends its own directory; restore the new entrypoint first.
if str(HERE) in sys.path:
    sys.path.remove(str(HERE))
sys.path.insert(0, str(HERE))


def load_module(name, path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


metrics = load_module('v610_reused_width_metrics', V69/'metrics.py')
phase_metrics = load_module('v610_reused_phase_metrics', V69/'phase_metrics.py')
_tracked = None
_tracked_lock = threading.Lock()


def prune_trial(folder, out):
    """Reclaim only terminal per-trial outputs after compact evidence is saved."""
    import storage
    global _tracked
    folder, out = Path(folder).resolve(), Path(out).resolve()
    folder.relative_to(out)
    out.relative_to(REPO/'experiments/vtr/tailcompile_v610/runs')
    if os.environ.get('TAILCOMPILE_V610_PRUNE_COMPLETED', '1') == '0':
        return None
    with _tracked_lock:
        if _tracked is None:
            _tracked = storage.prepare_paths(storage.tracked_files(REPO))
    result = storage.inspect_trial(folder, tracked=_tracked)
    if not result['eligible'] or not result['entries']:
        return result
    marker = folder/'payload_prune_pending.json'
    write(marker, dict(entries=result['entries'], created_unix=time.time()))
    deleted = []
    for entry in result['entries']:
        path = Path(entry['path'])
        if (path.is_symlink() or path.resolve().parent != folder or
                not path.is_file() or storage.snapshot(path) != entry['stat'] or
                sha(folder/'trial.json') != result['trial_sha256'] or
                sha(folder/'metrics.json') != result['metrics_sha256']):
            continue
        path.unlink()
        deleted.append(dict(path=str(path), kind=entry['kind'], bytes=entry['stat']['allocated_bytes']))
    previous = read(folder/'payload_pruned.json') if (folder/'payload_pruned.json').exists() else {'cleanups': []}
    previous['cleanups'].append(dict(deleted=deleted, unix_time=time.time()))
    write(folder/'payload_pruned.json', previous)
    marker.unlink()
    if deleted:
        emit('payload_pruned', directory=str(folder), files=len(deleted),
             bytes=sum(r['bytes'] for r in deleted))
    return previous
