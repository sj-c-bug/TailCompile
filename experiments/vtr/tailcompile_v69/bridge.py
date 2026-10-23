"""Reuse the audited v68 hook/process code without changing frozen v68 files."""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
V68 = HERE.with_name('tailcompile_v68')
REPO = HERE.parents[2]
for path in (V68.parent, V68):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common import emit, execute, fingerprint, freeze, read, recover, sha, write
import experiment as legacy_experiment
import parallel_timing as legacy_timing
from lifecycle import check_free_space, log_storage_failure, StoragePressureError, parallel


def prune_trial(folder, out):
    """Only completed v69 route payloads; keep Git/input/active-file protections."""
    import os
    import time
    import storage
    folder, out = Path(folder).resolve(), Path(out).resolve()
    if os.environ.get('TAILCOMPILE_V69_PRUNE_COMPLETED', '1') == '0':
        return
    folder.relative_to(out)
    out.relative_to(REPO / 'experiments/vtr/tailcompile_v69/runs')
    tracked = storage.tracked_files(REPO, folder=folder)
    result = storage.inspect_trial(folder, tracked=tracked)
    if not result['eligible'] or not result['entries']:
        return
    marker = folder / 'payload_prune_pending.json'
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
    old = read(folder/'payload_pruned.json') if (folder/'payload_pruned.json').exists() else {'cleanups': []}
    old['cleanups'].append(dict(deleted=deleted, unix_time=time.time()))
    write(folder/'payload_pruned.json', old)
    marker.unlink()
    emit('payload_pruned', directory=str(folder), files=len(deleted), bytes=sum(r['bytes'] for r in deleted))
