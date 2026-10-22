"""Bound transient VPR payload lifetime without changing a trial's inputs.

Filesystem free space is NOT a user's quota.  EDQUOT/ENOSPC still halt further
dispatch, drain active workers, and leave complete trials reusable.
"""
from __future__ import annotations

import concurrent.futures as cf
import errno
import math
import os
from pathlib import Path
import re
import shutil

from common import emit, sha, write

DISK_ERRNOS = {errno.ENOSPC, 122, getattr(errno, 'EDQUOT', 122)}


class StoragePressureError(RuntimeError):
    pass


def storage_error(error):
    return isinstance(error, StoragePressureError) or (
        isinstance(error, OSError) and error.errno in DISK_ERRNOS)


def minimum_free_gib():
    value = float(os.environ.get('TAILCOMPILE_V68_MIN_FREE_GIB', '10'))
    if value < 0 or not math.isfinite(value):
        raise ValueError('TAILCOMPILE_V68_MIN_FREE_GIB must be finite and nonnegative')
    return value


def check_free_space(folder):
    """Quick shared-filesystem guard; never claims to measure a user quota."""
    folder = Path(folder).resolve()
    while not folder.exists():
        folder = folder.parent
    free = shutil.disk_usage(folder).free
    reserve = minimum_free_gib() * 1024**3
    if free < reserve:
        raise StoragePressureError(f'Filesystem free space {free/1024**3:.2f} GiB '
            f'is below {reserve/1024**3:.2f} GiB reserve; clean storage before resuming. '
            'This filesystem check does not measure user/project quota.')
    return free


def log_storage_failure(text, returncode):
    return returncode != 0 and bool(re.search(
        r'disk quota exceeded|no space left on device|\bEDQUOT\b|\bENOSPC\b', text, re.I))


def retention_enabled():
    value = os.environ.get('TAILCOMPILE_V68_PRUNE_COMPLETED', '1')
    if value not in ('0', '1'):
        raise ValueError('TAILCOMPILE_V68_PRUNE_COMPLETED must be 0 or 1')
    return value == '1'


def record_policy(out):
    write(Path(out)/'storage_policy.json', dict(
        schema='tailcompile-v68-storage-policy-1',
        prune_completed_vpr_payloads=retention_enabled(),
        minimum_filesystem_free_gib=minimum_free_gib(),
        quota_visible=False,
        implementation_sha256={name:sha(Path(__file__).with_name(name))
                               for name in ('lifecycle.py','storage.py')},
        prune_after='Closed VPR child, trial.json and parsed metrics.json committed',
        keep=['canonical inputs', 'gp/structure TSVs', 'graphs and IR', 'RRG caches',
              'checkpoints', 'generated initializer and targets', 'JSON/CSV evidence', 'logs'],
        runtime_note='Raw-output cleanup is after the recorded VPR duration; not a VPR speedup.',
        deletion='Permanent unlink of reproducible per-trial raw payload only'))
    emit('storage_policy', prune_completed=retention_enabled(),
         minimum_filesystem_free_gib=minimum_free_gib(), quota_visible=False)


def after_metrics(folder, *, repo=None):
    if not retention_enabled():
        return None
    from storage import prune_completed_trial
    result = prune_completed_trial(folder, repo=repo)
    if result.get('deleted_files'):
        emit('payload_pruned', directory=str(folder), deleted_files=result['deleted_files'],
             released_GiB=round(result['released_bytes']/1024**3, 4),
             note='Raw derived outputs only; compact metrics and inputs retained.')
    return result


def parallel(function, tasks, jobs):
    """Keep only jobs tasks in flight; stop submitting after a disk failure.

    Other task failures retain the historical collect-all behavior. Active VPR
    jobs are NOT killed when a sibling encounters quota pressure: they finish,
    persist evidence and reclaim their own outputs if possible.
    """
    if jobs < 1:
        raise ValueError('jobs must be positive')
    iterator = iter(tasks)
    results, failures, pressure = [], [], None
    submitted = 0
    with cf.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = set()

        def submit_one():
            nonlocal submitted
            try:
                task = next(iterator)
            except StopIteration:
                return False
            futures.add(executor.submit(function, task))
            submitted += 1
            return True

        for _ in range(jobs):
            if not submit_one():
                break
        while futures:
            done, remaining = cf.wait(futures, return_when=cf.FIRST_COMPLETED)
            futures = remaining
            for future in done:
                try:
                    results.append(future.result())
                except Exception as error:
                    failures.append(str(error))
                    if storage_error(error):
                        pressure = error
                    emit('task_failed', error=str(error), storage_pressure=storage_error(error))
            if pressure is None:
                for _ in done:
                    if not submit_one():
                        break
            else:
                emit('storage_dispatch_paused', submitted=submitted, active=len(futures),
                     completed=len(results), failed=len(failures),
                     note='No new tasks dispatched; active jobs drain. Completed evidence is retained.')
    if pressure is not None:
        raise StoragePressureError(f'Storage quota/free-space failure; stopped after submitting '
            f'{submitted} tasks. Complete trials retained. Clean, recover interrupted jobs '
            f'after checking workers, then retry infrastructure failures. First: {pressure}') from pressure
    if failures:
        raise RuntimeError(f'{len(failures)} tasks failed; completed evidence retained. First: {failures[0]}')
    return results
