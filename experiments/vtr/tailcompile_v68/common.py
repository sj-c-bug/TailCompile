"""Small, resumable process/provenance helpers for the v6.8 experiment.

Timeouts are censored observations, not unroutable labels. A completed success
is never overwritten. Retry flags preserve an old attempt before re-running.
"""
from __future__ import annotations

import concurrent.futures as cf
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time


COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'
HERE = Path(__file__).resolve().parent
VTR_HERE = HERE.parent
ROUTE_OUTCOMES = {'success', 'unroutable'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2,
                                    allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def emit(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False, allow_nan=False), flush=True)


def freeze(path, value):
    path = Path(path)
    if path.exists() and read(path) != value:
        raise ValueError(f'Frozen inputs changed: {path}; use a new output directory')
    if not path.exists():
        write(path, value)


def parallel(function, tasks, jobs):
    if jobs < 1:
        raise ValueError('jobs must be positive')
    results, failures = [], []
    with cf.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(function, task) for task in tasks]
        for future in cf.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(str(error))
                emit('task_failed', error=str(error))
    if failures:
        raise RuntimeError(f'{len(failures)} tasks failed; completed evidence retained. First: {failures[0]}')
    return results


def classify(code, text):
    if code == 124:
        return 'timeout'
    if code in (-9, 137):
        return 'killed'
    if code == 0 and 'VPR succeeded' in text:
        return 'success'
    if code in (1, 2) and 'Circuit is unroutable with a channel width factor' in text:
        return 'unroutable'
    return 'other_failure'


def clean_env(extra=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith('TC_GP_')}
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               NUMEXPR_NUM_THREADS='1')
    env.update({k: str(v) for k, v in (extra or {}).items()})
    return env


def execute(command, folder, identity, *, env=None, timeout=7200, heartbeat=60,
            retry_failed=False, retry_timeouts=False, required_marker=None):
    """Stream a process to disk; resume only an exactly matching trial identity."""
    folder = Path(folder).resolve()
    folder.mkdir(parents=True, exist_ok=True)
    command = [str(x) for x in command]
    identity = dict(identity, command=command,
                    hook_env={k:str(v) for k,v in sorted((env or {}).items())})
    record, started = folder / 'trial.json', folder / 'started.json'
    if record.exists():
        previous = read(record)
        if previous['identity'] != identity:
            raise ValueError(f'Trial input identity changed: {folder}')
        status = previous['status']
        retry = (status == 'timeout' and retry_timeouts) or (
            status not in ROUTE_OUTCOMES | {'complete', 'timeout'} and retry_failed)
        if not retry:
            emit('reuse_trial', directory=str(folder), status=status)
            return previous
        archive = folder / '_attempts' / str(time.time_ns())
        archive.mkdir(parents=True)
        for name in ('trial.json', 'started.json', 'vpr.out'):
            source = folder / name
            if source.exists():
                source.replace(archive / name)
    elif started.exists():
        # An explicit recovery is necessary: another host's worker might be alive.
        raise RuntimeError(f'Interrupted/active trial: {folder}; inspect workers then run recover')
    metadata = dict(identity=identity, command=command, host=socket.gethostname(),
                    parent_pid=os.getpid(), started_unix=time.time(), timeout=timeout)
    write(started, metadata)
    emit('process_start', directory=str(folder), command=command)
    begin = time.monotonic()
    with (folder / 'vpr.out').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=folder, env=clean_env(env), stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=(os.name != 'nt'))
        write(started, dict(metadata, child_pid=process.pid))
        next_heartbeat = begin + heartbeat
        while process.poll() is None:
            now = time.monotonic()
            if now - begin >= timeout:
                if os.name == 'nt':
                    process.terminate()
                else:
                    os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if os.name == 'nt':
                        process.kill()
                    else:
                        os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                code = 124
                break
            if now >= next_heartbeat:
                emit('process_heartbeat', directory=str(folder), pid=process.pid,
                     elapsed_seconds=round(now - begin, 1))
                next_heartbeat = now + heartbeat
            try:
                process.wait(timeout=min(.05, max(.001, timeout - (now - begin))))
            except subprocess.TimeoutExpired:
                pass
        else:
            code = process.returncode
    text = (folder / 'vpr.out').read_text(encoding='utf-8', errors='replace')
    status = classify(code, text)
    if required_marker and code == 0 and required_marker in text:
        status = 'complete'
    elif required_marker and status == 'success':
        status = 'other_failure'
    result = dict(metadata, returncode=code, status=status, wall_seconds=time.monotonic()-begin,
                  completed_unix=time.time())
    write(record, result)
    emit('process_complete', directory=str(folder), status=status,
         seconds=round(result['wall_seconds'], 3), returncode=code)
    return result


def recover(root, confirm_remote_stopped=False):
    """Only quarantine incomplete task directories after same-host PID checks."""
    root = Path(root).resolve()
    moved = []
    for started in sorted(root.rglob('started.json')):
        if '_attempts' in started.parts or '_interrupted' in started.parts:
            continue
        folder = started.parent
        if (folder / 'trial.json').exists():
            continue
        info = read(started)
        if info.get('host') == socket.gethostname() and info.get('child_pid'):
            try:
                os.kill(info['child_pid'], 0)
            except ProcessLookupError:
                pass
            else:
                raise RuntimeError(f'Worker may still be alive: {folder}, PID={info["child_pid"]}')
        elif info.get('host') != socket.gethostname() and not confirm_remote_stopped:
            raise RuntimeError(f'Cannot verify worker on {info.get("host")}; inspect it before recovery: {folder}')
        target = root / '_interrupted' / str(time.time_ns()) / folder.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        folder.rename(target)
        moved.append(str(target))
    emit('recovered', count=len(moved), archived=moved)
    return moved
