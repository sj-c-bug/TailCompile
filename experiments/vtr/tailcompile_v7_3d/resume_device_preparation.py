"""Finish preparation-only cache work and resume the unchanged calibration CLI.

Only known extractor deadlines can trigger a retry, and each retry must publish
at least one new complete device cache. P&R failures retain their original labels.
"""
from __future__ import annotations

import argparse
from collections import deque
from contextlib import contextmanager
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

from io_utils import read, sha, write
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_PREPARATION_TIMEOUT = re.compile(
    r'(?:^|\n)TimeoutError:\s*(?:complete device graph extraction exceeded its preparation budget|'
    r'3D supply preparation exceeded [0-9]+(?:\.[0-9]+)?s)\s*(?:\n|$)')


class ResumeFailure(RuntimeError):
    def __init__(self, message, returncode=1):
        super().__init__(message)
        self.returncode = returncode


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f'preparation resume output escapes its owner: {path}')
    return path


def _signal_owned(process, signum):
    try:
        if os.name == 'posix':
            os.killpg(process.pid, signum)
        elif signum == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _terminate_owned(process):
    """Retire only the process group created for this calibration attempt."""
    _signal_owned(process, signal.SIGTERM)
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # The shell leader can exit before descendants have completed cleanup.
        _signal_owned(process, signal.SIGKILL if os.name == 'posix' else signal.SIGTERM)
        process.wait(timeout=5)


class SignalRelay:
    def __init__(self):
        self.active = None
        self.signum = None

    def stop(self, signum, frame):
        if self.signum is not None:
            return
        self.signum = int(signum)
        if self.active is not None:
            _signal_owned(self.active, signal.SIGTERM)
        raise KeyboardInterrupt(f'preparation resume interrupted by signal {signum}')

    @contextmanager
    def installed(self):
        enabled = threading.current_thread() is threading.main_thread()
        old = {}
        if enabled:
            for signum in (signal.SIGTERM, signal.SIGINT):
                old[signum] = signal.getsignal(signum)
                signal.signal(signum, self.stop)
        try:
            yield self
        finally:
            for signum, handler in old.items():
                signal.signal(signum, handler)


def run_calibration(entry, log_path, environment, *, relay=None):
    """Tee a real stage process without interpreting its output as a P&R label."""
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    tail = deque(maxlen=150)
    command = ['bash', str(entry), '--stage', 'calibrate']
    with log_path.open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=Path(entry).parent, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding='utf-8', errors='replace', start_new_session=(os.name == 'posix'))
        if relay is not None:
            relay.active = process
        try:
            for line in process.stdout:
                tail.append(line)
                log.write(line)
                log.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            code = process.wait()
        except BaseException:
            _terminate_owned(process)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()
            if relay is not None:
                relay.active = None
    return dict(returncode=code, output_tail=''.join(tail), log=str(log_path), command=command)


def resume(out, entry=None, max_seconds=3600, run_stage=None, complete=None, *, repo=REPO):
    if isinstance(max_seconds, bool) or not math.isfinite(float(max_seconds)) or max_seconds <= 0:
        raise ValueError('positive finite preparation budget required')
    repo, out = Path(repo).resolve(), Path(out).resolve()
    module = repo/'experiments/vtr/tailcompile_v7_3d'
    if module/'runs' not in out.parents:
        raise ValueError('preparation resume must use a canonical v7 cohort')
    validate_output_roots((out,), repo=repo)
    entry = Path(entry or HERE/'prepare_server.sh').resolve()
    expected_entry = _inside(module/'prepare_server.sh', module)
    if entry != expected_entry:
        raise ValueError('preparation resume must use the canonical frozen stage entry')
    config = read(out/'campaign.json')
    widths = config.get('width_grid', [])
    grids = {tuple(record['grid']) for record in config.get('designs', [])}
    if not widths or not grids:
        raise ValueError('frozen calibration panel is empty')
    limit = 4*len(widths)*len(grids)+1
    if complete is None:
        from complete_device_cache import complete_pending
        complete = complete_pending
    relay = SignalRelay()
    if run_stage is None:
        run_stage = lambda e, p, env: run_calibration(e, p, env, relay=relay)
    run_root = _inside(out/'device_preparation_resume'/f'run_{time.time_ns()}_{os.getpid()}', out)
    latest = _inside(out/'device_preparation_resume_report.json', out)
    begun = time.perf_counter()
    state = dict(schema='tailcompile-v7-preparation-resume-1', status='preparing',
        preparation_only=True, extraction_budget_seconds=max_seconds,
        extraction_concurrency=1, retry_limit=limit, attempts=[], completed_cache_keys=[],
        code_sha256={Path(__file__).name:sha(__file__)}, started_unix=time.time(),
        protocol_sha256=sha(out/'protocol.json'), original_pnr_labels_unchanged=True)
    helper = HERE/'complete_device_cache.py'
    if helper.is_file():
        state['code_sha256'][helper.name] = sha(helper)
    report = run_root/'report.json'
    def persist():
        state['elapsed_seconds'] = time.perf_counter()-begun
        write(report, state)
        write(latest, dict(report=str(report), status=state['status'],
            completed_cache_keys=state['completed_cache_keys'], stage_attempt_count=len(state['attempts'])))
    env = dict(os.environ, TAILCOMPILE_V7_RRG_JOBS='1', TAILCOMPILE_V7_OUT=str(out),
        TAILCOMPILE_V7_CONFIG=str(out/'campaign.json'),
        TAILCOMPILE_3D_QUALIFICATION=str(out/'qualification'), PYTHONDONTWRITEBYTECODE='1',
        TAILCOMPILE_3D_BACKEND_READ_ONLY='1', GIT_OPTIONAL_LOCKS='0', PYTHONUNBUFFERED='1')
    seen = set()
    with relay.installed():
        # The helper verifies the frozen protocol before any controller output is
        # created, holds its own campaign lock, and releases it before dispatch.
        keys = complete(out, max_seconds=max_seconds)
        seen.update(keys)
        state['completed_cache_keys'] = sorted(seen)
        persist()
        try:
            for index in range(1, limit+1):
                state['status'] = 'calibrating'
                persist()
                log = _inside(run_root/f'calibrate_{index:03d}.log', out)
                result = run_stage(entry, log, env)
                code = int(result['returncode'])
                tail = str(result.get('output_tail', ''))
                budget_timeout = code == 1 and bool(_PREPARATION_TIMEOUT.search(tail))
                state['attempts'].append(dict(attempt=index, returncode=code, log=str(log),
                    known_preparation_timeout=budget_timeout))
                persist()
                if code == 0:
                    state.update(status='complete', completed_unix=time.time())
                    persist()
                    return state
                if not budget_timeout:
                    state['status'] = 'stopped_non_preparation_failure'
                    persist()
                    raise ResumeFailure(f'calibration failed outside device extraction; inspect {log}', code)
                state['status'] = 'preparing'
                persist()
                keys = complete(out, max_seconds=max_seconds)
                new_keys = set(keys)-seen
                if not new_keys:
                    state['status'] = 'stopped_no_preparation_progress'
                    persist()
                    raise ResumeFailure(f'preparation timeout published no new cache; inspect {log}', code)
                seen.update(new_keys)
                state['completed_cache_keys'] = sorted(seen)
                persist()
            state['status'] = 'stopped_retry_limit'
            persist()
            raise ResumeFailure('bounded preparation resume retry limit reached')
        except KeyboardInterrupt:
            state.update(status='interrupted', interruption_signal=relay.signum)
            persist()
            raise
        except ResumeFailure:
            raise
        except Exception as error:
            state.update(status='failed_preparation_or_controller', error_type=type(error).__name__, error=str(error))
            persist()
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--entry', type=Path)
    parser.add_argument('--max-seconds', type=float, default=3600)
    args = parser.parse_args()
    try:
        state = resume(args.out, args.entry, args.max_seconds)
    except ResumeFailure as error:
        print(str(error), file=sys.stderr, flush=True)
        return error.returncode if 0 < error.returncode < 256 else 1
    except KeyboardInterrupt:
        print('Device preparation resume interrupted; existing evidence retained.', file=sys.stderr, flush=True)
        return 130
    print(dict(status=state['status'], repaired_cache_keys=state['completed_cache_keys'],
               stage_attempts=len(state['attempts'])), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
