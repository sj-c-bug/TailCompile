"""Small immutable manifests and bounded subprocesses for isolated 3D trials."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import threading
import time

try:
    from .repository_guard import validate_output_roots
except ImportError:
    from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tailcompile_v69.phase_metrics import parse_phase_metrics

COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'

_cancelled = threading.Event()
_process_lock = threading.RLock()
_owned_processes = {}
_stop_signal = None
_handlers_installed = False


def _signal_owned(process, signum):
    """Only process groups created by this module are eligible for signalling."""
    try:
        if os.name == 'posix':
            os.killpg(process.pid, signum)
        elif signum == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()
    except ProcessLookupError:
        pass


def _request_stop(signum, frame):
    global _stop_signal
    # Guard reentrant SIGINT/SIGTERM before Event.set takes its internal lock.
    if _stop_signal is not None:
        return
    _stop_signal = int(signum)
    _cancelled.set()
    with _process_lock:
        owned = list(_owned_processes.values())
    for process in owned:
        _signal_owned(process, signal.SIGTERM)


def install_signal_handlers():
    """Install cooperative TERM/INT handling from a CLI's main thread.

    Worker threads can share the event/registry, but cannot install Python
    signal handlers. An uncatchable SIGKILL still needs scheduler cleanup.
    """
    global _handlers_installed
    if threading.current_thread() is not threading.main_thread():
        return False
    if not _handlers_installed:
        signal.signal(signal.SIGTERM, _request_stop)
        signal.signal(signal.SIGINT, _request_stop)
        _handlers_installed = True
    return True


def raise_if_cancelled():
    if _cancelled.is_set():
        raise KeyboardInterrupt(f'campaign interrupted by signal {_stop_signal}')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def freeze(path, value):
    if Path(path).exists():
        if read(path) != value:
            raise ValueError(f'frozen input changed: {path}; use a new versioned output directory')
    else:
        write(path, value)
    return value


def emit(event, **values):
    print(json.dumps(dict(event=event, **values), ensure_ascii=False), flush=True)


def attempt_path_command(command, work, options):
    """Resolve declared relative output arguments inside this exact attempt.

    The signature keeps the logical relative argument, while subprocesses such
    as VTR's frontend receive an absolute path even after changing their cwd.
    """
    actual = list(command)
    work = Path(work).resolve()
    for option in options:
        if actual.count(option) != 1:
            raise ValueError(f'attempt path option must occur exactly once: {option}')
        index = actual.index(option) + 1
        if index >= len(actual) or not actual[index] or actual[index].startswith('-'):
            raise ValueError(f'attempt path option needs a relative path: {option}')
        relative = Path(actual[index])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'attempt path must stay inside its attempt: {option}')
        target = (work / relative).resolve()
        if target != work and work not in target.parents:
            raise ValueError(f'attempt path escapes through a symlink: {option}')
        actual[index] = str(target)
    return actual


def run_trial(folder, command, *, inputs=(), env=None, timeout=300, marker=None,
              metadata=None, affinity=None, prune=True, retry_infrastructure=False,
              attempt_path_options=()):
    """Resume completed exact signatures; never silently turn an error into a label."""
    install_signal_handlers()
    raise_if_cancelled()
    folder = Path(folder).resolve()
    validate_output_roots((folder,))
    command = list(map(str, command))
    signature = dict(command=command, inputs={str(Path(p).resolve()): sha(p) for p in inputs},
                     env={k: str(v) for k, v in (env or {}).items()}, timeout=timeout,
                     marker=marker, metadata=metadata or {}, affinity=affinity)
    options = tuple(attempt_path_options)
    if len(set(options)) != len(options):
        raise ValueError('duplicate attempt path options')
    if options:
        # Validate before creating a trial directory. Do not alter historical
        # signatures for callers which do not request this path policy.
        attempt_path_command(command, folder/'attempts'/'validation', options)
        signature['attempt_path_options'] = list(options)
    saved = folder / 'manifest.json'
    if saved.exists():
        row = read(saved)
        if row['signature'] != signature:
            raise ValueError(f'trial signature mismatch: {folder}')
        retry = row['status'] == 'infrastructure_failure' and (retry_infrastructure or
            os.environ.get('TAILCOMPILE_V7_RETRY_INFRASTRUCTURE') == '1')
        if retry and len(list((folder/'attempts').glob('attempt_*'))) >= 2:
            raise RuntimeError(f'infrastructure retry limit reached; inspect retained attempts: {folder}')
        if row['status'] in ('success', 'unroutable', 'complete', 'timeout', 'unsupported',
                             'packing_or_placement_failure', 'infrastructure_failure') and not retry:
            emit('trial_cached', folder=str(folder), status=row['status'])
            return row
    folder.mkdir(parents=True, exist_ok=True)
    attempts = folder / 'attempts'
    attempts.mkdir(exist_ok=True)
    work = attempts / f'attempt_{len(list(attempts.iterdir())) + 1:04d}'
    work.mkdir()
    actual = attempt_path_command(command, work, options)
    if affinity is not None:
        actual = ['taskset', '-c', str(affinity)] + actual
    timed = os.name == 'posix' and Path('/usr/bin/time').is_file()
    if timed:
        actual = ['/usr/bin/time', '-v', '-o', str(work / 'resources.txt')] + actual
    variables = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    for name in tuple(variables):
        if name.startswith('TC_GP_'):
            del variables[name]
    variables.update(signature['env'])
    started = time.perf_counter()
    write(work / 'started.json', dict(signature=signature, executed_command=actual,
                                     command_cwd=str(work), pid=os.getpid(), unix_time=time.time()))
    with (work / 'vpr.out').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(actual, cwd=work, stdout=log, stderr=subprocess.STDOUT,
                                   env=variables, start_new_session=(os.name == 'posix'))
        with _process_lock:
            _owned_processes[process.pid] = process
        try:
            deadline = time.monotonic() + timeout
            while True:
                if _cancelled.is_set():
                    _signal_owned(process, signal.SIGKILL if os.name == 'posix' else signal.SIGTERM)
                    process.wait()
                    code = 'interrupted'
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(actual, timeout)
                try:
                    code = process.wait(timeout=min(remaining, .2))
                    break
                except subprocess.TimeoutExpired:
                    continue
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            _signal_owned(process, signal.SIGKILL if os.name == 'posix' else signal.SIGTERM)
            process.wait()
            code = 'timeout' if isinstance(exc, subprocess.TimeoutExpired) else 'interrupted'
        finally:
            if _cancelled.is_set():
                # The /usr/bin/time leader can exit on TERM before a stubborn
                # descendant. Escalate the same known group before retiring it.
                _signal_owned(process, signal.SIGKILL if os.name == 'posix' else signal.SIGTERM)
                process.wait()
                code = 'interrupted'
            with _process_lock:
                _owned_processes.pop(process.pid, None)
    elapsed = time.perf_counter() - started
    output = (work / 'vpr.out').read_text(encoding='utf-8', errors='replace')
    routed = code == 0 and 'Circuit successfully routed' in output
    normal_failure = any(s in output.lower() for s in
        ('circuit is unroutable', 'failed to route', 'routing failed', 'circuit could not be routed'))
    memory_or_external_kill = code in (-9, 137) or any(s in output.lower() for s in
        ('std::bad_alloc', 'cannot allocate memory', 'out of memory', 'memoryerror:'))
    status = ('timeout' if code == 'timeout' else 'interrupted' if code == 'interrupted' else
              'success' if routed else 'complete' if code == 0 else
              'infrastructure_failure' if memory_or_external_kill else
              'unroutable' if normal_failure and code == 2 else
              'infrastructure_failure' if code in (126, 127) else 'packing_or_placement_failure')
    if marker and code == 0 and marker not in output:
        status = 'infrastructure_failure'
    cpd = re.search(r'Final critical path delay.*?:\s*([\d.eE+-]+) ns', output)
    wire = re.search(r'Total wirelength:\s*([\d.eE+-]+)', output)
    # Keep the GP table alive across hook observations without changing line numbers.
    phase_text = re.sub(r'^.*TAILCOMPILE_(?:V7|V68)_[^\n]*', '', output, flags=re.MULTILINE)
    phases = parse_phase_metrics(phase_text)
    metrics = dict(status=status, wall_seconds=elapsed,
                   failure_reason='unattributed_memory_or_external_kill' if memory_or_external_kill else None,
                   interruption_signal=_stop_signal if status == 'interrupted' else None,
                   cpd_ns=float(cpd[1]) if routed and cpd else None,
                   wirelength=float(wire[1]) if routed and wire else None, phases=phases)
    if timed and (work / 'resources.txt').exists():
        resources = (work / 'resources.txt').read_text(errors='replace')
        for label, key in (('Maximum resident set size (kbytes)', 'peak_rss_kib'),
                           ('User time (seconds)', 'user_cpu_seconds'),
                           ('System time (seconds)', 'system_cpu_seconds')):
            match = re.search(re.escape(label) + r':\s*([\d.]+)', resources)
            metrics[key] = float(match[1]) if match else None
    routes = list(work.glob('*.route'))
    metrics['unique_chanz_nodes'] = (len(set(re.findall(r'Node:\s*(\d+)\s+CHANZ\b',
        routes[0].read_text(errors='replace')))) if routes and routed else None)
    row = dict(signature=signature, status=status, returncode=code, workdir=str(work),
               log=str(work / 'vpr.out'), metrics=metrics, completed_unix=time.time())
    write(work / 'metrics.json', metrics)
    write(saved, row)
    if prune and status in ('success', 'unroutable', 'complete'):
        prune_derived(work, protected=inputs)
    emit('trial_finished', folder=str(folder), status=status, seconds=round(elapsed, 3))
    if code == 'interrupted':
        raise KeyboardInterrupt
    return row


def prune_derived(work, *, protected=()):
    """Reclaim only owned completed-trial payloads after metrics are durable.

    Inputs, AP teachers, flat placement traces and main RRG caches are outside
    this allowlist. File suffix alone never permits removing an input .net.
    """
    work = Path(work).resolve()
    if not (work/'metrics.json').is_file() or not (work/'started.json').is_file():
        raise ValueError('refuse payload cleanup before durable trial evidence')
    protected = {Path(p).resolve() for p in protected}
    removed = []
    for path in work.iterdir():
        allowed = (path.name.endswith(('.net', '.net.post_routing', '.net_postrouting',
                                        '.net.postrouting', '.route')))
        if not allowed or path.is_symlink() or not path.is_file() or path.resolve() in protected:
            continue
        if path.resolve().parent != work:
            raise ValueError('cleanup escaped exact trial directory')
        removed.append(dict(name=path.name, bytes=path.stat().st_size, sha256=sha(path)))
        path.unlink()
    write(work/'payload_pruned.json', dict(removed=removed,
        removed_bytes=sum(p['bytes'] for p in removed),
        reason='completed trial metrics durable; no downstream input or teacher dependency',
        raw_route_reanalysis_requires_rerun=bool(any(p['name'].endswith('.route') for p in removed))))


def vpr_command(binary, arch, blif, width, seed, *, rr_graph=None):
    command = [str(binary), str(arch), str(blif), '--analytical_place', '--route', '--analysis',
               '--ap_full_legalizer', 'appack', '--ap_detailed_placer', 'annealer',
               '--flat_routing', 'off', '--num_workers', '1', '--seed', str(seed),
               '--route_chan_width', str(width)]
    if rr_graph:
        command += ['--read_rr_graph', str(rr_graph)]
    return command
