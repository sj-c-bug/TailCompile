"""Single-core paired trial: charge fresh GNN compute + VPR, not loading."""
from __future__ import annotations

import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from types import SimpleNamespace

from bridge import read, write


THREAD_VARIABLES = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def _pin(cpu):
    for key in THREAD_VARIABLES:
        os.environ[key] = '1'
    os.sched_setaffinity(0, {cpu})
    allowed = sorted(os.sched_getaffinity(0))
    if allowed != [cpu]:
        raise RuntimeError('CPU affinity not applied')
    return dict(cpu=cpu, allowed_cpus=allowed,
                thread_env={key: os.environ[key] for key in THREAD_VARIABLES})


def _wait_gate(spec, folder):
    if not spec.get('start_gate'):
        return 0.
    timeout = float(spec.get('setup_timeout', 1800.))
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('setup_timeout must be positive and finite')
    gate = Path(spec['start_gate'])
    ready = Path(spec.get('ready_file', folder / 'ready.json'))
    write(ready, dict(ready=True, pid=os.getpid(), cpu=spec['cpu'], method=spec['method']))
    started = time.perf_counter()
    while True:
        if gate.exists():
            command = read(gate)
            if command.get('abort'):
                raise RuntimeError(f'Timing start aborted: {command["abort"]}')
            if command.get('go') is True:
                return time.perf_counter() - started
        if time.perf_counter() - started > timeout:
            raise TimeoutError(f'Timed worker setup gate was not released: {gate}')
        time.sleep(.05)


def _generate(spec, folder):
    # Pin the worker and set thread counts before importing PyTorch.
    from generate import generate
    return generate(spec['context'], spec['checkpoint'], folder / 'initializer',
                    spec['options'], spec.get('candidate', 0), reuse=False,
                    before_inference=lambda: _wait_gate(spec, folder))


def _stop_vpr(process):
    """Terminate the owned child group, including any hook subprocesses."""
    def send(sig):
        if os.name == 'posix':
            try:
                os.killpg(process.pid, sig)
            except ProcessLookupError:
                pass
        elif process.poll() is None:
            (process.terminate if sig == signal.SIGTERM else process.kill)()

    send(signal.SIGTERM)
    try:
        process.wait(timeout=5.)
    except subprocess.TimeoutExpired:
        pass
    # A descendant can outlive a leader which honored SIGTERM. Kill the owned
    # group even when the leader already exited, rather than leaving a hook.
    send(signal.SIGKILL if os.name == 'posix' else signal.SIGTERM)
    if process.poll() is None:
        process.kill()
    process.wait()


def _run_vpr(command, env, timeout, folder):
    """Use a child group, forwarding the outer watchdog's termination to it."""
    if timeout <= 0:
        return SimpleNamespace(returncode=124, timed_out=True, launched=False)
    launched_at = time.perf_counter()
    process = subprocess.Popen(command, env=env, start_new_session=(os.name == 'posix'))
    previous = {}

    def interrupted(signum, _frame):
        _stop_vpr(process)
        raise SystemExit(128 + signum)

    try:
        for sig in (signal.SIGTERM, signal.SIGINT):
            previous[sig] = signal.signal(sig, interrupted)
        write(folder / 'vpr_process.json', dict(pid=process.pid, worker_pid=os.getpid(),
            separate_process_group=(os.name == 'posix'), timeout_seconds=timeout))
        try:
            code = process.wait(timeout=max(0., timeout - (time.perf_counter() - launched_at)))
        except subprocess.TimeoutExpired:
            _stop_vpr(process)
            return SimpleNamespace(returncode=124, timed_out=True, launched=True)
        return SimpleNamespace(returncode=code, timed_out=False, launched=True)
    except BaseException:
        if process.poll() is None:
            _stop_vpr(process)
        raise
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)


def run_spec(spec, folder):
    started = time.perf_counter()
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    affinity = _pin(spec['cpu'])
    write(folder / 'affinity.json', affinity)
    env = {key: value for key, value in os.environ.items() if not key.startswith('TC_GP_')}
    generation = None
    gate_wait_seconds = 0.
    method = spec['method']
    if method not in ('default', 'fast'):
        raise ValueError(f'Unsupported timing method: {method}')
    measured_timeout = float(spec.get('measured_timeout', 7200.))
    if not math.isfinite(measured_timeout) or measured_timeout <= 0:
        raise ValueError('measured_timeout must be positive and finite')
    if method == 'fast':
        generation = _generate(spec, folder)
        if generation['ap_init_sha256'] != spec['expected_ap_sha256']:
            raise ValueError('Timed inference differs from the frozen scan candidate')
        env['TC_GP_INIT'] = generation['ap_init']
        gate_wait_seconds = generation.get('gate_wait_seconds', 0.)
    else:
        gate_wait_seconds = _wait_gate(spec, folder)
    compute = generation['compute_generation_seconds'] if generation else 0.
    if not math.isfinite(compute) or compute < 0:
        raise ValueError('compute_generation_seconds must be nonnegative and finite')
    # Both arms receive the same billable budget. Preloading/gate wait do not
    # consume it; GNN inference and decoding reduce the time left for VPR.
    remaining = max(0., measured_timeout - compute)
    before_vpr = time.perf_counter()
    result = _run_vpr(spec['vpr_command'], env, remaining, folder)
    vpr_seconds = time.perf_counter() - before_vpr
    worker_wall_seconds = time.perf_counter() - started
    cost = dict(vpr_seconds=vpr_seconds, compute_generation_seconds=compute,
        generation_seconds=compute, total_seconds=vpr_seconds + compute,
        excluded_setup_seconds=generation['excluded_setup_seconds'] if generation else 0.,
        gate_wait_seconds=gate_wait_seconds,
        worker_wall_seconds=worker_wall_seconds,
        excluded_worker_overhead_seconds=max(0., worker_wall_seconds - vpr_seconds - compute),
        returncode=result.returncode, timed_out=result.timed_out,
        vpr_launched=result.launched, measured_timeout_seconds=measured_timeout,
        vpr_remaining_budget_seconds=remaining,
        timeout_scope='GNN compute plus VPR wall; setup and gate wait excluded; termination grace included in observed wall',
        generation=generation,
        rrg_encoding_seconds=generation['timing']['rrg_encoding'] if generation else 0.,
        rrg_encoding_included_in_compute=True,
        runtime_scope='Preloaded-input GNN compute (both encoders + decode/repair/lowering) + full VPR subprocess wall; startup/loading excluded')
    write(folder / 'cost.json', cost)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', type=Path, required=True)
    args = parser.parse_args()
    return run_spec(read(args.spec), args.spec.parent)


if __name__ == '__main__':
    raise SystemExit(main())
