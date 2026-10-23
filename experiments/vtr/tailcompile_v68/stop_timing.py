"""Preview/stop only this cohort's selected timing coordinators and VPR groups.

No name-based pkill and no signal to an unverified/reused PID. Completed files
are not touched. Linux /proc and same-user process inspection are required.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import signal
import socket
import time

from common import emit, read, write

HERE = Path(__file__).resolve().parent


def inspect_process(pid):
    try:
        proc = Path('/proc') / str(pid)
        if pid <= 1 or pid == os.getpid() or proc.stat().st_uid != os.geteuid():
            return None
        raw = (proc/'stat').read_text()
        fields = raw[raw.rfind(')')+2:].split()
        if fields[0] == 'Z':
            return None
        argv = [x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        return dict(pid=pid, start_ticks=fields[19], command=argv,
                    cwd=str((proc/'cwd').resolve(strict=True)), pgid=os.getpgid(pid))
    except (OSError, ValueError, IndexError):
        return None


def argument(argv, name, default=None):
    try:
        return argv[argv.index(name)+1]
    except (ValueError, IndexError):
        return default


def coordinator(record, out, mode, serial_name, parallel_name):
    argv = record['command']
    target = argument(argv, '--out')
    if target is None or Path(target).resolve() != out:
        return False
    for script, phase, option, name, default in (
        ('experiment.py', 'timing', '--timing-name', serial_name, 'serial_001'),
        ('parallel_timing.py', 'run', '--run-name', parallel_name, 'paired_001')):
        if mode != 'both' and (mode == 'serial') != (script == 'experiment.py'):
            continue
        for i, token in enumerate(argv[:4]):
            if (Path(token).name == script and Path(record['cwd'], token).resolve() == HERE/script and
                argv[i+1:i+2] == [phase] and argument(argv, option, default) == name):
                return True
    return False


def targets(out, mode, serial_name, parallel_name):
    roots = []
    if mode in ('serial', 'both'):
        roots.append((out/'timing'/serial_name).resolve())
    if mode in ('parallel', 'both'):
        roots.append((out/'timing_parallel'/parallel_name/'waves').resolve())
    tasks = {}
    for root in roots:
        for started in root.rglob('started.json'):
            if '_attempts' in started.parts or '_interrupted' in started.parts:
                continue
            if (started.parent/'trial.json').exists():
                continue
            record = read(started)
            if record.get('host') == socket.gethostname():
                tasks[str(started.parent.resolve())] = record
    runners, workers = [], []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        record = inspect_process(int(path.name))
        if record is None:
            continue
        if coordinator(record, out, mode, serial_name, parallel_name):
            runners.append(record)
        # common.execute makes the affinity helper/VPR its own group leader.
        elif record['pid'] == record['pgid'] and record['cwd'] in tasks:
            task = tasks[record['cwd']]
            command = task.get('command', [])
            accepted = [command]
            if str(HERE/'affinity_exec.py') in command and '--' in command:
                accepted.append(command[command.index('--')+1:])
            # Check both folder and recorded argv/PID. Do not stop a user's
            # unrelated shell/editor merely because it has this working dir.
            if (record['command'] in accepted and
                (not task.get('child_pid') or task['child_pid'] == record['pid'])):
                workers.append(record)
    return runners, workers


def send(record, sig, group=False):
    current = inspect_process(record['pid'])
    if current is None:
        return False
    if any(current[key] != record[key] for key in ('start_ticks', 'command', 'cwd', 'pgid')):
        raise RuntimeError(f'PID identity changed; refusing signal: {record["pid"]}')
    try:
        if group:
            if current['pgid'] != current['pid']:
                raise RuntimeError('Refuse to signal a shared process group')
            os.killpg(current['pid'], sig)
        else:
            os.kill(current['pid'], sig)
        return True
    except ProcessLookupError:
        return False


def stop(out, mode='serial', serial_name='serial_001', parallel_name='paired_001', apply=False):
    out = Path(out).resolve()
    for name in (serial_name, parallel_name):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name):
            raise ValueError('Simple timing run names required')
    runners, workers = targets(out, mode, serial_name, parallel_name)
    emit('timing_stop_plan', apply=apply, mode=mode, coordinators=runners, worker_groups=workers,
         note='This host only; completed timing files are preserved.')
    if not apply:
        return
    for record in runners:
        send(record, signal.SIGTERM)
    deadline = time.monotonic()+3
    while time.monotonic() < deadline and any(inspect_process(r['pid']) for r in runners):
        time.sleep(.1)
    # Re-inspect after stopping dispatch to catch a child launched during preview.
    more_runners, workers = targets(out, mode, serial_name, parallel_name)
    if more_runners:
        raise RuntimeError('Timing coordinator is still running; inspect before stopping children')
    for record in workers:
        send(record, signal.SIGTERM, group=True)
    deadline = time.monotonic()+5
    while time.monotonic() < deadline and any(inspect_process(r['pid']) for r in workers):
        time.sleep(.1)
    for record in workers:
        send(record, signal.SIGKILL, group=True)
    remaining = targets(out, mode, serial_name, parallel_name)
    result = dict(mode=mode, stopped_coordinators=runners, stopped_worker_groups=workers,
                  remaining_coordinators=remaining[0], remaining_worker_groups=remaining[1],
                  completed_files_untouched=True, unix_time=time.time())
    path = out/'timing_stop'/f'{time.time_ns()}.json'
    write(path, result)
    emit('timing_stopped', log=str(path), complete=not any(remaining))
    if any(remaining):
        raise RuntimeError('Some scoped timing processes remain; do not start another measurement')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--mode', choices=['serial','parallel','both'], default='serial')
    p.add_argument('--serial-name', default='serial_001')
    p.add_argument('--parallel-name', default='paired_001')
    p.add_argument('--apply', action='store_true', help='Without this flag only print the exact scoped targets')
    a = p.parse_args()
    if os.name != 'posix' or not Path('/proc').is_dir():
        p.error('Linux same-user /proc inspection required')
    stop(a.out, a.mode, a.serial_name, a.parallel_name, a.apply)


if __name__ == '__main__':
    main()
