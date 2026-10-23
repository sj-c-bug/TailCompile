"""Separate, pinned AB/BA runtime measurements for the frozen v6.8 evaluation.

No generator/training/cohort source is changed. Completed synchronized waves
are reusable; a partial wave is archived as a unit, never mixed with a new load.
This measures latency under a specified concurrent load, not isolated latency.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import concurrent.futures as cf
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import sys
import threading
import time

import experiment as e
from common import clean_env, emit, execute, fingerprint, freeze, read, sha, write
from lifecycle import (StoragePressureError, after_metrics, check_free_space,
                       log_storage_failure, record_policy, storage_error)

HERE = Path(__file__).resolve().parent
METHODS = ['default', 'gnn_warmstart']


def read_text(path, default=None):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def topology():
    if not hasattr(os, 'sched_getaffinity'):
        raise RuntimeError('Controlled parallel timing requires Linux CPU affinity')
    rows = []
    for cpu in sorted(os.sched_getaffinity(0)):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}')
        package = read_text(base / 'topology/physical_package_id')
        core = read_text(base / 'topology/core_id')
        if package is None or core is None:
            raise RuntimeError('CPU topology unavailable; cannot claim physical-core isolation')
        nodes = sorted(base.glob('node[0-9]*'))
        rows.append(dict(cpu=cpu, package=int(package), core=int(core),
            numa=int(nodes[0].name[4:]) if nodes else None,
            siblings=read_text(base / 'topology/thread_siblings_list')))
    return rows


def quota_info(root=Path('/sys/fs/cgroup')):
    root = root.resolve()
    paths = {root}
    for line in (read_text('/proc/self/cgroup', '') or '').splitlines():
        if line.startswith('0::'):
            candidate = (root / line[3:].lstrip('/')).resolve()
            if candidate.is_relative_to(root):
                paths.add(candidate)
    for path in list(paths):
        paths.update(p for p in path.parents if p.is_relative_to(root))
    limits = []
    for path in paths:
        raw = read_text(path / 'cpu.max')
        if raw and raw.split()[0] != 'max':
            quota, period = map(int, raw.split())
            limits.append(quota / period)
    # Older hosts may expose v1 rather than v2 at the controller root.
    q = read_text(root / 'cpu/cpu.cfs_quota_us')
    p = read_text(root / 'cpu/cpu.cfs_period_us')
    if q and p and int(q) > 0:
        limits.append(int(q) / int(p))
    return min(limits) if limits else None


def parse_cpus(value):
    if value is None:
        return None
    if not isinstance(value, str):
        return list(value)
    cpus = []
    for part in value.split(','):
        if '-' in part:
            low, high = map(int, part.split('-'))
            if low < 0 or high < low:
                raise ValueError('Invalid CPU range')
            cpus.extend(range(low, high + 1))
        else:
            cpus.append(int(part))
    if len(set(cpus)) != len(cpus) or any(cpu < 0 for cpu in cpus):
        raise ValueError('CPU list contains duplicates/negative CPUs')
    return cpus


def hardware(requested, cpu_list=None, condition_count=None):
    if requested < 1:
        raise ValueError('Timing concurrency must be positive')
    rows, quota = topology(), quota_info()
    manual = parse_cpus(cpu_list)
    available = {row['cpu']: row for row in rows}
    unique, seen = [], set()
    for row in sorted(rows, key=lambda r: (r.get('numa') or 0, r['package'], r['core'], r['cpu'])):
        key = row['package'], row['core']
        if key not in seen:
            seen.add(key)
            unique.append(row)
    if manual is not None:
        if not manual or not set(manual) <= set(available):
            raise ValueError('CPU list is empty or outside allowed affinity/cpuset')
        chosen = [available[cpu] for cpu in manual]
        if len({(r['package'], r['core']) for r in chosen}) != len(chosen):
            raise ValueError('CPU list includes SMT siblings of the same physical core')
    else:
        chosen = unique
    actual = min(requested, len(chosen))
    if quota is not None:
        actual = min(actual, math.floor(quota))
    if condition_count is not None:
        actual = min(actual, condition_count)
    if actual < 1:
        raise ValueError('No paired conditions or fewer than one CPU equivalent available')
    return dict(host=socket.gethostname(), boot_id=read_text('/proc/sys/kernel/random/boot_id'),
        requested_concurrency=requested, concurrency=actual,
        allowed_logical_cpus=sorted(available), reported_physical_cores=len(unique),
        quota_cpu_equivalents=quota, selected_cores=chosen[:actual],
        memory_limit=read_text('/sys/fs/cgroup/memory.max'),
        note='Reported OS/VM topology; pinning does not reserve host cache/memory bandwidth.')


def timing_conditions(out):
    out = Path(out)
    if not read(out / 'evaluation_report.json')['all_route_outcomes_known']:
        raise ValueError('Finish/classify scan outcomes before timing selection')
    plan = read(out / 'evaluation_plan.json')
    if set(plan['methods']) != set(METHODS):
        raise ValueError('Expected the frozen Default/GNN warm-start comparison')
    seeds = plan['seeds'][:3]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Missing/duplicate timing seeds')
    conditions = []
    for design in sorted({row['design'] for row in plan['designs']}):
        for width in sorted({r['width'] for r in plan['designs'] if r['design'] == design}):
            root = out / 'evaluation/runs' / design / f'w{width}'
            if all(read(root / f's{seed}' / method / 'metrics.json')['status'] == 'success'
                   for seed in seeds for method in METHODS):
                conditions.append(dict(design=design, width=width, seeds=seeds))
                break
    return conditions


def schedule(conditions, cores):
    if not cores:
        raise ValueError('No timing cores selected')
    units = [dict(design=c['design'], width=c['width'], seed=seed)
             for c in conditions for seed in c['seeds']]
    waves = []
    for start in range(0, len(units), len(cores)):
        index = len(waves)
        lanes = [dict(unit, lane=lane, cpu=cores[(index + lane) % len(cores)]['cpu'],
                      order=METHODS[:] if (start + lane) % 2 == 0 else METHODS[::-1])
                 for lane, unit in enumerate(units[start:start + len(cores)])]
        waves.append(dict(index=index, effective_lanes=len(lanes), lanes=lanes))
    return waves


def run_root(a):
    if not re.fullmatch(r'[A-Za-z0-9_-]+', a.run_name):
        raise ValueError('Use a simple parallel timing run name')
    return a.out / 'timing_parallel' / a.run_name


def snapshot():
    return dict(unix_time=time.time(), loadavg=list(os.getloadavg()),
        cpu_stat=read_text('/sys/fs/cgroup/cpu.stat'),
        memory_current=read_text('/sys/fs/cgroup/memory.current'),
        memory_events=read_text('/sys/fs/cgroup/memory.events'))


def prepare(a):
    conditions = timing_conditions(a.out)
    pair_count = sum(len(c['seeds']) for c in conditions)
    hw = hardware(a.timing_jobs, a.cpu_list, pair_count)
    contexts = e.load_contexts(a)
    plan = read(a.out / 'evaluation_plan.json')
    if plan.get('contexts_sha256') and sha(a.out/'contexts.json') != plan['contexts_sha256']:
        raise ValueError('Evaluation contexts changed since the frozen width scan')
    inputs, input_files, cases = {}, {}, {}
    for condition in conditions:
        design, width = condition['design'], condition['width']
        context = next(c for c in contexts if c['design'] == design and c['width'] == width)
        path = a.out / 'evaluation/generation' / design / f'w{width}' / 'generation.json'
        generation = read(path)
        if generation['signature']['checkpoint_sha256'] != plan['checkpoint_sha256']:
            raise ValueError(f'Initializer checkpoint differs from frozen evaluation: {path}')
        if sha(generation['ap_init']) != generation['ap_init_sha256']:
            raise ValueError(f'Frozen initializer changed: {generation["ap_init"]}')
        if context.get('mapped_blif_sha256') and sha(context['mapped_blif']) != context['mapped_blif_sha256']:
            raise ValueError(f'Frozen mapped input changed: {design}')
        cases[design, width] = context, generation
        files = dict(mapped_blif=context['mapped_blif'], generation=str(path),
                     ap_init=generation['ap_init'], audit=context['audit'],
                     structure_trial=str(a.out / 'inputs' / design / 'structure_run/trial.json'))
        inputs[f'{design}/w{width}'] = {key: sha(value) for key, value in files.items()}
        input_files[f'{design}/w{width}'] = files
    protocol = dict(schema='tailcompile-v68-parallel-timing-1', identity=a.identity,
        conditions=conditions, hardware=hw, vpr_threads=1, timeout=a.timeout,
        checkpoint_sha256=plan['checkpoint_sha256'], input_sha256=inputs, input_files=input_files,
        contexts_file=str(a.out/'contexts.json'), contexts_sha256=sha(a.out/'contexts.json'),
        evaluation_plan_sha256=sha(a.out / 'evaluation_plan.json'),
        code_sha256={name: sha(HERE / name) for name in
                     ('parallel_timing.py', 'affinity_exec.py', 'experiment.py', 'common.py')},
        python=str(Path(sys.executable).resolve()),
        thread_env={key: clean_env()[key] for key in
                    ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS')},
        selection='Lowest frozen grid width successful for both methods/all first three timing seeds',
        policy='Same-core paired AB/BA; synchronized half-waves; partial wave reruns as a unit',
        scope='Runtime under recorded concurrent load; never pooled with serial timing')
    root = run_root(a)
    freeze(root / 'protocol.json', protocol)
    waves = schedule(conditions, hw['selected_cores'])
    freeze(root / 'schedule.json', dict(waves=waves))
    emit('parallel_timing_plan', directory=str(root), requested_concurrency=a.timing_jobs,
         actual_concurrency=hw['concurrency'], paired_conditions=pair_count,
         expected_trials=pair_count * 2, waves=len(waves), cpus=[r['cpu'] for r in hw['selected_cores']],
         skipped_designs=sorted({r['design'] for r in plan['designs']} - {c['design'] for c in conditions}),
         serial_results_included=False)
    return root, protocol, waves, cases


def wave_folder(root, wave):
    return root / 'waves' / f'w{wave["index"]:04d}'


def completed_wave(root, wave):
    path = wave_folder(root, wave) / 'wave.json'
    if not path.exists():
        return None
    result = read(path)
    if not result['complete']:
        return None
    if result['wave'] != wave:
        raise ValueError(f'Timing schedule evidence changed: {path}')
    expected = {f'{half}:{lane["lane"]}' for half in (0, 1) for lane in wave['lanes']}
    if set(result['results']) != expected:
        raise ValueError(f'Completed timing wave has missing/extra trials: {path}')
    for relative, digest in result['evidence_sha256'].items():
        item = path.parent / relative
        if not item.is_file() or sha(item) != digest:
            raise ValueError(f'Completed timing evidence changed: {item}')
    for half in (0, 1):
        for lane in wave['lanes']:
            directory = path.parent / f'half{half}' / f'lane{lane["lane"]:02d}'
            if read(directory/'metrics.json') != result['results'][f'{half}:{lane["lane"]}']:
                raise ValueError(f'Embedded wave results differ from original metrics: {directory}')
    return result


def assert_workers_stopped(folder, confirm_remote_stopped=False):
    """An incomplete wave can contain a still-running orphan after disconnection."""
    for path in Path(folder).rglob('started.json'):
        if (path.parent / 'trial.json').exists():
            continue
        record = read(path)
        if record['host'] != socket.gethostname():
            if not confirm_remote_stopped:
                raise RuntimeError(f'Inspect old worker on {record["host"]} before resuming: {path}')
            continue
        # Parent PID may still be active before child_pid was committed.
        for key in ('parent_pid', 'child_pid'):
            pid = record.get(key)
            if not pid or pid == os.getpid():
                continue
            stat = read_text(f'/proc/{pid}/stat')
            if stat and stat[stat.rfind(')')+2:].split()[0] == 'Z':
                continue  # Defunct children cannot still write this wave.
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            raise RuntimeError(f'Worker/parent PID {pid} may still be alive: {path}')


def archive_wave(root, wave, confirm_remote_stopped=False):
    folder = wave_folder(root, wave)
    if not folder.exists():
        return
    if folder.is_symlink():
        raise ValueError('Refuse symlink timing wave')
    folder.resolve().relative_to(root.resolve())
    assert_workers_stopped(folder, confirm_remote_stopped)
    target = root / '_attempts' / str(time.time_ns()) / folder.name
    target.parent.mkdir(parents=True)
    folder.rename(target)
    emit('parallel_wave_archived', destination=str(target),
         reason='Partial/retried wave must not mix old and new background loads')


def run_trial(a, context, lane, directory, generation, barrier, protocol, method):
    directory.mkdir(parents=True, exist_ok=True)
    init = generation['ap_init'] if method == 'gnn_warmstart' else None
    command = [sys.executable, str(HERE / 'affinity_exec.py'), '--cpu', str(lane['cpu']),
               '--audit', str(directory / 'affinity.json'), '--'] + [str(x) for x in
        e.vpr_command(a, context['mapped_blif'], lane['width'], lane['seed'], context['device_width'])]
    env = dict(TC_GP_INIT=init) if init else {}
    signature = dict(a.identity, stage='parallel_timing', design=lane['design'],
        width=lane['width'], seed=lane['seed'], method=method, cpu=lane['cpu'],
        protocol_sha256=fingerprint(protocol))
    try:
        check_free_space(a.out)
        barrier.wait(timeout=60)
    except BaseException:
        barrier.abort()
        raise
    result = execute(command, directory, signature, env=env, timeout=a.timeout,
                     heartbeat=a.heartbeat)
    text = (directory / 'vpr.out').read_text(encoding='utf-8', errors='replace')
    affinity_path = directory / 'affinity.json'
    affinity = read(affinity_path) if affinity_path.exists() else None
    if affinity is None or affinity['allowed_cpus'] != [lane['cpu']] or set(
            affinity['thread_env'].values()) != {'1'}:
        raise RuntimeError(f'CPU/thread isolation was not established: {directory}')
    if init and result['status'] in {'success', 'unroutable'} and (
        'TAILCOMPILE_V68_GP_INIT_CONSUMED' not in text or
        'TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1' not in text or
        'skipping the Global Placement' in text):
        raise RuntimeError(f'Initializer was not consumed by native GP: {directory}')
    def metric(pattern):
        values = re.findall(pattern, text)
        return float(values[-1]) if values else None
    row = dict(result, design=lane['design'], width=lane['width'], seed=lane['seed'],
        method=method, cpu=lane['cpu'], cpu_affinity=affinity,
        cpd=metric(r'Final critical path delay \(least slack\):\s*([\d.eE+-]+)'),
        wirelength=metric(r'Total wirelength:\s*([\d.eE+-]+)'),
        phases=e.parse_vpr_log(text))
    write(directory / 'metrics.json', row)
    if log_storage_failure(text, result['returncode']):
        raise StoragePressureError(f'VPR reported disk quota/full filesystem: {directory}')
    return row


def retry_wave(a, record):
    statuses = {r['status'] for r in record['results'].values()}
    return (a.retry_timeouts and 'timeout' in statuses) or (a.retry_failed and bool(
        statuses - {'success', 'unroutable', 'timeout'}))


def run(a):
    root, protocol, waves, cases = prepare(a)
    for wave in waves:
        old = completed_wave(root, wave)
        if old is not None and not retry_wave(a, old):
            emit('parallel_wave_reused', wave=wave['index'], completed_trials=len(old['results']))
            continue
        archive_wave(root, wave, a.confirm_remote_stopped)
        folder = wave_folder(root, wave)
        begin, results, errors, evidence = time.monotonic(), {}, [], {}
        before = snapshot()
        for half in (0, 1):
            if errors:
                break
            barrier = threading.Barrier(wave['effective_lanes'])
            emit('parallel_half_start', wave=wave['index'], half=half,
                 active_lanes=wave['effective_lanes'], remaining_waves=len(waves)-wave['index'])
            def one(lane):
                directory = folder / f'half{half}' / f'lane{lane["lane"]:02d}'
                context, generation = cases[lane['design'], lane['width']]
                try:
                    return lane, run_trial(a, context, lane, directory, generation, barrier,
                                           protocol, lane['order'][half])
                except BaseException:
                    barrier.abort()
                    raise
            with cf.ThreadPoolExecutor(max_workers=wave['effective_lanes']) as executor:
                futures = [executor.submit(one, lane) for lane in wave['lanes']]
                for future in cf.as_completed(futures):
                    try:
                        lane, row = future.result()
                        key = f'{half}:{lane["lane"]}'
                        results[key] = row
                        directory = folder / f'half{half}' / f'lane{lane["lane"]:02d}'
                        for name in ('metrics.json', 'trial.json', 'affinity.json'):
                            path = directory / name
                            evidence[str(path.relative_to(folder))] = sha(path)
                    except Exception as error:
                        errors.append(str(error))
                        emit('parallel_trial_error', wave=wave['index'], half=half,
                             error=str(error), storage_pressure=storage_error(error))
            # No cleanup I/O while another lane in this half is being timed.
            # Reclaim after all children have drained, before the next half.
            for lane in wave['lanes']:
                directory = folder / f'half{half}' / f'lane{lane["lane"]:02d}'
                if (directory/'metrics.json').exists():
                    try:
                        after_metrics(directory, repo=HERE.parents[2])
                    except Exception as error:
                        errors.append(str(error))
                        emit('parallel_cleanup_error', directory=str(directory), error=str(error))
            emit('parallel_half_complete', wave=wave['index'], half=half,
                 trials_completed=len(results), errors=len(errors))
        complete = len(results) == 2 * wave['effective_lanes'] and not errors
        write(folder / 'wave.json', dict(complete=complete, wave=wave, results=results,
            errors=errors, evidence_sha256=evidence, wall_seconds=time.monotonic()-begin,
            load_before=before, load_after=snapshot()))
        emit('parallel_wave_complete', wave=wave['index'], complete=complete,
             waves_complete=wave['index']+int(complete), waves_expected=len(waves))
        if not complete:
            report(a)
            raise RuntimeError(f'Incomplete timing wave retained; diagnose then resume: {folder}')
    return report(a)


def geomean(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def validate_inputs(protocol):
    """Reporting may not silently reprice a completed run with changed inputs."""
    if 'contexts_file' in protocol and sha(protocol['contexts_file']) != protocol['contexts_sha256']:
        raise ValueError('Frozen timing contexts changed; do not recompute the original report')
    for condition, files in protocol.get('input_files', {}).items():
        for key, path in files.items():
            if sha(path) != protocol['input_sha256'][condition][key]:
                raise ValueError(f'Frozen timing input/cost evidence changed: {path}')


def report(a):
    root = run_root(a)
    protocol = read(root / 'protocol.json')
    validate_inputs(protocol)
    waves = read(root / 'schedule.json')['waves']
    rows, loads, statuses = [], [], Counter()
    complete_waves, measured_seconds = 0, 0.
    for wave in waves:
        record = completed_wave(root, wave)
        if record is None:
            continue
        complete_waves += 1
        measured_seconds += record['wall_seconds']
        for half in (0, 1):
            group = [record['results'][f'{half}:{lane["lane"]}'] for lane in wave['lanes']]
            starts = [r['started_unix'] for r in group]
            loads.append(dict(wave=wave['index'], half=half, effective_lanes=wave['effective_lanes'],
                launch_skew_seconds=max(starts)-min(starts),
                shortest_seconds=min(r['wall_seconds'] for r in group),
                longest_seconds=max(r['wall_seconds'] for r in group)))
        statuses.update(r['status'] for r in record['results'].values())
        for lane in wave['lanes']:
            pair = {record['results'][f'{half}:{lane["lane"]}']['method']:
                    record['results'][f'{half}:{lane["lane"]}'] for half in (0, 1)}
            row = dict(design=lane['design'], width=lane['width'], seed=lane['seed'],
                       cpu=lane['cpu'], order=lane['order'], wave=wave['index'],
                       effective_lanes=wave['effective_lanes'],
                       statuses={method: r['status'] for method, r in pair.items()},
                       comparable=all(r['status'] == 'success' for r in pair.values()))
            if row['comparable']:
                baseline, gnn = pair['default'], pair['gnn_warmstart']
                cpd, wire = baseline.get('cpd'), baseline.get('wirelength')
                generation = read(a.out / 'evaluation/generation' / lane['design'] /
                                  f'w{lane["width"]}' / 'generation.json')
                context = next(c for c in read(a.out/'contexts.json')['contexts']
                               if c['design']==lane['design'] and c['width']==lane['width'])
                structure = read(a.out/'inputs'/lane['design']/'structure_run/trial.json')['wall_seconds']
                logic_cost = sum(v for key, v in read(context['audit']).get('timing', {}).items()
                                 if key not in ('device_preprocessing', 'device_provenance'))
                scopes = dict(vpr_only=gnn['wall_seconds'],
                    cached_graph_warm=gnn['wall_seconds']+generation['warm_inference_lowering_seconds'],
                    new_design_prepared_device=gnn['wall_seconds']+
                        generation['cold_from_prepared_graph_seconds']+structure+logic_cost)
                row.update(default_seconds=baseline['wall_seconds'], method_seconds=scopes,
                    speedup={key: baseline['wall_seconds']/value for key, value in scopes.items()},
                    cpd_ratio=gnn.get('cpd')/cpd if cpd and gnn.get('cpd') else None,
                    wirelength_ratio=gnn.get('wirelength')/wire if wire and gnn.get('wirelength') else None,
                    initializer_components=generation['timing'],
                    structure_extraction_seconds=structure, logic_feature_seconds=logic_cost)
            rows.append(row)
    groups = defaultdict(list)
    for row in rows:
        if row['comparable']:
            groups[row['design'], row['effective_lanes']].append(row)
    comparisons = [dict(design=design, effective_lanes=load, comparable_pairs=len(group),
        geomean_speedup={scope: geomean([r['speedup'][scope] for r in group])
                        for scope in ('vpr_only','cached_graph_warm','new_design_prepared_device')},
        geomean_cpd_ratio=geomean([r['cpd_ratio'] for r in group if r['cpd_ratio']]),
        geomean_wirelength_ratio=geomean([r['wirelength_ratio'] for r in group if r['wirelength_ratio']]))
        for (design, load), group in sorted(groups.items())]
    result = dict(schema='tailcompile-v68-parallel-timing-report-1',
        complete=complete_waves == len(waves), protocol=protocol,
        waves_complete=complete_waves, waves_expected=len(waves),
        status_counts=dict(statuses), rows=rows, comparisons=comparisons, half_wave_load=loads,
        all_route_outcomes_known=sum(statuses.values())>0 and not bool(
            set(statuses)-{'success','unroutable'}) and complete_waves==len(waves),
        serial_results_included=False, reusable_rrg_preprocessing_excluded=True,
        throughput=dict(completed_wave_seconds=measured_seconds, completed_trials=sum(statuses.values())),
        scope_note='Same v6.8 component scopes. New-design cost is an estimate using preparation-time '
                   'structure/logic costs, not a remeasured end-to-end cold process. VPR wall time '
                   'includes the identical Python affinity bootstrap in both arms. Reusable RRG '
                   'preprocessing excluded. RTL synthesis common/excluded. Cache-warm is not first-use.',
        caveat='Controlled concurrent wall-time, not isolated latency. Same-core pairing/ABBA do '
               'not eliminate cache, bandwidth, frequency or wave-tail interference. Load strata '
               'are separate; no serial measurements are included.')
    write(root / 'parallel_report.json', result)
    emit('parallel_timing_report', path=str(root/'parallel_report.json'), complete=result['complete'],
         waves_complete=complete_waves, waves_expected=len(waves), completed_trials=sum(statuses.values()),
         comparable_pairs=sum(r['comparable'] for r in rows))
    return result


def status(a):
    root = run_root(a)
    if not (root / 'schedule.json').exists():
        emit('parallel_status', run_name=a.run_name, initialized=False, complete=False)
        return
    waves = read(root / 'schedule.json')['waves']
    completed = sum(completed_wave(root, wave) is not None for wave in waves)
    counts = Counter()
    for wave in waves:
        folder = wave_folder(root, wave)
        for half in (0, 1):
            for lane in wave['lanes']:
                directory = folder / f'half{half}' / f'lane{lane["lane"]:02d}'
                if (directory/'metrics.json').exists():
                    counts[read(directory/'metrics.json')['status']] += 1
                else:
                    counts['started_or_interrupted' if (directory/'started.json').exists() else 'missing'] += 1
    report_path = root/'parallel_report.json'
    emit('parallel_status', run_name=a.run_name, initialized=True,
         waves_complete=completed, waves_expected=len(waves), trial_status_counts=dict(counts),
         report_exists=report_path.exists(),
         complete=completed==len(waves) and report_path.exists() and read(report_path)['complete'])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase', choices=['run','status','report'])
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--run-name', default='paired_001')
    p.add_argument('--vtr-root', dest='vtr', type=Path, default=os.environ.get('VTR_ROOT'))
    p.add_argument('--timing-jobs', type=int, default=16)
    p.add_argument('--cpu-list')
    p.add_argument('--timeout', type=float, default=7200)
    p.add_argument('--heartbeat', type=float, default=60)
    p.add_argument('--retry-failed', action='store_true')
    p.add_argument('--retry-timeouts', action='store_true')
    p.add_argument('--confirm-remote-stopped', action='store_true')
    a = p.parse_args(argv)
    a.out = a.out.resolve()
    if a.timing_jobs < 1 or a.timeout <= 0 or a.heartbeat <= 0:
        p.error('Positive concurrency/timeout/heartbeat required')
    run_root(a)
    return a


def main():
    a = parse_args()
    if a.phase == 'status':
        return status(a)
    if a.phase == 'report':
        return report(a)
    e.identity(a)
    record_policy(a.out)
    run(a)


if __name__ == '__main__':
    main()
