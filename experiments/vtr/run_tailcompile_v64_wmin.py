#!/usr/bin/env python3
"""Paired, resumable channel-width sweep of frozen v6.3 2D placements.

This is a *grid* minimum, not a mathematical proof of the minimum legal width.
Every width is run: VPR's heuristic success is not assumed monotone in width.
The same packed net, device size, placement seed, and frozen fplace are used
across all widths within a case. No model is retrained or reselected.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCHEMA = 'tailcompile-v64-fixed-placement-wmin-1'
DEFAULT_WIDTHS = {
    ('conv_layer', 140): list(range(80, 161, 10)),
    ('lenet', 160): list(range(140, 221, 10)),
}
DEFAULT_SEEDS = list(range(81, 91))
QOR = ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def width_map(specs):
    result = dict(DEFAULT_WIDTHS)
    for spec in specs:
        try:
            design, anchor, widths = spec.split(':', 2)
            key = design, int(anchor)
            values = [int(value) for value in widths.split(',')]
        except ValueError as exc:
            raise ValueError(f'invalid width specification {spec!r}; use DESIGN:ANCHOR:W1,W2,...') from exc
        if key not in DEFAULT_WIDTHS:
            raise ValueError(f'width override is not a frozen active v6.3 condition: {key}')
        if len(values) < 3 or values != sorted(set(values)) or any(value <= 0 for value in values):
            raise ValueError('widths must be >=3 sorted distinct positive integers')
        result[key] = values
    return result


def make_plan(selection_path, seeds, grids):
    selection_path = Path(selection_path).resolve()
    report = read(selection_path)
    if not set(seeds).isdisjoint(set(range(51, 74))) or len(seeds) < 3 or len(seeds) != len(set(seeds)):
        raise ValueError('use >=3 distinct fresh seeds outside v6.3 label/evaluation seeds 51-73')
    cases = []
    for row in report['selections']:
        key = row['design'], int(row['width'])
        if row['selected'] == 'noop':
            continue
        if key not in grids:
            raise ValueError(f'active selection lacks a preregistered width grid: {key}')
        inputs = Path(row['inputs']).resolve()
        blif, net = inputs / f'{key[0]}.pre-vpr.blif', inputs / f'{key[0]}.net'
        baseline, candidate = Path(row['baseline']).resolve(), Path(row['selected_fplace']).resolve()
        for path in (blif, net, baseline, candidate):
            if not path.is_file():
                raise FileNotFoundError(f'missing immutable campaign input: {path}')
        if sha(baseline) != row['baseline_sha256'] or sha(candidate) != row['selected_sha256']:
            raise ValueError(f'fplace hash differs from frozen selection: {key}')
        if sha(baseline) == sha(candidate):
            raise ValueError(f'selected action did not alter fplace: {key}')
        cases.append(dict(design=key[0], anchor_width=key[1], selected=row['selected'],
            widths=grids[key], blif=str(blif), blif_sha256=sha(blif),
            packed_net=str(net), packed_net_sha256=sha(net),
            baseline_fplace=str(baseline), baseline_sha256=sha(baseline),
            selected_fplace=str(candidate), selected_sha256=sha(candidate),
            device_width=int(row['device_width']), full_legalizer=row['full_legalizer']))
    if set((case['design'], case['anchor_width']) for case in cases) != set(DEFAULT_WIDTHS):
        raise ValueError('active v6.3 selections differ from the preregistered two-condition cohort')
    return dict(schema=SCHEMA, selection_report=str(selection_path),
        selection_report_sha256=sha(selection_path), seeds=seeds, cases=cases,
        task_count=sum(len(c['widths']) * len(seeds) * 3 for c in cases),
        protocol='fixed packed net and fplace per anchor; exhaustive predeclared width grid')


def tasks(plan, root):
    for case in plan['cases']:
        for width in case['widths']:
            for method in ('default', 'gnn_base', 'v63'):
                fplace = (None if method == 'default' else
                          case['baseline_fplace'] if method == 'gnn_base' else case['selected_fplace'])
                for seed in plan['seeds']:
                    directory = root / 'runs' / case['design'] / f'a{case["anchor_width"]}' / f'w{width}' / method / f's{seed}'
                    yield dict(case=case, width=width, method=method, seed=seed,
                               fplace=fplace, directory=directory)


def verify_inputs(plan):
    if sha(plan['selection_report']) != plan['selection_report_sha256']:
        raise ValueError('selection report changed after plan creation')
    for case in plan['cases']:
        for name, expected in (('blif', 'blif_sha256'), ('packed_net', 'packed_net_sha256'),
                               ('baseline_fplace', 'baseline_sha256'),
                               ('selected_fplace', 'selected_sha256')):
            if sha(case[name]) != case[expected]:
                raise ValueError(f'campaign input changed: {case[name]}')


def recover_interrupted(directory, root):
    """Move only a proven-inactive partial run under this campaign to history."""
    directory, root = directory.resolve(), root.resolve()
    relative = directory.relative_to(root)
    if not directory.exists() or not any(directory.iterdir()):
        return
    manifest = directory / 'manifest.json'
    if manifest.exists():
        try:
            read(manifest)
            return
        except (ValueError, json.JSONDecodeError):
            pass
    proc = Path('/proc')
    if not proc.is_dir():
        raise RuntimeError(f'incomplete run needs manual inspection: {directory}')
    needle = str(directory).encode()
    for cmdline in proc.glob('[0-9]*/cmdline'):
        try:
            if needle in cmdline.read_bytes():
                raise RuntimeError(f'incomplete run is still active: {directory}')
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    archive = root / '_interrupted_history' / relative.parent / f'{relative.name}-{time.time_ns()}'
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(archive))
    print(json.dumps(dict(stage='quarantine_interrupted', run=str(directory), archive=str(archive))), flush=True)


def run_task(task, root, backend, timeout):
    from prepare_tailcompile_v3_training import success_metrics
    from v5_vpr_retry import run_one

    case, directory = task['case'], task['directory']
    recover_interrupted(directory, root)
    manifest = run_one(backend, directory, case['blif'], case['packed_net'], task['width'],
        case['device_width'], task['seed'], task['fplace'], timeout=timeout,
        full_legalizer=case['full_legalizer'])
    manifest_hash = sha(directory / 'manifest.json')
    record_path = directory / 'trial.json'
    if record_path.is_file():
        previous = read(record_path)
        if previous.get('manifest_sha256') == manifest_hash:
            return previous
    record = dict(design=case['design'], anchor_width=case['anchor_width'],
        selected=case['selected'], width=task['width'], method=task['method'],
        seed=task['seed'], status=manifest['vpr_status'],
        manifest_sha256=manifest_hash, wall_seconds=manifest['wall_seconds'])
    if record['status'] == 'success':
        record.update(success_metrics(directory, case['design']))
        if any(record.get(key) is None for key in QOR):
            raise ValueError(f'successful VPR run lacks QoR metrics: {directory}')
    write(record_path, record)
    return record


def run_campaign(root, plan, jobs, timeout):
    if jobs < 1 or timeout < 1:
        raise ValueError('jobs and timeout must be positive')
    verify_inputs(plan)
    from run_tailcompile_v4 import Backend
    backend = Backend(os.environ.get('VTR_ROOT'))
    all_tasks = list(tasks(plan, root))
    if len(all_tasks) != plan['task_count']:
        raise AssertionError('task schedule differs from immutable plan')
    print(json.dumps(dict(stage='wmin_schedule', tasks=len(all_tasks), jobs=jobs,
                          expected_vtr=backend.root)), flush=True)
    completed = 0
    iterator = iter(all_tasks)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        pending = {}
        for _ in range(min(jobs, len(all_tasks))):
            task = next(iterator)
            pending[executor.submit(run_task, task, root, backend, timeout)] = task
        while pending:
            future = next(as_completed(pending))
            task = pending.pop(future)
            row = future.result()
            completed += 1
            print(json.dumps(dict(stage='wmin_trial', complete=completed, total=len(all_tasks),
                design=row['design'], anchor=row['anchor_width'], width=row['width'],
                method=row['method'], seed=row['seed'], status=row['status'])), flush=True)
            following = next(iterator, None)
            if following is not None:
                pending[executor.submit(run_task, following, root, backend, timeout)] = following


def load_trials(root, plan):
    rows, missing = [], []
    for task in tasks(plan, root):
        path = task['directory'] / 'trial.json'
        manifest = task['directory'] / 'manifest.json'
        if not path.is_file() or not manifest.is_file():
            missing.append(str(task['directory'].relative_to(root)))
            continue
        row = read(path)
        if row['manifest_sha256'] != sha(manifest):
            raise ValueError(f'trial was changed after metric extraction: {path}')
        expected = task['case']
        if (row['design'], row['anchor_width'], row['width'], row['method'], row['seed']) != (
                expected['design'], expected['anchor_width'], task['width'], task['method'], task['seed']):
            raise ValueError(f'trial identity mismatch: {path}')
        if row['status'] not in ('success', 'unroutable'):
            raise ValueError(f'non-routing failure cannot count as a width result: {path}')
        rows.append(row)
    return rows, missing


def width_result(widths, records):
    statuses = {row['width']: row['status'] for row in records}
    if set(statuses) != set(widths) or len(statuses) != len(records):
        raise ValueError('one result per configured width is required')
    successful = [width for width in widths if statuses[width] == 'success']
    if not successful:
        return dict(w_min_grid=None, censor='right', lower_bound_exclusive=max(widths),
                    monotonicity_violations=0)
    first = successful[0]
    return dict(w_min_grid=first, censor='left' if first == min(widths) else 'none',
        lower_bound_exclusive=None,
        monotonicity_violations=sum(statuses[width] != 'success' for width in widths if width > first))


def compare_pairs(candidate, baseline):
    pairs = []
    for seed in sorted(candidate):
        a, b = candidate[seed], baseline[seed]
        wa, wb = a['w_min_grid'], b['w_min_grid']
        delta = wa - wb if wa is not None and wb is not None else None
        if delta is not None:
            outcome = 'improved' if delta < 0 else 'regressed' if delta > 0 else 'equal_on_grid'
        elif wa is not None and wb is None:
            outcome = 'improved_at_least_to_grid_limit'
        elif wa is None and wb is not None:
            outcome = 'regressed_at_least_to_grid_limit'
        else:
            outcome = 'both_right_censored'
        pairs.append(dict(seed=seed, candidate=wa, baseline=wb, delta_grid=delta,
            candidate_censor=a['censor'], baseline_censor=b['censor'], outcome=outcome))
    deltas = [p['delta_grid'] for p in pairs if p['delta_grid'] is not None]
    return dict(pairs=pairs, exact_numeric_pairs=sum(p['candidate_censor'] == p['baseline_censor'] == 'none'
        for p in pairs), observed_numeric_pairs=len(deltas),
        median_delta_grid=statistics.median(deltas) if deltas else None,
        outcomes=dict(Counter(p['outcome'] for p in pairs)))


def summarize_rows(plan, rows):
    index = {(r['design'], r['anchor_width'], r['width'], r['method'], r['seed']): r for r in rows}
    if len(index) != len(rows) or len(rows) != plan['task_count']:
        raise ValueError('summary requires every distinct planned trial')
    conditions = []
    for case in plan['cases']:
        design, anchor, widths = case['design'], case['anchor_width'], case['widths']
        per_method = {}
        for method in ('default', 'gnn_base', 'v63'):
            by_seed = {}
            success_by_width = {}
            for seed in plan['seeds']:
                sample = [index[design, anchor, width, method, seed] for width in widths]
                by_seed[seed] = width_result(widths, sample)
            for width in widths:
                success_by_width[width] = sum(index[design, anchor, width, method, seed]['status'] == 'success'
                                               for seed in plan['seeds'])
            threshold = math.ceil(0.8 * len(plan['seeds']))
            robust = next((width for width in widths if success_by_width[width] >= threshold), None)
            per_method[method] = dict(per_seed={str(k): v for k, v in by_seed.items()},
                success_by_width={str(k): v for k, v in success_by_width.items()},
                w80_grid=robust, w80_censor='right' if robust is None else
                'left' if robust == min(widths) else 'none',
                monotonicity_violations=sum(v['monotonicity_violations'] for v in by_seed.values()))
        comparisons = {}
        for baseline in ('default', 'gnn_base'):
            comp = compare_pairs({int(k): v for k, v in per_method['v63']['per_seed'].items()},
                                 {int(k): v for k, v in per_method[baseline]['per_seed'].items()})
            candidate_robust, baseline_robust = per_method['v63']['w80_grid'], per_method[baseline]['w80_grid']
            comp['w80_delta_grid'] = (candidate_robust - baseline_robust
                if candidate_robust is not None and baseline_robust is not None else None)
            ratios = {key: [] for key in QOR}
            for width in widths:
                for seed in plan['seeds']:
                    a, b = index[design, anchor, width, 'v63', seed], index[design, anchor, width, baseline, seed]
                    if a['status'] == b['status'] == 'success':
                        for key in QOR:
                            if a[key] <= 0 or b[key] <= 0:
                                raise ValueError(f'nonpositive QoR value in successful run: {key}')
                            ratios[key].append(math.log(a[key] / b[key]))
            comp['joint_success_count'] = len(ratios[QOR[0]])
            comp['joint_success_geomean_ratio'] = {key: math.exp(statistics.mean(values)) if values else None
                for key, values in ratios.items()}
            comparisons[baseline] = comp
        conditions.append(dict(design=design, anchor_width=anchor, selected=case['selected'],
                               widths=widths, methods=per_method, comparisons=comparisons))
    return dict(schema=SCHEMA, complete=True, selection_report_sha256=plan['selection_report_sha256'],
        task_count=plan['task_count'], seeds=plan['seeds'], conditions=conditions,
        interpretation='W_min is the minimum successful tested width for a frozen placement; '
                       'left/right censoring and heuristic nonmonotonicity are explicit. '
                       'W80 is the minimum tested width with >=80% seed success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('plan', 'run', 'summarize', 'status'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', default=DEFAULT_SEEDS)
    parser.add_argument('--widths', action='append', default=[],
                        help='override before plan creation: DESIGN:ANCHOR:W1,W2,...')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=1800)
    args = parser.parse_args()
    root = args.out.resolve()
    plan_path = root / 'plan.json'
    if args.phase == 'plan':
        if args.selection is None:
            parser.error('plan requires --selection')
        plan = make_plan(args.selection, args.seeds, width_map(args.widths))
        if plan_path.is_file():
            if read(plan_path) != plan:
                raise ValueError('existing plan differs; use a new output directory')
        else:
            if root.exists() and any(root.iterdir()):
                raise ValueError('new campaign output directory must be empty')
            write(plan_path, plan)
        print(json.dumps(dict(stage='plan', cases=len(plan['cases']), seeds=len(plan['seeds']),
                              tasks=plan['task_count'], out=str(root))), flush=True)
        return
    if not plan_path.is_file():
        raise FileNotFoundError(f'run the plan phase first: {plan_path}')
    plan = read(plan_path)
    if plan['schema'] != SCHEMA:
        raise ValueError('wrong plan schema')
    if args.phase == 'run':
        run_campaign(root, plan, args.jobs, args.timeout)
    rows, missing = load_trials(root, plan)
    status = dict(stage='status', expected=plan['task_count'], completed=len(rows),
        missing=len(missing), statuses=dict(Counter(row['status'] for row in rows)))
    print(json.dumps(status), flush=True)
    if args.phase in ('run', 'summarize'):
        if missing:
            raise RuntimeError(f'campaign incomplete; first missing: {missing[0]}')
        report = summarize_rows(plan, rows)
        write(root / 'wmin_report.json', report)
        print(json.dumps(dict(stage='summary', out=str(root / 'wmin_report.json'),
                              conditions=len(report['conditions']))), flush=True)


if __name__ == '__main__':
    main()
