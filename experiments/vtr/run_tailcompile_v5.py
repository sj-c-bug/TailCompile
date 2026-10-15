#!/usr/bin/env python3
"""Run the frozen, heterogeneity-stratified TailCompile 2D confirmation."""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_tailcompile_v4 import Backend, summarize
from v5_vpr_retry import run_one
from tailcompile_v4 import build, load
from tailcompile_v4_risk import VTR_COMMIT, build_supply, digest, save


HERE = Path(__file__).resolve().parent
ALLOWED_METHODS = ('default', 'det_base', 'det_mean', 'det_tail',
                   'gnn_base', 'gnn_mean', 'gnn_tail')


def copy_locked(source, target):
    source, target = Path(source), Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and digest(target) != digest(source):
        raise ValueError(f'archived input changed: {target}')
    if not target.exists():
        shutil.copy2(source, target)


def gini(values):
    ordered = sorted(max(0.0, float(x)) for x in values)
    total, n = sum(ordered), len(ordered)
    if not n or total == 0:
        return 0.0
    return sum((2 * i - n - 1) * value for i, value in enumerate(ordered, 1)) / (n * total)


def heterogeneity_from_search(path):
    initial = load(path)['initial']
    ratios = [float(row['rho']) for row in initial['pools']]
    mean = float(initial['mean'])
    upper = max(1, math.ceil(0.10 * len(ratios)))
    return {'cvar': float(initial['cvar']), 'mean': mean,
            'cvar_mean_ratio': float(initial['cvar']) / max(mean, 1e-12),
            'max_mean_ratio': float(initial['max']) / max(mean, 1e-12),
            'gini': gini(ratios),
            'top10_demand_share': (sum(sorted(ratios, reverse=True)[:upper]) /
                                   max(sum(ratios), 1e-12)),
            'pool_count': len(ratios)}


def archive_inputs(root, record):
    design = record['design']
    target = root / design / 'inputs'
    paths = {'logic_ir': Path(record['logic_ir']), 'device_ir': Path(record['device_ir']),
             'mapped_blif': Path(record['mapped_blif']), 'packed_net': Path(record['packed_net'])}
    names = {'logic_ir': 'logic_ir.json', 'device_ir': 'device_ir.json',
             'mapped_blif': f'{design}.pre-vpr.blif', 'packed_net': f'{design}.net'}
    for key, path in paths.items():
        copy_locked(path, target / names[key])
    logic = load(target / 'logic_ir.json')
    if digest(target / f'{design}.net') != logic['packed_net_sha256']:
        raise ValueError(f'{design} packed net differs from Logic IR')
    if digest(target / f'{design}.pre-vpr.blif') != logic['mapped_blif_sha256']:
        raise ValueError(f'{design} BLIF differs from Logic IR')
    return target


def prepare(root, config, backend, timeout):
    checkpoint = Path(config['checkpoint'])
    if digest(checkpoint) != config['checkpoint_sha256']:
        raise ValueError('frozen checkpoint changed')
    profiles = []
    for record in config['designs']:
        design = record['design']
        inputs = archive_inputs(root, record)
        logic_path, device_path = inputs / 'logic_ir.json', inputs / 'device_ir.json'
        blif, packed = inputs / f'{design}.pre-vpr.blif', inputs / f'{design}.net'
        for width in record['widths']:
            work = root / design / f'w{width}'
            supply_path = work / 'supply.json'
            # The supply run only materializes the actual-width RRG.  Reuse a
            # screening seed that is explicitly excluded from confirmation so
            # no confirmatory VPR outcome exists before the strata are locked.
            provenance = run_one(backend, work / 'supply_run', blif, packed, width,
                                 int(record['device_width']), config['screen_seeds_excluded'][0],
                                 write_rr=True, timeout=timeout,
                                 full_legalizer=config['ap_full_legalizer'])
            if not supply_path.exists():
                supply = build_supply(work / 'supply_run/device.rr_graph.xml', load(device_path))
                if supply['channel_width'] != width:
                    raise ValueError('actual RRG width differs from requested width')
                supply['provenance'] = provenance['signature']
                save(supply_path, supply)
            gnn_dir = work / 'gnn_initial'
            if not (gnn_dir / 'build_summary.json').exists():
                clock = time.monotonic()
                subprocess.run([sys.executable, str(HERE / 'tailcompile_v3.py'),
                                '--logic-ir', str(logic_path), '--device-ir', str(device_path),
                                '--blif', str(blif), '--checkpoint', str(checkpoint),
                                '--channel-width', str(width), '--candidates', '1', '--seed', '2027',
                                '--out', str(gnn_dir)], check=True)
                save(gnn_dir / 'v4_initialization.json',
                     {'seconds': time.monotonic() - clock,
                      'checkpoint_sha256': config['checkpoint_sha256']})
            initial_summary = load(gnn_dir / 'build_summary.json')
            if initial_summary['checkpoint_sha256'] != config['checkpoint_sha256']:
                raise ValueError('GNN initializer checkpoint changed')
            gnn_lower = gnn_dir / initial_summary['candidates'][0]['lowering']
            from tailcompile_v2 import solve_binding
            from tailcompile_v1 import lower_fplace
            det_dir = work / 'det_initial'
            det_dir.mkdir(parents=True, exist_ok=True)
            det_lower = det_dir / 'tail_rrg_lowering.json'
            if not det_lower.exists():
                clock = time.monotonic()
                binding = solve_binding(load(logic_path), load(device_path), 'tail_rrg')
                lowering = lower_fplace(load(logic_path), load(device_path), binding,
                                        det_dir / 'tail_rrg.fplace')
                save(det_lower, lowering)
                save(det_dir / 'tail_rrg_binding.json', binding)
                save(det_dir / 'v4_initialization.json', {'seconds': time.monotonic() - clock})
            candidates = work / 'candidates'
            built = build(logic_path, device_path, blif, supply_path, det_lower, gnn_lower,
                          candidates, config['budget'], config['search_seed'],
                          matched_risk_scales=True, methods=('base', 'mean', 'tail'))
            metric = heterogeneity_from_search(candidates / 'det_base_search.json')
            profiles.append({'design': design, 'width': width, **metric,
                             'supply_sha256': digest(supply_path),
                             'build_summary_sha256': digest(candidates / 'build_summary.json'),
                             'candidate_count': len(built['candidates'])})
    by_design = {}
    for record in config['designs']:
        values = [row['cvar_mean_ratio'] for row in profiles if row['design'] == record['design']]
        by_design[record['design']] = statistics.median(values)
    ordered = sorted(by_design, key=lambda name: (by_design[name], name))
    strata = {}
    labels = ('low', 'medium', 'high')
    for index, design in enumerate(ordered):
        strata[design] = labels[min(2, 3 * index // len(ordered))]
    lock = {'schema': 'tailcompile-v5-heterogeneity-lock-v1',
            'created_before_confirmatory_vpr': True,
            'vpr_labels_used_for_stratification': False,
            'primary_heterogeneity': 'median_width_cvar_mean_ratio_on_det_base',
            'config_sha256': digest(root / 'config.json'), 'profiles': profiles,
            'design_scores': by_design, 'strata': strata,
            'stratum_rule': 'rank tertiles across the predeclared candidate cohort'}
    save(root / 'heterogeneity_lock.json', lock)
    return lock


def verify_lock(root):
    lock = load(root / 'heterogeneity_lock.json')
    if lock['config_sha256'] != digest(root / 'config.json'):
        raise ValueError('config changed after heterogeneity lock')
    for row in lock['profiles']:
        work = root / row['design'] / f'w{row["width"]}'
        if digest(work / 'supply.json') != row['supply_sha256']:
            raise ValueError('supply changed after heterogeneity lock')
        if digest(work / 'candidates/build_summary.json') != row['build_summary_sha256']:
            raise ValueError('candidate build changed after heterogeneity lock')
    return lock


def run_confirmation(root, config, backend, timeout, jobs=1):
    verify_lock(root)
    plan = []
    for record in config['designs']:
        design = record['design']
        inputs = root / design / 'inputs'
        blif, packed = inputs / f'{design}.pre-vpr.blif', inputs / f'{design}.net'
        for width in record['widths']:
            work = root / design / f'w{width}'
            built = load(work / 'candidates/build_summary.json')
            candidate_index = {row['method']: row for row in built['candidates']}
            for method in config['methods']:
                row = None if method == 'default' else candidate_index[method]
                fplace = None if row is None else work / 'candidates' / row['fplace']
                for seed in config['confirm_seeds']:
                    directory = work / 'runs' / f'{method}_s{seed}'
                    spec = {'design': design, 'width': width, 'method': method, 'seed': seed,
                            'ap_full_legalizer': config['ap_full_legalizer'],
                            'directory': directory.relative_to(root).as_posix(),
                            'inputs': inputs.relative_to(root).as_posix(),
                            'candidate': fplace.relative_to(root).as_posix() if fplace else None}
                    if row:
                        initial = method.split('_')[0]
                        init_seconds = load(work / f'{initial}_initial/v4_initialization.json')['seconds']
                        spec['preplacement_seconds'] = (init_seconds + row['search_seconds'] +
                                                       row['lowering_seconds'])
                        spec['search_evaluations'] = row['evaluations']
                    plan.append(spec)
    save(root / 'plan.json', {'schema': 'tailcompile-v5-confirmatory-plan-v1',
                              'config': config, 'parallel_jobs': jobs, 'runs': plan})
    records = {row['design']: row for row in config['designs']}

    def execute(spec):
        record = records[spec['design']]
        inputs = root / spec['inputs']
        return run_one(backend, root / spec['directory'],
                       inputs / f'{spec["design"]}.pre-vpr.blif',
                       inputs / f'{spec["design"]}.net',
                       spec['width'], int(record['device_width']), spec['seed'],
                       root / spec['candidate'] if spec['candidate'] else None,
                       timeout=timeout,
                       full_legalizer=config['ap_full_legalizer'])

    with ThreadPoolExecutor(max_workers=jobs) as executor:
        # Consume the iterator so worker exceptions stop the campaign.
        for _ in executor.map(execute, plan):
            pass
    summary = summarize(root)
    summary['interpretation'] = ('preregistered heterogeneity-stratified 2D confirmation; '
                                 'screening seeds excluded; strata locked before method VPR runs')
    save(root / 'summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vtr-root', default=os.environ.get('VTR_ROOT'))
    p.add_argument('--stage', choices=('prepare', 'run', 'summarize', 'all'), default='all')
    p.add_argument('--jobs', type=int,
                   default=int(os.environ.get('TAILCOMPILE_VPR_JOBS', '1')))
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    if a.jobs < 1:
        p.error('--jobs must be at least 1')
    config = load(a.config)
    if config['vtr_commit'] != VTR_COMMIT:
        raise ValueError('confirmatory config VTR commit mismatch')
    if tuple(config['methods']) != ALLOWED_METHODS:
        raise ValueError(f'method list must remain frozen as {ALLOWED_METHODS}')
    if config.get('ap_full_legalizer') != 'appack':
        raise ValueError('v5 confirmation requires the frozen appack full legalizer')
    a.out.mkdir(parents=True, exist_ok=True)
    archived = a.out / 'config.json'
    if archived.exists() and load(archived) != config:
        raise ValueError('confirmatory config changed; choose a new --out')
    save(archived, config)
    if a.stage == 'summarize':
        summarize(a.out)
        return
    backend = Backend(a.vtr_root)
    if a.stage in ('prepare', 'all'):
        prepare(a.out, config, backend, a.timeout)
    if a.stage in ('run', 'all'):
        result = run_confirmation(a.out, config, backend, a.timeout, a.jobs)
        if (result['missing'] or result['metric_errors'] or
                result['status_counts'].get('other_failure')):
            raise RuntimeError('confirmatory campaign failed completeness checks')
        print(json.dumps({'stage': 'complete', 'expected': result['expected'],
                          'observed': result['observed'],
                          'status_counts': result['status_counts']}), flush=True)


if __name__ == '__main__':
    main()
