#!/usr/bin/env python3
"""Find a Default-only route boundary before building any v5 Tail candidates."""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from run_tailcompile_v4 import Backend, run_one
from tailcompile_v4 import load
from tailcompile_v4_risk import VTR_COMMIT, digest, save


HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vtr-root', default=os.environ.get('VTR_ROOT'))
    p.add_argument('--checkpoint', type=Path,
                   default=HERE / 'tailcompile_v3_training/model_width_conditioned/ranker.pt')
    p.add_argument('--screen-seeds', nargs='+', type=int, default=[11, 12])
    p.add_argument('--confirm-seeds', nargs='+', type=int,
                   default=[21, 22, 23, 24, 25, 26, 27, 28])
    p.add_argument('--jobs', type=int,
                   default=int(os.environ.get('TAILCOMPILE_VPR_JOBS', '1')))
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    if a.jobs < 1:
        p.error('--jobs must be at least 1')
    if set(a.screen_seeds) & set(a.confirm_seeds):
        p.error('screen and confirm seeds must be disjoint')
    source = load(a.inputs)
    if source['vtr_commit'] != VTR_COMMIT:
        raise ValueError('input cohort VTR commit mismatch')
    backend = Backend(a.vtr_root)
    a.out.mkdir(parents=True, exist_ok=True)
    selected, screen_rows, censored = [], [], []
    for record in source['designs']:
        design = record['design']
        blif, packed = Path(record['mapped_blif']), Path(record['packed_net'])
        widths = sorted({int(x) for x in record['screen_widths']})
        tasks = [(width, seed,
                  a.out / 'runs' / design / f'w{width}' / f'default_s{seed}')
                 for width in widths for seed in a.screen_seeds]

        def execute(task):
            width, seed, directory = task
            result = run_one(backend, directory, blif, packed, width,
                             int(record['device_width']), seed, timeout=a.timeout)
            return width, seed, directory, result['vpr_status']

        # executor.map preserves task order, so the frozen summary is
        # deterministic even though independent VPR processes run in parallel.
        with ThreadPoolExecutor(max_workers=a.jobs) as executor:
            outcomes = list(executor.map(execute, tasks))
        by_width = {width: [] for width in widths}
        for width, seed, directory, status in outcomes:
            by_width[width].append(status)
            screen_rows.append({'design': design, 'width': width, 'seed': seed,
                                'status': status,
                                'directory': directory.relative_to(a.out).as_posix()})
        full_success = [w for w in widths if all(s == 'success' for s in by_width[w])]
        if not full_success:
            censored.append({'design': design, 'reason': 'no_full_success',
                             'tested_widths': widths})
            continue
        high = min(full_success)
        lower = [w for w in widths if w < high and not all(s == 'success' for s in by_width[w])]
        if not lower:
            censored.append({'design': design, 'reason': 'no_lower_failure',
                             'first_full_success': high, 'tested_widths': widths})
            continue
        low = max(lower)
        selected.append({**record, 'widths': [low, high],
                         'boundary_rule': 'largest_lower_non-full-success_and_smallest_full-success',
                         'screen_status': {str(w): by_width[w] for w in (low, high)}})
    screen = {'schema': 'tailcompile-v5-default-screen-v1', 'vtr_commit': VTR_COMMIT,
              'input_config_sha256': digest(a.inputs), 'screen_seeds': a.screen_seeds,
              'parallel_jobs': a.jobs,
              'rows': screen_rows, 'selected_design_count': len(selected), 'censored': censored}
    save(a.out / 'screen_summary.json', screen)
    if len(selected) < 3:
        raise RuntimeError('fewer than three uncensored holdout designs; expand screen widths')
    confirm = {
        'schema': 'tailcompile-v5-confirmatory-config-v1',
        'vtr_commit': VTR_COMMIT,
        'input_config': str(a.inputs.resolve()),
        'input_config_sha256': digest(a.inputs),
        'screen_summary': str((a.out / 'screen_summary.json').resolve()),
        'screen_summary_sha256': digest(a.out / 'screen_summary.json'),
        'screen_seeds_excluded': a.screen_seeds,
        'confirm_seeds': a.confirm_seeds,
        'checkpoint': str(a.checkpoint.resolve()),
        'checkpoint_sha256': digest(a.checkpoint),
        'budget': 512,
        'search_seed': 2027,
        'risk_normalization': 'mean/initial_mean; cvar/initial_cvar',
        'methods': ['default', 'det_base', 'det_mean', 'det_tail',
                    'gnn_base', 'gnn_mean', 'gnn_tail'],
        'designs': selected,
    }
    save(a.out / 'confirmatory_config.json', confirm)
    print(json.dumps({'stage': 'screen_complete', 'selected': len(selected),
                      'censored': len(censored),
                      'config': str(a.out / 'confirmatory_config.json')}), flush=True)


if __name__ == '__main__':
    main()
