#!/usr/bin/env python3
"""Audit compact v4 evidence without raw RRG/occupancy files or GPU dependencies."""
import argparse
import json
import statistics
from pathlib import Path

from tailcompile_v4_risk import VTR_COMMIT, digest, save


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def analyze(roots):
    comparisons, audits, trials = [], [], []
    for root in roots:
        summary = read(root / 'summary.json')
        plan = read(root / 'plan.json')
        if summary['missing'] or summary['metric_errors'] or summary['observed'] != len(plan['runs']):
            raise ValueError(f'incomplete campaign: {root}')
        for spec in plan['runs']:
            directory = root / spec['directory']
            manifest = read(directory / 'manifest.json')
            metrics = read(directory / 'metrics.json')
            if manifest['vtr_commit'] != VTR_COMMIT or manifest['vpr_status'] not in ('success', 'unroutable'):
                raise ValueError(f'invalid trial: {directory}')
            if metrics['manifest_sha256'] != digest(directory / 'manifest.json'):
                raise ValueError(f'metrics provenance changed: {directory}')
            if spec.get('candidate') and digest(root / spec['candidate']) != manifest['signature']['input_fplace_sha256']:
                raise ValueError(f'candidate changed: {directory}')
            trials.append(metrics)
        index = {(r['design'], r['width'], r['method']): r for r in summary['aggregates']}
        for key, row in index.items():
            if row['method'] not in ('det_tail', 'gnn_tail'):
                continue
            baseline = index[key[0], key[1], row['method'].split('_')[0] + '_base']
            build = read(root / row['design'] / f'w{row["width"]}' / 'candidates/build_summary.json')
            candidate = next(c for c in build['candidates'] if c['method'] == row['method'])
            paired = [p for p in summary['paired_changes'] if p['design'] == row['design']
                      and p['width'] == row['width'] and p['method'] == row['method']]
            comparisons.append({'design': row['design'], 'width': row['width'], 'method': row['method'],
                                'baseline_success': baseline['success'], 'refined_success': row['success'],
                                'trials_per_method': row['trials'],
                                'jointly_successful_pairs': sum(p['joint_success'] for p in paired),
                                'success_gains': sum(p['before_status'] != 'success' and p['after_status'] == 'success' for p in paired),
                                'success_losses': sum(p['before_status'] == 'success' and p['after_status'] != 'success' for p in paired),
                                'proxy_cvar_change_percent': 100 * (candidate['final_proxy']['cvar'] / candidate['initial_proxy']['cvar'] - 1),
                                'median_qor_change_percent': {
                                    m: 100 * (row[m] / baseline[m] - 1)
                                    for m in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')
                                    if row.get(m) is not None and baseline.get(m)},
                                'comparison_warning': 'median QoR ratios are descriptive; consult joint-success paired changes when success sets differ'})
        for path in sorted(root.glob('*/w*/candidates/*_search.json')):
            report = read(path)
            if report['evaluations'] != report['budget']:
                raise ValueError(f'budget not reached: {path}')
            if any(s['objective_after'] >= s['objective_before'] for s in report['trace']):
                raise ValueError(f'non-improving accepted move: {path}')
            audits.append({'file': str(path.relative_to(root)), 'evaluations': report['evaluations'],
                           'accepted': report['accepted'], 'seconds': report['seconds']})
    retention = [t['retention']['retained_fraction'] for t in trials
                 if t.get('retention') and t['retention']['retained_fraction'] is not None]
    return {'schema': 'tailcompile-v4-development-audit-v1', 'vtr_commit': VTR_COMMIT,
            'trial_count': len(trials), 'success': sum(t['status'] == 'success' for t in trials),
            'unroutable': sum(t['status'] == 'unroutable' for t in trials),
            'designs': sorted({t['design'] for t in trials}),
            'tail_comparisons': comparisons, 'search_audits': audits,
            'median_post_route_atom_region_retention': statistics.median(retention) if retention else None,
            'limitations': ['two placement seeds are not independent design replicates',
                            'spmv final input is unavailable locally; no spmv v4 conclusion',
                            'mean and tail use the same CVaR reference, not matched initial objective magnitude',
                            'all hotspot-search objectives share risk-informed proposal priorities',
                            'wire-pool proxy does not guarantee detailed RRG connectivity']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots', type=Path, nargs='+', required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    payload = analyze(args.roots)
    save(args.out, payload)
    print(json.dumps({k: payload[k] for k in ('trial_count', 'success', 'unroutable', 'designs')}))
