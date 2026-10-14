#!/usr/bin/env python3
"""Analyze conditional TailRisk benefit across locked heterogeneity strata."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path

from tailcompile_v4_risk import digest, save


METRICS = ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for pos in order[index:end]:
            result[pos] = rank
        index = end
    return result


def correlation(left, right):
    if len(left) < 3:
        return None
    lx, rx = ranks(left), ranks(right)
    lm, rm = statistics.mean(lx), statistics.mean(rx)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(lx, rx))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in lx) *
                            sum((b - rm) ** 2 for b in rx))
    return numerator / denominator if denominator else None


def bootstrap_ci(values, seed=2027, repetitions=10000):
    if not values:
        return None
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choice(values) for _ in values)
                   for _ in range(repetitions))
    return [means[int(0.025 * repetitions)], means[int(0.975 * repetitions)]]


def compare(index, designs, lock, candidate, reference):
    condition_rows, design_rows = [], []
    for design in designs:
        route_values, metric_values = [], defaultdict(list)
        keys = sorted({(width, seed) for d, width, method, seed in index
                       if d == design and method in (candidate, reference)})
        for width, seed in keys:
            tail = index.get((design, width, candidate, seed))
            base = index.get((design, width, reference, seed))
            if not tail or not base:
                raise ValueError(f'missing pair: {design} w{width} s{seed}')
            route_gain = int(tail['status'] == 'success') - int(base['status'] == 'success')
            route_values.append(route_gain)
            row = {'design': design, 'width': width, 'seed': seed,
                   'stratum': lock['strata'][design],
                   'heterogeneity': lock['design_scores'][design],
                   'candidate': candidate, 'reference': reference,
                   'route_success_gain': route_gain,
                   'joint_success': tail['status'] == base['status'] == 'success'}
            if row['joint_success']:
                for metric in METRICS:
                    # Positive means that TailRisk reduced a lower-is-better metric.
                    value = 100 * (float(base[metric]) - float(tail[metric])) / float(base[metric])
                    row[f'{metric}_improvement_percent'] = value
                    metric_values[metric].append(value)
            condition_rows.append(row)
        summary = {'design': design, 'stratum': lock['strata'][design],
                   'heterogeneity': lock['design_scores'][design],
                   'candidate': candidate, 'reference': reference,
                   'route_success_gain': statistics.mean(route_values),
                   'pair_count': len(route_values)}
        for metric in METRICS:
            summary[f'{metric}_improvement_percent'] = (
                statistics.mean(metric_values[metric]) if metric_values[metric] else None)
            summary[f'{metric}_joint_success_count'] = len(metric_values[metric])
        design_rows.append(summary)
    interactions = {'route_success_gain_spearman': correlation(
        [row['heterogeneity'] for row in design_rows],
        [row['route_success_gain'] for row in design_rows])}
    for metric in METRICS:
        usable = [row for row in design_rows if row[f'{metric}_improvement_percent'] is not None]
        interactions[f'{metric}_improvement_spearman'] = correlation(
            [row['heterogeneity'] for row in usable],
            [row[f'{metric}_improvement_percent'] for row in usable])
    strata = {}
    for stratum in ('low', 'medium', 'high'):
        rows = [row for row in design_rows if row['stratum'] == stratum]
        strata[stratum] = {'design_count': len(rows)}
        for key in ('route_success_gain', *(f'{m}_improvement_percent' for m in METRICS)):
            values = [row[key] for row in rows if row.get(key) is not None]
            strata[stratum][key] = statistics.mean(values) if values else None
            strata[stratum][key + '_design_bootstrap_95ci'] = bootstrap_ci(values)
    return condition_rows, design_rows, interactions, strata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    root = a.root.resolve()
    out = a.out or root / 'confirmatory_analysis.json'
    summary, plan, lock = read(root / 'summary.json'), read(root / 'plan.json'), read(root / 'heterogeneity_lock.json')
    if summary['missing'] or summary['metric_errors'] or summary['observed'] != summary['expected']:
        raise ValueError('campaign is incomplete')
    if not lock['created_before_confirmatory_vpr'] or lock['vpr_labels_used_for_stratification']:
        raise ValueError('heterogeneity strata are not preregistered')
    rows, index = [], {}
    for spec in plan['runs']:
        directory = root / spec['directory']
        metrics = read(directory / 'metrics.json')
        manifest = read(directory / 'manifest.json')
        if metrics['manifest_sha256'] != digest(directory / 'manifest.json'):
            raise ValueError(f'metrics provenance mismatch: {directory}')
        if spec.get('candidate') and manifest['signature']['input_fplace_sha256'] != digest(root / spec['candidate']):
            raise ValueError(f'candidate provenance mismatch: {directory}')
        key = spec['design'], int(spec['width']), spec['method'], int(spec['seed'])
        if key in index:
            raise ValueError(f'duplicate trial: {key}')
        index[key] = metrics
        rows.append(metrics)
    designs = sorted(lock['strata'])
    comparisons = {}
    all_condition_rows, all_design_rows = [], []
    comparison_plan = (
        ('gnn_tail', 'gnn_mean'),
        ('gnn_tail', 'gnn_base'),
        ('gnn_tail', 'default'),
        ('det_tail', 'det_mean'),
        ('det_tail', 'det_base'),
        ('det_tail', 'default'),
    )
    for candidate, reference in comparison_plan:
        condition, design, interaction, strata = compare(
            index, designs, lock, candidate, reference)
        key = f'{candidate}_vs_{reference}'
        comparisons[key] = {'interaction': interaction, 'strata': strata}
        all_condition_rows.extend(condition)
        all_design_rows.extend(design)
    high = comparisons['gnn_tail_vs_gnn_mean']['strata']['high']
    low = comparisons['gnn_tail_vs_gnn_mean']['strata']['low']
    verdict = {
        'primary_claim_supported': bool(
            high['route_success_gain'] is not None and high['route_success_gain'] >= 0 and
            high['region_p99_improvement_percent'] is not None and
            high['region_p99_improvement_percent'] > 0 and
            comparisons['gnn_tail_vs_gnn_mean']['interaction']['region_p99_improvement_spearman'] is not None and
            comparisons['gnn_tail_vs_gnn_mean']['interaction']['region_p99_improvement_spearman'] > 0),
        'rule': ('high-H Tail must not reduce route success, must improve region P99, and '
                 'the design-level H-by-P99-benefit Spearman coefficient must be positive'),
        'low_stratum_is_negative_control': low,
        'warning': 'two stratum means alone are descriptive; design-family bootstrap CIs and raw pairs remain primary evidence',
    }
    payload = {'schema': 'tailcompile-v5-conditional-confirmation-v1',
               'trial_count': len(rows), 'design_count': len(designs),
               'status_counts': summary['status_counts'],
               'heterogeneity_lock': lock, 'comparisons': comparisons,
               'verdict': verdict,
               'primary_comparison': 'gnn_tail_vs_gnn_mean',
               'deterministic_replication': 'det_tail_vs_det_mean',
               'claim_scope': ('conditional benefit on unseen spatially heterogeneous designs; '
                               'no universal-improvement claim')}
    save(out, payload)
    csv_path = out.with_suffix('.csv')
    if all_design_rows:
        fields = sorted({key for row in all_design_rows for key in row})
        with csv_path.open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_design_rows)
    save(out.with_name(out.stem + '_paired_rows.json'), all_condition_rows)
    print(json.dumps({'stage': 'analysis_complete', 'trial_count': len(rows),
                      'design_count': len(designs), 'verdict': verdict}), flush=True)


if __name__ == '__main__':
    main()
