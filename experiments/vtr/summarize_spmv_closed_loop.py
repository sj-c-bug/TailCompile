#!/usr/bin/env python3
"""Summarize the spmv holdout closed loop while raw VPR occupancy files exist."""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics
from pathlib import Path

from prepare_tailcompile_v3_training import success_metrics


HERE = Path(__file__).resolve().parent
METRICS = ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')


def candidate_hashes(pool: Path) -> set[str]:
    summary = json.loads((pool / 'build_summary.json').read_text())
    return {row['fplace_sha256'] for row in summary['candidates']}


def median_metrics(rows: list[dict]) -> dict:
    result = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        result[metric] = statistics.median(values) if values else None
    return result


def summarize(method: str, width: int, rows: list[dict]) -> dict:
    statuses = collections.Counter(row['status'] for row in rows)
    successful = [row['metrics'] for row in rows if row['metrics'] is not None]
    by_action = collections.defaultdict(list)
    for row in rows:
        by_action[row['action']].append(row)
    action_rates = [sum(row['status'] == 'success' for row in trials) / len(trials)
                    for trials in by_action.values()]
    return {
        'method': method, 'channel_width': width, 'trial_count': len(rows),
        'action_count': len(by_action), 'status': dict(statuses),
        'route_success_rate': statuses['success'] / max(len(rows), 1),
        'action_success_rate_median': statistics.median(action_rates) if action_rates else None,
        'action_success_rate_range': ([min(action_rates), max(action_rates)]
                                      if action_rates else None),
        'successful_trial_metrics_median': median_metrics(successful),
        'metric_extraction_error_count': sum('metric_error' in row for row in rows),
    }


def load_trial(manifest_path: Path, design: str) -> dict:
    meta = json.loads(manifest_path.read_text())
    result = {
        'status': meta['vpr_status'],
        'seed': int(meta['seed']), 'width': int(meta['route_chan_width']),
        'action': meta.get('input_fplace_sha256') or f'default-seed-{meta["seed"]}',
        'metrics': None,
    }
    if result['status'] == 'success':
        try:
            result['metrics'] = success_metrics(manifest_path.parent, design)
        except (FileNotFoundError, ValueError, ZeroDivisionError) as error:
            result['metric_error'] = f'{type(error).__name__}: {error}'
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=HERE / 'tailcompile_v3_training/spmv_closed_loop_summary.json')
    parser.add_argument('--csv', type=Path,
                        default=HERE / 'tailcompile_v3_training/spmv_closed_loop_summary.csv')
    args = parser.parse_args()

    v3_root = HERE / 'tailcompile_v3/spmv/4x4'
    bootstrap = candidate_hashes(v3_root / 'pools/bootstrap_24')
    trained = {
        width: candidate_hashes(v3_root / f'pools/trained_w{width}_24')
        for width in (80, 90, 100, 110)
    }
    records = collections.defaultdict(list)
    unclassified = []

    for manifest in sorted((HERE / 'tailcompile_v2/spmv/4x4/runs').glob('*/manifest.json')):
        meta = json.loads(manifest.read_text())
        if meta['method'] not in ('default_ap', 'tail_rrg'):
            continue
        method = 'default_ap' if meta['method'] == 'default_ap' else 'deterministic_shared'
        trial = load_trial(manifest, 'spmv')
        records[(method, trial['width'])].append(trial)

    for manifest in sorted((v3_root / 'runs').glob('*/manifest.json')):
        meta = json.loads(manifest.read_text())
        action = meta.get('input_fplace_sha256')
        width = int(meta['route_chan_width'])
        if action in bootstrap:
            method = 'bootstrap_gnn'
        elif action in trained.get(width, set()):
            method = 'trained_gnn'
        else:
            unclassified.append(str(manifest.relative_to(HERE.parent.parent)))
            continue
        records[(method, width)].append(load_trial(manifest, 'spmv'))

    rows = [summarize(method, width, trials)
            for (method, width), trials in sorted(records.items())]
    deterministic_hashes = json.loads(
        (HERE / 'tailcompile_v2/spmv/4x4/build_summary.json').read_text())['methods']
    payload = {
        'schema': 'tailcompile-spmv-closed-loop-summary-v1',
        'rows': rows,
        'method_aliases': {
            'rrg_only': 'deterministic_shared',
            'mean_rrg': 'deterministic_shared',
            'tail_rrg': 'deterministic_shared',
        },
        'deterministic_fplace_hashes': {
            key: value['fplace_sha256'] for key, value in deterministic_hashes.items()
        },
        'unclassified_manifest_count': len(unclassified),
        'unclassified_manifest_examples': unclassified[:20],
    }
    args.out.write_text(json.dumps(payload, indent=2) + '\n')
    with args.csv.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=(
            'method', 'channel_width', 'trial_count', 'action_count',
            'route_success_rate', 'action_success_rate_median',
            'region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns',
            'metric_extraction_error_count'))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                **{key: row[key] for key in (
                    'method', 'channel_width', 'trial_count', 'action_count',
                    'route_success_rate', 'action_success_rate_median',
                    'metric_extraction_error_count')},
                **row['successful_trial_metrics_median'],
            })
    print(json.dumps(payload, indent=2))


if __name__ == '__main__':
    main()
