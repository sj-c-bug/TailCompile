#!/usr/bin/env python3
"""Audit the seed-aggregated TailCompile dataset and design-family holdout."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path,
                        default=HERE / 'tailcompile_v3_training/bundle/dataset.json')
    parser.add_argument('--training-report', type=Path,
                        default=HERE / 'tailcompile_v3_training/model_width_conditioned/training_report.json')
    parser.add_argument('--out', type=Path,
                        default=HERE / 'tailcompile_v3_training/holdout_audit.json')
    args = parser.parse_args()

    dataset = json.loads(args.dataset.read_text())
    training = json.loads(args.training_report.read_text())
    designs = {}
    accepted = set()
    for design in dataset['designs']:
        status_by_width = {}
        for width in sorted({int(trial['channel_width'])
                             for action in design['actions'] for trial in action['trials']}):
            trials = [(action, trial) for action in design['actions']
                      for trial in action['trials'] if int(trial['channel_width']) == width]
            statuses = collections.Counter(trial['status'] for _, trial in trials)
            rates = []
            for action in design['actions']:
                rows = [trial for trial in action['trials']
                        if int(trial['channel_width']) == width]
                if rows:
                    rates.append(sum(row['status'] == 'success' for row in rows) / len(rows))
                for row in rows:
                    accepted.add((design['design'], action['id'], int(row['placement_seed']), width))
            status_by_width[str(width)] = {
                'trial_count': len(trials), 'status': dict(statuses),
                'action_count': len(rates),
                'action_success_rate_range': [min(rates), max(rates)],
            }
        designs[design['design']] = {
            'cluster_count': design['cluster_count'],
            'action_count': design['action_count'],
            'trial_count': design['trial_count'],
            'rejected_trial_count': design['rejected_trial_count'],
            'by_width': status_by_width,
        }

    manifest_versions = collections.Counter()
    manifest_status = collections.Counter()
    orphaned = []
    for manifest_path in sorted((HERE / 'tailcompile_v3').glob('*/4x4/runs/*/manifest.json')):
        row = json.loads(manifest_path.read_text())
        design = row['design']
        manifest_versions[(design, row.get('vtr_commit', 'missing'))] += 1
        manifest_status[(design, row['vpr_status'])] += 1
        key = (design, row.get('input_fplace_sha256'), int(row['seed']),
               int(row['route_chan_width']))
        if key not in accepted:
            orphaned.append(str(manifest_path.relative_to(HERE.parent.parent)))

    deterministic = json.loads(
        (HERE / 'tailcompile_v2/spmv/4x4/build_summary.json').read_text())['methods']
    deterministic_hashes = {
        method: record['fplace_sha256'] for method, record in deterministic.items()
    }
    final_validation = training['final_validation']
    validation_accuracy = float(final_validation['accuracy'])
    result = {
        'schema': 'tailcompile-v3-holdout-audit-v1',
        'dataset_schema': dataset['schema'],
        'design_count': len(dataset['designs']),
        'action_count': sum(row['action_count'] for row in dataset['designs']),
        'trial_count': sum(row['trial_count'] for row in dataset['designs']),
        'rejected_trial_count': sum(row['rejected_trial_count'] for row in dataset['designs']),
        'designs': designs,
        'training_designs': ['attention_layer', 'sha'],
        'validation_designs': ['spmv'],
        'train_pair_count': training['train_pair_count'],
        'validation_pair_count': training['validation_pair_count'],
        'final_train': training['final_train'],
        'final_validation': final_validation,
        'best_logged_validation_accuracy': max(
            ({'epoch': row['epoch'], **row['validation']} for row in training['history']),
            key=lambda row: row['accuracy']),
        'best_logged_validation_loss': min(
            ({'epoch': row['epoch'], **row['validation']} for row in training['history']),
            key=lambda row: row['mean_pair_loss']),
        'manifest_status': {f'{design}:{status}': count
                            for (design, status), count in sorted(manifest_status.items())},
        'manifest_versions': {f'{design}:{commit}': count
                              for (design, commit), count in sorted(manifest_versions.items())},
        'manifest_count': sum(manifest_status.values()),
        'dataset_orphan_manifest_count': len(orphaned),
        'dataset_orphan_manifest_examples': orphaned[:20],
        'spmv_deterministic_fplace_hashes': deterministic_hashes,
        'spmv_deterministic_methods_are_identical': len(set(deterministic_hashes.values())) == 1,
        'assessment': (
            'weak_holdout_generalization_proceed_to_closed_loop_as_a_falsification_test'
            if validation_accuracy < 0.60 else
            'promising_holdout_ranking_proceed_to_closed_loop_confirmation'
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
