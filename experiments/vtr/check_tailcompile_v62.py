"""Compact v6.2 phase and acceptance report."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--require', choices=('prepare', 'train', 'pilot', 'full'))
    a = p.parse_args()
    pool = a.out / 'pool/pool_report.json'
    labels = a.out / 'labels/label_results.json'
    data = a.out / 'data/dataset_report.json'
    models = a.out / 'models/report.json'
    selection = a.out / 'selection/selection_report.json'
    pilot = a.out / 'pilot/vpr_results.json'
    full = a.out / 'full/vpr_results.json'
    result = {
        'pool_candidates': len(read(pool)['candidates']) if pool.exists() else 0,
        'label_complete': labels.exists() and read(labels).get('complete') is True,
        'label_trials': len(read(labels).get('rows', [])) if labels.exists() else 0,
        'training_examples': read(data).get('examples', 0) if data.exists() else 0,
        'model_checkpoints': len(list((a.out / 'models').glob('target_*_member*.pt'))),
        'selection_count': len(read(selection).get('selections', [])) if selection.exists() else 0,
        'pilot_complete': pilot.exists() and read(pilot).get('complete') is True,
        'pilot_trials': len(read(pilot).get('rows', [])) if pilot.exists() else 0,
        'pilot_gate_passed': read(pilot).get('pilot_gate_passed') if pilot.exists() else None,
        'full_complete': full.exists() and read(full).get('complete') is True,
        'full_trials': len(read(full).get('rows', [])) if full.exists() else 0,
    }
    result['prepare_ready'] = (result['pool_candidates'] == 84 and result['label_complete'] and
                               result['label_trials'] == 56 and result['training_examples'] == 28)
    result['train_ready'] = result['prepare_ready'] and models.exists() and result['model_checkpoints'] == 18
    result['pilot_ready'] = result['train_ready'] and result['selection_count'] == 12 and \
        result['pilot_complete'] and result['pilot_trials'] == 24
    result['full_ready'] = result['pilot_ready'] and result['full_complete'] and result['full_trials'] == 72
    print(json.dumps(result, indent=2, ensure_ascii=False))
    required = {'prepare': 'prepare_ready', 'train': 'train_ready',
                'pilot': 'pilot_ready', 'full': 'full_ready'}
    if a.require and not result[required[a.require]]:
        raise SystemExit(f'v6.2 {a.require} phase incomplete')


if __name__ == '__main__':
    main()
