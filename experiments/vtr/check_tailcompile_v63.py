"""Check complete trial coverage for the expanded v6.3 experiment."""
import argparse
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else None


def check(root):
    pool = read(root / 'pool/pool_report.json')
    labels = read(root / 'labels/label_results.json')
    data = read(root / 'data/dataset_report.json')
    models = read(root / 'models/report.json')
    selection = read(root / 'selection/selection_report.json')
    probe = read(root / 'probe/probe_report.json')
    evaluation = read(root / 'evaluation/vpr_results.json')
    result = dict(probe_trials=len(probe['rows']) if probe else 0,
        pool_candidates=len(pool['candidates']) if pool else 0,
        label_trials=len(labels['rows']) if labels else 0,
        reused_v62_trials=labels.get('reused_v62_trials', 0) if labels else 0,
        training_examples=data.get('examples', 0) if data else 0,
        model_checkpoints=len(list((root / 'models').glob('target_*_member*.pt'))),
        selection_count=len(selection['selections']) if selection else 0,
        evaluation_trials=len(evaluation['rows']) if evaluation else 0)
    result['probe_ready'] = bool(probe and probe.get('complete') and result['probe_trials'] == 6)
    result['prepare_ready'] = bool(pool and labels and labels.get('complete') and data and
        result['pool_candidates'] == 168 and result['label_trials'] == 504 and
        result['training_examples'] == 168)
    result['train_ready'] = bool(result['prepare_ready'] and models and result['model_checkpoints'] == 18)
    result['evaluate_ready'] = bool(result['train_ready'] and selection and evaluation and
        evaluation.get('complete') and result['selection_count'] == 12 and
        result['evaluation_trials'] == 108)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--require', choices=('probe', 'prepare', 'train', 'evaluate'))
    args = parser.parse_args()
    result = check(args.out)
    print(json.dumps(result, indent=2))
    if args.require and not result[args.require + '_ready']:
        raise SystemExit(f'v6.3 {args.require} phase incomplete')


if __name__ == '__main__':
    main()
