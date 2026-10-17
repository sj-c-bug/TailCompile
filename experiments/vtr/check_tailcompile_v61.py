"""Report TailCompile v6.1 phase completeness without loading large tensors."""
import argparse
import json
from collections import Counter
from pathlib import Path


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(root):
    prepare = root / 'data/prepare_report.json'
    bundle = root / 'data/bundle.pt'
    models = root / 'models'
    candidates = root / 'candidates'
    generation = candidates / 'generation_report.json'
    results = candidates / 'vpr_results.json'

    prepared = prepare.is_file() and bundle.is_file() and bundle.stat().st_size > 0
    design_count = None
    if prepare.is_file():
        report = read(prepare)
        design_count = len({row['design'] for row in report.get('conditions', [])})
        if not design_count:
            design_count = report.get('design_count')

    checkpoint_count = len(list(models.glob('holdout_*.pt')))
    model_report = models / 'report.json'
    trained = model_report.is_file() and checkpoint_count > 0
    if model_report.is_file():
        design_count = design_count or len(read(model_report).get('folds', []))

    selected_count = len(list(candidates.glob('*/w*/selected.fplace')))
    generated = generation.is_file() and selected_count > 0
    expected_candidates = None
    if generation.is_file():
        expected_candidates = len(read(generation).get('candidates', []))

    rows, complete, comparisons = [], False, False
    if results.is_file():
        report = read(results)
        rows = report.get('rows', [])
        complete = report.get('complete') is True
        comparisons = bool(report.get('comparisons'))
    keys = [(r.get('design'), r.get('width'), r.get('method'), r.get('seed')) for r in rows]
    status_counts = Counter(r.get('status') for r in rows)
    expected_trials = (expected_candidates * 3 * 2
                       if expected_candidates is not None else None)
    valid_statuses = all(r.get('status') in ('success', 'unroutable') for r in rows)
    accepted = (prepared and trained and generated and complete and comparisons and
                len(keys) == len(set(keys)) and valid_statuses and
                expected_trials is not None and len(rows) == expected_trials and
                (design_count is None or checkpoint_count == design_count) and
                selected_count == expected_candidates)
    return dict(accepted=accepted, prepared=prepared, trained=trained, generated=generated,
                result_complete=complete, design_count=design_count,
                checkpoint_count=checkpoint_count, selected_fplace_count=selected_count,
                expected_candidates=expected_candidates, trial_count=len(rows),
                expected_trials=expected_trials, duplicate_trials=len(keys) - len(set(keys)),
                status_counts=dict(status_counts), comparisons_present=comparisons)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--require-complete', action='store_true')
    args = parser.parse_args()
    result = check(args.out)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.require_complete and not result['accepted']:
        raise SystemExit('TailCompile v6.1 is incomplete')


if __name__ == '__main__':
    main()
