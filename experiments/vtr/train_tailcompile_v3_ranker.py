#!/usr/bin/env python3
"""Train the TailCompile v3 GNN with pairwise structured ranking.

The VPR/legalization path is discrete.  We therefore rank complete candidate
actions: the preferred assignment/coordinate action must receive a higher
model energy than a worse action evaluated under the same VPR condition.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import random
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from torch.nn import functional as F

from tailcompile_v3 import TwoTowerPlacementGNN, graph_inputs


DEFAULT_WEIGHTS = {
    'route_failure': 4.0,
    'region_p99': 0.40,
    'channel_p99': 0.20,
    'wirelength': 0.20,
    'final_cpd_ns': 0.20,
}


def source_path(bundle: Path, record: dict) -> Path:
    path = Path(record['path'])
    return path if path.is_absolute() else bundle / path


def normalized_costs(rows: list[tuple[dict, dict]], weights: dict) -> dict[str, float]:
    costs = {action['id']: 0.0 for action, _ in rows}
    for action, trial in rows:
        if trial['status'] != 'success':
            costs[action['id']] += weights['route_failure']
    successful = [(action, trial) for action, trial in rows if trial['status'] == 'success']
    for metric in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns'):
        values = [float(trial['metrics'][metric]) for _, trial in successful
                  if trial['metrics'].get(metric) is not None]
        if len(values) < 2:
            continue
        low, high = min(values), max(values)
        scale = max(high - low, 1e-12)
        for action, trial in successful:
            value = trial['metrics'].get(metric)
            if value is not None:
                costs[action['id']] += weights[metric] * (float(value) - low) / scale
    return costs


def make_pairs(design: dict, weights: dict, margin: float) -> list[dict]:
    conditions = {}
    for action in design['actions']:
        for trial in action['trials']:
            key = (trial['placement_seed'], trial['channel_width'])
            conditions.setdefault(key, []).append((action, trial))
    pairs = []
    for condition, rows in conditions.items():
        if len(rows) < 2:
            continue
        costs = normalized_costs(rows, weights)
        for (left, _), (right, _) in itertools.combinations(rows, 2):
            gap = costs[left['id']] - costs[right['id']]
            if abs(gap) <= margin:
                continue
            preferred, other = (left, right) if gap < 0 else (right, left)
            pairs.append({'design': design['design'], 'condition': condition,
                          'preferred': preferred, 'other': other, 'cost_gap': abs(gap)})
    return pairs


def action_score(scores: torch.Tensor, coordinates: torch.Tensor, action: dict,
                 coordinate_weight: float) -> torch.Tensor:
    assignment = torch.tensor(action['assignment'], dtype=torch.long)
    nodes = torch.arange(len(action['assignment']), dtype=torch.long)
    region_score = scores[nodes, assignment].mean()
    target_xy = torch.tensor(action['relative_xy'], dtype=torch.float32)
    coordinate_score = -F.mse_loss(coordinates, target_xy)
    return region_score + coordinate_weight * coordinate_score


def evaluate(model, pairs: list[dict], graphs: dict, coordinate_weight: float) -> dict:
    model.eval()
    correct, losses = 0, []
    with torch.no_grad():
        for pair in pairs:
            scores, coordinates = model(graphs[pair['design']])
            preferred = action_score(scores, coordinates, pair['preferred'], coordinate_weight)
            other = action_score(scores, coordinates, pair['other'], coordinate_weight)
            correct += int(preferred > other)
            losses.append(float(F.softplus(-(preferred - other))))
    return {'pair_count': len(pairs), 'accuracy': correct / max(len(pairs), 1),
            'mean_pair_loss': sum(losses) / max(len(losses), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-5)
    parser.add_argument('--coordinate-weight', type=float, default=0.2)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--pair-margin', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--validation-designs', nargs='*', default=[])
    parser.add_argument('--smoke-train-all', action='store_true',
                        help='Allow training/evaluation on the same tiny dataset')
    args = parser.parse_args()
    if args.epochs < 1 or args.temperature <= 0:
        parser.error('epochs and temperature must be positive')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    dataset = json.loads(args.dataset.read_text())
    bundle = args.dataset.parent
    weights = dict(DEFAULT_WEIGHTS)
    graphs, all_pairs = {}, []
    for design in dataset['designs']:
        sources = design['sources']
        logic_path = source_path(bundle, sources['logic_ir'])
        device_path = source_path(bundle, sources['device_ir'])
        blif_path = source_path(bundle, sources['mapped_blif'])
        logic = json.loads(logic_path.read_text())
        device = json.loads(device_path.read_text())
        graphs[design['design']] = graph_inputs(logic, device, blif_path)
        all_pairs.extend(make_pairs(design, weights, args.pair_margin))
    if not all_pairs:
        raise SystemExit('no controlled candidate pairs; collect >=2 actions under one condition')
    validation_names = set(args.validation_designs)
    train_pairs = [row for row in all_pairs if row['design'] not in validation_names]
    validation_pairs = [row for row in all_pairs if row['design'] in validation_names]
    if not train_pairs:
        raise SystemExit('validation split leaves no training pairs')
    if not validation_pairs:
        if not args.smoke_train_all:
            raise SystemExit('no validation pairs; pass --validation-designs or --smoke-train-all')
        validation_pairs = train_pairs
    sample_graph = next(iter(graphs.values()))
    model = TwoTowerPlacementGNN(sample_graph['cluster_x'].shape[1],
                                 sample_graph['region_x'].shape[1], args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        shuffled = list(train_pairs)
        random.shuffle(shuffled)
        epoch_loss = 0.0
        for pair in shuffled:
            scores, coordinates = model(graphs[pair['design']])
            preferred = action_score(scores, coordinates, pair['preferred'], args.coordinate_weight)
            other = action_score(scores, coordinates, pair['other'], args.coordinate_weight)
            loss = F.softplus(-(preferred - other) / args.temperature)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach())
        if epoch in {1, args.epochs} or epoch % max(args.epochs // 10, 1) == 0:
            history.append({'epoch': epoch,
                            'train_loss': epoch_loss / max(len(shuffled), 1),
                            'validation': evaluate(model, validation_pairs, graphs,
                                                   args.coordinate_weight)})
    args.out.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        'schema': 'tailcompile-v3-structured-ranker-v1',
        'model_state': model.state_dict(),
        'hidden_width': args.hidden,
        'cluster_feature_width': sample_graph['cluster_x'].shape[1],
        'region_feature_width': sample_graph['region_x'].shape[1],
        'coordinate_weight': args.coordinate_weight,
        'metric_weights': weights,
        'training_designs': sorted({row['design'] for row in train_pairs}),
        'validation_designs': sorted({row['design'] for row in validation_pairs}),
        'smoke_train_all': args.smoke_train_all,
    }
    checkpoint_path = args.out / 'ranker.pt'
    torch.save(checkpoint, checkpoint_path)
    report = {
        'schema': 'tailcompile-v3-training-report-v1',
        'dataset': str(args.dataset), 'checkpoint': str(checkpoint_path),
        'epochs': args.epochs, 'train_pair_count': len(train_pairs),
        'validation_pair_count': len(validation_pairs),
        'final_train': evaluate(model, train_pairs, graphs, args.coordinate_weight),
        'final_validation': evaluate(model, validation_pairs, graphs, args.coordinate_weight),
        'history': history,
        'warning': ('smoke result is not an unbiased generalization estimate'
                    if args.smoke_train_all else None),
    }
    (args.out / 'training_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
