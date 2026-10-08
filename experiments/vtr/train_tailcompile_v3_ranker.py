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
import time
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
    assignment = torch.as_tensor(action['assignment'], dtype=torch.long, device=scores.device)
    nodes = torch.arange(len(action['assignment']), dtype=torch.long)
    nodes = nodes.to(scores.device)
    region_score = scores[nodes, assignment].mean()
    target_xy = torch.as_tensor(action['relative_xy'], dtype=torch.float32,
                                device=coordinates.device)
    coordinate_score = -F.mse_loss(coordinates, target_xy)
    return region_score + coordinate_weight * coordinate_score


def score_unique_actions(scores: torch.Tensor, coordinates: torch.Tensor,
                         pairs: list[dict], coordinate_weight: float) -> dict[str, torch.Tensor]:
    actions = {}
    for pair in pairs:
        actions[pair['preferred']['id']] = pair['preferred']
        actions[pair['other']['id']] = pair['other']
    return {action_id: action_score(scores, coordinates, action, coordinate_weight)
            for action_id, action in actions.items()}


def evaluate(model, pairs: list[dict], graphs: dict, coordinate_weight: float) -> dict:
    model.eval()
    correct, losses = 0, []
    with torch.no_grad():
        by_design = {}
        for pair in pairs:
            by_design.setdefault(pair['design'], []).append(pair)
        for design, design_pairs in by_design.items():
            # Graph embeddings only depend on the design, not the candidate pair.
            # Reusing this forward pass avoids thousands of redundant full-graph
            # GNN evaluations per epoch.
            scores, coordinates = model(graphs[design])
            action_scores = score_unique_actions(scores, coordinates, design_pairs,
                                                  coordinate_weight)
            for pair in design_pairs:
                preferred = action_scores[pair['preferred']['id']]
                other = action_scores[pair['other']['id']]
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
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto')
    parser.add_argument('--log-every', type=int, default=10)
    parser.add_argument('--checkpoint-every', type=int, default=20)
    parser.add_argument('--validation-designs', nargs='*', default=[])
    parser.add_argument('--smoke-train-all', action='store_true',
                        help='Allow training/evaluation on the same tiny dataset')
    args = parser.parse_args()
    if (args.epochs < 1 or args.temperature <= 0 or args.log_every < 1
            or args.checkpoint_every < 1):
        parser.error('epochs, temperature, log-every and checkpoint-every must be positive')
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('--device cuda requested but CUDA is unavailable')
    torch_device = torch.device('cuda' if (args.device == 'cuda'
                                           or args.device == 'auto'
                                           and torch.cuda.is_available())
                                else 'cpu')
    print(json.dumps({
        'stage': 'startup', 'python_pid': os.getpid(), 'torch': torch.__version__,
        'requested_device': args.device, 'selected_device': str(torch_device),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device': (torch.cuda.get_device_name(0)
                        if torch_device.type == 'cuda' else None),
    }), flush=True)
    started = time.time()
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
        device_ir = json.loads(device_path.read_text())
        graph = graph_inputs(logic, device_ir, blif_path)
        graphs[design['design']] = {
            key: value.to(torch_device) if isinstance(value, torch.Tensor) else value
            for key, value in graph.items()
        }
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
                                 sample_graph['region_x'].shape[1], args.hidden).to(torch_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=args.weight_decay)
    args.out.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        'stage': 'training_start', 'design_count': len(graphs),
        'train_pair_count': len(train_pairs),
        'validation_pair_count': len(validation_pairs), 'epochs': args.epochs,
    }), flush=True)
    history = []
    train_by_design = {}
    for pair in train_pairs:
        train_by_design.setdefault(pair['design'], []).append(pair)
    for epoch in range(1, args.epochs + 1):
        model.train()
        design_order = list(train_by_design)
        random.shuffle(design_order)
        epoch_loss = 0.0
        epoch_pairs = 0
        for design in design_order:
            design_pairs = train_by_design[design]
            scores, coordinates = model(graphs[design])
            action_scores = score_unique_actions(scores, coordinates, design_pairs,
                                                  args.coordinate_weight)
            pair_losses = []
            for pair in design_pairs:
                preferred = action_scores[pair['preferred']['id']]
                other = action_scores[pair['other']['id']]
                pair_losses.append(F.softplus(-(preferred - other) / args.temperature))
            loss = torch.stack(pair_losses).mean()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += float(loss.detach()) * len(design_pairs)
            epoch_pairs += len(design_pairs)
        if epoch in {1, args.epochs} or epoch % args.log_every == 0:
            row = {'epoch': epoch,
                   'train_loss': epoch_loss / max(epoch_pairs, 1),
                   'validation': evaluate(model, validation_pairs, graphs,
                                          args.coordinate_weight),
                   'elapsed_seconds': round(time.time() - started, 3)}
            history.append(row)
            print(json.dumps({'stage': 'epoch', **row}), flush=True)
            (args.out / 'progress.json').write_text(json.dumps({
                'schema': 'tailcompile-v3-training-progress-v1',
                'device': str(torch_device), 'latest': row, 'history': history,
            }, indent=2) + '\n')
        if epoch % args.checkpoint_every == 0 and epoch != args.epochs:
            torch.save({'schema': 'tailcompile-v3-intermediate-checkpoint-v1',
                        'epoch': epoch, 'model_state': model.state_dict(),
                        'hidden_width': args.hidden}, args.out / 'ranker_latest.pt')
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
        'device': str(torch_device),
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
        'device': str(torch_device), 'elapsed_seconds': round(time.time() - started, 3),
        'warning': ('smoke result is not an unbiased generalization estimate'
                    if args.smoke_train_all else None),
    }
    (args.out / 'training_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'stage': 'training_complete', **report}, indent=2), flush=True)


if __name__ == '__main__':
    main()
