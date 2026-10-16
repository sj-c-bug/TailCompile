#!/usr/bin/env python3
"""Leave-one-design-out training for the v6 early/placement information test."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from tailcompile_v2 import contract_edges
from tailcompile_v3 import graph_inputs
from tailcompile_v4_risk import RiskEvaluator, save
from tailcompile_v6 import (QOR_NAMES, RoutabilityGNN, apply_tailrisk_ablation,
                            move_graph, spatial_proxy)
from tailrisk_core import parse_blif


VARIANTS = (
    ('early_no_tail', 'early', False),
    ('early_with_tail', 'early', True),
    ('placement_no_tail', 'placement', False),
    ('placement_with_tail', 'placement', True),
)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def ranks(values):
    order = sorted(range(len(values)), key=lambda i: values[i])
    result, cursor = [0.0] * len(values), 0
    while cursor < len(order):
        end = cursor + 1
        while end < len(order) and values[order[end]] == values[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2
        for index in order[cursor:end]:
            result[index] = rank
        cursor = end
    return result


def spearman(left, right):
    if len(left) < 3:
        return None
    left, right = ranks(left), ranks(right)
    lm, rm = statistics.mean(left), statistics.mean(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) *
                            sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator else None


def auc(labels, scores):
    positive = [score for label, score in zip(labels, scores) if label >= 0.5]
    negative = [score for label, score in zip(labels, scores) if label < 0.5]
    if not positive or not negative:
        return None
    wins = sum((a > b) + 0.5 * (a == b) for a in positive for b in negative)
    return wins / (len(positive) * len(negative))


def build_examples(dataset):
    examples, graph_cache, evaluator_cache = [], {}, {}
    for design_row in dataset['designs']:
        design = design_row['design']
        logic, device = load(design_row['logic_ir']), load(design_row['device_ir'])
        primitives, _ = parse_blif(Path(design_row['mapped_blif']))
        mapping = {atom: int(cluster['id']) for cluster in logic['clusters']
                   for atom in cluster['atoms']}
        edges, _ = contract_edges(primitives, mapping)
        for condition in design_row['conditions']:
            width = int(condition['width'])
            cache_key = design, width
            if cache_key not in graph_cache:
                graph_cache[cache_key] = graph_inputs(
                    logic, device, Path(design_row['mapped_blif']), width)
                evaluator_cache[cache_key] = RiskEvaluator(
                    logic, device, load(condition['supply']), edges)
            region_spatial, global_spatial = spatial_proxy(
                logic, device, evaluator_cache[cache_key], condition['assignment'])
            hotspot = condition['actual_hotspot']
            examples.append({
                'design': design, 'width': width, 'method': condition['method'],
                'route': float(condition['route_success_rate']),
                'qor': [condition['qor'][name] for name in QOR_NAMES],
                'hotspot': ([row['utilization'] for row in hotspot]
                            if hotspot is not None else None),
                'graph_key': cache_key,
                'spatial_region_x': region_spatial,
                'spatial_global_x': global_spatial,
                'assignment': torch.tensor(condition['assignment'], dtype=torch.long),
            })
    return examples, graph_cache


def normalization(examples):
    result = []
    for index in range(len(QOR_NAMES)):
        values = [math.log1p(float(row['qor'][index])) for row in examples
                  if row['qor'][index] is not None]
        mean = statistics.mean(values) if values else 0.0
        std = statistics.pstdev(values) if len(values) > 1 else 1.0
        result.append((mean, max(std, 1e-6)))
    return result


def targets(row, norms, device):
    qor, mask = [], []
    for value, (mean, std) in zip(row['qor'], norms):
        mask.append(value is not None)
        qor.append((math.log1p(float(value)) - mean) / std if value is not None else 0.0)
    hotspot = row['hotspot'] if row['hotspot'] is not None else [0.0] * 16
    return (torch.tensor(row['route'], device=device),
            torch.tensor(qor, device=device), torch.tensor(mask, device=device),
            torch.tensor(hotspot, device=device), row['hotspot'] is not None)


def model_data(row, graph_cache, use_tailrisk, device):
    data = dict(graph_cache[row['graph_key']])
    data['spatial_region_x'] = row['spatial_region_x']
    data['spatial_global_x'] = row['spatial_global_x']
    data['assignment'] = row['assignment']
    return move_graph(apply_tailrisk_ablation(data, use_tailrisk), device)


def evaluate(model, rows, graph_cache, norms, mode, use_tailrisk, device):
    records = []
    model.eval()
    with torch.no_grad():
        for row in rows:
            output = model(model_data(row, graph_cache, use_tailrisk, device), mode)
            probability = torch.sigmoid(output['route_logit']).item()
            prediction = []
            for value, (mean, std) in zip(output['qor'].tolist(), norms):
                prediction.append(math.expm1(value * std + mean))
            records.append({**row, 'predicted_route': probability,
                            'predicted_qor': prediction,
                            'predicted_hotspot': output['hotspot'].tolist()})
    labels = [row['route'] for row in records]
    probabilities = [row['predicted_route'] for row in records]
    groups = defaultdict(list)
    for row in records:
        groups[row['design'], row['width']].append(row)
    correlations = []
    for values in groups.values():
        observed = [row['route'] - (0.05 * math.log1p(row['qor'][0])
                    if row['qor'][0] is not None else 0.0) for row in values]
        predicted = [row['predicted_route'] - 0.05 * math.log1p(
            max(row['predicted_qor'][0], 0.0)) for row in values]
        value = spearman(observed, predicted)
        if value is not None:
            correlations.append(value)
    hotspot_errors = [abs(predicted - actual)
                      for row in records if row['hotspot'] is not None
                      for predicted, actual in zip(row['predicted_hotspot'], row['hotspot'])]
    eps = 1e-7
    return {
        'example_count': len(records),
        'brier': statistics.mean((a - b) ** 2 for a, b in zip(probabilities, labels)),
        'log_loss': statistics.mean(-(label * math.log(max(probability, eps)) +
                                      (1 - label) * math.log(max(1 - probability, eps)))
                                    for label, probability in zip(labels, probabilities)),
        'binary_auc': auc(labels, probabilities),
        'candidate_rank_spearman': statistics.mean(correlations) if correlations else None,
        'hotspot_mae': statistics.mean(hotspot_errors) if hotspot_errors else None,
        'records': records,
    }


def train_fold(train_rows, test_rows, graph_cache, mode, use_tailrisk, args, device):
    norms = normalization(train_rows)
    sample = graph_cache[train_rows[0]['graph_key']]
    torch.manual_seed(args.seed)
    model = RoutabilityGNN(sample['cluster_x'].shape[1], sample['region_x'].shape[1],
                           args.hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                                  weight_decay=1e-4)
    rng = random.Random(args.seed)
    trace = []
    for epoch in range(args.epochs):
        order = list(train_rows)
        rng.shuffle(order)
        model.train()
        total = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, row in enumerate(order, 1):
            output = model(model_data(row, graph_cache, use_tailrisk, device), mode)
            route, qor, mask, hotspot, hotspot_valid = targets(row, norms, device)
            loss = F.binary_cross_entropy_with_logits(output['route_logit'], route)
            if mask.any():
                loss = loss + args.qor_weight * F.mse_loss(output['qor'][mask], qor[mask])
            if hotspot_valid:
                loss = loss + args.hotspot_weight * F.mse_loss(output['hotspot'], hotspot)
                if mode == 'placement':
                    cluster_hotspot = hotspot[row['assignment'].to(device)]
                    target = cluster_hotspot / cluster_hotspot.amax().clamp_min(1e-6)
                    loss = loss + args.release_weight * F.mse_loss(
                        torch.sigmoid(output['intervention_logit']), target)
            loss.backward()
            total += loss.item()
            if step % args.accumulate == 0 or step == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        if epoch == 0 or (epoch + 1) % 10 == 0 or epoch + 1 == args.epochs:
            trace.append({'epoch': epoch + 1, 'mean_loss': total / len(order)})
    return model, norms, trace, evaluate(
        model, test_rows, graph_cache, norms, mode, use_tailrisk, device)


def mean_metric(folds, key):
    values = [fold['metrics'][key] for fold in folds if fold['metrics'][key] is not None]
    return statistics.mean(values) if values else None


def train(dataset_path, out, args):
    dataset = load(dataset_path)
    examples, graph_cache = build_examples(dataset)
    designs = sorted({row['design'] for row in examples})
    if len(designs) < 3:
        raise ValueError('leave-one-design-out validation requires at least three designs')
    device = torch.device(args.device if args.device != 'auto'
                          else 'cuda' if torch.cuda.is_available() else 'cpu')
    report = {'schema': 'tailcompile-v6-information-ablation-v1',
              'device': str(device), 'designs': designs,
              'fairness_lock': dataset['comparison'], 'variants': {}}
    out.mkdir(parents=True, exist_ok=True)
    for name, mode, use_tailrisk in VARIANTS:
        folds = []
        for heldout in designs:
            train_rows = [row for row in examples if row['design'] != heldout]
            test_rows = [row for row in examples if row['design'] == heldout]
            model, norms, trace, metrics = train_fold(
                train_rows, test_rows, graph_cache, mode, use_tailrisk, args, device)
            checkpoint = out / f'{name}_holdout_{heldout}.pt'
            torch.save({'model': model.state_dict(), 'mode': mode,
                        'use_tailrisk': use_tailrisk, 'qor_normalization': norms,
                        'heldout_design': heldout}, checkpoint)
            folds.append({'heldout_design': heldout, 'trace': trace, 'metrics': metrics,
                          'checkpoint': checkpoint.name})
            print(json.dumps({'stage': 'fold_complete', 'variant': name,
                              'heldout': heldout, 'metrics': {k: v for k, v in metrics.items()
                                                             if k != 'records'}}), flush=True)
        report['variants'][name] = {
            'mode': mode, 'use_tailrisk': use_tailrisk, 'folds': folds,
            'mean_heldout_brier': mean_metric(folds, 'brier'),
            'mean_heldout_log_loss': mean_metric(folds, 'log_loss'),
            'mean_heldout_auc': mean_metric(folds, 'binary_auc'),
            'mean_candidate_rank_spearman': mean_metric(folds, 'candidate_rank_spearman'),
            'mean_hotspot_mae': mean_metric(folds, 'hotspot_mae'),
        }
    early = report['variants']['early_with_tail']
    placed = report['variants']['placement_with_tail']
    relative_brier = ((early['mean_heldout_brier'] - placed['mean_heldout_brier']) /
                      max(early['mean_heldout_brier'], 1e-12))
    rank_gain = ((placed['mean_candidate_rank_spearman'] or 0.0) -
                 (early['mean_candidate_rank_spearman'] or 0.0))
    report['decision'] = {
        'relative_brier_improvement_from_placement': relative_brier,
        'candidate_rank_spearman_gain_from_placement': rank_gain,
        'early_prediction_is_poor': bool(early['mean_heldout_brier'] >= 0.20 or
                                         (early['mean_heldout_auc'] or 0.0) <= 0.60),
        'coarse_placement_is_worth_latency': bool(
            relative_brier >= 0.10 or
            (rank_gain >= 0.15 and (placed['mean_candidate_rank_spearman'] or 0.0) >= 0.30)),
        'preregistered_rule': ('accept coarse-placement latency if held-out Brier improves by >=10%, '
                               'or candidate ranking Spearman improves by >=0.15 and reaches >=0.30; '
                               'flag early mode poor at Brier>=0.20 or AUC<=0.60'),
        'tailrisk_increment_early_brier': (report['variants']['early_no_tail']['mean_heldout_brier'] -
                                           early['mean_heldout_brier']),
        'tailrisk_increment_placement_brier': (
            report['variants']['placement_no_tail']['mean_heldout_brier'] -
            placed['mean_heldout_brier']),
    }
    save(out / 'evaluation_report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--hidden', type=int, default=48)
    parser.add_argument('--learning-rate', type=float, default=2e-3)
    parser.add_argument('--qor-weight', type=float, default=0.20)
    parser.add_argument('--hotspot-weight', type=float, default=0.50)
    parser.add_argument('--release-weight', type=float, default=0.10)
    parser.add_argument('--accumulate', type=int, default=8)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    report = train(args.dataset, args.out, args)
    print(json.dumps({'stage': 'v6_training_complete', 'decision': report['decision']},
                     indent=2), flush=True)


if __name__ == '__main__':
    main()
