#!/usr/bin/env python3
"""Regression tests for seed aggregation and device/width conditioning."""

from __future__ import annotations

import unittest

import torch

from tailcompile_v3 import GraphLayer, normalized_channel_width
from train_tailcompile_v3_ranker import DEFAULT_WEIGHTS, make_pairs


def trial(seed: int, width: int, status: str, value: float = 1.0) -> dict:
    metrics = None if status != 'success' else {
        'region_p99': value,
        'channel_p99': value,
        'wirelength': value,
        'final_cpd_ns': value,
    }
    return {'placement_seed': seed, 'channel_width': width,
            'status': status, 'metrics': metrics}


def action(name: str, trials: list[dict]) -> dict:
    return {'id': name, 'assignment': [0], 'relative_xy': [[0.0, 0.0]],
            'trials': trials}


class TrainingDefinitionTest(unittest.TestCase):
    def test_seed_is_aggregated_not_used_as_condition(self):
        design = {
            'design': 'tiny',
            'actions': [
                action('a', [trial(11, 100, 'success', 1.0),
                             trial(12, 100, 'success', 1.2)]),
                action('b', [trial(11, 100, 'success', 2.0),
                             trial(12, 100, 'success', 2.2)]),
            ],
        }
        pairs = make_pairs(design, DEFAULT_WEIGHTS, margin=0.05)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['channel_width'], 100)
        self.assertEqual(pairs[0]['preferred']['id'], 'a')
        self.assertNotIn('placement_seed', pairs[0]['condition'])
        self.assertEqual(pairs[0]['preferred_summary']['placement_seed_count'], 2)

    def test_route_failure_rate_is_aggregated(self):
        design = {
            'design': 'tiny',
            'actions': [
                action('unstable', [trial(11, 90, 'success'),
                                    trial(12, 90, 'unroutable')]),
                action('stable', [trial(11, 90, 'success'),
                                  trial(12, 90, 'success')]),
            ],
        }
        pairs = make_pairs(design, DEFAULT_WEIGHTS, margin=0.05)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]['preferred']['id'], 'stable')

    def test_channel_width_feature_is_monotonic(self):
        self.assertLess(normalized_channel_width(80), normalized_channel_width(160))
        with self.assertRaises(ValueError):
            normalized_channel_width(0)

    def test_graph_layer_keeps_temporary_tensors_on_device(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        layer = GraphLayer(4).to(device)
        x = torch.randn(3, 4, device=device)
        src = torch.tensor([0, 1], dtype=torch.long, device=device)
        dst = torch.tensor([1, 2], dtype=torch.long, device=device)
        weight = torch.ones(2, device=device)
        result = layer(x, src, dst, weight)
        self.assertEqual(result.device, device)


if __name__ == '__main__':
    unittest.main()
