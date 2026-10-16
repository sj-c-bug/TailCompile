import os
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch

from tailcompile_v6 import RoutabilityGNN, apply_tailrisk_ablation, gini, release_mask
from train_tailcompile_v6 import auc, load_fold, report_record, spearman


def graph():
    return {
        'cluster_x': torch.randn(5, 8),
        'logic_src': torch.tensor([0, 1, 2, 3]),
        'logic_dst': torch.tensor([1, 2, 3, 4]),
        'logic_weight': torch.ones(4),
        'region_x': torch.randn(4, 13),
        'region_src': torch.tensor([0, 1, 2, 3]),
        'region_dst': torch.tensor([1, 2, 3, 0]),
        'region_weight': torch.ones(4),
        'spatial_region_x': torch.zeros(4, 7),
        'spatial_global_x': torch.zeros(9),
        'assignment': torch.tensor([0, 1, 2, 3, 0]),
    }


class V6Test(unittest.TestCase):
    def test_tail_ablation_is_targeted(self):
        data = graph()
        data['spatial_region_x'][:, 0] = 2
        data['spatial_region_x'][:, 1:] = 3
        data['spatial_global_x'][:] = torch.arange(9)
        result = apply_tailrisk_ablation(data, False)
        self.assertEqual(result['cluster_x'][0, 7], 0)
        self.assertTrue(torch.equal(result['spatial_region_x'][:, 0],
                                    data['spatial_region_x'][:, 0]))
        self.assertTrue(torch.all(result['spatial_global_x'][1:5] == 0))

    def test_early_ignores_spatial_candidate_but_placement_uses_it(self):
        torch.manual_seed(2)
        model = RoutabilityGNN(8, 13, hidden=12).eval()
        first, second = graph(), graph()
        for key in ('cluster_x', 'logic_src', 'logic_dst', 'logic_weight',
                    'region_x', 'region_src', 'region_dst', 'region_weight'):
            second[key] = first[key].clone()
        second['spatial_global_x'][:] = 5
        second['spatial_region_x'][:] = 5
        with torch.no_grad():
            early_a = model(first, 'early')['route_logit']
            early_b = model(second, 'early')['route_logit']
            place_a = model(first, 'placement')['route_logit']
            place_b = model(second, 'placement')['route_logit']
        self.assertTrue(torch.equal(early_a, early_b))
        self.assertFalse(torch.equal(place_a, place_b))

    def test_statistics(self):
        self.assertAlmostEqual(gini([0, 1]), 0.5)
        self.assertAlmostEqual(spearman([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(auc([0, 1], [0.1, 0.9]), 1.0)

    def test_release_is_confidence_and_budget_limited(self):
        mask = release_mask(torch.tensor([5.0, 4.0, 3.0, -5.0]),
                            threshold=0.70, max_fraction=0.25)
        self.assertEqual(mask.tolist(), [True, False, False, False])

    def test_report_record_excludes_tensors(self):
        row = {'design': 'd', 'width': 80, 'method': 'm', 'route': 0.5,
               'qor': [1, None, 2.0, 3], 'hotspot': [0.1],
               'assignment': torch.tensor([0]),
               'spatial_region_x': torch.zeros(1, 7)}
        record = report_record(row, 0.6, [1, 2, 3, 4], [0.2])
        json.dumps(record, allow_nan=False)
        self.assertNotIn('assignment', record)

    def test_completed_fold_can_be_reused(self):
        data = graph()
        row = {'design': 'heldout', 'width': 80, 'method': 'm', 'route': 1.0,
               'qor': [1.0, 1.0, 1.0, 1.0], 'hotspot': [0.1] * 4,
               'assignment': data['assignment'], 'spatial_region_x': data['spatial_region_x'],
               'spatial_global_x': data['spatial_global_x'], 'graph_key': ('d', 80)}
        model = RoutabilityGNN(8, 13, hidden=12)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'fold.pt'
            torch.save({'model': model.state_dict(), 'mode': 'placement',
                        'use_tailrisk': True, 'qor_normalization': [(0.0, 1.0)] * 4,
                        'heldout_design': 'heldout'}, checkpoint)
            _, _, trace, metrics = load_fold(
                checkpoint, [row], [row], {('d', 80): data}, 'placement', True,
                SimpleNamespace(hidden=12), torch.device('cpu'))
        self.assertTrue(trace[0]['reused_checkpoint'])
        json.dumps(metrics, allow_nan=False)


if __name__ == '__main__':
    unittest.main()
