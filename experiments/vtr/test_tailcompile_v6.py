import os
import unittest

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch

from tailcompile_v6 import RoutabilityGNN, apply_tailrisk_ablation, gini, release_mask
from train_tailcompile_v6 import auc, spearman


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


if __name__ == '__main__':
    unittest.main()
