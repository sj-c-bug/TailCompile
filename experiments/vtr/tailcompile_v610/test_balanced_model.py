"""Meaningful invariants for frozen-weight repair of discrete density collapse."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import unittest
from unittest.mock import patch

import numpy as np
import torch

from balanced_model import BalancedHierarchicalInitializer, repair_assignment
from tailcompile_v69.fast_model import FastHierarchicalInitializer
from tailcompile_v68.model import to_device


def graph(n=64, columns=8, rows=8):
    torch.manual_seed(11)
    r = columns * rows
    bounds = torch.tensor([[x/columns, (x+1)/columns, y/rows, (y+1)/rows]
                          for y in range(rows) for x in range(columns)])
    demand = torch.zeros(n, 5)
    demand[:, 0] = 1
    return dict(cluster_x=torch.randn(n, 8), region_x=torch.randn(r, 13),
        logic_src=torch.arange(n-1), logic_dst=torch.arange(1, n),
        logic_weight=torch.ones(n-1), region_src=torch.arange(r-1),
        region_dst=torch.arange(1, r), region_weight=torch.ones(r-1),
        pool_x=torch.zeros(0, 8), pool_ids=torch.empty(0, dtype=torch.long),
        pool_regions=torch.empty(0, dtype=torch.long), parents=[],
        demand=demand, capacity=torch.ones(r, 5), region_bounds=bounds,
        movable=torch.ones(n, dtype=torch.bool), fixed_region=torch.full((n,), -1),
        fixed_xy=torch.full((n, 2), -1.), channel_width=100.)


def model():
    result = BalancedHierarchicalInitializer(8, 13, hidden=16, chunk_size=17).eval()
    # Exact matching ties reproduce independent argmax collapse without relying
    # on randomly initialized untrained networks to exhibit a particular failure.
    with torch.no_grad():
        result.query.weight.zero_()
    return result


class BalancedDecoderTest(unittest.TestCase):
    def test_discrete_collapse_repaired_beyond_topk_first_bins(self):
        data, m = graph(), model()
        baseline = m.predict(data, repair_passes=0)
        self.assertEqual(baseline['assignment'].unique().numel(), 1)
        with patch('tailcompile_v68.model.capacity_decode', side_effect=AssertionError('hard decode')):
            repaired = m.predict(data, repair_topk=32)
        diag = repaired['repair_diagnostics']
        self.assertLess(diag['after']['normalized_squared_overflow'],
                        diag['before']['normalized_squared_overflow'] / 100)
        self.assertGreater(diag['after']['occupied_region_count'], 48)
        self.assertLessEqual(diag['node_visits'], 2*len(data['demand']))
        self.assertFalse(repaired['hard_regional_capacity_enforced'])
        self.assertFalse(diag['packing_legality_claimed'])

    def test_proxy_capacity_impossible_still_bounded_and_explicit(self):
        data = graph(n=64, columns=2, rows=2)
        result = model().predict(data)
        d = result['repair_diagnostics']
        self.assertGreater(d['after']['normalized_squared_overflow'], 0)
        self.assertLess(d['after']['normalized_squared_overflow'], d['before']['normalized_squared_overflow'])
        self.assertLessEqual(d['moves_attempted_and_accepted'], 128)
        self.assertLessEqual(d['node_visits'], 128)
        self.assertTrue(d['residual_overflow_allowed'])
        one = model().predict(graph(n=10, columns=1, rows=1))
        self.assertEqual(one['repair_diagnostics']['changed_nodes'], 0)
        self.assertEqual(one['repair_diagnostics']['termination'], 'no_improving_move')

    def test_fixed_known_and_unknown_regions_preserved_and_no_mutation(self):
        data = graph(n=20, columns=4, rows=4)
        data['movable'][:2] = False
        data['fixed_region'][0] = 1
        data['fixed_xy'][:2] = torch.tensor([[.3, .1], [.2, .3]])
        original = {k: v.clone() for k, v in data.items() if torch.is_tensor(v)}
        m = model()
        weights = {k: v.clone() for k, v in m.state_dict().items()}
        result = m.predict(data, score_noise_std=.05, coordinate_noise_std=.1, seed=7)
        self.assertEqual(result['assignment'][:2].tolist(), [1, -1])
        self.assertTrue(torch.equal(result['xy'][:2], data['fixed_xy'][:2]))
        self.assertEqual(result['capacity_diagnostics']['fixed_region_unknown_count'], 1)
        for key, value in original.items():
            self.assertTrue(torch.equal(value, data[key]), key)
        for key, value in weights.items():
            self.assertTrue(torch.equal(value, m.state_dict()[key]), key)
        self.assertTrue(all(p.grad is None for p in m.parameters()))

    def test_heterogeneous_multiresource_masks_and_coordinate_conditioning(self):
        data = graph(n=24, columns=4, rows=4)
        data['demand'][::2, 0] = 0
        data['demand'][::2, 1] = 1
        data['capacity'][::2, 0] = 0
        data['capacity'][1::2, 1] = 0
        m = model()
        result = m.predict(data)
        assigned = result['assignment']
        self.assertTrue(torch.all(data['demand'] <= data['capacity'][assigned] + 1e-6))
        expected_xy = m.coordinates_from_encoded(data, m.encode_logic(data), m.encode_device(data), assigned)
        self.assertTrue(torch.allclose(result['xy'], expected_xy, atol=1e-7))
        bounds = data['region_bounds'][assigned]
        self.assertTrue(torch.all(result['xy'] >= bounds[:, [0, 2]]))
        self.assertTrue(torch.all(result['xy'] < bounds[:, [1, 3]]))

    def test_disable_exact_v69_and_weight_loading(self):
        data, m = graph(n=16, columns=4, rows=4), model()
        base = FastHierarchicalInitializer(**m.config).eval()
        base.load_state_dict(m.state_dict(), strict=True)
        a = base.predict(data)
        for kwargs in (dict(repair_passes=0), dict(repair_max_moves=0)):
            b = m.predict(data, **kwargs)
            self.assertTrue(torch.equal(a['assignment'], b['assignment']))
            self.assertTrue(torch.equal(a['xy'], b['xy']))
            self.assertTrue(torch.equal(a['usage'], b['usage']))

    def test_determinism_cached_embeddings_and_timing_scope(self):
        data, m = graph(n=40, columns=6, rows=6), model()
        kwargs = dict(score_noise_std=.1, coordinate_noise_std=.02, seed=17)
        a, b = m.predict(data, **kwargs), m.predict(data, **kwargs)
        c = m.predict(data, logic_embedding=m.encode_logic(data), device_embedding=m.encode_device(data), **kwargs)
        for result in (b, c):
            self.assertTrue(torch.equal(a['assignment'], result['assignment']))
            self.assertTrue(torch.equal(a['xy'], result['xy']))
        phases = a['phase_seconds']
        self.assertGreater(phases['discrete_density_repair'], 0)
        self.assertGreater(phases['repair_matching'], 0)
        self.assertGreaterEqual(phases['total'] + 1e-6, sum(v for k, v in phases.items() if k != 'total'))
        self.assertEqual(c['phase_seconds']['encoding_logic'], 0)
        self.assertEqual(c['phase_seconds']['encoding_device'], 0)

    def test_repair_objective_is_invariant_to_resource_units_and_move_budget(self):
        assignment = np.zeros(8, dtype=np.int64)
        demand = np.ones((8, 2))
        capacity = np.full((4, 2), 2.)
        candidates = np.tile(np.arange(4), (8, 1))
        scores = np.zeros((8, 4))
        centers = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
        common = dict(nodes=np.arange(8), candidate_regions=candidates,
                      candidate_scores=scores, centers=centers, max_moves=3)
        a, _, da = repair_assignment(assignment, demand, capacity, **common)
        d2, c2 = demand.copy(), capacity.copy()
        d2[:, 0] *= 1e6
        c2[:, 0] *= 1e6
        b, _, db = repair_assignment(assignment, d2, c2, **common)
        self.assertTrue(np.array_equal(a, b))
        self.assertAlmostEqual(da['after']['normalized_squared_overflow'], db['after']['normalized_squared_overflow'])
        self.assertEqual(da['moves_attempted_and_accepted'], 3)
        self.assertEqual(da['termination'], 'move_budget')

    def test_invalid_options_and_corrupt_resources_fail(self):
        m, data = model(), graph()
        for kwargs in (dict(repair_passes=-1), dict(repair_passes=True),
                       dict(repair_topk=0), dict(repair_topk=257),
                       dict(repair_max_moves=-1), dict(repair_max_moves=True),
                       dict(repair_locality_weight=float('nan'))):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                m.predict(data, **kwargs)
        data['capacity'][0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            m.predict(data)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_contract(self):
        device = torch.device('cuda:0')
        data, m = to_device(graph(), device), model().to(device)
        result = m.predict(data)
        self.assertEqual(result['xy'].device, device)
        self.assertEqual(result['assignment'].device.type, 'cpu')
        self.assertEqual(result['usage'].device.type, 'cpu')
        self.assertTrue(torch.isfinite(result['xy']).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
