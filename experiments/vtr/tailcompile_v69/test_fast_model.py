"""Regression tests for bounded soft-density initialization (CPU/CUDA)."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import unittest
from unittest.mock import patch

import torch

from fast_model import (CapacityError, FastHierarchicalInitializer,
                        HierarchicalInitializer, to_device)


def graph(columns=5, rows=3):
    """Small multi-resource AP graph with non-square multilevel device leaves."""
    torch.manual_seed(13)
    n, r = 12, columns * rows
    cells = [(x, y) for y in range(rows) for x in range(columns)]
    parents = []
    while len(cells) > 1:
        coarse = [(x // 2, y // 2) for x, y in cells]
        unique = sorted(set(coarse), key=lambda item: (item[1], item[0]))
        ids = {value: i for i, value in enumerate(unique)}
        parents.append(torch.tensor([ids[value] for value in coarse]))
        cells = unique
    bounds = torch.tensor([[x / columns, (x + 1) / columns, y / rows, (y + 1) / rows]
                           for y in range(rows) for x in range(columns)])
    return dict(cluster_x=torch.randn(n, 8), logic_src=torch.arange(n - 1),
                logic_dst=torch.arange(1, n), logic_weight=torch.ones(n - 1),
                region_x=torch.randn(r, 13), region_src=torch.arange(r - 1),
                region_dst=torch.arange(1, r), region_weight=torch.ones(r - 1),
                parents=parents, pool_x=torch.randn(2, 8),
                pool_ids=torch.tensor([0, 0, 1, 1]),
                pool_regions=torch.tensor([0, min(1, r - 1), max(0, r - 2), r - 1]),
                demand=torch.nn.functional.one_hot(torch.arange(n) % 2, 5).float() * .25,
                capacity=torch.ones(r, 5), region_bounds=bounds,
                movable=torch.ones(n, dtype=torch.bool), channel_width=100.)


class FastInitializerTest(unittest.TestCase):
    def model(self, **kwargs):
        return FastHierarchicalInitializer(8, 13, hidden=16, chunk_size=3, **kwargs).eval()

    def test_checkpoint_compatible_and_variable_hierarchy_preserved(self):
        original = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        model = FastHierarchicalInitializer(**original.config)
        model.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(model.config, original.config)
        for shape in ((9, 7), (5, 3), (2, 2), (1, 1)):
            data = graph(*shape)
            result = model.predict(data)
            self.assertEqual(result['candidate_region_count'], shape[0] * shape[1])
            self.assertEqual(result['assignment'].shape, (12,))
            bounds = data['region_bounds'][result['assignment']]
            self.assertTrue(torch.all(result['xy'] >= bounds[:, [0, 2]]))
            self.assertTrue(torch.all(result['xy'] < bounds[:, [1, 3]]))

    def test_no_hard_decode_sort_or_repair_and_soft_overflow_reported(self):
        data = graph(1, 1)
        model = self.model()
        # Each molecule fits, but total demand cannot fit this one leaf.
        data['capacity'][:] = .3
        with patch('tailcompile_v68.model.capacity_decode', side_effect=AssertionError('hard decode called')):
            with patch('torch.argsort', side_effect=AssertionError('sorting called')):
                result = model.predict(data)
        quality = result['capacity_diagnostics']
        self.assertFalse(result['hard_regional_capacity_enforced'])
        self.assertFalse(quality['packing_legality_claimed'])
        self.assertEqual(quality['overflow_region_count'], 1)
        self.assertGreater(quality['total_capacity_overflow'], 0.)
        self.assertGreater(float(result['usage'].max()), float(data['capacity'].max()))

    def test_resource_absence_rejected_even_without_aggregate_constraint(self):
        data = graph(2, 2)
        data['capacity'][:, 1] = 0
        with self.assertRaisesRegex(CapacityError, 'individually feasible'):
            self.model().predict(data)

    def test_heterogeneous_resources_never_assigned_to_incompatible_leaf(self):
        data = graph(2, 2)
        data['capacity'][::2, 0] = 0
        data['capacity'][1::2, 1] = 0
        for passes in (0, 1, 2, 8):
            result = self.model().predict(data, soft_density_passes=passes)
            self.assertTrue(torch.all(data['demand'] <= data['capacity'][result['assignment']] + 1e-6))

    def test_fixed_positions_and_regions_unchanged_with_noise(self):
        data = graph(2, 2)
        data['movable'][:2] = False
        data['fixed_region'] = torch.full((12,), -1)
        data['fixed_region'][0] = 1
        data['fixed_xy'] = torch.zeros(12, 2)
        data['fixed_xy'][:2] = torch.tensor([[.7, .2], [.1, .3]])
        result = self.model().predict(data, score_noise_std=1., coordinate_noise_std=20., seed=5)
        self.assertEqual(result['assignment'][:2].tolist(), [1, -1])
        self.assertTrue(torch.equal(result['xy'][:2], data['fixed_xy'][:2]))
        self.assertEqual(result['capacity_diagnostics']['fixed_region_unknown_count'], 1)
        self.assertGreaterEqual(float(result['usage'][1, 0]), .25)
        data.pop('fixed_xy')
        with self.assertRaisesRegex(ValueError, 'fixed_xy'):
            self.model().predict(data)

    def test_seeded_sampling_and_cached_embeddings_repeat(self):
        data, model = graph(), self.model()
        options = dict(score_noise_std=3., coordinate_noise_std=.15, seed=23)
        a, b = model.predict(data, **options), model.predict(data, **options)
        cached = model.predict(data, logic_embedding=model.encode_logic(data),
                               device_embedding=model.encode_device(data), **options)
        c = model.predict(data, **dict(options, seed=24))
        self.assertTrue(torch.equal(a['assignment'], b['assignment']))
        self.assertTrue(torch.equal(a['xy'], b['xy']))
        self.assertTrue(torch.equal(a['assignment'], cached['assignment']))
        self.assertTrue(torch.equal(a['xy'], cached['xy']))
        self.assertFalse(torch.equal(a['xy'], c['xy']))
        self.assertEqual(cached['phase_seconds']['encoding_logic'], 0.)
        self.assertEqual(cached['phase_seconds']['encoding_device'], 0.)

    def test_deployment_movable_coordinate_sentinels_are_ignored_without_mutation(self):
        for fixed_count in (0, 2):
            with self.subTest(fixed_count=fixed_count):
                data = graph(2, 2)
                # v68 data.ap_graph stores no coordinates for movable nodes.
                data['fixed_xy'] = torch.full((12, 2), -1.)
                data['fixed_region'] = torch.full((12,), -1)
                if fixed_count:
                    data['movable'][:2] = False
                    data['fixed_xy'][:2] = torch.tensor([[.7, .2], [.1, .3]])
                    data['fixed_region'][0] = 1
                original = {name: value.clone() for name, value in data.items()
                            if torch.is_tensor(value)}
                result = self.model().predict(data, score_noise_std=.2,
                                              coordinate_noise_std=.3, seed=9)
                for name, value in original.items():
                    self.assertTrue(torch.equal(data[name], value), name)
                movable = data['movable']
                self.assertTrue(torch.all(result['assignment'][movable] >= 0))
                self.assertTrue(torch.all((result['xy'][movable] >= 0) &
                                         (result['xy'][movable] <= 1)))
                self.assertTrue(torch.equal(result['xy'][~movable],
                                            original['fixed_xy'][~movable]))

    def test_fixed_coordinate_validation_keeps_range_structure_and_finite_guards(self):
        model = self.model()
        for value in (-1., -.01, 1.01):
            data = graph(2, 2)
            data['fixed_xy'] = torch.full((12, 2), -1.)
            data['movable'][0] = False
            data['fixed_xy'][0] = torch.tensor([value, .2])
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, 'nonmovable.*normalized'):
                    model.predict(data)
        for bad_xy in (torch.full((12, 2), float('nan')),
                       torch.full((12, 2), float('inf')),
                       torch.zeros((11, 2)),
                       torch.empty((12, 2), device='meta')):
            data = graph(2, 2)
            data['fixed_xy'] = bad_xy
            with self.assertRaisesRegex(ValueError, 'finite Nx2 on the model device'):
                model.predict(data)

    def test_direct_mode_matches_dense_argmax(self):
        data, model = graph(), self.model()
        dense = model(data)
        result = model.predict(data, soft_density_passes=0)
        self.assertTrue(torch.equal(result['assignment'], dense['assignment']))
        self.assertTrue(torch.allclose(result['xy'], dense['xy'], atol=1e-7))
        timings = result['phase_seconds']
        self.assertEqual(timings['soft_density'], 0.)
        self.assertGreater(timings['matching'], 0.)
        self.assertGreater(timings['coordinate'], 0.)
        self.assertGreaterEqual(timings['total'] + 1e-6,
                                sum(value for key, value in timings.items() if key != 'total'))

    def test_invalid_inputs_fail_instead_of_silently_relaxing(self):
        model = self.model()
        for field in ('capacity', 'demand', 'cluster_x', 'region_bounds'):
            data = graph()
            data[field].flatten()[0] = float('nan')
            with self.assertRaisesRegex(ValueError, 'finite'):
                model.predict(data)
        for passes in (-1, 9, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'soft_density_passes'):
                model.predict(graph(), soft_density_passes=passes)
        with self.assertRaises(ValueError):
            model.predict(graph(), density_step=float('inf'))
        with self.assertRaises(FloatingPointError):
            model.predict(graph(), logic_embedding=torch.full((12, 16), float('nan')))

    def test_training_loss_updates_same_deployed_generator(self):
        data, model = graph(2, 2), self.model()
        target = torch.arange(12) % 4
        bounds = data['region_bounds'][target]
        xy = (bounds[:, [0, 2]] + bounds[:, [1, 3]]) / 2
        model.loss(data, target, xy).backward()
        self.assertGreater(float(model.query.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(model.coordinate_head[-1].weight.grad.abs().sum()), 0.)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_keeps_device_contract(self):
        device = torch.device('cuda:0')
        model, data = self.model().to(device), to_device(graph(), device)
        result = model.predict(data, score_noise_std=.1, coordinate_noise_std=.1)
        self.assertEqual(result['xy'].device, device)
        self.assertEqual(result['assignment'].device.type, 'cpu')
        self.assertEqual(result['usage'].device.type, 'cpu')
        self.assertTrue(torch.isfinite(result['xy']).all())


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
