"""CPU/CUDA regression tests for the actual v6.8 hierarchical generator."""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import unittest
import torch

from model import HierarchicalInitializer, CapacityError, capacity_decode, to_device


def graph(columns=5, rows=3):
    torch.manual_seed(13)
    n, r = 12, columns * rows
    # Non-square/non-power-of-two leaves and recursive geometry-defined parents.
    coords = [(x, y) for y in range(rows) for x in range(columns)]
    levels = []
    while len(coords) > 1:
        parents = [(x // 2, y // 2) for x, y in coords]
        unique = sorted(set(parents), key=lambda p: (p[1], p[0]))
        ids = {value: index for index, value in enumerate(unique)}
        levels.append(torch.tensor([ids[value] for value in parents]))
        coords = unique
    bounds = torch.tensor([[x / columns, (x + 1) / columns,
                            y / rows, (y + 1) / rows]
                           for y in range(rows) for x in range(columns)])
    resource = torch.arange(n) % 2
    return dict(cluster_x=torch.randn(n, 8), logic_src=torch.arange(n - 1),
                logic_dst=torch.arange(1, n), logic_weight=torch.ones(n - 1),
                logic_criticality=torch.linspace(0, 1, n - 1),
                region_x=torch.randn(r, 13), region_src=torch.arange(r - 1),
                region_dst=torch.arange(1, r), region_weight=torch.ones(r - 1),
                parents=levels, pool_x=torch.randn(2, 8),
                pool_ids=torch.tensor([0, 0, 1, 1]),
                pool_regions=torch.tensor([0, min(1, r - 1), max(0, r - 2), r - 1]),
                demand=torch.nn.functional.one_hot(resource, 5).float() * .25,
                capacity=torch.ones(r, 5), region_bounds=bounds,
                movable=torch.ones(n, dtype=torch.bool), channel_width=100.)


class GeneratorTest(unittest.TestCase):
    def test_variable_graph_hierarchy_and_region_conditioned_coordinates(self):
        for columns, rows in ((9, 7), (5, 3), (2, 2), (1, 1)):
            data = graph(columns, rows)
            model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
            output = model(data)
            self.assertEqual(output['logits'].shape, (12, columns * rows))
            bounds = data['region_bounds'][output['assignment']]
            self.assertTrue(torch.all(output['xy'] >= bounds[:, [0, 2]]))
            self.assertTrue(torch.all(output['xy'] <= bounds[:, [1, 3]]))
            first = model(data, torch.zeros(12, dtype=torch.long))['xy']
            last = model(data, torch.full((12,), columns * rows - 1))['xy']
            if columns * rows > 1:
                self.assertFalse(torch.allclose(first, last))

    def test_capacity_upper_bound_is_not_exact_proportional_balance(self):
        scores = torch.tensor([[9., 0.], [8., 0.], [7., 0.]])
        assignment, usage = capacity_decode(scores, torch.ones(3, 1), torch.full((2, 1), 10.))
        self.assertEqual(assignment.tolist(), [0, 0, 0])
        self.assertEqual(usage.tolist(), [[3.], [0.]])

    def test_weighted_multiresource_sparse_decode_and_fixed_reservation(self):
        scores = torch.tensor([[1., 0.], [2., 0.], [1., 0.], [1., 0.]])
        demands = torch.tensor([[.7, 0.], [.4, .2], [0., .8], [.4, .2]])
        capacities = torch.ones(2, 2)
        assignment, usage = capacity_decode(scores, demands, capacities,
            movable=torch.tensor([False, True, True, True]),
            fixed_region=torch.tensor([0, -1, -1, -1]))
        self.assertEqual(int(assignment[0]), 0)
        self.assertTrue(torch.all(usage <= capacities + 1e-6))
        with self.assertRaises(CapacityError):
            capacity_decode(torch.ones(1, 2), torch.full((1, 2), 2.), capacities)

    def test_loss_updates_actual_generator_and_predict_is_capacity_safe(self):
        data = graph()
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        target = torch.arange(12) % 15
        bounds = data['region_bounds'][target]
        xy = (bounds[:, [0, 2]] + bounds[:, [1, 3]]) / 2
        loss, details = model.loss(data, target, xy, return_details=True)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name in ('logic_input.weight', 'demand_input.weight',
                     'region_input.weight', 'capacity_input.weight', 'pool.0.weight',
                     'up.0.weight', 'down.0.weight', 'query.weight',
                     'coordinate_head.0.weight'):
            gradient = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(gradient, name)
            self.assertGreater(float(gradient.abs().sum()), 0, name)
        result = model.predict(data, topk=1)
        self.assertTrue(torch.all(result['usage'] <= data['capacity'] + 1e-6))
        self.assertTrue(result['regional_capacity_is_proxy'])
        self.assertEqual(details['supervised'], 12)

    def test_empty_edges_pools_and_skipped_fixed_nodes(self):
        data = graph(2, 2)
        for name in ('logic_src', 'logic_dst', 'region_src', 'region_dst',
                     'pool_ids', 'pool_regions'):
            data[name] = torch.empty(0, dtype=torch.long)
        data['logic_weight'] = data['logic_criticality'] = torch.empty(0)
        data['region_weight'] = torch.empty(0)
        data['pool_x'] = torch.empty(0, 8)
        data['movable'][0] = False
        data['fixed_xy'] = torch.zeros(12, 2)
        data['fixed_xy'][0] = torch.tensor([.1, .2])
        result = HierarchicalInitializer(8, 13, hidden=16).predict(data)
        self.assertEqual(int(result['assignment'][0]), -1)
        self.assertTrue(torch.equal(result['xy'][0], data['fixed_xy'][0]))
        data['fixed_region'] = torch.full((12,), -1)
        data['fixed_region'][0] = 0
        result = HierarchicalInitializer(8, 13, hidden=16).predict(data)
        self.assertEqual(int(result['assignment'][0]), 0)
        self.assertTrue(torch.equal(result['xy'][0], data['fixed_xy'][0]))

    def test_supervised_generator_can_learn_small_teacher(self):
        data = graph(2, 2)
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=4)
        target = torch.arange(12) % 4
        bounds = data['region_bounds'][target]
        xy = (bounds[:, [0, 2]] + bounds[:, [1, 3]]) / 2
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        initial = float(model.loss(data, target, xy).detach())
        for _ in range(40):
            optimizer.zero_grad()
            loss = model.loss(data, target, xy)
            loss.backward()
            optimizer.step()
        final = float(model.loss(data, target, xy).detach())
        self.assertLess(final, initial * .5)

    def test_training_only_sampling_is_seeded_and_stays_inside_selected_leaf(self):
        data = graph(5, 3)
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        options = dict(score_noise_std=3., coordinate_noise_std=.15)
        first = model.predict(data, seed=15, **options)
        repeat = model.predict(data, seed=15, **options)
        other = model.predict(data, seed=16, **options)
        self.assertTrue(torch.equal(first['assignment'], repeat['assignment']))
        self.assertTrue(torch.equal(first['xy'], repeat['xy']))
        self.assertFalse(torch.equal(first['assignment'], other['assignment']))
        self.assertFalse(torch.equal(first['xy'], other['xy']))
        bounds = data['region_bounds'][first['assignment']]
        self.assertTrue(torch.all(first['xy'] >= bounds[:, [0, 2]]))
        self.assertTrue(torch.all(first['xy'] < bounds[:, [1, 3]]))
        cached = model.predict(data, seed=15,
            logic_embedding=model.encode_logic(data), device_embedding=model.encode_device(data),
            **options)
        self.assertTrue(torch.equal(first['assignment'], cached['assignment']))
        self.assertTrue(torch.equal(first['xy'], cached['xy']))
        clipped = model.predict(data, coordinate_noise_std=100., seed=15)
        bounds = data['region_bounds'][clipped['assignment']]
        self.assertTrue(torch.all(clipped['xy'] >= bounds[:, [0, 2]]))
        self.assertTrue(torch.all(clipped['xy'] < bounds[:, [1, 3]]))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_forward_loss_decode_keep_device_contract(self):
        device = torch.device('cuda:0')
        data = to_device(graph(), device)
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3).to(device)
        target = torch.arange(12, device=device) % 15
        bounds = data['region_bounds'][target]
        xy = (bounds[:, [0, 2]] + bounds[:, [1, 3]]) / 2
        model.loss(data, target, xy).backward()
        result = model.predict(data)
        self.assertEqual(result['xy'].device, device)
        self.assertEqual(result['assignment'].device.type, 'cpu')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
