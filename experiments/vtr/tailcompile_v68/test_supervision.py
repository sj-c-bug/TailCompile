"""Resource-aware teacher projection tests; no VPR or GPU is required."""
import copy
import math
import os
from pathlib import Path
import sys
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import unittest
import torch

from model import CapacityError, HierarchicalInitializer, to_device
from supervision import project_teacher
from test_model import graph

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tailcompile_v4_risk import region_at


RESOURCE_KEYS = ('lut', 'ff', 'memory', 'dsp_top', 'io')


def target_for(data):
    n, regions = len(data['demand']), len(data['capacity'])
    assignment = torch.arange(n) % regions
    bounds = data['region_bounds'][assignment]
    xy = (bounds[:, [0, 2]] + bounds[:, [1, 3]]) / 2
    return dict(teacher_region=assignment.tolist(), teacher_xy=xy.tolist(),
                movable=data['movable'].tolist(), teacher_sha256='raw-teacher-hash')


def dsp_conflict():
    """Same resource conflict as BNN: DSP demand 1, teacher leaf DSP capacity 0."""
    data = graph(2, 2)
    data['capacity'][:, 3] = 0
    data['capacity'][1, 3] = 1
    data['demand'][0] = torch.tensor([0., 0., 0., 1., 0.])
    target = target_for(data)
    target['teacher_region'][0] = 0
    target['teacher_xy'][0] = [.2, .2]
    return data, target


class SupervisionTest(unittest.TestCase):
    def assert_graph_equal(self, first, second):
        self.assertEqual(first.keys(), second.keys())
        for name, value in first.items():
            if torch.is_tensor(value):
                self.assertTrue(torch.equal(value, second[name]), name)
            elif name == 'parents':
                self.assertEqual(len(value), len(second[name]))
                for a, b in zip(value, second[name]):
                    self.assertTrue(torch.equal(a, b))
            else:
                self.assertEqual(value, second[name], name)

    def assert_target_feasible(self, data, target):
        selected = torch.tensor(target['teacher_region'], dtype=torch.long)
        mask = torch.tensor(target['movable'], dtype=torch.bool) & (selected >= 0)
        self.assertTrue(torch.all(data['demand'][mask] <=
                                  data['capacity'][selected[mask]] + 1e-6))
        bounds = data['region_bounds'][selected[mask]]
        xy = torch.tensor(target['teacher_xy'], dtype=bounds.dtype)[mask]
        self.assertTrue(torch.isfinite(xy).all())
        self.assertTrue(torch.all(xy >= bounds[:, [0, 2]]))
        self.assertTrue(torch.all(xy < bounds[:, [1, 3]]))

    def test_bnn_dsp_teacher_moves_to_nearest_feasible_leaf_without_mutation(self):
        data, target = dsp_conflict()
        original_data, original_target = copy.deepcopy(data), copy.deepcopy(target)
        projected, audit = project_teacher(data, target)
        self.assertEqual(projected['teacher_region'][0], 1)
        self.assertGreater(projected['teacher_xy'][0][0], .5)
        self.assertLess(projected['teacher_xy'][0][0], .500001)
        self.assertAlmostEqual(projected['teacher_xy'][0][1], .2)
        self.assert_target_feasible(data, projected)
        self.assertEqual(target, original_target)
        self.assert_graph_equal(data, original_data)
        self.assertIsNot(projected, target)
        self.assertEqual(projected['teacher_region'][1:], target['teacher_region'][1:])
        self.assertEqual(projected['teacher_xy'][1:], target['teacher_xy'][1:])
        self.assertEqual(audit['schema'], 'tailcompile-v68-resource-aware-teacher-1')
        self.assertEqual(audit['movable_nodes'], 12)
        self.assertEqual(audit['conflicted_nodes'], 1)
        self.assertEqual(audit['reassigned_nodes'], 1)
        self.assertEqual(audit['conflict_resources']['dsp_top'], 1)
        self.assertEqual(set(audit['conflict_resources']), set(RESOURCE_KEYS))
        self.assertTrue(math.isfinite(audit['max_displacement']))
        self.assertTrue(math.isfinite(audit['mean_displacement']))
        self.assertGreater(audit['max_displacement'], 0)

    def test_valid_teacher_is_identical_and_projection_is_idempotent(self):
        data = graph(5, 3)
        target = target_for(data)
        first, audit = project_teacher(data, target)
        second, repeat = project_teacher(data, first)
        self.assertEqual(first['teacher_region'], target['teacher_region'])
        self.assertEqual(first['teacher_xy'], target['teacher_xy'])
        self.assertEqual(second['teacher_region'], first['teacher_region'])
        self.assertEqual(second['teacher_xy'], first['teacher_xy'])
        self.assertEqual(audit['conflicted_nodes'], 0)
        self.assertEqual(repeat['reassigned_nodes'], 0)

    def test_half_open_upper_edge_is_clipped_without_region_change(self):
        data = graph(2, 2)
        target = target_for(data)
        target['teacher_region'][0] = 0
        target['teacher_xy'][0] = [.5, .5]
        projected, audit = project_teacher(data, target)
        self.assertEqual(projected['teacher_region'][0], 0)
        actual = torch.tensor(projected['teacher_xy'][0], dtype=torch.float32)
        self.assertTrue(torch.all(actual >= 0))
        self.assertTrue(torch.all(actual < .5))
        self.assertEqual(audit['conflicted_nodes'], 0)
        self.assertEqual(audit['boundary_adjusted_nodes'], 1)
        self.assert_target_feasible(data, projected)

    def test_fixed_nodes_are_not_projected_even_if_region_has_no_resource(self):
        data, target = dsp_conflict()
        data['movable'][0] = False
        target['movable'][0] = False
        projected, audit = project_teacher(data, target)
        self.assertEqual(projected['teacher_region'][0], target['teacher_region'][0])
        self.assertEqual(projected['teacher_xy'][0], target['teacher_xy'][0])
        self.assertEqual(audit['movable_nodes'], 11)
        self.assertEqual(audit['conflicted_nodes'], 0)

    def test_no_individually_feasible_leaf_is_not_silently_relaxed_or_dropped(self):
        data, target = dsp_conflict()
        data['capacity'][:, 3] = 0
        original = copy.deepcopy(target)
        with self.assertRaisesRegex(CapacityError, r'node\s*0'):
            project_teacher(data, target)
        self.assertEqual(target, original)

    def test_zero_demand_can_remain_in_empty_resource_leaf(self):
        data = graph(2, 2)
        data['capacity'][0] = 0
        data['demand'][0] = 0
        # Other nodes have nonzero LUT/FF demand and must be projected away.
        projected, audit = project_teacher(data, target_for(data))
        self.assertEqual(projected['teacher_region'][0], 0)
        self.assertGreater(audit['reassigned_nodes'], 0)
        self.assert_target_feasible(data, projected)

    def test_chunk_size_does_not_change_projection_or_audit(self):
        data, target = dsp_conflict()
        first, first_audit = project_teacher(data, target, chunk_size=1)
        for size in (2, 5, 512):
            projected, audit = project_teacher(data, target, chunk_size=size)
            self.assertEqual(projected, first)
            self.assertEqual(audit, first_audit)

    def test_movable_masks_must_agree_instead_of_silently_changing_supervision(self):
        data, target = dsp_conflict()
        target['movable'][0] = False
        with self.assertRaisesRegex(ValueError, r'movable|mask'):
            project_teacher(data, target)

    def test_nearest_bbox_distance_precedes_center_distance(self):
        data = dict(demand=torch.tensor([[0., 0., 0., 1., 0.]]),
                    capacity=torch.tensor([[0., 0., 0., 0., 0.],
                                           [0., 0., 0., 1., 0.],
                                           [0., 0., 0., 1., 0.]]),
                    region_bounds=torch.tensor([[.25, .75, .25, .75],
                                                [0., .25, .25, .75],
                                                [.8125, .875, .25, .75]]),
                    movable=torch.tensor([True]))
        target = dict(teacher_region=[0], teacher_xy=[[.5, .5]], movable=[True])
        projected, _ = project_teacher(data, target)
        self.assertEqual(projected['teacher_region'], [1])

    def test_rectangular_device_uses_physical_axis_scale_when_provided(self):
        data = dict(demand=torch.tensor([[0., 0., 0., 1., 0.]]),
                    capacity=torch.tensor([[0., 0., 0., 0., 0.],
                                           [0., 0., 0., 1., 0.],
                                           [0., 0., 0., 1., 0.]]),
                    region_bounds=torch.tensor([[.25, .75, .25, .625],
                                                [0., .25, .25, .75],
                                                [.25, .75, .625, .875]]),
                    movable=torch.tensor([True]))
        target = dict(teacher_region=[0], teacher_xy=[[.5, .5]], movable=[True])
        normalized, _ = project_teacher(data, target)
        self.assertEqual(normalized['teacher_region'], [2])
        data['teacher_axis_scale'] = [1., 4.]
        physical, _ = project_teacher(data, target)
        self.assertEqual(physical['teacher_region'], [1])

    def test_projected_coordinate_round_trip_owns_new_leaf_on_84_tile4_device(self):
        size, tile_size = 84, 4
        regions = []
        for row, yl in enumerate(range(1, size - 1, tile_size)):
            for col, xl in enumerate(range(1, size - 1, tile_size)):
                regions.append(dict(id=len(regions), row=row, column=col,
                    x_low=xl, x_high=min(xl + tile_size - 1, size - 2),
                    y_low=yl, y_high=min(yl + tile_size - 1, size - 2)))
        device = dict(grid_width=size, grid_height=size, tile_size=tile_size,
                      region_grid=dict(columns=21, rows=21), regions=regions)
        bounds = torch.tensor([[
            (0 if r['x_low'] == 1 else r['x_low'] - .5) / size,
            (size if r['x_high'] == size - 2 else r['x_high'] + .5) / size,
            (0 if r['y_low'] == 1 else r['y_low'] - .5) / size,
            (size if r['y_high'] == size - 2 else r['y_high'] + .5) / size,
        ] for r in regions], dtype=torch.float32)
        capacity = torch.zeros(len(regions), 5)
        capacity[1, 3] = 1
        data = dict(demand=torch.tensor([[0., 0., 0., 1., 0.]]),
                    capacity=capacity, region_bounds=bounds,
                    movable=torch.tensor([True]), teacher_axis_scale=[size, size])
        target = dict(teacher_region=[0], teacher_xy=[[4. / size, 2. / size]],
                      movable=[True])
        projected, _ = project_teacher(data, target)
        self.assertEqual(projected['teacher_region'], [1])
        physical_xy = torch.tensor(projected['teacher_xy'][0]) * size
        self.assertEqual(region_at(device, *physical_xy.tolist()), 1)
        self.assert_target_feasible(data, projected)

    def test_bbox_ties_use_center_distance_then_region_index(self):
        data = dict(demand=torch.tensor([[0., 0., 0., 1., 0.]]),
                    capacity=torch.tensor([[0., 0., 0., 0., 0.],
                                           [0., 0., 0., 1., 0.],
                                           [0., 0., 0., 1., 0.]]),
                    region_bounds=torch.tensor([[.25, .75, .25, .75],
                                                [0., .25, .25, .75],
                                                [.75, .875, .25, .75]]),
                    movable=torch.tensor([True]))
        target = dict(teacher_region=[0], teacher_xy=[[.5, .5]], movable=[True])
        projected, _ = project_teacher(data, target)
        self.assertEqual(projected['teacher_region'], [2])
        data['region_bounds'][2] = torch.tensor([.75, 1., .25, .75])
        projected, _ = project_teacher(data, target)
        self.assertEqual(projected['teacher_region'], [1])

    def test_nonfinite_graph_values_and_movable_teacher_xy_fail_clearly(self):
        for field in ('demand', 'capacity', 'region_bounds'):
            for value in (float('nan'), float('inf')):
                data, target = dsp_conflict()
                data[field].view(-1)[0] = value
                with self.subTest(field=field, value=value):
                    with self.assertRaisesRegex(ValueError, r'finite|NaN|Inf'):
                        project_teacher(data, target)
        data, target = dsp_conflict()
        target['teacher_xy'][0][0] = float('nan')
        with self.assertRaisesRegex(ValueError, r'finite|NaN|Inf'):
            project_teacher(data, target)

    def test_projected_teacher_trains_and_decoder_keeps_capacity_bounds(self):
        data, target = dsp_conflict()
        projected, _ = project_teacher(data, target, chunk_size=3)
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        loss, details = model.loss(data,
            torch.tensor(projected['teacher_region']),
            torch.tensor(projected['teacher_xy']),
            mask=torch.tensor(projected['movable']), return_details=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(details['supervised'], 12)
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertGreater(sum(float(value.abs().sum()) for value in gradients), 0)
        prediction = model.predict(data, topk=1)
        self.assertTrue(torch.all(prediction['usage'] <= data['capacity'] + 1e-6))
        self.assertEqual(int(prediction['assignment'][0]), 1)
        self.assertTrue(torch.isfinite(prediction['xy']).all())

    def test_geometry_protection_can_retain_raw_teacher_wire_geometry(self):
        data, target = dsp_conflict()
        projected, _ = project_teacher(data, target)
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        regions = torch.tensor(projected['teacher_region'])
        projected_xy = torch.tensor(projected['teacher_xy'])
        raw_xy = torch.tensor(target['teacher_xy'])
        loss, details = model.loss(data, regions, projected_xy,
            geometry_teacher_xy=raw_xy, return_details=True)
        prediction = model(data, regions)['xy']
        src, dst = data['logic_src'], data['logic_dst']
        predicted_length = (prediction[src] - prediction[dst]).abs().sum(-1)
        raw_length = (raw_xy[src] - raw_xy[dst]).abs().sum(-1)
        weight = data['logic_weight'] * (1 + data['logic_criticality'])
        expected = (torch.relu(predicted_length - raw_length) * weight).sum() / weight.sum()
        self.assertAlmostEqual(details['geometry'], float(expected.detach()), places=6)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_projection_audits_aggregate_overflow_without_inflating_capacity(self):
        data, target = dsp_conflict()
        for node in (0, 4, 8):
            data['demand'][node] = torch.tensor([0., 0., 0., 1., 0.])
        original_capacity = data['capacity'].clone()
        projected, audit = project_teacher(data, target)
        self.assertEqual(audit['conflicted_nodes'], 3)
        self.assertEqual(audit['aggregate_overflow_projected']['dsp_top'], 2.)
        self.assertTrue(torch.equal(data['capacity'], original_capacity))
        self.assert_target_feasible(data, projected)  # Individual, not total packing.

    def test_generator_can_learn_resource_projected_teacher_with_raw_geometry_guard(self):
        data, target = dsp_conflict()
        projected, _ = project_teacher(data, target)
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        region = torch.tensor(projected['teacher_region'])
        xy = torch.tensor(projected['teacher_xy'])
        raw_xy = torch.tensor(target['teacher_xy'])
        initial = float(model.loss(data, region, xy, geometry_teacher_xy=raw_xy).detach())
        for _ in range(40):
            optimizer.zero_grad()
            loss = model.loss(data, region, xy, geometry_teacher_xy=raw_xy)
            loss.backward()
            optimizer.step()
        final = float(model.loss(data, region, xy, geometry_teacher_xy=raw_xy).detach())
        self.assertLess(final, initial * .7)

    def test_nonfinite_logits_are_not_misreported_as_teacher_capacity_conflict(self):
        data = graph(2, 2)
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        with self.assertRaisesRegex(FloatingPointError, 'NaN/Inf'):
            model._supervised_chunk(torch.full((1, 16), float('nan')),
                torch.zeros(4, 16), torch.zeros(1, 5), torch.tensor([0]), data['capacity'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_projection_rejects_cuda_graph_instead_of_implicitly_copying(self):
        data, target = dsp_conflict()
        data = to_device(data, torch.device('cuda:0'))
        with self.assertRaisesRegex(ValueError, r'CPU|cpu'):
            project_teacher(data, target)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
