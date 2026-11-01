"""008 target projection: same constraints and deployed XY range as decoding."""
import copy
from pathlib import Path
import unittest

import torch

from aligned_teacher import (SCHEMA, TeacherAlignmentError, TeacherProjectionFailure,
    project_teacher, verify_teacher, deployed_xy_bounds)
from test_model import fixture


def target(data, regions):
    bounds = data['region_bounds'][regions].double()
    return dict(teacher_region=list(regions), teacher_layer=data['region_layer'][regions].tolist(),
                teacher_xy=((bounds[:, [0, 2]]+bounds[:, [1, 3]])*.5).tolist())


class AlignedTeacherTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_feasible_teacher_is_unchanged_and_raw_immutable(self):
        data = fixture()
        raw = target(data, [0, 1, 0, 2, 3, 2]); before = copy.deepcopy(raw)
        new, audit = project_teacher(data, raw, chunk_size=2)
        self.assertEqual(raw, before)
        self.assertEqual(new['schema'], SCHEMA)
        self.assertEqual(new['teacher_region'], raw['teacher_region'])
        self.assertEqual(new['teacher_layer'], raw['teacher_layer'])
        self.assertEqual(new['teacher_xy'], raw['teacher_xy'])
        self.assertEqual(audit['changed_layer_count'], 0)
        self.assertEqual(audit['changed_xy_count'], 0)
        self.assertEqual(audit['projection_kind'], 'already_feasible_unchanged')
        self.assertTrue(verify_teacher(data, new)['hard_layer_capacity_satisfied'])

    def test_rare_memory_overflow_repaired_without_moving_lut_majority(self):
        data = fixture()
        data['demand'].zero_()
        data['demand'][:4, 2] = 1.
        data['demand'][4:, 0] = 1.
        raw = target(data, [0]*6)
        before = copy.deepcopy(raw); cap = data['capacity'].clone()
        new, audit = project_teacher(data, raw)
        self.assertEqual(raw, before); self.assertTrue(torch.equal(cap, data['capacity']))
        self.assertEqual(audit['raw_capacity']['layer_overflow_by_resource'][2], 2.)
        self.assertEqual(audit['changed_layer_count'], 2)
        self.assertEqual(new['teacher_layer'][4:], [0, 0])
        self.assertEqual(audit['aligned_capacity']['layer_overflow_by_resource'], [0.]*5)
        self.assertFalse(audit['physical_packing_certified'])
        self.assertFalse(audit['aggregate_leaf_capacity_certified'])

    def test_known_fixed_layer_and_axes_survive_repair(self):
        data = fixture()
        data['fixed_layer'][0] = 1
        data['fixed_xy'][0] = torch.tensor([.5, .7])
        data['movable'][0] = False
        raw = target(data, [0]*6)
        new, audit = project_teacher(data, raw)
        self.assertEqual(new['teacher_layer'][0], 1)
        self.assertAlmostEqual(new['teacher_xy'][0][0], .5, places=7)
        self.assertAlmostEqual(new['teacher_xy'][0][1], .7, places=7)
        self.assertTrue(audit['aligned_capacity']['fixed_axes_passed'])

    def test_all_fixed_memory_overflow_is_explicit_projection_failure(self):
        data = fixture(n=3)
        data['demand'].zero_(); data['demand'][:, 2] = 1.
        data['fixed_layer'][:] = 0
        raw = target(data, [0]*3)
        with self.assertRaisesRegex(TeacherProjectionFailure, 'not physical infeasibility'):
            project_teacher(data, raw)
        self.assertEqual(raw['teacher_layer'], [0, 0, 0])

    def test_moved_layer_keeps_compatible_spatial_leaf(self):
        data = fixture()
        raw = target(data, [3]*6)
        raw['teacher_xy'] = [[.8, .5] for _ in range(6)]
        new, audit = project_teacher(data, raw)
        moved = [i for i, z in enumerate(new['teacher_layer']) if z == 0]
        self.assertEqual(len(moved), 2)
        self.assertTrue(all(new['teacher_region'][i] == 1 for i in moved))
        self.assertTrue(all(abs(new['teacher_xy'][i][0]-.8) < 1e-7 for i in moved))
        self.assertEqual(audit['changed_layer_count'], 2)

    def test_one_bounded_allocator_fallback_can_repair_a_needed_swap(self):
        data = fixture(n=4)
        data['capacity'].zero_()
        data['capacity'][[0, 2], :2] = 5.
        data['demand'].zero_()
        data['demand'][:, :2] = torch.tensor([[4., 1.], [2., 1.], [1., 3.], [0., 2.]])
        # Layer0 LUT is overfull, while layer1 FF is full. No single source
        # move fits; a valid solution swaps B and D. No unbounded swap search.
        raw = target(data, [0, 0, 2, 2])
        new, audit = project_teacher(data, raw)
        self.assertEqual(audit['projection_kind'], 'single_deployment_allocator_fallback')
        self.assertEqual(audit['fallback_allocator_calls'], 1)
        self.assertEqual(new['teacher_layer'], [0, 1, 1, 0])
        self.assertEqual(audit['aligned_capacity']['layer_overflow_by_resource'], [0.]*5)

    def test_target_in_leaf_but_outside_deployed_blend_is_projected_once(self):
        data = fixture()
        data['density_spread_fraction'] = .75
        raw = target(data, [0, 1, 0, 2, 3, 2])
        raw['teacher_xy'][0] = [.00001, .00001]
        new, audit = project_teacher(data, raw)
        self.assertGreater(audit['blend_reachability_changed_count'], 0)
        regions = torch.tensor(new['teacher_region'])
        low, high = deployed_xy_bounds(data, regions)
        xy = torch.tensor(new['teacher_xy'], dtype=torch.float64)
        self.assertTrue((xy >= low-1e-6).all()); self.assertTrue((xy <= high+1e-6).all())
        self.assertTrue(torch.allclose(xy[0], low[0]))
        twice, second = project_teacher(data, new)
        self.assertEqual(twice, new)
        self.assertEqual(second['changed_xy_count'], 0)

    def test_verification_recomputes_bounds_instead_of_trusting_boolean(self):
        data = fixture()
        new, _ = project_teacher(data, target(data, [0, 1, 0, 2, 3, 2]))
        altered = copy.deepcopy(new)
        altered['teacher_region'] = [0]*6; altered['teacher_layer'] = [0]*6
        altered['hard_layer_capacity_satisfied'] = True
        with self.assertRaises(TeacherAlignmentError): verify_teacher(data, altered)

    def test_verification_rejects_layer_leaf_disagreement_and_stale_capacity(self):
        data = fixture()
        new, _ = project_teacher(data, target(data, [0, 1, 0, 2, 3, 2]))
        other = copy.deepcopy(new); other['teacher_layer'][0] = 1
        with self.assertRaisesRegex(TeacherAlignmentError, 'layer and leaf disagree'):
            verify_teacher(data, other)
        data['capacity'][0, 0] += 1
        with self.assertRaisesRegex(TeacherAlignmentError, 'different constraints'):
            verify_teacher(data, new)

    def test_noninteger_ids_are_not_silently_truncated(self):
        data = fixture(); raw = target(data, [0, 1, 0, 2, 3, 2])
        raw['teacher_region'][0] = .5
        with self.assertRaisesRegex(TeacherAlignmentError, 'finite integers'):
            project_teacher(data, raw)

    def test_expired_budget_does_not_return_undocumented_relaxation(self):
        data = fixture(); raw = target(data, [0]*6)
        with self.assertRaises(TeacherProjectionFailure): project_teacher(data, raw, max_seconds=1e-12)

    def test_audit_is_compact_and_does_not_embed_per_node_labels(self):
        data = fixture(n=100)
        data['capacity'][:, 0] = 30.
        new, audit = project_teacher(data, target(data, [0]*100))
        self.assertEqual(len(new['teacher_layer']), 100)
        self.assertLessEqual(len(audit['modified_samples']), 32)
        self.assertNotIn('teacher_layer', audit)
        self.assertNotIn('projected_layer', audit)
        self.assertEqual(len(audit['aligned_capacity']['layer_usage']), 2)


if __name__ == '__main__': unittest.main()
