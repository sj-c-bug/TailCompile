"""Meaningful CPU tests of 3D constraints and actual generator gradients."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from model import (CapacityError, Hierarchical3DInitializer, allocate_layers,
                   capacity_only, decode)
from generation import lower_initialization
from supervision import native_layer, project_teacher, select_checkpoint
from train import feedback_loss, outcome_reward, train


def fixture(n=6):
    torch.manual_seed(2)
    demand = torch.zeros(n, 5); demand[:, 0] = 1
    capacity = torch.tensor([[2., 2., 1., 1., 2.]] * 4)
    bounds = torch.tensor([[0., .5, 0., 1.], [.5, 1., 0., 1.]] * 2)
    return dict(cluster_x=torch.randn(n, 3), region_x=torch.randn(4, 5),
        logic_src=torch.arange(n - 1), logic_dst=torch.arange(1, n),
        logic_weight=torch.ones(n - 1), logic_criticality=torch.zeros(n - 1),
        region_src=torch.tensor([0, 1, 0, 2, 2, 3]),
        region_dst=torch.tensor([1, 0, 2, 0, 3, 2]), region_weight=torch.ones(6),
        region_bounds=bounds, region_layer=torch.tensor([0, 0, 1, 1]),
        pool_x=torch.ones(2, 8), pool_ids=torch.tensor([0, 0, 1, 1]),
        pool_regions=torch.tensor([0, 2, 1, 3]),
        parents=[torch.tensor([0, 0, 1, 1])],
        demand=demand, capacity=capacity, movable=torch.ones(n, dtype=torch.bool),
        fixed_region=torch.full((n,), -1), fixed_layer=torch.full((n,), -1),
        fixed_xy=torch.full((n, 2), -1.), channel_width=64.,
        teacher_axis_scale=[16., 16.], density_spread_fraction=.15)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_hard_layer_limit_even_when_logits_collapse(self):
        data = fixture()
        scores = torch.tensor([[100., 100., -100., -100.]] * 6)
        result = decode(data, lambda nodes: scores[nodes])
        self.assertEqual(int((result['layer'] == 0).sum()), 4)
        self.assertTrue((result['layer_usage'][:, 0] <= 4).all())

    def test_fixed_partial_axes_and_hard_compatibility(self):
        data = fixture()
        data['movable'][0] = False
        data['fixed_xy'][0] = torch.tensor([.2, -1.])
        data['fixed_layer'][0] = 1
        data['legal_layers'] = torch.tensor([[False, True]] + [[True, True]] * 5)
        result = capacity_only(data)
        self.assertEqual(int(result['layer'][0]), 1)
        self.assertAlmostEqual(float(result['xy'][0, 0]), .2, places=6)
        self.assertGreaterEqual(float(result['xy'][0, 1]), 0)

    def test_capacity_failure_is_explicit(self):
        data = fixture(n=9)
        with self.assertRaises(CapacityError):
            capacity_only(data)

    def test_single_molecule_larger_than_every_leaf_rejected(self):
        data = fixture(); data['demand'][0, 0] = 3
        with self.assertRaises(CapacityError):
            capacity_only(data)

    def test_capacity_baseline_spreads_points_deterministically(self):
        first = capacity_only(fixture())
        second = capacity_only(fixture())
        self.assertTrue(torch.equal(first['assignment'], second['assignment']))
        self.assertTrue(torch.equal(first['xy'], second['xy']))
        self.assertGreater(len(torch.unique(first['xy'], dim=0)), 1)

    def test_supervised_forward_backward_and_loss_declines(self):
        data = fixture()
        model = Hierarchical3DInitializer(3, 5, hidden=16)
        baseline = capacity_only(data)
        target = dict(teacher_region=baseline['assignment'].tolist(), teacher_xy=baseline['xy'].tolist())
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        initial = float(model.supervision_loss(data, target).detach())
        for _ in range(15):
            optimizer.zero_grad(); loss = model.supervision_loss(data, target)
            loss.backward(); optimizer.step()
        final = float(model.supervision_loss(data, target).detach())
        self.assertLess(final, initial)
        self.assertIsNotNone(model.layer_geometry[0].weight.grad)
        self.assertIsNotNone(model.coordinate_head[-1].weight.grad)
        result = model.predict(data)
        self.assertTrue(torch.isfinite(result['xy']).all())
        self.assertTrue((result['layer_usage'][:, 0] <= 4).all())

    def test_noise_repeats_across_density_passes(self):
        data = fixture(); model = Hierarchical3DInitializer(3, 5, hidden=16)
        a = model.predict(data, seed=123, score_noise_std=.2, coordinate_noise_std=.05)
        b = model.predict(data, seed=123, score_noise_std=.2, coordinate_noise_std=.05)
        self.assertTrue(torch.equal(a['assignment'], b['assignment']))
        self.assertTrue(torch.equal(a['xy'], b['xy']))

    def test_true_signed_feedback_changes_generator_preference(self):
        data = fixture(); model = Hierarchical3DInitializer(3, 5, hidden=16)
        base = capacity_only(data)
        action = dict(assignment=base['assignment'].tolist(), xy=base['xy'].tolist())
        other = copy.deepcopy(action)
        other['assignment'] = [(region + 2) % 4 for region in other['assignment']]
        a = model.action_log_probability(data, action)
        b = model.action_log_probability(data, other)
        initial = float((a - b).detach())
        optimizer = torch.optim.Adam(model.parameters(), lr=.005)
        feedback = [dict(action=action, advantage=.7), dict(action=other, advantage=-.6)]
        for _ in range(5):
            optimizer.zero_grad(); loss, detail = feedback_loss(model, data, feedback)
            loss.backward(); optimizer.step()
        difference = float((model.action_log_probability(data, action) -
                            model.action_log_probability(data, other)).detach())
        self.assertGreater(difference, initial)
        self.assertEqual(detail['negative_actions'], 1)
        self.assertEqual(detail['positive_actions'], 1)

    def test_projection_uses_half_up_layer_and_preserves_raw(self):
        data = fixture(n=2)
        data['capacity'][:, 3] = torch.tensor([0., 0., 1., 1.])
        data['demand'][0] = torch.tensor([0., 0., 0., 1., 0.])
        target = dict(teacher_xy=[[.1, .3], [.6, .3]], teacher_raw_z=[.49, .5])
        original = copy.deepcopy(target)
        derived, audit = project_teacher(data, target)
        self.assertEqual(target, original)
        self.assertEqual(audit['mapped_layer'], [0, 1])
        self.assertEqual(derived['teacher_layer'][0], 1)
        self.assertEqual(audit['changed_layer_count'], 1)

    def test_native_z_mapping(self):
        self.assertEqual(native_layer(.5, 2), 1)
        self.assertEqual(native_layer(.4999999, 2), 0)
        for value in (-.01, 1.6, float('nan')):
            with self.assertRaises(ValueError): native_layer(value, 2)

    def test_feedback_infrastructure_not_negative_and_timeout_no_fake_time(self):
        baseline = [dict(seed=521, status='success', total_seconds=10., cpd_ns=1., wirelength=10.)]
        timeout = outcome_reward([dict(seed=521, status='timeout')], baseline)
        self.assertTrue(timeout['eligible']); self.assertEqual(timeout['advantage'], -1.)
        self.assertFalse(timeout['failure_runtime_imputed'])
        infrastructure = outcome_reward([dict(seed=521, status='infrastructure_failure')], baseline)
        self.assertFalse(infrastructure['eligible'])
        slower = outcome_reward([dict(seed=521, status='success', total_seconds=12., cpd_ns=1., wirelength=10.)], baseline)
        self.assertLess(slower['advantage'], 0)

    def test_checkpoint_selection_requires_real_vpr_evidence(self):
        report = dict(validation_signature='same', evidence='real_vpr_validation',
                      success_count=2, default_success_count=2, cpd_ratio=1., wirelength_ratio=1.,
                      total_runtime_ratio=.94, epoch=2, checkpoint='epoch2.pt')
        result = select_checkpoint([report])
        self.assertEqual(result['checkpoint'], 'epoch2.pt')
        with self.assertRaises(ValueError):
            select_checkpoint([dict(report, evidence='teacher_loss')])
        rejected = select_checkpoint([dict(report, strata_success_protection_passed=False)])
        self.assertIsNone(rejected['checkpoint'])
        rejected = select_checkpoint([dict(report, cpd_ratio=float('nan'))])
        self.assertIsNone(rejected['checkpoint'])

    def test_lowering_molecule_atoms_keep_xyz_and_fixed_axis(self):
        data = fixture(n=2)
        prediction = capacity_only(data)
        logical = dict(clusters=[dict(atoms=['a', 'b']), dict(atoms=['c'])])
        with tempfile.TemporaryDirectory() as folder:
            result = lower_initialization(logical, dict(grid_width=16, grid_height=16, layer_count=2),
                                          prediction, folder)
            rows = [line.split() for line in Path(result['ap_init']).read_text().splitlines() if not line.startswith('#')]
            self.assertEqual(rows[0][1:], rows[1][1:])
            self.assertEqual(result['atom_count'], 3)
            self.assertEqual(result['target_sha256'], hashlib.sha256(Path(result['target']).read_bytes()).hexdigest())

    def test_training_emits_candidates_not_teacher_loss_best(self):
        data = fixture(n=3); initial = capacity_only(data)
        record = dict(graph=data, teacher=dict(teacher_region=initial['assignment'].tolist(),
                      teacher_xy=initial['xy'].tolist()), split='train', condition_id='c1', family='family1')
        with tempfile.TemporaryDirectory() as folder:
            report = train([record], folder, epochs=3, hidden=16)
            self.assertEqual(len(report['checkpoints']), 3)
            self.assertIsNone(report['selected_checkpoint'])
            self.assertIn('real_vpr', report['checkpoint_selection'])
        with self.assertRaises(ValueError):
            train([dict(record, split='validation')], 'unused', epochs=1)


if __name__ == '__main__':
    unittest.main()
