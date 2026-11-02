"""Behavioral 010 checks: actual coordinate gradients and safe deployment."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from aligned_model import action_digest
from aligned_teacher import constraints_digest
from qor_model import QorInitializer, generate as generate009
from train_qor import save_model as save009
from structured_model import (StructuredInitializer, generate, project_teacher,
    validate_collected_action, balanced_layer_loss, leaf_noise, ACTION_SCHEMA, DECODER_SCHEMA)
from structured_decoder import validate_leaf_capacity, StructuredCapacityError
from test_aligned_model import graph, target
from train_structured import from009, save_model, load_model
from train import state_sha256


class StructuredModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(10301)
        return StructuredInitializer(3, 5, hidden=16, chunk_size=2)

    def test_geometry_reaches_actual_xy_head_without_tail(self):
        data = graph(); model = self.model()
        with patch('vertical_risk.risk_loss', side_effect=AssertionError('Tail called')):
            geometry = model.differentiable_objectives(data)
            geometry['wire'].backward()
        grad = sum(float(p.grad.abs().sum()) for p in model.shared_xy.coordinate_head.parameters() if p.grad is not None)
        self.assertGreater(grad, 0)
        self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
        self.assertTrue(torch.equal(geometry['action']['xy'], geometry['xy'].detach().cpu()))

    def test_soft_density_gradient_penalizes_concentration(self):
        data = graph(); model = self.model()
        with torch.no_grad(): model.layer_query.weight.zero_()
        bias = torch.tensor([8., -8., 8., -8.], requires_grad=True)
        with patch.object(model.shared_xy, 'matching_logits', side_effect=lambda a,b: bias[None].expand(len(a),-1)):
            result = model.differentiable_objectives(data)
            gradient = torch.autograd.grad(result['density'], bias)[0]
        self.assertGreater(float(result['density']), 0)
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertTrue(torch.isfinite(gradient).all())

    def test_fixed_boundary_axis_has_no_soft_movement_gradient(self):
        data = graph(); model = self.model()
        # x=.5 is individually compatible with leaves on both sides. Their
        # centres differ, but this physical x is fixed and cannot be moved.
        data['fixed_xy'][0, 0] = .5
        result = model.differentiable_objectives(data)
        gradient = torch.autograd.grad(result['wire'], result['soft_region_xy'], retain_graph=True)[0]
        self.assertEqual(float(gradient[0, 0]), 0.)
        self.assertGreater(float(gradient[1:].abs().sum()), 0.)
        self.assertEqual(float(result['xy'][0, 0]), .5)

    def test_minority_teacher_layer_has_equal_group_gradient(self):
        logits = torch.zeros((10, 2), requires_grad=True)
        labels = torch.tensor([0]*9+[1]); demand = torch.ones((10, 5)); movable = torch.ones(10, dtype=torch.bool)
        loss = balanced_layer_loss(logits, labels, demand, movable)
        loss.backward()
        self.assertAlmostEqual(float(logits.grad[:9].abs().sum()), float(logits.grad[9:].abs().sum()), places=6)

    def test_fixed_axes_and_hard_leaf_constraints_hold(self):
        data = graph(); data['fixed_layer'][0] = 1; data['fixed_xy'][0,0] = .2; data['movable'][0] = False
        model = self.model(); action = generate(model, data)
        self.assertEqual(int(action['layer'][0]), 1)
        self.assertAlmostEqual(float(action['xy'][0,0]), .2, places=6)
        self.assertTrue(validate_leaf_capacity(data, action['assignment'])['valid'])
        self.assertEqual(action['capacity_diagnostics']['overflow_region_count'], 0)
        self.assertFalse(action['risk_diagnostics_computed'])

    def test_projection_is_supervision_only_and_validates_original_stamp(self):
        data = graph(); old = target(data); projected = project_teacher(data, old)
        self.assertFalse(projected['projection']['old_qor_labels_reused_for_projection'])
        self.assertTrue(validate_leaf_capacity(data, projected['teacher_region'])['valid'])
        self.assertEqual(project_teacher(data, projected), projected)
        bad = copy.deepcopy(old); bad['teacher_layer'][0] = 1-bad['teacher_layer'][0]
        with self.assertRaises(ValueError): project_teacher(data, bad)

    def test_leaf_jitter_is_invariant_to_chunk_order_and_seeded(self):
        full = leaf_noise(torch.arange(8), 4, 10301, torch.float32)
        ids = torch.tensor([6, 2, 7])
        self.assertTrue(torch.equal(full[ids], leaf_noise(ids, 4, 10301, torch.float32)))
        self.assertFalse(torch.equal(full, leaf_noise(torch.arange(8), 4, 10302, torch.float32)))
        model = self.model(); data = graph()
        a = generate(model, data, score_noise_std=2., seed=11)
        b = generate(model, data, score_noise_std=2., seed=11)
        self.assertEqual(a['deployed_action_sha256'], b['deployed_action_sha256'])

    def test_migration_keeps_initial_logits_and_checkpoint_is_isolated(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_structured_') as folder:
            folder = Path(folder); source = QorInitializer(3,5,hidden=16,chunk_size=2,arm='X').eval()
            save009(source, folder/'009.pt', epoch=10)
            model = from009(folder/'009.pt').eval(); data = graph()
            self.assertTrue(torch.equal(source.layer_distribution(data)[0], model.layer_distribution(data)[0]))
            self.assertEqual(state_sha256(source.shared_xy), state_sha256(model.shared_xy))
            self.assertEqual(state_sha256(model), state_sha256(from009(folder/'009.pt')))
            save_model(model, folder/'010.pt'); loaded = load_model(folder/'010.pt')
            self.assertEqual(state_sha256(loaded), state_sha256(model))
            with self.assertRaisesRegex(ValueError, 'migration'): load_model(folder/'009.pt')
            p = torch.load(folder/'010.pt', weights_only=False); p['state_dict']['layer_query.weight'][0,0] += 1
            torch.save(p, folder/'tampered.pt')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): load_model(folder/'tampered.pt')

    def test_collected_action_payload_cannot_change_after_routing(self):
        data = graph(); action = generate(self.model(), data)
        payload = dict(schema=ACTION_SCHEMA, decoder_kind=DECODER_SCHEMA,
            assignment=action['assignment'].tolist(), layer=action['layer'].tolist(), xy=action['xy'].tolist(),
            constraints_sha256=constraints_digest(data), collection_checkpoint_sha256='1'*64,
            deployed_action_sha256=action['deployed_action_sha256'])
        self.assertTrue(validate_collected_action(data, payload)['valid'])
        payload['xy'][0][0] += .001
        with self.assertRaisesRegex(ValueError, 'digest'): validate_collected_action(data, payload)


if __name__ == '__main__': unittest.main()
