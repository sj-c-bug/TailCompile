"""007 behavior tests: explicit marginals, hard proxy bounds and frozen XY."""
import copy
import json
import unittest
import torch

from layer_model import ExplicitLayerInitializer, generate, deterministic_layers
from model import CapacityError, capacity_only
from test_model import fixture
from train import state_sha256


class ExplicitLayerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self):
        torch.manual_seed(2801)
        return ExplicitLayerInitializer(3, 5, hidden=16, chunk_size=2)

    def test_layer_probability_independent_of_shared_leaf_scores(self):
        data = fixture(); model = self.model()
        before = model.layer_distribution(data)[1].detach().clone()
        with torch.no_grad():
            model.shared_xy.query.weight.mul_(100.)
            model.shared_xy.key.weight.fill_(100.)
        self.assertTrue(torch.equal(before, model.layer_distribution(data)[1]))
        self.assertTrue(torch.allclose(before.sum(1), torch.ones(6)))

    def test_fixed_and_resource_capacity_constraints_apply_to_layer_probability(self):
        data = fixture(); model = self.model().freeze_shared_xy()
        data['fixed_layer'][0] = 1; data['fixed_xy'][0, 0] = .2; data['movable'][0] = False
        prob = model.layer_distribution(data)[1]
        self.assertEqual(float(prob[0, 0]), 0.)
        self.assertEqual(float(prob[0, 1]), 1.)
        for method in ('D', 'L', 'T'):
            result = generate(model, data, method=method)
            self.assertEqual(int(result['layer'][0]), 1)
            self.assertAlmostEqual(float(result['xy'][0, 0]), .2, places=6)
            self.assertTrue((result['layer_usage'] <= data['capacity'].reshape(2, 2, 5).sum(1)+1e-6).all())

    def test_shared_encoder_and_coordinates_stay_identical_after_layer_updates(self):
        data = fixture(); model = self.model().freeze_shared_xy()
        baseline = capacity_only(data)
        teacher = dict(teacher_region=baseline['assignment'].tolist(), teacher_xy=baseline['xy'].tolist())
        before_hash = state_sha256(model.shared_xy)
        before_conditional = model.conditional_positions(data).detach().clone()
        layer_before = state_sha256(model.layer_network)
        optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=.01)
        for _ in range(3):
            optimizer.zero_grad(); loss, _ = model.supervision_loss(data, teacher)
            loss.backward(); optimizer.step()
        self.assertEqual(before_hash, state_sha256(model.shared_xy))
        self.assertTrue(torch.equal(before_conditional, model.conditional_positions(data)))
        self.assertNotEqual(layer_before, state_sha256(model.layer_network))
        self.assertIsNone(model.shared_xy.logic_input.weight.grad)

    def test_conditioned_leaf_xy_equal_for_equal_layer_assignment(self):
        data = fixture(); data['legal_layers'] = torch.tensor([[True, False]]*3+[[False, True]]*3)
        model = self.model().freeze_shared_xy()
        a, b = generate(model, data, method='D'), generate(model, data, method='L')
        self.assertTrue(torch.equal(a['layer'], b['layer']))
        self.assertTrue(torch.equal(a['assignment'], b['assignment']))
        self.assertTrue(torch.equal(a['xy'], b['xy']))

    def test_action_likelihood_uses_exact_explicit_layer_probability(self):
        data = fixture(); model = self.model().freeze_shared_xy()
        action = generate(model, data, method='L')
        prob = model.layer_distribution(data)[1]
        expected = prob.gather(1, action['layer'][:, None]).log().mean()
        observed = model.action_log_probability(data, action)
        self.assertTrue(torch.allclose(expected, observed))
        with self.assertRaises(ValueError):
            model.action_log_probability(data, dict(layer=1-action['layer'], assignment=action['assignment']))

    def test_large_demand_reports_decoder_failure_without_relaxing(self):
        data = fixture(n=9); model = self.model().freeze_shared_xy()
        with self.assertRaises(CapacityError):
            generate(model, data)

    def test_bounded_partition_does_not_move_fixed_layer(self):
        data = fixture(); data['fixed_layer'][0] = 1
        model = self.model(); allowed = model.layer_distribution(data)[2]
        assignment, usage = deterministic_layers(data, allowed, passes=2)
        self.assertEqual(int(assignment[0]), 1)
        with self.assertRaises(ValueError): deterministic_layers(data, allowed, passes=3)

    def test_real_vertical_risk_flows_through_layer_head_and_serializes(self):
        from test_vertical_risk import summary
        data = fixture(); data['vertical_risk_device'] = summary()
        data['risk_nets'] = [dict(id=0, pins=[0, 1, 2], drivers=[0], sinks=[1, 2], ignored=False)]
        model = self.model().freeze_shared_xy()
        base = capacity_only(data)
        teacher = dict(teacher_region=base['assignment'].tolist(), teacher_xy=base['xy'].tolist())
        loss, detail = model.supervision_loss(data, teacher, tail_weight=.1)
        loss.backward()
        self.assertTrue(torch.isfinite(model.layer_query.weight.grad).all())
        self.assertGreater(detail['vertical_tail'], 0.)
        self.assertIsNone(model.shared_xy.logic_input.weight.grad)
        json.dumps(detail, allow_nan=False)
        for method in ('D', 'L', 'T'):
            generated = generate(model, data, method=method)
            self.assertIn('hard_vertical_risk', generated)
            json.dumps(generated['soft_vertical_risk'], allow_nan=False)
            json.dumps(generated['hard_vertical_risk'], allow_nan=False)


if __name__ == '__main__':
    unittest.main()
