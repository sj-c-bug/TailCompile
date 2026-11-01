"""008 behavioral checks of risk forward parity, constraints and gradients."""
import copy
import json
import unittest
from unittest.mock import patch

import torch
from aligned_model import AlignedLayerInitializer, generate, validate_action
from aligned_teacher import project_teacher, TeacherAlignmentError
from model import capacity_only
from test_model import fixture
from test_vertical_risk import summary
from train import state_sha256
from vertical_risk import hard_risk, risk_loss


def graph():
    data = fixture()
    data['vertical_risk_device'] = summary(capacities=(2.,))
    data['risk_nets'] = [dict(id=0, pins=[0, 1], drivers=[0], sinks=[1], ignored=False)]
    return data


def target(data):
    result = capacity_only(data)
    raw = dict(teacher_region=result['assignment'].tolist(), teacher_xy=result['xy'].tolist())
    return project_teacher(data, raw)[0]


class AlignedModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def model(self, frozen=True):
        torch.manual_seed(2801)
        model = AlignedLayerInitializer(3, 5, hidden=16, chunk_size=2)
        with torch.no_grad(): model.layer_query.weight.zero_()
        return model.freeze_shared_xy() if frozen else model

    def test_raw_or_tampered_teacher_cannot_enter_training(self):
        data = graph(); model = self.model()
        with self.assertRaises(TeacherAlignmentError):
            model.supervision_loss(data, {'teacher_region':[0]*6, 'teacher_xy':[[.25,.5]]*6})
        bad = target(data)
        bad['teacher_region'] = [0]*6; bad['teacher_layer'] = [0]*6
        with self.assertRaises(TeacherAlignmentError): model.supervision_loss(data, bad)

    def test_training_risk_is_the_exact_deployed_hard_action(self):
        data = graph(); model = self.model(); teacher = target(data)
        with patch('vertical_risk.risk_loss', wraps=risk_loss) as observed:
            loss, detail = model.supervision_loss(data, teacher, tail_weight=.1)
        self.assertEqual(observed.call_count, 1)
        action = generate(model, data, diagnostics=True)
        self.assertEqual(detail['deployed_action_sha256'], action['deployed_action_sha256'])
        self.assertEqual(detail['vertical_tail'], action['hard_vertical_risk']['cvar95'])
        self.assertEqual(detail['vertical_risk']['pool_demand'], action['hard_vertical_risk']['pool_demand'])
        probability = observed.call_args.args[1]
        self.assertTrue(torch.equal(probability.detach(), torch.nn.functional.one_hot(action['layer'],2).float()))
        self.assertIn('straight-through', detail['risk_gradient_semantics'])
        json.dumps(detail, allow_nan=False)
        loss.backward()
        self.assertTrue(torch.isfinite(model.layer_query.weight.grad).all())
        self.assertGreater(float(model.layer_query.weight.grad.abs().sum()), 0.)
        self.assertIsNone(model.shared_xy.logic_input.weight.grad)

    def test_tail_gradient_is_a_nonzero_biased_surrogate_with_hard_forward(self):
        data = graph(); model = self.model(); teacher = target(data)
        base, _ = model.supervision_loss(data, teacher, tail_weight=0.)
        added, details = model.supervision_loss(data, teacher, tail_weight=.1)
        self.assertGreater(details['vertical_tail'], 0.)
        self.assertAlmostEqual(float((added-base).detach()), .1*details['vertical_tail']/model.vertical_tail_scale, places=6)
        gradient = torch.autograd.grad(added-base, model.layer_query.weight)[0]
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()),0.)

    def test_noisy_or_clamp_gradient_limit_is_explicit(self):
        data = dict(vertical_risk_device=summary(capacities=(2.,)),
            risk_nets=[dict(id=0,pins=[0,1],drivers=[0],sinks=[1],ignored=False)])
        soft=torch.tensor([[.6,.4],[.4,.6]],requires_grad=True)
        hard=torch.tensor([[1.,0.],[0.,1.]])
        st=hard+(soft-soft.detach())
        xy=torch.tensor([[[.25,.5]]*2]*2)
        loss,_=risk_loss(data,st,xy)
        gradient=torch.autograd.grad(loss,soft)[0]
        self.assertGreater(float(gradient[0,0]),0.)
        # At exactly-one sink occupancy the existing noisy-OR clamp masks this
        # sink derivative. The surrogate must not be advertised as exact.
        self.assertEqual(float(gradient[1,1]),0.)

    def test_deployment_has_no_risk_analysis_by_default(self):
        data = graph(); model = self.model()
        with patch('vertical_risk.hard_risk', side_effect=AssertionError('diagnostic cost leaked into serving')):
            result = generate(model, data)
        self.assertFalse(result['risk_diagnostics_computed'])
        self.assertNotIn('hard_vertical_risk', result)
        with self.assertRaises(ValueError): generate(model, data, density_passes=0)

    def test_actual_capacity_and_fixed_constraints_are_preserved(self):
        data = graph(); data['fixed_layer'][0]=1; data['fixed_xy'][0,0]=.2; data['movable'][0]=False
        model = self.model()
        for method in ('D','L','T'):
            action = generate(model,data,method=method)
            self.assertEqual(int(action['layer'][0]),1)
            self.assertAlmostEqual(float(action['xy'][0,0]),.2,places=6)
            self.assertTrue(validate_action(data,action)['valid'])
        action['layer'][0]=0
        with self.assertRaises(ValueError): validate_action(data,action)

    def test_action_indices_cannot_be_fractionally_truncated(self):
        data=graph(); action=generate(self.model(),data)
        for key in ('assignment','layer'):
            bad=copy.deepcopy(action); bad[key]=bad[key].float(); bad[key][0]+=.25
            with self.assertRaisesRegex(ValueError,'integers'): validate_action(data,bad)
        bad=copy.deepcopy(action); bad['assignment']=bad['assignment'].float(); bad['assignment'][0]=float('nan')
        with self.assertRaisesRegex(ValueError,'finite integers'): validate_action(data,bad)

    def test_action_inside_leaf_but_outside_blend_reach_is_rejected(self):
        data=graph(); action=generate(self.model(),data)
        action['xy'][0]=data['region_bounds'][action['assignment'][0]][[0,2]]
        with self.assertRaisesRegex(ValueError,'blend-reachable'): validate_action(data,action)

    def test_rare_resource_ce_has_its_own_weight_and_capacity_is_not_averaged(self):
        data=graph(); data['demand'][0,2]=1.
        model=self.model(); teacher=target(data)
        _,detail=model.supervision_loss(data,teacher)
        self.assertIsNotNone(detail['resource_ce'][2])
        self.assertAlmostEqual(detail['expected_capacity_overflow'],sum(detail['expected_capacity_per_resource']),places=6)

    def test_shared_training_still_trains_xy_against_aligned_targets(self):
        data=graph(); model=self.model(frozen=False)
        before=state_sha256(model.shared_xy)
        loss,_=model.supervision_loss(data,target(data),shared=True)
        loss.backward()
        self.assertIsNotNone(model.shared_xy.coordinate_head[0].weight.grad)
        opt=torch.optim.Adam(model.parameters(),lr=.01); opt.step()
        self.assertNotEqual(before,state_sha256(model.shared_xy))


if __name__=='__main__': unittest.main()
