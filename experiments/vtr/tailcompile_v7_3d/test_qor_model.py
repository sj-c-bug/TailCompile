"""009 tests exercise actual coordinate gradients and hard-action parity."""
import copy
import unittest
from unittest.mock import patch

import torch

from aligned_teacher import deployed_xy_bounds, constraints_digest
from qor_model import QorInitializer, generate, validate_collected_action, DECODER_SCHEMA
from test_aligned_model import graph,target
from train import state_sha256


def model(arm):
    torch.manual_seed(9301)
    result=QorInitializer(3,5,hidden=16,chunk_size=2,arm=arm)
    with torch.no_grad(): result.layer_query.weight.zero_()
    return result


def collected(action,data,xy_hash):
    return dict(schema='tailcompile-v7-aligned-action-008-1',decoder_kind='aligned-hard-layer-density-008-1',
        shared_xy_sha256=xy_hash,constraints_sha256=constraints_digest(data),
        assignment=action['assignment'].tolist(),layer=action['layer'].tolist(),xy=action['xy'].tolist())


class QorModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_cut_removal_is_an_isolated_objective_change(self):
        data=graph(); teacher=target(data)
        q,c=model('Q'),model('C')
        self.assertEqual(state_sha256(q),state_sha256(c))
        a,da=q.supervision_loss(data,teacher); b,db=c.supervision_loss(data,teacher)
        self.assertAlmostEqual(float((a-b).detach()),.1*da['decoded_timing_proxy_cut'],places=6)
        self.assertEqual(da['deployed_action_sha256'],db['deployed_action_sha256'])
        self.assertEqual(db['cut_weight'],0.)

    def test_tail_is_the_only_x_t_loss_difference_and_uses_deployed_hard_action(self):
        data=graph(); teacher=target(data); x,t=model('X'),model('T')
        a,dx=x.supervision_loss(data,teacher); b,dt=t.supervision_loss(data,teacher)
        action=generate(t,data,diagnostics=True)
        self.assertEqual(dt['deployed_action_sha256'],action['deployed_action_sha256'])
        self.assertEqual(dt['vertical_tail'],action['hard_vertical_risk']['cvar95'])
        self.assertEqual(dt['vertical_risk']['pool_demand'],action['hard_vertical_risk']['pool_demand'])
        self.assertAlmostEqual(float((b-a).detach()),.1*dt['vertical_tail']/t.vertical_tail_scale,places=5)
        self.assertEqual(dx['deployed_action_sha256'],dt['deployed_action_sha256'])

    def test_full_candidate_score_changes_coordinate_head_and_actual_output(self):
        data=graph(); m=model('X'); m.eval()
        before=generate(m,data)
        action=copy.deepcopy(before)
        low,high=deployed_xy_bounds(data,action['assignment'])
        action['xy']=.2*low.float()+.8*high.float()
        # Force a nonzero target residual while keeping the selected feasible leaf.
        score=m.action_score(data,action)
        score.backward()
        gradient=m.shared_xy.coordinate_head[-1].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()),0.)
        old=state_sha256(m.shared_xy)
        opt=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),lr=.01)
        opt.step()
        after=generate(m,data)
        self.assertNotEqual(old,state_sha256(m.shared_xy))
        self.assertFalse(torch.equal(before['xy'],after['xy']))

    def test_teacher_coordinate_forward_is_trainable_without_decoder_gradient(self):
        data=graph(); m=model('X')
        loss,detail=m.supervision_loss(data,target(data))
        loss.backward()
        gradient=m.shared_xy.coordinate_head[0].weight.grad
        self.assertGreater(float(gradient.abs().sum()),0.)
        self.assertTrue(detail['xy_trainable'])

    def test_valid_historical_action_survives_extreme_layer_logit_underflow(self):
        data=graph();m=model('Q');action=generate(m,data)
        # Each selected historical label is deliberately the 2000-logit loser.
        # Its ordinary float softmax is zero, while log_softmax and its actor
        # gradient must remain finite. Hard-masked labels still stay -inf.
        labels=action['layer']
        logits=torch.full((len(labels),2),1000.)
        logits.scatter_(1,labels[:,None],-1000.)
        logits.requires_grad_()
        self.assertTrue((logits.softmax(1).gather(1,labels[:,None])==0).all())
        with patch.object(m,'layer_distribution',return_value=(logits,None,None,None)):
            score=m.action_score(data,action,constant=torch.tensor(0.))
        self.assertTrue(torch.isfinite(score))
        score.backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertGreater(float(logits.grad.abs().sum()),0.)
        masked=logits.detach().clone()
        masked[0,labels[0]]=-torch.inf
        with self.assertRaisesRegex(ValueError,'incompatible'):
            m.action_score(data,action,log_probability=masked.log_softmax(1),constant=torch.tensor(0.))

    def test_frozen_arms_have_no_xy_gradients(self):
        data=graph(); teacher=target(data)
        for arm in ('R','Q','C'):
            m=model(arm); initial=state_sha256(m.shared_xy)
            loss,detail=m.supervision_loss(data,teacher); loss.backward()
            self.assertTrue(m.shared_xy_frozen)
            self.assertTrue(all(p.grad is None for p in m.shared_xy.parameters()))
            opt=torch.optim.Adam((p for p in m.parameters() if p.requires_grad),lr=.01);opt.step()
            self.assertEqual(initial,state_sha256(m.shared_xy))
            self.assertFalse(detail['xy_trainable'])

    def test_collection_identity_and_support_are_not_silently_rewritten(self):
        data=graph(); m=model('Q'); identity=state_sha256(m.shared_xy)
        action=collected(generate(m,data),data,identity)
        self.assertTrue(validate_collected_action(data,action,allowed_xy_sha256=identity)['valid'])
        foreign=copy.deepcopy(action);foreign['shared_xy_sha256']='1'*64
        self.assertTrue(validate_collected_action(data,foreign)['valid'])
        with self.assertRaisesRegex(ValueError,'outside frozen'):validate_collected_action(data,foreign,allowed_xy_sha256=identity)
        wrong=copy.deepcopy(action);wrong['constraints_sha256']='2'*64
        with self.assertRaisesRegex(ValueError,'constraints'):validate_collected_action(data,wrong)
        with self.assertRaisesRegex(ValueError,'explicitly collected'):validate_collected_action(data,dict(layer=[0]*6))

    def test_new_serving_keeps_constraints_and_omits_risk_diagnostic_cost(self):
        data=graph(); data['fixed_layer'][0]=1;data['fixed_xy'][0,0]=.2;data['movable'][0]=False
        for arm in ('Q','X','T'):
            m=model(arm)
            with patch('vertical_risk.hard_risk',side_effect=AssertionError('unrequested diagnostic')):
                action=generate(m,data,method=arm)
            self.assertEqual(action['decoder_kind'],DECODER_SCHEMA)
            self.assertFalse(action['risk_diagnostics_computed'])
            self.assertEqual(int(action['layer'][0]),1)
            self.assertAlmostEqual(float(action['xy'][0,0]),.2,places=6)
            with self.assertRaises(ValueError):generate(m,data,method='R' if arm!='R' else 'Q')


if __name__=='__main__':unittest.main()
