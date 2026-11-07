"""015 same-action ablation, null-budget matching and truthful failure tests."""
import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from aligned_model import action_digest
from attribution_actions import make_attribution_candidates, GENERATOR
from residual_actions import make_candidates, validate_residual_action
from test_residual_actions import graph_and_base


class AttributionActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_joint_is_original_weak_and_components_project_exactly_that_action(self):
        data, base = graph_and_base(400)
        old = copy.deepcopy(base)
        original = next(a for a in make_candidates(data, base, 'J', **GENERATOR)
                        if a['candidate_id']=='j_s1')
        actions = make_attribution_candidates(data, base)
        self.assertEqual(tuple(actions), ('D', 'L', 'Y', 'J', 'R1', 'R2'))
        self.assertEqual(action_digest(actions['D']), action_digest(base))
        self.assertEqual(action_digest(actions['J']), action_digest(original))
        for field in ('assignment', 'layer'):
            self.assertTrue(torch.equal(actions['L'][field], actions['J'][field]))
            self.assertTrue(torch.equal(actions['Y'][field], base[field]))
        self.assertTrue(torch.equal(actions['L']['xy'], base['xy']))
        self.assertTrue(torch.equal(actions['Y']['xy'], actions['J']['xy']))
        for field in old:
            self.assertTrue(torch.equal(old[field], base[field]))
        for name, arm in (('L','L'),('Y','Y'),('J','J'),('R1','J'),('R2','J')):
            self.assertTrue(validate_residual_action(data, actions[name], base_action=base, arm=arm)['valid'])

    def test_random_controls_match_joint_counts_resource_mass_and_norms(self):
        data, base = graph_and_base(400)
        actions = make_attribution_candidates(data, base)
        for name in ('R1','R2'):
            action = actions[name]
            match = action['attribution_diagnostics']['matching']
            self.assertTrue(match['matched_control'])
            self.assertEqual(match['requested'], match['achieved'])
            self.assertEqual(match['requested_resource_mass'], match['achieved_resource_mass'])
            self.assertEqual(match['matched_templates'], match['requested_templates'])
            for assignment in match['template_assignments']:
                original, null = assignment['template_node'], assignment['random_node']
                old_norm = torch.linalg.norm(actions['J']['xy'][original]-base['xy'][original])
                new_norm = torch.linalg.norm(action['xy'][null]-base['xy'][null])
                self.assertAlmostEqual(float(old_norm), float(new_norm), places=6)
                self.assertTrue(torch.equal(data['demand'][original], data['demand'][null]))
                self.assertEqual(int(base['layer'][original]), int(base['layer'][null]))
        self.assertNotEqual(action_digest(actions['R1']), action_digest(actions['R2']))

    def test_random_uses_local_seed_and_is_independent_of_global_rng(self):
        data, base = graph_and_base(400)
        first = make_attribution_candidates(data, base)
        np.random.seed(999); np.random.randn(400)
        torch.manual_seed(171); torch.randn(80)
        second = make_attribution_candidates(data, base)
        self.assertEqual([action_digest(v) for v in first.values()],
                         [action_digest(v) for v in second.values()])

    def test_immovable_fixed_axis_layer_and_resource_type_stay_valid(self):
        data, base = graph_and_base(400)
        data['movable'][0] = False
        data['fixed_xy'][1, 0] = base['xy'][1, 0]
        data['fixed_layer'][2] = int(base['layer'][2])
        data['fixed_region'][3] = int(base['assignment'][3])
        # This unique DSP can only occupy leaf0, not an arbitrary random leaf.
        data['demand'][4] = torch.tensor([0.,0.,0.,1.,0.])
        data['capacity'][:, 3] = 0.; data['capacity'][0,3] = 2.
        actions = make_attribution_candidates(data, base)
        for action in actions.values():
            self.assertTrue(torch.equal(action['xy'][0], base['xy'][0]))
            self.assertEqual(int(action['assignment'][0]), int(base['assignment'][0]))
            self.assertEqual(float(action['xy'][1,0]), float(base['xy'][1,0]))
            self.assertEqual(int(action['layer'][2]), int(base['layer'][2]))
            self.assertEqual(int(action['assignment'][3]), int(base['assignment'][3]))
            self.assertEqual(int(action['assignment'][4]), 0)
            self.assertEqual(action['capacity_diagnostics']['overflow_region_count'], 0)

    def test_unmatched_null_is_valid_and_explicit_not_fake_equal_budget(self):
        data, base = graph_and_base(400)
        # Simulate a bounded geometric sampler unable to fit any XY template.
        with patch('attribution_actions._random_delta', return_value=None):
            actions = make_attribution_candidates(data, base)
        for name in ('R1','R2'):
            action = actions[name]
            audit = action['attribution_diagnostics']
            self.assertFalse(audit['matched_control'])
            self.assertTrue(audit['matching']['unmatched_template_nodes'])
            self.assertNotEqual(audit['matching']['requested'], audit['matching']['achieved'])
            self.assertTrue(validate_residual_action(data, action, base_action=base, arm='J')['valid'])

    def test_no_change_remains_honest_identity_for_all_arms(self):
        data, base = graph_and_base(8)
        data['capacity'][:,0] = 2.
        data['logic_src'] = data['logic_src'][:0]
        data['logic_dst'] = data['logic_dst'][:0]
        data['logic_weight'] = data['logic_weight'][:0]
        data['logic_criticality'] = data['logic_criticality'][:0]
        actions = make_attribution_candidates(data, base)
        self.assertEqual({action_digest(v) for v in actions.values()}, {action_digest(base)})
        for name in ('R1','R2'):
            self.assertTrue(actions[name]['attribution_diagnostics']['matched_control'])
            self.assertEqual(actions[name]['attribution_diagnostics']['matching']['requested_templates'], 0)

    def test_impossible_xy_projection_is_rejected_instead_of_repaired(self):
        data, base = graph_and_base(400)
        bank = make_candidates(data, base, 'J', **GENERATOR)
        joint = next(a for a in bank if a['candidate_id']=='j_s1')
        # Supply a valid-in-J coordinate that leaves the old base region; the
        # component projection must fail loudly rather than change the action.
        node = int(torch.where(joint['layer'] != base['layer'])[0][0])
        old_leaf = int(base['assignment'][node])
        new_leaf = int(joint['assignment'][node])
        axis = 0
        joint['xy'][node, axis] = .1 if float(base['xy'][node,axis]) > .5 else .9
        data['region_bounds'][new_leaf] = torch.tensor([0.,1.,0.,1.])
        joint['deployed_action_sha256'] = action_digest(joint)
        with patch('attribution_actions.make_candidates', return_value=bank):
            with self.assertRaisesRegex(ValueError, 'cannot be exactly decomposed'):
                make_attribution_candidates(data, base)


if __name__ == '__main__':
    unittest.main()
