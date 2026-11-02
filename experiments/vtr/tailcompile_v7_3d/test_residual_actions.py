"""011 edit attribution, bounded work, constraints and truthful action identity."""
import copy
import math
import unittest

import torch

from aligned_model import action_digest
from structured_model import validate_structured_action
from residual_actions import (make_candidates, action_features, ACTION_FEATURE_DIM,
                              validate_residual_action, SCHEMA, DECODER_SCHEMA)
from test_model import fixture


def graph_and_base(n=6):
    data = fixture(n)
    data['density_spread_fraction'] = 0.
    data['capacity'] = torch.tensor([[float(n), float(n), 2., 2., 2.]]*4)
    assignment = torch.arange(n) % 4
    layer = data['region_layer'][assignment]
    bounds = data['region_bounds'][assignment]
    xy = (bounds[:, [0, 2]]+bounds[:, [1, 3]])/2
    xy[:, 1] = torch.linspace(.2, .8, n)
    action = dict(assignment=assignment, layer=layer, xy=xy)
    validate_structured_action(data, action)
    return data, action


class ResidualActionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_l_y_j_have_exact_attribution_and_shared_sparse_budget(self):
        data, base = graph_and_base(40)
        original = copy.deepcopy(base)
        for arm in ('L', 'Y', 'J'):
            candidates = make_candidates(data, base, arm, max_moves=7, fraction=1.)
            self.assertEqual(candidates[0]['candidate_id'], 'identity')
            self.assertEqual(len(candidates), 3)
            self.assertEqual(len({action_digest(a) for a in candidates}), len(candidates))
            for action in candidates:
                audit = validate_residual_action(data, action, base_action=base, arm=arm)
                self.assertLessEqual(audit['changed_nodes'], 7)
                self.assertEqual(audit['overflow_region_count'], 0)
                if arm == 'L':
                    self.assertTrue(torch.equal(action['xy'], base['xy']))
                if arm == 'Y':
                    self.assertTrue(torch.equal(action['assignment'], base['assignment']))
                    self.assertTrue(torch.equal(action['layer'], base['layer']))
                self.assertFalse(action['residual_diagnostics']['convergence_loop'])
        for key in original:
            self.assertTrue(torch.equal(base[key], original[key]))

    def test_immovable_pins_fixed_axis_and_resource_absence_are_preserved(self):
        data, base = graph_and_base(20)
        data['movable'][0] = False
        data['fixed_xy'][1, 0] = base['xy'][1, 0]
        data['fixed_layer'][2] = int(base['layer'][2])
        data['fixed_region'][3] = int(base['assignment'][3])
        # A DSP APBlock may only use leaf0, which also pins its legal layer.
        data['demand'][4] = torch.tensor([0., 0., 0., 1., 0.])
        data['capacity'][:, 3] = 0.; data['capacity'][0, 3] = 2.
        for arm in ('L', 'Y', 'J'):
            candidates = make_candidates(data, base, arm, fraction=1.)
            self.assertGreater(len(candidates), 1)
            for action in candidates:
                self.assertEqual(int(action['assignment'][0]), int(base['assignment'][0]))
                self.assertTrue(torch.equal(action['xy'][0], base['xy'][0]))
                self.assertEqual(float(action['xy'][1, 0]), float(base['xy'][1, 0]))
                self.assertEqual(int(action['layer'][2]), int(base['layer'][2]))
                self.assertEqual(int(action['assignment'][3]), int(base['assignment'][3]))
                self.assertEqual(int(action['assignment'][4]), 0)

    def test_fully_loaded_target_layers_deduplicate_to_identity(self):
        data, base = graph_and_base(8)
        data['capacity'][:, 0] = 2.
        candidates = make_candidates(data, base, 'L', fraction=1.)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]['candidate_id'], 'identity')
        self.assertLessEqual(candidates[0]['residual_diagnostics']['proposal_nodes_considered'], 32)

    def test_zero_budget_and_isolated_xy_produce_no_fake_new_actions(self):
        data, base = graph_and_base()
        for arm in ('L', 'Y', 'J'):
            self.assertEqual(len(make_candidates(data, base, arm, max_moves=0)), 1)
        data['logic_src'] = data['logic_src'][:0]
        data['logic_dst'] = data['logic_dst'][:0]
        data['logic_weight'] = data['logic_weight'][:0]
        data['logic_criticality'] = data['logic_criticality'][:0]
        self.assertEqual(len(make_candidates(data, base, 'Y', fraction=1.)), 1)

    def test_deterministic_actions_and_finite_feature_contract(self):
        data, base = graph_and_base(20)
        first = make_candidates(data, base, 'J', fraction=1., seed=71)
        torch.manual_seed(799); torch.randn(100)
        second = make_candidates(data, base, 'J', fraction=1., seed=71)
        self.assertEqual([a['deployed_action_sha256'] for a in first],
                         [a['deployed_action_sha256'] for a in second])
        for action in first:
            features = action_features(data, base, action)
            self.assertEqual(tuple(features.shape), (ACTION_FEATURE_DIM,))
            self.assertTrue(torch.isfinite(features).all())
        self.assertEqual(float(action_features(data, base, first[0])[0]), 1.)
        self.assertEqual(float(action_features(data, base, first[-1])[0]), 0.)

    def test_capacity_and_integer_tampering_cannot_keep_old_reward_identity(self):
        data, base = graph_and_base(12)
        action = make_candidates(data, base, 'Y', fraction=1.)[-1]
        self.assertEqual(action['schema'], SCHEMA)
        self.assertEqual(action['decoder_kind'], DECODER_SCHEMA)
        bad = copy.deepcopy(action); bad['xy'][0, 0] += .001
        with self.assertRaisesRegex(ValueError, 'digest mismatch'):
            validate_residual_action(data, bad, base_action=base)
        bad = copy.deepcopy(action); bad['assignment'] = bad['assignment'].float(); bad['assignment'][0] = .5
        with self.assertRaisesRegex(ValueError, 'integers'):
            validate_residual_action(data, bad, base_action=base)
        bad = copy.deepcopy(action); bad['base_action_sha256'] = '0'*64
        with self.assertRaisesRegex(ValueError, 'base action digest'):
            validate_residual_action(data, bad, base_action=base)

    def test_original_D_must_be_feasible_not_silently_repaired(self):
        data, base = graph_and_base()
        data['capacity'][:, 0] = 2.
        base['assignment'].fill_(0); base['layer'].fill_(0)
        base['xy'][:, 0] = .25
        with self.assertRaises(ValueError):
            make_candidates(data, base, 'Y')

    def test_new_coordinate_domain_is_explicitly_separate_from_old_shared_head(self):
        data, base = graph_and_base()
        data['density_spread_fraction'] = .8
        # At the exact old D centre the spread-blend interval may not contain
        # arbitrary centres, so construct a truly valid old D first.
        from aligned_teacher import deployed_xy_bounds
        low, high = deployed_xy_bounds(data, base['assignment'])
        base['xy'] = ((low+high)/2).float()
        action = make_candidates(data, base, 'Y', fraction=1.)[0]
        action['xy'][0] = torch.tensor([.01, .01])
        action['residual_diagnostics']['edit_limit'] = 1
        action['deployed_action_sha256'] = action_digest(action)
        self.assertTrue(validate_residual_action(data, action, base_action=base)['valid'])
        with self.assertRaisesRegex(ValueError, 'blend-reachable'):
            validate_structured_action(data, action)

    def test_layer_move_attempts_are_bounded_when_spatial_overlap_is_absent(self):
        data, base = graph_and_base(2000)
        data['legal_layers'] = torch.nn.functional.one_hot(base['layer'], 2).bool()
        candidates = make_candidates(data, base, 'L', fraction=.01, max_moves=20)
        self.assertEqual(len(candidates), 1)

    def test_invalid_budget_parameters_fail_before_generation(self):
        data, base = graph_and_base()
        for kwargs in (dict(max_moves=True), dict(fraction=float('nan')),
                       dict(strengths=(.1,.2,.3)), dict(strengths=(.2,.2))):
            with self.assertRaises(ValueError):
                make_candidates(data, base, 'L', **kwargs)


if __name__ == '__main__':
    unittest.main()
