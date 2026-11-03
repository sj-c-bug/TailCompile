"""012 local edit descriptors: identity, attribution, invariance and bounded work."""
import copy
import unittest

import torch

from edit_context_features import (local_features, edit_feature_dim, FEATURE_SCHEMA,
    EMBEDDING_BLOCKS, SCALAR_FEATURE_NAMES)
from residual_actions import action_features, make_candidates
from test_residual_actions import graph_and_base


def symmetric_case():
    data, base = graph_and_base(8)
    bounds = data['region_bounds'][base['assignment']]
    base['xy'] = (bounds[:, [0, 2]]+bounds[:, [1, 3]])/2
    data['logic_src'] = torch.tensor([0, 4])
    data['logic_dst'] = torch.tensor([1, 5])
    data['logic_weight'] = torch.ones(2)
    data['logic_criticality'] = torch.tensor([.5, .5])
    a, b = copy.deepcopy(base), copy.deepcopy(base)
    a['xy'][0, 0] += .05; b['xy'][4, 0] += .05
    a['candidate_strength'] = b['candidate_strength'] = .15
    logic = torch.arange(8*4, dtype=torch.float32).reshape(8, 4)/12-1
    region = torch.tensor([[.1, .3, .2, -.2], [.2, .2, -.3, .4],
                           [-.5, -.6, .3, .1], [.6, -.1, .5, .7]])
    return data, base, [base, a, b], logic, region


class EditContextFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_global_descriptors_distinguish_which_nodes_and_neighbors_changed(self):
        data, base, actions, logic, region = symmetric_case()
        self.assertTrue(torch.equal(action_features(data, base, actions[1]),
                                    action_features(data, base, actions[2])))
        result = local_features(data, base, actions, logic, region)
        self.assertEqual(result.shape, (3, edit_feature_dim(4)))
        self.assertTrue(torch.equal(result[0], torch.zeros_like(result[0])))
        self.assertFalse(torch.allclose(result[1], result[2]))
        # The difference is in local representations, not numeric node IDs:
        # every scalar edit/geometry/resource/edge summary is identical here.
        self.assertTrue(torch.allclose(result[1, 40:], result[2, 40:]))
        self.assertEqual(len(SCALAR_FEATURE_NAMES), 77)
        self.assertEqual(len(EMBEDDING_BLOCKS), 10)
        self.assertIn('012', FEATURE_SCHEMA)

    def test_embedding_and_local_neighbor_dependence_with_no_label_access(self):
        data, base, actions, logic, region = symmetric_case()
        class Forbidden:
            def __array__(self): raise AssertionError('teacher/route label read')
            def __iter__(self): raise AssertionError('teacher/route label read')
        for name in ('teacher', 'teacher_xy', 'cpd_ns', 'wirelength', 'route'):
            data[name] = Forbidden()
        original = local_features(data, base, actions, logic, region)
        changed = logic.clone(); changed[1] += 1.
        result = local_features(data, base, actions, changed, region)
        self.assertFalse(torch.equal(original[1], result[1]))
        self.assertTrue(torch.equal(original[2], result[2]))

    def test_source_destination_region_direction_is_explicit(self):
        data, base = graph_and_base(20)
        bank = make_candidates(data, base, 'L', max_moves=4, fraction=1.)
        self.assertGreater(len(bank), 1)
        torch.manual_seed(12001)
        logic = torch.randn(20, 5); region = torch.randn(4, 5)
        normal = local_features(data, base, bank, logic, region)
        swapped = local_features(data, bank[-1], [bank[-1], base], logic, region)
        directional = EMBEDDING_BLOCKS.index('destination_minus_source_region_mean')
        start, stop = directional*5, (directional+1)*5
        self.assertTrue(torch.allclose(normal[-1, start:stop], -swapped[-1, start:stop], atol=1e-6))

    def test_relabeling_logical_nodes_and_regions_preserves_features(self):
        data, base, actions, logic, region = symmetric_case()
        original = local_features(data, base, actions, logic, region)
        order = torch.tensor([4, 1, 7, 3, 0, 5, 2, 6]); inverse = torch.argsort(order)
        relabeled = copy.deepcopy(data)
        for key in ('cluster_x','demand','movable','fixed_region','fixed_layer','fixed_xy'):
            relabeled[key] = data[key][order]
        relabeled['logic_src'] = inverse[data['logic_src']]
        relabeled['logic_dst'] = inverse[data['logic_dst']]
        moved = [{k:(v[order] if k in ('assignment','layer','xy') else v) for k,v in a.items()}
                 for a in actions]
        result = local_features(relabeled, moved[0], moved, logic[order], region)
        self.assertTrue(torch.allclose(original, result, atol=1e-6))
        order = torch.tensor([3, 0, 2, 1]); inverse = torch.argsort(order)
        for key in ('capacity','region_bounds','region_layer'):
            relabeled[key] = relabeled[key][order]
        moved = [dict(a, assignment=inverse[a['assignment']]) for a in moved]
        result = local_features(relabeled, moved[0], moved, logic[torch.tensor([4,1,7,3,0,5,2,6])], region[order])
        self.assertTrue(torch.allclose(original, result, atol=1e-6))

    def test_detached_deterministic_and_preserves_actions_fixed_axes_and_graph(self):
        data, base = graph_and_base(20)
        data['movable'][0] = False; data['fixed_xy'][1, 0] = base['xy'][1, 0]
        bank = make_candidates(data, base, 'J', fraction=1., max_moves=5)
        before_data = {k:v.clone() for k,v in data.items() if isinstance(v, torch.Tensor)}
        before = [{k:v.clone() for k,v in a.items() if isinstance(v, torch.Tensor)} for a in bank]
        logic = torch.randn(20, 4, requires_grad=True); region = torch.randn(4, 4, requires_grad=True)
        first = local_features(data, base, bank, logic, region)
        torch.manual_seed(9999); torch.rand(100)
        second = local_features(data, base, bank, logic, region)
        self.assertTrue(torch.equal(first, second))
        self.assertFalse(first.requires_grad); self.assertIsNone(first.grad_fn)
        self.assertEqual(first.device.type, 'cpu')
        self.assertTrue(torch.isfinite(first).all())
        self.assertLessEqual(float(first.abs().max()), 1.)
        for key,value in before_data.items():self.assertTrue(torch.equal(data[key], value))
        for action,old in zip(bank,before):
            for key,value in old.items():self.assertTrue(torch.equal(action[key], value))

    def test_isolated_and_zero_criticality_edges_remain_finite(self):
        data, base, actions, logic, region = symmetric_case()
        for key in ('logic_src','logic_dst','logic_weight','logic_criticality'):
            data[key] = data[key][:0]
        result = local_features(data, base, actions, logic, region)
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue(torch.equal(result[0], torch.zeros_like(result[0])))

    def test_budget_nonfinite_and_malformed_coverage_are_rejected(self):
        data, base, actions, logic, region = symmetric_case()
        with self.assertRaisesRegex(ValueError, 'one to three'):
            local_features(data,base,actions+[base],logic,region)
        with self.assertRaisesRegex(ValueError, 'dimensions disagree'):
            local_features(data,base,actions,logic[:3],region)
        data['logic_weight'][0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            local_features(data,base,actions,logic,region)
        data,base = graph_and_base(300)
        modified=copy.deepcopy(base); modified['xy'][:,1] += .01
        with self.assertRaisesRegex(ValueError, '256'):
            local_features(data,base,[base,modified],torch.zeros(300,4),torch.zeros(4,4))


if __name__ == '__main__':unittest.main()
