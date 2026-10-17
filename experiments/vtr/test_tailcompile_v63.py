import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import unittest

import torch
from torch import nn

from run_tailcompile_v63 import (EVAL_SEEDS, LABEL_SEEDS, PlacementChoiceMatcher,
                                 assert_complete, expected_count)
from test_tailcompile_v61 import graph
from train_tailcompile_v61 import group_loss


class ExpandedCampaignTest(unittest.TestCase):
    def test_trial_coverage_and_seed_isolation(self):
        candidates = [dict(design='a', width=80, candidate='default'),
                      dict(design='a', width=80, candidate='noop')]
        rows = [dict(**candidate, seed=seed) for candidate in candidates for seed in LABEL_SEEDS]
        self.assertEqual(expected_count(candidates, LABEL_SEEDS), 6)
        self.assertFalse(set(LABEL_SEEDS) & set(EVAL_SEEDS))
        assert_complete(rows, candidates, LABEL_SEEDS)
        with self.assertRaises(ValueError):
            assert_complete(rows[:-1], candidates, LABEL_SEEDS)

    def test_default_branch_has_trainable_distinct_outputs(self):
        class FixedCore(nn.Module):
            def forward(self, graph):
                return dict(embedding=torch.ones(8), route_logit=torch.tensor(0.),
                            qor=torch.zeros(4), score=torch.tensor(0.),
                            hotspot=torch.zeros(16))

        model = PlacementChoiceMatcher(hidden=8)
        model.core = FixedCore()
        with torch.no_grad():
            model.default_delta[0].weight.zero_()
            model.default_delta[0].bias.fill_(1.)
            model.default_delta[2].weight.zero_()
            model.default_delta[2].bias.copy_(torch.tensor([1., 2., 3., 4., 5., 6.]))
        baseline = model({'is_default': False})
        default = model({'is_default': True})
        self.assertEqual(float(baseline['score']), 0.)
        self.assertEqual(float(default['score'].detach()), 6.)
        self.assertEqual(float(default['route_logit'].detach()), 1.)
        self.assertEqual(default['qor'].detach().tolist(), [2., 3., 4., 5.])

    def test_default_branch_participates_in_pair_training(self):
        torch.manual_seed(3)
        model = PlacementChoiceMatcher(hidden=12)
        baseline = graph(2, 2)
        rows = []
        for method, route in (('default', 1.), ('noop', 0.)):
            rows.append(dict(design='d', width=80, method=method, route=route,
                qor=[.5, .5, 100., 5.] if route else [None] * 4,
                hotspot=None, graph=dict(baseline, is_default=method == 'default')))
        loss = group_loss(model, rows, [(0., 1.)] * 4, torch.device('cpu'))
        loss.backward()
        self.assertGreater(float(model.default_delta[2].weight.grad.abs().sum()), 0.)


if __name__ == '__main__':
    unittest.main()
