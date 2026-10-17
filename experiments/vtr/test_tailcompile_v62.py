import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import math
import unittest
import torch
from run_tailcompile_v62 import (PILOT, conservative_audit, compare,
                                 evaluate_interventions, evaluation_scope)


class V62Test(unittest.TestCase):
    def test_pilot_scope_uses_exact_widths(self):
        report = [dict(design=design, width=width) for design, width in PILOT]
        report += [dict(design=design, width=width + 20) for design, width in PILOT]
        specs = [f'{design}:{width}' for design, width in PILOT]
        selected, conditions = evaluation_scope(report, [], specs)
        self.assertEqual({(r['design'], r['width']) for r in selected}, set(PILOT))
        self.assertEqual(conditions, set(PILOT))

    def test_heldout_evaluation_uses_noop_baseline(self):
        class Model(torch.nn.Module):
            def forward(self, graph):
                return {'score': graph['score']}
        rows = [dict(design='d', width=80, method='noop', route=0., qor=[None] * 4,
                     graph={'score': torch.tensor(0.)}),
                dict(design='d', width=80, method='swap01_r0', route=1., qor=[1.] * 4,
                     graph={'score': torch.tensor(1.)})]
        report = evaluate_interventions(Model(), rows, torch.device('cpu'))
        self.assertEqual(report['selections'][0]['chosen'], 'swap01_r0')
        self.assertEqual(report['selections'][0]['baseline_route'], 0.)

    def test_conservative_policy_accepts_unanimous_safe_gain(self):
        zero = [dict(score=1., route=.5, qor=[0., 0., 4., 2.]) for _ in range(3)]
        good = [dict(score=1.1, route=.51, qor=[0., 0., 3.99, 1.99]) for _ in range(3)]
        audit = conservative_audit('x', 2, good, zero, .01, 1., .01)
        self.assertTrue(audit['eligible'])
        uncertain = [dict(good[0]), dict(good[0]), dict(score=.8, route=.4, qor=[0., 0., 4.1, 2.1])]
        self.assertFalse(conservative_audit('x', 2, uncertain, zero, .01, 1., .01)['eligible'])

    def test_compare_success_first(self):
        rows = []
        for method, status in (('default', 'success'), ('gnn_base', 'unroutable'), ('v62', 'success')):
            row = dict(design='d', width=80, seed=61, candidate=method, status=status)
            if status == 'success':
                row.update(region_p99=1., channel_p99=1., wirelength=10., final_cpd_ns=2.)
            rows.append(row)
        result = compare(rows)
        self.assertEqual(result['default']['net_success_gain'], 0)
        self.assertEqual(result['gnn_base']['net_success_gain'], 1)


if __name__ == '__main__':
    unittest.main()
