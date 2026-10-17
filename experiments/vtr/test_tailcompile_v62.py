import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import math
import unittest
from run_tailcompile_v62 import conservative_audit, compare


class V62Test(unittest.TestCase):
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
