from pathlib import Path
import tempfile
import unittest
from qor_route_metrics import summarize_route


class RouteUsageTests(unittest.TestCase):
    def test_shared_branches_deduplicated_per_net(self):
        text='''Net 0 (a)
Node: 1 CHANX (1,2,0) to (4,2,0)
Node: 2 CHANZ (4,2,0) to (4,2,1)
Node: 2 CHANZ (4,2,0) to (4,2,1)
Net 1 (b)
Node: 2 CHANZ (4,2,0) to (4,2,1)
'''
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.route'; p.write_text(text); r=summarize_route(p)
        self.assertEqual(r['unique_chanz_nodes'],1)
        self.assertEqual(r['per_net_unique_chanz_nodes'],2)
        self.assertEqual(r['z_hop_sum'],2)
        self.assertEqual(r['xy_tile_span_sum'],4)
        self.assertEqual(r['repeated_branch_rows_ignored'],1)

    def test_missing_z_is_unknown(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'tiny.route'; p.write_text('Net 0 (a)\nNode: 2 CHANZ (4,2)\n')
            r=summarize_route(p)
        self.assertIsNone(r['z_hop_sum'])
        self.assertEqual(r['unique_chanz_nodes'],1)


if __name__=='__main__': unittest.main()
