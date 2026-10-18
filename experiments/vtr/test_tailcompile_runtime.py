"""Offline checks for VPR stage parsing and success-stratified runtime accounting."""
import unittest
from pathlib import Path

from analyze_tailcompile_runtime import parse_vpr_log, ratio_summary


HERE = Path(__file__).parent


class RuntimeAuditTest(unittest.TestCase):
    def test_real_success_and_failure_logs(self):
        root = HERE / 'tailcompile_v4/runs/server_spmv_001/spmv/w90/runs'
        success = root / 'gnn_base_s11/vpr.out'
        failure = root / 'default_s11/vpr.out'
        if not success.exists() or not failure.exists():
            self.skipTest('optional archived VPR logs absent')
        good = parse_vpr_log(success.read_text(errors='replace'))
        bad = parse_vpr_log(failure.read_text(errors='replace'))
        self.assertEqual(good['routing_iterations'], 22)
        self.assertEqual(bad['routing_iterations'], 50)
        self.assertAlmostEqual(good['placement_seconds'], 23.34)
        self.assertAlmostEqual(bad['routing_seconds'], 6.70)
        self.assertEqual(good['placement_swaps'], 670428)

    def test_missing_log_fields_are_not_fabricated(self):
        result = parse_vpr_log('VPR succeeded\n')
        self.assertIsNone(result['routing_iterations'])
        self.assertIsNone(result['placement_seconds'])

    def test_paired_geomean_and_win_counts(self):
        rows = [({'wall_seconds': 8}, {'wall_seconds': 10}),
                ({'wall_seconds': 18}, {'wall_seconds': 20})]
        result = ratio_summary(rows, 'wall_seconds')
        self.assertEqual(result['pairs'], 2)
        self.assertEqual(result['faster'], 2)
        self.assertAlmostEqual(result['geomean_ratio'], (0.8 * 0.9) ** 0.5)


if __name__ == '__main__':
    unittest.main()
