import unittest

from analyze_tailcompile_v5 import compare, correlation
from run_tailcompile_v5 import gini


class ConfirmatoryUtilitiesTest(unittest.TestCase):
    def test_gini(self):
        self.assertEqual(gini([0, 0, 0]), 0.0)
        self.assertAlmostEqual(gini([0, 1]), 0.5)
        self.assertAlmostEqual(gini([1, 1, 1]), 0.0)

    def test_rank_interaction(self):
        self.assertAlmostEqual(correlation([1, 2, 3], [10, 20, 30]), 1.0)
        self.assertAlmostEqual(correlation([1, 2, 3], [30, 20, 10]), -1.0)
        self.assertIsNone(correlation([1, 2], [1, 2]))

    def test_paired_comparison_uses_design_heterogeneity(self):
        index = {
            ('a', 80, 'gnn_tail', 21): {
                'status': 'success', 'region_p99': 9, 'channel_p99': 9,
                'wirelength': 90, 'final_cpd_ns': 9},
            ('a', 80, 'gnn_mean', 21): {
                'status': 'success', 'region_p99': 10, 'channel_p99': 10,
                'wirelength': 100, 'final_cpd_ns': 10},
            ('b', 80, 'gnn_tail', 21): {
                'status': 'success', 'region_p99': 11, 'channel_p99': 11,
                'wirelength': 110, 'final_cpd_ns': 11},
            ('b', 80, 'gnn_mean', 21): {
                'status': 'success', 'region_p99': 10, 'channel_p99': 10,
                'wirelength': 100, 'final_cpd_ns': 10},
            ('c', 80, 'gnn_tail', 21): {
                'status': 'success', 'region_p99': 6, 'channel_p99': 7,
                'wirelength': 80, 'final_cpd_ns': 8},
            ('c', 80, 'gnn_mean', 21): {
                'status': 'success', 'region_p99': 10, 'channel_p99': 10,
                'wirelength': 100, 'final_cpd_ns': 10},
        }
        lock = {'strata': {'a': 'medium', 'b': 'low', 'c': 'high'},
                'design_scores': {'a': 2.0, 'b': 1.0, 'c': 3.0}}
        _, rows, interaction, strata = compare(
            index, ['a', 'b', 'c'], lock, 'gnn_tail', 'gnn_mean')
        self.assertEqual(len(rows), 3)
        self.assertAlmostEqual(strata['high']['region_p99_improvement_percent'], 40.0)
        self.assertEqual(strata['low']['route_success_gain'], 0)
        self.assertGreater(interaction['region_p99_improvement_spearman'], 0)


if __name__ == '__main__':
    unittest.main()
