"""Regression tests: python experiments/vtr/tailcompile_v69/test_metrics.py."""
import json
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import aggregate_widths, compare_runtime, choose_on_validation


SEEDS = [11, 12, 13, 14, 15]
GRID = [90, 100, 102, 110]


def scans():
    return [dict(design='circuit', width=w, seed=s, method=m,
                 status='success' if w >= (100 if m == 'default' else 102) else 'unroutable')
            for m in ('default', 'fast') for w in GRID for s in SEEDS]


def width_report(rows=None):
    return aggregate_widths(scans() if rows is None else rows,
                            seeds=SEEDS, widths_by_design={'circuit': GRID})


def conditions():
    return [dict(design='circuit', width=110, seed=s) for s in SEEDS]


def timing():
    return [dict(**c, method=m, status='success',
                 vpr_seconds=100 if m == 'default' else 85,
                 generation_seconds=0 if m == 'default' else 5,
                 cpd=10 if m == 'default' else 10.2,
                 wirelength=1000 if m == 'default' else 1020)
            for c in conditions() for m in ('default', 'fast')]


def compare(rows=None, widths=None, **kwargs):
    return compare_runtime(timing() if rows is None else rows,
                           expected_conditions=conditions(),
                           width_report=width_report() if widths is None else widths, **kwargs)


class WidthMetricsTest(unittest.TestCase):
    def test_arithmetic_mean_of_five_minima_not_luckiest_seed(self):
        rows = scans()
        for r in rows:
            if r['method'] == 'fast' and r['seed'] == 11 and r['width'] == 100:
                r['status'] = 'success'
        g = width_report(rows)['groups'][1]
        self.assertEqual(g['five_seed_mean_observed_wmin'], 101.6)
        self.assertEqual(len(g['per_seed']), 5)

    def test_exactly_five_distinct_seeds_required(self):
        for seeds in (SEEDS[:-1], SEEDS + [16], [1, 1, 2, 3, 4]):
            with self.assertRaises(ValueError):
                aggregate_widths([], seeds=seeds, widths_by_design={'x': [100]})

    def test_missing_seed_is_retained_and_no_four_seed_mean(self):
        rows = [r for r in scans() if not (r['method'] == 'fast' and r['seed'] == 15)]
        g = width_report(rows)['groups'][1]
        self.assertIsNone(g['five_seed_mean_observed_wmin'])
        self.assertEqual(g['unknown_seed_count'], 1)
        self.assertEqual(g['per_seed'][-1]['unknown_widths'], GRID)

    def test_timeout_infra_missing_are_not_unroutable_or_right_censored(self):
        for failure in ('timeout', 'other_failure', 'killed', 'missing'):
            rows = scans()
            for r in rows:
                if r['method'] == 'fast' and r['seed'] == 15:
                    r['status'] = failure
            g = width_report(rows)['groups'][1]
            self.assertIsNone(g['five_seed_mean_observed_wmin'])
            self.assertEqual(g['right_censored_seed_count'], 0)
            self.assertTrue(g['per_seed'][-1]['no_observed_success'])

    def test_right_and_left_censoring_visible(self):
        rows = scans()
        for r in rows:
            if r['method'] == 'fast' and r['seed'] == 15:
                r['status'] = 'unroutable'
            if r['method'] == 'default' and r['seed'] == 11:
                r['status'] = 'success'
        a, b = width_report(rows)['groups']
        self.assertEqual(a['left_censored_seed_count'], 1)
        self.assertEqual(b['right_censored_seed_count'], 1)
        self.assertIsNone(b['five_seed_mean_observed_wmin'])

    def test_nonmonotonic_success_then_failure_does_not_move_minimum(self):
        rows = scans()
        for r in rows:
            if r['method'] == 'default' and r['width'] == 110:
                r['status'] = 'unroutable'
        g = width_report(rows)['groups'][0]
        self.assertEqual(g['five_seed_mean_observed_wmin'], 100)
        self.assertEqual(g['nonmonotone_seed_count'], 5)
        self.assertEqual(g['per_seed'][0]['failures_above_first_success'], [110])

    def test_unknown_above_minimum_stays_visible_and_blocks_complete_mean(self):
        rows = scans()
        for r in rows:
            if r['method'] == 'default' and r['width'] == 110 and r['seed'] == 11:
                r['status'] = 'timeout'
        g = width_report(rows)['groups'][0]
        self.assertEqual(g['per_seed'][0]['observed_minimum_width'], 100)
        self.assertEqual(g['per_seed'][0]['unknown_widths'], [110])
        self.assertIsNone(g['five_seed_mean_observed_wmin'])

    def test_duplicate_and_unplanned_trials_rejected(self):
        rows = scans()
        with self.assertRaises(ValueError):
            width_report(rows + [rows[0]])
        with self.assertRaises(ValueError):
            width_report(rows + [dict(rows[0], seed=999)])


class RuntimeMetricsTest(unittest.TestCase):
    def test_small_quality_regression_accepted_for_inclusive_runtime_gain(self):
        r = compare()
        self.assertTrue(r['passed'])
        self.assertAlmostEqual(r['aggregate']['generation_inclusive_speedup'], 100 / 90)
        self.assertAlmostEqual(r['by_design'][0]['width_guard']['ratio'], 1.02)
        self.assertGreater(r['aggregate']['candidate_over_baseline_geomean']['cpd'], 1)
        self.assertEqual(choose_on_validation(r, split='validation')['selected_method'], 'fast')

    def test_generation_cost_cannot_be_omitted_or_hide_slowdown(self):
        for cost in (None, 20, -1, float('nan'), float('inf')):
            rows = timing()
            for row in rows:
                if row['method'] == 'fast':
                    row['generation_seconds'] = cost
            r = compare(rows)
            self.assertFalse(r['passed'])
            json.dumps(r, allow_nan=False)
        rows = timing()
        for row in rows:
            del row['generation_seconds']
        self.assertFalse(compare(rows)['passed'])

    def test_timeout_does_not_pass_by_excluding_bad_pair(self):
        rows = timing()
        rows[1]['status'] = 'timeout'
        r = compare(rows)
        self.assertFalse(r['passed'])
        self.assertEqual(r['aggregate']['expected_pairs'], 5)
        self.assertEqual(r['aggregate']['joint_success_pairs'], 4)
        self.assertEqual(r['aggregate']['unknown_outcome_pairs'], 1)
        self.assertEqual(r['aggregate']['joint_success_coverage'], .8)

    def test_success_loss_is_not_compensated_by_other_gain(self):
        rows = timing()
        rows[1]['status'] = 'unroutable'
        rows[2]['status'] = 'unroutable'
        r = compare(rows)
        self.assertFalse(r['passed'])
        self.assertEqual(r['aggregate']['success_loss_pairs'], 1)
        self.assertEqual(r['aggregate']['success_gain_pairs'], 1)

    def test_worse_width_or_cpd_rejected_unless_explicit_budget(self):
        self.assertFalse(compare(budgets={'width_regression': .01})['passed'])
        self.assertFalse(compare(budgets={'cpd_regression': .01})['passed'])
        self.assertTrue(compare(budgets={'width_regression': .02, 'cpd_regression': .02})['passed'])

    def test_nan_negative_and_zero_metrics_fail_without_invalid_json(self):
        for name in ('cpd', 'wirelength', 'vpr_seconds'):
            for value in (float('nan'), float('inf'), -1, 0):
                rows = timing()
                rows[1][name] = value
                report = compare(rows)
                self.assertFalse(report['passed'])
                json.dumps(report, allow_nan=False)

    def test_missing_width_report_and_missing_timing_block_acceptance(self):
        self.assertFalse(compare_runtime(timing(), expected_conditions=conditions())['passed'])
        self.assertFalse(compare(timing()[:-1])['passed'])

    def test_nested_metrics_and_current_v68_aliases(self):
        rows = timing()
        for r in rows:
            r['metrics'] = dict(final_cpd_ns=r.pop('cpd'), wirelength=r.pop('wirelength'))
            r['wall_seconds'] = r.pop('vpr_seconds')
        self.assertTrue(compare(rows)['passed'])

    def test_no_test_label_selection(self):
        with self.assertRaises(ValueError):
            choose_on_validation(compare(), split='test')

    def test_aggregate_cannot_hide_a_design_quality_regression(self):
        rows = timing()
        second = [dict(r, design='second') for r in timing()]
        for row in second:
            if row['method'] == 'fast':
                row['cpd'] = 10.4
        scans_all = scans() + [dict(r, design='second') for r in scans()]
        widths = aggregate_widths(scans_all, seeds=SEEDS,
                                  widths_by_design={'circuit': GRID, 'second': GRID})
        expected = conditions() + [dict(c, design='second') for c in conditions()]
        report = compare_runtime(rows + second, expected_conditions=expected, width_report=widths)
        self.assertLess(report['aggregate']['candidate_over_baseline_geomean']['cpd'], 1.03)
        self.assertFalse(report['passed'])

    def test_aggregate_runtime_can_pass_with_one_nonbenefiting_design(self):
        rows = timing()
        for row in rows:
            if row['method'] == 'fast':
                row['vpr_seconds'] = 75  # total ratio .80
        second = [dict(r, design='second') for r in timing()]
        for row in second:
            if row['method'] == 'fast':
                row['vpr_seconds'] = 100  # total ratio 1.05
        scans_all = scans() + [dict(r, design='second') for r in scans()]
        widths = aggregate_widths(scans_all, seeds=SEEDS,
                                  widths_by_design={'circuit': GRID, 'second': GRID})
        expected = conditions() + [dict(c, design='second') for c in conditions()]
        report = compare_runtime(rows + second, expected_conditions=expected, width_report=widths)
        self.assertTrue(report['passed'])
        self.assertTrue(report['complete'])
        self.assertTrue(report['evidence_complete'])
        self.assertAlmostEqual(report['aggregate']['candidate_over_baseline_geomean']['total'], (.8 * 1.05) ** .5)
        slower = report['by_design'][1]
        self.assertFalse(slower['runtime_target_met'])
        self.assertTrue(slower['runtime_regressed'])
        self.assertTrue(slower['quality_and_evidence_passed'])
        self.assertEqual(report['aggregate']['designs_with_runtime_regression'], 1)

    def test_runtime_designs_equal_weight_not_number_of_conditions(self):
        rows = timing()  # five pairs with total ratio .9
        second = [dict(r, design='second') for r in timing()[:2]]
        for row in second:
            if row['method'] == 'fast':
                row['vpr_seconds'] = 105  # one pair with ratio 1.1
        scans_all = scans() + [dict(r, design='second') for r in scans()]
        widths = aggregate_widths(scans_all, seeds=SEEDS,
                                  widths_by_design={'circuit': GRID, 'second': GRID})
        expected = conditions() + [dict(conditions()[0], design='second')]
        report = compare_runtime(rows + second, expected_conditions=expected, width_report=widths)
        # Pooled pairs would falsely pass at ~.931, but equal designs give ~.995.
        self.assertLess(report['aggregate']['condition_weighted_candidate_over_baseline_geomean']['total'], .95)
        self.assertAlmostEqual(report['aggregate']['candidate_over_baseline_geomean']['total'], (.9 * 1.1) ** .5)
        self.assertFalse(report['passed'])

    def test_no_mix_of_widths_or_attempts(self):
        rows = timing()
        with self.assertRaises(ValueError):
            compare(rows + [rows[0]])
        rows[1]['width'] = 112
        with self.assertRaises(ValueError):
            compare(rows)

    def test_empty_and_invalid_budgets_rejected(self):
        for kwargs in ({'cpd_regression': -.01}, {'min_total_time_reduction': 1}, {'unknown': 0}):
            with self.assertRaises(ValueError):
                compare(budgets=kwargs)
        with self.assertRaises(ValueError):
            compare_runtime([], expected_conditions=[])


if __name__ == '__main__':
    unittest.main()
