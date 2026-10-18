"""Offline regression checks for the frozen-width sweep and censoring rules."""
import json
import tempfile
import unittest
from pathlib import Path

from run_tailcompile_v64_wmin import (compare_pairs, make_plan, sha, summarize_rows,
                                      width_result)


class WidthSweepTest(unittest.TestCase):
    def test_grid_minimum_and_nonmonotonicity(self):
        widths = [100, 110, 120, 130]
        records = [dict(width=w, status=s) for w, s in zip(widths,
                   ('unroutable', 'success', 'unroutable', 'success'))]
        self.assertEqual(width_result(widths, records), dict(w_min_grid=110,
            censor='none', lower_bound_exclusive=None, monotonicity_violations=1))
        self.assertEqual(width_result(widths, [dict(width=w, status='success') for w in widths])['censor'], 'left')
        self.assertEqual(width_result(widths, [dict(width=w, status='unroutable') for w in widths])['censor'], 'right')

    def test_right_censoring_is_not_mistaken_for_numeric_minimum(self):
        candidate = {81: dict(w_min_grid=110, censor='none'),
                     82: dict(w_min_grid=100, censor='left')}
        baseline = {81: dict(w_min_grid=None, censor='right'),
                    82: dict(w_min_grid=120, censor='none')}
        result = compare_pairs(candidate, baseline)
        self.assertEqual(result['observed_numeric_pairs'], 1)
        self.assertEqual(result['exact_numeric_pairs'], 0)
        self.assertEqual(result['outcomes']['improved_at_least_to_grid_limit'], 1)
        self.assertEqual(result['median_delta_grid'], -20)

    def test_summary_uses_identical_widths_and_seeds(self):
        plan = dict(schema='test', task_count=18, seeds=[81, 82, 83],
            selection_report_sha256='frozen', cases=[dict(design='x', anchor_width=120,
                selected='swap02_r0', widths=[100, 110])])
        rows = []
        for width in (100, 110):
            for method in ('default', 'gnn_base', 'v63'):
                for seed in plan['seeds']:
                    success = width == 110 or method == 'v63'
                    row = dict(design='x', anchor_width=120, width=width,
                        method=method, seed=seed, status='success' if success else 'unroutable')
                    if success:
                        row.update(region_p99=.7, channel_p99=.8, wirelength=100, final_cpd_ns=5)
                    rows.append(row)
        report = summarize_rows(plan, rows)
        condition = report['conditions'][0]
        self.assertEqual(condition['methods']['v63']['w80_grid'], 100)
        self.assertEqual(condition['methods']['gnn_base']['w80_grid'], 110)
        self.assertEqual(condition['comparisons']['gnn_base']['w80_delta_grid'], -10)
        self.assertEqual(condition['comparisons']['default']['outcomes']['improved'], 3)

    def test_plan_is_frozen_to_active_actions_and_fresh_seeds(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as folder:
            root = Path(folder)
            selections = []
            for design, anchor in (('conv_layer', 140), ('lenet', 160)):
                inputs = root / design
                inputs.mkdir()
                (inputs / f'{design}.pre-vpr.blif').write_text('.model test\n')
                (inputs / f'{design}.net').write_text('net')
                baseline, selected = inputs / 'base.fplace', inputs / 'selected.fplace'
                baseline.write_text('base')
                selected.write_text('selected')
                selections.append(dict(design=design, width=anchor, selected='swap02_r0',
                    inputs=str(inputs), baseline=str(baseline), baseline_sha256=sha(baseline),
                    selected_fplace=str(selected), selected_sha256=sha(selected),
                    device_width=20, full_legalizer='appack'))
            path = root / 'selection_report.json'
            path.write_text(json.dumps(dict(selections=selections)))
            plan = make_plan(path, [81, 82, 83], {('conv_layer', 140): [100, 110, 120],
                ('lenet', 160): [160, 170, 180]})
            self.assertEqual(plan['task_count'], 54)
            guided = make_plan(path, [101, 102, 103],
                {('conv_layer', 140): [100, 110, 120],
                 ('lenet', 160): [160, 170, 180]}, 'guided_tail')
            self.assertEqual(guided['candidate_method'], 'guided_tail')
            with self.assertRaises(ValueError):
                make_plan(path, [71, 82, 83], {('conv_layer', 140): [100, 110, 120],
                    ('lenet', 160): [160, 170, 180]})


if __name__ == '__main__':
    unittest.main()
