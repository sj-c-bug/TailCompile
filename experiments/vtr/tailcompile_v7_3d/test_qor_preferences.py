"""009 mathematical/coverage contracts; no Torch, training or router required."""
import copy
import json
import math
from pathlib import Path
import tempfile
import unittest

from qor_preferences import build_preferences, evaluate, choose_epoch, METHODS, LEARNED


def action(index=0, xy_hash='frozen', xy=None):
    return dict(shared_xy_sha256=xy_hash, layer=[index], assignment=[index], xy=xy or [[.2, .3]])


def entry(name, widths=(10., 10.), cpds=(10., 10.), times=(10., 10.),
          statuses=('success', 'success'), index=0, xy_hash='frozen'):
    return dict(condition_id='train/H0Z0/w128', candidate_id=name, split='train',
        action=action(index, xy_hash), trials=[dict(seed=s+1, status=statuses[s],
            wirelength=widths[s], cpd_ns=cpds[s], vpr_seconds=times[s]) for s in range(2)])


def measured(method, *, seed=1, condition='a/H0Z0/w128', family='a', architecture='H0Z0',
             wl=10., cpd=10., total=10., status='success', epoch=None):
    row = dict(method=method, seed=seed, condition_id=condition, family=family, architecture=architecture,
               width_role='reference', status=status, wirelength=wl, cpd_ns=cpd,
               total_seconds=total, vpr_seconds=total)
    if epoch is not None: row['epoch'] = epoch
    return row


def validation():
    rows = [measured('B', seed=s, epoch=0) for s in (1, 2)]
    rows += [measured(m, seed=s, epoch=e) for m in LEARNED for e in (5, 10) for s in (1, 2)]
    return rows


class PreferenceTests(unittest.TestCase):
    def test_qor_arithmetic_seed_means_reverse_runtime_ranking(self):
        a = entry('a', widths=(1., 9.), cpds=(4., 4.), times=(1., 1.), index=0)
        b = entry('b', widths=(4., 4.), cpds=(4., 4.), times=(2., 2.), index=1)
        result = build_preferences([a, b])
        pair = result['pairs_by_condition'][a['condition_id']][0]
        self.assertEqual(pair['better'], b['action'])
        self.assertAlmostEqual(pair['raw_delta'], .5*math.log(5/4))
        self.assertAlmostEqual(result['scales']['qor'], .5*math.log(5/4))
        self.assertAlmostEqual(result['scales']['runtime'], math.log(2))
        old = build_preferences([a, b], objective='runtime', scales=result['scales'])
        self.assertEqual(old['pairs_by_condition'][a['condition_id']][0]['better'], a['action'])

    def test_success_count_is_categorical_and_failure_time_unused(self):
        a = entry('a', statuses=('success', 'unroutable'), index=0)
        b = entry('b', statuses=('unroutable', 'timeout'), widths=(None, None),
                  cpds=(None, None), times=(-999., None), index=1)
        for objective in ('qor', 'runtime'):
            result = build_preferences([a, b], objective=objective)
            pair = result['pairs_by_condition'][a['condition_id']][0]
            self.assertEqual(pair['kind'], 'success_count')
            self.assertEqual(pair['weight'], 1.)

    def test_equal_success_count_different_seeds_has_no_preference(self):
        a = entry('a', statuses=('success', 'unroutable'))
        b = entry('b', statuses=('unroutable', 'success'), index=1)
        self.assertEqual(build_preferences([a, b])['total_pairs'], 0)

    def test_unknown_is_not_a_bad_training_example(self):
        a, b = entry('a'), entry('b', index=1, statuses=('unknown', 'success'))
        result = build_preferences([a, b])
        self.assertEqual(result['total_pairs'], 0)
        self.assertFalse(result['audit'][1]['eligible'])
        self.assertEqual(result['audit'][1]['reason'], 'infrastructure_or_unknown_status')

    def test_missing_qor_is_not_filled_with_runtime(self):
        a, b = entry('a'), entry('b', cpds=(None, 10.), index=1)
        result = build_preferences([a, b])
        self.assertEqual(result['total_pairs'], 0)
        self.assertEqual(result['audit'][1]['reason'], 'missing_measured_qor_metric')

    def test_train_boundary_and_duplicate_candidates(self):
        a = entry('a'); a['split'] = 'validation'
        with self.assertRaisesRegex(ValueError, 'train-only'):
            build_preferences([a])
        a = entry('a')
        with self.assertRaisesRegex(ValueError, 'duplicate candidate'):
            build_preferences([a, copy.deepcopy(a)])
        a['trials'][1]['seed'] = 1
        self.assertFalse(build_preferences([a])['audit'][0]['eligible'])

    def test_scales_and_inputs_immutable_weight_capped(self):
        a, b = entry('a'), entry('b', widths=(1., 1.), cpds=(1., 1.), index=1)
        before = copy.deepcopy([a, b])
        scales = dict(qor=.001, runtime=1., fit_train_only=True)
        scale_copy = dict(scales)
        result = build_preferences([a, b], scales=scales)
        self.assertEqual(result['pairs_by_condition'][a['condition_id']][0]['weight'], 5.)
        self.assertEqual(scales, scale_copy)
        self.assertEqual([a, b], before)
        with self.assertRaisesRegex(ValueError, 'positive finite'):
            build_preferences([a, b], scales=dict(qor=0, runtime=1))

    def test_frozen_actor_support_and_xy_only_difference(self):
        a, b = entry('a'), entry('b', index=1, xy_hash='updated', widths=(1., 1.))
        result = build_preferences([a, b], allowed_xy_sha256='frozen')
        self.assertEqual(result['total_pairs'], 0)
        self.assertEqual(result['audit'][1]['reason'], 'outside_frozen_xy_support')
        b = entry('b', widths=(1., 1.)); b['action']['xy'] = [[.4, .5]]
        result = build_preferences([a, b], allowed_xy_sha256='frozen')
        self.assertEqual(result['total_pairs'], 0)
        self.assertEqual(result['pair_counts']['same_layer_leaf_no_frozen_actor_signal'], 1)
        self.assertEqual(build_preferences([a, b])['total_pairs'], 1)

    def test_relative_payload_paths_and_collection_names(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_qor_preferences_') as directory:
            base = Path(directory)
            entries = []
            for i in range(2):
                path = base/f'{i}.json'; path.write_text(json.dumps(action(i)))
                row = entry('T', widths=(10.-i, 10.-i), index=i)
                row['action'] = path.name
                entries.append(row)
            result = build_preferences(entries, allowed_xy_sha256='frozen', base=base)
            self.assertEqual(result['total_pairs'], 1)

    def test_ties_and_all_failures_do_not_make_up_signal(self):
        result = build_preferences([entry('a'), entry('b', index=1)])
        self.assertEqual(result['total_pairs'], 0)
        self.assertEqual(result['scales']['qor'], 1.)
        self.assertTrue(result['scales']['zero_signal_fallback']['qor'])
        a = entry('a', statuses=('unroutable', 'unroutable'))
        b = entry('b', index=1, statuses=('unroutable', 'unroutable'))
        self.assertEqual(build_preferences([a, b])['total_pairs'], 0)


class EvaluationTests(unittest.TestCase):
    def contrast(self, report, method='T', baseline='B'):
        return next(r for r in report['comparisons'] if r['method'] == method and r['baseline'] == baseline)

    def test_seed_mean_before_ratio_not_geometric_seed_ratio(self):
        rows = [measured('B', seed=s, wl=4.) for s in (1, 2)]
        rows += [measured('T', seed=1, wl=1.), measured('T', seed=2, wl=9.)]
        c = self.contrast(evaluate(rows, methods=('B', 'T')))
        self.assertAlmostEqual(c['metrics']['wirelength']['ratio'], 5/4)
        self.assertEqual(c['classification'], 'average_joint_regression')

    def test_architectures_and_families_are_equal_weight(self):
        rows = []
        for condition, arch, ratio, count in [('a/one/x','one',2.,3), ('a/one/y','one',2.,1), ('a/two/z','two',.5,1)]:
            for seed in range(count):
                rows.extend([measured('B', condition=condition, architecture=arch, seed=seed),
                             measured('T', condition=condition, architecture=arch, seed=seed, wl=10*ratio)])
        c = self.contrast(evaluate(rows, methods=('B', 'T')))
        self.assertAlmostEqual(c['metrics']['wirelength']['ratio'], 1.)

    def test_slower_joint_qor_gain_is_a_gain(self):
        rows = [measured('B'), measured('T', wl=9., cpd=9., total=100.)]
        expected = {(r['condition_id'], r['seed'], r['method']) for r in rows}
        c = self.contrast(evaluate(rows, methods=('B', 'T'), expected=expected))
        self.assertTrue(c['full_population_joint_improvement_claim'])
        self.assertEqual(c['classification'], 'average_joint_improvement')
        self.assertAlmostEqual(c['metrics']['total_seconds']['ratio'], 10.)

    def test_tradeoff_is_not_joint_gain_even_when_combined_score_good(self):
        rows = [measured('B'), measured('T', wl=7., cpd=12.)]
        expected = {(r['condition_id'], r['seed'], r['method']) for r in rows}
        c = self.contrast(evaluate(rows, methods=('B', 'T'), expected=expected))
        self.assertLess(c['joint_q'], 0)
        self.assertEqual(c['classification'], 'qor_tradeoff')
        self.assertFalse(c['full_population_joint_improvement_claim'])

    def test_failed_seed_not_a_runtime_win_or_global_qor_claim(self):
        rows = [measured('B', seed=s) for s in (1, 2)]
        rows += [measured('T', seed=1, wl=9., cpd=9.),
                 measured('T', seed=2, total=.1, status='unroutable', wl=None, cpd=None)]
        expected = {(r['condition_id'], r['seed'], r['method']) for r in rows}
        c = self.contrast(evaluate(rows, methods=('B', 'T'), expected=expected))
        self.assertEqual(c['jointly_successful'], 1)
        self.assertIn('conditional', c['estimate_population'])
        self.assertFalse(c['full_population_joint_improvement_claim'])
        self.assertEqual(c['metrics']['total_seconds']['ratio'], 1.)

    def test_missing_task_and_missing_metric_do_not_look_complete(self):
        rows = [measured('B'), measured('T', wl=9., cpd=None)]
        expected = {(r['condition_id'], r['seed'], r['method']) for r in rows}
        expected.add((rows[0]['condition_id'], 2, 'T'))
        report = evaluate(rows, methods=('B', 'T'), expected=expected)
        self.assertFalse(report['complete'])
        self.assertEqual(len(report['missing']), 1)
        self.assertEqual(self.contrast(report)['classification'], 'insufficient_qor')

    def test_duplicate_and_unexpected_method_rejected(self):
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluate([measured('B'), measured('B')])
        with self.assertRaisesRegex(ValueError, 'unexpected'):
            evaluate([measured('other')])

    def test_missing_both_metrics_on_one_success_condition_is_not_global_claim(self):
        rows = [measured('B'), measured('T', wl=9., cpd=9.),
                measured('B', condition='b/H0Z0/w128', family='b'),
                measured('T', condition='b/H0Z0/w128', family='b', wl=None, cpd=None)]
        expected = {(r['condition_id'], r['seed'], r['method']) for r in rows}
        c = self.contrast(evaluate(rows, methods=('B','T'), expected=expected))
        self.assertTrue(c['consistent_qor_population'])
        self.assertFalse(c['full_qor_metric_coverage'])
        self.assertFalse(c['full_population_joint_improvement_claim'])


class SelectionTests(unittest.TestCase):
    def test_real_qor_selects_slower_checkpoint(self):
        rows = validation()
        for row in rows:
            if row['epoch'] == 10:
                row.update(wirelength=8., cpd_ns=9., total_seconds=100.)
        selection = choose_epoch(rows)
        self.assertEqual(selection['epoch'], 10)
        self.assertTrue(selection['performance_selection_completed'])
        self.assertFalse(selection['runtime_used_for_selection'])

    def test_success_coverage_precedes_common_population_qor(self):
        rows = validation()
        for row in rows:
            if row['epoch'] == 10:
                row.update(wirelength=1., cpd_ns=1.)
                if row['method'] == 'T' and row['seed'] == 1:
                    row['status'] = 'unroutable'
        selection = choose_epoch(rows)
        self.assertEqual(selection['epoch'], 5)
        self.assertEqual(selection['common_success_conditions'], 1)

    def test_missing_other_arm_and_duplicates_rejected(self):
        rows = validation()
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            choose_epoch([r for r in rows if not (r['method'] == 'R' and r['epoch'] == 10 and r['seed'] == 2)])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            choose_epoch(rows+[copy.deepcopy(rows[0])])

    def test_qor_tie_uses_earlier_epoch_not_runtime(self):
        rows = validation()
        for row in rows:
            if row['epoch'] == 5: row['total_seconds'] = 1000.
        self.assertEqual(choose_epoch(rows)['epoch'], 5)

    def test_no_common_success_reports_no_performance_selection(self):
        rows = validation()
        for row in rows:
            if row['method'] == 'B': row['status'] = 'unroutable'
        selection = choose_epoch(rows)
        self.assertFalse(selection['performance_selection_completed'])
        self.assertEqual(selection['epoch'], 5)

    def test_successful_task_without_qor_does_not_silently_select(self):
        rows = validation(); rows[0]['cpd_ns'] = None
        with self.assertRaisesRegex(ValueError, 'missing QoR metrics'):
            choose_epoch(rows)


if __name__ == '__main__':
    unittest.main()
