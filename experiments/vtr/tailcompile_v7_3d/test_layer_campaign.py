"""Registered budget, mean-positive estimates and checkpoint selection contracts."""
import copy
import unittest
from layer_campaign import queue_plan, evaluate, choose_epoch, METHODS


def row(family, method, seconds, seed=1, status='success', epoch=0):
    return dict(condition_id=family+'/H0Z0/w64', family=family, architecture='H0Z0',
                width_role='reference', seed=seed, method=method, status=status,
                total_seconds=seconds, cpd_ns=1., wirelength=100., epoch=epoch)


class LayerCampaignTests(unittest.TestCase):
    def test_registered_budget_and_arms(self):
        p = queue_plan()
        self.assertEqual(len(METHODS), 7)
        self.assertEqual(p['total_runs_upper'], 12+160+80+224)
        self.assertFalse(p['per_case_qor_veto'])

    def test_average_gain_allows_a_regression(self):
        rows = [row('a', 'B', 100), row('a', 'T-S', 80), row('b', 'B', 100), row('b', 'T-S', 110)]
        expected = [(r['condition_id'], r['seed'], r['method']) for r in rows]
        report = evaluate(rows, methods=('B', 'T-S'), expected=expected)
        r = report['comparisons'][0]
        self.assertAlmostEqual(r['metrics']['total_seconds']['ratio'], (0.8*1.1)**.5)
        self.assertTrue(r['full_population_runtime_gain_claim'])
        self.assertEqual(r['runtime_losses'], 1)

    def test_fast_failure_does_not_count_as_speedup(self):
        rows = [row('a', 'B', 100), row('a', 'T-S', 1, status='unroutable')]
        r = evaluate(rows, methods=('B', 'T-S'))['comparisons'][0]
        self.assertIsNone(r['metrics']['total_seconds']['ratio'])
        self.assertFalse(r['full_population_runtime_gain_claim'])

    def test_missing_tasks_cannot_be_called_full_population_gain(self):
        rows = [row('a', 'B', 100), row('a', 'T-S', 80)]
        expected = [(r['condition_id'], 1, r['method']) for r in rows]+[('missing', 1, 'B')]
        report = evaluate(rows, methods=('B', 'T-S'), expected=expected)
        self.assertFalse(report['complete'])
        self.assertFalse(report['comparisons'][0]['full_population_runtime_gain_claim'])

    def test_epoch_is_shared_and_real_time_not_teacher_loss(self):
        rows = [row('a', 'B', 100)]
        for e in (5, 10):
            for m in ('L', 'T'):
                rows.append(row('a', m, 90 if e == 5 else 80, epoch=e))
        self.assertEqual(choose_epoch(rows)['epoch'], 10)
        broken = copy.deepcopy(rows)
        broken[-1]['status'] = 'unroutable'
        self.assertFalse(choose_epoch(broken)['performance_selection_completed'])
        self.assertEqual(choose_epoch(broken)['epoch'], 5)


if __name__ == '__main__':
    unittest.main()
