"""Regression checks for 014's prospective attribution and cross-seed scoring."""
from copy import deepcopy
import hashlib
from itertools import product
import math
import unittest

from prospective_evaluation import ARCHES, CANDIDATES, NATIVE, SCORERS, SEEDS, evaluate


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def fixture():
    conditions, rows, predictions = [], [], {'conditions': {}}
    for family, arch in product(('alpha','beta'), ARCHES):
        cid = f'{family}/{arch}/w128'
        c = dict(id=cid, family=family, design=family, architecture=arch, width=128)
        conditions.append(c)
        candidates = {m: dict(ap_init_sha256=digest(cid+'/input/'+m),
                              deployed_action_sha256=digest(cid+'/action/'+m)) for m in CANDIDATES}
        pred = dict(condition_id=cid, candidates=candidates, device_signature=digest(cid+'/device'),
                    generation_seconds=10., prediction_worker_internal_seconds=11.5,
                    generation_seconds_excludes_scorer_times=True, choices={})
        for name, selected in zip(SCORERS, ('j_s1','D','j_s2')):
            pred['choices'][name] = dict(method=selected, **candidates[selected], scoring_seconds=.5,
                candidate_ids=list(CANDIDATES), scores=[float(m==selected) for m in CANDIDATES])
        predictions['conditions'][cid] = pred
        for seed, method in product(SEEDS, NATIVE):
            i = SEEDS.index(seed)
            # j_s1 wins on the first two seeds. On the third it is worse than D.
            wl = (5., 20., 120.)[i] if method == 'j_s1' else (10., 40., 60.)[i]
            if method == 'j_s2':
                wl *= 2.
            if method == 'j_s1' and family == 'beta':
                wl *= .75
            rows.append(dict(condition_id=cid, family=family, design=family, architecture=arch,
                width=128, method=method, seed=seed, status='success', wirelength=wl, cpd_ns=2.,
                unique_chanz_nodes=0 if method=='Default' else 3, vpr_seconds=1.2,
                input_consumed=method != 'Default', native_solver_continued=method != 'Default',
                device_signature=pred['device_signature'], execution_signature=digest(cid+str(seed)),
                vpr_manifest=f'{cid}/{seed}/{method}/manifest.json',
                ap_init_sha256=candidates.get(method, {}).get('ap_init_sha256'),
                deployed_action_sha256=candidates.get(method, {}).get('deployed_action_sha256')))
    return rows, conditions, predictions


class ProspectiveEvaluationTests(unittest.TestCase):
    def test_precommitted_choices_use_native_rows_and_ratio_of_common_seed_means(self):
        rows, conditions, pred = fixture()
        sample = conditions[0]['id']
        report = evaluate(rows, conditions, pred)
        result = report['comparisons']['E012/D']
        alpha = 145/110
        self.assertAlmostEqual(result['metrics']['wirelength']['family_ratios']['alpha'], alpha)
        self.assertAlmostEqual(result['metrics']['wirelength']['ratio'], math.sqrt(alpha*alpha*.75))
        self.assertEqual(report['comparisons']['E012/j_s1']['metrics']['wirelength']['ratio'], 1.)
        self.assertEqual(report['comparisons']['G012/D']['metrics']['cpd_ns']['ratio'], 1.)
        self.assertEqual(report['comparisons']['E013/D']['metrics']['wirelength']['ratio'], 2.)
        self.assertEqual((result['planned_seed_pairs'], result['jointly_successful']), (24,24))
        zero = report['comparisons']['E012/Default']['metrics']['final_unique_chanz_nodes']
        self.assertIsNone(zero['ratio'])
        self.assertEqual(zero['conditions'][sample]['baseline_seed_mean'], 0.)
        self.assertEqual(report['measured_costs'][sample]['scoring_seconds']['E012'], .5)
        self.assertNotIn('total_seconds', result['metrics'])
        self.assertEqual(report['unique_native_manifests'], 96)

    def test_loso_uses_only_other_seeds_even_when_third_seed_reverses_winner(self):
        rows, conditions, pred = fixture()
        cid = conditions[0]['id']
        report = evaluate(rows, conditions, pred)
        self.assertEqual(report['pool_oracle'][cid]['method'], 'D')
        leave = report['leave_one_seed_out'][cid][str(SEEDS[2])]
        self.assertEqual(leave['selection_seeds'], list(SEEDS[:2]))
        self.assertEqual(leave['method'], 'j_s1')
        modified = deepcopy(rows)
        for r in modified:
            if (r['condition_id'], r['seed'], r['method']) == (cid, SEEDS[2], 'j_s1'):
                r['wirelength'] = 1.
        new = evaluate(modified, conditions, pred)
        self.assertEqual(new['leave_one_seed_out'][cid][str(SEEDS[2])], leave)
        self.assertEqual(new['pool_oracle'][cid]['method'], 'j_s1')
        self.assertNotEqual(new['comparisons']['leave_seed_out/D']['metrics']['wirelength']['ratio'],
                            report['comparisons']['leave_seed_out/D']['metrics']['wirelength']['ratio'])

    def test_failures_keep_denominators_use_common_seeds_and_success_first(self):
        rows, conditions, pred = fixture()
        cid = conditions[0]['id']
        for r in rows:
            if r['condition_id'] == cid and r['method'] == 'D' and r['seed'] != SEEDS[0]:
                r.update(status='unroutable', wirelength=None, cpd_ns=None)
            if r['condition_id'] == cid and r['method'] == 'j_s2':
                r.update(status='packing_or_placement_failure', confirmed_candidate_infeasibility=True,
                         wirelength=None, cpd_ns=None)
        report = evaluate(rows, conditions, pred)
        a = report['comparisons']['E012/D']
        self.assertEqual((a['method_successes'], a['baseline_successes'], a['jointly_successful']), (24,22,22))
        self.assertEqual(a['metrics']['wirelength']['conditions'][cid]['common_success_seeds'], [SEEDS[0]])
        self.assertEqual(report['pool_oracle'][cid]['method'],'j_s1')
        self.assertIn(cid, report['comparisons']['E013/D']['conditions_without_common_success'])
        # No common success among the tied top-count candidates in the two selection seeds.
        for r in rows:
            if r['condition_id'] == cid and r['method'] == 'j_s1' and r['seed'] == SEEDS[0]:
                r.update(status='unroutable', wirelength=None, cpd_ns=None)
        record = evaluate(rows, conditions, pred)['leave_one_seed_out'][cid][str(SEEDS[2])]
        self.assertTrue(record['no_common_qor'])
        self.assertEqual(record['method'], 'D')

    def test_missing_unmatched_tampered_or_infrastructure_results_block_attribution(self):
        rows, conditions, pred = fixture()
        with self.assertRaisesRegex(ValueError, 'incomplete native'):
            evaluate(rows[:-1], conditions, pred)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            evaluate(rows+[rows[0]], conditions, pred)
        for change, expected in [
                ({'execution_signature':'0'*64},'backend/netlist'),
                ({'ap_init_sha256':'0'*64},'initializer'),
                ({'device_signature':'0'*64},'device'),
                ({'status':'timeout'},'infrastructure'),
                ({'status':'packing_or_placement_failure'},'infrastructure'),
                ({'native_solver_continued':False},'consumer evidence'),
                ({'vpr_manifest':rows[0]['vpr_manifest']},'shared native manifest')]:
            with self.subTest(change=change):
                copied = deepcopy(rows)
                copied[1].update(change)
                with self.assertRaisesRegex(ValueError, expected):
                    evaluate(copied, conditions, pred)
        copied = deepcopy(pred)
        cid = conditions[0]['id']
        copied['conditions'][cid]['choices']['E012']['scores'] = [2., 1., 0.]
        with self.assertRaisesRegex(ValueError, 'scores'):
            evaluate(rows, conditions, copied)

    def test_deduplicated_inputs_share_qor_without_counting_extra_native_trials(self):
        rows, conditions, pred = fixture()
        cid = conditions[0]['id']
        # Distinct logical candidate labels can lower to identical input bytes.
        target = pred['conditions'][cid]
        target['candidates']['j_s2']['ap_init_sha256'] = target['candidates']['j_s1']['ap_init_sha256']
        target['choices']['E013']['ap_init_sha256'] = target['candidates']['j_s2']['ap_init_sha256']
        for strong in rows:
            if (strong['condition_id'],strong['method']) == (cid,'j_s2'):
                weak = next(r for r in rows if (r['condition_id'],r['seed'],r['method'])==(cid,strong['seed'],'j_s1'))
                for key in ('vpr_manifest','ap_init_sha256','wirelength','cpd_ns','vpr_seconds','unique_chanz_nodes'):
                    strong[key] = weak[key]
        report = evaluate(rows, conditions, pred)
        self.assertEqual((report['logical_native_trials'], report['unique_native_manifests']),(96,93))


if __name__ == '__main__':
    unittest.main()
