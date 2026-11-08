"""017 pairwise populations, complete failure coverage and frozen input contracts."""
import copy
from itertools import product
import unittest

from factorial_evaluation import evaluate, METHODS, SEEDS, ARCHES


def fixture():
    conditions, rows = [], []
    predictions = {'conditions': {}}
    for f, arch in product(range(6), ARCHES):
        cid = f'f{f}/{arch}/w128'
        c = dict(id=cid, family=f'f{f}', design=f'd{f}', architecture=arch, width=128,
                 training_exposure='development', split='validation')
        conditions.append(c)
        candidates = {m:dict(ap_init_sha256=str(i)*64, deployed_action_sha256=str(i+3)*64,
                     diagnostics=dict(stop_reason='no_improvement')) for i,m in enumerate(METHODS[1:],1)}
        for arm,mode,swaps in (('W','wl',False),('WS','wl',True),('T','joint',False),('TS','joint',True)):
            candidates[arm]['diagnostics'].update(setup_complete=True,objective_mode=mode,pair_swaps=swaps,
                normalizers=dict(wirelength=100.,timing=1.))
        predictions['conditions'][cid] = dict(condition_id=cid, device_signature='a'*64, candidates=candidates)
        for seed, method in product(SEEDS, METHODS):
            factor = {'Default':1., 'D':1.03, 'W':1.02, 'WS':1.01, 'T':1., 'TS':.97}[method]
            rows.append(dict(condition_id=cid, family=c['family'], design=c['design'], architecture=arch,
                width=128, seed=seed, method=method, device_signature='a'*64, execution_signature='b'*64,
                status='success', input_consumed=method!='Default', native_solver_continued=method!='Default',
                vpr_manifest=f'{cid}/{seed}/{method}/manifest.json', wirelength=100*factor, cpd_ns=10*factor,
                unique_chanz_nodes=0, vpr_seconds=5.,
                **{k:candidates.get(method,{}).get(k) for k in ('ap_init_sha256','deployed_action_sha256')}))
    return rows, conditions, predictions


class FactorialEvaluationTests(unittest.TestCase):
    def test_factorial_interaction_uses_correct_four_arms(self):
        rows,c,p=fixture()
        r=evaluate(rows,c,p,bootstrap_samples=0)
        self.assertEqual(r['logical_native_trials'],432)
        self.assertAlmostEqual(r['comparisons']['TS/Default']['metrics']['wirelength']['percent_change'],-3.)
        self.assertTrue(r['joint_outcomes']['TS/Default']['average_both_improve'])
        self.assertEqual(r['joint_outcomes']['TS/Default']['case_counts'],{'both_improve':24})
        self.assertAlmostEqual(r['interaction']['metrics']['wirelength']['ratio'],.97*1.02/(1.*1.01))
        self.assertEqual(r['factors']['swaps'],{'W':False,'WS':True,'T':False,'TS':True})

    def test_one_metric_improving_does_not_count_as_a_joint_gain(self):
        rows,c,p=fixture()
        for row in rows:
            if row['method']=='TS':
                row.update(wirelength=90.,cpd_ns=10.1)
        r=evaluate(rows,c,p,bootstrap_samples=0)
        self.assertFalse(r['joint_outcomes']['TS/Default']['average_both_improve'])
        self.assertEqual(r['joint_outcomes']['TS/Default']['case_counts'],{'tradeoff_or_one_equal':24})

    def test_factor_failure_does_not_filter_default_d_population(self):
        rows,c,p=fixture()
        next(r for r in rows if r['method']=='TS').update(status='unroutable',wirelength=None,cpd_ns=None,unique_chanz_nodes=None)
        r=evaluate(rows,c,p,bootstrap_samples=0)
        self.assertEqual(r['comparisons']['D/Default']['common_success_seed_pairs'],72)
        self.assertEqual(r['comparisons']['TS/Default']['common_success_seed_pairs'],71)
        self.assertEqual(r['factorial_common_success_comparisons']['T/W']['common_success_seed_pairs'],71)
        self.assertEqual(r['success_by_method']['TS']['planned'],72)
        self.assertEqual(r['success_by_method']['TS']['successes'],71)

    def test_incomplete_matrix_invalid_digest_infra_and_shared_manifest_rejected(self):
        rows,c,p=fixture()
        for mutation in ('missing','digest','infra','manifest','seed'):
            changed=copy.deepcopy(rows)
            if mutation=='missing':changed.pop()
            elif mutation=='digest':changed[-1]['ap_init_sha256']='a'*64
            elif mutation=='infra':changed[-1]['status']='timeout'
            elif mutation=='seed':changed[-1]['seed']=16101
            else:changed[-1]['vpr_manifest']=changed[-2]['vpr_manifest']
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):
                evaluate(changed,c,p,bootstrap_samples=0)

    def test_proxy_and_terminal_qor_are_joined_without_equating_them(self):
        rows,c,p=fixture();cid=c[0]['id']
        p['conditions'][cid]['candidates']['TS']['diagnostics'].update(
            initial_objective=1.,final_objective=.95,initial_wirelength=100.,final_wirelength=90.,
            initial_timing=2.,final_timing=2.1)
        r=evaluate(rows,c,p,bootstrap_samples=0)
        d=r['proxy_to_qor_by_condition'][cid]['TS']
        self.assertAlmostEqual(d['proxy_gain'],.05)
        self.assertEqual(d['final_timing'],2.1)
        self.assertAlmostEqual(d['routed_CPD_percent_change'],100*(.97/1.03-1))

    def test_wrong_factor_missing_setup_or_normalization_is_rejected(self):
        rows,c,p=fixture();cid=c[0]['id']
        for key,value in (('setup_complete',False),('objective_mode','wl'),('pair_swaps',False),
                          ('normalizers',dict(wirelength=101.,timing=1.))):
            bad=copy.deepcopy(p)
            bad['conditions'][cid]['candidates']['TS']['diagnostics'][key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):
                evaluate(rows,c,bad,bootstrap_samples=0)


if __name__=='__main__':
    unittest.main()

