"""016 pairwise populations, complete failure coverage and frozen input contracts."""
import copy
from itertools import product
import unittest

from explainable_evaluation import evaluate, METHODS, SEEDS, ARCHES


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
        predictions['conditions'][cid] = dict(condition_id=cid, device_signature='a'*64, candidates=candidates)
        for seed, method in product(SEEDS, METHODS):
            factor = {'Default':1., 'D':1.02, 'J':1.01, 'E':.99}[method]
            rows.append(dict(condition_id=cid, family=c['family'], design=c['design'], architecture=arch,
                width=128, seed=seed, method=method, device_signature='a'*64, execution_signature='b'*64,
                status='success', input_consumed=method!='Default', native_solver_continued=method!='Default',
                vpr_manifest=f'{cid}/{seed}/{method}/manifest.json', wirelength=100*factor, cpd_ns=10*factor,
                unique_chanz_nodes=0, vpr_seconds=5.,
                **{k:candidates.get(method,{}).get(k) for k in ('ap_init_sha256','deployed_action_sha256')}))
    return rows, conditions, predictions


class ExplainableEvaluationTests(unittest.TestCase):
    def test_pairwise_comparisons_and_zero_vertical_usage(self):
        rows, cond, pred = fixture()
        r = evaluate(rows, cond, pred, bootstrap_samples=0)
        self.assertEqual(r['logical_native_trials'], 288)
        self.assertAlmostEqual(r['comparisons']['E/Default']['metrics']['wirelength']['percent_change'], -1.)
        self.assertAlmostEqual(r['comparisons']['D/Default']['metrics']['cpd_ns']['percent_change'], 2.)
        self.assertIsNone(r['comparisons']['E/Default']['metrics']['final_unique_chanz_nodes']['ratio'])
        self.assertEqual(r['refinement_stop_reasons'], {'no_improvement':24})

    def test_new_algorithm_failure_never_filters_standalone_gnn_baseline(self):
        rows, cond, pred = fixture()
        row = next(r for r in rows if r['method']=='E')
        row.update(status='unroutable',wirelength=None,cpd_ns=None,unique_chanz_nodes=None)
        result = evaluate(rows,cond,pred,bootstrap_samples=0)
        self.assertEqual(result['comparisons']['D/Default']['common_success_seed_pairs'],72)
        self.assertEqual(result['comparisons']['E/Default']['common_success_seed_pairs'],71)
        self.assertEqual(result['all_four_common_success_comparisons']['D/Default']['common_success_seed_pairs'],71)
        self.assertEqual(result['success_by_method']['E']['planned'],72)
        self.assertEqual(result['success_by_method']['E']['successes'],71)

    def test_missing_trials_wrong_seed_consumption_and_infra_failure_rejected(self):
        rows, cond, pred = fixture()
        for mutation in ('missing','seed','init','failure','device','duplicate_manifest'):
            changed = copy.deepcopy(rows)
            if mutation=='missing': changed.pop()
            elif mutation=='seed': changed[-1]['seed']=15101
            elif mutation=='init': changed[-1]['ap_init_sha256']='f'*64
            elif mutation=='failure': changed[-1]['status']='timeout'
            elif mutation=='device': changed[-1]['device_signature']='f'*64
            else: changed[-1]['vpr_manifest']=changed[-2]['vpr_manifest']
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                evaluate(changed,cond,pred,bootstrap_samples=0)

    def test_complete_architecture_blocks_required(self):
        rows, cond, pred = fixture()
        with self.assertRaises(ValueError):
            evaluate(rows,cond[:-1],pred,bootstrap_samples=0)

    def test_proxy_gain_is_joined_to_physical_qor_without_claiming_equivalence(self):
        rows, cond, pred = fixture()
        cid=cond[0]['id']
        d=pred['conditions'][cid]['candidates']['E']['diagnostics']
        d.update(initial_objective=100.,final_objective=90.,full_net_coverage=True)
        result=evaluate(rows,cond,pred,bootstrap_samples=0)
        evidence=result['proxy_to_qor_by_condition'][cid]
        self.assertEqual(evidence['proxy_gain'],10.)
        self.assertEqual(evidence['common_success_seeds'],list(SEEDS))
        self.assertAlmostEqual(evidence['routed_WL_percent_change'],100*(.99/1.02-1))
        d['final_objective']=101.
        with self.assertRaisesRegex(ValueError,'increasing objective'):
            evaluate(rows,cond,pred,bootstrap_samples=0)


if __name__=='__main__':
    unittest.main()
