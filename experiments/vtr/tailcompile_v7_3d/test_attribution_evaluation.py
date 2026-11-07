"""Scientific pairing and family weighting checks for the 015 attribution."""
from copy import deepcopy
import hashlib
from itertools import product
import json
import math
import unittest

from attribution_evaluation import ARCHES, CANDIDATES, METHODS, SEEDS, evaluate


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def fixture(families=6):
    conditions, rows, predictions = [], [], {'conditions':{}}
    factors = dict(Default=1.1, D=1., L=.95, Y=.9, J=.8, R1=.99, R2=1.02)
    for index, arch in product(range(families), ARCHES):
        family = 'family'+str(index)
        cid = f'{family}/{arch}/w128'
        condition = dict(id=cid, family=family, design=family, architecture=arch, width=128,
            split='train' if index < 2 else 'validation',
            training_exposure='training_exposed' if index < 2 else 'heldout_development')
        conditions.append(condition)
        candidates = {m:dict(ap_init_sha256=digest(cid+'/input/'+m),
                            deployed_action_sha256=digest(cid+'/action/'+m),
                            diagnostics=dict(matched_control=True)) for m in CANDIDATES}
        pred = dict(condition_id=cid, device_signature=digest(cid+'/device'), candidates=candidates)
        predictions['conditions'][cid] = pred
        for seed, method in product(SEEDS, METHODS):
            value = 100.*(SEEDS.index(seed)+1)*(index+1)
            row = dict(condition_id=cid, family=family, design=family, architecture=arch,
                width=128, method=method, seed=seed, status='success',
                wirelength=value*factors[method], cpd_ns=10.*factors[method],
                unique_chanz_nodes=0 if method=='Default' else 3,
                final_route_usage=dict(unique_chanz_nodes=0 if method=='Default' else 3), vpr_seconds=1.,
                input_consumed=method!='Default', native_solver_continued=method!='Default',
                device_signature=pred['device_signature'], execution_signature=digest(cid+str(seed)),
                vpr_manifest=f'{cid}/{seed}/{method}/manifest.json',
                ap_init_sha256=candidates.get(method,{}).get('ap_init_sha256'),
                deployed_action_sha256=candidates.get(method,{}).get('deployed_action_sha256'))
            rows.append(row)
    return rows, conditions, predictions


def review(rows, conditions, pred, **kw):
    return evaluate(rows, conditions, pred, bootstrap_samples=32, **kw)


class AttributionEvaluationTests(unittest.TestCase):
    def test_factorial_ratios_interaction_and_strata(self):
        rows, conditions, pred = fixture()
        report = review(rows, conditions, pred)
        self.assertEqual(report['logical_native_trials'],504)
        self.assertEqual(report['unique_native_manifests'],504)
        self.assertEqual(report['independent_family_count'],6)
        ratio = report['comparisons']['J/D']['metrics']['wirelength']
        self.assertAlmostEqual(ratio['ratio'], .8)
        self.assertEqual(ratio['effective_family_count'],6)
        self.assertEqual(ratio['training_exposure_strata']['training_exposed']['effective_family_count'],2)
        self.assertEqual(ratio['training_exposure_strata']['heldout_development']['effective_family_count'],4)
        self.assertAlmostEqual(report['factorial_interaction']['metrics']['wirelength']['ratio'],.8/(.95*.9))
        self.assertAlmostEqual(report['comparisons']['J/Default']['metrics']['wirelength']['ratio'],.8/1.1)
        # Zero baseline usage has a meaningful absolute difference but no usage ratio.
        z = report['comparisons']['J/Default']['metrics']['final_unique_chanz_nodes']
        self.assertIsNone(z['ratio'])
        self.assertEqual(z['mean_absolute_difference'],3.)
        self.assertFalse(report['Wmin_measured'])
        self.assertFalse(report['final_blind_test'])
        json.dumps(report, allow_nan=False)

    def test_all_factorial_methods_use_the_same_success_seeds(self):
        rows, conditions, pred = fixture()
        cid = conditions[0]['id']
        for row in rows:
            if row['condition_id']==cid and row['seed']==SEEDS[2]:
                if row['method']=='L':
                    row.update(status='unroutable', wirelength=None, cpd_ns=None)
                elif row['method']=='J':
                    row['wirelength'] *= 100.
        report = review(rows, conditions, pred)
        a = report['comparisons']['J/D']
        b = report['comparisons']['Y/D']
        for result in (a,b):
            self.assertEqual(result['metrics']['wirelength']['conditions'][cid]['common_success_seeds'],list(SEEDS[:2]))
        self.assertAlmostEqual(a['metrics']['wirelength']['ratio'],.8)
        self.assertEqual(a['planned_seed_pairs'],72)
        self.assertEqual(a['common_success_seed_pairs'],71)
        self.assertEqual(report['success_by_method']['L']['status_counts']['unroutable'],1)

    def test_confirmed_scientific_failures_remain_and_partial_families_are_marked(self):
        rows, conditions, pred = fixture()
        cid = conditions[0]['id']
        for row in rows:
            if row['condition_id']==cid and row['method']=='J':
                row.update(status='packing_or_placement_failure', confirmed_candidate_infeasibility=True,
                           wirelength=None, cpd_ns=None)
        report = review(rows, conditions, pred)
        self.assertEqual(report['success_by_method']['J']['planned'],72)
        self.assertEqual(report['success_by_method']['J']['successes'],69)
        result = report['comparisons']['J/D']
        self.assertIn(cid,result['conditions_without_common_success'])
        self.assertEqual(result['metrics']['wirelength']['effective_family_count'],5)
        self.assertEqual(result['metrics']['wirelength']['incomplete_architecture_families'],['family0'])

    def test_unmatched_random_proposal_is_not_a_negative_control_result(self):
        rows, conditions, pred = fixture()
        cid = conditions[0]['id']
        pred['conditions'][cid]['candidates']['R1']['diagnostics']['matched_control']=False
        for row in rows:
            if row['condition_id']==cid and row['method']=='R1':
                row['wirelength'] *= 100.
        report = review(rows, conditions, pred)
        random = report['comparisons']['J/R1']
        self.assertEqual(random['unmatched_random_control_conditions'],[cid])
        self.assertNotIn(cid,random['metrics']['wirelength']['conditions'])
        self.assertAlmostEqual(random['metrics']['wirelength']['ratio'],.8/.99)
        self.assertEqual(random['metrics']['wirelength']['effective_family_count'],5)
        self.assertEqual(report['comparisons']['J/R2']['metrics']['wirelength']['effective_family_count'],6)

    def test_design_size_does_not_determine_weight_and_family_sensitivity_is_visible(self):
        rows, conditions, pred = fixture()
        for row in rows:
            if row['method']=='J':
                row['wirelength'] = row['wirelength']/.8*(.5 if row['family']=='family0' else 1.)
            # Make the one improving family large. It must still count only once.
            if row['family']=='family0':
                row['wirelength'] *= 10000.
        ratio = review(rows,conditions,pred)['comparisons']['J/D']['metrics']['wirelength']
        self.assertAlmostEqual(ratio['ratio'],.5**(1/6))
        self.assertAlmostEqual(ratio['leave_one_family_out']['family0'],1.)
        self.assertAlmostEqual(ratio['leave_one_family_out']['family1'],.5**(1/5))
        self.assertEqual(ratio['family_bootstrap_95']['unit'],'independent family')
        self.assertEqual(ratio['family_bootstrap_95'],review(rows,conditions,pred)['comparisons']['J/D']['metrics']['wirelength']['family_bootstrap_95'])

    def test_matrix_and_frozen_signatures_must_match(self):
        rows, conditions, pred = fixture()
        for altered, text in ((rows[:-1],'incomplete native'),(rows+[rows[0]],'duplicate')):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError,text):
                review(altered,conditions,pred)
        for changes, text in [
            ({'status':'timeout'},'infrastructure'),
            ({'status':'packing_or_placement_failure'},'infrastructure'),
            ({'ap_init_sha256':'0'*64},'initializer'),
            ({'device_signature':'0'*64},'device'),
            ({'execution_signature':'0'*64},'execution'),
            ({'native_solver_continued':False},'consumer'),
            ({'wirelength':float('nan')},'positive WL'),
            ({'vpr_manifest':rows[0]['vpr_manifest']},'shared native manifest')]:
            altered=deepcopy(rows)
            altered[1].update(changes)
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError,text):
                review(altered,conditions,pred)

    def test_independent_family_and_exposure_checks(self):
        rows, conditions, pred=fixture(2)
        with self.assertRaisesRegex(ValueError,'independent families'):
            review(rows,conditions,pred)
        self.assertEqual(review(rows,conditions,pred,minimum_families=2)['independent_family_count'],2)
        rows, conditions, pred=fixture()
        altered=deepcopy(conditions)
        for c in altered:
            if c['family']=='family1':
                c['design']='family0'
        with self.assertRaisesRegex(ValueError,'same design'):
            review(rows,altered,pred)
        altered=deepcopy(conditions)
        altered[0]['training_exposure']='heldout_development'
        with self.assertRaisesRegex(ValueError,'leakage'):
            review(rows,altered,pred)
        altered=deepcopy(conditions)
        for c in altered:
            c['record']={'mapped_blif_sha256':digest('one-netlist')}
        with self.assertRaisesRegex(ValueError,'duplicated'):
            review(rows,altered,pred)

    def test_aliases_require_identical_native_input_and_result(self):
        rows, conditions, pred=fixture()
        cid=conditions[0]['id']
        target=pred['conditions'][cid]
        target['candidates']['L']['ap_init_sha256']=target['candidates']['D']['ap_init_sha256']
        for row in rows:
            if (row['condition_id'],row['method'])==(cid,'L'):
                original=next(r for r in rows if (r['condition_id'],r['seed'],r['method'])==(cid,row['seed'],'D'))
                for k in ('ap_init_sha256','vpr_manifest','wirelength','cpd_ns'):
                    row[k]=original[k]
        self.assertEqual(review(rows,conditions,pred)['unique_native_manifests'],501)
        altered=deepcopy(rows)
        next(r for r in altered if r['condition_id']==cid and r['method']=='L')['wirelength']+=1.
        with self.assertRaisesRegex(ValueError,'shared native manifest'):
            review(altered,conditions,pred)

    def test_registered_panel_metadata_is_compatible_without_loading_server_payloads(self):
        # Real, hash-checked archived metadata; explicitly synthetic QoR below.
        # This boundary test catches field/schema drift between loader and report.
        from attribution_panel import load_panel
        panel=load_panel(verify_payload=False)
        rows,conditions,pred=fixture(len(panel['designs']))
        targets=dict(zip((c['id'] for c in conditions),panel['conditions']))
        remapped={}
        for cid,p in pred['conditions'].items():
            target=targets[cid]
            p['condition_id']=target['id']
            remapped[target['id']]=p
        for row in rows:
            target=targets[row['condition_id']]
            row.update(condition_id=target['id'],**{k:target[k] for k in ('family','design','architecture','width')})
            row['vpr_manifest']=f'synthetic/{target["id"]}/{row["seed"]}/{row["method"]}/manifest.json'
        report=review(rows,panel['conditions'],dict(conditions=remapped))
        strata=report['comparisons']['J/D']['metrics']['wirelength']['training_exposure_strata']
        self.assertEqual(set(strata),{c['exposure'] for c in panel['conditions']})
        self.assertNotIn('unspecified',strata)
        self.assertEqual(sum(v['effective_family_count'] for v in strata.values()),6)
        self.assertAlmostEqual(report['comparisons']['J/D']['metrics']['wirelength']['ratio'],.8)

    def test_mechanism_uses_native_atom_denominators_and_preserves_missing_evidence(self):
        rows,conditions,pred=fixture()
        selected=[r for r in rows if r['family']=='family0' and r['method']=='J']
        selected[0]['residual_observations']=dict(edited_atoms=10,stages=dict(
            post_gp=dict(status='observed',matched_atoms=10,matched_layer_edited_atoms=2,
                layer_target_retained_fraction=1.,mean_normalized_xy_l1_to_target=.2,
                mean_normalized_xy_l1_to_base=.4)))
        selected[1]['residual_observations']=dict(edited_atoms=30,stages=dict(
            post_gp=dict(status='partial_coverage',matched_atoms=20,matched_layer_edited_atoms=8,
                layer_target_retained_fraction=.25,mean_normalized_xy_l1_to_target=.5,
                mean_normalized_xy_l1_to_base=None)))
        selected[2]['residual_observations']=dict(edited_atoms=5,stages=dict(
            post_gp=dict(status='missing',matched_atoms=None,layer_target_retained_fraction=None)))
        selected[3]['residual_observations']=dict(edited_atoms=0,stages=dict(
            post_gp=dict(status='no_edits',matched_atoms=0,layer_target_retained_fraction=None)))
        mechanism=review(rows,conditions,pred)['mechanism_observations']
        summary=mechanism['by_method']['J']['by_family']['family0']['stages']['post_gp']
        self.assertEqual(summary['matched_atoms'],30)
        self.assertEqual(summary['matched_layer_edited_atoms'],10)
        self.assertAlmostEqual(summary['layer_target_retained_fraction'],.4)
        self.assertAlmostEqual(summary['mean_normalized_xy_l1_to_target'],.4)
        self.assertAlmostEqual(summary['mean_normalized_xy_l1_to_base'],.4)
        self.assertEqual(summary['metric_denominators'],dict(layer_target_retained_fraction=10,
            mean_normalized_xy_l1_to_target=30,mean_normalized_xy_l1_to_base=10))
        self.assertEqual(summary['missing_metric_atoms']['mean_normalized_xy_l1_to_base'],20)
        self.assertEqual(summary['status_counts'],dict(observed=1,partial_coverage=1,missing=1,no_edits=1,missing_observation=8))
        final=mechanism['by_method']['J']['by_family']['family0']['stages']['final']
        self.assertEqual(final['status_counts']['missing_stage'],4)
        self.assertIsNone(final['matched_atoms'])
        self.assertIsNone(final['layer_target_retained_fraction'])
        self.assertTrue(mechanism['atom_observations_are_not_independent_samples'])
        self.assertTrue(mechanism['retention_is_not_a_success_metric'])


if __name__=='__main__':
    unittest.main()
