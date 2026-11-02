"""011 matrix, validation-only replacement gate and frozen-source identities."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import residual_campaign as campaign
from io_utils import read, write, sha


def row(cid, method, *, epoch=0, seed=11201, wl=100., cp=10., status='success'):
    family,arch=cid.split('/')[:2]
    return dict(condition_id=cid,method=method,epoch=epoch,seed=seed,family=family,architecture=arch,
        wirelength=wl,cpd_ns=cp,status=status,total_seconds=10.,vpr_seconds=9.)


def validation():
    rows=[]
    for family in ('blob_merge','diffeq'):
        for arch in ('H0Z0','H0Z1','H1Z0','H1Z1'):
            cid=f'{family}/{arch}/w128'
            for seed in campaign.VALIDATION_SEEDS:
                rows.append(row(cid,'D',seed=seed))
                for arm in campaign.ARMS:
                    for epoch in campaign.EPOCHS:
                        rows.append(row(cid,arm,epoch=epoch,seed=seed,wl=95. if epoch==2 else 97.))
    return rows


class ResidualCampaignTests(unittest.TestCase):
    def test_budget_and_frozen_backbone_objective(self):
        p=campaign.queue_plan()
        self.assertEqual(p['research_runs_upper'],304)
        self.assertEqual(p['feedback_runs_upper']+p['validation_runs']+p['comparison_runs'],304)
        self.assertEqual(p['methods'],['B','D','L','Y','J'])
        self.assertFalse(p['backbone_trainable']);self.assertFalse(p['teacher_targets_used'])
        self.assertFalse(p['projected_rewards_reused']);self.assertEqual(p['tail_weight'],0)
        self.assertEqual(p['backend_layer_prior'],0);self.assertEqual(p['new_backend_builds'],0)
        for first,second in ((campaign.FEEDBACK_SEEDS,campaign.TEST_SEEDS),(campaign.VALIDATION_SEEDS,campaign.TEST_SEEDS)):
            self.assertFalse(set(first)&set(second))

    def test_namespace_refuses_all_old_cohorts(self):
        for name in ('development_010_structured','development_009_qor','development_008_aligned'):
            with self.assertRaises(ValueError):campaign.owned(campaign.HERE/'runs'/name)
        self.assertEqual(campaign.owned(campaign.HERE/'runs/development_011_test'),campaign.HERE/'runs/development_011_test')

    def test_missing_source_read_only(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            result=campaign.check(SimpleNamespace(source=Path(d)))
            self.assertFalse(result['ready']);self.assertEqual(list(Path(d).iterdir()),[])

    def test_gate_improvement_vs_D_not_default_or_runtime(self):
        rows=validation()
        for r in rows:
            if r['epoch']==2:r['total_seconds']=10000.
        result=campaign.choose_policies(rows)
        self.assertEqual(len(rows),112)
        self.assertTrue(all(a['enabled'] and a['epoch']==2 for a in result['arms'].values()))
        self.assertFalse(result['runtime_used_for_selection']);self.assertFalse(result['test_seeds_used'])

    def test_no_improvement_or_exact_tie_deploys_D(self):
        rows=validation()
        for r in rows:
            if r['method']=='L':r['wirelength']=100.
            if r['method']=='Y':r['wirelength']=110.
        result=campaign.choose_policies(rows)
        for arm in ('L','Y'):
            self.assertFalse(result['arms'][arm]['enabled']);self.assertEqual(result['arms'][arm]['epoch'],0)
        self.assertTrue(result['arms']['J']['enabled'])

    def test_failure_veto_not_masked_by_conditional_good_qor(self):
        rows=validation()
        next(r for r in rows if r['method']=='L' and r['epoch']==2)['status']='unroutable'
        result=campaign.choose_policies(rows)
        self.assertEqual(result['arms']['L']['epoch'],5)
        self.assertFalse(result['arms']['L']['scores'][0]['eligible'])
        self.assertEqual(result['arms']['L']['scores'][0]['common_success_pairs'],15)
        self.assertEqual(result['arms']['L']['scores'][1]['common_success_pairs'],15)

    def test_packing_failure_remains_negative_scientific_evidence(self):
        self.assertIn('packing_or_placement_failure',campaign.SCIENTIFIC_STATUSES)
        self.assertNotIn('infrastructure_failure',campaign.SCIENTIFIC_STATUSES)
        self.assertNotIn('interface_failure',campaign.SCIENTIFIC_STATUSES)
        rows=validation()
        bad=next(r for r in rows if r['method']=='Y' and r['epoch']==2)
        bad.update(status='packing_or_placement_failure',confirmed_candidate_infeasibility=True)
        result=campaign.choose_policies(rows)
        self.assertEqual(result['arms']['Y']['epoch'],5)
        self.assertEqual(result['arms']['Y']['scores'][0]['successes'],15)

    def test_backend_assertion_does_not_become_candidate_infeasibility(self):
        status,confirmed=campaign.classify_native_failure('packing_or_placement_failure',1,
            'Message: Failed to find device which satisfies resource requirements required memory:294 available248')
        self.assertEqual(status,'packing_or_placement_failure');self.assertTrue(confirmed)
        for code,text in ((134,'Assertion failed'),(-11,'Segmentation fault'),(1,'Unknown error'),
            (1,'Failed to find device which satisfies resource requirements\nAssertion failed')):
            status,confirmed=campaign.classify_native_failure('packing_or_placement_failure',code,text)
            self.assertEqual(status,'infrastructure_failure');self.assertFalse(confirmed)
        rows=validation();rows[0].update(status='packing_or_placement_failure')
        with self.assertRaisesRegex(ValueError,'infrastructure'):campaign.choose_policies(rows)

    def test_missing_or_duplicate_validation_rejected(self):
        rows=validation()
        with self.assertRaises(ValueError):campaign.choose_policies(rows[:-1])
        with self.assertRaises(ValueError):campaign.choose_policies(rows+[rows[0]])

    def test_same_seeds_mean_before_ratio(self):
        rows=[row('spmv/H0Z0/w128','D',seed=s,wl=1.) for s in (1,2)]
        rows+=[row('spmv/H0Z0/w128','J',seed=1,wl=1.),row('spmv/H0Z0/w128','J',seed=2,wl=9.)]
        self.assertAlmostEqual(campaign.contrast(rows,'J','D')['metrics']['wirelength']['ratio'],5.)

    def test_candidate_opportunity_is_training_diagnostic_not_deployable_score(self):
        trials=lambda wl:[row('spmv/H0Z0/w128','D',seed=s,wl=wl) for s in campaign.FEEDBACK_SEEDS]
        actions=[dict(condition_id='spmv/H0Z0/w128',arm=a,candidate_id=n,trials=trials(wl))
            for a,n,wl in (('D','identity',100.),('L','l_s1',90.),('Y','y_s1',110.))]
        result=campaign.candidate_diagnostics(actions)
        self.assertEqual(result['better_candidates_by_arm'],dict(L=1,Y=0,J=0))
        self.assertFalse(result['oracle_is_deployable'])
        self.assertTrue(result['all_negative_results_retained'])

    def test_worker_changed_checkpoint_rejected_before_torch_or_output(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            path=Path(d)/'source.pt';path.write_bytes(b'another valid checkpoint')
            out=Path(d)/'never-created'
            with self.assertRaisesRegex(ValueError,'frozen identity'):
                campaign.worker(dict(checkpoint=str(path),checkpoint_sha256='0'*64),out)
            self.assertFalse(out.exists())

    def test_worker_cache_signature_pins_all_consumed_raw_inputs(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            directory=Path(d);target=directory/'generation'
            request=dict(prefix=str(directory/'snapshot'),blif=str(directory/'input.blif'),
                architecture=str(directory/'arch.xml'),device=str(directory/'device_graph.json'),
                vertical_summary=str(directory/'vertical_summary.json'),
                checkpoint=str(directory/'selector.pt'),base_checkpoint=str(directory/'frozenD.pt'))
            runner=campaign.ResidualCampaign.__new__(campaign.ResidualCampaign)
            runner.guard=lambda:None;runner.request=lambda *args,**kwargs:request
            runner.args=SimpleNamespace(timeout=60)
            write(target/'payload/generation_report.json',dict(status='complete'))
            with patch.object(campaign,'run_trial',return_value=dict(status='complete',metrics=dict(wall_seconds=1.))) as trial:
                runner.generate({},target,method='J')
            pinned={str(p) for p in trial.call_args.kwargs['inputs']}
            for key in ('blif','architecture','device','vertical_summary','checkpoint','base_checkpoint'):
                self.assertIn(request[key],pinned)
            for suffix in ('.blocks.tsv','.nets.tsv','.meta.tsv'):
                self.assertIn(request['prefix']+suffix,pinned)
            self.assertIn(str(target/'request.json'),pinned)

    def test_zero_feedback_stops_validation(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            runner=campaign.ResidualCampaign.__new__(campaign.ResidualCampaign);runner.out=Path(d)
            write(runner.out/'trained_report.json',dict(feedback_pairs=0,checkpoints=[]))
            with self.assertRaisesRegex(ValueError,'requires effective'):runner.validate()
            self.assertFalse((runner.out/'validation').exists())

    def test_trained_checkpoint_change_stops_validation(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            runner=campaign.ResidualCampaign.__new__(campaign.ResidualCampaign);runner.out=Path(d)
            paths=[Path(d)/f'epoch{e}.pt' for e in (2,5)]
            for p in paths:p.write_bytes(b'original snapshot')
            wanted={str(p):sha(p) for p in paths}
            write(runner.out/'trained_report.json',dict(feedback_pairs=2,checkpoints=[str(p) for p in paths],checkpoint_sha256=wanted))
            paths[1].write_bytes(b'replaced valid checkpoint')
            with self.assertRaisesRegex(ValueError,'frozen identity'):runner.validate()
            self.assertFalse((runner.out/'validation').exists())


if __name__=='__main__':unittest.main()
