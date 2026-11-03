"""012 predeclared folds, aggregation and read-only/offline runner guards."""
from collections import Counter
import copy
import itertools
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import edit_context_campaign as campaign
from io_utils import read,write,sha


def banks():
    return [dict(condition_id=f'{family}/{arch}/w128',family=family,architecture=arch,arm=arm)
            for family in ('spmv','mcml') for arch in ('H0Z0','H0Z1','H1Z0','H1Z1') for arm in campaign.ARMS]


def row(cid,arm,wl=1.,cpd=1.,base=1.):
    return dict(condition_id=cid,family=cid.split('/')[0],arm=arm,selected_WL=wl,D_WL=base,
        selected_CPD=cpd,D_CPD=base,selected_candidate_id='identity',selected_success_count=2,
        D_success_count=2,selected_is_offline_best=True,q_regret=0.)


class EditContextCampaignTests(unittest.TestCase):
    def source_fixture(self,folder):
        folder=Path(folder);source=folder/'source011';old=folder/'source008'
        model=folder/'D.pt';model.write_bytes(b'synthetic model bytes for read-only input audit')
        identity=source/'binary_identity.json';write(identity,dict(sha256='b'*64))
        ids=[f'{family}/{arch}/w128' for family in ('spmv','mcml') for arch in ('H0Z0','H0Z1','H1Z0','H1Z1')]
        records=[];pins={};feedback=[]
        for index,cid in enumerate(ids):
            graph=folder/f'graph{index}.pt';graph.write_bytes(b'synthetic graph bytes')
            records.append(dict(condition_id=cid,family=cid.split('/')[0],split='train',graph=str(graph)))
            pins[str(graph)]=sha(graph)
            for arm,name in [('D','identity')]+[(a,a.lower()+f'_s{k}') for a,k in itertools.product(campaign.ARMS,(1,2))]:
                target=source/cid/name/'target.json';write(target,dict(synthetic=True))
                init=target.parent/'init.ap.tsv';init.write_bytes(b'not used by preflight')
                gen=dict(target_sha256=sha(target),ap_init_sha256=sha(init));trials=[]
                for seed in (11401,11402):
                    manifest=target.parent/f's{seed}'/'vpr/manifest.json'
                    write(manifest,dict(status='success',returncode=0,metrics=dict(wirelength=100.,cpd_ns=10.),
                        signature=dict(command=['vpr','--seed',str(seed),'--route_chan_width','128'],
                            env=dict(TC_GP_INIT=str(init)),inputs={str(init):sha(init),str(identity):sha(identity)})))
                    trials.append(dict(condition_id=cid,width=128,seed=seed,status='success',wirelength=100.,cpd_ns=10.,
                        ap_init_sha256=sha(init),vpr_manifest=str(manifest)))
                feedback.append(dict(condition_id=cid,arm=arm,candidate_id=name,action=str(target),generation=gen,trials=trials))
        path=old/'teacher_records.json';write(path,dict(records=records));pins[str(path.resolve())]=sha(path)
        write(source/'residual_protocol.json',dict(source008=str(old),implementation={},source_inputs=pins,
            base_checkpoint_sha256=sha(model),binary_sha256='b'*64))
        write(source/'prepared_report.json',dict(checkpoint=str(model),checkpoint_sha256=sha(model)))
        write(source/'feedback.json',feedback);write(source/'feedback_report.json',dict(actions=56,native_runs=112))
        write(source/'comparison/report.json',dict(rows=[dict(condition_id=c,seed=s,method=m)
            for c,s,m in itertools.product(ids,(11101,11102),('B','D','L','Y','J'))]))
        return source,feedback

    def test_fixed_budget_is_offline_and_backbone_exposure_disclosed(self):
        plan=campaign.plan()
        self.assertEqual(plan['offline_fits'],28);self.assertEqual(plan['new_pnr'],0)
        self.assertEqual(plan['epochs'],80);self.assertEqual(plan['selection_epoch'],80)
        self.assertTrue(plan['frozen_backbone_family_exposure'])
        self.assertFalse(plan['holdout_used_for_epoch_or_hyperparameter_selection'])
        self.assertEqual(plan['head_width'],32)
        self.assertTrue(plan['train_fold_only_pair_scale'])

    def test_folds_hold_out_scorer_labels_and_cover_each_condition_once(self):
        folds=campaign.splits(banks());self.assertEqual(len(folds),7)
        full=folds[0];self.assertFalse(full['heldout']);self.assertEqual(full['train_ids'],full['evaluation_ids'])
        for group,count in (('scorer_leave_family_out',2),('scorer_leave_architecture_out',4)):
            selected=[s for s in folds if s['group']==group]
            self.assertEqual(len(selected),count)
            ids=[]
            for fold in selected:
                self.assertFalse(set(fold['train_ids']) & set(fold['evaluation_ids']))
                self.assertEqual(len(fold['train_ids'])+len(fold['evaluation_ids']),8)
                ids.extend(fold['evaluation_ids'])
            self.assertEqual(len(ids),8);self.assertTrue(all(n==1 for n in Counter(ids).values()))

    def test_wrong_panel_cannot_claim_natural_holdout(self):
        with self.assertRaisesRegex(ValueError,'two-family'):
            campaign.splits(banks()[:6])

    def test_summary_uses_equal_family_then_architecture_geometric_ratios(self):
        rows=[row('spmv/H0Z0/w128','L',wl=2),row('spmv/H0Z1/w128','L',wl=8),
              row('mcml/H0Z0/w128','L',wl=1)]
        out=campaign.aggregate(rows)['L']
        self.assertAlmostEqual(out['metrics']['WL']['ratio'],2.)
        self.assertEqual(out['conditions'],3)
        self.assertEqual(out['selected_successes'],6)
        with self.assertRaisesRegex(ValueError,'duplicate'):
            campaign.aggregate(rows+[rows[0]])

    def test_missing_source_is_read_only(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs',prefix='ctx-source-') as folder:
            result=campaign.inspect_source(folder)
            self.assertFalse(result['ready']);self.assertEqual(list(Path(folder).iterdir()),[])
            self.assertTrue(any(p.endswith('feedback.json') for p in result['missing']))

    def test_output_cannot_overlap_old_cohorts(self):
        for name in ('development_011_residual','development_010_structured','development_009_qor'):
            with self.assertRaisesRegex(ValueError,'development_012'):
                campaign.owned(campaign.HERE/'runs'/name)
        self.assertEqual(campaign.owned(campaign.HERE/'runs/development_012_test'),campaign.HERE/'runs/development_012_test')

    def test_report_rejects_incomplete_fits_without_importing_torch(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs',prefix='ctx-report-') as folder:
            runner=campaign.Campaign.__new__(campaign.Campaign);runner.out=Path(folder)
            write(runner.out/'trained_report.json',dict(complete=False,fits=[]))
            with self.assertRaisesRegex(ValueError,'incomplete'):runner.report()
            self.assertFalse((runner.out/'edit_context_report.json').exists())

    def test_no_vpr_functions_are_imported_or_called_by_offline_runner(self):
        self.assertFalse(hasattr(campaign,'vpr_command'))
        self.assertFalse(hasattr(campaign,'run_trial'))
        self.assertFalse(hasattr(campaign.Campaign,'pnr'))

    def test_matched_control_requires_same_init_budget_labels_and_calibration(self):
        fields=('initial_state_sha256','parameter_count','train_ids','train_pairs','train_pairs_by_arm',
                'calibration','source_checkpoint_sha256','backbone_sha256','cache_sha256',
                'global_dim','local_dim','epochs','seed','device')
        fits=[dict(split=dict(id='a'),mode=mode,seed=12301,
                   training=dict({k:'same' for k in fields},mode=mode)) for mode in campaign.MODES]
        self.assertTrue(campaign.validate_paired_fits(fits)['paired_controls_verified'])
        self.assertEqual(campaign.validate_paired_fits(fits)['pairs'],1)
        for field in ('parameter_count','initial_state_sha256','calibration','train_ids'):
            changed=copy.deepcopy(fits);changed[1]['training'][field]='tampered'
            with self.assertRaisesRegex(ValueError,'unmatched'):
                campaign.validate_paired_fits(changed)
        with self.assertRaisesRegex(ValueError,'missing'):
            campaign.validate_paired_fits(fits[:1])

    def test_E_G_contrast_uses_same_condition_actions_not_their_D_values(self):
        g=[dict(row('spmv/H0Z0/w128','L',wl=2.,cpd=4.,base=100),selected_digest='g')]
        e=[dict(row('spmv/H0Z0/w128','L',wl=1.,cpd=2.,base=100),selected_digest='e')]
        result=campaign.feature_contrast(e,g)['L']
        self.assertEqual(result['metrics']['WL']['ratio'],.5)
        self.assertEqual(result['metrics']['CPD']['ratio'],.5)
        self.assertEqual(result['same_candidate_choices'],0)
        self.assertEqual(result['E_selected_successes'],2)
        with self.assertRaisesRegex(ValueError,'populations'):
            campaign.feature_contrast(e,[])

    def test_source_native_labels_and_pin_chain_are_checked_before_cache_creation(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs',prefix='ctx-native-') as folder:
            source,feedback=self.source_fixture(folder)
            result=campaign.inspect_source(source)
            self.assertTrue(result['ready']);self.assertEqual(len(result['records']),8)
            self.assertEqual(len(result['feedback']),56)
            self.assertFalse((source/'feature_cache.pt').exists())
            manifest=Path(feedback[0]['trials'][0]['vpr_manifest']);row=read(manifest)
            row['signature']['command'][2]='99999';write(manifest,row)
            with self.assertRaisesRegex(ValueError,'seed/width/condition'):
                campaign.inspect_source(source)


if __name__=='__main__':unittest.main()
