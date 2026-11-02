"""010 finite budget, independent metric aggregation and stage guards."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import structured_campaign as campaign
from io_utils import read, write, sha


def row(condition, method, *, epoch=0, seed=10201, wl=100., cp=10., status='success'):
    family, architecture = condition.split('/')[:2]
    return dict(condition_id=condition, method=method, family=family, architecture=architecture,
                epoch=epoch, seed=seed, wirelength=wl, cpd_ns=cp, total_seconds=10., vpr_seconds=9., status=status)


class StructuredCampaignTests(unittest.TestCase):
    def test_budget_and_no_tail_or_rebuild(self):
        p = campaign.queue_plan()
        self.assertEqual(p['feedback_runs_upper']+p['validation_runs']+p['comparison_runs'],104)
        self.assertEqual(p['training_epochs'],5)
        self.assertEqual(p['methods'],['B','X','D','N'])
        self.assertEqual(p['tail_weight'],0.)
        self.assertEqual(p['backend_layer_prior'],0)
        self.assertEqual(p['new_backend_builds'],0)
        self.assertFalse(p['feedback_relabel_after_projection'])

    def test_010_namespace_does_not_allow_old_cohorts(self):
        for old in ('development_009_qor','development_008_aligned/x','development_007_layer'):
            with self.assertRaises(ValueError): campaign.owned(campaign.HERE/'runs'/old)
        self.assertEqual(campaign.owned(campaign.HERE/'runs/development_010_test'), campaign.HERE/'runs/development_010_test')

    def test_missing_check_is_read_only(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            source=Path(d)
            result=campaign.check(SimpleNamespace(source=source))
            self.assertFalse(result['ready'])
            self.assertEqual(list(source.iterdir()),[])

    def test_seed_arithmetic_mean_precedes_ratio(self):
        rows=[row('spmv/H0Z0/w128','B',seed=s,wl=1.) for s in (1,2)]
        rows += [row('spmv/H0Z0/w128','N',seed=1,wl=1.),row('spmv/H0Z0/w128','N',seed=2,wl=9.)]
        result=campaign.contrast(rows,'N','B')
        self.assertAlmostEqual(result['metrics']['wirelength']['ratio'],5.)
        self.assertEqual(result['jointly_successful'],2)

    def validation(self):
        rows=[]
        for family in ('blob_merge','diffeq'):
            for arch in ('H0Z0','H0Z1','H1Z0','H1Z1'):
                c=f'{family}/{arch}/w128'
                rows += [row(c,'B'),row(c,'N',epoch=2,wl=95.),row(c,'N',epoch=5,wl=97.)]
        return rows

    def test_validation_uses_qor_not_runtime(self):
        rows=self.validation()
        for r in rows:
            if r['epoch']==2:r['total_seconds']=1000.
        result=campaign.choose_epoch(rows)
        self.assertEqual(result['epoch'],2)
        self.assertTrue(result['performance_selection_completed'])
        self.assertFalse(result['runtime_used_for_selection'])
        with self.assertRaises(ValueError):campaign.choose_epoch(rows[:-1])

    def test_validation_success_coverage_precedes_common_qor(self):
        rows=self.validation()
        next(r for r in rows if r['epoch']==2)['status']='decode_failure'
        self.assertEqual(campaign.choose_epoch(rows)['epoch'],5)

    def test_zero_supported_pairs_stops_before_native_validation(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            runner=campaign.StructuredCampaign.__new__(campaign.StructuredCampaign);runner.out=Path(d)
            write(runner.out/'trained_report.json',dict(feedback_pairs=0,checkpoints=[]))
            with self.assertRaisesRegex(ValueError,'requires effective'):
                runner.validate()
            self.assertFalse((runner.out/'validation').exists())

    def test_valid_replacement_checkpoint_rejected_before_feedback(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            runner=campaign.StructuredCampaign.__new__(campaign.StructuredCampaign);runner.out=Path(d)
            checkpoint=runner.out/'control.pt';checkpoint.write_bytes(b'valid original checkpoint')
            wanted=sha(checkpoint)
            write(runner.out/'prepared_report.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=wanted))
            checkpoint.write_bytes(b'another otherwise valid checkpoint')
            with self.assertRaisesRegex(ValueError,'frozen identity'):runner.feedback()
            self.assertFalse((runner.out/'feedback').exists())

    def test_worker_rejects_checkpoint_mismatch_before_graph_or_torch(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            checkpoint=Path(d)/'other-valid.pt';checkpoint.write_bytes(b'otherwise valid replacement')
            out=Path(d)/'not-created'
            with self.assertRaisesRegex(ValueError,'frozen identity'):
                campaign.worker(dict(checkpoint=str(checkpoint),checkpoint_sha256='0'*64),out)
            self.assertFalse(out.exists())

    def test_trained_snapshot_replacement_rejected_before_validation(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            runner=campaign.StructuredCampaign.__new__(campaign.StructuredCampaign);runner.out=Path(d)
            checkpoints=[runner.out/f'epoch_{e}.pt' for e in (2,5)]
            for p in checkpoints:p.write_bytes(b'original checkpoint')
            wanted={str(p):sha(p) for p in checkpoints}
            write(runner.out/'trained_report.json',dict(feedback_pairs=1,checkpoints=[str(p) for p in checkpoints],checkpoint_sha256=wanted))
            checkpoints[1].write_bytes(b'another valid checkpoint')
            with self.assertRaisesRegex(ValueError,'frozen identity'):runner.validate()
            self.assertFalse((runner.out/'validation').exists())

    def test_selected_model_replacement_rejected_before_comparison(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            runner=campaign.StructuredCampaign.__new__(campaign.StructuredCampaign);runner.out=Path(d)
            control=runner.out/'control.pt';control.write_bytes(b'control checkpoint')
            selected=runner.out/'selected.pt';selected.write_bytes(b'selected checkpoint')
            wanted=sha(selected)
            write(runner.out/'prepared_report.json',dict(checkpoint=str(control),checkpoint_sha256=sha(control)))
            write(runner.out/'trained_report.json',dict(checkpoint_sha256={str(selected):wanted}))
            write(runner.out/'selection.json',dict(performance_selection_completed=True,checkpoint=str(selected),checkpoint_sha256=wanted))
            selected.write_bytes(b'another valid selected model')
            with self.assertRaisesRegex(ValueError,'frozen identity'):runner.compare()
            self.assertFalse((runner.out/'comparison').exists())

    def test_feedback_duplicate_sha_skips_native_run(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            runner=campaign.StructuredCampaign.__new__(campaign.StructuredCampaign)
            runner.out=Path(d);runner.budget=dict(physical_cores=[0],effective=1)
            runner.reference=[dict(id='spmv/H0Z0/w128',family='spmv')]
            checkpoint=runner.out/'source010.pt';checkpoint.write_bytes(b'control')
            write(runner.out/'prepared_report.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint)))
            calls=[]
            runner.generate=lambda *args,**kwargs: dict(ap_init='init.tsv',ap_init_sha256='a'*64,target='action.json')
            runner.pnr=lambda *args,**kwargs: calls.append(args) or dict(status='success',seed=10401)
            runner.feedback()
            result=read(runner.out/'feedback_report.json')
            self.assertEqual(len(calls),1)
            self.assertEqual(result['native_runs'],1)
            self.assertEqual(len(result['duplicates']),1)
            self.assertFalse(result['old_labels_reused_after_projection'])


if __name__=='__main__':unittest.main()
