import contextlib
import csv
import io
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import run
from bridge import read, write, sha, REPO, metrics


class SweepContractTest(unittest.TestCase):
    def test_runtime_excludes_setup_but_charges_decode(self):
        value=run.cost_fields({'wall_seconds':1020},dict(vpr_seconds=12,
            compute_generation_seconds=3,excluded_setup_seconds=1000))
        self.assertEqual(value['total_seconds'],15)
        self.assertEqual(value['excluded_setup_seconds'],1000)
        self.assertIsNone(run.cost_fields({'wall_seconds':20},None)['total_seconds'])
        with self.assertRaises(ValueError):
            run.cost_fields({'wall_seconds':20},dict(vpr_seconds=12,compute_generation_seconds=float('nan')))

    def test_timing_widths_do_not_depend_on_gnn_success(self):
        rows=[dict(design='x',width=w,reference_width=100,gnn_success=(w==80)) for w in [120,80,100]]
        self.assertEqual([r['width'] for r in run.timing_contexts(rows)],[100,120])
        rows[0]['gnn_success']=True
        self.assertEqual([r['width'] for r in run.timing_contexts(rows)],[100,120])

    def test_five_seeds_required_and_no_calibration_overlap(self):
        out=REPO/'experiments/vtr/tailcompile_v610/runs/test'
        for seeds in [['1','2','3'],['401','412','413','414','415']]:
            with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
                run.parse_args(['status','--out',str(out),'--seeds',*seeds])
        args=run.parse_args(['status','--out',str(out)])
        self.assertEqual(args.generation_jobs,32)
        self.assertEqual(args.rr_jobs,32)

    def test_graph_change_after_preparation_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp); cp=root/'checkpoint.pt'; cp.write_bytes(b'frozen')
            graph=root/'graph.pt'; graph.write_bytes(b'original')
            write(root/'protocol.json',dict(code={},checkpoint=str(cp),checkpoint_sha256=sha(cp),timeout=1))
            write(root/'contexts.json',dict(contexts=[dict(design='x',width=100)]))
            write(root/'inventory.json',dict(designs=[]))
            write(root/'prepared_signature.json',dict(contexts_sha256=sha(root/'contexts.json'),
                inventory_sha256=sha(root/'inventory.json'),artifact_sha256={str(graph):sha(graph)}))
            args=SimpleNamespace(out=root)
            self.assertEqual(run.load(args)[1][0]['design'],'x')
            graph.write_bytes(b'modified')
            with self.assertRaisesRegex(ValueError,'graph/IR changed'):
                run.load(args)

    def test_five_seed_means_keep_nonmonotonicity_and_unknowns(self):
        seeds=[411,412,413,414,415]
        rows=[dict(design='x',width=w,seed=s,method=m,
            status='success' if w==104 else 'unroutable')
            for s in seeds for w in [100,104,108] for m in ['default','fast']]
        result=metrics.aggregate_widths(rows,seeds=seeds,widths_by_design={'x':[100,104,108]},methods=['default','fast'])
        self.assertEqual(result['groups'][0]['five_seed_mean_observed_wmin'],104)
        self.assertEqual(result['groups'][0]['nonmonotone_seed_count'],5)
        rows[0]['status']='timeout'
        result=metrics.aggregate_widths(rows,seeds=seeds,widths_by_design={'x':[100,104,108]},methods=['default','fast'])
        self.assertIsNone(result['groups'][0]['five_seed_mean_observed_wmin'])

    def test_timing_generation_does_not_overwrite_full_coverage(self):
        import generate
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp)
            write(root/'generation/generation_report.json',dict(expected=160))
            a=SimpleNamespace(out=root,checkpoint=root/'cp',retry_failed=False,generation_jobs=32)
            c=dict(design='x',width=100)
            with patch.object(generate,'signature',return_value={}),patch.object(generate,'validate_generation',return_value={'ok':True}),contextlib.redirect_stdout(io.StringIO()):
                run.generate_all(a,dict(options={},checkpoint_sha256='frozen'),[c],report_name='timing_generation_report.json')
            self.assertEqual(read(root/'generation/generation_report.json')['expected'],160)
            self.assertEqual(read(root/'generation/timing_generation_report.json')['expected'],1)

    def test_group_report_does_not_drop_missing_metrics(self):
        def row(name,total):
            return dict(design=name,family='same',candidate_over_baseline_geomean={k:total for k in ('vpr','total','cpd','wirelength')},
                success_loss_pairs=0,success_gain_pairs=0,passed=False)
        group=run.grouped_summaries([row('small',.5),row('large',None)],'family')[0]
        self.assertIsNone(group['candidate_over_baseline_geomean']['total'])
        self.assertEqual(len(group['designs']),2)

    def test_ready_batch_results_keep_deferred_designs_visible_without_fake_labels(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);a=SimpleNamespace(out=root,run_name='paired')
            seeds=[411,412,413,414,415]
            p=dict(seeds=seeds,methods=['default','fast'],checkpoint_sha256='frozen',
                evaluation_role='Exploratory coverage',budgets=metrics.DEFAULT_BUDGETS)
            contexts=[dict(design='ready',family='ready-family',width=w,reference_width=104,
                original_split='sweep',checkpoint_exposure='unseen') for w in (100,104,108)]
            deferred=dict(design='slow',status='timeout',stage='synthesis')
            write(root/'preparation_report.json',dict(complete=True,requested_design_count=2,
                eligible_design_count=1,deferred_design_count=1,deferred=[deferred],coverage_complete=False))
            write(root/'inventory.json',dict(designs=[dict(design='ready'),dict(design='slow',family='slow-family')]))
            for c in contexts:
                for seed in seeds:
                    for method in p['methods']:
                        write(root/'evaluation/runs/ready'/f'w{c["width"]}'/f's{seed}'/method/'metrics.json',
                            dict(design='ready',width=c['width'],seed=seed,method=method,
                                status='unroutable' if c['width']==100 else 'success'))
            rows=[dict(design='ready',width=w,seed=s,method=m,status='success',cpd=1.,wirelength=10.,
                vpr_seconds=100. if m=='default' else 90.,generation_seconds=0. if m=='default' else 1.)
                for w in (104,108) for s in seeds for m in p['methods']]
            with patch.object(run,'load',return_value=(p,contexts)),contextlib.redirect_stdout(io.StringIO()):
                report=run.runtime_report(a,p,contexts,rows,protocol={})
            self.assertTrue(report['complete'])
            self.assertTrue(report['passed'])
            self.assertFalse(report['coverage_complete'])
            self.assertFalse(report['full_inventory_passed'])
            self.assertEqual(report['preparation_coverage']['deferred'],[deferred])
            self.assertEqual([r['design'] for r in report['by_design']],['ready'])
            scan=read(root/'evaluation_report.json')
            self.assertFalse(scan['coverage_complete'])
            self.assertEqual(scan['expected_trials'],30)
            with (root/'benchmark_summary.csv').open(encoding='utf-8-sig',newline='') as stream:
                table=list(csv.DictReader(stream))
            slow=next(r for r in table if r['design']=='slow')
            self.assertEqual(slow['preparation_status'],'deferred_timeout')
            self.assertTrue(all(slow[k]=='' for k in ('default_mean_wmin','gnn_mean_wmin',
                'wmin_ratio','compute_plus_vpr_ratio','cpd_ratio','wirelength_ratio','passed')))

    def test_old_complete_cohorts_still_report_full_coverage(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            a=SimpleNamespace(out=Path(tmp))
            contexts=[dict(design='x',width=w) for w in (100,104)]
            write(a.out/'preparation_report.json',dict(inventory_design_count=1,complete=True))
            report=run.preparation_coverage(a,contexts)
            self.assertTrue(report['coverage_complete'])
            self.assertEqual(report['evaluated_design_count'],1)
            self.assertEqual(report['deferred_design_count'],0)

    def test_formal_load_checks_deferred_coverage_against_the_frozen_contexts(self):
        import preparation_admission
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);cp=root/'checkpoint.pt';cp.write_bytes(b'frozen')
            write(root/'protocol.json',dict(code={},checkpoint=str(cp),checkpoint_sha256=sha(cp),timeout=7200))
            write(root/'contexts.json',dict(contexts=[dict(design='ready',width=100)]))
            write(root/'inventory.json',dict(designs=[dict(design='ready'),dict(design='slow')]))
            write(root/'prepared_signature.json',dict(contexts_sha256=sha(root/'contexts.json'),
                inventory_sha256=sha(root/'inventory.json'),artifact_sha256={}))
            write(root/'preparation_admission.json',{})
            deferred=dict(design='slow',status='timeout')
            write(root/'preparation_report.json',dict(deferred=[deferred]))
            a=SimpleNamespace(out=root)
            with patch.object(preparation_admission,'load_deferred',return_value={'slow':deferred}):
                self.assertEqual(len(run.load(a)[1]),1)
                write(root/'preparation_report.json',dict(deferred=[]))
                with self.assertRaisesRegex(ValueError,'coverage changed'):
                    run.load(a)
            with patch.object(preparation_admission,'load_deferred',return_value={}):
                with self.assertRaisesRegex(ValueError,'context coverage differs'):
                    run.load(a)


class StorageTest(unittest.TestCase):
    def test_terminal_cleanup_keeps_canonical_input_and_evidence(self):
        import bridge
        import storage
        base=REPO/'experiments/vtr/tailcompile_v610/runs';base.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=base,prefix='test-storage-') as tmp:
            out=Path(tmp);folder=out/'evaluation/runs/x/w100/s1/default';folder.mkdir(parents=True)
            retained=folder/'source.net';retained.write_text('canonical')
            derived=folder/'x.net';derived.write_text('derived')
            route=folder/'x.route';route.write_text('derived route')
            write(folder/'trial.json',dict(status='success',identity=dict(stage='route'),command=['vpr','--net_file',str(retained)]))
            write(folder/'metrics.json',dict(status='success'))
            with patch.object(bridge,'_tracked',storage.prepare_paths(set())),patch.object(storage,'active_paths',return_value=set()),patch.dict(os.environ,{'TAILCOMPILE_V610_PRUNE_COMPLETED':'1'}),contextlib.redirect_stdout(io.StringIO()):
                bridge.prune_trial(folder,out)
                bridge.prune_trial(folder,out)
            self.assertTrue(retained.exists())
            self.assertFalse(derived.exists())
            self.assertFalse(route.exists())
            self.assertTrue((folder/'trial.json').exists())
            self.assertTrue((folder/'payload_pruned.json').exists())


if __name__=='__main__':
    unittest.main()
