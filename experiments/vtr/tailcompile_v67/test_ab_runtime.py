import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import ab_runtime as ab

p=ab.p


class AbRuntimeTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.selection=dict(fixed_seeds=[251,252,253],comparisons=[
            dict(design='test',baseline='A_default',candidate=c,pairs=[dict(width=100,seed=s) for s in (251,252,253)])
            for c in ('B_gnn','C_default_refine','D_gnn_refine')])

    def test_filter_retains_all_original_ab_pairs_and_no_refinement(self):
        selected=ab.ab_selection(self.selection)
        self.assertEqual(len(self.selection['comparisons']),3)
        self.assertEqual(len(selected['comparisons']),1)
        self.assertEqual(selected['comparisons'][0]['pairs'],self.selection['comparisons'][0]['pairs'])
        self.assertEqual(selected['selected_task_count'],6)
        self.assertEqual({r[-1] for r in selected['selected_tasks']},{'A_default','B_gnn'})
        with self.assertRaises(ValueError):ab.ab_selection(dict(fixed_seeds=[],comparisons=[]))

    def test_status_reports_individual_finished_missing_failed_trials(self):
        a=SimpleNamespace(out=self.root,run_name='ab_test')
        folder=p.run_dir(a)
        schedule=dict(waves=p.wave_schedule(ab.ab_selection(self.selection),[dict(cpu=0),dict(cpu=1)]))
        p.r.f.write(folder/'schedule.json',schedule)
        base=p.wave_folder(folder,schedule['waves'][0])
        p.r.f.write(base/'half0/lane00/trial.json',dict(status='success'))
        p.r.f.write(base/'half0/lane01/trial.json',dict(status='other_failure'))
        result=ab.status(a)
        self.assertEqual(result['waves_complete'],0)
        self.assertEqual(result['trials_expected'],6)
        self.assertEqual(result['trial_status_counts'],{'success':1,'other_failure':1,'missing':4})

    def test_report_only_ab_and_preserves_generation_cost_scopes(self):
        a=SimpleNamespace(out=self.root,run_name='ab_test')
        folder=p.run_dir(a);generation=self.root/'generation.json'
        p.r.f.write(generation,dict(timing={k:1 for keys in p.r.f.SCOPES.values() for k in keys}))
        p.r.f.write(self.root/'plan.json',dict(cases=[dict(design='test',width=100,generation=str(generation))]))
        waves=p.wave_schedule(ab.ab_selection(self.selection),[dict(cpu=0),dict(cpu=1)])
        p.r.f.write(folder/'schedule.json',dict(waves=waves))
        p.r.f.write(folder/'protocol.json',dict(concurrency=2))
        for wave in waves:
            results={}
            for half in (0,1):
                for lane in wave['lanes']:
                    method=lane['methods'][half]
                    results[f'{half}:{lane["lane"]}']=dict(method=method,status='success',
                        started_unix=half*100,wall_seconds=30 if method=='A_default' else 20,
                        wirelength=100,cpd=10)
            p.r.f.write(p.wave_folder(folder,wave)/'wave.json',dict(complete=True,
                trial_hashes={},results=results,wall_seconds=50))
        ab.report(a)
        result=p.r.f.read(folder/'parallel_report.json')
        self.assertTrue(result['complete']);self.assertFalse(result['old_mixed_runtime_reused'])
        self.assertFalse(result['local_refinement'])
        self.assertEqual({c['effective_lanes'] for c in result['comparisons']},{1,2})
        for c in result['comparisons']:
            self.assertEqual(c['candidate'],'B_gnn');self.assertEqual(c['baseline'],'A_default')
            self.assertAlmostEqual(c['vpr_wall_speedup'],1.5)
            self.assertLess(c['scope_speedup']['warm_inference_lowering'],1.5)

    @unittest.skipUnless(os.name=='posix','real affinity requires Linux')
    def test_execute_pins_ab_children_and_resumes_complete_waves(self):
        cores=p.select_cores(p.topology(),1)
        a=SimpleNamespace(out=self.root,run_name='ab_test',timeout=20,heartbeat=10)
        folder=p.run_dir(a);packed=self.root/'source.net';packed.write_text('synthetic')
        case=dict(design='test',width=100,device_width=14,packed=str(packed),sources={},fplace_sha256='candidate')
        selection=dict(fixed_seeds=[251],comparisons=[dict(design='test',baseline='A_default',
            candidate='B_gnn',pairs=[dict(width=100,seed=251)])])
        schedule=dict(waves=p.wave_schedule(selection,cores))
        setup=(folder,dict(identity={'binary':'synthetic'}),schedule,dict(cases=[case]),None)
        command=[sys.executable,'-u','-c','print("VPR succeeded",flush=True)']
        with patch.object(ab,'prepare',return_value=setup),patch.object(p.r.f,'command',return_value=command):
            ab.execute(a)
        row=p.r.f.read(p.wave_folder(folder,schedule['waves'][0])/'wave.json')
        self.assertTrue(row['complete'])
        self.assertEqual({v['method'] for v in row['results'].values()},{'A_default','B_gnn'})
        self.assertTrue(all(v['cpu_affinity']['allowed_cpus']==[cores[0]['cpu']] for v in row['results'].values()))
        self.assertTrue(all(not v['hook_env'] for v in row['results'].values()))
        with patch.object(ab,'prepare',return_value=setup),patch.object(p.r,'run_process',side_effect=AssertionError('must reuse')):
            ab.execute(a)

    def test_storage_failure_is_retained_and_retry_archives_incomplete_wave(self):
        a=SimpleNamespace(out=self.root,run_name='ab_test')
        folder=p.run_dir(a)
        selection=dict(fixed_seeds=[251],comparisons=[dict(design='test',baseline='A_default',
            candidate='B_gnn',pairs=[dict(width=100,seed=251)])])
        schedule=dict(waves=p.wave_schedule(selection,[dict(cpu=0)]))
        setup=(folder,dict(identity={}),schedule,dict(cases=[dict(design='test',width=100)]),None)
        with patch.object(ab,'prepare',return_value=setup),patch.object(p.r,'run_process',side_effect=OSError(122,'Disk quota exceeded')):
            with self.assertRaisesRegex(RuntimeError,'check storage quota'):ab.execute(a)
        old=p.r.f.read(p.wave_folder(folder,schedule['waves'][0])/'wave.json')
        self.assertFalse(old['complete']);self.assertIn('Disk quota exceeded',old['errors'][0])
        def success(args,backend,identity,case,seed,method,directory):
            row=dict(method=method,status='success')
            p.r.f.write(directory/'trial.json',row)
            return row
        with patch.object(ab,'prepare',return_value=setup),patch.object(p.r,'run_process',side_effect=success):
            ab.execute(a)
        self.assertEqual(len(list((folder/'_attempts').rglob('wave.json'))),1)
        self.assertTrue(p.r.f.read(p.wave_folder(folder,schedule['waves'][0])/'wave.json')['complete'])


if __name__=='__main__':unittest.main()
