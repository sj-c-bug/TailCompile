import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import parallel_runtime as p


class ParallelRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()

    def test_core_selection_avoids_smt_and_honors_cpuset_quota(self):
        rows=[dict(cpu=cpu,package=0,core=core,numa=0) for cpu,core in [(0,0),(1,1),(8,0),(9,1)]]
        self.assertEqual([r['cpu'] for r in p.select_cores(rows,2)],[0,1])
        with self.assertRaises(ValueError):p.select_cores(rows,2,[0,8])
        with self.assertRaises(ValueError):p.select_cores(rows,3)
        with self.assertRaises(ValueError):p.select_cores(rows,2,quota=1)
        with self.assertRaises(ValueError):p.select_cores(rows,2,[0,99])

    def test_schedule_pairs_share_core_and_counterbalance_order(self):
        selection=dict(fixed_seeds=[251,252,253],comparisons=[dict(design='test',candidate='B_gnn',baseline='A_default',
            pairs=[dict(width=100,seed=s) for s in (251,252,253)])])
        waves=p.wave_schedule(selection,[{'cpu':2},{'cpu':4}])
        self.assertEqual([w['effective_lanes'] for w in waves],[2,1])
        self.assertEqual(waves[0]['lanes'][0]['methods'],['A_default','B_gnn'])
        self.assertEqual(waves[0]['lanes'][1]['methods'],['B_gnn','A_default'])
        self.assertEqual(waves[1]['lanes'][0]['cpu'],4)

    def test_incomplete_wave_archives_all_attempts(self):
        folder=self.root/'timing_parallel/c2';wave=folder/'waves/w0000'
        p.r.f.write(wave/'half0/lane00/trial.json',{'status':'success'})
        p.archive_wave(folder,wave)
        self.assertFalse(wave.exists())
        self.assertEqual(len(list((folder/'_attempts').rglob('trial.json'))),1)

    def test_report_keeps_tail_load_separate_and_ignores_serial_results(self):
        folder=self.root/'timing_parallel/c2';a=SimpleNamespace(out=self.root,run_name='c2')
        generation=self.root/'generation.json'
        p.r.f.write(generation,dict(timing={k:1 for keys in p.r.f.SCOPES.values() for k in keys}))
        p.r.f.write(self.root/'plan.json',dict(cases=[dict(design='test',width=100,generation=str(generation))]))
        selection=dict(fixed_seeds=[251,252,253],comparisons=[dict(design='test',candidate='B_gnn',baseline='A_default',
            pairs=[dict(width=100,seed=s) for s in (251,252,253)])])
        waves=p.wave_schedule(selection,[{'cpu':2},{'cpu':4}])
        p.r.f.write(folder/'schedule.json',dict(waves=waves))
        p.r.f.write(folder/'protocol.json',dict(concurrency=2))
        for wave in waves:
            results={}
            for h in (0,1):
                for lane in wave['lanes']:
                    method=lane['methods'][h]
                    results[f'{h}:{lane["lane"]}']=dict(method=method,status='success',started_unix=h*100,
                        wall_seconds=10 if method=='A_default' else 5,wirelength=100,cpd=10)
            p.r.f.write(p.wave_folder(folder,wave)/'wave.json',dict(complete=True,trial_hashes={},results=results,wall_seconds=20))
        p.r.f.write(self.root/'timing/test/w100/s251/B_gnn/trial.json',dict(wall_seconds=99999))
        p.report(a)
        result=p.r.f.read(folder/'parallel_report.json')
        self.assertTrue(result['complete'])
        self.assertFalse(result['serial_results_included'])
        self.assertEqual({c['effective_lanes'] for c in result['comparisons']},{1,2})
        self.assertTrue(all(c['vpr_wall_speedup']==2 for c in result['comparisons']))

    @unittest.skipUnless(os.name=='posix','real affinity requires Linux')
    def test_real_affinity_barriers_and_completed_wave_resume(self):
        cores=p.select_cores(p.topology(),min(2,len({(r['package'],r['core']) for r in p.topology()})))
        if len(cores)<2:self.skipTest('need two reported physical cores')
        folder=self.root/'timing_parallel/c2';a=SimpleNamespace(out=self.root,run_name='c2',timeout=20,heartbeat=10)
        packed=self.root/'source.net';packed.write_text('input')
        c=dict(design='test',width=100,device_width=14,packed=str(packed),sources={},fplace_sha256='candidate')
        plan=dict(cases=[c])
        selection=dict(fixed_seeds=[251,252],comparisons=[dict(design='test',candidate='B_gnn',baseline='A_default',
            pairs=[dict(width=100,seed=s) for s in (251,252)])])
        schedule=dict(waves=p.wave_schedule(selection,cores))
        setup=(folder,dict(identity={'binary':'test'}),schedule,plan,None)
        command=[sys.executable,'-u','-c','import time; time.sleep(.1); print("VPR succeeded",flush=True)']
        with patch.object(p,'prepare',return_value=setup),patch.object(p.r.f,'command',return_value=command):
            p.execute(a)
        result=p.r.f.read(p.wave_folder(folder,schedule['waves'][0])/'wave.json')
        self.assertTrue(result['complete'])
        for lane in schedule['waves'][0]['lanes']:
            rows=[result['results'][f'{h}:{lane["lane"]}'] for h in (0,1)]
            self.assertTrue(all(row['cpu_affinity']['allowed_cpus']==[lane['cpu']] for row in rows))
            self.assertTrue(all(set(row['cpu_affinity']['thread_env'].values())=={'1'} for row in rows))
        first=[v['finished_unix'] for k,v in result['results'].items() if k.startswith('0:')]
        second=[v['started_unix'] for k,v in result['results'].items() if k.startswith('1:')]
        self.assertLessEqual(max(first),min(second))
        with patch.object(p,'prepare',return_value=setup),patch.object(p.r,'run_process',side_effect=AssertionError('must reuse complete wave')):
            p.execute(a)


if __name__=='__main__':unittest.main()
