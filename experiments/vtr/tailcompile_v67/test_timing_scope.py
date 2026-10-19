import os
from pathlib import Path
import tempfile
import subprocess
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import resume as r


class TimingScopeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.a=SimpleNamespace(out=self.root,timeout=7200,jobs=4,retry_timeouts=False,retry_failed=False,
            time_scan_timeouts=False,heartbeat=120,timing_seed_count=3,timing_width_count=1,replan_timing=False)
        paths={}
        for name in ('blif','packed','fplace'):
            p=self.root/name;p.write_text(name);paths[name]=str(p)
        self.c=dict(design='test',width=100,device_width=14,**paths,
            fplace_sha256=r.f.sha(paths['fplace']),sources=dict(mapped_blif=r.f.sha(paths['blif']),packed_net=r.f.sha(paths['packed'])))
        generation=self.root/'generation.json'
        r.f.write(generation,dict(timing={k:1 for keys in r.f.SCOPES.values() for k in keys}))
        self.plan=dict(signature=dict(code=r.f.direct.implementation()),
            cases=[dict(self.c,width=w,generation=str(generation)) for w in (80,90,100)],seeds=[251,252,253,254,255])
        r.f.write(self.root/'plan.json',self.plan)
        for task in r.f.tasks(self.a,self.plan,False):
            c,s,m,folder=task
            r.f.write(folder/'trial.json',dict(width=c['width'],method=m,seed=s,status='success' if c['width']==100 else 'unroutable',
                process_returncode=0 if c['width']==100 else 2,wall_seconds=100,wirelength=100,cpd=10))

    def test_boundary_selection_fixed_seeds_and_dedup(self):
        plan=r.timing_plan(self.a)
        self.assertEqual(plan['fixed_seeds'],[251,252,253])
        self.assertEqual(plan['selected_task_count'],12)
        self.assertTrue(all(t[1]==100 and t[2] in (251,252,253) for t in plan['selected_tasks']))
        # An arbitrarily fast/slow existing timing value cannot change selection.
        folder=self.root/'timing/test/w100/s251/A_default'
        r.f.write(folder/'trial.json',dict(status='success',wall_seconds=.001))
        self.assertEqual(r.timing_plan(self.a),plan)
        progress=r.timing_progress(self.a,self.plan,plan)
        self.assertEqual(progress['expected'],12)
        self.assertEqual(progress['remaining'],11)

    def test_scan_summary_needs_no_timing_and_reports_wmin(self):
        with patch.object(r,'run_process',side_effect=AssertionError('summary must not launch VPR')):
            summary=r.scan_report(self.a)
        self.assertTrue(summary['scan_complete'])
        self.assertEqual(len(summary['wmin_comparisons']),5)
        self.assertTrue(all(v['w80']==100 for v in summary['width_results']))
        self.assertFalse((self.root/'timing').exists())

    def test_unknowns_are_not_promoted_to_success_and_plan_changes_are_explicit(self):
        original=r.timing_plan(self.a)
        for w in (80,90,100):
            p=self.root/'runs/test'/f'w{w}'/'s251/C_default_refine/trial.json'
            r.f.write(p,dict(r.f.read(p),status='other_failure',process_returncode=124))
        with self.assertRaises(ValueError):r.timing_plan(self.a)
        self.a.replan_timing=True
        selected=r.timing_plan(self.a)
        self.assertFalse(any(t[2]==251 and t[3]=='C_default_refine' for t in selected['selected_tasks']))
        self.assertTrue(any((self.root/'timing_selection_history').iterdir()))
        row=next(p for p in selected['comparisons'] if p['candidate']=='C_default_refine')
        self.assertEqual(row['no_joint_success_seeds'],[251])

    def test_selected_success_is_reused_without_vpr(self):
        identity={'binary':'same'}
        folder=self.root/'timing/test/w100/s251/A_default'
        r.f.write(folder/'trial.json',dict(signature=r.signature(self.a,identity,self.c,251,'A_default'),
            status='success',process_returncode=0,wall_seconds=50,method='A_default',wirelength=100,cpd=10))
        with patch.object(r,'run_process',side_effect=AssertionError('must reuse')):
            row=r.run_trial(self.a,None,identity,(self.c,251,'A_default',folder),True)
        self.assertTrue(row['reused'])
        self.assertEqual(row['wall_seconds'],50)

    @unittest.skipUnless(os.name=='posix','execution requires Linux/WSL')
    def test_stop_targets_only_same_output_controller(self):
        script=str(Path(r.__file__).resolve())
        procs=[subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)',script,
            'timing','--out',str(folder)]) for folder in (self.root,self.root/'other_cohort')]
        wait=threading.Thread(target=procs[0].wait);wait.start()
        try:
            r.request_stop(self.a)
            wait.join(timeout=3)
            self.assertIsNotNone(procs[0].poll())
            self.assertIsNone(procs[1].poll())
        finally:
            for p in procs:
                if p.poll() is None:p.terminate()
                p.wait()
            wait.join(timeout=3)

    @unittest.skipUnless(os.name=='posix','execution requires Linux/WSL')
    def test_execute_launches_only_selected_tasks(self):
        backend=SimpleNamespace(root=str(self.root),binary_hash='binary',arch_hash='arch')
        with patch.dict(os.environ,{'VTR_ROOT':str(self.root)}), \
             patch.object(r.f.direct,'DirectBackend',return_value=backend), \
             patch.object(r.f,'verify',return_value='hook'), \
             patch.object(r,'run_trial',return_value={'status':'success'}) as run:
            r.execute(self.a,True)
        self.assertEqual(run.call_count,12)
        for call in run.call_args_list:
            c,seed,method,folder=call.args[3]
            self.assertEqual(c['width'],100)
            self.assertIn(seed,[251,252,253])


if __name__=='__main__':unittest.main()
