import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import resume as r


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve()
        self.a=SimpleNamespace(out=self.root,timeout=7200,retry_timeouts=False,retry_failed=False,
            time_scan_timeouts=False,heartbeat=120)
        self.identity={'binary':'frozen'}
        self.case=dict(design='test',width=100,device_width=14,blif='/blif',packed='/packed',
                       fplace='/gnn.fplace',fplace_sha256='candidate',sources={'blif':'hash'})
        self.folder=self.root/'runs/test/w100/s251/C_default_refine'

    def row(self,status='success',code=0,folder=None):
        value=dict(signature=r.signature(self.a,self.identity,self.case,251,'C_default_refine'),
            design='test',width=100,seed=251,method='C_default_refine',status=status,
            process_returncode=code,wall_seconds=7200 if code==124 else 100,wirelength=100,cpd=10)
        r.f.write((folder or self.folder)/'trial.json',value)
        return value

    def test_reuses_success_without_launch_and_preserves_original_bytes(self):
        self.row();before=(self.folder/'trial.json').read_bytes()
        with patch.object(r,'run_process',side_effect=AssertionError('must not rerun')):
            result=r.run_trial(self.a,None,self.identity,(self.case,251,'C_default_refine',self.folder))
        self.assertEqual(result['status'],'success')
        self.assertTrue(result['reused'])
        self.assertEqual((self.folder/'trial.json').read_bytes(),before)

    def test_legacy_timeout_is_terminal_and_timing_skip_is_not_a_measurement(self):
        self.row('other_failure',124)
        with patch.object(r,'run_process',side_effect=AssertionError('must not rerun')):
            result=r.run_trial(self.a,None,self.identity,(self.case,251,'C_default_refine',self.folder))
            timing=self.root/'timing/test/w100/s251/C_default_refine'
            skipped=r.run_trial(self.a,None,self.identity,(self.case,251,'C_default_refine',timing),True)
        self.assertEqual(result['status'],'timeout')
        self.assertEqual(r.f.read(self.folder/'trial.json')['status'],'other_failure')
        self.assertEqual(skipped['status'],'not_run_after_scan_timeout')
        self.assertIsNone(skipped['wall_seconds'])
        self.assertFalse((timing/'trial.json').exists())
        self.assertEqual(r.observation(timing)['status'],'not_run_after_scan_timeout')
        # The server's absolute path may be unavailable after Git synchronization.
        record=r.f.read(timing/'outcome.json');record['source_trial']='/old/server/path/trial.json'
        r.f.write(timing/'outcome.json',record)
        self.assertEqual(r.observation(timing)['status'],'not_run_after_scan_timeout')

    def test_changed_binary_rejected_even_for_prior_success(self):
        self.row()
        with self.assertRaises(ValueError):
            r.run_trial(self.a,None,{'binary':'new'},(self.case,251,'C_default_refine',self.folder))

    def test_retry_archives_old_attempt_before_launch(self):
        self.row('other_failure',124)
        r.f.write(self.folder/'evidence.json',{'keep':'yes'})
        self.a.retry_timeouts=True
        with patch.object(r,'run_process',return_value={'status':'success'}) as launch:
            r.run_trial(self.a,None,self.identity,(self.case,251,'C_default_refine',self.folder))
        self.assertEqual(launch.call_count,1)
        saved=list((self.root/'_attempts').rglob('evidence.json'))
        self.assertEqual(len(saved),1)
        self.assertEqual(r.f.read(saved[0]),{'keep':'yes'})

    def test_interrupted_folder_archived_automatically(self):
        r.f.write(self.folder/'started.json',r.signature(self.a,self.identity,self.case,251,'C_default_refine'))
        with patch.object(r,'run_process',return_value={'status':'success'}) as launch:
            r.run_trial(self.a,None,self.identity,(self.case,251,'C_default_refine',self.folder))
        self.assertTrue(launch.called)
        self.assertEqual(len(list((self.root/'_attempts').rglob('started.json'))),1)

    def test_timeout_below_success_cannot_be_reported_as_exact_wmin(self):
        rows=[dict(width=80,status='unroutable'),dict(width=90,status='timeout'),dict(width=100,status='success')]
        result=r.width_bounds([80,90,100],rows)
        self.assertIsNone(result['w_min_grid'])
        self.assertEqual(result['observed_min_success_width'],100)
        self.assertEqual(result['possible_min_on_grid'],90)
        rows[1]['status']='unroutable'
        self.assertEqual(r.width_bounds([80,90,100],rows)['w_min_grid'],100)

    def test_reports_unknowns_and_excluded_pairs(self):
        generation=self.root/'generation.json'
        r.f.write(generation,dict(timing={k:1 for keys in r.f.SCOPES.values() for k in keys}))
        cases=[dict(self.case,width=w,generation=str(generation)) for w in (80,90,100)]
        r.f.write(self.root/'plan.json',dict(cases=cases,seeds=[251,252,253]))
        for phase in ('runs','timing'):
            for c in cases:
                for s in (251,252,253):
                    for m in r.f.METHODS:
                        folder=self.root/phase/'test'/f'w{c["width"]}'/f's{s}'/m
                        value=dict(width=c['width'],method=m,status='success',process_returncode=0,
                                   wall_seconds=100,wirelength=100,cpd=10)
                        if m=='C_default_refine' and c['width']==80:
                            if phase=='runs':value.update(status='other_failure',process_returncode=124)
                            else:value.update(status='not_run_after_scan_timeout',wall_seconds=None)
                        r.f.write(folder/('outcome.json' if value['status'].startswith('not_run') else 'trial.json'),value)
        r.report(self.a)
        report=r.f.read(self.root/'resume_evaluation_report.json')
        self.assertTrue(report['complete'])
        self.assertFalse(report['all_scan_route_outcomes_known'])
        self.assertFalse(report['all_timing_executed'])
        c=next(w for w in report['width_results'] if w['method']=='C_default_refine')
        self.assertIsNone(c['w80'])
        pair=next(p for p in report['runtime_comparisons'] if p['candidate']=='C_default_refine' and p['baseline']=='A_default')
        self.assertEqual(pair['joint_success_pairs'],3)
        self.assertEqual(pair['selected_pair_count'],3)
        self.assertEqual(sum(pair['excluded_pairs'].values()),0)
        self.assertGreater(report['timing_selection_progress']['existing_records_outside_scope'],0)

    @unittest.skipUnless(os.name=='posix','real process groups require Linux/WSL')
    def test_execute_finishes_other_cases_after_timeout(self):
        paths={}
        for name in ('blif','packed','fplace'):
            path=self.root/name;path.write_text(name);paths[name]=str(path)
        c=dict(self.case,**paths,sources=dict(mapped_blif=r.f.sha(paths['blif']),packed_net=r.f.sha(paths['packed'])),
               fplace_sha256=r.f.sha(paths['fplace']))
        r.f.write(self.root/'plan.json',dict(signature=dict(code=r.f.direct.implementation()),cases=[c],seeds=[251,252,253]))
        self.a.jobs=2
        backend=SimpleNamespace(root=str(self.root),binary_hash='binary',arch_hash='arch')
        def trial(a,b,identity,task,timing):
            return dict(status='timeout' if task[2]=='C_default_refine' else 'success')
        with patch.dict(os.environ,{'VTR_ROOT':str(self.root)}), \
             patch.object(r.f.direct,'DirectBackend',return_value=backend), \
             patch.object(r.f,'verify',return_value='hook'),patch.object(r,'run_trial',side_effect=trial) as run:
            r.execute(self.a)
        self.assertEqual(run.call_count,12)
        session=r.f.read(next((self.root/'resume_sessions').glob('*.json')))
        self.assertEqual(session['status_counts'],{'timeout':3,'success':9})

    @unittest.skipUnless(os.name=='posix','real process groups require Linux/WSL')
    def test_real_timeout_kills_group_and_returns_record_instead_of_raising(self):
        packed=self.root/'source.net';packed.write_text('input')
        c=dict(self.case,packed=str(packed))
        a=SimpleNamespace(**dict(vars(self.a),timeout=1,heartbeat=1))
        code='import time; print("## AP Global Placer",flush=True); time.sleep(30)'
        with patch.object(r.f,'command',return_value=[sys.executable,'-u','-c',code]):
            result=r.run_process(a,None,self.identity,c,251,'C_default_refine',self.folder)
        self.assertEqual(result['status'],'timeout')
        self.assertEqual(result['process_returncode'],124)
        self.assertEqual(result['last_observed_stage'],'global_placement')
        self.assertFalse(r.process_alive(self.folder))


if __name__=='__main__':unittest.main()
