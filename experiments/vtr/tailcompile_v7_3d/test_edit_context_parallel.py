"""012 independent fit processes: concurrency, cancellation and determinism."""
from __future__ import annotations

import json
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import edit_context_campaign as campaign
from io_utils import read, write, sha


class EditContextParallelTests(unittest.TestCase):
    def folder(self):
        return tempfile.TemporaryDirectory(dir=campaign.HERE/'runs',
                                          prefix='development_012_parallel_test_')

    def test_execution_limits_do_not_oversubscribe_gpu(self):
        self.assertEqual(campaign.execution_options('cpu')['jobs'],4)
        self.assertEqual(campaign.execution_options('cuda')['jobs'],1)
        self.assertEqual(campaign.execution_options('cpu',8)['threads_per_worker'],1)
        for jobs in (0,-1,True,1.5):
            with self.assertRaises(ValueError):campaign.execution_options('cpu',jobs)
        with self.assertRaisesRegex(ValueError,'CUDA'):
            campaign.execution_options('cuda:0',2)

    def test_real_process_intervals_overlap_and_results_keep_planned_order(self):
        main_thread=threading.get_ident();callbacks=[]
        code=('import os,sys,time,json; start=time.time(); time.sleep(float(sys.argv[2])); '
              'print(json.dumps(dict(task=int(sys.argv[1]),pid=os.getpid(),start=start,end=time.time())))')
        def launch(task):
            index,delay=task
            done=subprocess.run([sys.executable,'-c',code,str(index),str(delay)],
                                capture_output=True,text=True,check=True,timeout=10)
            return json.loads(done.stdout)
        def complete(index,value):callbacks.append((index,threading.get_ident()))
        values=campaign.parallel_fit_map([(0,.5),(1,.05),(2,.05)],jobs=2,
                                         run_task=launch,on_complete=complete)
        self.assertEqual([v['task'] for v in values],[0,1,2])
        self.assertEqual(len({v['pid'] for v in values}),3)
        self.assertLess(max(values[0]['start'],values[1]['start']),
                        min(values[0]['end'],values[1]['end']))
        self.assertEqual({thread for _,thread in callbacks},{main_thread})
        self.assertEqual({index for index,_ in callbacks},{0,1,2})
        events=sorted((stamp,delta) for value in values for stamp,delta in
                      ((value['start'],1),(value['end'],-1)))
        active=peak=0
        for _,delta in events:active+=delta;peak=max(active,peak)
        self.assertEqual(peak,2)

    def test_first_failure_cancels_active_work_without_submitting_rest(self):
        started=[];both_started=threading.Event();cancelled=threading.Event()
        lock=threading.Lock();cancel_calls=[]
        def work(task):
            with lock:
                started.append(task)
                if len(started)==2:both_started.set()
            if not both_started.wait(5):raise AssertionError('expected two active tasks')
            if task==0:raise RuntimeError('intentional fit failure')
            if not cancelled.wait(5):raise AssertionError('active worker was not cancelled')
            return 'cancelled'
        def cancel():cancel_calls.append(True);cancelled.set()
        with self.assertRaisesRegex(RuntimeError,'intentional fit failure'):
            campaign.parallel_fit_map(list(range(8)),jobs=2,run_task=work,cancel_running=cancel)
        self.assertEqual(set(started),{0,1})
        self.assertEqual(cancel_calls,[True])
        self.assertTrue(cancelled.is_set())

    def test_changed_cache_pin_fails_before_output_creation(self):
        with self.folder() as tmp:
            folder=Path(tmp);cache=folder/'cache.pt';cache.write_bytes(b'changed cache')
            out=folder/'must-not-exist'
            with self.assertRaisesRegex(ValueError,'cache path/bytes changed'):
                campaign.fit_worker(dict(cache=str(cache),cache_file_sha256='0'*64),out)
            self.assertFalse(out.exists())

    def test_resume_does_not_freeze_volatile_available_memory(self):
        with self.folder() as tmp:
            folder=Path(tmp)
            banks=[dict(condition_id=f'{f}/{a}/w64',family=f,architecture=a)
                   for f in ('alpha','beta') for a in ('H0Z0','H0Z1','H1Z0','H1Z1')]
            cache=dict(banks=banks,cache_sha256='a'*64,source_checkpoint_sha256='b'*64)
            write(folder/'prepared_report.json',dict(cache=str(folder/'cache.pt'),cache_file_sha256='c'*64))
            runner=campaign.Campaign(SimpleNamespace(out=folder,device='cpu',jobs=4,fit_timeout=120))
            runner.cache=lambda:cache
            # All fits are already complete. Resume must verify and retain them,
            # rather than start a worker because free RAM or actual lanes changed.
            for split in campaign.splits(banks):
                for mode in campaign.MODES:
                    for seed in campaign.TRAIN_SEEDS:
                        fit=dict(split=split,mode=mode,seed=seed,worker_threads=1,
                            training=dict(mode=mode,seed=seed,epochs=80,device='cpu',
                                cache_sha256=cache['cache_sha256'],source_checkpoint_sha256=cache['source_checkpoint_sha256'],
                                train_ids=sorted(split['train_ids']),checkpoint_sha256={}))
                        write(folder/'models'/split['id']/mode/f's{seed}'/'fit_report.json',fit)
            budgets=[dict(effective=4,physical_cores=[0,1,2,3],available_mib=10000),
                     dict(effective=2,physical_cores=[0,1],available_mib=3500)]
            with patch.object(campaign,'resource_budget',side_effect=budgets), \
                 patch.object(campaign,'run_fit_process',side_effect=AssertionError('cached fit rerun')), \
                 contextlib.redirect_stdout(io.StringIO()):
                runner.train()
                protocol=sha(folder/'execution/train/parallel_protocol.json')
                summary=sha(folder/'trained_report.json')
                runner.train()
            self.assertEqual(protocol,sha(folder/'execution/train/parallel_protocol.json'))
            self.assertEqual(summary,sha(folder/'trained_report.json'))
            observations=[read(p) for p in sorted((folder/'execution/train/parallel_runs').glob('attempt_*/metrics.json'))]
            self.assertEqual([v['budget']['available_mib'] for v in observations],[10000,3500])
            self.assertEqual([v['effective_jobs'] for v in observations],[4,2])
            self.assertEqual(read(folder/'train_progress_report.json')['completed'],28)

    @unittest.skipUnless(os.name=='posix','Linux process-group cancellation')
    def test_failed_fit_terminates_owned_sibling_process_group(self):
        with self.folder() as tmp:
            folder=Path(tmp)
            controller=folder/'controller.py'
            controller.write_text('''import json,sys,time
from pathlib import Path
sys.path.insert(0,sys.argv[1])
from edit_context_campaign import parallel_fit_map,_stop_fit_processes
from io_utils import run_trial,write
root=Path(sys.argv[2]);started=[]
child="import os,sys,time; from pathlib import Path; Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(float(sys.argv[2])); raise SystemExit(int(sys.argv[3]))"
def task(index):
    started.append(index)
    result=run_trial(root/f'trial_{index}',[sys.executable,'-c',child,str(root/f'pid_{index}'),'.3' if index==0 else '20','1' if index==0 else '0'],timeout=30,prune=False)
    if result['status']!='complete':raise RuntimeError('intentional child fit failure')
    return result
begin=time.monotonic()
try:
    parallel_fit_map(list(range(5)),jobs=2,run_task=task,cancel_running=_stop_fit_processes)
except RuntimeError as exc:
    assert 'intentional child fit failure' in str(exc)
    write(root/'cancel_report.json',dict(elapsed=time.monotonic()-begin,started=sorted(started)))
else:raise AssertionError('expected original fit error')
''',encoding='utf-8')
            result=subprocess.run([sys.executable,str(controller),str(campaign.HERE),str(folder)],
                                  capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            report=read(folder/'cancel_report.json')
            self.assertEqual(report['started'],[0,1])
            self.assertLess(report['elapsed'],10)
            self.assertFalse((folder/'pid_2').exists())
            for index in (0,1):
                self.assertTrue((folder/f'pid_{index}').is_file())
                pid=int((folder/f'pid_{index}').read_text())
                with self.assertRaises(ProcessLookupError):os.kill(pid,0)

    @unittest.skipUnless(os.name=='posix','Linux/WSL independent Torch workers')
    def test_parallel_real_head_fit_matches_serial_weights_and_predictions(self):
        import torch
        from edit_context_features import FEATURE_SCHEMA
        from train_edit_context import CACHE_SCHEMA,cache_digest,save_cache
        torch.set_num_threads(1)
        with self.folder() as tmp:
            folder=Path(tmp)
            banks=[]
            for family in ('alpha','beta'):
                for architecture in ('H0Z0','H0Z1','H1Z0','H1Z1'):
                    for arm in ('L','Y','J'):
                        trials=[]
                        for index,wl in enumerate((100.,90.,110.)):
                            trials.append({seed:dict(seed=seed,status='success',wirelength=wl,
                                cpd_ns=10.-index*.2) for seed in (11401,11402)})
                        ids=['identity',arm.lower()+'_s1',arm.lower()+'_s2']
                        banks.append(dict(condition_id=f'{family}/{architecture}/w64',family=family,
                            architecture=architecture,arm=arm,
                            global_features=torch.tensor([[1.,0.,0.,.2],[0.,1.,0.,.2],[0.,0.,1.,.2]]),
                            local_features=torch.tensor([[0.,0.,0.],[.4,.5,.2],[-.3,.1,.2]]),
                            candidate_ids=ids,digests=[str(i)*64 for i in range(3)],trials=trials,
                            generated_candidate_count=3,measured_candidate_count=3))
            cache=dict(schema=CACHE_SCHEMA,feature_schema=FEATURE_SCHEMA,banks=banks,
                global_dim=4,local_dim=3,source_checkpoint_sha256='a'*64,backbone_sha256='b'*64,
                source_inputs={},feedback_sha256='c'*64,synthetic_fixture=True,new_pnr=0)
            cache['cache_sha256']=cache_digest(cache)
            cache_path=Path(save_cache(cache,folder/'cache.pt'))
            split=campaign.splits(banks)[0]
            implementation={name:sha(campaign.HERE/name) for name in campaign.CODE}
            requests=[dict(cache=str(cache_path),cache_file_sha256=sha(cache_path),
                cache_sha256=cache['cache_sha256'],split=split,mode='E',seed=seed,device='cpu',
                implementation=implementation,fit_timeout=1800) for seed in (12301,12302)]
            worker_code=('import sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); '
                'from io_utils import read; from edit_context_campaign import fit_worker; '
                'fit_worker(read(sys.argv[2]),Path(sys.argv[3]))')
            def launch(item):
                index,request=item;target=folder/f'worker_{index}'
                request_path=folder/f'request_{index}.json';write(request_path,request)
                env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
                         NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
                done=subprocess.run([sys.executable,'-c',worker_code,str(campaign.HERE),
                    str(request_path),str(target)],env=env,capture_output=True,text=True,timeout=90)
                (folder/f'worker_{index}.log').write_text(done.stdout+done.stderr,encoding='utf-8')
                self.assertEqual(done.returncode,0,done.stdout+done.stderr)
                self.assertIn('TAILCOMPILE_EDIT_CONTEXT_FIT_COMPLETE',done.stdout)
                return read(target/'fit_report.json')
            parallel=campaign.parallel_fit_map(list(enumerate(requests)),jobs=2,run_task=launch)
            serial=launch((2,requests[0]))
            self.assertNotEqual(parallel[0]['worker_pid'],parallel[1]['worker_pid'])
            self.assertNotIn(os.getpid(),[r['worker_pid'] for r in parallel])
            self.assertTrue(all(r['worker_threads']==1 for r in parallel+[serial]))
            self.assertEqual(parallel[0]['training']['initial_state_sha256'],serial['training']['initial_state_sha256'])
            self.assertEqual(parallel[0]['training']['final_state_sha256'],serial['training']['final_state_sha256'])
            self.assertEqual(parallel[0]['evaluation'],serial['evaluation'])
            self.assertEqual(parallel[0]['seed_evaluation'],serial['seed_evaluation'])
            self.assertEqual(parallel[0]['training']['calibration'],serial['training']['calibration'])


if __name__=='__main__':unittest.main()
