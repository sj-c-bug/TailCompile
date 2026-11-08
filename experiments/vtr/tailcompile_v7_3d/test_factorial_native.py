"""017 actual child process, resume, hard deadline and cleanup contracts."""
import os
import json
import hashlib
import signal
import subprocess
import sys
import time
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from contextlib import nullcontext
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from campaign import resource_budget
from io_utils import read, write, sha
from factorial_campaign import arguments, Controller, verify_measurements, prepare_vertical_summary
from factorial_native import run_native, refinement_deadline, refine_arms, METHODS, SEEDS

HERE=Path(__file__).resolve().parent


class FactorialNativeTests(unittest.TestCase):
    @unittest.skipUnless(hasattr(signal,'setitimer'),'POSIX hard deadline')
    def test_noncooperative_refinement_is_interrupted_without_a_fallback(self):
        previous=signal.getsignal(signal.SIGALRM)
        started=time.perf_counter()
        with self.assertRaisesRegex(TimeoutError,'hard deadline'):
            with refinement_deadline(.03):
                while True:
                    pass
        self.assertLess(time.perf_counter()-started,2.)
        self.assertEqual(signal.getsignal(signal.SIGALRM),previous)
        self.assertEqual(signal.getitimer(signal.ITIMER_REAL)[0],0.)

    def test_budget_configuration_and_namespace_rejected_before_any_run(self):
        for flag,value in (('--refine-max-seconds','nan'),('--refine-max-seconds','21'),
                           ('--refine-setup-seconds','61'),('--refine-setup-seconds','inf'),
                           ('--refine-hard-timeout','121'),('--refine-hard-timeout','80'),
                           ('--worker-timeout','599'),('--worker-timeout','901')):
            with self.subTest(flag=flag,value=value),self.assertRaises(SystemExit):
                arguments(['all',flag,value])
        args=arguments(['plan'])
        self.assertEqual(args.refine_max_seconds,20.)
        self.assertEqual(args.refine_setup_seconds,60.)
        self.assertEqual(args.refine_hard_timeout,120)
        self.assertEqual(args.worker_timeout,600)
        self.assertEqual(METHODS,('Default','D','W','WS','T','TS'))
        self.assertEqual(SEEDS,(17101,17102,17103))

    def test_old_cohort_is_never_a_write_target(self):
        with self.assertRaises(SystemExit):
            arguments(['all','--out',str(HERE/'runs/development_014_prospective_joint')])

    def test_parallel_really_dispatches_independent_tasks(self):
        barrier=Barrier(3)
        controller=SimpleNamespace(budget={'effective':3},guard=lambda:None)
        def work(i):
            barrier.wait(timeout=5)
            return i*i
        self.assertEqual(sorted(Controller.parallel(controller,range(3),work)),[0,1,4])

    def test_incomplete_setup_aborts_entire_candidate_panel(self):
        failed=dict(refinement_diagnostics=dict(setup_complete=False))
        with patch('factorial_native.refinement_deadline',return_value=nullcontext()), \
             patch('factorial_refinement.prepare_problem',return_value=object()), \
             patch('factorial_refinement.refine',return_value=failed) as refine:
            with self.assertRaisesRegex(RuntimeError,'incomplete refinement setup'):
                refine_arms({}, {}, {}, {}, {},120,'case')
            self.assertEqual(refine.call_count,1)

    def test_arms_with_different_D_normalizers_are_rejected_before_lowering(self):
        normal=dict(refinement_diagnostics=dict(setup_complete=True,normalizers=dict(wirelength=1.,timing=1.)))
        actions=[deepcopy(normal) for _ in range(4)]
        actions[-1]['refinement_diagnostics']['normalizers']['timing']=2.
        with patch('factorial_native.refinement_deadline',return_value=nullcontext()), \
             patch('factorial_refinement.prepare_problem',return_value=object()), \
             patch('factorial_refinement.refine',side_effect=actions):
            with self.assertRaisesRegex(RuntimeError,'did not share D normalizers'):
                refine_arms({}, {}, {}, {}, {},120,'case')

    @unittest.skipUnless(os.name=='posix','Linux child process contract')
    def test_native_consumes_init_continues_and_reclaims_disposable_files(self):
        with tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='development_017_local_native_') as tmp:
            out=Path(tmp)
            binary=out/'fake-vpr'
            binary.write_text('''#!/usr/bin/env python3
import os
from pathlib import Path
if 'TC_GP_INIT' in os.environ:
    print('TAILCOMPILE_V7_GP_INIT_CONSUMED path='+os.environ['TC_GP_INIT'])
    print('TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1')
Path('design.route').write_text('Net 0 (a)\\nNode: 2 CHANZ (1,2,0) to (1,2,1)\\n')
Path('design.net').write_text('temporary')
Path('design.net.post_routing').write_text('temporary')
Path('post_route.fplace').write_text('a 1 2 0 0\\n')
print('Circuit successfully routed')
print('Total wirelength: 10')
print('Final critical path delay: 5 ns')
''')
            binary.chmod(0o755)
            files={k:out/name for k,name in dict(arch='arch.xml',blif='design.blif',rr_graph='rr.xml').items()}
            for file in files.values(): file.write_text(file.name)
            device=out/'device.json';write(device,{})
            manifest=out/'rr_manifest.json'
            write(manifest,dict(binary_sha256='older-producer',channel_width=128,
                architecture_sha256=sha(files['arch']),rr_graph_sha256=sha(files['rr_graph']),graph_sha256=sha(device)))
            fields={k:str(v) for k,v in files.items()}
            fields.update(device=str(device),device_manifest=str(manifest))
            c=dict(id='x/H0Z0/w128',family='x',design='x',architecture='H0Z0',width=128,fields=fields)
            controller=SimpleNamespace(out=out,binary=binary,binary_hash=sha(binary),
                binary_reference=binary,guard=lambda:None,args=SimpleNamespace(timeout=60))
            a=run_native(controller,c,17101,out/'comparison'/c['id']/'s17101/Default','Default')
            init=out/'predictions/TS/init.ap.tsv';init.parent.mkdir(parents=True);init.write_text('a 1 2 0 0\n')
            edits=init.parent/'residual_edits.json';write(edits,dict(grid=[4,4],atoms=[]))
            candidate=dict(ap_init=str(init),ap_init_sha256=sha(init),residual_edits=str(edits),
                residual_edits_sha256=sha(edits),deployed_action_sha256='a'*64)
            folder=out/'comparison'/c['id']/'s17101/TS'
            b=run_native(controller,c,17101,folder,'TS',candidate)
            self.assertEqual(a['execution_signature'],b['execution_signature'])
            self.assertTrue(b['input_consumed'] and b['native_solver_continued'])
            self.assertEqual(b['unique_chanz_nodes'],1)
            work=Path(read(folder/'vpr/manifest.json')['workdir'])
            self.assertFalse((work/'design.net').exists())
            self.assertFalse((work/'design.net.post_routing').exists())
            self.assertFalse((work/'design.route').exists())
            self.assertTrue((work/'payload_pruned.json').is_file())
            self.assertEqual(b,run_native(controller,c,17101,folder,'TS',candidate))
            verify_measurements(out,[a,b])
            mutated=dict(b,wirelength=99.)
            with self.assertRaisesRegex(ValueError,'summary differs'):
                verify_measurements(out,[mutated])
            init.write_text('different\n')
            with self.assertRaisesRegex(ValueError,'payload changed'):
                run_native(controller,c,17101,folder,'TS',candidate)

    @unittest.skipUnless(os.name=='posix' and Path('/proc').is_dir(),'POSIX process groups')
    def test_parallel_cancellation_stops_owned_children_and_dispatch(self):
        """Exercise the actual017 dispatcher and shared subprocess registry."""
        with tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='development_017_local_cancel_') as tmp:
            out=Path(tmp)
            child=('import os,signal,time;from pathlib import Path;'
                   'signal.signal(signal.SIGTERM,signal.SIG_IGN);'
                   'Path(__import__("sys").argv[1]).write_text(str(os.getpid()));time.sleep(30)')
            script=(
                'import sys;from pathlib import Path;from types import SimpleNamespace\n'
                f'sys.path.insert(0,{str(HERE)!r})\n'
                'from factorial_campaign import Controller\n'
                'from io_utils import run_trial,install_signal_handlers,raise_if_cancelled,write\n'
                f'out=Path({str(out)!r});child={child!r}\n'
                'install_signal_handlers()\n'
                'controller=SimpleNamespace(budget={"effective":2},guard=raise_if_cancelled)\n'
                'def work(i):\n'
                ' return run_trial(out/f"trial{i}",[sys.executable,"-c",child,str(out/f"child{i}.pid")],timeout=60)\n'
                'try:\n'
                ' Controller.parallel(controller,range(4),work)\n'
                'except KeyboardInterrupt:\n'
                ' write(out/"stopped.json",{"cancelled":True})\n')
            unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'],start_new_session=True)
            process=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
            try:
                deadline=time.monotonic()+8
                while not all((out/f'child{i}.pid').is_file() for i in range(2)):
                    if process.poll() is not None:
                        stdout,stderr=process.communicate()
                        self.fail('parallel controller exited before dispatch: '+stdout+stderr)
                    if time.monotonic()>deadline:
                        self.fail('parallel children did not start')
                    time.sleep(.01)
                pids=[int((out/f'child{i}.pid').read_text()) for i in range(2)]
                os.kill(process.pid,signal.SIGTERM)
                stdout,stderr=process.communicate(timeout=8)
                self.assertEqual(process.returncode,0,stdout+stderr)
                self.assertTrue(read(out/'stopped.json')['cancelled'])
                self.assertFalse((out/'trial2').exists())
                self.assertFalse((out/'trial3').exists())
                for i in range(2):
                    self.assertEqual(read(out/f'trial{i}/manifest.json')['status'],'interrupted')
                for pid in pids:
                    stat=Path(f'/proc/{pid}/stat')
                    self.assertTrue(not stat.exists() or stat.read_text().split()[2] in ('Z','X'))
                self.assertIsNone(unrelated.poll())
            finally:
                for proc in (process,unrelated):
                    if proc.poll() is None:
                        proc.kill();proc.wait(timeout=5)


class FactorialCacheTests(unittest.TestCase):
    """A reused cache needs the complete pre-measurement receipt chain."""
    def setUp(self):
        import vertical_risk
        owned=tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='development_017_local_cache_')
        source=tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='development_016_local_cache_')
        self.addCleanup(owned.cleanup);self.addCleanup(source.cleanup)
        self.out=Path(owned.name);self.source=Path(source.name);self.cid='x/H0Z0/w128'
        self.target=self.out/'predictions'/self.cid/'vertical_summary.json'
        self.cache=self.source/'predictions'/self.cid/'vertical_summary.json'
        self.rr=self.out/'rr.xml';self.rr.write_text('hashable but deliberately not parseable XML')
        self.device=dict(grid_width=4,grid_height=4,layers=2,tile_size=4)
        device_path=self.out/'device.json';write(device_path,self.device)
        self.summary=dict(signature=dict(schema=vertical_risk.SCHEMA,
            implementation=sha(Path(vertical_risk.__file__)),rr_graph_sha256=sha(self.rr),
            device_sha256=hashlib.sha256(json.dumps(self.device,sort_keys=True,allow_nan=False).encode()).hexdigest()),pools=[])
        write(self.cache,self.summary)
        request=self.cache.parent/'request.json'
        write(request,dict(device_signature='device017',vertical_summary=str(self.cache),inputs={str(self.cache):sha(self.cache)}))
        report=self.source/'predictions/prediction_report.json'
        write(report,dict(conditions={self.cid:dict(device_signature='device017',request_sha256=sha(request))}))
        write(report.parent/'selection.json',dict(pre_measurement=True,predictions_sha256=sha(report)))
        self.condition=dict(id=self.cid,fields=dict(rr_graph=str(self.rr),device=str(device_path)))

    def prepare(self):
        return prepare_vertical_summary(self.condition,self.target,max_seconds=5,
            expected_device_signature='device017',source_root=self.source)

    def test_verified016_cache_copy_skips_parse_and_old_payload_stays_unchanged(self):
        original=self.cache.read_bytes()
        with patch('vertical_risk.ET.iterparse',side_effect=AssertionError('receipt verified cache needs no parse')):
            receipt=self.prepare()
            self.assertEqual(receipt['mode'],'reused_016')
            self.assertEqual(self.target.read_bytes(),original)
            self.assertEqual(self.cache.read_bytes(),original)
            self.assertEqual(self.prepare(),receipt)

    def test_cache_tamper_with_same_signature_rebuilds_only_owned017(self):
        write(self.cache,dict(self.summary,tampered=True));original=self.cache.read_bytes()
        def derive(rr,device,out,*,max_seconds):
            self.assertFalse(Path(out).exists());write(out,dict(self.summary,rebuilt=True))
        with patch('vertical_risk.prepare_device',side_effect=derive):
            receipt=self.prepare()
        self.assertEqual(receipt['mode'],'rebuilt')
        self.assertIn('differs from frozen request',receipt['reuse_rejected_reasons'][0]['reason'])
        self.assertEqual(self.cache.read_bytes(),original)
        self.assertTrue(read(self.target)['rebuilt'])

    def test_resume_rejects_owned_digest_tamper(self):
        self.prepare();write(self.target,dict(self.summary,tampered=True))
        with self.assertRaisesRegex(ValueError,'frozen payload changed'):
            self.prepare()

    def test_unpublished_owned_copy_can_recover_from_verified_source(self):
        write(self.target,dict(unpublished=True))
        with patch('vertical_risk.ET.iterparse',side_effect=AssertionError('verified source should be used')):
            self.assertEqual(self.prepare()['mode'],'reused_016')
        self.assertEqual(self.target.read_bytes(),self.cache.read_bytes())


if __name__=='__main__':
    unittest.main()
