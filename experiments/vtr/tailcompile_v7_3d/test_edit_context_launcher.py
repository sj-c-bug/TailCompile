"""012 real Bash contracts: read-only preflight, no P&R, bounded output paths."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE=Path(__file__).resolve().parent


@unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Linux launcher contract')
class EditContextLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='edit-context-launch-')
        self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name).resolve()/'TailCompile'
        self.module=self.repo/'experiments/vtr/tailcompile_v7_3d';self.module.mkdir(parents=True)
        (self.repo/'REPOSITORY_WORKFLOW.md').write_text('fixture')
        self.launcher=self.module/'run_edit_context_campaign.sh'
        shutil.copyfile(HERE/'run_edit_context_campaign.sh',self.launcher)
        self.source=self.module/'runs/development_011_residual';self.source.mkdir(parents=True)
        (self.source/'keep.json').write_text('{"read_only":true}')
        self.out=self.module/'runs/development_012_edit_context';self.calls=self.repo/'calls.jsonl'
        (self.module/'edit_context_campaign.py').write_text('''import json, os, sys
from pathlib import Path
stage=sys.argv[1]
record=dict(stage=stage,args=sys.argv[2:],omp=os.environ['OMP_NUM_THREADS'],
    tmp=os.environ.get('TMPDIR'),cache=os.environ.get('XDG_CACHE_HOME'))
with open(os.environ['FAKE_CALLS'],'a') as f: f.write(json.dumps(record)+'\\n')
if os.environ.get('FAIL_STAGE')==stage: raise SystemExit(19)
''')
        self.env={k:v for k,v in os.environ.items() if not k.startswith('TAILCOMPILE_')}
        self.env.update(FAKE_CALLS=str(self.calls),TAILCOMPILE_3D_PYTHON=sys.executable,
            TAILCOMPILE_V7_RESIDUAL_OUT=str(self.source),TAILCOMPILE_V7_OUT=str(self.repo/'old_2d'),
            TAILCOMPILE_SCRATCH=str(self.repo/'old_scratch'),OMP_NUM_THREADS='99')

    def launch(self,*args):
        return subprocess.run(['bash',str(self.launcher),*args],env=self.env,capture_output=True,text=True,timeout=20)

    def records(self):
        return [json.loads(x) for x in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def test_plan_and_check_are_read_only(self):
        for stage in ('plan','check'):
            result=self.launch(stage);self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            self.assertFalse(self.out.exists());self.assertFalse((self.repo/'logs').exists())
            self.assertFalse((self.repo/'vtr_work').exists())
        self.assertEqual([r['stage'] for r in self.records()],['plan','check'])

    def test_all_is_offline_and_preserves_old_outputs(self):
        before=(self.source/'keep.json').read_bytes();self.env['TAILCOMPILE_3D_TRAIN_DEVICE']='cuda:1'
        result=self.launch('all');self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        rows=self.records();self.assertEqual([r['stage'] for r in rows],['check','prepare','train','report'])
        for row in rows:
            args=dict(zip(row['args'][::2],row['args'][1::2]))
            self.assertEqual(args['--source'],str(self.source));self.assertEqual(args['--out'],str(self.out))
            self.assertEqual(args['--device'],'cuda:1');self.assertNotIn('--backend',args)
            self.assertEqual(row['omp'],'1')
            self.assertEqual(Path(row['tmp']),self.repo/'vtr_work/tailcompile_v7_3d/edit_context/tmp')
        self.assertEqual((self.source/'keep.json').read_bytes(),before)
        self.assertFalse((self.repo/'old_2d').exists());self.assertFalse((self.repo/'old_scratch').exists())

    def test_failure_stops_later_stages(self):
        self.env['FAIL_STAGE']='train';result=self.launch('all')
        self.assertEqual(result.returncode,19,result.stdout+result.stderr)
        self.assertEqual([r['stage'] for r in self.records()],['check','prepare','train'])
        self.assertIn('V7_EDIT_CONTEXT_FAILED stage=train exit=19',result.stderr)
        self.assertNotIn('GROUP_COMPLETE',result.stdout)

    def test_old_output_rejected_before_background_log_creation(self):
        self.env['TAILCOMPILE_V7_EDIT_CONTEXT_OUT']=str(self.source)
        result=self.launch('all','--background');self.assertEqual(result.returncode,2)
        self.assertFalse((self.repo/'logs').exists());self.assertEqual(self.records(),[])

    def test_source_nested_inside_output_rejected(self):
        self.env['TAILCOMPILE_V7_EDIT_CONTEXT_SOURCE']=str(self.out/'borrowed')
        result=self.launch('prepare');self.assertEqual(result.returncode,2)
        self.assertIn('disjoint',result.stderr);self.assertFalse(self.out.exists())

    def test_escaping_cache_symlink_rejected(self):
        scratch=self.repo/'vtr_work/tailcompile_v7_3d/edit_context';scratch.mkdir(parents=True)
        outside=Path(self.tmp.name)/'outside';outside.mkdir()
        (scratch/'tmp').symlink_to(outside,target_is_directory=True)
        result=self.launch('prepare');self.assertEqual(result.returncode,2)
        self.assertEqual(self.records(),[]);self.assertEqual(list(outside.iterdir()),[])
        self.assertFalse(self.out.exists())

    def test_background_log_pid_and_completion(self):
        result=self.launch('prepare','--background');self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        values=dict(line.split('=',1) for line in result.stdout.splitlines())
        self.assertTrue(Path(values['LOG']).is_file());self.assertTrue(Path(values['PIDFILE']).is_file())
        self.assertEqual(values['OUT'],str(self.out))
        for _ in range(100):
            text=Path(values['LOG']).read_text()
            if 'V7_EDIT_CONTEXT_GROUP_COMPLETE' in text:break
            time.sleep(.02)
        self.assertIn('V7_EDIT_CONTEXT_GROUP_COMPLETE',text)
        self.assertEqual([r['stage'] for r in self.records()],['prepare'])

    def test_bad_arguments_do_not_start_work(self):
        for argv in [('feedback',),('check','--background'),('prepare','--invalid')]:
            self.assertEqual(self.launch(*argv).returncode,2)
        self.env['TAILCOMPILE_3D_TRAIN_DEVICE']='cuda:1junk'
        self.assertEqual(self.launch('prepare').returncode,2);self.assertEqual(self.records(),[])


if __name__=='__main__':unittest.main()
