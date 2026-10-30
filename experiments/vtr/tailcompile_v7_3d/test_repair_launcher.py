"""Qualified learning stages keep old cohorts read-only and fail as a group."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux learning launcher')
class RepairLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=HERE/'runs', prefix='repair-launch-')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()/'TailCompile'
        self.module = self.repo/'experiments/vtr/tailcompile_v7_3d'
        self.module.mkdir(parents=True)
        self.launcher = self.module/'run_learning_repair.sh'
        shutil.copyfile(HERE/'run_learning_repair.sh', self.launcher)
        (self.repo/'REPOSITORY_WORKFLOW.md').write_text('fixture')
        self.source = self.module/'runs/development_004_capacity'
        self.source.mkdir(parents=True)
        for name in ('protocol.json', 'campaign.json', 'prepared.json', 'conditions.json',
                     'conditions_manifest.json', 'binary_identity.json'):
            (self.source/name).write_text('{"keep": true}')
        self.out = self.module/'runs/development_006_learning_repair'
        self.prior = self.module/'runs/development_005_qualified_learning'
        for name in ('selection.json','teacher_records.json','models/pretrain/completed.json','models/pretrain/input_signature.json'):
            target=self.prior/name
            target.parent.mkdir(parents=True,exist_ok=True)
            target.write_text('{}')
        self.calls = self.repo/'calls.jsonl'
        (self.module/'repair_learning.py').write_text('''import json, os, sys
from pathlib import Path
stage = sys.argv[1]
record = dict(stage=stage, args=sys.argv[2:],
              tmp=os.environ['TMPDIR'], cache=os.environ['XDG_CACHE_HOME'],
              omp=os.environ['OMP_NUM_THREADS'], mkl=os.environ['MKL_NUM_THREADS'],
              legacy_out=os.environ.get('TAILCOMPILE_V7_OUT'))
assert Path(record['tmp']).is_dir()
assert Path(record['cache']).is_dir()
with open(os.environ['FAKE_CALLS'], 'a') as stream:
    stream.write(json.dumps(record) + '\\n')
if os.environ.get('FAIL_STAGE') == stage:
    raise SystemExit(17)
''')
        self.env = {k:v for k,v in os.environ.items() if not k.startswith('TAILCOMPILE_')}
        import sys
        self.env.update(FAKE_CALLS=str(self.calls), TAILCOMPILE_3D_PYTHON=sys.executable,
                        TAILCOMPILE_V7_OUT=str(self.repo/'old_2d_cohort'),
                        TAILCOMPILE_V7_CONFIG=str(self.repo/'old_2d_config'),
                        TAILCOMPILE_SCRATCH=str(self.repo/'old_2d_scratch'),
                        OMP_NUM_THREADS='99', MKL_NUM_THREADS='99')

    def run_launcher(self, *args):
        return subprocess.run(['bash', str(self.launcher), *args], env=self.env,
                              capture_output=True, text=True, timeout=15)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()]

    def test_teachers_group_inherits_python_and_uses_new_isolated_paths(self):
        before = {p:p.read_bytes() for p in self.source.iterdir()}
        result = self.run_launcher('cpu-prepare')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        rows = self.records()
        self.assertEqual([r['stage'] for r in rows], ['prepare'])
        for row in rows:
            args = dict(zip(row['args'][::2], row['args'][1::2]))
            self.assertEqual(args, dict([
                ('--source', str(self.source)), ('--prior',str(self.prior)), ('--out', str(self.out)),
                ('--jobs', '4'), ('--per-job-mib', '4096'), ('--timeout', '3600'),
                ('--device', 'cpu')]))
            self.assertEqual(row['omp'], '1')
            self.assertEqual(row['mkl'], '1')
            self.assertEqual(row['legacy_out'], str(self.repo/'old_2d_cohort'))
            self.assertEqual(Path(row['tmp']), self.repo/'vtr_work/tailcompile_v7_3d/learning_repair/tmp')
        self.assertEqual(before, {p:p.read_bytes() for p in self.source.iterdir()})
        self.assertFalse((self.repo/'old_2d_scratch').exists())
        self.assertFalse((self.repo/'old_2d_cohort').exists())
        self.assertIn('NEXT=cpu-feedback', result.stdout)

    def test_gpu_default_and_explicit_cpu_training(self):
        result = self.run_launcher('gpu-train')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        row = self.records()[-1]
        self.assertEqual(row['args'][row['args'].index('--device')+1], 'cuda')
        self.env['TAILCOMPILE_3D_TRAIN_DEVICE'] = 'cpu'
        result = self.run_launcher('gpu-train')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        row = self.records()[-1]
        self.assertEqual(row['args'][row['args'].index('--device')+1], 'cpu')

    def test_evaluation_stops_after_failed_validation(self):
        self.env['FAIL_STAGE'] = 'validate'
        result = self.run_launcher('cpu-evaluate')
        self.assertEqual(result.returncode, 17, result.stdout+result.stderr)
        self.assertEqual([r['stage'] for r in self.records()], ['validate'])
        self.assertIn('V7_REPAIR_FAILED stage=validate exit=17', result.stderr)
        self.assertNotIn('GROUP_COMPLETE', result.stdout)

    def test_evaluation_executes_all_three_stages_in_order(self):
        result = self.run_launcher('cpu-evaluate', '--foreground')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual([r['stage'] for r in self.records()], ['validate', 'compare', 'report'])
        self.assertIn('no formal acceptance claim', result.stdout)

    def test_all_executes_order_and_uses_cuda_for_training(self):
        result=self.run_launcher('all')
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        rows=self.records()
        self.assertEqual([r['stage'] for r in rows],['prepare','feedback','train','validate','compare','report'])
        training=next(r for r in rows if r['stage']=='train')
        self.assertEqual(training['args'][training['args'].index('--device')+1],'cuda')

    def test_prior_overlap_rejected_before_launch(self):
        self.env['TAILCOMPILE_V7_REPAIR_OUT']=str(self.prior/'overwrite')
        result=self.run_launcher('all','--background')
        self.assertEqual(result.returncode,2)
        self.assertIn('disjoint',result.stderr)
        self.assertFalse(self.calls.exists())

    def test_source_overlap_rejected_before_any_output_creation(self):
        self.env['TAILCOMPILE_V7_REPAIR_OUT'] = str(self.source/'overwrite')
        result = self.run_launcher('prepare', '--background')
        self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
        self.assertIn('disjoint', result.stderr)
        self.assertFalse((self.source/'overwrite').exists())
        self.assertFalse((self.repo/'logs').exists())
        self.assertFalse(self.calls.exists())

    def test_log_symlink_escape_rejected_before_background_launch(self):
        external = Path(self.tmp.name)/'outside'
        external.mkdir()
        (self.repo/'logs').symlink_to(external, target_is_directory=True)
        result = self.run_launcher('prepare', '--background')
        self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
        self.assertIn('namespace', result.stderr)
        self.assertEqual(list(external.iterdir()), [])
        self.assertFalse(self.out.exists())
        self.assertFalse(self.calls.exists())

    def test_cache_symlink_escape_rejected_before_python(self):
        scratch = self.repo/'vtr_work/tailcompile_v7_3d/learning_repair'
        scratch.mkdir(parents=True)
        external = Path(self.tmp.name)/'outside'
        external.mkdir()
        (scratch/'tmp').symlink_to(external, target_is_directory=True)
        result = self.run_launcher('prepare')
        self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
        self.assertIn('symlink', result.stderr)
        self.assertFalse(self.calls.exists())
        self.assertFalse(self.out.exists())

    def test_background_returns_pid_and_durable_log(self):
        result = self.run_launcher('cpu-prepare', '--background')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        values = dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        log = Path(values['LOG'])
        pidfile = Path(values['PIDFILE'])
        self.assertEqual(log.parent, self.repo/'logs')
        self.assertEqual(pidfile.read_text().strip(), values['PID'])
        deadline = time.monotonic()+10
        while time.monotonic() < deadline:
            contents = log.read_text()
            if 'V7_REPAIR_GROUP_COMPLETE' in contents or 'V7_REPAIR_FAILED' in contents:
                break
            time.sleep(.05)
        self.assertIn('V7_REPAIR_GROUP_COMPLETE group=cpu-prepare', log.read_text())
        self.assertEqual([r['stage'] for r in self.records()], ['prepare'])

    def test_invalid_stage_or_resource_does_not_launch(self):
        result = self.run_launcher('calibrate')
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.calls.exists())
        self.env['TAILCOMPILE_V7_JOBS'] = '0'
        result = self.run_launcher('prepare')
        self.assertEqual(result.returncode, 2)
        self.assertFalse(self.out.exists())
        self.assertFalse(self.calls.exists())

    def test_group_lock_prevents_duplicate_dispatch(self):
        import fcntl
        self.out.mkdir()
        with (self.out/'.learning-repair-launch.lock').open('w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX|fcntl.LOCK_NB)
            result = self.run_launcher('prepare')
        self.assertEqual(result.returncode, 2, result.stdout+result.stderr)
        self.assertIn('Another qualified-learning', result.stderr)
        self.assertFalse(self.calls.exists())


if __name__ == '__main__':
    unittest.main()
