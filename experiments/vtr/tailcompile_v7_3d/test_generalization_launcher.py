"""013 Bash boundary contracts with an offline mock controller."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux launcher contract')
class GeneralizationLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=HERE/'runs', prefix='generalization-launch-')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()/'TailCompile'
        self.module = self.repo/'experiments/vtr/tailcompile_v7_3d'
        self.module.mkdir(parents=True)
        (self.repo/'REPOSITORY_WORKFLOW.md').write_text('fixture')
        self.launcher = self.module/'run_generalization_campaign.sh'
        shutil.copyfile(HERE/'run_generalization_campaign.sh', self.launcher)
        self.source = self.module/'runs/development_012_edit_context_parallel'
        self.source.mkdir(parents=True)
        self.source_marker = self.source/'keep.json'
        self.source_marker.write_text('{"original_cache":"immutable"}')
        self.out = self.module/'runs/development_013_generalization'
        self.calls = self.repo/'calls.jsonl'
        (self.module/'generalization_campaign.py').write_text('''import json, os, sys, time
stage = sys.argv[1]
record = dict(stage=stage, args=sys.argv[2:], omp=os.environ['OMP_NUM_THREADS'],
    tmp=os.environ.get('TMPDIR'), cache=os.environ.get('XDG_CACHE_HOME'))
with open(os.environ['FAKE_CALLS'], 'a') as stream:
    stream.write(json.dumps(record)+'\\n')
if os.environ.get('HOLD_STAGE') == stage:
    time.sleep(float(os.environ.get('HOLD_SECONDS', '2')))
if os.environ.get('FAIL_STAGE') == stage:
    raise SystemExit(19)
''', encoding='utf-8')
        self.env = {k: v for k, v in os.environ.items() if not k.startswith('TAILCOMPILE_')}
        self.env.update(FAKE_CALLS=str(self.calls), TAILCOMPILE_3D_PYTHON=sys.executable,
                        TAILCOMPILE_V7_EDIT_CONTEXT_OUT=str(self.source),
                        TAILCOMPILE_V7_OUT=str(self.repo/'old_2d'),
                        TAILCOMPILE_SCRATCH=str(self.repo/'old_scratch'), OMP_NUM_THREADS='99')

    def launch(self, *args):
        return subprocess.run(['bash', str(self.launcher), *args], env=self.env,
                              capture_output=True, text=True, timeout=20)

    def records(self):
        return [json.loads(line) for line in self.calls.read_text().splitlines()] if self.calls.exists() else []

    def assert_no_writes(self):
        self.assertEqual(self.records(), [])
        self.assertFalse(self.out.exists())
        self.assertFalse((self.repo/'logs').exists())
        self.assertFalse((self.repo/'vtr_work').exists())

    def background_log(self, *args, marker):
        result = self.launch(*args, '--background')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        values = dict(line.split('=', 1) for line in result.stdout.splitlines())
        self.assertEqual(values['SOURCE'], str(self.source))
        self.assertEqual(values['OUT'], str(self.out))
        self.assertEqual(Path(values['PIDFILE']).read_text().strip(), values['PID'])
        for _ in range(150):
            content = Path(values['LOG']).read_text()
            if marker in content:
                return content
            time.sleep(.02)
        self.fail('background marker missing\n'+content)

    def test_plan_and_check_do_not_create_runtime_directories(self):
        for stage in ('plan', 'check'):
            result = self.launch(stage)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            self.assertFalse(self.out.exists())
            self.assertFalse((self.repo/'logs').exists())
            self.assertFalse((self.repo/'vtr_work').exists())
        self.assertEqual([row['stage'] for row in self.records()], ['plan', 'check'])

    def test_all_has_cpu_parallel_defaults_and_preserves_old_versions(self):
        original = self.source_marker.read_bytes()
        result = self.launch('all')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        self.assertEqual([row['stage'] for row in self.records()], ['check', 'prepare', 'train', 'report'])
        for row in self.records():
            args = dict(zip(row['args'][::2], row['args'][1::2]))
            self.assertEqual(args, {'--source': str(self.source), '--out': str(self.out),
                                    '--device': 'cpu', '--jobs': '16', '--fit-timeout': '1800'})
            self.assertEqual(row['omp'], '1')
            self.assertEqual(Path(row['tmp']), self.repo/'vtr_work/tailcompile_v7_3d/generalization/tmp')
        self.assertEqual(original, self.source_marker.read_bytes())
        self.assertFalse((self.repo/'old_2d').exists())
        self.assertFalse((self.repo/'old_scratch').exists())
        self.assertIn('V7_GENERALIZATION_GROUP_COMPLETE group=all', result.stdout)

    def test_explicit_cpu_jobs_timeout_and_cuda_device_are_passed(self):
        self.env['TAILCOMPILE_V7_GENERALIZATION_JOBS'] = '8'
        self.env['TAILCOMPILE_V7_GENERALIZATION_FIT_TIMEOUT'] = '600'
        result = self.launch('check')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        args = dict(zip(self.records()[0]['args'][::2], self.records()[0]['args'][1::2]))
        self.assertEqual(args['--jobs'], '8')
        self.assertEqual(args['--fit-timeout'], '600')
        del self.env['TAILCOMPILE_V7_GENERALIZATION_JOBS']
        self.env['TAILCOMPILE_3D_TRAIN_DEVICE'] = 'cuda:1'
        result = self.launch('check')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        args = dict(zip(self.records()[-1]['args'][::2], self.records()[-1]['args'][1::2]))
        self.assertEqual(args['--device'], 'cuda:1')
        self.assertEqual(args['--jobs'], '1')

    def test_bad_arguments_fail_before_work(self):
        for argv in [('feedback',), ('check', '--background'), ('all', '--invalid')]:
            self.assertEqual(self.launch(*argv).returncode, 2)
        for variable in ('TAILCOMPILE_V7_GENERALIZATION_JOBS', 'TAILCOMPILE_V7_GENERALIZATION_FIT_TIMEOUT'):
            for value in ('', '0', '-1', '1.5', 'many'):
                self.env[variable] = value
                self.assertEqual(self.launch('all', '--background').returncode, 2, (variable, value))
            del self.env[variable]
        self.env['TAILCOMPILE_3D_TRAIN_DEVICE'] = 'cuda:0extra'
        self.assertEqual(self.launch('prepare').returncode, 2)
        self.assert_no_writes()

    def test_cuda_parallel_jobs_rejected_before_runtime_writes(self):
        self.env['TAILCOMPILE_3D_TRAIN_DEVICE'] = 'cuda'
        self.env['TAILCOMPILE_V7_GENERALIZATION_JOBS'] = '2'
        result = self.launch('all', '--background')
        self.assertEqual(result.returncode, 2)
        self.assertIn('CUDA uses one fit process', result.stderr)
        self.assert_no_writes()

    def test_known_repository_alias_is_accepted(self):
        alias = Path(self.tmp.name)/'project-public'
        alias.symlink_to(self.repo, target_is_directory=True)
        self.env['TAILCOMPILE_REPO'] = str(alias)+'/'
        self.env['TAILCOMPILE_V7_GENERALIZATION_SOURCE'] = str(alias/self.source.relative_to(self.repo))
        self.env['TAILCOMPILE_V7_GENERALIZATION_OUT'] = str(alias/self.out.relative_to(self.repo))
        result = self.launch('check')
        self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
        args = dict(zip(self.records()[0]['args'][::2], self.records()[0]['args'][1::2]))
        self.assertEqual(args['--source'], str(self.source))
        self.assertEqual(args['--out'], str(self.out))
        self.assertFalse(self.out.exists())

    def test_source_and_destination_must_use_distinct_version_namespaces(self):
        self.env['TAILCOMPILE_V7_GENERALIZATION_OUT'] = str(self.source)
        result = self.launch('all', '--background')
        self.assertEqual(result.returncode, 2)
        self.assertIn('disjoint', result.stderr)
        self.assert_no_writes()
        self.env['TAILCOMPILE_V7_GENERALIZATION_OUT'] = str(self.repo/'outside_013')
        self.assertEqual(self.launch('prepare', '--background').returncode, 2)
        del self.env['TAILCOMPILE_V7_GENERALIZATION_OUT']
        self.env['TAILCOMPILE_V7_GENERALIZATION_SOURCE'] = str(self.module/'runs/development_011_residual')
        self.assertEqual(self.launch('prepare', '--background').returncode, 2)
        self.assert_no_writes()

    def test_source_nested_under_destination_rejected(self):
        self.env['TAILCOMPILE_V7_GENERALIZATION_SOURCE'] = str(self.out/'borrowed')
        result = self.launch('prepare', '--background')
        self.assertEqual(result.returncode, 2)
        self.assertIn('disjoint', result.stderr)
        self.assert_no_writes()

    def test_output_redirect_below_repository_alias_rejected(self):
        alias = Path(self.tmp.name)/'project-public'
        alias.symlink_to(self.repo, target_is_directory=True)
        self.env['TAILCOMPILE_REPO'] = str(alias)
        self.env['TAILCOMPILE_V7_GENERALIZATION_OUT'] = str(alias/self.out.relative_to(self.repo))
        self.out.symlink_to(self.source, target_is_directory=True)
        result = self.launch('all', '--background')
        self.assertEqual(result.returncode, 2)
        self.assertIn('path redirected outside namespace', result.stderr)
        self.assertEqual(self.records(), [])
        self.assertFalse((self.repo/'logs').exists())
        self.assertEqual(self.source_marker.read_text(), '{"original_cache":"immutable"}')

    def test_cache_and_log_redirects_rejected(self):
        outside = Path(self.tmp.name)/'outside'
        outside.mkdir()
        (self.repo/'logs').symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.launch('all', '--background').returncode, 2)
        self.assertEqual(list(outside.iterdir()), [])
        (self.repo/'logs').unlink()
        scratch = self.repo/'vtr_work/tailcompile_v7_3d/generalization'
        scratch.mkdir(parents=True)
        (scratch/'tmp').symlink_to(outside, target_is_directory=True)
        self.assertEqual(self.launch('all', '--background').returncode, 2)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertEqual(self.records(), [])
        self.assertFalse(self.out.exists())

    def test_failure_stops_later_stages(self):
        self.env['FAIL_STAGE'] = 'train'
        result = self.launch('all')
        self.assertEqual(result.returncode, 19, result.stdout+result.stderr)
        self.assertEqual([row['stage'] for row in self.records()], ['check', 'prepare', 'train'])
        self.assertIn('V7_GENERALIZATION_FAILED stage=train exit=19', result.stderr)
        self.assertNotIn('GROUP_COMPLETE', result.stdout)

    def test_background_alias_success_and_failure_are_logged(self):
        alias = Path(self.tmp.name)/'project-public'
        alias.symlink_to(self.repo, target_is_directory=True)
        self.env['TAILCOMPILE_REPO'] = str(alias)
        self.env['TAILCOMPILE_V7_GENERALIZATION_OUT'] = str(alias/self.out.relative_to(self.repo))
        text = self.background_log('prepare', marker='V7_GENERALIZATION_GROUP_COMPLETE')
        self.assertIn('V7_GENERALIZATION_GROUP_COMPLETE group=prepare', text)
        self.env['FAIL_STAGE'] = 'train'
        text = self.background_log('all', marker='V7_GENERALIZATION_FAILED')
        self.assertIn('V7_GENERALIZATION_FAILED stage=train exit=19', text)
        self.assertNotIn('GROUP_COMPLETE', text)
        self.assertEqual([row['stage'] for row in self.records()], ['prepare', 'check', 'prepare', 'train'])

    def test_second_stage_group_is_rejected_by_lock(self):
        self.env['HOLD_STAGE'] = 'prepare'
        leader = subprocess.Popen(['bash', str(self.launcher), 'prepare'], env=self.env,
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.addCleanup(lambda: leader.poll() is None and leader.kill())
        for _ in range(100):
            if self.records():
                break
            time.sleep(.02)
        self.assertEqual([row['stage'] for row in self.records()], ['prepare'])
        follower = self.launch('train')
        self.assertEqual(follower.returncode, 2, follower.stdout+follower.stderr)
        self.assertIn('Another 013 stage group is active', follower.stderr)
        stdout, stderr = leader.communicate(timeout=10)
        self.assertEqual(leader.returncode, 0, stdout+stderr)
        self.assertEqual([row['stage'] for row in self.records()], ['prepare'])


if __name__ == '__main__':
    unittest.main()
