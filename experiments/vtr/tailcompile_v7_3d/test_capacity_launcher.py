"""The recovery launcher must replace stale cohort settings and run CPU stages in order."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux recovery launcher')
class CapacityLauncherTests(unittest.TestCase):
    def test_foreground_recovery_replaces_old_cohort_and_preserves_backend(self):
        with tempfile.TemporaryDirectory(dir=HERE/'runs', prefix='capacity-launch-') as temporary:
            repo = Path(temporary).resolve()/'TailCompile'
            module = repo/'experiments/vtr/tailcompile_v7_3d'
            module.mkdir(parents=True)
            calls = repo/'calls.txt'
            shutil.copyfile(HERE/'recover_capacity.sh', module/'recover_capacity.sh')
            (module/'prepare_server.sh').write_text(
                '#!/bin/bash\nset -eu\n'
                'printf "%s|%s|%s|%s|%s\\n" "$*" "$TAILCOMPILE_V7_OUT" '
                '"$TAILCOMPILE_V7_CONFIG" "$TAILCOMPILE_3D_QUALIFICATION" '
                '"$TAILCOMPILE_3D_BACKEND_READ_ONLY" >> "$FAKE_CALLS"\n')
            python = repo/'python'
            python.write_text('#!/bin/bash\nprintf "recover:%s\\n" "$*" >> "$FAKE_CALLS"\n')
            python.chmod(0o755)
            backend = Path(temporary)/'read_only_backend'
            backend.mkdir()
            preserved = backend/'preserve'
            preserved.write_text('unchanged')
            env = {k:v for k,v in os.environ.items() if not k.startswith('TAILCOMPILE_')}
            env.update(FAKE_CALLS=str(calls), TAILCOMPILE_3D_PYTHON=str(python),
                       TAILCOMPILE_3D_VTR_ROOT=str(backend), TAILCOMPILE_V7_OUT=str(repo/'old'),
                       TAILCOMPILE_V7_CONFIG=str(repo/'old/campaign.json'),
                       TAILCOMPILE_3D_QUALIFICATION=str(repo/'old/qualification'))
            result = subprocess.run(['bash', str(module/'recover_capacity.sh'), '--foreground'],
                env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            lines = calls.read_text().splitlines()
            self.assertEqual([line.split('|')[0] for line in lines if not line.startswith('recover:')],
                             ['--check', '--stage plan', '--stage prepare', '--stage calibrate'])
            out = module/'runs/development_004_capacity'
            for line in lines:
                if not line.startswith('recover:'):
                    self.assertEqual(line.split('|')[1:],
                        [str(out), str(out/'campaign.json'), str(out/'qualification'), '1'])
            self.assertEqual(preserved.read_text(), 'unchanged')
            self.assertFalse((repo/'old').exists())
            self.assertIn('V7_CAPACITY_BASELINE_FINISHED_REVIEW_BOUNDARIES', result.stdout)


if __name__ == '__main__':
    unittest.main()
