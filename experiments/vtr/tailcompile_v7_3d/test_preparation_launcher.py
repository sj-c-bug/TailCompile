"""The large-device recovery keeps the existing cohort and serializes XML work."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux preparation launcher')
class PreparationLauncherTests(unittest.TestCase):
    def test_existing_cohort_and_old_results_are_preserved_with_serial_preparation(self):
        with tempfile.TemporaryDirectory(dir=HERE/'runs', prefix='prep-launch-') as temporary:
            repo = Path(temporary).resolve()/'TailCompile'
            module = repo/'experiments/vtr/tailcompile_v7_3d'
            module.mkdir(parents=True)
            shutil.copyfile(HERE/'resume_device_preparation.sh', module/'resume_device_preparation.sh')
            cohort = module/'runs/development_004_capacity'
            for name in ('protocol.json', 'campaign.json', 'prepared.json', 'qualification/qualification_report.json'):
                path = cohort/name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"preserved":true}')
            calls = repo/'calls.txt'
            (module/'prepare_server.sh').write_text('#!/bin/bash\nprintf "preflight:%s\\n" "$*" >> "$FAKE_CALLS"\n')
            python = repo/'python'
            python.write_text('#!/bin/bash\nprintf "%s|%s|%s|%s|%s|%s\\n" "$*" '
                '"$TAILCOMPILE_V7_OUT" "$TAILCOMPILE_V7_CONFIG" "$TAILCOMPILE_3D_QUALIFICATION" '
                '"$TAILCOMPILE_V7_RRG_JOBS" "$TAILCOMPILE_V7_JOBS" >> "$FAKE_CALLS"\n')
            python.chmod(0o755)
            before = {str(p):p.read_bytes() for p in cohort.rglob('*') if p.is_file()}
            env = {k:v for k,v in os.environ.items() if not k.startswith('TAILCOMPILE_')}
            env.update(FAKE_CALLS=str(calls), TAILCOMPILE_3D_PYTHON=str(python),
                       TAILCOMPILE_V7_OUT=str(repo/'old003'), TAILCOMPILE_V7_CONFIG=str(repo/'old003/config'),
                       TAILCOMPILE_3D_QUALIFICATION=str(repo/'old003/qualification'),
                       TAILCOMPILE_V7_JOBS='16', TAILCOMPILE_V7_RRG_JOBS='4')
            result = subprocess.run(['bash', str(module/'resume_device_preparation.sh'), '--foreground'],
                env=env, capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout+result.stderr)
            line = calls.read_text().splitlines()[-1].split('|')
            self.assertIn('resume_device_preparation.py', line[0])
            self.assertIn('--max-seconds 3600', line[0])
            self.assertEqual(line[1:], [str(cohort), str(cohort/'campaign.json'),
                                      str(cohort/'qualification'), '1', '16'])
            self.assertEqual(before, {p:Path(p).read_bytes() for p in before})
            self.assertFalse((repo/'old003').exists())


if __name__ == '__main__':
    unittest.main()
