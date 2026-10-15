import os
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from v5_vpr_retry import run_one
from tailcompile_v4 import load


class RetryTest(unittest.TestCase):
    def test_timeout_archive_resume_and_input_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blif, net = root / 'a.blif', root / 'a.net'
            blif.write_text('blif')
            net.write_text('net')
            backend = SimpleNamespace(binary_hash='binary', arch_hash='arch',
                                      binary='vpr', arch='arch', distro=None,
                                      command=lambda args, cwd: args)
            attempts = []

            def execute(command, **kwargs):
                attempts.append(kwargs['timeout'])
                if len(attempts) == 1:
                    kwargs['stdout'].write('routing in progress')
                    raise subprocess.TimeoutExpired(command, kwargs['timeout'])
                kwargs['stdout'].write('VPR succeeded')
                return SimpleNamespace(returncode=0)

            args = (backend, root / 'run', blif, net, 50, 56, 11)
            with patch.dict(os.environ, TAILCOMPILE_VPR_RETRY_TIMEOUT='7200'), \
                    patch('run_tailcompile_v4.subprocess.run', side_effect=execute):
                result = run_one(*args)
                self.assertEqual(attempts, [1800, 7200])
                self.assertEqual(result['vpr_status'], 'success')
                history = list((root / '_timeout_history').glob('*/manifest.json'))
                self.assertEqual(len(history), 1)
                self.assertEqual(load(history[0])['process_returncode'], 124)
                self.assertEqual(run_one(*args), result)
                self.assertEqual(len(attempts), 2)
                blif.write_text('changed')
                with self.assertRaisesRegex(ValueError, 'inputs changed'):
                    run_one(*args)

    def test_retry_exhaustion_is_not_unroutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blif, net = root / 'a.blif', root / 'a.net'
            blif.write_text('blif')
            net.write_text('net')
            backend = SimpleNamespace(binary_hash='binary', arch_hash='arch',
                                      binary='vpr', arch='arch', distro=None,
                                      command=lambda args, cwd: args)
            args = (backend, root / 'run', blif, net, 50, 56, 11)
            with patch.dict(os.environ, TAILCOMPILE_VPR_RETRY_TIMEOUT='7200'), \
                    patch('run_tailcompile_v4.subprocess.run',
                          side_effect=subprocess.TimeoutExpired('vpr', 1800)) as launch:
                with self.assertRaises(RuntimeError):
                    run_one(*args)
                with self.assertRaisesRegex(RuntimeError, 'budget exhausted'):
                    run_one(*args)
                self.assertEqual(launch.call_count, 2)
                self.assertEqual(load(root / 'run/manifest.json')['process_returncode'], 124)


if __name__ == '__main__':
    unittest.main()
