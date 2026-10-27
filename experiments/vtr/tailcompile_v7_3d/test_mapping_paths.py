"""Exercise frontend cwd changes, isolated retries and server output boundaries."""
from pathlib import Path
import os
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import campaign
from io_utils import read, run_trial


HERE = Path(__file__).resolve().parent


class MappingPathTests(unittest.TestCase):
    def temporary(self):
        # Keep this contract's generated payload in the source repository.
        runs = HERE/'runs'
        runs.mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='mapping-contract-', dir=runs)

    def command(self, root, retry=False):
        script = root/'frontend.py'
        script.write_text(
            'import os,sys\nfrom pathlib import Path\n'
            'flow=Path(sys.argv[sys.argv.index("-temp_dir")+1])\n'
            'assert flow.is_absolute(), "VTR frontend needs an absolute path"\n'
            'flow.mkdir()\n(flow/"odin_config.xml").write_text("attempt-owned")\n'
            'attempt=Path.cwd().name\nos.chdir(flow)\n'
            'assert (flow/"odin_config.xml").read_text()=="attempt-owned"\n'
            'print("FRONTEND_PATH_OK",flow,flush=True)\n'
            f'sys.exit(127 if {retry!r} and attempt=="attempt_0001" else 0)\n',
            encoding='utf-8')
        return [sys.executable, str(script), '-temp_dir', 'flow']

    def test_frontend_receives_absolute_attempt_path_and_exact_resume(self):
        with self.temporary() as tmp:
            root = Path(tmp).resolve()
            command = self.command(root)
            folder = root/'trial'
            result = run_trial(folder, command, timeout=10, prune=False,
                               attempt_path_options=('-temp_dir',))
            self.assertEqual(result['status'], 'complete')
            self.assertEqual(result['signature']['command'][-1], 'flow')
            self.assertEqual(result['signature']['attempt_path_options'], ['-temp_dir'])
            started = read(Path(result['workdir'])/'started.json')
            actual = started['executed_command']
            self.assertEqual(actual[actual.index('-temp_dir')+1],
                             str(Path(result['workdir'])/'flow'))
            cached = run_trial(folder, command, timeout=10, prune=False,
                               attempt_path_options=('-temp_dir',))
            self.assertEqual(cached['workdir'], result['workdir'])
            self.assertEqual(len(list((folder/'attempts').iterdir())), 1)

    def test_retry_uses_fresh_absolute_flow_and_preserves_failed_attempt(self):
        with self.temporary() as tmp:
            root = Path(tmp).resolve()
            command = self.command(root, retry=True)
            folder = root/'trial'
            first = run_trial(folder, command, timeout=10, prune=False,
                              attempt_path_options=('-temp_dir',))
            self.assertEqual(first['status'], 'infrastructure_failure')
            old_log = Path(first['log']).read_bytes()
            second = run_trial(folder, command, timeout=10, prune=False,
                               attempt_path_options=('-temp_dir',), retry_infrastructure=True)
            self.assertEqual(second['status'], 'complete')
            self.assertNotEqual(first['workdir'], second['workdir'])
            self.assertEqual(Path(first['log']).read_bytes(), old_log)
            for result in (first, second):
                self.assertEqual((Path(result['workdir'])/'flow/odin_config.xml').read_text(),
                                 'attempt-owned')
            self.assertEqual(read(Path(first['workdir'])/'metrics.json')['status'],
                             'infrastructure_failure')

    def test_invalid_attempt_path_is_rejected_before_trial_directory_creation(self):
        with self.temporary() as tmp:
            root = Path(tmp)
            for value in ('../escape', str(root.resolve()/'absolute')):
                with self.subTest(value=value):
                    folder = root/'must_not_exist'
                    with self.assertRaisesRegex(ValueError, 'inside its attempt'):
                        run_trial(folder, [sys.executable, '-temp_dir', value],
                                  attempt_path_options=('-temp_dir',))
                    self.assertFalse(folder.exists())

    def test_mapping_failure_surfaces_nested_parmys_error(self):
        with self.temporary() as tmp:
            work = Path(tmp)
            (work/'flow').mkdir()
            (work/'flow/parmys.out').write_text('ERROR_PARSE_ARCH: Failed to open file\n')
            (work/'vpr.out').write_text('Executable yosys failed\n')
            message = campaign.mapping_failure_message(dict(workdir=str(work),
                log=str(work/'vpr.out'), status='packing_or_placement_failure', returncode=1))
            self.assertIn('ERROR_PARSE_ARCH', message)
            self.assertIn('Executable yosys failed', message)
            self.assertIn('returncode=1', message)

    def test_server_output_guard_refuses_sibling_and_resolved_symlink(self):
        with self.temporary() as tmp:
            root = Path(tmp).resolve()
            repo = root/'repo'
            repo.mkdir()
            fake_here = repo/'experiments/vtr/tailcompile_v7_3d'
            args = SimpleNamespace(allow_local=False, out=repo/'runs/a', scratch=repo/'scratch')
            with patch.object(campaign, 'HERE', fake_here):
                campaign.validate_server_output_roots(args)
                args.out = root/'outside'
                with self.assertRaisesRegex(ValueError, 'outside'):
                    campaign.validate_server_output_roots(args)
                self.assertFalse(args.out.exists())
                args.allow_local = True
                campaign.validate_server_output_roots(args)
                if os.name == 'posix':
                    args.allow_local = False
                    (repo/'alias').symlink_to(root/'outside', target_is_directory=True)
                    args.out = repo/'alias/results'
                    with self.assertRaisesRegex(ValueError, 'outside'):
                        campaign.validate_server_output_roots(args)


if __name__ == '__main__':
    unittest.main()
