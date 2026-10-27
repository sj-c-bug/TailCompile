from pathlib import Path
import os
import tempfile
import unittest
from unittest.mock import patch

import install_hook
from io_utils import COMMIT, read, sha, write
from recover_prepare import recover

HERE = Path(__file__).resolve().parent


class PrepareRecoveryTests(unittest.TestCase):
    def fixture(self, root):
        repo = root/'repo'
        runs = repo/'experiments/vtr/tailcompile_v7_3d/runs'
        old, new, backend = runs/'old', runs/'new', root/'external_read_only_backend'
        (backend/'vpr').mkdir(parents=True)
        (backend/'vpr/vpr').write_bytes(b'qualified executable')
        sources = {}
        for name in (*install_hook.FILES, *install_hook.HETERO_FILES):
            target = backend/name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(name)
            sources[name] = sha(target)
        digest = sha(backend/'vpr/vpr')
        write(backend/'backend_manifest.json', dict(vtr_commit=COMMIT, binary_sha256=digest,
              hook_sha256='verified-hook', source_sha256=sources))
        cases, seeds = ['aligned', 'shifted', 'fixed_mixed', 'sha'], [701, 702]
        write(old/'qualification/protocol.json', dict(commit=COMMIT, binary_sha256=digest, cases=cases, seeds=seeds))
        write(old/'qualification/qualification_report.json', dict(complete=True, passed=True, harness_version=2,
              conditions=[dict(case=c, seed=s, passed=True) for c in cases for s in seeds]))
        write(old/'campaign.json', dict(designs=['unchanged-inputs']))
        write(old/'protocol.json', dict(config=read(old/'campaign.json')))
        write(old/'inputs/sha/mapping/manifest.json', dict(status='packing_or_placement_failure'))
        return repo, old, new, backend

    def test_reuse_keeps_old_failure_and_external_backend_unchanged_and_is_idempotent(self):
        with tempfile.TemporaryDirectory(dir=HERE/'runs') as temporary:
            repo, old, new, backend = self.fixture(Path(temporary))
            before = {str(p): sha(p) for base in (old, backend) for p in base.rglob('*') if p.is_file()}
            with patch('recover_prepare.install_hook.verify', return_value='verified-hook'):
                first = recover(old, new, backend, repo=repo)
                self.assertEqual(first, recover(old, new, backend, repo=repo))
            self.assertEqual(before, {p: sha(p) for p in before})
            self.assertTrue(read(new/'qualification/qualification_report.json')['passed'])
            self.assertFalse((new/'protocol.json').exists())
            self.assertFalse((new/'inputs/sha/mapping/manifest.json').exists())

    def test_bad_binary_partial_qualification_and_external_output_rejected_before_writes(self):
        for failure in ('binary', 'qualification', 'output'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory(dir=HERE/'runs') as temporary:
                repo, old, new, backend = self.fixture(Path(temporary))
                if failure == 'binary':
                    (backend/'vpr/vpr').write_bytes(b'changed executable')
                elif failure == 'qualification':
                    report = read(old/'qualification/qualification_report.json')
                    report['conditions'].pop()
                    write(old/'qualification/qualification_report.json', report)
                else:
                    new = Path(temporary)/'outside'
                with patch('recover_prepare.install_hook.verify', return_value='verified-hook'), self.assertRaises(ValueError):
                    recover(old, new, backend, repo=repo)
                self.assertFalse(new.exists())

    @unittest.skipUnless(os.name == 'posix', 'directory symlink contract')
    def test_existing_qualification_symlink_cannot_redirect_recovery_writes(self):
        with tempfile.TemporaryDirectory(dir=HERE/'runs') as temporary:
            repo, old, new, backend = self.fixture(Path(temporary))
            external = Path(temporary)/'outside'
            external.mkdir()
            new.mkdir()
            (new/'qualification').symlink_to(external, target_is_directory=True)
            with patch('recover_prepare.install_hook.verify', return_value='verified-hook'), self.assertRaisesRegex(ValueError, 'symlink'):
                recover(old, new, backend, repo=repo)
            self.assertEqual(list(external.iterdir()), [])
            self.assertFalse((new/'recovery.json').exists())


if __name__ == '__main__':
    unittest.main()
