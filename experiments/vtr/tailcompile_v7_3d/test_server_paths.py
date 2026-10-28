"""Server bootstrap path contracts; fake tools never build or run VPR."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'


@unittest.skipUnless(os.name == 'posix' and shutil.which('bash'), 'Linux shell path contract')
class ServerPathTests(unittest.TestCase):
    def setUp(self):
        parent = HERE/'runs'
        parent.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix='server_paths_', dir=parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.repo = self.root/'TailCompile'
        self.module = self.repo/'experiments/vtr/tailcompile_v7_3d'
        self.module.mkdir(parents=True)
        self.entry = self.module/'prepare_server.sh'
        shutil.copyfile(HERE/'prepare_server.sh', self.entry)
        (self.repo/'REPOSITORY_WORKFLOW.md').write_text('test fixture\n')
        (self.module/'campaign.example.json').write_text('{}\n')
        self.fake_log = self.repo/'fake_calls.txt'
        self.tools = self.repo/'fake_tools'
        self.tools.mkdir()
        self.script(self.tools/'git',
                    '#!/bin/sh\ncase "$*" in\n*--show-toplevel*) printf "%s\\n" "$FAKE_REPO";;\n'
                    f'*) printf "%s\\n" "{COMMIT}";;\nesac\n')
        self.script(self.tools/'python', '#!/bin/sh\nprintf "%s\\n" "$*" >> "$FAKE_LOG"\n')
        self.script(self.module/'run_pipeline.sh', '#!/bin/sh\nprintf "stage:%s\\n" "$*" >> "$FAKE_LOG"\n')
        self.mapping = self.root/'vtr_tailcompile/vtr-verilog-to-routing'
        for relative in ('abc/abc', 'build/bin/yosys', 'vpr/vpr'):
            self.script(self.mapping/relative, '#!/bin/sh\nexit 0\n')
        (self.mapping/'vtr_flow/scripts').mkdir(parents=True)
        (self.mapping/'vtr_flow/scripts/run_vtr_flow.py').write_text('# fixture\n')
        self.env = {key:value for key,value in os.environ.items()
                    if not key.startswith(('TAILCOMPILE_', 'TC_GP_'))}
        self.env.update(PATH=str(self.tools)+os.pathsep+os.environ['PATH'],
                        FAKE_REPO=str(self.repo), FAKE_LOG=str(self.fake_log),
                        TAILCOMPILE_3D_PYTHON=str(self.tools/'python'))

    @staticmethod
    def script(path, content):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)

    def run_entry(self, *arguments, **overrides):
        return subprocess.run(['bash', str(self.entry), *arguments], cwd=self.repo,
                              env=dict(self.env, **overrides), capture_output=True,
                              text=True, timeout=15)

    def assert_no_runtime_directories(self):
        for name in ('.deps', 'logs', 'vtr_work', 'env_3d_v7.sh'):
            self.assertFalse((self.repo/name).exists(), name)
        self.assertFalse((self.module/'runs').exists())

    def test_default_backend_is_owned_repo_dependency(self):
        row = self.run_entry('--check')
        self.assertEqual(row.returncode, 0, row.stdout+row.stderr)
        self.assertIn('3D_ROOT='+str(self.repo/'.deps/tailcompile_v7_3d/vtr-verilog-to-routing'), row.stdout)
        self.assert_no_runtime_directories()

    def test_external_writable_backend_and_escaping_deps_symlink_fail_before_writes(self):
        outside = self.root/'old_sibling_backend'
        row = self.run_entry('--check', TAILCOMPILE_3D_VTR_ROOT=str(outside))
        self.assertNotEqual(row.returncode, 0)
        self.assertIn('Writable 3D backend must stay under repo/.deps/', row.stderr)
        self.assertFalse(outside.exists())
        self.assert_no_runtime_directories()
        outside.mkdir()
        (self.repo/'.deps').symlink_to(outside, target_is_directory=True)
        row = self.run_entry('--check')
        self.assertNotEqual(row.returncode, 0)
        self.assertIn('namespace resolves outside', row.stderr)
        self.assertEqual(list(outside.iterdir()), [])
        self.assertFalse((self.repo/'vtr_work').exists())

    def test_readonly_mode_requires_existing_backend_before_runtime_writes(self):
        row = self.run_entry('--check', TAILCOMPILE_3D_VTR_ROOT=str(self.root/'missing'),
                             TAILCOMPILE_3D_BACKEND_READ_ONLY='1')
        self.assertNotEqual(row.returncode, 0)
        self.assertIn('Read-only 3D backend needs', row.stderr)
        self.assert_no_runtime_directories()

    def test_readonly_stage_verifies_existing_backend_without_install_or_mutation(self):
        backend = self.root/'old_sibling_backend'
        self.script(backend/'vpr/vpr', '#!/bin/sh\nexit 0\n')
        (backend/'backend_manifest.json').write_text('{}\n')
        before = {str(path.relative_to(backend)):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in backend.rglob('*') if path.is_file()}
        row = self.run_entry('--stage', 'prepare', TAILCOMPILE_3D_VTR_ROOT=str(backend),
                             TAILCOMPILE_3D_BACKEND_READ_ONLY='1')
        self.assertEqual(row.returncode, 0, row.stdout+row.stderr)
        calls = self.fake_log.read_text()
        self.assertIn('install_hook.py '+str(backend)+' --verify', calls)
        self.assertIn('stage:prepare', calls)
        self.assertNotIn('install_vtr.sh', calls)
        after = {str(path.relative_to(backend)):hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in backend.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        self.assertFalse((self.repo/'.deps').exists())

    def test_readonly_bootstrap_preserves_external_backend_and_pins_runtime_caches(self):
        backend = self.root/'old_sibling_backend'
        self.script(backend/'vpr/vpr', '#!/bin/sh\nexit 0\n')
        (backend/'backend_manifest.json').write_text('{}\n')
        self.script(self.tools/'python',
                    '#!/bin/sh\nprintf "%s\\n" "$*" >> "$FAKE_LOG"\n'
                    'for owned_path in "$TMPDIR" "$XDG_CACHE_HOME" "$CCACHE_DIR" "$TORCH_EXTENSIONS_DIR" "$CUDA_CACHE_PATH" "$TRITON_CACHE_DIR"; do\n'
                    ' case "$owned_path" in "$FAKE_REPO"/vtr_work/tailcompile_v7_3d/*) ;; *) exit 9;; esac\ndone\n')
        before = {str(path.relative_to(backend)):hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in backend.rglob('*') if path.is_file()}
        row = self.run_entry(TAILCOMPILE_3D_VTR_ROOT=str(backend),
                             TAILCOMPILE_3D_BACKEND_READ_ONLY='1')
        self.assertEqual(row.returncode, 0, row.stdout+row.stderr)
        self.assertIn('CPU_PREPARE_OK', row.stdout)
        calls = self.fake_log.read_text()
        self.assertIn('stage:prepare', calls)
        self.assertNotIn('install_vtr.sh', calls)
        after = {str(path.relative_to(backend)):hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in backend.rglob('*') if path.is_file()}
        self.assertEqual(before, after)
        env = (self.repo/'env_3d_v7.sh').read_text()
        self.assertIn('export TAILCOMPILE_3D_BACKEND_READ_ONLY=1', env)

    def test_escaping_scratch_or_envfile_symlink_rejected_before_calls(self):
        for name, directory in (('vtr_work', True), ('env_3d_v7.sh', False)):
            with self.subTest(name=name):
                outside = self.root/('outside_'+name)
                if directory:
                    outside.mkdir()
                else:
                    outside.write_text('preserve\n')
                alias = self.repo/name
                alias.symlink_to(outside, target_is_directory=directory)
                row = self.run_entry('--check')
                self.assertNotEqual(row.returncode, 0)
                self.assertIn('resolves outside', row.stderr)
                self.assertFalse(self.fake_log.exists())
                alias.unlink()

    def test_child_temporary_and_cache_symlink_escapes_fail_before_imports(self):
        scratch = self.repo/'vtr_work/tailcompile_v7_3d'
        for relative in ('tmp', 'cache', 'cache/torch_extensions'):
            with self.subTest(relative=relative):
                alias = scratch/relative
                alias.parent.mkdir(parents=True, exist_ok=True)
                outside = self.root/('outside_'+relative.replace('/', '_'))
                outside.mkdir()
                alias.symlink_to(outside, target_is_directory=True)
                row = self.run_entry('--check')
                self.assertNotEqual(row.returncode, 0)
                self.assertIn('temporary/cache path resolves outside', row.stderr)
                self.assertFalse(self.fake_log.exists())
                self.assertEqual(list(outside.iterdir()), [])
                alias.unlink()


if __name__ == '__main__':
    unittest.main()
