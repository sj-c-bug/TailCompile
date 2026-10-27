"""Build an isolated exact-object replacement VPR from an audited local archive.

This is an incremental build, explicitly not a clean rebuild. Five objects are
compiled from pinned, reviewed source; base headers/libraries/flags are frozen
in a provenance manifest. The original source, archive and executable are read-only.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time

import install_hook


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda: f.read(1024*1024), b''):
            value.update(data)
    return value.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--patch', type=Path, required=True)
    p.add_argument('--jobs', type=int, default=2)
    a = p.parse_args()
    base, root = a.base.resolve(), a.root.resolve()
    from repository_guard import validate_output_roots
    validate_output_roots((root,),dependency_roots=(base,))
    if root.exists():
        p.error('isolated root must be new; existing experiment evidence is preserved')
    if install_hook.git(base, 'rev-parse', 'HEAD').strip() != install_hook.COMMIT:
        p.error('base commit mismatch')
    changed = install_hook.git(base, 'diff', '--name-only', '--ignore-submodules', 'HEAD').splitlines()
    if set(changed) - {install_hook.FLOW, 'vpr/vpr'}:
        p.error(f'base contains unreviewed non-flow changes: {changed}')
    patch_text = a.patch.read_text()
    patch_files = [line[6:] for line in patch_text.splitlines() if line.startswith('+++ b/')]
    if set(patch_files) != set(install_hook.HETERO_FILES):
        p.error('requires the complete reviewed heterogeneity v2 patch')
    original_archive = base/'build/vpr/libvpr.a'
    paths = [base/rel for rel in (*install_hook.FILES, *install_hook.HETERO_FILES)]
    paths += [original_archive, base/'vpr/vpr']
    before = {str(path): sha(path) for path in paths}
    started = time.monotonic()
    # The local research snapshot may deliberately omit unrelated Git blobs.
    # Copy its audited working tree/index instead of forcing a full checkout of
    # pruned documentation/benchmark history. Restore every compiled file from
    # the exact pinned blob; do not inherit the legacy flow hook in the copy.
    def omit_root_builds(directory, names):
        if Path(directory).resolve() != base:
            return []
        return [name for name in names if name == 'build' or
                name.startswith('build_tailcompile_') or name == 'tailcompile_frozen_binaries']
    shutil.copytree(base, root, symlinks=True, ignore=omit_root_builds)
    shutil.copy2(a.patch, root/'tailcompile_heterogeneity_v2.patch')
    for rel in (*install_hook.FILES, *install_hook.HETERO_FILES):
        (root/rel).write_text(install_hook.git(base, 'show', f'{install_hook.COMMIT}:{rel}'))
    subprocess.run(['git', '-C', str(root), 'apply', '--check', str(a.patch.resolve())], check=True)
    subprocess.run(['git', '-C', str(root), 'apply', str(a.patch.resolve())], check=True)
    subprocess.run(['python3', str(Path(install_hook.__file__).resolve()), str(root)], check=True)
    build = root/'build_tailcompile_v7'
    build.mkdir()
    values = {}
    flags = base/'build/vpr/CMakeFiles/libvpr.dir/flags.make'
    for line in flags.read_text().splitlines():
        if ' = ' in line:
            key, value = line.split(' = ', 1)
            values[key] = shlex.split(value)
    commands = []
    objects = []
    files = (*install_hook.FILES, *install_hook.HETERO_FILES)
    def compile_one(rel):
        obj = build/(Path(rel).name+'.o')
        command = ['/usr/bin/c++', *values['CXX_DEFINES'], *values['CXX_INCLUDES'],
                   *values['CXX_FLAGS'], '-o', str(obj), '-c', str(root/rel)]
        with (build/(Path(rel).name+'.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return obj, command
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,a.jobs)) as pool:
        for obj, command in pool.map(compile_one, files):
            objects.append(obj)
            commands.append(command)
    archive = build/'libvpr.a'
    shutil.copy2(original_archive, archive)
    members = subprocess.check_output(['ar','t',str(archive)],text=True).splitlines()
    if any(members.count(obj.name) != 1 for obj in objects):
        raise ValueError('each replacement object must occur exactly once in original archive')
    command = ['ar','r',str(archive),*map(str,objects)]
    subprocess.run(command,check=True)
    commands.append(command)
    command = shlex.split((base/'build/vpr/CMakeFiles/vpr.dir/link.txt').read_text())
    binary = build/'vpr'
    command[command.index('-o')+1] = str(binary)
    command[command.index('libvpr.a')] = str(archive)
    with (build/'link.log').open('w') as log:
        subprocess.run(command,cwd=base/'build/vpr',stdout=log,stderr=subprocess.STDOUT,check=True)
    commands.append(command)
    entry = root/'vpr/vpr'
    if entry.exists() or entry.is_symlink():
        if entry.is_dir():
            raise ValueError('unexpected executable directory')
        entry.unlink()
    entry.symlink_to(binary)
    after = {str(path):sha(path) for path in paths}
    if before != after:
        raise AssertionError('original source/build modified during isolated compilation')
    manifest = dict(schema='tailcompile-v7-backend-1',vtr_commit=install_hook.COMMIT,
        build_method='exact_five_object_replacement_not_clean_rebuild',
        source_snapshot='audited_working_tree_copy_with_five_pristine_pinned_blobs',
        hook_sha256=install_hook.verify(root),heterogeneity_patch_sha256=sha(a.patch),
        source_sha256={rel:sha(root/rel) for rel in files},
        original_artifacts_before=before,original_artifacts_after=after,
        original_unchanged=True,compile_flags_sha256=sha(flags),commands=commands,
        binary=str(binary),binary_sha256=sha(binary),elapsed_seconds=time.monotonic()-started)
    (root/'backend_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({key:manifest[key] for key in ('binary','binary_sha256','original_unchanged','elapsed_seconds')}))


if __name__ == '__main__':
    main()
