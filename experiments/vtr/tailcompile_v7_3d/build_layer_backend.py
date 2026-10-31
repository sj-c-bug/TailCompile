"""Read-only v7 archive -> isolated three-object 007 VPR replacement.

Supports verified incremental and normal CMake v7 backends. Their source,
objects, headers, libraries and binary remain read-only.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import install_hook
import install_layer_hook as layer
from io_utils import sha, write
from repository_guard import validate_output_roots


def normal_commands(base, manifest):
    build_dir=Path(manifest['build_directory'])
    if sha(build_dir/'CMakeCache.txt') != manifest['cmake_cache_sha256']:
        raise ValueError('normal CMake cache signature mismatch')
    work=build_dir/'vpr'
    flags=work/'CMakeFiles/libvpr.dir/flags.make'
    link=work/'CMakeFiles/vpr.dir/link.txt'
    if flags.is_file() and link.is_file():
        values={}
        for line in flags.read_text().splitlines():
            if ' = ' in line:
                key,value=line.split(' = ',1); values[key]=shlex.split(value)
        cache=(build_dir/'CMakeCache.txt').read_text().splitlines()
        compiler=next((s.split('=',1)[1] for s in cache if s.startswith('CMAKE_CXX_COMPILER:FILEPATH=')),'/usr/bin/c++')
        reference=[compiler,*values['CXX_DEFINES'],*values['CXX_INCLUDES'],*values['CXX_FLAGS'],
                   '-o','unused.o','-c',str(base/layer.SOLVER)]
        return reference,shlex.split(link.read_text()),work,work/'libvpr.a'
    if (build_dir/'build.ninja').is_file():
        commands=[shlex.split(line) for line in subprocess.check_output(
            ['ninja','-C',str(build_dir),'-t','commands','vpr'],text=True).splitlines()]
        reference=next(c for c in commands if '-c' in c and c[c.index('-c')+1].endswith('/analytical_solver.cpp'))
        link=next(c for c in reversed(commands) if '-o' in c and c[c.index('-o')+1].endswith('vpr/vpr'))
        if any(token in link for token in ('&&',';')):
            raise ValueError('Ninja link command wrapper unsupported; retain explicit compile/link commands')
        return reference,link,build_dir,build_dir/'vpr/libvpr.a'
    raise ValueError('normal backend requires retained Make flags/link or readable Ninja commands')


def build(base, root, jobs=2, resume=False):
    base, root = Path(base).resolve(), Path(root).resolve()
    validate_output_roots((root,), dependency_roots=(base,))
    if root.exists() and not resume: raise ValueError('new isolated output root required (or --resume for owned incomplete build)')
    if root.exists() and (not (root/'source_overlay.json').exists() or
        Path(json.loads((root/'source_overlay.json').read_text())['base_root']) != base or
        (root/'backend_manifest.json').exists()):
        raise ValueError('resume requires an incomplete owned overlay of the same base')
    signature = install_hook.verify(base)
    original = json.loads((base/'backend_manifest.json').read_text())
    if original['hook_sha256'] != signature or sha(original['binary']) != original['binary_sha256']:
        raise ValueError('audited base manifest mismatch')
    if original.get('build_method')=='normal_cmake_build':
        reference,link_command,original_build,archive=normal_commands(base,original)
        files=(layer.GP,layer.SOLVER,layer.ROUTE)
    else:
        incremental_archive = base/'build_tailcompile_v7/libvpr.a'
        archive = incremental_archive if incremental_archive.exists() else Path(next(
            p for p in original['original_artifacts_before'] if p.endswith('/build/vpr/libvpr.a')))
        if archive != incremental_archive and sha(archive) != original['original_artifacts_before'][str(archive)]:
            raise ValueError('original archive checksum mismatch')
        reference = next(c for c in original['commands'] if '-c' in c)
        includes = [arg[2:] for arg in reference if arg.startswith('-I')]
        original_build = Path(next(i[:-len('/vpr/src/base')] for i in includes if i.endswith('/vpr/src/base')))/'build/vpr'
        link_command=list(original['commands'][-1])
        files=(layer.GP,layer.SOLVER,layer.ROUTE) if archive==incremental_archive else layer.FILES
    watched = [archive, Path(original['binary']), *[base/p for p in install_hook.FILES]]
    before = {str(p): sha(p) for p in watched}
    started = time.monotonic()
    root.mkdir(parents=True,exist_ok=True)
    write(root/'source_overlay.json', dict(schema='tailcompile-layer-source-overlay-1',
        base_root=str(base), vtr_commit=layer.COMMIT, base_hook_sha256=signature,
        semantics='read-only base headers/libraries; exact patched source overlay'))
    for rel, text in layer.expected(root).items():
        (root/rel).parent.mkdir(parents=True, exist_ok=True)
        (root/rel).write_text(text)
    build_dir = root/'build_tailcompile_layer'
    build_dir.mkdir(exist_ok=True)
    def compile_one(rel):
        command = list(reference)
        command[command.index('-o')+1] = str(build_dir/(Path(rel).name+'.o'))
        command[command.index('-c')+1] = str(root/rel)
        # Ninja compile commands can retain dependency-output flags. Redirect
        # them into this owned build rather than writing back into the base.
        for option in ('-MF','-MT','-MQ'):
            if option in command:
                command[command.index(option)+1]=str(build_dir/(Path(rel).name+'.d' if option=='-MF' else Path(rel).name+'.o'))
        for i,arg in enumerate(command):
            if arg.startswith('-I') and len(arg)>2 and not Path(arg[2:]).is_absolute():
                command[i]='-I'+str((original_build/arg[2:]).resolve())
            elif i and command[i-1]=='-isystem' and not Path(arg).is_absolute():
                command[i]=str((original_build/arg).resolve())
        with (build_dir/(Path(rel).name+'.log')).open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        return command
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        commands = list(executor.map(compile_one, files))
    new_archive = build_dir/'libvpr.a'
    shutil.copy2(archive, new_archive)
    objects = [build_dir/(Path(rel).name+'.o') for rel in files]
    members = subprocess.check_output(['ar', 't', str(new_archive)], text=True).splitlines()
    if any(members.count(p.name) != 1 for p in objects): raise ValueError('ambiguous archive member')
    subprocess.run(['ar', 'r', str(new_archive), *map(str, objects)], check=True)
    command = list(link_command)
    command[command.index('-o')+1] = str(build_dir/'vpr')
    archive_tokens=[i for i,p in enumerate(command) if Path(p).name=='libvpr.a']
    if len(archive_tokens)!=1: raise ValueError('one libvpr.a link input required')
    command[archive_tokens[0]] = str(new_archive)
    with (build_dir/'link.log').open('w') as log:
        subprocess.run(command, cwd=original_build, stdout=log, stderr=subprocess.STDOUT, check=True)
    commands.append(command)
    binary = build_dir/'vpr'
    (root/'vpr/vpr').symlink_to(binary)
    after = {str(p): sha(p) for p in watched}
    if before != after: raise AssertionError('read-only original changed')
    manifest = dict(schema='tailcompile-layer-backend-007', vtr_commit=layer.COMMIT,
        hook_sha256=layer.verify(root), binary=str(binary), binary_sha256=sha(binary),
        build_method=f'exact_{len(files)}_object_replacement_not_clean_build',
        base_manifest_sha256=sha(base/'backend_manifest.json'), base_archive_sha256=before[str(archive)],
        original_unchanged=before==after, original_before=before, original_after=after,
        commands=commands, elapsed_seconds=time.monotonic()-started,
        env={'TC_GP_LAYER_PRIOR':'0 or 1; 1 requires TC_GP_INIT', 'TC_GP_Z_USAGE':'optional output prefix'})
    write(root/'backend_manifest.json', manifest)
    return manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=2)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    m=build(a.base,a.root,a.jobs,a.resume)
    print(json.dumps({k:m[k] for k in ('binary','binary_sha256','original_unchanged','elapsed_seconds')}))


if __name__=='__main__': main()
