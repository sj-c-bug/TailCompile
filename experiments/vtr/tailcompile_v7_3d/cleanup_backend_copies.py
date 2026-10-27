"""Two-phase reclamation of this task's exact failed backend copies only."""
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import time

from io_utils import read, sha, write

TOOLS=Path('/home/phoenix/tools')
KEPT=TOOLS/'vtr-40f8d197-tailcompile-v7-3d-r2'
TARGETS=(TOOLS/'vtr-40f8d197-tailcompile-v7-3d',
         TOOLS/'vtr-40f8d197-tailcompile-v7-3d-r1',
         KEPT/'build_tailcompile_v7/libvpr.a')


def summarize(path):
    if path.is_symlink() or path.resolve()!=path:
        raise ValueError(f'cleanup target must be exact resolved nonsymlink: {path}')
    entries=[path]
    if path.is_dir():
        for directory,dirs,files in os.walk(path,followlinks=False):
            entries.extend(Path(directory)/name for name in dirs+files)
    digest=hashlib.sha256()
    logical=allocated=files=shared=0
    inodes={}
    for item in sorted(entries):
        value=item.lstat()
        rel=str(item.relative_to(path)) if item!=path else '.'
        digest.update(f'{rel}\0{value.st_mode}\0{value.st_ino}\0{value.st_size}\0{value.st_mtime_ns}\0{value.st_nlink}\n'.encode())
        logical+=value.st_size
        key=(value.st_dev,value.st_ino)
        if key not in inodes:
            allocated+=value.st_blocks*512
            inodes[key]=value.st_nlink
        if stat.S_ISREG(value.st_mode):
            files+=1
            shared+=value.st_nlink>1
    root=path.lstat()
    return dict(path=str(path),kind='directory' if path.is_dir() else 'file',
                logical_bytes=logical,allocated_bytes=allocated,file_count=files,
                regular_entries_with_multiple_links=shared,inode=root.st_ino,
                device=root.st_dev,metadata_sha256=digest.hexdigest(),
                file_sha256=sha(path) if path.is_file() else None)


def references():
    active=[]
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():
            continue
        try:
            args=(process/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
            for target in TARGETS:
                exact=str(target)
                # The kept -r2 directory shares a prefix with the failed name;
                # only a path boundary can identify a real failed-copy reference.
                import re
                if re.search(re.escape(exact)+r'(?=/|\s|$)',args):
                    active.append(dict(pid=int(process.name),kind='command',target=exact))
            for fd in (process/'fd').iterdir():
                try:
                    opened=Path(os.readlink(fd))
                except (FileNotFoundError,PermissionError,OSError):
                    continue
                for target in TARGETS:
                    if opened==target or (target.is_dir() and target in opened.parents):
                        active.append(dict(pid=int(process.name),kind='open_fd',target=str(target)))
        except (FileNotFoundError,PermissionError,ProcessLookupError):
            continue
    return active


def anchor_hashes():
    paths=[TOOLS/'vtr-40f8d197-tailcompile2d/vpr/vpr',
           TOOLS/'vtr-40f8d197-tailcompile2d/build/vpr/libvpr.a',
           TOOLS/'vtr-40f8d197-tailcompile-v68/vpr/vpr',
           KEPT/'vpr/vpr',KEPT/'backend_manifest.json',KEPT/'tailcompile_heterogeneity_v2.patch']
    import install_hook
    paths.extend(KEPT/rel for rel in (*install_hook.FILES,*install_hook.HETERO_FILES,install_hook.HOOK))
    return {str(path):sha(path) for path in paths}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    a=p.parse_args()
    plan_path=a.plan.resolve()
    if not a.apply:
        if plan_path.exists():
            p.error('new plan path required; never overwrite prior cleanup evidence')
        for target in TARGETS:
            if not target.exists():
                p.error(f'expected owned failed artifact is absent: {target}')
        patch=Path(__file__).parents[3]/'experiments/vtr3d/patches/vtr40f8_layer_heterogeneity_v2.patch'
        kept_patch=KEPT/'tailcompile_heterogeneity_v2.patch'
        if kept_patch.exists() and sha(kept_patch)!=sha(patch):
            raise ValueError('kept patch differs; cleanup stopped')
        if not kept_patch.exists():
            shutil.copy2(patch,kept_patch)
        active=references()
        if active:
            raise ValueError(f'active failed-copy/build dependency: {active}')
        targets=[summarize(path) for path in TARGETS]
        plan=dict(schema='tailcompile-v7-owned-backend-copy-cleanup-1',created_unix=time.time(),
            targets=targets,active_references=active,preserved_sha256=anchor_hashes(),
            total_logical_bytes=sum(item['logical_bytes'] for item in targets),
            total_allocated_bytes=sum(item['allocated_bytes'] for item in targets),
            kept_small_objects=[str(path) for path in sorted((KEPT/'build_tailcompile_v7').glob('*.o'))],
            rebuild_note='Deleted duplicate static archive can be reconstructed by copying the unchanged base archive and replacing the five kept object members using backend_manifest commands. The linked VPR executable does not load a static archive at runtime.',
            deletion_scope='two exact failed independent checkout copies and one duplicate linked static archive; never old v68/2d or the kept r2 source/binary')
        write(plan_path,plan)
        print(json.dumps(dict(plan=str(plan_path),targets=targets,
            total_allocated_bytes=plan['total_allocated_bytes'],active_references=active)))
        return
    plan=read(plan_path)
    if [item['path'] for item in plan['targets']]!=list(map(str,TARGETS)):
        raise ValueError('plan does not name the exact three authorized task artifacts')
    if anchor_hashes()!=plan['preserved_sha256'] or references():
        raise ValueError('preserved artifact or process dependency changed; cleanup stopped')
    if [summarize(path) for path in TARGETS]!=plan['targets']:
        raise ValueError('cleanup target changed after review; cleanup stopped')
    journal=plan_path.with_name(plan_path.stem+'_applied.json')
    if journal.exists():
        raise ValueError('cleanup application already has a ledger')
    free_before=shutil.disk_usage(TOOLS).free
    done=[]
    for target,item in zip(TARGETS,plan['targets']):
        write(journal,dict(status='applying',plan_sha256=sha(plan_path),completed=done,
                           current_target=str(target)))
        if item['kind']=='directory':
            shutil.rmtree(target)
        else:
            target.unlink()
        done.append(item)
    after=anchor_hashes()
    if after!=plan['preserved_sha256']:
        raise AssertionError('preserved artifacts changed during owned cleanup')
    ledger=dict(status='complete',plan_sha256=sha(plan_path),removed=done,
        logical_bytes_removed=sum(item['logical_bytes'] for item in done),
        allocated_bytes_in_removed_entries=sum(item['allocated_bytes'] for item in done),
        filesystem_free_bytes_change_observed=shutil.disk_usage(TOOLS).free-free_before,
        observed_free_change_may_include_concurrent_work=True,preserved_sha256_after=after,
        preserved_unchanged=True,kept_small_objects=plan['kept_small_objects'],
        recovery='Failed copies can be recreated; duplicate static archive can be reconstructed from unchanged base archive plus kept five objects. No user source or accepted result removed.')
    write(journal,ledger)
    print(json.dumps({key:value for key,value in ledger.items() if key not in ('preserved_sha256_after','removed')}))


if __name__=='__main__':
    main()
