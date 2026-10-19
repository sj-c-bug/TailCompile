"""Plan/apply deletion of reproducible OLD VPR payloads, preserving inputs/evidence.

No directory deletion. v6.6/v6.7, training bundles, inputs and Git-tracked files
are excluded. apply validates exact paths/stat snapshots and active processes.
Deleted route/RRG/report payloads require rerunning VPR to regenerate.
"""
import argparse
from collections import Counter,defaultdict
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import uuid

OLD={f'tailcompile_{v}' for v in ('v0','v1','v2','v3','v4','v5','v6','v61','v62','v63','v64','v65')}
PROTECTED={'inputs','input','ir','pool','pools','model','models','bundle','dataset','checkpoints','designs','.git'}
DONE={'success','unroutable','timeout','other_failure','interrupted'}


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n',encoding='utf-8');temporary.replace(path)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tracked(repo):
    result=subprocess.run(['git','-C',str(repo),'ls-files','-z'],check=True,capture_output=True)
    return {(repo/os.fsdecode(name)).absolute() for name in result.stdout.split(b'\0') if name}


def raw_kind(name):
    lower=name.lower()
    if re.search(r'(rr[_-]?graph|rrg).*\.(xml|bin)(\.gz)?$',lower):return 'rr_graph'
    if lower.endswith(('.route','.route.gz')):return 'route'
    if lower.endswith(('.rpt','.echo')):return 'detailed_report'
    return None


def finished(folder):
    for name in ('manifest.json','trial.json'):
        path=folder/name
        if path.is_file() and not path.is_symlink():
            try:
                row=load(path)
                if row.get('vpr_status',row.get('status')) in DONE:return path
            except (OSError,ValueError):pass
    return None


def snapshot(path):
    s=path.lstat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,device=s.st_dev,inode=s.st_ino,
        allocated_bytes=getattr(s,'st_blocks',(s.st_size+511)//512)*512,
        links=s.st_nlink,regular=stat.S_ISREG(s.st_mode))


def controller_versions(args):
    versions=set()
    for arg in args:
        if not arg.endswith(('.py','.sh')):continue
        for old in OLD:
            if re.search(re.escape(old)+r'(?![0-9A-Za-z_])',Path(arg).name) or '/'+old+'/' in arg:
                versions.add(old)
    if '--variant' in args:
        i=args.index('--variant')
        if i+1<len(args) and 'tailcompile_'+args[i+1] in OLD:versions.add('tailcompile_'+args[i+1])
    return versions


def scoped_directories(directories,repo,scratch):
    roots=[repo/'experiments/vtr'/v for v in OLD]
    if scratch:roots.extend(scratch/v for v in OLD)
    # An idle shell in the repo (or /) must not mark the entire tree as busy.
    return {p for p in directories if any(p.is_relative_to(root) for root in roots)}


def active(repo,scratch=None):
    """No command lines/credentials are saved; only live file/directory paths."""
    directories=set();files=set();versions=set();unreadable=0
    if os.name!='posix' or not Path('/proc').is_dir():
        raise RuntimeError('active-process safety checks require Linux/WSL')
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():continue
        try:
            cwd=(process/'cwd').resolve(strict=True)
            directories.add(cwd)
            args=(process/'cmdline').read_bytes().decode(errors='replace').split('\0')
            for arg in args:
                if arg.startswith('/'):
                    p=Path(arg)
                    if p.exists():
                        p=p.resolve()
                        (directories if p.is_dir() else files).add(p)
            # Running old controllers can launch more cases after inspection.
            versions.update(controller_versions(args))
            for fd in (process/'fd').iterdir():
                try:
                    p=fd.resolve(strict=True)
                    if p.is_file():files.add(p)
                except OSError:pass
        except PermissionError:unreadable+=1
        except (OSError,ValueError):pass
    roots={repo/'experiments/vtr'/v for v in versions}
    if scratch:roots.update(scratch/v for v in versions)
    directories.update(roots)
    return scoped_directories(directories,repo,scratch),files,unreadable


def busy(path,directories,files):
    return path in files or any(path.is_relative_to(p) for p in directories)


def scratch_evidence(version_root,folder,repo):
    parts=folder.relative_to(version_root).parts
    version=version_root.name
    if version in ('tailcompile_v0','tailcompile_v1') and len(parts)>=2:
        base=repo/'experiments/vtr'/version/parts[0]/'runs'/parts[1]
    elif version in ('tailcompile_v2','tailcompile_v3') and len(parts)>=3 and parts[1] in ('2x2','2x4','4x4'):
        base=repo/'experiments/vtr'/version/parts[0]/parts[1]/'runs'/parts[2]
    else:return None
    return finished(base)


def roots_for(repo,scratch):
    result=[]
    for name in sorted(OLD):
        path=repo/'experiments/vtr'/name
        if path.is_dir() and not path.is_symlink():result.append((path,False))
    if scratch:
        if scratch in (repo,Path.home().resolve(),Path(scratch.anchor)):
            raise ValueError('scratch cannot be repository, home, or filesystem root')
        for name in sorted(OLD):
            path=scratch/name
            if path.is_dir() and not path.is_symlink():result.append((path,True))
    return result


def plan(a):
    repo=a.repo.resolve();scratch=a.scratch.resolve() if a.scratch else None
    if a.plan.exists():raise ValueError('plan already exists; choose a new --plan filename')
    protected=tracked(repo);directories,open_files,unreadable=active(repo,scratch)
    entries=[];skipped=Counter();summaries=defaultdict(lambda:dict(files=0,allocated_bytes=0,logical_bytes=0))
    for root,is_scratch in roots_for(repo,scratch):
        for base,dirs,names in os.walk(root,followlinks=False):
            folder=Path(base)
            dirs[:]=[d for d in dirs if d not in PROTECTED and not (folder/d).is_symlink()]
            evidence=scratch_evidence(root,folder,repo) if is_scratch else finished(folder)
            if evidence is None:continue
            for name in names:
                kind=raw_kind(name)
                if not kind:continue
                path=folder/name
                if path.is_symlink():skipped['symlink']+=1;continue
                actual=path.resolve()
                if not actual.is_relative_to(root.resolve()):skipped['outside_scope']+=1;continue
                if actual in protected:skipped['git_tracked']+=1;continue
                if busy(actual,directories,open_files):skipped['active']+=1;continue
                s=snapshot(path)
                if not s['regular'] or s['links']!=1:skipped['not_regular_or_hardlinked']+=1;continue
                if time.time_ns()-s['mtime_ns']<a.min_age_hours*3600*1e9:skipped['too_recent']+=1;continue
                entries.append(dict(path=str(actual),root=str(root.resolve()),scratch=is_scratch,
                    kind=kind,stat=s,evidence=str(evidence.resolve()),evidence_sha256=digest(evidence)))
                key=('scratch/' if is_scratch else 'experiments/')+root.name+'/'+kind
                summaries[key]['files']+=1;summaries[key]['allocated_bytes']+=s['allocated_bytes'];summaries[key]['logical_bytes']+=s['size']
    data=dict(schema='tailcompile-old-payload-cleanup-v1',repo=str(repo),scratch=str(scratch) if scratch else None,
        created_unix=time.time(),tool_sha256=digest(__file__),min_age_hours=a.min_age_hours,
        excluded_versions=['tailcompile_v66','tailcompile_v67','tailcompile_v3_training'],
        summaries={k:dict(v,allocated_GiB=round(v['allocated_bytes']/1024**3,3)) for k,v in summaries.items()},
        candidate_files=len(entries),allocated_bytes=sum(e['stat']['allocated_bytes'] for e in entries),
        allocated_GiB=round(sum(e['stat']['allocated_bytes'] for e in entries)/1024**3,3),
        skipped=dict(skipped),unreadable_processes=unreadable,entries=entries,
        deletion='permanent unlink of listed raw payloads only; regeneration requires rerunning VPR')
    save(a.plan,data)
    print(json.dumps({k:v for k,v in data.items() if k!='entries'},indent=2))


def apply(a):
    data=load(a.plan);repo=a.repo.resolve()
    if data['repo']!=str(repo) or data['tool_sha256']!=digest(__file__):raise ValueError('repository/tool changed; create a new plan')
    scratch=Path(data['scratch']) if data['scratch'] else None
    protected=tracked(repo);allowed={str(path.resolve()) for path,_ in roots_for(repo,scratch)}
    directories,open_files,unreadable=active(repo,scratch)
    deleted=[];skipped=[];markers=defaultdict(list);refreshed=time.monotonic()
    run_id=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
    receipt=a.plan.with_name(a.plan.stem+'.'+run_id+'.receipt.json')
    journal=receipt.with_suffix('.jsonl')
    # Append after each unlink so an interrupted apply retains its audit trail.
    journal.touch(exist_ok=False)
    for entry in data['entries']:
        path=Path(entry['path']);reason=None
        if time.monotonic()-refreshed>5:
            directories,open_files,unreadable=active(repo,scratch);refreshed=time.monotonic()
        try:
            root=Path(entry['root'])
            if str(root) not in allowed or not path.is_relative_to(root) or any(p in PROTECTED for p in path.relative_to(root).parts[:-1]):reason='outside_allowed_payload_scope'
            elif not path.is_file() or path.is_symlink() or path.resolve()!=path:reason='missing_or_symlink'
            elif path in protected or busy(path,directories,open_files):reason='tracked_or_active'
            elif raw_kind(path.name)!=entry['kind'] or snapshot(path)!=entry['stat']:reason='payload_changed'
            else:
                evidence=scratch_evidence(root,path.parent,repo) if entry['scratch'] else finished(path.parent)
                if evidence is None or evidence.resolve()!=Path(entry['evidence']) or digest(evidence)!=entry['evidence_sha256']:
                    reason='evidence_changed'
            if reason:skipped.append(dict(path=str(path),reason=reason));continue
            path.unlink();deleted.append(entry)
            with journal.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(path=str(path),bytes=entry['stat']['allocated_bytes'],unix_time=time.time()))+'\n')
            markers[Path(entry['evidence']).parent].append(dict(path=str(path),kind=entry['kind'],bytes=entry['stat']['allocated_bytes']))
        except OSError as error:skipped.append(dict(path=str(path),reason=str(error)))
    save(receipt,dict(plan=str(a.plan),plan_sha256=digest(a.plan),journal=str(journal),deleted_files=len(deleted),
        released_allocated_bytes=sum(e['stat']['allocated_bytes'] for e in deleted),
        deleted_paths=[e['path'] for e in deleted],skipped=skipped,finished_unix=time.time(),unreadable_processes=unreadable))
    for folder,rows in markers.items():
        marker=folder/'payload_pruned.json'
        history=load(marker) if marker.exists() else dict(cleanups=[])
        history['cleanups'].append(dict(receipt=str(receipt),deleted=rows,unix_time=time.time()))
        save(marker,history)
    print(json.dumps(dict(receipt=str(receipt),deleted_files=len(deleted),
        released_GiB=sum(e['stat']['allocated_bytes'] for e in deleted)/1024**3,skipped_files=len(skipped)),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['plan','apply']);p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True);p.add_argument('--scratch',type=Path)
    p.add_argument('--min-age-hours',type=float,default=24)
    a=p.parse_args()
    if a.min_age_hours<0:p.error('min age must be nonnegative')
    globals()[a.phase](a)


if __name__=='__main__':main()
