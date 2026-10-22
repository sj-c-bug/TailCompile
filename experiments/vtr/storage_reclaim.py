"""Inventory and reviewed reclamation of CLOSED v0-v6.7 VTR payloads.

Current v6.8, canonical inputs, models, IR, generated candidates and compact
evidence are never deleted. No directory deletion. Netlist hardlinking is an
explicit immutable-archive operation, not a transparent change to inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import hashlib
import heapq
import json
import math
import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time
import uuid

SCHEMA = 'tailcompile-legacy-storage-reclaim-1'
VERSIONS = {f'tailcompile_{v}' for v in ('v0','v1','v2','v3','v4','v5','v6',
            'v61','v62','v63','v64','v65','v66','v67')}
PROTECTED = {'inputs','input','ir','graphs','graph','device_cache','pool','pools',
             'generation','models','model','bundle','dataset','checkpoints','designs','.git'}
DONE = {'success','unroutable','timeout','other_failure','interrupted','killed','complete'}
NETS = {'packed_net','postrouting_net'}


class Progress:
    """Rate-limited JSON diagnostics; the command's stdout stays machine readable."""

    def __init__(self, operation, seconds=30, *, stream=None, clock=None):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('progress_seconds must be finite and positive')
        self.operation = operation
        self.seconds = seconds
        self.stream = sys.stderr if stream is None else stream
        self.clock = time.monotonic if clock is None else clock
        self.started = self.clock()
        self.phase_started = self.started
        self.last = self.started
        self.phase_name = None
        self.details = {}
        self.hashed_bytes = 0
        self.hashed_files = 0

    def phase(self, name, **details):
        self.phase_name = name
        self.phase_started = self.clock()
        self.details = details
        self.update(force=True)

    def update(self, *, force=False, **details):
        self.details.update(details)
        now = self.clock()
        if not force and now-self.last < self.seconds:
            return
        row = dict(event='storage_progress', operation=self.operation, phase=self.phase_name,
                   elapsed_seconds=round(now-self.started,3),
                   phase_elapsed_seconds=round(now-self.phase_started,3), **self.details)
        print(json.dumps(row,ensure_ascii=False),file=self.stream,flush=True)
        self.last = now


def _digest(path, progress=None, *, total_bytes=None):
    if progress is None:
        return digest(path)
    before = progress.hashed_bytes

    def advanced(bytes_read):
        progress.hashed_bytes = before+bytes_read
        progress.update(current_path=str(path), file_bytes_read=bytes_read,
                        file_size_bytes=total_bytes, hashed_bytes=progress.hashed_bytes,
                        hashed_files=progress.hashed_files)

    value = digest(path, progress=advanced)
    progress.hashed_files += 1
    progress.update(hashed_files=progress.hashed_files)
    return value


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def digest(path, progress=None):
    value = hashlib.sha256()
    bytes_read = 0
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
            bytes_read += len(block)
            if progress is not None:
                progress(bytes_read)
    if progress is not None and bytes_read == 0:
        progress(0)
    return value.hexdigest()


def snapshot(path):
    s = Path(path).lstat()
    return dict(size=s.st_size, mtime_ns=s.st_mtime_ns, device=s.st_dev, inode=s.st_ino,
                links=s.st_nlink, mode=stat.S_IMODE(s.st_mode), regular=stat.S_ISREG(s.st_mode),
                allocated_bytes=getattr(s,'st_blocks',(s.st_size+511)//512)*512)


def raw_kind(name):
    lower = name.lower()
    if re.search(r'(?:\.net[._]?post[_-]?routing|[._]post[_-]?routing\.net)$', lower):
        return 'postrouting_net'
    if lower.endswith('.net'):
        return 'packed_net'
    if re.search(r'(rr[_-]?graph|rrg).*\.(xml|bin)(\.gz)?$', lower):
        return 'rr_graph'
    if lower.endswith(('.route','.route.gz')):
        return 'route'
    if lower.endswith(('.rpt','.echo')):
        return 'detailed_report'
    return None


def file_type(name):
    return raw_kind(name) or ''.join(Path(name.lower()).suffixes[-2:]) or '[no_extension]'


def roots_for(repo, scratch=None):
    repo = Path(repo).resolve()
    if not repo.is_dir() or repo in (Path(repo.anchor),Path.home().resolve()):
        raise ValueError('repo must be an explicit working directory, not home or filesystem root')
    result, seen = [], set()
    bases = [(repo/'experiments/vtr','experiments')]
    if scratch is not None:
        scratch = Path(scratch).resolve()
        if scratch in (repo, Path.home().resolve(), Path(scratch.anchor)):
            raise ValueError('scratch must be an explicit scratch directory, not repository/home/root')
        bases.append((scratch,'scratch'))
    for base,location in bases:
        for version in sorted(VERSIONS):
            path = base/version
            if path.is_dir() and not path.is_symlink():
                resolved = path.resolve()
                if resolved not in seen:
                    result.append((resolved, location, version))
                    seen.add(resolved)
    return result


def tracked_files(repo):
    repo = Path(repo).resolve()
    result = subprocess.run(['git','-C',str(repo),'ls-files','-z'], capture_output=True, check=True)
    return {(repo/os.fsdecode(name)).resolve() for name in result.stdout.split(b'\0') if name}


def controller_versions(args):
    versions = set()
    for arg in args:
        if arg.endswith(('.py','.sh')):
            for version in VERSIONS:
                if '/'+version+'/' in arg.replace('\\','/') or re.search(
                    re.escape(version)+r'(?![0-9A-Za-z_])', Path(arg).name):
                    versions.add(version)
    if '--variant' in args:
        index = args.index('--variant')
        if index+1 < len(args) and 'tailcompile_'+args[index+1] in VERSIONS:
            versions.add('tailcompile_'+args[index+1])
    return versions


def active_paths(repo, scratch=None, progress=None):
    if os.name != 'posix' or not Path('/proc').is_dir():
        raise RuntimeError('Destructive cleanup requires Linux/WSL process inspection')
    directories,files,versions = set(),set(),set()
    unreadable = processes = 0
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name)==os.getpid():
            continue
        processes += 1
        if progress is not None:
            progress.update(processes=processes, current_path=str(process))
        try:
            try:
                directories.add((process/'cwd').resolve(strict=True))
            except OSError:
                pass
            args = (process/'cmdline').read_bytes().decode(errors='replace').split('\0')
            versions.update(controller_versions(args))
            for arg in args:
                if arg.startswith('/'):
                    p = Path(arg)
                    if p.exists():
                        (directories if p.is_dir() else files).add(p.resolve())
            for fd in (process/'fd').iterdir():
                try:
                    p = fd.resolve(strict=True)
                    if p.is_file():
                        files.add(p)
                except OSError:
                    pass
        except PermissionError:
            unreadable += 1
        except OSError:
            pass
    roots = roots_for(repo,scratch)
    directories = {p for p in directories if any(p.is_relative_to(r) for r,_,_ in roots)}
    directories.update(r for r,_,version in roots if version in versions)
    return directories,files,unreadable


def busy(path, active):
    directories,files,_ = active
    return path in files or any(path.is_relative_to(p) for p in directories)


class References(set):
    def __init__(self):
        super().__init__()
        self.current = set()
        self.unreadable = []


def _looks_path(value):
    lower = value.lower()
    return ('/' in value or '\\' in value or raw_kind(value) is not None or
            lower.endswith(('.blif','.fplace','.place','.pt','.tsv','.csv','.json')))


def _paths(value, folder, progress=None):
    result = set()
    if isinstance(value, dict):
        for index,child in enumerate(value.values()):
            if progress is not None and index % 1024 == 0:
                progress.update()
            result.update(_paths(child, folder, progress))
    elif isinstance(value, list):
        for index,child in enumerate(value):
            if progress is not None and index % 1024 == 0:
                progress.update()
            result.update(_paths(child, folder, progress))
    elif isinstance(value, str):
        # Avoid resolving millions of ordinary atom/resource names in logic IR.
        if not _looks_path(value):
            return result
        tokens = [value]
        try:
            tokens.extend(shlex.split(value))
        except ValueError:
            pass
        for token in tokens:
            if '=' in token and token.startswith('-'):
                token = token.split('=',1)[1]
            # Only paths to a potentially removable RAW kind are dependencies
            # here. Hierarchical atom names contain '/' too; resolving millions
            # of those would issue millions of pointless NAS metadata requests.
            if not token or token.startswith('-') or raw_kind(token) is None:
                continue
            p = Path(token)
            if not p.is_absolute():
                p = folder/p
            try:
                result.add(p.resolve())
            except (OSError,ValueError):
                pass
    return result


def streamed_paths(path, progress=None):
    """Extract JSON string references without materializing huge IR arrays.

    Carry an incomplete STRING, never an arbitrary tail that could reverse JSON
    quote pairing. Huge numeric arrays need no retained buffer. Abnormally long
    strings fail closed rather than silently losing possible input references.
    """
    result,tail,characters_read = set(),'',0
    token = re.compile(r'"(?:[^"\\]|\\.)*"')
    with Path(path).open(encoding='utf-8',errors='strict') as stream:
        while True:
            chunk = stream.read(1024*1024)
            if not chunk:
                break
            characters_read += len(chunk)
            text = tail+chunk
            position,tail = 0,''
            while True:
                opening = text.find('"',position)
                if opening < 0:
                    break
                match = token.match(text,opening)
                if match is None:
                    # Do not search escaped quotes INSIDE an unfinished string.
                    tail = text[opening:]
                    break
                position = match.end()
                literal = match.group()
                if not any(mark in literal for mark in ('/','\\','.net','.route','.xml','.rpt','.echo')):
                    continue
                result.update(_paths(json.loads(literal),Path(path).parent))
            if len(tail) > 65536:
                raise ValueError(f'Unusually long JSON string in provenance: {path}')
            if progress is not None:
                progress(characters_read)
    if tail:
        raise ValueError(f'Incomplete JSON string in provenance: {path}')
    return result


def collect_references(repo, scratch=None, progress=None):
    """Read small provenance JSON, including current and unfinished consumers.

    Huge feature/target arrays are not parsed; every .net path in eligible JSON
    is retained. Malformed provenance fails closed for netlist deletion.
    """
    repo = Path(repo).resolve()
    result = References()
    directories = files_seen = json_files = 0
    if progress is not None:
        progress.phase('collect_references', directories=0, files_seen=0, json_files=0,
                       references=0, unreadable_dependency_records=0)
    bases = [repo/'experiments/vtr']
    if scratch is not None and Path(scratch).resolve() != repo/'vtr_work':
        bases.append(Path(scratch).resolve())
    elif scratch is not None:
        bases.append(Path(scratch).resolve())
    for base in bases:
        if not base.is_dir():
            continue
        for directory,dirs,names in os.walk(base, followlinks=False):
            folder = Path(directory)
            directories += 1
            files_seen += len(names)
            if progress is not None:
                progress.update(current_path=str(folder), directories=directories,
                                files_seen=files_seen, json_files=json_files,
                                references=len(result),
                                unreadable_dependency_records=len(result.unreadable))
            dirs[:] = [d for d in dirs if d not in {'.git','__pycache__','models','model','checkpoints'}
                       and not (folder/d).is_symlink()]
            for name in names:
                if not name.endswith('.json') or name == 'target.json':
                    continue
                path = folder/name
                if path.is_symlink():
                    continue
                # Generated evidence/plans contain paths to OUTPUTS, not input
                # dependencies, and must not make previous cleanups self-pinning.
                if name.startswith(('payload_prun','storage_')) or 'cleanup' in name or 'reclaim' in name:
                    continue
                json_files += 1
                try:
                    if progress is not None:
                        progress.update(current_path=str(path),json_files=json_files,
                                        json_characters_read=0)
                    paths = (streamed_paths(path, progress=(lambda count: progress.update(
                                 json_characters_read=count)) if progress is not None else None)
                             if path.stat().st_size > 64*1024**2
                             else _paths(load(path),folder,progress))
                    result.update(paths)
                    if not any(part in VERSIONS for part in path.parts):
                        result.current.update(paths)
                except (OSError,ValueError):
                    # Even IR/summary records may carry the only live input
                    # reference; unreadable provenance must not fail open.
                    result.unreadable.append(str(path))
    if progress is not None:
        progress.update(force=True, status='complete', directories=directories,
                        files_seen=files_seen, json_files=json_files, references=len(result),
                        unreadable_dependency_records=len(result.unreadable))
    return result


def finished(folder):
    for name in ('manifest.json','trial.json','outcome.json'):
        path = folder/name
        if not path.is_file() or path.is_symlink():
            continue
        try:
            row = load(path)
            status = row.get('vpr_status',row.get('status'))
            if status in DONE:
                return path,row
        except (OSError,ValueError):
            pass
    return None,None


def evidence_for(root,folder,location,version,repo):
    direct = finished(folder)
    if direct[0] or location != 'scratch':
        return direct
    parts = folder.relative_to(root).parts
    if version in {'tailcompile_v0','tailcompile_v1'} and len(parts)==2:
        return finished(Path(repo)/'experiments/vtr'/version/parts[0]/'runs'/parts[1])
    if version in {'tailcompile_v2','tailcompile_v3'} and len(parts)==3 and parts[1] in {'2x2','2x4','4x4'}:
        return finished(Path(repo)/'experiments/vtr'/version/parts[0]/parts[1]/'runs'/parts[2])
    return None,None


def archived_input_copy(path,folder,row):
    """v0-v3 archives duplicate a scratch input twice; neither archive is read.

    Exact references are checked separately. Never extend this to the actual
    scratch input or an arbitrary input.net without the archived-copy evidence.
    """
    if not any(tag in path.name.lower() for tag in ('.locked-input.net','.input-packed.net')):
        return False
    if not all(row.get(key) for key in ('scratch','archive','input_packed_net','input_packed_net_sha256')):
        return False
    try:
        return Path(row['archive']).resolve()==folder and Path(row['scratch']).resolve()!=folder
    except (OSError,ValueError):
        return False


def _allowed_file(path,root,active,tracked,min_age_hours):
    if path.is_symlink():
        return None,'symlink'
    actual = path.resolve()
    if actual.parent != path.parent.resolve() or not actual.is_relative_to(root):
        return None,'outside_scope'
    info = snapshot(path)
    if not info['regular'] or info['links'] != 1:
        return None,'not_regular_or_hardlinked'
    if actual in tracked:
        return None,'git_tracked'
    if busy(actual,active):
        return None,'active'
    if time.time_ns()-info['mtime_ns'] < min_age_hours*3600*1e9:
        return None,'too_recent'
    return info,None


def plan_reclaim(repo, scratch=None, *, include_netlists=False, deduplicate_netlists=False,
                 min_age_hours=24, active=None, tracked=None, references=None, progress=None):
    if progress is not None:
        progress.phase('enumerate_roots')
    repo = Path(repo).resolve()
    scratch = Path(scratch).resolve() if scratch is not None else None
    roots = roots_for(repo,scratch)
    if min_age_hours < 0 or not math.isfinite(min_age_hours):
        raise ValueError('min_age_hours must be finite and nonnegative')
    if progress is not None:
        progress.phase('inspect_processes', roots=len(roots))
    active = active_paths(repo,scratch,progress) if active is None else active
    if progress is not None:
        progress.phase('collect_tracked_files')
    tracked = tracked_files(repo) if tracked is None else {Path(p).resolve() for p in tracked}
    references = collect_references(repo,scratch,progress) if references is None else references
    references = references if isinstance(references,References) else set(Path(p).resolve() for p in references)
    blocked_nets = bool(getattr(references,'unreadable',[]))
    entries,skipped,dedup = [],Counter(),defaultdict(list)
    summaries = defaultdict(lambda:dict(files=0,allocated_bytes=0,logical_bytes=0))
    directories = files_seen = 0
    if progress is not None:
        progress.phase('scan_payloads', roots=len(roots), directories=0, files_seen=0,
                       candidate_files=0)
    for root,location,version in roots:
        for base,dirs,names in os.walk(root,followlinks=False):
            folder = Path(base)
            directories += 1
            dirs[:] = [d for d in dirs if d not in PROTECTED and not (folder/d).is_symlink()]
            evidence,row = evidence_for(root,folder,location,version,repo)
            # All payloads in this folder use the same completed record. Hash
            # it once while planning; apply still revalidates every entry.
            evidence_path = evidence_sha256 = None
            for name in names:
                files_seen += 1
                if progress is not None:
                    progress.update(current_path=str(folder/name),directories=directories,
                                    files_seen=files_seen,candidate_files=len(entries))
                kind = raw_kind(name)
                if kind is None:
                    continue
                path = folder/name
                if evidence is None:
                    skipped['no_completed_evidence'] += 1
                    continue
                info,reason = _allowed_file(path,root,active,tracked,min_age_hours)
                if reason:
                    skipped[reason] += 1
                    continue
                actual = path.resolve()
                if evidence_sha256 is None:
                    evidence_path = str(evidence.resolve())
                    evidence_sha256 = _digest(evidence,progress)
                entry = dict(action='delete', path=str(actual), root=str(root),
                    location=location,version=version,kind=kind,stat=info,
                    evidence=evidence_path, evidence_sha256=evidence_sha256)
                protected_name = (name.lower()=='input.net' or 'locked-input' in name.lower() or 'input-packed' in name.lower())
                if protected_name and archived_input_copy(actual,folder,row):
                    protected_name = False
                referenced = actual in references
                can_delete = not (protected_name or referenced or (kind in NETS and (not include_netlists or blocked_nets)))
                if can_delete:
                    entries.append(entry)
                else:
                    skipped['input_dependency' if protected_name or referenced else 'netlists_not_enabled_or_unreadable_provenance'] += 1
                    if (deduplicate_netlists and kind in NETS and not blocked_nets and
                            actual not in getattr(references,'current',set())):
                        dedup[info['device'],info['size']].append(entry)
    if deduplicate_netlists:
        groups = [group for group in dedup.values() if len(group)>=2]
        if progress is not None:
            progress.hashed_bytes = progress.hashed_files = 0
            progress.phase('hash_netlist_duplicates', groups=len(groups),
                           files_to_hash=sum(len(group) for group in groups),
                           bytes_to_hash=sum(entry['stat']['size'] for group in groups for entry in group),
                           hashed_files=0, hashed_bytes=0)
        for group in groups:
            hashes = defaultdict(list)
            for entry in group:
                value = _digest(entry['path'],progress,total_bytes=entry['stat']['size'])
                if snapshot(entry['path'])!=entry['stat']:
                    skipped['changed_during_hash'] += 1
                else:
                    hashes[value].append(entry)
            for checksum,matches in hashes.items():
                if len(matches)<2:
                    continue
                matches.sort(key=lambda e:e['path'])
                master = matches[0]
                for entry in matches[1:]:
                    entries.append(dict(entry,action='hardlink',canonical=master['path'],
                        canonical_stat=master['stat'],canonical_evidence=master['evidence'],
                        canonical_evidence_sha256=master['evidence_sha256'],sha256=checksum,
                        freeze_readonly=True))
    if progress is not None:
        progress.phase('summarize_plan', candidate_files=len(entries),
                       skipped_files=sum(skipped.values()),directories=directories,files_seen=files_seen)
    for entry in entries:
        key = entry['location']+'/'+entry['version']+'/'+entry['action']+'/'+entry['kind']
        summaries[key]['files'] += 1
        summaries[key]['allocated_bytes'] += entry['stat']['allocated_bytes']
        summaries[key]['logical_bytes'] += entry['stat']['size']
    allocated = sum(e['stat']['allocated_bytes'] for e in entries)
    return dict(schema=SCHEMA,repo=str(repo),scratch=str(scratch) if scratch else None,
        created_unix=time.time(),tool_sha256=_digest(__file__,progress),min_age_hours=min_age_hours,
        include_netlists=include_netlists,deduplicate_netlists=deduplicate_netlists,
        candidate_files=len(entries),deletion_files=sum(e['action']=='delete' for e in entries),
        dedup_files=sum(e['action']=='hardlink' for e in entries),allocated_bytes=allocated,
        allocated_GiB=allocated/1024**3,reclaimable_GiB=allocated/1024**3,
        summaries={k:dict(v,allocated_GiB=v['allocated_bytes']/1024**3) for k,v in summaries.items()},
        skipped=dict(skipped),unreadable_dependency_records=getattr(references,'unreadable',[]),
        unreadable_processes=active[2],entries=entries,
        warning='Delete is permanent; raw outputs require VPR rerun. Hardlinks require immutable archived runs: same bytes/paths, shared inode; root can bypass read-only mode.')


def _journal(path,value):
    if path is None:
        return
    with Path(path).open('a',encoding='utf-8') as stream:
        stream.write(json.dumps(value,ensure_ascii=False)+'\n')
        stream.flush()
        os.fsync(stream.fileno())


def apply_reclaim(plan, *, repo=None, scratch=None, active=None, tracked=None,
                  references=None, journal=None, ack_immutable_archives=False, progress=None):
    if progress is not None:
        progress.phase('validate_plan',entries=len(plan.get('entries',[])))
    repo = Path(repo or plan['repo']).resolve()
    scratch = Path(scratch).resolve() if scratch is not None else (Path(plan['scratch']) if plan['scratch'] else None)
    if plan.get('schema')!=SCHEMA or plan.get('tool_sha256')!=_digest(__file__,progress) or str(repo)!=plan['repo']:
        raise ValueError('Plan schema, tool or repository changed; generate a new plan')
    if (str(scratch) if scratch else None)!=plan['scratch']:
        raise ValueError('Scratch scope changed')
    if any(e['action']=='hardlink' for e in plan['entries']) and not ack_immutable_archives:
        raise ValueError('Hardlink dedup requires --ack-immutable-archives; never rerun these archives in place')
    if progress is not None:
        progress.phase('enumerate_roots')
    scopes = {str(r):(location,version) for r,location,version in roots_for(repo,scratch)}
    allowed = set(scopes)
    dynamic_active = active is None
    if progress is not None:
        progress.phase('inspect_processes',roots=len(scopes))
    active = active_paths(repo,scratch,progress) if active is None else active
    checked_active = time.monotonic()
    if progress is not None:
        progress.phase('collect_tracked_files')
    tracked = tracked_files(repo) if tracked is None else {Path(p).resolve() for p in tracked}
    references = collect_references(repo,scratch,progress) if references is None else references
    references = references if isinstance(references,References) else {Path(p).resolve() for p in references}
    results,skipped,masters = [],[],{}
    if progress is not None:
        progress.phase('apply_entries',entries=len(plan['entries']),entries_checked=0,
                       completed_files=0,skipped_files=0)
    for index,entry in enumerate(plan['entries']):
        path,root = Path(entry['path']),Path(entry['root'])
        if progress is not None:
            progress.update(current_path=str(path),entries_checked=index,
                            completed_files=len(results),skipped_files=len(skipped))
        reason = None
        try:
            if dynamic_active and time.monotonic()-checked_active > 5:
                active = active_paths(repo,scratch,progress)
                checked_active = time.monotonic()
            if (str(root) not in allowed or not path.is_absolute() or not path.is_relative_to(root) or
                    any(p in PROTECTED for p in path.relative_to(root).parts[:-1]) or
                    path.is_symlink() or path.resolve()!=path or not path.is_file()):
                reason = 'outside_scope_or_missing'
            elif path in tracked or busy(path,active):
                reason = 'tracked_or_active'
            elif raw_kind(path.name)!=entry['kind'] or snapshot(path)!=entry['stat']:
                reason = 'payload_changed'
            else:
                location,version = scopes[str(root)]
                evidence,row = evidence_for(root,path.parent,location,version,repo)
                if (evidence is None or evidence.resolve()!=Path(entry['evidence']) or
                        _digest(evidence,progress)!=entry['evidence_sha256']):
                    reason = 'evidence_changed_or_not_completed'
            if reason is None and entry['action']=='delete' and entry['kind'] in NETS:
                lower = path.name.lower()
                named_input = lower=='input.net' or 'locked-input' in lower or 'input-packed' in lower
                if not plan.get('include_netlists'):
                    reason = 'netlist_deletion_not_enabled'
                elif named_input and not archived_input_copy(path,path.parent,row):
                    reason = 'protected_input_name'
            if reason is None and entry['action']=='delete' and (path in references or (entry['kind'] in NETS and getattr(references,'unreadable',[]))):
                reason = 'input_dependency'
            elif reason is None and entry['action']=='hardlink' and path in getattr(references,'current',set()):
                reason = 'current_dependency'
            elif reason is None and entry['action']=='hardlink' and getattr(references,'unreadable',[]):
                reason = 'unreadable_dependency_records'
            if reason:
                skipped.append(dict(path=str(path),reason=reason))
                continue
            event = dict(path=str(path),action=entry['action'],bytes=entry['stat']['allocated_bytes'],unix_time=time.time())
            if entry['action']=='delete':
                _journal(journal,dict(event,event='delete_intent'))
                path.unlink()
            elif entry['action']=='hardlink':
                canonical = Path(entry['canonical'])
                croot = next((Path(r) for r in allowed if canonical.is_relative_to(Path(r))),None)
                if (croot is None or canonical.is_symlink() or canonical.resolve()!=canonical or not canonical.is_file() or
                        canonical in tracked or canonical in getattr(references,'current',set()) or busy(canonical,active) or
                        any(p in PROTECTED for p in canonical.relative_to(croot).parts[:-1])):
                    raise ValueError('Canonical archive source is protected or outside allowed scope')
                if canonical not in masters:
                    if snapshot(canonical)!=entry['canonical_stat']:
                        raise ValueError('Canonical archive source changed')
                    clocation,cversion = scopes[str(croot)]
                    cevidence,_ = evidence_for(croot,canonical.parent,clocation,cversion,repo)
                    if (cevidence is None or cevidence.resolve()!=Path(entry['canonical_evidence']) or
                            _digest(cevidence,progress)!=entry['canonical_evidence_sha256']):
                        raise ValueError('Canonical archive evidence changed')
                    masters[canonical] = entry['sha256']
                if (_digest(canonical,progress,total_bytes=entry['canonical_stat']['size'])!=entry['sha256'] or
                        _digest(path,progress,total_bytes=entry['stat']['size'])!=entry['sha256']):
                    raise ValueError('Dedup content changed')
                _journal(journal,dict(event,event='hardlink_intent',canonical=str(canonical),freeze_readonly=True))
                # Explicitly freeze historical copies; canonical source inputs
                # were excluded from planning and cannot become a shared master.
                os.chmod(canonical,entry['canonical_stat']['mode'] & ~0o222)
                temporary = path.with_name(path.name+'.reclaim-link-'+uuid.uuid4().hex)
                try:
                    os.link(canonical,temporary)
                    temporary.replace(path)
                finally:
                    temporary.unlink(missing_ok=True)
            else:
                raise ValueError('Unknown plan action')
            results.append(event)
            _journal(journal,dict(event,event='completed'))
        except (OSError,ValueError) as error:
            skipped.append(dict(path=str(path),reason=str(error)))
    released = sum(e['bytes'] for e in results)
    if progress is not None:
        progress.update(force=True,status='complete',entries_checked=len(plan['entries']),
                        completed_files=len(results),skipped_files=len(skipped),released_bytes=released)
    return dict(deleted_files=sum(e['action']=='delete' for e in results),
        deduplicated_files=sum(e['action']=='hardlink' for e in results),released_bytes=released,
        released_GiB=released/1024**3,completed=results,skipped=skipped,
        deletion='Permanent unlink; hardlinks freeze only historical copies. Keep journal/receipt.')


def inventory(repo,scratch=None,report=None,progress=None):
    if progress is not None:
        progress.phase('enumerate_roots')
    roots = roots_for(repo,scratch)
    if scratch is not None and Path(scratch).is_dir():
        # Include unknown old temp layouts for diagnosis, NOT deletion scope.
        known = {r for r,_,_ in roots}
        for path in Path(scratch).resolve().iterdir():
            if path.is_dir() and not path.is_symlink() and path.resolve() not in known:
                roots.append((path.resolve(),'scratch_unclassified',path.name))
    groups,seen,largest,errors = defaultdict(lambda:dict(files=0,logical_bytes=0,allocated_bytes=0,hardlinked_files=0)),set(),[],[]
    directories = files_seen = 0
    if progress is not None:
        progress.phase('inventory_files',roots=len(roots),directories=0,files_seen=0)
    for root,location,version in roots:
        for base,dirs,names in os.walk(root,followlinks=False):
            folder = Path(base)
            directories += 1
            dirs[:] = [d for d in dirs if d!='.git' and not (folder/d).is_symlink()]
            for name in names:
                path = folder/name
                files_seen += 1
                if progress is not None:
                    progress.update(current_path=str(path),directories=directories,
                                    files_seen=files_seen,errors=len(errors))
                try:
                    if path.is_symlink():
                        continue
                    info = snapshot(path)
                    if not info['regular']:
                        continue
                    group = groups[location+'/'+version+'/'+file_type(name)]
                    group['files'] += 1
                    group['logical_bytes'] += info['size']
                    group['hardlinked_files'] += int(info['links']>1)
                    inode = info['device'],info['inode']
                    if inode not in seen:
                        group['allocated_bytes'] += info['allocated_bytes']
                        seen.add(inode)
                    row = info['size'],str(path)
                    if len(largest)<30:
                        heapq.heappush(largest,row)
                    elif row>largest[0]:
                        heapq.heapreplace(largest,row)
                except OSError as error:
                    errors.append(dict(path=str(path),error=str(error)))
    result = dict(schema='tailcompile-legacy-storage-inventory-1',repo=str(Path(repo).resolve()),
        summaries={k:dict(v,allocated_GiB=v['allocated_bytes']/1024**3) for k,v in sorted(groups.items(),key=lambda x:-x[1]['allocated_bytes'])},
        allocated_bytes=sum(v['allocated_bytes'] for v in groups.values()),
        logical_bytes=sum(v['logical_bytes'] for v in groups.values()),
        largest_files=[dict(path=p,logical_bytes=n) for n,p in sorted(largest,reverse=True)],errors=errors)
    if report:
        save(report,result)
    if progress is not None:
        progress.update(force=True,status='complete',directories=directories,
                        files_seen=files_seen,errors=len(errors),allocated_bytes=result['allocated_bytes'])
    return result


@contextmanager
def lock_archives(repo,scratch=None,progress=None):
    if progress is not None:
        progress.phase('lock_archives',directories=0,locks=0)
    if os.name!='posix':
        raise RuntimeError('Cleanup CLI requires Linux/WSL process and flock checks')
    import fcntl
    locks=[]
    directories = 0
    try:
        for root,_,_ in roots_for(repo,scratch):
            for base,dirs,names in os.walk(root,followlinks=False):
                folder=Path(base)
                directories += 1
                if progress is not None:
                    progress.update(current_path=str(folder),directories=directories,locks=len(locks))
                dirs[:]=[d for d in dirs if d not in PROTECTED and not (folder/d).is_symlink()]
                if '.lock' in names and not (folder/'.lock').is_symlink():
                    fd=os.open(folder/'.lock',os.O_RDONLY)
                    locks.append(fd)
                    try:
                        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        raise RuntimeError(f'Old cohort still active: {folder}')
        yield
    finally:
        for fd in locks:
            os.close(fd)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('inventory','plan','apply'))
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--scratch',type=Path)
    parser.add_argument('--report',type=Path)
    parser.add_argument('--plan',type=Path)
    parser.add_argument('--include-netlists',action='store_true')
    parser.add_argument('--deduplicate-netlists',action='store_true')
    parser.add_argument('--ack-immutable-archives',action='store_true')
    parser.add_argument('--min-age-hours',type=float,default=24)
    parser.add_argument('--progress-seconds',type=float,default=30,
                        help='stderr JSON progress interval in seconds (default: 30)')
    args=parser.parse_args()
    if not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0:
        parser.error('--progress-seconds must be finite and positive')
    progress=Progress(args.phase,args.progress_seconds)
    progress.phase('starting',repo=str(args.repo),scratch=str(args.scratch) if args.scratch else None)
    if args.phase=='inventory':
        result=inventory(args.repo,args.scratch,args.report,progress=progress)
    else:
        if args.plan is None:
            parser.error('--plan required; use a writable workspace logs/storage directory')
        with lock_archives(args.repo,args.scratch,progress=progress):
            if args.phase=='plan':
                if args.plan.exists():
                    raise ValueError('Plan already exists; choose a new filename')
                result=plan_reclaim(args.repo,args.scratch,include_netlists=args.include_netlists,
                    deduplicate_netlists=args.deduplicate_netlists,min_age_hours=args.min_age_hours,
                    progress=progress)
                progress.phase('write_plan',current_path=str(args.plan),candidate_files=result['candidate_files'])
                save(args.plan,result)
            else:
                progress.phase('load_plan',current_path=str(args.plan))
                plan=load(args.plan)
                stamp=time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
                journal=args.plan.with_name(args.plan.stem+'.'+stamp+'.journal.jsonl')
                journal.touch(exist_ok=False)
                result=apply_reclaim(plan,repo=args.repo,scratch=args.scratch,journal=journal,
                    ack_immutable_archives=args.ack_immutable_archives,progress=progress)
                receipt=args.plan.with_name(args.plan.stem+'.'+stamp+'.receipt.json')
                progress.phase('write_receipt',current_path=str(receipt))
                save(receipt,result)
                result.update(receipt=str(receipt),journal=str(journal))
    progress.phase('complete')
    print(json.dumps({k:v for k,v in result.items() if k not in {'entries','completed'}},indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
