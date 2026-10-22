"""Inventory and prune reproducible v6.8 VPR payloads, never source inputs.

Manual cleanup requires the cohort lock and a reviewed plan. Automatic cleanup
is called only after VPR has exited and trial.json + metrics.json are committed.
The raw payload is reproducible, not recoverable from the compact evidence.
"""
from __future__ import annotations

import argparse
from collections.abc import Set
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
import socket
import stat
import subprocess
import sys
import threading
import time
import uuid

SCHEMA = 'tailcompile-v68-storage-plan-1'
TERMINAL = {'success', 'unroutable', 'timeout', 'other_failure', 'killed'}
PROTECTED = {'inputs', 'input', 'graphs', 'graph', 'device_cache', 'generation',
             'models', 'model', 'checkpoints', 'bundle', 'dataset', 'ir', '.git',
             '_interrupted', '_attempts'}


class PreparedPaths(frozenset):
    """Operation-local, already canonicalized safety snapshot; never persistent."""


def prepare_paths(values, *, already_resolved=False):
    if isinstance(values, PreparedPaths):
        return values
    if already_resolved:
        return PreparedPaths(values)
    return PreparedPaths(Path(value).resolve() for value in values)


def prepare_dependencies(values):
    # Membership-only sets must not be copied once per folder. Materialize a
    # one-shot iterable only once at the operation boundary.
    if values is None:
        return frozenset()
    return values if isinstance(values, Set) else frozenset(values)


def notify(progress, phase, **fields):
    if progress is not None:
        progress(phase, **fields)


class ProgressReporter:
    """CLI stderr heartbeat, including time spent in slow setup/syscalls."""
    def __init__(self, operation, seconds=30):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('progress-seconds must be finite and positive')
        self.operation, self.seconds = operation, seconds
        self.started = self.phase_started = time.monotonic()
        self.state = dict(phase='startup')
        self.lock, self.stop = threading.Lock(), threading.Event()
        self.thread = threading.Thread(target=self._heartbeat, daemon=True)

    def _emit_locked(self):
        now = time.monotonic()
        print(json.dumps(dict(event='storage_progress', operation=self.operation,
            elapsed_seconds=round(now-self.started, 1),
            phase_elapsed_seconds=round(now-self.phase_started, 1), **self.state),
            ensure_ascii=True), file=sys.stderr, flush=True)

    def __call__(self, phase, **fields):
        with self.lock:
            changed = phase != self.state['phase']
            if changed:
                self.state = dict(phase=phase)
                self.phase_started = time.monotonic()
            self.state.update(fields)
            if changed:
                self._emit_locked()

    def _heartbeat(self):
        while not self.stop.wait(self.seconds):
            with self.lock:
                self._emit_locked()

    def __enter__(self):
        with self.lock:
            self._emit_locked()
        self.thread.start()
        return self

    def __exit__(self, kind, value, traceback):
        self.stop.set()
        self.thread.join()
        self('failed' if kind else 'complete', **({'error':str(value)} if kind else {}))


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def snapshot(path):
    value = Path(path).lstat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                device=value.st_dev, inode=value.st_ino, links=value.st_nlink,
                allocated_bytes=getattr(value, 'st_blocks', (value.st_size + 511)//512)*512,
                regular=stat.S_ISREG(value.st_mode))


def raw_kind(name):
    lower = name.lower()
    if re.search(r'(?:\.net[._]?post[_-]?routing|[._]post[_-]?routing\.net)$', lower):
        return 'postrouting_net'
    if lower.endswith('.net'):
        return 'packed_net'
    if lower.endswith(('.route', '.route.gz')):
        return 'route'
    if lower.endswith(('.rpt', '.echo')):
        return 'detailed_report'
    return None


def file_type(name):
    lower = name.lower()
    if raw_kind(lower):
        return raw_kind(lower)
    if re.search(r'(rr[_-]?graph|rrg).*\.(xml|bin)(\.gz)?$', lower):
        return 'rr_graph'
    suffix = ''.join(Path(lower).suffixes[-2:])
    return suffix or '[no_extension]'


def tracked_files(repo, folder=None, *, progress=None):
    if repo is None:
        return set()
    repo = Path(repo).resolve()
    command = ['git', '-C', str(repo), 'ls-files', '-z']
    if folder is not None:
        relative = Path(folder).resolve().relative_to(repo).as_posix()
        # Literal pathspec avoids interpreting design names as Git wildcards.
        command += ['--', ':(literal)'+relative]
    result = subprocess.run(command, capture_output=True, check=True)
    names = [name for name in result.stdout.split(b'\0') if name]
    paths = set()
    for index, name in enumerate(names, 1):
        paths.add((repo/os.fsdecode(name)).resolve())
        if index % 256 == 0 or index == len(names):
            notify(progress, 'tracked_files', processed_paths=index, total_paths=len(names))
    return PreparedPaths(paths)


def active_paths(root=None):
    """Open files and worker cwd; no command lines/credentials are recorded."""
    result = set()
    active_paths.unreadable_unrelated_processes = 0
    scope = Path(root).resolve() if root is not None else None
    if os.name != 'posix' or not Path('/proc').is_dir():
        return result  # Portable unit tests; the destructive CLI requires Linux.
    for process in Path('/proc').iterdir():
        if not process.name.isdigit() or int(process.name) == os.getpid():
            continue
        cwd = None
        try:
            try:
                cwd = (process/'cwd').resolve(strict=True)
                result.add(cwd)
            except (OSError, RuntimeError):
                pass
            for descriptor in (process/'fd').iterdir():
                try:
                    target = descriptor.resolve(strict=True)
                    if target.is_file():
                        result.add(target)
                except (OSError, RuntimeError):
                    pass
        except PermissionError:
            # Restricted unrelated system services are common in containers.
            # They may be excluded only with readable argv proving an unrelated
            # command; an unknown or relevant same-user worker remains a blocker.
            try:
                if process.stat().st_uid == os.getuid():
                    args = (process/'cmdline').read_bytes().decode(errors='replace').split('\0')
                    related = (scope is None or (cwd is not None and cwd.is_relative_to(scope)) or
                               any(str(scope) in arg or 'tailcompile' in arg.lower() or
                                   Path(arg).name == 'vpr' for arg in args))
                    if related or not any(args):
                        raise RuntimeError(f'Cannot inspect relevant/unknown same-user process {process.name}; stop it or fix /proc permissions')
                    active_paths.unreadable_unrelated_processes += 1
            except FileNotFoundError:
                pass
        except (OSError, RuntimeError) as error:
            if isinstance(error, RuntimeError):
                raise
    return result


active_paths.unreadable_unrelated_processes = 0


def input_dependencies(trial, folder):
    """Conservatively retain every command/env path, not only --net_file."""
    paths = set()
    def visit(value):
        if isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
        elif isinstance(value, str):
            tokens = [value]
            try:
                tokens.extend(shlex.split(value))
            except ValueError:
                pass
            for token in tokens:
                if token.startswith('-') or not token:
                    continue
                candidate = Path(token)
                if not candidate.is_absolute():
                    candidate = folder/candidate
                try:
                    paths.add(candidate.resolve())
                except (OSError, ValueError):
                    pass
    visit(trial.get('identity', {}))
    visit(trial.get('command', []))
    return paths


def _busy(path, active):
    # An idle shell with cwd=repository/cohort must not freeze every payload.
    # Only route-scoped cwd ancestors count; file membership catches readers.
    return path in active or any(parent in active and _route_scope(parent)
                                  for parent in path.parents)


def _route_scope(folder):
    parts = folder.parts
    if any(part in PROTECTED for part in parts):
        return False
    # Frozen inputs and shared caches are not route working directories.
    return ('feedback' in parts and 'runs' in parts) or ('evaluation' in parts and 'runs' in parts) or any(
        part in parts for part in ('teachers', 'screen', 'timing'))


def inspect_trial(folder, *, repo=None, active=None, tracked=None, dependencies=None):
    folder = Path(folder).resolve()
    empty = dict(eligible=False, reason='', entries=[], skipped={})
    if not _route_scope(folder):
        return dict(empty, reason='outside_route_scope')
    trial_path, metrics_path = folder/'trial.json', folder/'metrics.json'
    if any(not p.is_file() or p.is_symlink() for p in (trial_path, metrics_path)):
        return dict(empty, reason='missing_completed_evidence')
    try:
        trial, metrics = load(trial_path), load(metrics_path)
    except (OSError, ValueError):
        return dict(empty, reason='invalid_completed_evidence')
    if (trial.get('status') not in TERMINAL or metrics.get('status') != trial.get('status') or
            ('identity' in metrics and metrics['identity'] != trial.get('identity')) or
            trial.get('identity', {}).get('stage') != 'route'):
        return dict(empty, reason='incomplete_or_nonroute_evidence')
    if repo is not None:
        runs = Path(repo).resolve()/'experiments/vtr/tailcompile_v68/runs'
        if not folder.is_relative_to(runs) or len(folder.relative_to(runs).parts) < 3:
            return dict(empty, reason='outside_v68_cohort')
    payloads = [path for path in folder.iterdir() if raw_kind(path.name)]
    evidence = dict(trial_sha256=digest(trial_path), metrics_sha256=digest(metrics_path))
    if not payloads:
        # A resume of thousands of already-pruned trials should not perform
        # thousands of needless Git and whole-/proc inspections.
        return dict(eligible=True, reason='completed_route', entries=[], skipped={}, **evidence)
    active = (prepare_paths(active_paths(folder), already_resolved=True) if active is None
              else prepare_paths(active))
    tracked = (prepare_paths(tracked_files(repo, folder=folder), already_resolved=True)
               if tracked is None else prepare_paths(tracked))
    dependencies = prepare_dependencies(dependencies)
    local_dependencies = input_dependencies(trial, folder)
    entries, skipped, retained = [], Counter(), []
    for path in payloads:
        kind = raw_kind(path.name)
        if not kind:
            continue
        if path.is_symlink():
            skipped['symlink'] += 1
            continue
        resolved = path.resolve()
        info = snapshot(path)
        lower = path.name.lower()
        reason = None
        if (lower in {'input.net', 'input.net.post_routing'} or 'locked-input' in lower or
                lower.startswith(('gp.', 'structure.')) or resolved in dependencies or
                resolved in local_dependencies):
            reason = 'input_dependency'
        elif resolved in tracked:
            reason = 'git_tracked'
        elif _busy(resolved, active):
            reason = 'active'
        elif not info['regular'] or info['links'] != 1:
            reason = 'not_regular_or_hardlinked'
        elif resolved.parent != folder:
            reason = 'outside_trial'
        if reason:
            skipped[reason] += 1
            retained.append(dict(path=str(path), kind=kind, reason=reason, stat=info))
        else:
            entries.append(dict(path=str(resolved), folder=str(folder), kind=kind,
                                stat=info, **evidence))
    return dict(eligible=True, reason='completed_route', entries=entries,
                retained=retained, skipped=dict(skipped), **evidence)


def trial_folders(root):
    for base, dirs, names in os.walk(root, followlinks=False):
        folder = Path(base)
        dirs[:] = [name for name in dirs if name not in PROTECTED and not (folder/name).is_symlink()]
        # Missing evidence remains a deletion blocker, but must be visible in
        # the diagnostic report rather than silently omitted from the plan.
        if _route_scope(folder) and ('trial.json' in names or 'metrics.json' in names or
                any(raw_kind(name) for name in names)):
            yield folder


def referenced_paths(root, *, progress=None):
    """Include interrupted trials: a later resume may still need their inputs."""
    result = set()
    count = 0
    for base, dirs, names in os.walk(root, followlinks=False):
        folder = Path(base)
        dirs[:] = [name for name in dirs if name not in PROTECTED and not (folder/name).is_symlink()]
        for name in ('trial.json', 'started.json'):
            if name not in names:
                continue
            path = folder/name
            if path.is_symlink():
                continue
            notify(progress, 'dependencies', processed_records=count, current_record=str(path))
            try:
                result.update(input_dependencies(load(path), folder))
                count += 1
            except (OSError, ValueError):
                # A malformed dependency record is not a license to delete.
                raise ValueError(f'Invalid dependency record, inspect before cleanup: {path}')
    notify(progress, 'dependencies', processed_records=count, complete=True)
    return result


def _validate_root(root, repo=None):
    root = Path(root).resolve()
    if root in (Path(root.anchor), Path.home().resolve()) or not root.is_dir():
        raise ValueError('Choose an existing, explicit cohort output directory, not home or filesystem root')
    if repo is not None:
        repo = Path(repo).resolve()
        runs = repo/'experiments/vtr/tailcompile_v68/runs'
        if root.parent != runs.resolve():
            raise ValueError('Choose exactly one v6.8 cohort: repo/experiments/vtr/tailcompile_v68/runs/<cohort>')
    return root


def plan_cleanup(root, *, repo=None, min_age_hours=0, active=None, tracked=None, progress=None):
    root = _validate_root(root, repo)
    if min_age_hours < 0:
        raise ValueError('min_age_hours must be nonnegative')
    notify(progress, 'active_paths')
    active = (prepare_paths(active_paths(root), already_resolved=True) if active is None
              else prepare_paths(active))
    notify(progress, 'tracked_files')
    tracked = (prepare_paths(tracked_files(repo, progress=progress), already_resolved=True)
               if tracked is None else prepare_paths(tracked))
    notify(progress, 'enumerate_folders', discovered_folders=0)
    folders = []
    for folder in trial_folders(root):
        folders.append(folder)
        notify(progress, 'enumerate_folders', discovered_folders=len(folders), current_folder=str(folder))
    notify(progress, 'dependencies')
    dependencies = prepare_dependencies(referenced_paths(root, progress=progress))
    entries, skipped, kinds = [], Counter(), defaultdict(lambda: dict(files=0, allocated_bytes=0, logical_bytes=0))
    retained_groups, retained_seen = {}, set()

    def record_retained(path, kind, reason, info):
        key = reason+'/'+kind
        group = retained_groups.setdefault(key, dict(files=0, logical_bytes=0,
            allocated_bytes=0, examples=[]))
        group['files'] += 1
        if info['regular']:
            group['logical_bytes'] += info['size']
            inode = info['device'], info['inode']
            if inode not in retained_seen:
                group['allocated_bytes'] += info['allocated_bytes']
                retained_seen.add(inode)
        if len(group['examples']) < 3:
            group['examples'].append(str(path))

    notify(progress, 'inspect_folders', completed_folders=0, total_folders=len(folders))
    for index, folder in enumerate(folders, 1):
        inspection = inspect_trial(folder, repo=repo, active=active, tracked=tracked, dependencies=dependencies)
        skipped.update(inspection['skipped'])
        if not inspection['eligible']:
            skipped[inspection['reason']] += 1
            for path in folder.iterdir():
                kind = raw_kind(path.name)
                if kind:
                    record_retained(path, kind, inspection['reason'], snapshot(path))
        for retained in inspection.get('retained', []):
            record_retained(retained['path'], retained['kind'], retained['reason'], retained['stat'])
        for entry in inspection['entries']:
            if time.time_ns()-entry['stat']['mtime_ns'] < min_age_hours*3600*1e9:
                skipped['too_recent'] += 1
                record_retained(entry['path'], entry['kind'], 'too_recent', entry['stat'])
                continue
            entries.append(entry)
            group = kinds[entry['kind']]
            group['files'] += 1
            group['allocated_bytes'] += entry['stat']['allocated_bytes']
            group['logical_bytes'] += entry['stat']['size']
        notify(progress, 'inspect_folders', completed_folders=index,
               total_folders=len(folders), current_folder=str(folder), candidate_files=len(entries))
    allocated = sum(e['stat']['allocated_bytes'] for e in entries)
    policy_path = root/'storage_policy.json'
    try:
        observed_policy = load(policy_path) if policy_path.is_file() else None
    except (OSError, ValueError) as error:
        observed_policy = dict(read_error=str(error))
    return dict(schema=SCHEMA, root=str(root), repo=str(Path(repo).resolve()) if repo else None,
                created_unix=time.time(), tool_sha256=digest(__file__), min_age_hours=min_age_hours,
                candidate_files=len(entries), allocated_bytes=allocated, allocated_GiB=allocated/1024**3,
                summaries=dict(kinds), skipped=dict(skipped), entries=entries,
                inspected_folders=len(folders), observed_storage_policy=observed_policy,
                retained_summaries={key:dict(value, allocated_GiB=value['allocated_bytes']/1024**3)
                    for key,value in sorted(retained_groups.items(), key=lambda row:-row[1]['allocated_bytes'])},
                retention_scope='Route directories only; protected inputs, models, generation, shared caches and historical attempts are excluded. Retained allocated bytes count each regular inode once.',
                unreadable_unrelated_processes=getattr(active_paths, 'unreadable_unrelated_processes', 0),
                deletion='Permanent listed raw payload deletion; regeneration requires VPR rerun. Inputs and compact evidence retained.')


def apply_cleanup(plan, *, repo=None, active=None, tracked=None, journal=None, progress=None):
    if plan.get('schema') != SCHEMA or plan.get('tool_sha256') != digest(__file__):
        raise ValueError('Cleanup schema/tool changed; create a new reviewed plan')
    expected_repo = str(Path(repo).resolve()) if repo else None
    if plan.get('repo') != expected_repo:
        raise ValueError('Cleanup repository changed')
    root = _validate_root(plan['root'], repo)
    dynamic_active = active is None
    notify(progress, 'active_paths')
    active = (prepare_paths(active_paths(root), already_resolved=True) if dynamic_active
              else prepare_paths(active))
    checked_active = time.monotonic()
    notify(progress, 'tracked_files')
    tracked = (prepare_paths(tracked_files(repo, progress=progress), already_resolved=True)
               if tracked is None else prepare_paths(tracked))
    notify(progress, 'dependencies')
    dependencies = prepare_dependencies(referenced_paths(root, progress=progress))
    deleted, skipped, markers = [], [], defaultdict(list)
    cached_folder, current_entries = None, {}
    notify(progress, 'validate_entries', processed_entries=0, total_entries=len(plan['entries']))
    for index, entry in enumerate(plan['entries'], 1):
        notify(progress, 'validate_entries', processed_entries=index-1, current_entry=index,
               total_entries=len(plan['entries']), deleted_files=len(deleted), current_file=entry['path'])
        path, folder = Path(entry['path']), Path(entry['folder'])
        if (not path.is_absolute() or not folder.is_relative_to(root) or path.parent != folder or
                path.resolve() != path or path.is_symlink() or not path.is_file()):
            skipped.append(dict(path=str(path), reason='missing_or_outside_scope'))
            continue
        if dynamic_active and time.monotonic()-checked_active >= 5:
            active = prepare_paths(active_paths(root), already_resolved=True)
            checked_active = time.monotonic()
        if folder != cached_folder:
            inspection = inspect_trial(folder, repo=repo, active=active, tracked=tracked, dependencies=dependencies)
            cached_folder = folder
            current_entries = {row['path']:row for row in inspection['entries']}
        current = current_entries.get(str(path))
        if current != entry:
            skipped.append(dict(path=str(path), reason='payload_or_evidence_changed_or_protected'))
            continue
        # Cache directory enumeration/static parsing, NEVER the final mutation
        # guards. A sibling can change payload/evidence after an earlier unlink.
        evidence_paths = (folder/'trial.json', folder/'metrics.json')
        try:
            valid = (path not in tracked and not _busy(path, active) and path not in dependencies
                     and snapshot(path) == entry['stat']
                     and all(p.is_file() and not p.is_symlink() for p in evidence_paths)
                     and digest(evidence_paths[0]) == entry['trial_sha256']
                     and digest(evidence_paths[1]) == entry['metrics_sha256'])
        except (OSError, ValueError):
            valid = False
        if not valid:
            skipped.append(dict(path=str(path), reason='payload_or_evidence_changed_or_protected'))
            continue
        record = dict(path=str(path), kind=entry['kind'], allocated_bytes=entry['stat']['allocated_bytes'], unix_time=time.time())
        if journal:
            # Write intent before unlink. Keep plan/journal in the designated
            # workspace; if it is unwritable, stop before deleting anything.
            with Path(journal).open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(dict(record, event='delete_intent')) + '\n')
                stream.flush()
                os.fsync(stream.fileno())
        try:
            path.unlink()
            deleted.append(record)
            markers[folder].append(record)
        except OSError as error:
            skipped.append(dict(path=str(path), reason=str(error)))
    marker_errors = []
    for folder, records in markers.items():
        path = folder/'payload_pruned.json'
        previous = load(path) if path.exists() else dict(schema='tailcompile-v68-payload-pruned-1', cleanups=[])
        previous['cleanups'].append(dict(deleted=records, unix_time=time.time(), journal=str(journal) if journal else None))
        try:
            save(path, previous)
        except OSError as error:
            marker_errors.append(dict(folder=str(folder), error=str(error)))
    released = sum(row['allocated_bytes'] for row in deleted)
    notify(progress, 'validate_entries', processed_entries=len(plan['entries']),
           total_entries=len(plan['entries']), deleted_files=len(deleted), released_GiB=released/1024**3)
    return dict(deleted_files=len(deleted), released_bytes=released, released_GiB=released/1024**3,
                skipped=skipped, deleted=deleted, marker_errors=marker_errors, root=str(root))


def prune_completed_trial(folder, *, repo=None):
    """Post-metrics auto retention; never remove an unfinished VPR working set."""
    folder = Path(folder).resolve()
    inspection = inspect_trial(folder, repo=repo)
    if not inspection['eligible'] or not inspection['entries']:
        return dict(deleted_files=0, released_bytes=0, reason=inspection['reason'], skipped=inspection['skipped'])
    # The helper uses the exact single-trial snapshot, not a repository scan.
    # Evidence is written first; if the quota cannot fit a compact marker, stop
    # without unlinking so manual external-journal cleanup remains auditable.
    marker = folder/'payload_prune_pending.json'
    prior_pending = load(marker) if marker.exists() else None
    save(marker, dict(schema=SCHEMA, entries=inspection['entries'], created_unix=time.time(),
                      prior_pending=prior_pending))
    deleted, skipped = [], []
    for entry in inspection['entries']:
        path = Path(entry['path'])
        if (path.is_symlink() or path.resolve() != path or not path.is_file() or
                snapshot(path) != entry['stat'] or
                digest(folder/'trial.json') != inspection['trial_sha256'] or
                digest(folder/'metrics.json') != inspection['metrics_sha256']):
            skipped.append(str(path))
            continue
        path.unlink()
        deleted.append(dict(path=str(path), kind=entry['kind'], allocated_bytes=entry['stat']['allocated_bytes']))
    previous = load(folder/'payload_pruned.json') if (folder/'payload_pruned.json').exists() else dict(schema='tailcompile-v68-payload-pruned-1', cleanups=[])
    previous['cleanups'].append(dict(deleted=deleted, unix_time=time.time(), automatic=True,
                                     prior_pending=prior_pending))
    save(folder/'payload_pruned.json', previous)
    marker.unlink(missing_ok=True)
    return dict(deleted_files=len(deleted), released_bytes=sum(e['allocated_bytes'] for e in deleted), skipped=skipped)


def inventory(root, *, repo=None, report=None, top=30, progress=None):
    root = _validate_root(root, repo)
    groups, seen, largest, errors = defaultdict(lambda: dict(files=0, logical_bytes=0, allocated_bytes=0, hardlinked_files=0)), set(), [], []
    symlinks = 0
    count = 0
    notify(progress, 'inventory')
    for base, dirs, names in os.walk(root, followlinks=False):
        folder = Path(base)
        notify(progress, 'inventory', processed_files=count, current_folder=str(folder))
        dirs[:] = [name for name in dirs if not (folder/name).is_symlink() and name != '.git']
        for name in names:
            path = folder/name
            try:
                if path.is_symlink():
                    symlinks += 1
                    continue
                info = snapshot(path)
                if not info['regular']:
                    continue
                count += 1
                key = info['device'], info['inode']
                group = groups[file_type(name)]
                group['files'] += 1
                group['logical_bytes'] += info['size']
                group['hardlinked_files'] += int(info['links'] > 1)
                if key not in seen:
                    group['allocated_bytes'] += info['allocated_bytes']
                    seen.add(key)
                row = (info['size'], str(path))
                if len(largest) < top:
                    heapq.heappush(largest, row)
                elif row > largest[0]:
                    heapq.heapreplace(largest, row)
            except OSError as error:
                errors.append(dict(path=str(path), error=str(error)))
    result = dict(schema='tailcompile-v68-storage-inventory-1', root=str(root), host=socket.gethostname(),
                  created_unix=time.time(), file_types=dict(sorted(groups.items(), key=lambda kv:-kv[1]['allocated_bytes'])),
                  allocated_bytes=sum(g['allocated_bytes'] for g in groups.values()),
                  logical_bytes=sum(g['logical_bytes'] for g in groups.values()),
                  symlink_files_skipped=symlinks, largest_files=[dict(path=p, logical_bytes=n) for n,p in sorted(largest, reverse=True)], errors=errors)
    if report:
        notify(progress, 'write_inventory', report=str(report), processed_files=count)
        save(report, result)
    return result


@contextmanager
def cohort_lock(root):
    if os.name != 'posix':
        raise RuntimeError('Manual cleanup requires Linux/WSL process and flock checks')
    import fcntl
    lock_path = Path(root)/'.lock'
    # O_RDONLY is enough for flock and can succeed at a full project quota when
    # the pipeline's existing lock file is already present.
    descriptor = os.open(lock_path, os.O_RDONLY) if lock_path.exists() else os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('Cohort pipeline still active; stop its controller and workers before cleanup')
        yield
    finally:
        os.close(descriptor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('inventory', 'plan', 'apply'))
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--report', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--min-age-hours', type=float, default=0)
    parser.add_argument('--progress-seconds', type=float, default=30,
                        help='stderr heartbeat interval; stdout remains the final JSON summary')
    args = parser.parse_args()
    if not math.isfinite(args.progress_seconds) or args.progress_seconds <= 0:
        parser.error('--progress-seconds must be finite and positive')
    with ProgressReporter(args.phase, args.progress_seconds) as progress:
        progress('validate_root')
        root = _validate_root(args.out, args.repo)
        if args.phase == 'inventory':
            result = inventory(root, repo=args.repo, report=args.report, progress=progress)
        else:
            if not args.plan:
                parser.error('--plan is required for plan/apply; use a writable workspace logs/storage directory')
            progress('cohort_lock')
            with cohort_lock(root):
                if args.phase == 'plan':
                    if args.plan.exists():
                        raise ValueError('Plan already exists; choose a fresh filename')
                    result = plan_cleanup(root, repo=args.repo, min_age_hours=args.min_age_hours,
                                          progress=progress)
                    progress('write_plan', path=str(args.plan), candidate_files=result['candidate_files'])
                    save(args.plan, result)
                else:
                    progress('read_plan', path=str(args.plan))
                    data = load(args.plan)
                    if Path(data['root']).resolve() != root:
                        raise ValueError('Plan cohort changed')
                    stamp = time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8]
                    journal = args.plan.with_name(args.plan.stem+'.'+stamp+'.journal.jsonl')
                    journal.touch(exist_ok=False)
                    result = apply_cleanup(data, repo=args.repo, journal=journal, progress=progress)
                    receipt = args.plan.with_name(args.plan.stem+'.'+stamp+'.receipt.json')
                    progress('write_receipt', path=str(receipt))
                    save(receipt, result)
                    result.update(receipt=str(receipt), journal=str(journal))
    print(json.dumps({k:v for k,v in result.items() if k not in {'entries', 'deleted'}}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
