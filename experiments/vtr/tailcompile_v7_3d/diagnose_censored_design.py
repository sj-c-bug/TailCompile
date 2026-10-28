"""Read existing calibration evidence; never launch VPR or edit the source cohort.

The report is diagnostic, not a routability acceptance or a causal attribution.
Only the explicitly named JSON report in a separate v7 cohort is written.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS_REL = Path('experiments/vtr/tailcompile_v7_3d/runs')
SCHEMA = 'tailcompile-v7-censored-diagnosis-1'
ARCHITECTURES = ('H0Z0', 'H0Z1', 'H1Z0', 'H1Z1')
CALIBRATION_SEEDS = (501, 502)
NORMAL = {'success', 'unroutable'}
LOG_TAIL_BYTES = 256 * 1024
# These are reported as observed messages, never silently promoted to causes.
PATTERNS = {
    'missing_dependency': r'error while loading shared libraries|cannot open shared object file|command not found',
    'memory_failure': r'std::bad_alloc|cannot allocate memory|out of memory|MemoryError',
    'resource_capacity': r'Failed to find device which satisfies resource requirements|Packing failed to fit',
    'assertion_or_crash': r'Assertion .*failed|assertion failure|Segmentation fault|Aborted \(core dumped\)',
    'route_disconnection': r'no path.*(?:sink|source)|(?:sink|source).*not reachable|unreachable.*(?:sink|source)',
    'routing_failure': r'circuit is unroutable|failed to route|routing failed|circuit could not be routed',
    'routing_limit': r'max(?:imum)?.*routing iterations|routing.*iteration limit|exceeded.*iterations',
    'congestion': r'overused|overuse|overfull|congested',
    'timeout': r'timed out|timeout expired|time limit exceeded',
}


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _within(path, root):
    return path == root or root in path.parents


def _runs_root(repo):
    repo = Path(repo).resolve()
    runs = (repo / RUNS_REL).resolve()
    if not _within(runs, repo):
        raise ValueError('v7 runs root escapes repository through a symlink')
    return runs


def _recorded_path(value, runs):
    """Relocate synced absolute server paths by the versioned runs suffix only."""
    if not value:
        return None
    original = Path(value)
    if original.is_absolute() and _within(original.resolve(), runs):
        return original.resolve()
    normalized = str(value).replace('\\', '/')
    marker = RUNS_REL.as_posix() + '/'
    if marker not in normalized:
        return None
    relative = normalized.split(marker, 1)[1]
    resolved = (runs / relative).resolve()
    if not _within(resolved, runs):
        raise ValueError(f'recorded input escapes v7 runs: {value}')
    return resolved


def _log_evidence(path):
    if path is None or not path.is_file():
        return dict(available=False, path=str(path) if path else None,
                    observations=[], cause_established=False)
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - LOG_TAIL_BYTES))
        payload = stream.read(LOG_TAIL_BYTES)
    lines = payload.decode('utf-8', errors='replace').splitlines()
    if size > LOG_TAIL_BYTES:
        lines = lines[1:]  # First byte can be in the middle of a UTF-8 line.
    observations = []
    for index, line in enumerate(lines):
        tags = [tag for tag, pattern in PATTERNS.items() if re.search(pattern, line, re.I)]
        if tags:
            observations.append(dict(tail_line=index + 1, tags=tags, text=line[:1200]))
    return dict(available=True, path=str(path), total_bytes=size, tail_bytes=len(payload),
                tail_sha256=hashlib.sha256(payload).hexdigest(),
                tail_truncated=size > LOG_TAIL_BYTES,
                observations=observations[-40:], cause_established=False)


def _category(status, code, log):
    tags = {tag for row in log['observations'] for tag in row['tags']}
    if status in ('timeout', 'interrupted', 'out_of_memory', 'oom', 'infrastructure_failure') or code in (127, 'timeout', 'interrupted'):
        return 'infrastructure_or_interruption'
    if tags & {'missing_dependency', 'memory_failure'}:
        return 'infrastructure_or_interruption'
    if status in NORMAL:
        return 'normal_pnr_outcome'
    if 'resource_capacity' in tags:
        return 'resource_capacity_failure'
    return 'other_execution_failure_needs_review'


def diagnose(source, *, design='or1200', repo=None):
    """Return bounded evidence without writes, subprocesses, or payload loading."""
    if not re.fullmatch(r'[A-Za-z0-9_-]+', design):
        raise ValueError('design must be a simple identifier')
    runs = _runs_root(repo or REPO)
    source = Path(source).resolve()
    if (not _within(source, runs) or source == runs or not source.is_dir()
            or len(source.relative_to(runs).parts) != 1):
        raise ValueError('source must be an existing cohort under v7 runs')
    paths = dict(trials=source/'calibration'/design/'trials.json',
                 boundary=source/'calibration'/design/'boundary.json')
    boundary = _read(paths['boundary'])
    rows = _read(paths['trials'])
    if not isinstance(rows, list) or not rows:
        raise ValueError('calibration trial list is empty or invalid')
    grid = boundary['record']['grid']
    audit_path = source/'architectures'/f'{grid[0]}x{grid[1]}'/'actual_audit.json'
    audit = _read(audit_path) if audit_path.is_file() else None
    if audit is not None:
        paths['rrg_audit'] = audit_path
    input_hashes = {str(p): _sha(p) for p in paths.values()}
    protected = {source}
    cells = defaultdict(list)
    details = []
    seen = set()
    issues = []
    for row in rows:
        key = (row['architecture'], int(row['width']), int(row['seed']))
        if key in seen:
            raise ValueError(f'duplicate calibration condition: {key}')
        seen.add(key)
        if row.get('design') != design or row.get('method') != 'default':
            raise ValueError('diagnosis accepts only the requested design Default trials')
        manifest_path = _recorded_path(row.get('vpr_manifest'), runs)
        manifest = None
        if manifest_path is not None:
            protected.add(runs / manifest_path.relative_to(runs).parts[0])
            if manifest_path.is_file():
                manifest = _read(manifest_path)
                input_hashes[str(manifest_path)] = _sha(manifest_path)
        code = manifest.get('returncode') if manifest else None
        if manifest and manifest.get('status') != row['status']:
            issues.append(dict(kind='status_mismatch', condition=key,
                               trial_status=row['status'], manifest_status=manifest.get('status')))
        if manifest and row['status'] in NORMAL and code != {'success': 0, 'unroutable': 2}[row['status']]:
            issues.append(dict(kind='returncode_mismatch', condition=key,
                               trial_status=row['status'], returncode=code))
        log_path = _recorded_path(manifest.get('log'), runs) if manifest else None
        # Some synced manifests retain an old mount name; the attempt suffix is
        # resolved above. Never guess a different attempt after a failed lookup.
        log = _log_evidence(log_path)
        if log_path is not None:
            protected.add(runs / log_path.relative_to(runs).parts[0])
        phases = row.get('phases', {})
        item = dict(architecture=key[0], width=key[1], seed=key[2], status=row['status'],
                    returncode=code, category=_category(row['status'], code, log),
                    routing_seconds=phases.get('seconds', {}).get('routing_seconds'),
                    routing_iterations=phases.get('counters', {}).get('routing_iteration_rows_observed'),
                    vpr_manifest=str(manifest_path) if manifest_path else None,
                    manifest_available=manifest is not None, log=log)
        details.append(item)
        cells[(key[0], key[2])].append(item)
    widths = sorted({r['width'] for r in details})
    by_width = []
    for width in widths:
        panel = [r for r in details if r['width'] == width]
        expected = {(a, width, s) for a in ARCHITECTURES for s in CALIBRATION_SEEDS}
        present = {(r['architecture'], width, r['seed']) for r in panel}
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing or extra:
            issues.append(dict(kind='incomplete_or_unexpected_panel', width=width,
                               missing=missing, unexpected=extra))
        successes = sum(r['status'] == 'success' for r in panel)
        by_width.append(dict(width=width, total=len(panel), success=successes,
                             success_rate=successes / len(panel),
                             statuses=dict(Counter(r['status'] for r in panel)),
                             complete_expected_panel=not missing and not extra))
    nonmonotonic = []
    cell_summary = []
    for (architecture, seed), panel in sorted(cells.items()):
        panel.sort(key=lambda r: r['width'])
        lower_success = None
        for row in panel:
            if row['status'] == 'success' and lower_success is None:
                lower_success = row['width']
            elif row['status'] == 'unroutable' and lower_success is not None:
                nonmonotonic.append(dict(architecture=architecture, seed=seed,
                                        lower_success_width=lower_success,
                                        higher_failure_width=row['width']))
        cell_summary.append(dict(architecture=architecture, seed=seed,
                                 observed_min_success_width=lower_success,
                                 statuses_by_width={str(r['width']): r['status'] for r in panel}))
    z = {arch: (audit or {}).get('width_coupling', {}).get(arch, {}).get('z_capacity_invariant')
         for arch in ARCHITECTURES}
    return dict(schema=SCHEMA, source=str(source), design=design, scope='diagnosis_only',
                no_vpr_execution=True, source_read_only=True,
                status='needs_evidence_review' if issues else 'diagnostic_complete',
                boundary_status=boundary.get('status'), W_ref=boundary.get('W_ref'),
                W_stress=boundary.get('W_stress'), trial_count=len(details),
                statuses=dict(Counter(r['status'] for r in details)),
                categories=dict(Counter(r['category'] for r in details)),
                by_width=by_width, cells=cell_summary, nonmonotonic_events=nonmonotonic,
                log_coverage=dict(available=sum(r['log']['available'] for r in details), total=len(details)),
                rrg=dict(available=audit is not None,
                         four_cell_matches=(audit or {}).get('four_cell_matches'),
                         z_capacity_invariant=z,
                         limitation='Matched supply does not establish equal pin access or detailed routability.'),
                issues=issues, trials=sorted(details, key=lambda r: (r['width'], r['architecture'], r['seed'])),
                input_sha256=input_hashes, protected_source_cohorts=sorted(str(p) for p in protected),
                conclusion=dict(cause_established=False,
                                interpretation='Default outcomes and log excerpts only; no learned-method efficacy conclusion.',
                                width_interpretation='Nonmonotonic P&R outcomes are not a fixed-placement routing monotonicity test.',
                                next_evidence='Inspect reported failure excerpts and per-cell routing counters; absent logs leave the cause unresolved.',
                                learning_blocked=False))


def write_report(report, output, *, repo=None):
    runs = _runs_root(repo or REPO)
    output = Path(output).resolve()
    if not _within(output, runs) or len(output.relative_to(runs).parts) < 2 or output.suffix != '.json':
        raise ValueError('output must be an explicit JSON report inside a separate v7 runs cohort')
    for protected in report['protected_source_cohorts']:
        if _within(output, Path(protected).resolve()):
            raise ValueError('output overlaps a read-only source cohort')
    if output.exists():
        old = _read(output)
        if old.get('schema') != SCHEMA or old.get('source') != report['source'] or old.get('design') != report['design']:
            raise ValueError('refusing to replace an unrelated existing artifact')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f'.{os.getpid()}.tmp')
    created = False
    try:
        with temporary.open('x', encoding='utf-8') as stream:
            created = True
            json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write('\n')
        temporary.replace(output)
    finally:
        if created and temporary.exists():
            temporary.unlink()
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--design', default='or1200')
    args = parser.parse_args()
    report = diagnose(args.source, design=args.design)
    output = write_report(report, args.output)
    print(json.dumps(dict(report=str(output), status=report['status'], trial_count=report['trial_count'],
                          statuses=report['statuses'], log_coverage=report['log_coverage'],
                          nonmonotonic_events=len(report['nonmonotonic_events']),
                          cause_established=False, learning_blocked=False)))


if __name__ == '__main__':
    main()
