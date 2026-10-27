"""Copy an audited, compact v7 dependency/evidence snapshot into the sole repo.

Historical workspace sources are never moved or deleted. Existing destinations
must match the intended content exactly; conflicts abort before any copy.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

from io_utils import write


HELPERS = ('heterogeneity_fixtures.py', 'pilot_architectures.py', 'device_supply_3d.py',
           'prepare_natural_blif.py', 'probe_layer_heterogeneity.py', 'test_device_supply_3d.py',
           'regress_layer_heterogeneity_2d.py', 'test_layer_heterogeneity.py',
           'validate_layer_heterogeneity.py', 'cluster_sa_probe.py')
DOCS = ('EXPERIMENT_PLAN_3D_V68_2026-10-23.md', 'PRE_GNN_EXECUTION_PLAN_2026-10-23.md',
        'EXTERNAL_COMPARISON_PLAN_2026-10-23.md', 'RESULTS_ARCHITECTURES_2026-10-24.md',
        'RESULTS_V7_PREPARATION_2026-10-25.md')


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def compact(value):
    if isinstance(value, dict):
        return {key: compact(item) for key, item in value.items() if key != 'positions'}
    if isinstance(value, list):
        return [compact(item) for item in value]
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--repo', type=Path, required=True)
    a = p.parse_args()
    workspace, repo = a.workspace.resolve(), a.repo.resolve()
    source = workspace/'experiments/vtr3d'
    destination = repo/'experiments/vtr3d'
    report = destination/'migration_v7_20261025.json'
    if report.exists():
        raise ValueError('migration ledger already exists; keep immutable and use a new dated ledger')
    if source == destination or source in destination.parents or destination in source.parents:
        raise ValueError('historical source and canonical target must be disjoint')
    protected = ['experiments/vtr/tailcompile_v68', 'experiments/vtr/tailcompile_v69']
    diff_before = subprocess.check_output(['git', '-C', str(repo), 'diff', '--numstat', '--', *protected], text=True, timeout=30)
    if diff_before.strip():
        raise ValueError('frozen 2D trees have an existing tracked diff; report it before migration')
    required = list(HELPERS) + list(DOCS) + [
        'patches/vtr40f8_layer_heterogeneity_v2.patch', 'patches/vtr40f8_layer_heterogeneity_v2.md']
    for name in required:
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    paths = {source/name for name in required}
    paths.update(path for path in (source/'source_archive').rglob('*') if path.is_file())
    paths.update(path for folder in (source/'runs').glob('v7_*') for path in folder.rglob('*') if path.is_file())
    copies, excluded, payloads = [], [], {}
    for path in sorted(paths):
        relative = path.relative_to(source)
        raw = path.read_bytes()
        record = dict(source=str(path), destination=str(destination/relative),
                      relative=relative.as_posix(), source_bytes=len(raw), source_sha256=digest(raw))
        reason = None
        if path.is_symlink() or '__pycache__' in relative.parts:
            reason = 'symlink or generated Python cache'
        elif relative.parts[0] == 'runs':
            if path.name in ('device_graph.json', 'logic.json', 'architecture.json'):
                reason = 'regenerable device/logic/slot graph payload; summary/provenance retained separately'
            elif path.suffix in ('.gz', '.tsv', '.blif', '.net', '.route', '.place', '.fplace',
                                 '.xml', '.rpt', '.pt', '.pth', '.pkl', '.npz', '.zip', '.tar'):
                reason = 'input, graph, model, raw placement, timing report or derived payload'
            elif path.suffix not in ('.json', '.md', '.csv', '.log', '.out', '.txt', '.py'):
                reason = 'outside compact evidence extension allowlist'
            elif len(raw) > 512*1024:
                reason = 'over 512 KiB; compact audit records failure instead of full raw output'
        if reason:
            excluded.append(dict(record, reason=reason))
            continue
        transformed = False
        if path.suffix == '.json':
            original = json.loads(raw)
            minimal = compact(original)
            if minimal != original:
                raw = (json.dumps(minimal, indent=2, ensure_ascii=False, allow_nan=False)+'\n').encode('utf-8')
                transformed = True
        target = destination/relative
        expected = digest(raw)
        if target.exists() and (not target.is_file() or digest(target.read_bytes()) != expected):
            raise ValueError(f'canonical destination differs; refusing overwrite: {target}')
        copies.append(dict(record, destination_bytes=len(raw), destination_sha256=expected,
                           transformation='remove positions only' if transformed else None,
                           destination_already_identical=target.exists()))
        payloads[target] = raw
    # All collisions are checked before the first write.
    for target, raw in payloads.items():
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
    diff_after = subprocess.check_output(['git', '-C', str(repo), 'diff', '--numstat', '--', *protected], text=True, timeout=30)
    if diff_before != diff_after:
        raise ValueError('frozen 2D tracked diff changed during migration')
    result = dict(schema='tailcompile-v7-canonical-migration-1', utc=datetime.now(timezone.utc).isoformat(),
                  workflow='REPOSITORY_WORKFLOW.md', source=str(source), destination=str(destination),
                  source_files_removed=False, git_commit_or_push_performed=False, vtr_modified=False,
                  protected_2d_paths=protected, protected_2d_tracked_diff_before=diff_before,
                  protected_2d_tracked_diff_after=diff_after,
                  selected_file_count=len(copies), selected_bytes=sum(row['destination_bytes'] for row in copies),
                  excluded_file_count=len(excluded), excluded_bytes=sum(row['source_bytes'] for row in excluded),
                  exclusion_reasons=dict(Counter(row['reason'] for row in excluded)), files=copies, excluded=excluded,
                  historical_markdown_links='Legacy workspace paths in archived historical evidence retain original provenance; active v7 runtime uses canonical helpers.',
                  server_input_policy='No BLIF qualification package in Git; prepare_server.sh remaps source from fixed VTR commit.')
    write(report, result)
    print({key: result[key] for key in ('selected_file_count', 'selected_bytes', 'excluded_file_count',
                                       'excluded_bytes', 'protected_2d_tracked_diff_after')})


if __name__ == '__main__':
    main()
