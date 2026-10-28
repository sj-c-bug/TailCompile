"""Audit a capacity-only input revision and reuse unaffected development evidence.

All original artifacts are read-only. Runtime hooks return original references;
they never forge a new executed-command signature for an old VPR measurement.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re

import install_hook
from io_utils import COMMIT, freeze, read, sha
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BASE_CAMPAIGN_SHA = '94de021232d83659eda21b3d4e534762696a0720bf386143449c363b33db1d2e'
CORE = ('architectures.py', 'device_graph.py', 'data.py', 'model.py',
        'generation.py', 'supervision.py', 'train.py', 'io_utils.py', 'repository_guard.py')
KINDS = ('io', 'clb', 'dsp_top', 'memory')


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f'capacity recovery path outside its allowed root: {path}')
    return path


def require_hash(path, digest):
    if sha(path) != digest:
        raise ValueError(f'capacity recovery artifact changed: {path}')


def resource_error(text):
    matches = re.findall(r'resource requirements required:\s*(.*?)\s*\(available\s+(.*?)\)', text)
    if not matches:
        raise ValueError('expected an explicit VPR packed-resource capacity failure')
    required, available = matches[-1]
    parse = lambda value: {key: int(count) for key, count in re.findall(r'(\w+):\s*(\d+)', value)}
    required, available = parse(required), parse(available)
    if set(required) != set(KINDS) or set(available) != set(KINDS):
        raise ValueError('incomplete VPR packed-resource capacity counts')
    if not any(required[k] > available[k] for k in KINDS):
        raise ValueError('resource error does not demonstrate insufficient device capacity')
    return required, available


def choose_grid(required, old_grid, available, tile_size=4, maximum_utilization=.9):
    """Smallest nonshrinking aligned square satisfying the declared headroom."""
    if tile_size < 1 or not 0 < maximum_utilization <= 1:
        raise ValueError('invalid capacity margin rule')
    w, h = old_grid
    if available['memory'] != 4*(h-2) or available['dsp_top'] != 4*(h-2):
        raise ValueError('capacity recovery requires the fixed four-hard-column architecture')
    io_roots = 2*(2*w+2*h-8)
    if available['io'] % io_roots:
        raise ValueError('unexpected IO capacity multiplier')
    io_units = available['io']//io_roots
    for size in range(math.ceil(max(w, h)/tile_size)*tile_size, 4097, tile_size):
        capacity = dict(memory=4*(size-2), dsp_top=4*(size-2),
                        clb=2*(size-6)*(size-2), io=2*(4*size-8)*io_units)
        if all(required[k] <= maximum_utilization*capacity[k] for k in KINDS):
            return [size, size], capacity
    raise ValueError('no bounded candidate grid meets the resource margin')


def compatible_implementation(protocol):
    implementation = protocol.get('implementation', {})
    if implementation.get('campaign.py') != BASE_CAMPAIGN_SHA:
        raise ValueError('unregistered previous campaign implementation')
    if set(implementation) != {'campaign.py', *CORE}:
        raise ValueError('unexpected previous implementation dependency set')
    for name in CORE:
        require_hash(HERE/name, implementation[name])
    for path, digest in protocol.get('inherited_implementation', {}).items():
        inside(path, REPO)
        require_hash(path, digest)
    if not protocol.get('inherited_implementation'):
        raise ValueError('previous campaign lacks inherited dependency provenance')


def qualify_backend(previous, backend):
    manifest = read(backend/'backend_manifest.json')
    binary_hash = sha(backend/'vpr/vpr')
    if manifest.get('vtr_commit') != COMMIT or manifest.get('binary_sha256') != binary_hash:
        raise ValueError('backend binary/commit differs from its manifest')
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    if install_hook.verify(backend) != manifest.get('hook_sha256'):
        raise ValueError('backend hook/source differs from its manifest')
    for name in (*install_hook.FILES, *install_hook.HETERO_FILES):
        require_hash(backend/name, manifest.get('source_sha256', {}).get(name))
    protocol = read(previous/'qualification/protocol.json')
    report = read(previous/'qualification/qualification_report.json')
    if protocol.get('commit') != COMMIT or protocol.get('binary_sha256') != binary_hash:
        raise ValueError('qualification belongs to another backend')
    expected = {(c, s) for c in ('aligned', 'shifted', 'fixed_mixed', 'sha') for s in (701, 702)}
    rows = report.get('conditions', [])
    if (not report.get('complete') or not report.get('passed') or report.get('harness_version') != 2 or
            len(rows) != 8 or {(r['case'], r['seed']) for r in rows} != expected or not all(r.get('passed') for r in rows)):
        raise ValueError('full passed qualification evidence required')
    return binary_hash, protocol, report


def certify_record(previous, record, artifacts):
    def pin(path, expected=None):
        path = inside(path, previous)
        digest = sha(path)
        if expected is not None and expected != digest:
            raise ValueError(f'prepared artifact hash mismatch: {path}')
        artifacts[str(path)] = digest
    source_path = previous/'inputs'/record['design']/'source.json'
    pin(source_path)
    source = read(source_path)
    require_hash(source['source'], source['sha256'])
    if source.get('include'):
        require_hash(source['include'], source['include_sha256'])
    pin(record['blif'], record['mapped_blif_sha256'])
    for suffix, digest in record['snapshot_sha256'].items():
        pin(record['structure_prefix']+suffix, digest)
    return source_path


def calibration_evidence(previous, record, config, artifacts):
    """Only completed, normally terminated whole-design calibration is reused."""
    from campaign import select_widths
    folder = previous/'calibration'/record['design']
    boundary_path, trials_path = folder/'boundary.json', folder/'trials.json'
    if not boundary_path.is_file() or not trials_path.is_file():
        return None
    boundary, rows = read(boundary_path), read(trials_path)
    if not rows or any(r.get('status') not in ('success', 'unroutable') for r in rows):
        return None
    recomputed = select_widths(rows, config['width_grid'])
    for field in ('status', 'W_ref', 'W_stress', 'success_rates'):
        # JSON object keys are strings after persistence.
        if identity(boundary.get(field)) != identity(json.loads(json.dumps(recomputed.get(field)))):
            raise ValueError('previous calibration boundary disagrees with its trial table')
    evidence = dict(boundary=str(boundary_path), trials=str(trials_path), timeout=None)
    for path in (boundary_path, trials_path):
        artifacts[str(path)] = sha(path)
    for row in rows:
        if row.get('method') != 'default' or row.get('design') != record['design'] or row.get('seed') not in (501, 502):
            raise ValueError('unexpected previous calibration condition')
        p = inside(row['vpr_manifest'], previous)
        manifest = read(p)
        if manifest.get('status') != row['status'] or manifest.get('returncode') not in (0, 2):
            raise ValueError('previous VPR status and calibration result disagree')
        signature = manifest['signature']
        if signature.get('metadata') != dict(condition_id=row['condition_id'], method='default', seed=row['seed'], checkpoint_sha256=None):
            raise ValueError('previous calibration metadata does not match its condition')
        if evidence['timeout'] not in (None, signature['timeout']):
            raise ValueError('mixed timeout budgets in previous calibration')
        evidence['timeout'] = signature['timeout']
        for path, digest in signature['inputs'].items():
            inside(path, previous)
            require_hash(path, digest)
            artifacts[str(Path(path).resolve())] = digest
        artifacts[str(p)] = sha(p)
    return evidence


def recover(previous, out, backend, *, repo=REPO):
    previous, out, backend = (Path(p).resolve() for p in (previous, out, backend))
    runs = Path(repo).resolve()/'experiments/vtr/tailcompile_v7_3d/runs'
    if runs not in previous.parents or runs not in out.parents or previous == out or previous in out.parents or out in previous.parents:
        raise ValueError('capacity recovery needs separate cohorts in canonical v7 runs')
    validate_output_roots((out,), repo=repo, dependency_roots=(backend,))
    old_config, protocol, prepared = (read(previous/name) for name in ('campaign.json', 'protocol.json', 'prepared.json'))
    if protocol.get('config') != old_config or not prepared.get('complete'):
        raise ValueError('previous preparation/configuration was not complete and frozen')
    compatible_implementation(protocol)
    binary_hash, qualification_protocol, qualification_report = qualify_backend(previous, backend)
    if protocol.get('binary_sha256') != binary_hash or protocol.get('backend_manifest_sha256') != sha(backend/'backend_manifest.json'):
        raise ValueError('previous campaign backend provenance changed')
    originals = {r['design']: r for r in old_config['designs']}
    records = {r['design']: r for r in prepared['records']}
    if set(records) != set(originals) or 'mcml' not in records:
        raise ValueError('previous prepared design panel is incomplete')
    artifacts = {str(previous/name): sha(previous/name) for name in ('campaign.json', 'protocol.json', 'prepared.json')}
    for name, record in records.items():
        if any(record.get(k) != v for k, v in originals[name].items()):
            raise ValueError('prepared record differs from frozen configuration')
        certify_record(previous, record, artifacts)

    demands, available, failure_evidence = [], None, []
    manifests = sorted((previous/'calibration/mcml').glob('*/w32/s*/vpr/manifest.json'))
    if len(manifests) != 8:
        raise ValueError('capacity correction requires all eight MCML w32 conditions')
    for path in manifests:
        row = read(path)
        if row.get('status') != 'packing_or_placement_failure' or row.get('returncode') != 1:
            raise ValueError('unexpected MCML outcome; cannot infer a capacity-only correction')
        log = inside(row['log'], previous)
        required, actual = resource_error(log.read_text(errors='replace'))
        if available is not None and available != actual:
            raise ValueError('four H/Z cells disagree on actual total resources')
        available = actual
        demands.append(required)
        for p in (path, log):
            artifacts[str(p)] = sha(p)
        failure_evidence.append(dict(manifest=str(path), log=str(log), required=required, available=actual))
    required = {k: max(d[k] for d in demands) for k in KINDS}
    grid, capacity = choose_grid(required, records['mcml']['grid'], available, int(old_config.get('tile_size', 4)))
    config = copy.deepcopy(old_config)
    changed = next(r for r in config['designs'] if r['design'] == 'mcml')
    source = Path(records['mcml']['blif'])
    adapter = previous/'inputs/mcml/mapping_adapter.json'
    mapping_manifest = previous/'inputs/mcml/mapping/manifest.json'
    mapping = read(mapping_manifest) if mapping_manifest.is_file() else None
    if mapping is not None:
        if mapping.get('status') != 'complete':
            raise ValueError('previous MCML frontend mapping did not complete')
        artifacts[str(mapping_manifest)] = sha(mapping_manifest)
    if adapter.is_file():
        a = read(adapter)
        raw = Path(a['source'])
        if raw.is_file() and a.get('derived_sha256') == records['mcml']['mapped_blif_sha256']:
            inside(raw, previous)
            require_hash(raw, a['source_sha256'])
            if mapping is not None and raw.parent.resolve() != (Path(mapping['workdir'])/'flow').resolve():
                raise ValueError('raw MCML BLIF is not linked to its completed mapping attempt')
            source = raw
            artifacts[str(adapter)] = sha(adapter)
    artifacts[str(source)] = sha(source)
    changed.update(grid=grid, mapped_blif=str(source), qualification_status='capacity_corrected_pending_default')
    config['note'] = (str(config.get('note', '')) +
        ' Capacity input revision: MCML mapping completed; packed BRAM demand exceeds the old grid. '
        'The new tile-aligned grid uses a declared maximum packed-resource utilization of 0.90; '
        'new-grid Default calibration remains pending.').strip()

    calibration = {}
    for name, record in records.items():
        if name != 'mcml':
            evidence = calibration_evidence(previous, record, old_config, artifacts)
            if evidence:
                calibration[name] = evidence
    devices = {}
    for path in sorted((previous/'device_cache').glob('*/manifest.json')):
        value = read(path)
        if value.get('binary_sha256') == binary_hash and (path.parent/'rr_graph.xml').is_file() and (path.parent/'device_graph.json').is_file():
            devices[path.parent.name] = dict(folder=str(path.parent), manifest_sha256=sha(path))
    catalog = dict(schema='tailcompile-v7-capacity-recovery-1', previous_out=str(previous),
        binary_sha256=binary_hash, artifacts=artifacts, old_config=old_config,
        old_records=records, unchanged_designs=sorted(set(records)-{'mcml'}), calibration=calibration,
        devices=devices, correction=dict(design='mcml', original_grid=records['mcml']['grid'],
            grid=grid, required=required, old_available=available, declared_capacity=capacity,
            maximum_utilization=.9, tile_alignment=int(old_config.get('tile_size', 4)),
            basis='maximum packed demand across all eight failed Default w32 runs; no learned-method outcomes',
            failure_evidence=failure_evidence), target_config_identity=identity(config))
    targets = [out/name for name in ('capacity_recovery_catalog.json', 'campaign.json',
               'qualification/protocol.json', 'qualification/qualification_report.json')]
    if any(out not in target.resolve().parents for target in targets):
        raise ValueError('capacity recovery target escapes its cohort through a symlink')
    if (out/'protocol.json').exists() and not (out/'capacity_recovery_catalog.json').is_file():
        raise ValueError('target already belongs to another campaign')
    # No writes occur before all original evidence has been certified.
    freeze(out/'capacity_recovery_catalog.json', catalog)
    config['input_recovery'] = dict(catalog=str(out/'capacity_recovery_catalog.json'),
                                   sha256=sha(out/'capacity_recovery_catalog.json'))
    freeze(out/'campaign.json', config)
    freeze(out/'qualification/protocol.json', qualification_protocol)
    freeze(out/'qualification/qualification_report.json', qualification_report)
    return catalog


def _catalog(campaign):
    if not campaign.config.get('input_recovery'):
        return None
    if hasattr(campaign, '_capacity_recovery_catalog'):
        return campaign._capacity_recovery_catalog
    pointer = campaign.config['input_recovery']
    path = inside(pointer['catalog'], campaign.out)
    require_hash(path, pointer['sha256'])
    catalog = read(path)
    config = copy.deepcopy(campaign.config)
    config.pop('input_recovery')
    if catalog.get('binary_sha256') != campaign.binary_hash or catalog.get('target_config_identity') != identity(config):
        raise ValueError('capacity recovery catalog does not match this campaign')
    for path, digest in catalog['artifacts'].items():
        inside(path, catalog['previous_out'])
        require_hash(path, digest)
    compatible_implementation(read(Path(catalog['previous_out'])/'protocol.json'))
    campaign._capacity_recovery_catalog = catalog
    return catalog


def reused_prepared(campaign, original):
    catalog = _catalog(campaign)
    if not catalog or original['design'] not in catalog['unchanged_designs']:
        return None
    old_original = next(r for r in catalog['old_config']['designs'] if r['design'] == original['design'])
    if original != old_original:
        raise ValueError('cannot reuse preparation for a changed design')
    previous = Path(catalog['previous_out'])
    source = read(previous/'inputs'/original['design']/'source.json')
    require_hash(source['source'], source['sha256'])
    if source.get('include'):
        require_hash(source['include'], source['include_sha256'])
    folder = campaign.out/'inputs'/original['design']
    inside(folder, campaign.out)
    freeze(folder/'source.json', source)
    return copy.deepcopy(catalog['old_records'][original['design']])


def reused_calibration(campaign, record):
    catalog = _catalog(campaign)
    if not catalog or record['design'] not in catalog['calibration']:
        return None
    evidence = catalog['calibration'][record['design']]
    if evidence['timeout'] != campaign.args.timeout:
        raise ValueError('calibration reuse requires the same timeout budget')
    if record != catalog['old_records'][record['design']]:
        raise ValueError('cannot reuse calibration after prepared inputs changed')
    from campaign import select_widths
    for path in (evidence['boundary'], evidence['trials']):
        require_hash(path, catalog['artifacts'][str(Path(path).resolve())])
    rows = read(evidence['trials'])
    for row in rows:
        architecture = campaign.architecture(record, row['architecture'])
        require_hash(row['vpr_manifest'], catalog['artifacts'][str(Path(row['vpr_manifest']).resolve())])
        manifest = read(row['vpr_manifest'])
        command = manifest['signature']['command']
        old_arch = next((p for p in manifest['signature']['inputs'] if Path(p).name == 'arch.xml'), None)
        if old_arch is None or sha(architecture) != manifest['signature']['inputs'][old_arch]:
            raise ValueError('reused calibration architecture differs from the current XML')
        if (int(command[command.index('--route_chan_width')+1]) != row['width'] or
                int(command[command.index('--seed')+1]) != row['seed']):
            raise ValueError('reused VPR command disagrees with its width/seed')
    boundary = select_widths(rows, campaign.config['width_grid'])
    boundary.update(design=record['design'], family=record['family'], split=record['split'], record=record,
        reused_from=dict(previous_out=catalog['previous_out'], boundary=evidence['boundary'], trials=evidence['trials']))
    return boundary, rows


def reused_device_folder(campaign, key, arch, width):
    catalog = _catalog(campaign)
    entry = catalog['devices'].get(key) if catalog else None
    if not entry:
        return None
    folder = inside(entry['folder'], catalog['previous_out'])
    require_hash(folder/'manifest.json', entry['manifest_sha256'])
    manifest = read(folder/'manifest.json')
    if (manifest['architecture_sha256'] != sha(arch) or manifest['binary_sha256'] != campaign.binary_hash or
            manifest['channel_width'] != int(width)):
        raise ValueError('shared RRG recovery identity mismatch')
    return folder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', type=Path, required=True)
    args = parser.parse_args()
    catalog = recover(args.previous_out, args.out, args.backend)
    print(dict(recovery='ready', out=str(args.out.resolve()), grid=catalog['correction']['grid'],
               memory_capacity=catalog['correction']['declared_capacity']['memory'],
               reused_calibration_designs=sorted(catalog['calibration']), backend_read_only=True), flush=True)


if __name__ == '__main__':
    main()
