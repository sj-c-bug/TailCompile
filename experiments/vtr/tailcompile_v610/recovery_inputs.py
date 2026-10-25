"""Fresh-cohort input recovery; imported sources and prior runs stay read-only.

The pre-GP device estimate is not a packing certificate. A changed device needs
a fresh AP-structure export, but never a second synthesis of an unchanged BLIF.
Device qualification and routing are handled by the recovery orchestrator.
"""
from __future__ import annotations

from copy import copy
import math
from pathlib import Path
import re
import shutil
import sys

from bridge import e, emit, freeze, read, sha, StoragePressureError, log_storage_failure, storage_error
from data import read_ap_snapshot
from inventory import SAFE_NAME
from prepare import PreparationFailure, STRUCTURE_SUFFIXES

SOURCE_KEYS = ('design', 'source', 'source_sha256', 'include_sha256')
STALE_FIELDS = {
    'graph', 'logic', 'device', 'supply', 'audit', 'width', 'channel_width', 'rr_graph',
    'widths', 'reference_width', 'reference_is_success', 'calibration',
    'calibration_status', 'calibration_observations', 'boundary_selection',
}


def _clean_record(record):
    if not SAFE_NAME.fullmatch(record['design']):
        raise ValueError('Unsafe benchmark name')
    return {key:value for key,value in record.items() if key not in STALE_FIELDS}


def _budget(a, name):
    value = float(getattr(a, name))
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    worker = copy(a)
    worker.timeout = value
    return worker


def _source_signature(a, record):
    bench = (Path(a.vtr)/'vtr_flow/benchmarks/verilog').resolve()
    source = (bench/record['source']).resolve()
    source.relative_to(bench)
    if not source.is_file() or sha(source) != record['source_sha256']:
        raise ValueError(f'Canonical source missing or changed: {source}')
    include = source.parent/'hard_block_include.v' if record['source'].startswith('koios/') else None
    include_hash = sha(include) if include else None
    if include_hash != record.get('include_sha256'):
        raise ValueError(f'Canonical architecture include changed: {include}')
    return source, include, {key:record.get(key) for key in SOURCE_KEYS}


def _check_mapped(record):
    path = Path(record['mapped_blif'])
    if not path.is_file() or sha(path) != record['mapped_blif_sha256']:
        raise ValueError(f'Canonical mapped BLIF missing or changed: {path}')


def _fail_result(row, folder, stage, message):
    log = Path(folder)/'vpr.out'
    text = log.read_text(encoding='utf-8', errors='replace') if log.is_file() else ''
    if log_storage_failure(text, row.get('returncode', 1)):
        raise StoragePressureError(f'Storage failure during {stage}: {folder}')
    status = row.get('status')
    if status not in {'timeout', 'killed'}:
        status = 'preparation_failure'
    error=PreparationFailure(status, f'{message}; see {folder}/vpr.out', folder)
    error.device_capacity_failure=is_device_capacity_failure(row,folder)
    raise error


def _call(a, command, folder, signature, **kwargs):
    try:
        return e.call(a, command, folder, signature, **kwargs)
    except Exception as error:
        if storage_error(error) or isinstance(error, PreparationFailure):
            raise
        # Interrupted/active trials and identity conflicts must remain visible.
        raise PreparationFailure('preparation_failure', str(error), folder) from error


def _canonical_mapping(a, record):
    record = _clean_record(record)
    source, include, source_signature = _source_signature(a, record)
    root = Path(a.out)/'inputs'/record['design']
    blif = root/(record['design']+'.pre-vpr.blif')
    mapping = root/'mapping.json'
    signature = dict(source_signature, backend_identity=a.identity)
    freeze(root/'source.json', source_signature)
    if mapping.is_file():
        saved = read(mapping)
        if saved['signature'] != signature:
            raise ValueError(f'Mapping provenance changed: {mapping}')
        value = dict(record, mapped_blif=str(blif), mapped_blif_sha256=saved['mapped_blif_sha256'])
        _check_mapped(value)
        return value
    worker = _budget(a, 'synthesis_timeout')
    flow, folder = root/'mapping', root/'synthesis'
    command = [sys.executable, Path(a.vtr)/'vtr_flow/scripts/run_vtr_flow.py', source, a.arch,
               '-temp_dir', flow, '-ending_stage', 'abc', '-timeout', str(worker.timeout)]
    if include is not None:
        command += ['-include', include]
    result = _call(worker, command, folder,
                   dict(signature, stage='synthesis', budget_seconds=worker.timeout))
    generated = flow/(record['design']+'.abc.blif')
    if result['returncode'] != 0 or not generated.is_file():
        _fail_result(result, folder, 'synthesis', 'Synthesis did not produce a canonical BLIF')
    blif.parent.mkdir(parents=True, exist_ok=True)
    temporary = blif.with_suffix(blif.suffix+'.tmp')
    shutil.copyfile(generated, temporary)
    temporary.replace(blif)
    digest = sha(blif)
    freeze(mapping, dict(schema='tailcompile-v610-recovery-mapping-1', signature=signature,
        mapped_blif_sha256=digest, synthesis_budget_seconds=worker.timeout,
        synthesis_trial_sha256=sha(folder/'trial.json')))
    emit('recovery_mapping_ready', design=record['design'], mapped_blif=str(blif))
    return dict(record, mapped_blif=str(blif), mapped_blif_sha256=digest)


def _structure(a, record, device_width=None):
    record = _clean_record(record)
    _check_mapped(record)
    name = 'auto' if device_width is None else f'w{device_width}'
    root = Path(a.out)/'inputs'/record['design']/'structures'/name
    prefix, folder, manifest = root/'structure', root/'run', root/'prepared.json'
    signature = dict(backend_identity=a.identity, design=record['design'],
        source_sha256=record['source_sha256'], include_sha256=record.get('include_sha256'),
        mapped_blif_sha256=record['mapped_blif_sha256'], requested_device_width=device_width,
        channel_width=a.width_start, seed=a.calibration_seeds[0])
    if manifest.is_file():
        saved = read(manifest)
        if saved['signature'] != signature:
            raise ValueError(f'Structure attempt provenance changed: {manifest}')
        result = dict(saved['record'])
        _check_structure(result)
        return dict(result, **{key:value for key,value in record.items()
                              if key not in {'structure_prefix','structure_sha256','device_width','device_height',
                                             'input_origin','requested_device_width'}})
    worker = _budget(a, 'calibration_timeout')
    command = e.vpr_command(worker, record['mapped_blif'], a.width_start,
                            a.calibration_seeds[0], device_width)
    result = _call(worker, command, folder, dict(signature, stage='structure', budget_seconds=worker.timeout),
        env={'TC_GP_STRUCTURE':prefix, 'TC_GP_STRUCTURE_ONLY':1},
        marker='TAILCOMPILE_V68_STRUCTURE_ONLY_COMPLETE')
    if result['status'] != 'complete':
        _fail_result(result, folder, 'structure', 'Pre-GP structure export did not complete')
    try:
        snapshot = read_ap_snapshot(prefix, structure=True)
    except (OSError, ValueError) as error:
        raise PreparationFailure('preparation_failure', f'Invalid structure export: {error}', folder) from error
    width,height = snapshot['size']
    if device_width is not None and width != device_width:
        raise PreparationFailure('preparation_failure',
            f'VPR exported device width {width}, requested {device_width}', folder)
    value = dict(record, structure_prefix=str(prefix), device_width=width, device_height=height,
        structure_sha256={suffix:sha(str(prefix)+suffix) for suffix in STRUCTURE_SUFFIXES},
        input_origin='recovery_structure_export', requested_device_width=device_width)
    freeze(manifest, dict(schema='tailcompile-v610-recovery-structure-1', signature=signature, record=value))
    emit('recovery_structure_ready', design=record['design'], grid=[width,height], attempt=name)
    return value


def _check_structure(record):
    prefix = str(record['structure_prefix'])
    hashes = {suffix:sha(prefix+suffix) for suffix in STRUCTURE_SUFFIXES}
    if record.get('structure_sha256') is not None and hashes != record['structure_sha256']:
        raise ValueError(f'Canonical AP structure changed: {prefix}')
    snapshot = read_ap_snapshot(prefix, structure=True)
    if snapshot['size'] != [record['device_width'],record['device_height']]:
        raise ValueError(f'Canonical AP structure device dimensions changed: {prefix}')
    return hashes


def prepare_input(a, record, imported=None):
    """Reuse imported validated inputs read-only, otherwise synthesize and export."""
    record = _clean_record(record)
    if imported is not None:
        if any(imported.get(key) != record.get(key) for key in SOURCE_KEYS):
            raise ValueError(f'Imported input differs from frozen inventory: {record["design"]}')
        _check_mapped(imported)
        structure_hashes = _check_structure(imported)
        value = dict(_clean_record(imported), **record)
        value.update(structure_sha256=structure_hashes, input_origin='verified_read_only_recovery_input')
        emit('recovery_input_reused', design=record['design'], mapped_blif=imported['mapped_blif'])
        return value
    return _structure(a, _canonical_mapping(a, record))


def resize_input(a, record, device_width):
    """Export a new geometry while retaining the exact existing mapped BLIF."""
    if isinstance(device_width,bool) or not isinstance(device_width,int) or device_width < 3:
        raise ValueError('device_width must be an integer >=3')
    if device_width <= record['device_width']:
        raise ValueError('Device recovery may only increase the qualified width')
    return _structure(a, record, device_width)


def is_device_capacity_failure(row, folder):
    """Match known VPR capacity/legal-site failures, never timeout/unroutability."""
    code = row.get('returncode', row.get('process_returncode'))
    if code not in (1,2) or row.get('status') in {'success','complete','unroutable','timeout','killed'}:
        return False
    path = Path(folder)/'vpr.out'
    if not path.is_file():
        return False
    text = path.read_text(encoding='utf-8', errors='replace')
    if log_storage_failure(text,code):
        return False
    return bool(re.search(
        r'Message:\s*Failed to find device which satisfies resource requirements required:\s*[^\n]+\(available\s+[^\n]+\)',
        text) or re.search(
        r'Message:\s*\d+ blocks could not be placed during initial placement;\s*no spaces were available for them on the grid\.',
        text))
