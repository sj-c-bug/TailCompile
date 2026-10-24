"""Frozen-weight initialization with explicit preloaded-input runtime accounting.

Model and graph loading are outside the charged interval. Both GNN encoders,
matching, density repair, coordinate prediction and artifact production are
inside it. Device/logic embeddings are deliberately recomputed on every call.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

from bridge import HERE, V68, V69, emit, read, sha, write

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch
from tailcompile_v68.model import SCHEMA
from tailcompile_v69.fast_model import FastHierarchicalInitializer
from balanced_model import BalancedHierarchicalInitializer


GENERATION_SCHEMA = 'tailcompile-v610-preloaded-frozen-generation-1'
# One CPU model per process. No growing graph/embedding cache across designs.
_MODEL_CACHE = {}


def clear_model_cache():
    _MODEL_CACHE.clear()


def _options(options):
    values = dict(options)
    decoder = values.pop('decoder', 'balanced')
    if decoder not in ('balanced', 'legacy_soft'):
        raise ValueError('decoder must be balanced or legacy_soft')
    allowed = {'topk', 'soft_density_passes', 'density_price_weight', 'density_step',
               'repair_passes', 'repair_topk', 'repair_max_moves', 'repair_locality_weight'}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f'Unknown frozen decoder options: {sorted(unknown)}')
    if decoder == 'legacy_soft':
        values = {key: value for key, value in values.items() if not key.startswith('repair_')}
    return decoder, values


def signature(context, checkpoint, options, candidate=0):
    if candidate != 0:
        raise ValueError('Frozen benchmark sweep uses deterministic candidate 0 only')
    _options(options)
    return dict(schema=GENERATION_SCHEMA, checkpoint_sha256=sha(checkpoint),
        options=dict(options), candidate=0, sampling='none',
        inputs={key: sha(context[key]) for key in ('graph', 'logic', 'device', 'supply')},
        implementation={str(path.relative_to(HERE.parent)): sha(path) for path in
            (Path(__file__), HERE / 'balanced_model.py', V69 / 'fast_model.py', V68 / 'model.py')})


def validate_generation(folder, expected):
    folder = Path(folder)
    if not (folder / 'generation.json').exists():
        return None
    saved = read(folder / 'generation.json')
    if saved.get('signature') != expected:
        raise ValueError(f'Generation identity changed: {folder}; use a new output directory')
    for key in ('ap_init', 'fplace', 'target'):
        if sha(saved[key]) != saved[key + '_sha256']:
            raise ValueError(f'Changed generation artifact: {saved[key]}')
    return saved


def _load_model(checkpoint, digest, decoder, measured):
    key = (str(Path(checkpoint).resolve()), digest, decoder)
    if _MODEL_CACHE.get('key') == key:
        return _MODEL_CACHE['model'], _MODEL_CACHE['metadata'], True
    saved = measured('checkpoint_load', lambda: torch.load(
        checkpoint, map_location='cpu', weights_only=True))
    if saved.get('schema') != SCHEMA:
        raise ValueError('Expected the frozen v68/v69 hierarchical coordinate generator checkpoint')

    def construct():
        cls = BalancedHierarchicalInitializer if decoder == 'balanced' else FastHierarchicalInitializer
        model = cls(**saved['model_config'])
        model.load_state_dict(saved['model_state'], strict=True)
        return model.eval()

    model = measured('model_setup', construct)
    metadata = {key: saved.get(key, []) or [] for key in
                ('training_families', 'validation_families')}
    _MODEL_CACHE.clear()
    _MODEL_CACHE.update(key=key, model=model, metadata=metadata)
    return model, metadata, False


def _lower(folder, logical, physical, target):
    """Lower whole movable AP molecules; leave fixed atoms to native VPR."""
    rows, flat, seen = [], [], set()
    if len(logical['clusters']) != len(target['movable']):
        raise ValueError('Logic/AP graph node count mismatch')
    for index, block in enumerate(logical['clusters']):
        atoms = block.get('atoms') if isinstance(block, dict) else None
        if not isinstance(atoms, list) or not atoms:
            raise ValueError(f'AP cluster {index} must contain a nonempty atoms list')
        for atom in atoms:
            if (not isinstance(atom, str) or not atom or atom.startswith('#') or
                    any(c.isspace() or not c.isprintable() for c in atom)):
                raise ValueError(f'Invalid AP atom name in cluster {index}')
            if atom in seen:
                raise ValueError(f'Duplicate AP atom {atom!r}; fixed/movable molecules must be disjoint')
            seen.add(atom)
    width, height = float(physical['grid_width']), float(physical['grid_height'])
    if not all(math.isfinite(v) and v > 0 for v in (width, height)):
        raise ValueError('Invalid physical device dimensions')
    for index, block in enumerate(logical['clusters']):
        if not target['movable'][index]:
            continue
        x, y = target['teacher_xy'][index]
        if not all(math.isfinite(v) and 0 <= v <= 1 for v in (x, y)):
            raise ValueError('Invalid native AP coordinate')
        x = min(x * width, math.nextafter(width, 0.))
        y = min(y * height, math.nextafter(height, 0.))
        for atom in block['atoms']:
            rows.append(f'{atom}\t{x:.17g}\t{y:.17g}\t0\t0\n')
            flat.append(f'{atom}\t{math.floor(x)}\t{math.floor(y)}\t0\t0\n')
    (folder / 'init.ap.tsv').write_text(
        '# v610 frozen-weight native GP initialization; regional capacity is a proxy\n' + ''.join(rows),
        encoding='utf-8')
    (folder / 'init.fplace').write_text(
        '# Diagnostic only; not a legalized placement or GP bypass\n' + ''.join(flat), encoding='utf-8')
    write(folder / 'target.json', target)


def generate(context, checkpoint, folder, options, candidate=0, reuse=True, *, before_inference=None):
    started = time.perf_counter()
    torch.set_num_threads(1)
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    expected = signature(context, checkpoint, options, candidate)
    if reuse:
        old = validate_generation(folder, expected)
        if old is not None:
            return old
    elif (folder / 'generation.json').exists():
        raise ValueError('Timed generation requires a fresh measured directory')
    decoder, predict_options = _options(options)
    timing = {'checkpoint_load': 0., 'model_setup': 0.}

    def measured(name, function):
        before = time.perf_counter()
        value = function()
        timing[name] = time.perf_counter() - before
        return value

    model, metadata, model_cache_hit = _load_model(
        checkpoint, expected['checkpoint_sha256'], decoder, measured)
    data = measured('graph_load', lambda: torch.load(
        context['graph'], map_location='cpu', weights_only=True))
    # Both metadata files are loaded BEFORE the first GNN inference. Loading
    # after prediction and subtracting an estimated cost would be misleading.
    logical, physical = measured('metadata_load', lambda: (
        read(context['logic']), read(context['device'])))
    if len(logical['clusters']) != len(data['movable']):
        raise ValueError('Logic/AP graph node count mismatch')
    if len(physical['regions']) != len(data['capacity']):
        raise ValueError('Physical IR/graph leaf count mismatch')
    excluded_setup_seconds = time.perf_counter() - started

    gate_started = time.perf_counter()
    if before_inference is not None:
        before_inference()
    gate_wait_seconds = time.perf_counter() - gate_started if before_inference is not None else 0.

    compute_started = time.perf_counter()
    with torch.no_grad():
        device_embedding = measured('rrg_encoding', lambda: model.encode_device(data))
        logic_embedding = measured('logic_forward', lambda: model.encode_logic(data))
        prediction = measured('matching_decode_coordinate', lambda: model.predict(
            data, device_embedding=device_embedding, logic_embedding=logic_embedding,
            **predict_options))
    target = measured('target_serialization', lambda: dict(
        teacher_region=prediction['assignment'].tolist(),
        teacher_xy=prediction['xy'].detach().cpu().tolist(),
        movable=data['movable'].tolist()))
    measured('lowering_and_target_io', lambda: _lower(folder, logical, physical, target))
    artifacts = {}

    def hash_artifacts():
        for key, name in (('ap_init', 'init.ap.tsv'), ('fplace', 'init.fplace'), ('target', 'target.json')):
            artifacts[key] = str((folder / name).resolve())
            artifacts[key + '_sha256'] = sha(folder / name)
    measured('artifact_hashing', hash_artifacts)
    compute_generation_seconds = time.perf_counter() - compute_started
    family = context.get('family')
    exposure = ('training_family' if family in metadata['training_families'] else
                'validation_family' if family in metadata['validation_families'] else
                'not_listed_in_checkpoint')
    result = dict(signature=expected, timing=timing, phase_seconds=prediction.get('phase_seconds', {}),
        capacity_diagnostics=prediction['capacity_diagnostics'],
        capacity_diagnostics_before=prediction.get('capacity_diagnostics_before'),
        repair_diagnostics=prediction.get('repair_diagnostics'),
        capacity_usage=prediction['usage'].tolist(),
        candidate_region_count=prediction['candidate_region_count'],
        compute_generation_seconds=compute_generation_seconds,
        inference_seconds=timing['rrg_encoding'] + timing['logic_forward'],
        excluded_setup_seconds=excluded_setup_seconds,
        gate_wait_seconds=gate_wait_seconds,
        measured_generation_seconds=time.perf_counter() - started,
        model_cache_hit=model_cache_hit, graph_embedding_cache_hit=False,
        hard_regional_capacity_enforced=False, regional_capacity_is_proxy=True,
        native_gp_required=True, final_legality='Certified only by VPR, not this initializer',
        leaf_count=len(physical['regions']), hierarchy_level_count=len(data['parents']),
        decoder_kind=prediction.get('decoder_kind', decoder),
        checkpoint_exposure=dict(family=family, category=exposure,
            original_split=context.get('original_split', context.get('source_split', context.get('split'))),
            interpretation='Frozen-weight coverage sweep, not a new unseen-family generalization test'),
        runtime_scope='From fresh device+logic GNN inference through decode/repair/coordinate/lowering/artifact hashing; excludes startup and model/graph/metadata loading',
        excluded_setup_scope='Pre-inference checkpoint/model/graph/metadata loading and validation/provenance setup; interpreter startup measured by parent only',
        rrg_encoding_included_in_compute=True, **artifacts)
    # Reporting the measured interval is bookkeeping, not initializer work.
    write(folder / 'generation.json', result)
    emit('generated', design=context['design'], width=context['width'], candidate=0,
         compute_generation_seconds=compute_generation_seconds,
         excluded_setup_seconds=excluded_setup_seconds, decoder=decoder,
         capacity_diagnostics=result['capacity_diagnostics'], repair=result['repair_diagnostics'])
    return result


def worker(task):
    try:
        return generate(*task)
    except Exception as error:
        context = task[0]
        raise RuntimeError(f'Frozen generation failed: {context["design"]}/w{context["width"]}: {error}') from error
