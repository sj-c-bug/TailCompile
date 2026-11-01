"""008 targets in the same layer-capacity/compatibility set as deployment.

This projection preserves already valid targets. Otherwise it performs at most
two scalar overload-repair sweeps and one bounded deployment-allocator fallback.
It certifies only the declared five-resource layer proxy, never physical packing
or aggregate leaf capacity. Each input row remains one atomic APBlock.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import time

import torch

try:
    from .model import validate, compatibility_mask, allocate_layers, coordinate_interior, dispersed_xy, CapacityError
except ImportError:
    from model import validate, compatibility_mask, allocate_layers, coordinate_interior, dispersed_xy, CapacityError

SCHEMA = 'tailcompile-v7-aligned-teacher-008-1'
AUDIT_SCHEMA = 'tailcompile-v7-teacher-alignment-audit-008-1'
RESOURCE_KEYS = ('lut', 'ff', 'memory', 'dsp_top', 'io')
CONSTRAINT_KEYS = ('demand', 'capacity', 'region_layer', 'region_bounds', 'layer_capacity',
                   'legal_layers', 'fixed_layer', 'fixed_region', 'fixed_xy', 'movable', 'density_spread_fraction')
TOLERANCE = 1e-6


class TeacherAlignmentError(ValueError):
    """The supplied label violates the declared aligned-target contract."""


class TeacherProjectionFailure(CapacityError):
    """Bounded projection failed; not a certificate of physical infeasibility."""


def _digest(value):
    def serial(v):
        if isinstance(v, torch.Tensor):
            return dict(dtype=str(v.dtype), shape=list(v.shape), values=v.detach().cpu().tolist())
        if isinstance(v, dict): return {k: serial(x) for k, x in v.items()}
        if isinstance(v, (list, tuple)): return [serial(x) for x in v]
        return v
    return hashlib.sha256(json.dumps(serial(value), sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def constraints_digest(data):
    return _digest({key: data[key] for key in CONSTRAINT_KEYS if key in data})


def _arrays(data, target):
    n, r = len(data['demand']), len(data['capacity'])
    device = data['demand'].device
    source_regions = torch.as_tensor(target['teacher_region'], device=device)
    if source_regions.dtype == torch.bool or not torch.isfinite(source_regions).all() or (source_regions != source_regions.round()).any():
        raise TeacherAlignmentError('teacher leaf IDs must be finite integers')
    regions = source_regions.long()
    xy = torch.as_tensor(target['teacher_xy'], dtype=torch.float64, device=device)
    if regions.shape != (n,) or (regions < 0).any() or (regions >= r).any():
        raise TeacherAlignmentError('teacher leaf coverage/range mismatch')
    if xy.shape != (n, 2) or not torch.isfinite(xy).all() or (xy < 0).any() or (xy >= 1).any():
        raise TeacherAlignmentError('teacher XY must be finite normalized Nx2 in [0,1)')
    layers = data['region_layer'][regions].long()
    if 'teacher_layer' in target:
        source_layer = torch.as_tensor(target['teacher_layer'], device=device)
        if source_layer.dtype == torch.bool or not torch.isfinite(source_layer).all() or (source_layer != source_layer.round()).any():
            raise TeacherAlignmentError('teacher layer IDs must be finite integers')
        supplied = source_layer.long()
        if supplied.shape != (n,) or not torch.equal(layers, supplied):
            raise TeacherAlignmentError('teacher layer and leaf disagree')
    return regions, layers, xy


def _capacity_audit(data, regions, layers, meta):
    layer_capacity = meta[1]
    usage = layer_capacity.new_zeros(layer_capacity.shape)
    usage.index_add_(0, layers, data['demand'])
    overflow = (usage-layer_capacity).clamp_min(0)
    leaf_usage = data['capacity'].new_zeros(data['capacity'].shape)
    leaf_usage.index_add_(0, regions, data['demand'])
    leaf_overflow = (leaf_usage-data['capacity']).clamp_min(0)
    return dict(layer_usage=usage.detach().cpu().tolist(),
        layer_capacity=layer_capacity.detach().cpu().tolist(),
        layer_overflow=overflow.detach().cpu().tolist(),
        layer_overflow_by_resource=overflow.sum(0).detach().cpu().tolist(),
        leaf_overflow_by_resource=leaf_overflow.sum(0).detach().cpu().tolist(),
        leaf_overflow_count=int((leaf_overflow > TOLERANCE).any(1).sum()),
        hard_layer_capacity_satisfied=bool((usage <= layer_capacity+TOLERANCE).all()),
        aggregate_leaf_capacity_certified=False, physical_packing_certified=False)


def _compatibility(data, regions, meta, chunk_size, deadline=None):
    n, layer_count = len(data['demand']), len(meta[1])
    device = data['demand'].device
    allowed = torch.zeros((n, layer_count), dtype=torch.bool, device=device)
    selected_ok = torch.zeros(n, dtype=torch.bool, device=device)
    for start in range(0, n, chunk_size):
        if deadline is not None and time.perf_counter() > deadline:
            raise TeacherProjectionFailure('teacher projection time budget exceeded; no infeasibility claim')
        ids = torch.arange(start, min(n, start+chunk_size), device=device)
        mask = compatibility_mask(data, ids, meta)
        selected_ok[ids] = mask.gather(1, regions[ids, None]).squeeze(1)
        for z in range(layer_count):
            allowed[ids, z] = mask[:, meta[0] == z].any(1)
    return allowed, selected_ok


def deployed_xy_bounds(data, assignment, metadata=None):
    """Exact range of the shared head plus deployed finite density blend.

    The final assignment determines low-discrepancy ranks, including fixed rows.
    A fixed axis is overwritten by deployment after blending, hence its range
    has the single declared fixed value rather than an artificial blend bound.
    """
    meta = metadata or validate(data)
    blend = float(data.get('density_spread_fraction', .15))
    if not math.isfinite(blend) or not 0 <= blend <= 1:
        raise ValueError('density_spread_fraction must be finite in [0,1]')
    low, high = coordinate_interior(data['region_bounds'][assignment])
    spread = dispersed_xy(data, assignment)
    low = ((1-blend)*low + blend*spread).double()
    high = ((1-blend)*high + blend*spread).double()
    known = meta[4] >= 0
    low[known] = meta[4].double()[known]
    high[known] = meta[4].double()[known]
    return low, high


def _geometry_audit(data, regions, xy, meta):
    bounds = data['region_bounds'][regions].double()
    low, high = bounds[:, [0, 2]], bounds[:, [1, 3]]
    inside = ((xy >= low-TOLERANCE) & (xy <= high+TOLERANCE)).all(1)
    known = meta[4] >= 0
    fixed_ok = ((xy-meta[4].double()).abs() <= TOLERANCE) | ~known
    deployed_low, deployed_high = deployed_xy_bounds(data, regions, meta)
    reachable = ((xy >= deployed_low-TOLERANCE) & (xy <= deployed_high+TOLERANCE)).all(1)
    return bool(inside.all()), bool(fixed_ok.all()), bool(reachable.all())


def verify_teacher(data, teacher, *, chunk_size=256):
    """Recompute the certificate; never trusts a saved success boolean."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError('positive chunk_size required')
    if teacher.get('schema') != SCHEMA or 'teacher_layer' not in teacher:
        raise TeacherAlignmentError('008 aligned teacher schema/layer required')
    meta = validate(data)
    regions, layers, xy = _arrays(data, teacher)
    if teacher.get('constraints_sha256') != constraints_digest(data):
        raise TeacherAlignmentError('aligned teacher belongs to different constraints')
    _, compatible = _compatibility(data, regions, meta, chunk_size)
    geometry_ok, fixed_ok, reachable = _geometry_audit(data, regions, xy, meta)
    capacity = _capacity_audit(data, regions, layers, meta)
    if not compatible.all() or not geometry_ok or not fixed_ok or not reachable or not capacity['hard_layer_capacity_satisfied']:
        raise TeacherAlignmentError('aligned teacher violates layer capacity, compatibility, geometry, fixed axes or deployed XY range')
    return dict(schema=AUDIT_SCHEMA, valid=True, compatibility_passed=True,
        fixed_axes_passed=fixed_ok, xy_in_selected_leaf=geometry_ok, xy_in_deployed_blend_range=reachable,
        layer_matches_selected_leaf=True, logical_unit='atomic APBlock row',
        constraints_sha256=teacher['constraints_sha256'], **capacity)


def _repair_layers(data, raw_layer, allowed, meta, deadline):
    """Preserve teachers, move large overload contributors, then one fallback.

    This minimizes disturbance heuristically, not an optimal cardinality solve.
    No capacity relaxation, recursive search or repeated optimizer is allowed.
    """
    n, count = allowed.shape
    demand = data['demand'].double().tolist()
    cap = meta[1].double().tolist()
    permit = allowed.tolist()
    original = raw_layer.tolist()
    chosen = list(original)
    for i in range(n):
        options = [z for z in range(count) if permit[i][z]]
        if not options:
            raise TeacherProjectionFailure(f'APBlock {i} has no compatible layer; no physical infeasibility claim')
        if not permit[i][chosen[i]]:
            chosen[i] = min(options, key=lambda z: (abs(z-original[i]), z))
    used = [[0.]*5 for _ in range(count)]
    for i, z in enumerate(chosen):
        for t in range(5): used[z][t] += demand[i][t]
    def feasible():
        return all(used[z][t] <= cap[z][t]+TOLERANCE for z in range(count) for t in range(5))
    if feasible():
        return torch.tensor(chosen), 'teacher_layers_retained_or_constraint_corrected', 0
    pressure = [max(demand[i][t]/max(max(c[t] for c in cap), 1.) for t in range(5)) for i in range(n)]
    order = sorted(range(n), key=lambda i: (-pressure[i], i))
    moves = 0
    for sweep in range(2):
        for at, i in enumerate(order):
            if at % 256 == 0 and time.perf_counter() > deadline:
                raise TeacherProjectionFailure('teacher layer repair exceeded budget; no infeasibility claim')
            old = chosen[i]
            overloaded = [t for t in range(5) if used[old][t] > cap[old][t]+TOLERANCE]
            if not overloaded or not any(demand[i][t] > 0 for t in overloaded):
                continue
            possible = [z for z in range(count) if z != old and permit[i][z] and
                        all(used[z][t]+demand[i][t] <= cap[z][t]+TOLERANCE for t in range(5))]
            if not possible:
                continue
            z = min(possible, key=lambda z: (z != original[i], abs(z-original[i]),
                        sum((used[z][t]+demand[i][t])/max(cap[z][t], 1.) for t in range(5)), z))
            for t in range(5):
                used[old][t] -= demand[i][t]; used[z][t] += demand[i][t]
            chosen[i] = z; moves += 1
        if feasible():
            return torch.tensor(chosen), 'two_pass_overload_repair', moves
    if time.perf_counter() > deadline:
        raise TeacherProjectionFailure('teacher projection budget expired before fallback')
    # The exact same bounded capacity allocator as actual deployment. A large
    # finite preference preserves the teacher layer wherever greedy feasibility permits.
    scores = torch.zeros((n, count), dtype=torch.float64)
    for z in range(count): scores[:, z] = -(raw_layer-z).abs().double()/max(1, count)
    scores.scatter_add_(1, raw_layer[:, None], torch.full((n, 1), 10., dtype=torch.float64))
    try:
        chosen, _ = allocate_layers(scores, data['demand'], meta[1], allowed)
    except CapacityError as exc:
        raise TeacherProjectionFailure('bounded teacher projection failed; not physical infeasibility: '+str(exc)) from exc
    if time.perf_counter() > deadline:
        raise TeacherProjectionFailure('teacher projection budget exceeded after bounded fallback')
    return chosen, 'single_deployment_allocator_fallback', moves


def project_teacher(data, teacher, *, chunk_size=256, max_seconds=120.):
    """Derive an immutable, layer-feasible 008 target on CPU.

    An unchanged, already valid teacher is preserved bit-for-bit in label values.
    Corrected target coordinates are clipped to a compatible leaf, while every
    known fixed axis is preserved. No density blend is applied twice to a label.
    """
    if data['demand'].device.type != 'cpu':
        raise ValueError('teacher targets must be derived on CPU')
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError('positive chunk_size required')
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('positive finite projection budget required')
    started = time.perf_counter(); deadline = started+max_seconds
    try:
        meta = validate(data)
    except CapacityError as exc:
        raise TeacherProjectionFailure('declared deployment constraints have no legal layer: '+str(exc)) from exc
    raw_regions, raw_layers, raw_xy = _arrays(data, teacher)
    input_sha = _digest(teacher)
    constraints_sha = constraints_digest(data)
    if teacher.get('schema') == SCHEMA:
        audit = verify_teacher(data, teacher, chunk_size=chunk_size)
        return copy.deepcopy(teacher), dict(audit, projection_kind='already_aligned',
            changed_layer_count=0, changed_region_count=0, changed_xy_count=0,
            source_teacher_sha256=input_sha, derived_sha256=input_sha,
            raw_inputs_modified=False, projection_seconds=time.perf_counter()-started)
    try:
        allowed, raw_compatible = _compatibility(data, raw_regions, meta, chunk_size, deadline)
    except CapacityError as exc:
        raise TeacherProjectionFailure('teacher has no compatible deployment position: '+str(exc)) from exc
    raw_capacity = _capacity_audit(data, raw_regions, raw_layers, meta)
    raw_geometry, raw_fixed, raw_reachable = _geometry_audit(data, raw_regions, raw_xy, meta)
    if raw_capacity['hard_layer_capacity_satisfied'] and raw_compatible.all() and raw_geometry and raw_fixed and raw_reachable:
        chosen, kind, repair_moves = raw_layers.clone(), 'already_feasible_unchanged', 0
    else:
        chosen, kind, repair_moves = _repair_layers(data, raw_layers, allowed, meta, deadline)
    regions = raw_regions.clone()
    xy = raw_xy.clone()
    known = meta[4] >= 0
    xy[known] = torch.where((xy-meta[4].double()).abs() > TOLERANCE, meta[4].double(), xy)[known]
    changed = ~raw_compatible | (meta[0][regions] != chosen)
    ids_to_project = torch.where(changed)[0]
    bounds = data['region_bounds'].double()
    low, high = bounds[:, [0, 2]], bounds[:, [1, 3]]
    scale = torch.as_tensor(data.get('teacher_axis_scale', [1., 1.]), dtype=torch.float64)
    if scale.shape != (2,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('teacher_axis_scale requires two positive physical dimensions')
    for start in range(0, len(ids_to_project), chunk_size):
        if time.perf_counter() > deadline:
            raise TeacherProjectionFailure('teacher leaf projection exceeded budget')
        ids = ids_to_project[start:start+chunk_size]
        mask = compatibility_mask(data, ids, meta) & (meta[0][None, :] == chosen[ids, None])
        if (~mask.any(1)).any():
            raise TeacherProjectionFailure('assigned teacher layer has no compatible leaf')
        point = xy[ids, None, :]
        distance = (torch.maximum(low[None, :, :]-point, point-high[None, :, :]).clamp_min(0)*scale).square().sum(2)
        best_distance = distance.masked_fill(~mask, torch.inf).min(1).values
        candidates = mask & (distance == best_distance[:, None])
        center = (((low+high)[None, :, :]*.5-point)*scale).square().sum(2)
        best_center = center.masked_fill(~candidates, torch.inf).min(1).values
        tie = candidates & (center == best_center[:, None])
        numbers = torch.arange(len(bounds))[None, :].expand(len(ids), -1)
        regions[ids] = numbers.masked_fill(~tie, len(bounds)).min(1).values
    deployed_low, deployed_high = deployed_xy_bounds(data, regions, meta)
    outside_deployed = ((xy < deployed_low-TOLERANCE) | (xy > deployed_high+TOLERANCE)).any(1)
    project_xy = changed | outside_deployed
    xy[project_xy] = torch.maximum(deployed_low[project_xy], torch.minimum(deployed_high[project_xy], xy[project_xy]))
    xy[known] = meta[4].double()[known]
    # Preserve exact feasible source coordinates, including float-valued fixed
    # axes accepted by the shared validator's tolerance.
    if kind == 'already_feasible_unchanged': xy = raw_xy.clone()
    result = dict(schema=SCHEMA, teacher_region=regions.tolist(), teacher_layer=chosen.tolist(),
        teacher_xy=xy.tolist(), source_teacher_sha256=input_sha, constraints_sha256=constraints_sha,
        logical_unit='atomic APBlock row', hard_layer_capacity_satisfied=True)
    verified = verify_teacher(data, result, chunk_size=chunk_size)
    if time.perf_counter() > deadline:
        raise TeacherProjectionFailure('teacher verification exceeded projection budget')
    changed_layer = chosen != raw_layers
    changed_region = regions != raw_regions
    changed_xy = (xy != raw_xy).any(1)
    displacement = ((xy-raw_xy)*scale).norm(dim=1)
    samples = torch.where(changed_layer | changed_region | changed_xy)[0][:32].tolist()
    audit = dict(schema=AUDIT_SCHEMA, valid=True, source_teacher_sha256=input_sha,
        derived_sha256=_digest(result), constraints_sha256=constraints_sha,
        implementation_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        resource_keys=list(RESOURCE_KEYS), projection_kind=kind, maximum_repair_sweeps=2,
        fallback_allocator_calls=int(kind == 'single_deployment_allocator_fallback'), repair_moves=repair_moves,
        global_minimum_changed_labels_claimed=False, changed_layer_count=int(changed_layer.sum()),
        changed_region_count=int(changed_region.sum()), changed_xy_count=int(changed_xy.sum()),
        blend_reachability_changed_count=int(outside_deployed.sum()), raw_xy_in_deployed_blend_range=raw_reachable,
        xy_tile_displacement_max=float(displacement.max()), xy_tile_displacement_mean=float(displacement.mean()),
        modified_samples=[dict(node=i, source_layer=int(raw_layers[i]), target_layer=int(chosen[i]),
                               source_leaf=int(raw_regions[i]), target_leaf=int(regions[i])) for i in samples],
        raw_capacity=raw_capacity, aligned_capacity={k:v for k,v in verified.items() if k not in ('schema',)},
        raw_inputs_modified=False, projection_seconds=time.perf_counter()-started,
        aggregate_leaf_capacity_certified=False, physical_packing_certified=False)
    if _digest(teacher) != input_sha:
        raise AssertionError('raw teacher was modified')
    return result, audit
