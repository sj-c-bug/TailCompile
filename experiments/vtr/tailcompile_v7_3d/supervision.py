"""Immutable raw 3D GP labels and individually resource-compatible targets."""
from __future__ import annotations

import hashlib
import json
import math

SCHEMA = 'tailcompile-v7-3d-projected-teacher-1'


def native_layer(raw_z, layer_count):
    """40f8 density import uses std::round: nonnegative half ties go upward.

    Source: flat_placement_density_manager.cpp:274-278. Never clamp a rounded
    layer outside the device, which would conceal an export/interface error.
    """
    if not math.isfinite(raw_z) or raw_z < 0:
        raise ValueError('raw GP Z must be finite and nonnegative')
    mapped = math.floor(raw_z + .5)
    if not 0 <= mapped < layer_count:
        raise ValueError(f'raw GP Z {raw_z} rounds outside {layer_count} layers')
    return mapped


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def project_teacher(data, target):
    import torch
    try:
        from .model import compatibility_mask, validate, coordinate_interior
    except ImportError:
        from model import compatibility_mask, validate, coordinate_interior
    meta = validate(data)
    layers, layer_capacity = meta[:2]
    n = len(data['demand'])
    if data['demand'].device.type != 'cpu':
        raise ValueError('teacher projection must be derived on CPU')
    raw_xy = torch.as_tensor(target['teacher_xy'], dtype=torch.float64).clone()
    if raw_xy.shape != (n, 2) or not torch.isfinite(raw_xy).all():
        raise ValueError('raw teacher XY must be finite Nx2')
    if (raw_xy < 0).any() or (raw_xy >= 1).any():
        raise ValueError('raw teacher XY must use normalized [0,1)')
    raw_z = target.get('teacher_raw_z', target.get('teacher_layer'))
    if raw_z is None:
        raise ValueError('raw teacher Z required, not inferred from planar leaf ids')
    if len(raw_z) != n:
        raise ValueError('raw teacher Z count mismatch')
    raw_z = [float(z) for z in raw_z]
    mapped_layer = torch.tensor([native_layer(z, len(layer_capacity)) for z in raw_z])
    bounds = data['region_bounds'].double()
    low, high = bounds[:, [0, 2]], bounds[:, [1, 3]]
    safe_low, safe_high = coordinate_interior(data['region_bounds'])
    safe_low, safe_high = safe_low.double(), safe_high.double()
    movable = data.get('movable', torch.ones(n, dtype=torch.bool)).bool()
    scale = torch.tensor(data.get('teacher_axis_scale', [1., 1.]), dtype=torch.float64)
    if scale.shape != (2,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('teacher_axis_scale must be two positive physical dimensions')
    projected_region = torch.full((n,), -1, dtype=torch.long)
    raw_region = torch.full((n,), -1, dtype=torch.long)
    projected_xy = raw_xy.clone()
    projected_layer = mapped_layer.clone()
    for node in range(n):
        point = raw_xy[node]
        inside = ((point >= low - 1e-10) & (point <= high + 1e-10)).all(1)
        original = torch.where(inside & (layers == mapped_layer[node]))[0]
        if not len(original):
            raise ValueError(f'raw teacher node {node} not in its mapped layer geometry')
        # Leaves are emitted in deterministic ID order; native lower-index ties.
        raw_region[node] = int(original[0])
        compatible = compatibility_mask(data, torch.tensor([node]), meta)[0]
        if compatible[raw_region[node]]:
            projected_region[node] = raw_region[node]
            continue
        if not movable[node]:
            raise ValueError('fixed teacher is incompatible with its immutable resource constraints')
        available = torch.where(compatible)[0]
        layer_hops = (layers[available] - mapped_layer[node]).abs()
        distance = (torch.maximum(low[available] - point, point - high[available])
                    .clamp_min(0) * scale).square().sum(1)
        center_distance = (((low[available] + high[available]) * .5 - point) * scale).square().sum(1)
        rank = min(range(len(available)), key=lambda i: (int(layer_hops[i]), float(distance[i]),
                                                        float(center_distance[i]), int(available[i])))
        region = int(available[rank])
        projected_region[node] = region
        projected_layer[node] = layers[region]
        projected_xy[node] = torch.maximum(safe_low[region], torch.minimum(safe_high[region], point))
    def overflow(assignment):
        usage = torch.zeros_like(data['capacity'])
        usage.index_add_(0, assignment, data['demand'])
        per_layer = torch.zeros_like(layer_capacity)
        per_layer.index_add_(0, layers, usage)
        return dict(leaf_overflow_by_resource=(usage - data['capacity']).clamp_min(0).sum(0).tolist(),
                    layer_overflow_by_resource=(per_layer - layer_capacity).clamp_min(0).sum(0).tolist())
    derived = dict(schema=SCHEMA, teacher_region=projected_region.tolist(),
                   teacher_xy=projected_xy.tolist(), teacher_layer=projected_layer.tolist(),
                   teacher_raw_z=raw_z, movable=movable.tolist())
    audit = dict(schema=SCHEMA, raw_sha256=_digest(target), derived_sha256=_digest(derived),
                 native_z_mapping='std::round_nonnegative_floor(z+0.5)',
                 projected_node_count=int((projected_region != raw_region).sum()),
                 changed_layer_count=int((projected_layer != mapped_layer).sum()),
                 raw_z=raw_z, mapped_layer=mapped_layer.tolist(),
                 projected_layer=projected_layer.tolist(),
                 xy_tile_displacement=((projected_xy - raw_xy) * scale).norm(dim=1).tolist(),
                 raw_capacity=overflow(raw_region), projected_capacity=overflow(projected_region),
                 raw_inputs_modified=False, aggregate_packing_legality_claimed=False)
    return derived, audit


def select_checkpoint(reports, *, runtime_ratio_limit=.95, quality_ratio_limit=1.03):
    """Select only from real validation reports, never teacher reconstruction loss.

    Every report must aggregate the same pre-registered validation conditions.
    This helper refuses a manifest mismatch rather than comparing unlike work.
    """
    signatures = {report.get('validation_signature') for report in reports}
    if len(signatures) != 1 or None in signatures:
        raise ValueError('checkpoint selection needs identical validation signatures')
    eligible = []
    for report in reports:
        if report.get('evidence') != 'real_vpr_validation':
            raise ValueError('checkpoint selection requires actual VPR validation')
        if report.get('unknown_count', 0) or report['success_count'] < report['default_success_count']:
            continue
        if not report.get('strata_success_protection_passed', True):
            continue
        if any(not math.isfinite(report[key]) or report[key] <= 0 for key in
               ('cpd_ratio', 'wirelength_ratio', 'total_runtime_ratio')):
            continue
        if any(report[key] > quality_ratio_limit for key in ('cpd_ratio', 'wirelength_ratio')):
            continue
        eligible.append(report)
    if not eligible:
        return dict(status='no_checkpoint_passed_protection', checkpoint=None)
    selected = min(eligible, key=lambda item: (item['total_runtime_ratio'], item['cpd_ratio'], item['epoch']))
    return dict(status='selected_by_real_vpr_validation', checkpoint=selected['checkpoint'],
                epoch=selected['epoch'], runtime_target_met=selected['total_runtime_ratio'] <= runtime_ratio_limit,
                report=selected)
