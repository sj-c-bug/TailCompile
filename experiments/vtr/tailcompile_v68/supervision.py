"""Resource-aware supervision derived from immutable pre-legalization teachers.

A native GP coordinate is not a physical site assignment.  Move only labels
whose original leaf cannot individually accommodate their AP demand to the
nearest individually compatible leaf.  This is NOT packing legalization:
aggregate overflow is audited and remains a separate model penalty.  No graph
features, capacities, raw teachers, or final-test labels are modified here.
"""
from __future__ import annotations

import hashlib
import json

import torch

from model import CapacityError, coordinate_interior

SUPERVISION_SCHEMA = 'tailcompile-v68-resource-aware-teacher-1'
RESOURCE_KEYS = ('lut', 'ff', 'memory', 'dsp_top', 'io')


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def interior_bounds(bounds):
    """Float32-safe interior, including native region_at's lower-index ties.

    Merely clamping to a leaf's low edge can put an AP point in its neighbor
    when exported back to native coordinates.  A few float32 ULPs avoid both
    normalization rounding and saturation of a coordinate head at either edge.
    """
    low, high = coordinate_interior(bounds)
    return low.double(), high.double()


def project_teacher(data, target, *, chunk_size=512):
    """Return derived JSON labels and audit; CPU-only, bounded chunk x leaf work.

    Distance is point-to-leaf rectangle, then leaf-center distance, then index.
    Optional teacher_axis_scale=[W,H] measures distance in physical tile units;
    otherwise it is explicitly normalized AP distance.  Fixed nodes never move.
    Raw xy remains available to the caller for original-teacher edge protection.
    """
    if chunk_size < 1:
        raise ValueError('chunk_size must be positive')
    for key in ('demand', 'capacity', 'region_bounds', 'movable'):
        if torch.is_tensor(data.get(key)) and data[key].device.type != 'cpu':
            raise ValueError('teacher projection requires CPU graph tensors')
    demand, capacity = data['demand'].double(), data['capacity'].double()
    bounds = data['region_bounds'].double()
    regions = torch.as_tensor(target['teacher_region'], dtype=torch.long).clone()
    xy = torch.as_tensor(target['teacher_xy'], dtype=torch.float64).clone()
    movable = torch.as_tensor(target['movable'], dtype=torch.bool).clone()
    n = len(demand)
    if (demand.ndim != 2 or capacity.ndim != 2 or not len(capacity) or
            demand.shape[1] != capacity.shape[1] or bounds.shape != (len(capacity), 4) or
            regions.shape != (n,) or xy.shape != (n, 2) or movable.shape != (n,)):
        raise ValueError('teacher projection graph/label dimensions do not match')
    if any(not torch.isfinite(value).all() for value in (demand, capacity, bounds, xy)):
        raise ValueError('teacher projection inputs contain NaN/Inf')
    if (demand < 0).any() or (capacity < 0).any():
        raise ValueError('teacher projection demand/capacity must be nonnegative')
    if 'movable' in data and not torch.equal(data['movable'].bool(), movable):
        raise ValueError('teacher and graph movable masks differ')
    if ((regions < -1) | (regions >= len(capacity))).any() or (regions[movable] < 0).any():
        raise ValueError('teacher region outside device or missing for movable node')
    if ((xy[movable] < 0) | (xy[movable] >= 1)).any():
        raise ValueError('movable teacher coordinates must be in global [0,1)')
    low, high = bounds[:, [0, 2]], bounds[:, [1, 3]]
    if not (low < high).all() or (low < 0).any() or (high > 1).any():
        raise ValueError('invalid normalized region bounds')
    scale = torch.as_tensor(data.get('teacher_axis_scale', [1., 1.]), dtype=torch.float64)
    if scale.shape != (2,) or not torch.isfinite(scale).all() or (scale <= 0).any():
        raise ValueError('teacher_axis_scale must contain two finite positive dimensions')
    safe_low, safe_high = interior_bounds(data['region_bounds'])
    original_regions, original_xy = regions.clone(), xy.clone()
    nodes = torch.where(movable)[0]
    excessive = demand[nodes] > capacity[regions[nodes]] + 1e-6
    conflicted = nodes[excessive.any(-1)]
    centers = (low + high) * .5
    # Only incompatible nodes need an all-leaf search; ordinary labels stay put.
    for start in range(0, len(conflicted), chunk_size):
        active = conflicted[start:start + chunk_size]
        feasible = torch.ones((len(active), len(capacity)), dtype=torch.bool)
        for resource in range(demand.shape[1]):
            feasible &= demand[active, resource, None] <= capacity[None, :, resource] + 1e-6
        impossible = ~feasible.any(-1)
        if impossible.any():
            node = int(active[torch.where(impossible)[0][0]])
            raise CapacityError(f'node {node}: no individually feasible teacher region; '
                                f'demand={demand[node].tolist()}; proxy is not relaxed')
        distance = torch.zeros_like(feasible, dtype=torch.float64)
        center_distance = torch.zeros_like(distance)
        for axis in range(2):
            point = original_xy[active, axis, None]
            outside = torch.maximum((low[None, :, axis] - point).clamp_min(0),
                                    (point - high[None, :, axis]).clamp_min(0))
            distance += (outside * scale[axis]).square()
            center_distance += ((point - centers[None, :, axis]) * scale[axis]).square()
        distance.masked_fill_(~feasible, torch.inf)
        closest = distance == distance.min(-1, keepdim=True).values
        center_distance.masked_fill_(~closest, torch.inf)
        # argmin resolves remaining exact ties by stable lowest leaf index.
        regions[active] = center_distance.argmin(-1)
    xy[nodes] = torch.maximum(safe_low[regions[nodes]],
                              torch.minimum(safe_high[regions[nodes]], original_xy[nodes]))
    if (demand[nodes] > capacity[regions[nodes]] + 1e-6).any():
        raise AssertionError('internal teacher projection feasibility error')
    delta = (xy[nodes] - original_xy[nodes]) * scale
    displacement = torch.linalg.vector_norm(delta, dim=-1)
    changed = (xy != original_xy).any(-1) & movable
    boundary_adjusted = changed & (regions == original_regions)
    names = RESOURCE_KEYS if demand.shape[1] == len(RESOURCE_KEYS) else tuple(
        f'resource_{i}' for i in range(demand.shape[1]))

    def overflow(assignment):
        usage = torch.zeros_like(capacity)
        valid = assignment >= 0
        usage.index_add_(0, assignment[valid], demand[valid])
        excess = (usage - capacity).clamp_min(0).sum(0)
        return dict(zip(names, excess.tolist()))

    projected = dict(target, teacher_region=regions.tolist(), teacher_xy=xy.tolist(),
                     movable=movable.tolist(), teacher_projection_schema=SUPERVISION_SCHEMA,
                     raw_teacher_sha256=target.get('teacher_sha256'))
    moved_distances = displacement[displacement > 0]
    quantiles = (torch.quantile(moved_distances, torch.tensor([.5, .95, .99], dtype=torch.float64))
                 if len(moved_distances) else torch.zeros(3, dtype=torch.float64))
    audit = dict(schema=SUPERVISION_SCHEMA, movable_nodes=int(movable.sum()),
        conflicted_nodes=len(conflicted), reassigned_nodes=int((regions != original_regions).sum()),
        boundary_adjusted_nodes=int(boundary_adjusted.sum()),
        max_displacement=float(displacement.max()) if len(nodes) else 0.,
        mean_displacement=float(displacement.mean()) if len(nodes) else 0.,
        displacement_quantiles=dict(zip(('p50', 'p95', 'p99'), quantiles.tolist())),
        distance_units='physical_tiles' if 'teacher_axis_scale' in data else 'normalized_AP',
        axis_scale=scale.tolist(),
        conflict_resources=dict(zip(names, excessive.sum(0).tolist())),
        aggregate_overflow_raw=overflow(original_regions),
        aggregate_overflow_projected=overflow(regions),
        raw_target_sha256=_digest(target), projected_target_sha256=_digest(projected),
        note='Derived individually resource-compatible teacher, NOT a legal packing. '
             'Distance mean includes all movable nodes; quantiles include displaced nodes. '
             'Raw teacher geometry is retained for edge-length protection; capacities are unchanged.')
    return projected, audit
