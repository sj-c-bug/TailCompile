"""Layer-aware hierarchical AP initializer with bounded layer-capacity decoding.

The graph tensors extend v6.8 with region_layer[R], legal_layers[N,L], and
optional fixed_layer[N]. Layer capacities are hard *proxy* upper bounds. Leaf
density is a bounded soft objective; native GP/APPack certifies actual legality.
There is no local cut-risk search and no dense AP-node by physical-site matrix.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

VTR_DIRECTORY = Path(__file__).resolve().parents[1]
if str(VTR_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VTR_DIRECTORY))
from tailcompile_v68.model import (CapacityError, HierarchicalInitializer,
                                  coordinate_interior, to_device)

SCHEMA = 'tailcompile-v7-3d-hierarchical-generator-1'
DECODER_SCHEMA = 'tailcompile-v7-3d-layer-capacity-leaf-density-1'
RESOURCE_KEYS = ('lut', 'ff', 'memory', 'dsp_top', 'io')


def validate(data):
    demand, capacity = data['demand'], data['capacity']
    n, r = len(demand), len(capacity)
    if demand.ndim != 2 or capacity.ndim != 2 or not n or not r:
        raise ValueError('nonempty demand/capacity matrices required')
    if demand.shape[1] != 5 or capacity.shape[1] != 5:
        raise ValueError('explicit LUT/FF/memory/DSP/IO proxy dimensions required')
    for key in ('demand', 'capacity', 'cluster_x', 'region_x', 'region_bounds',
                'pool_x', 'logic_weight', 'region_weight'):
        if not torch.isfinite(data[key]).all():
            raise ValueError(f'nonfinite graph field: {key}')
    if (demand < 0).any() or (capacity < 0).any():
        raise ValueError('resource demand/capacity must be nonnegative')
    bounds = data['region_bounds']
    if bounds.shape != (r, 4) or (bounds < 0).any() or (bounds > 1).any():
        raise ValueError('bounds must be normalized Rx4')
    coordinate_interior(bounds)
    layers = data['region_layer'].long()
    if layers.shape != (r,) or (layers < 0).any():
        raise ValueError('one nonnegative layer per leaf required')
    layer_count = int(layers.max()) + 1
    if set(layers.cpu().tolist()) != set(range(layer_count)):
        raise ValueError('layer ids must be contiguous')
    layer_capacity = capacity.new_zeros((layer_count, 5))
    layer_capacity.index_add_(0, layers, capacity)
    if 'layer_capacity' in data:
        supplied = data['layer_capacity']
        if supplied.shape != layer_capacity.shape or not torch.allclose(supplied, layer_capacity):
            raise ValueError('layer_capacity must equal disjoint leaf capacity sum')
    legal = data.get('legal_layers', torch.ones((n, layer_count), dtype=torch.bool,
                                               device=demand.device)).bool()
    if legal.shape != (n, layer_count):
        raise ValueError('legal_layers shape mismatch')
    fixed_layer = data.get('fixed_layer', torch.full((n,), -1, dtype=torch.long,
                                                    device=demand.device)).long()
    fixed_region = data.get('fixed_region', torch.full((n,), -1, dtype=torch.long,
                                                      device=demand.device)).long()
    fixed_xy = data.get('fixed_xy', demand.new_full((n, 2), -1.))
    if fixed_layer.shape != (n,) or fixed_region.shape != (n,) or fixed_xy.shape != (n, 2):
        raise ValueError('fixed metadata shape mismatch')
    if ((fixed_layer < -1) | (fixed_layer >= layer_count)).any():
        raise ValueError('fixed layer outside device')
    if ((fixed_region < -1) | (fixed_region >= r)).any():
        raise ValueError('fixed region outside device')
    if not torch.isfinite(fixed_xy).all() or ((fixed_xy < 0) & (fixed_xy != -1)).any() or (fixed_xy >= 1).any():
        raise ValueError('fixed_xy must contain normalized [0,1) or -1 placeholders')
    pinned = fixed_layer >= 0
    legal = legal.clone()
    legal[pinned] &= torch.arange(layer_count, device=demand.device)[None, :] == fixed_layer[pinned, None]
    pinned_regions = fixed_region >= 0
    legal[pinned_regions] &= torch.arange(layer_count, device=demand.device)[None, :] == layers[fixed_region[pinned_regions], None]
    if (~legal.any(1)).any():
        raise CapacityError('a molecule has no legal layer')
    return layers, layer_capacity, legal, fixed_region, fixed_xy


def compatibility_mask(data, nodes, metadata=None):
    layers, _, legal, fixed_region, fixed_xy = metadata or validate(data)
    mask = legal[nodes][:, layers].clone()
    demand, capacity = data['demand'][nodes], data['capacity']
    for resource in range(5):
        mask &= demand[:, resource, None] <= capacity[None, :, resource] + 1e-6
    fixed = fixed_region[nodes]
    pinned = fixed >= 0
    if pinned.any():
        mask[pinned] &= torch.arange(len(capacity), device=nodes.device)[None, :] == fixed[pinned, None]
    bounds = data['region_bounds']
    for axis, lo, hi in ((0, 0, 1), (1, 2, 3)):
        position = fixed_xy[nodes, axis]
        known = position >= 0
        if known.any():
            # A known fixed_region resolves native boundary ownership exactly.
            mask[known] &= ((position[known, None] >= bounds[None, :, lo] - 1e-7) &
                            (position[known, None] <= bounds[None, :, hi] + 1e-7))
    if (~mask.any(1)).any():
        node = int(nodes[torch.where(~mask.any(1))[0][0]])
        raise CapacityError(f'molecule {node}: no individually compatible leaf')
    return mask


def allocate_layers(scores, demand, capacity, allowed, *, balanced=False):
    """O(N L T) deterministic greedy capacity allocation, no recursive search.

    Failure is reported as decoder failure, not physical infeasibility. More
    constrained/larger molecules are processed first. Fixed obligations are
    represented by single-layer allowed masks and consume capacity normally.
    """
    scores, demand, capacity = scores.cpu().double(), demand.cpu().double(), capacity.cpu().double()
    allowed = allowed.cpu().bool()
    n, layer_count = scores.shape
    if allowed.shape != scores.shape or demand.shape != (n, 5) or capacity.shape != (layer_count, 5):
        raise ValueError('layer allocation shape mismatch')
    if not torch.isfinite(demand).all() or not torch.isfinite(capacity).all():
        raise ValueError('nonfinite layer resource inputs')
    if (demand < 0).any() or (capacity < 0).any():
        raise ValueError('negative layer resources')
    if not torch.isfinite(scores[allowed]).all():
        raise ValueError('nonfinite allowed layer score')
    pressure = (demand / capacity.max(0).values.clamp_min(1.)).max(1).values
    compatible_counts, pressure_values = allowed.sum(1).tolist(), pressure.tolist()
    order = sorted(range(n), key=lambda i: (compatible_counts[i], -pressure_values[i], i))
    # Tiny LxT bookkeeping in native Python avoids one Torch dispatch (and on
    # CUDA one host/device synchronization) per resource and per molecule.
    score_values, demands = scores.tolist(), demand.tolist()
    capacities, allowed_values = capacity.tolist(), allowed.tolist()
    remaining = [row[:] for row in capacities]
    assignment = [-1] * n
    for node in order:
        possible = [layer for layer in range(layer_count) if allowed_values[node][layer] and
                    all(demands[node][t] <= remaining[layer][t] + 1e-6 for t in range(5))]
        if not possible:
            raise CapacityError(f'molecule {node}: bounded layer decoder cannot satisfy proxy capacity')
        def value(layer):
            projected = sum(((capacities[layer][t] - remaining[layer][t] + demands[node][t]) /
                             max(capacities[layer][t], 1.)) ** 2 for t in range(5)) / 5.
            return -projected if balanced else score_values[node][layer] - .05 * projected
        layer = max(possible, key=lambda z: (value(z), -z))
        assignment[node] = layer
        for resource in range(5):
            remaining[layer][resource] -= demands[node][resource]
    usage = capacity - torch.tensor(remaining, dtype=capacity.dtype)
    if (usage > capacity + 1e-5).any():
        raise AssertionError('layer capacity enforcement failed')
    return torch.tensor(assignment, dtype=torch.long), usage


def dispersed_xy(data, assignment):
    """Deterministic low-discrepancy points within each selected leaf."""
    bounds = data['region_bounds'][assignment]
    low, high = coordinate_interior(bounds)
    order = torch.argsort(assignment, stable=True)
    sorted_regions = assignment[order]
    position = torch.arange(len(assignment), device=assignment.device)
    new_group = torch.ones(len(assignment), dtype=torch.bool, device=assignment.device)
    new_group[1:] = sorted_regions[1:] != sorted_regions[:-1]
    group_start = torch.cummax(torch.where(new_group, position, 0), 0).values
    ranks = torch.empty(len(assignment), dtype=torch.float64, device=assignment.device)
    ranks[order] = (position - group_start).double()
    x = torch.frac((ranks + .5) * .6180339887498949)
    y = torch.frac((ranks + .5) * .4142135623730951)
    return low + (high - low) * torch.stack((x, y), -1).to(low)


def decode(data, logits, *, balanced=False, density_passes=2, coordinate=None):
    """Decode chunked scores into capacity-protected layers and soft-density leaves.

    logits(nodes) returns [len(nodes), R] on data's device. Leaf assignment uses
    a constant pass count; never retries or performs local risk search.
    """
    if isinstance(density_passes, bool) or not isinstance(density_passes, int) or not 0 <= density_passes <= 4:
        raise ValueError('density_passes must be an integer in [0,4]')
    meta = validate(data)
    layers, layer_capacity, _, _, fixed_xy = meta
    device, n, r = data['demand'].device, len(data['demand']), len(data['capacity'])
    layer_scores = data['demand'].new_full((n, len(layer_capacity)), -torch.inf)
    allowed = torch.zeros_like(layer_scores, dtype=torch.bool)
    chunk_size = int(data.get('matching_chunk_size', 256))
    for start in range(0, n, chunk_size):
        nodes = torch.arange(start, min(start + chunk_size, n), device=device)
        mask = compatibility_mask(data, nodes, meta)
        values = logits(nodes).masked_fill(~mask, -torch.inf)
        for layer in range(len(layer_capacity)):
            selected = layers == layer
            count = mask[:, selected].sum(1)
            allowed[nodes, layer] = count > 0
            layer_scores[nodes, layer] = torch.logsumexp(values[:, selected], 1) - count.clamp_min(1).to(values).log()
    chosen_layers, layer_usage = allocate_layers(layer_scores, data['demand'], layer_capacity,
                                                  allowed, balanced=balanced)
    chosen_layers = chosen_layers.to(device)
    assignment = torch.full((n,), -1, dtype=torch.long, device=device)
    prices = torch.zeros_like(data['capacity'])
    for pass_index in range(density_passes + 1):
        expected = torch.zeros_like(data['capacity'])
        for start in range(0, n, chunk_size):
            nodes = torch.arange(start, min(start + chunk_size, n), device=device)
            mask = compatibility_mask(data, nodes, meta)
            mask &= layers[None, :] == chosen_layers[nodes, None]
            values = logits(nodes) - data['demand'][nodes] @ (prices / data['capacity'].clamp_min(1.)).T
            if balanced:
                # Distinct ID ties spread equal-resource molecules across leaves.
                ids = torch.arange(r, device=device)[None, :]
                values = values - ((ids - nodes[:, None]) % r).to(values) * 1e-4
            values = values.masked_fill(~mask, -torch.inf)
            if pass_index == density_passes:
                assignment[nodes] = values.argmax(1)
            else:
                expected.add_(values.softmax(1).T @ data['demand'][nodes])
        if pass_index < density_passes:
            prices = (prices + (expected - data['capacity']) / data['capacity'].clamp_min(1.)).clamp(0, 100.)
    xy = coordinate(assignment) if coordinate is not None else dispersed_xy(data, assignment)
    spread = dispersed_xy(data, assignment)
    # One frozen density pass blends within-leaf points; no search/convergence.
    blend = float(data.get('density_spread_fraction', .15))
    if not 0 <= blend <= 1:
        raise ValueError('density_spread_fraction must be in [0,1]')
    xy = (1 - blend) * xy + blend * spread
    known = fixed_xy >= 0
    xy = torch.where(known, fixed_xy.to(xy), xy)
    if not torch.isfinite(xy).all() or (xy < 0).any() or (xy >= 1).any():
        raise ValueError('decoded XY outside normalized device')
    usage = torch.zeros_like(data['capacity'])
    usage.index_add_(0, assignment, data['demand'])
    overflow = (usage - data['capacity']).clamp_min(0)
    return dict(assignment=assignment.cpu(), layer=chosen_layers.cpu(), xy=xy,
                usage=usage.cpu(), layer_usage=layer_usage,
                capacity_diagnostics=dict(hard_layer_capacity_enforced=True,
                    hard_leaf_capacity_enforced=False, packing_legality_claimed=False,
                    overflow_by_resource=overflow.sum(0).cpu().tolist(),
                    overflow_region_count=int((overflow > 1e-6).any(1).sum()),
                    used_region_count=int(torch.unique(assignment).numel())),
                decoder_kind=DECODER_SCHEMA)


@torch.no_grad()
def capacity_only(data, *, density_passes=2):
    start = time.perf_counter()
    result = decode(data, lambda nodes: data['capacity'].new_zeros((len(nodes), len(data['capacity']))),
                    balanced=True, density_passes=density_passes)
    result['phase_seconds'] = dict(total=time.perf_counter() - start)
    result['initializer'] = 'capacity_only'
    return result


class Hierarchical3DInitializer(HierarchicalInitializer):
    def __init__(self, cluster_width, region_width, hidden=64, chunk_size=256,
                 pool_width=8, demand_width=5):
        super().__init__(cluster_width, region_width, hidden, chunk_size, pool_width, demand_width)
        self.layer_geometry = nn.Sequential(nn.Linear(1, hidden), nn.GELU(), nn.Linear(hidden, hidden))

    def encode_device(self, data):
        encoded = super().encode_device(data)
        layer = data['region_layer'].to(encoded)
        return encoded + self.layer_geometry((layer / layer.max().clamp_min(1.)).unsqueeze(1))

    def coordinates_from_encoded(self, data, logic, regions, assignment):
        # Unlike v6.8, partially fixed axes are legal and retain -1 placeholders.
        unconstrained = dict(data)
        unconstrained.pop('movable', None)
        xy = super().coordinates_from_encoded(unconstrained, logic, regions, assignment)
        fixed = data.get('fixed_xy')
        return torch.where(fixed >= 0, fixed.to(xy), xy) if fixed is not None else xy

    def logits_for(self, data, logic, regions, nodes, metadata=None):
        logits = self.matching_logits(logic[nodes], regions)
        return logits.masked_fill(~compatibility_mask(data, nodes, metadata), -torch.inf)

    def deployment_coordinates(self, data, logic, regions, assignment):
        """Match the finite within-leaf density blend used by actual decoding."""
        xy = self.coordinates_from_encoded(data, logic, regions, assignment)
        blend = float(data.get('density_spread_fraction', .15))
        spread = dispersed_xy(data, assignment)
        xy = (1 - blend) * xy + blend * spread
        fixed = data.get('fixed_xy')
        return torch.where(fixed >= 0, fixed.to(xy), xy) if fixed is not None else xy

    def supervision_loss(self, data, target, *, coordinate_weight=5., capacity_weight=.1,
                         return_details=False):
        meta = validate(data)
        logic, regions = self.encode(data)
        device = logic.device
        teacher = torch.as_tensor(target['teacher_region'], device=device, dtype=torch.long)
        teacher_xy = torch.as_tensor(target['teacher_xy'], device=device, dtype=logic.dtype)
        movable = data.get('movable', torch.ones(len(logic), dtype=torch.bool, device=device)).bool()
        if teacher.shape != (len(logic),) or teacher_xy.shape != (len(logic), 2):
            raise ValueError('teacher shape mismatch')
        if (teacher < 0).any() or (teacher >= len(regions)).any():
            raise ValueError('every teacher, including fixed nodes, needs a valid leaf')
        if not torch.isfinite(teacher_xy).all() or (teacher_xy < 0).any() or (teacher_xy >= 1).any():
            raise ValueError('teacher coordinates must be finite normalized [0,1)')
        if not movable.any():
            raise ValueError('no movable supervised molecules')
        ce = logic.new_zeros(())
        expected_layer = logic.new_zeros(meta[1].shape)
        for start in range(0, len(logic), self.chunk_size):
            nodes = torch.arange(start, min(start + self.chunk_size, len(logic)), device=device)
            active = nodes[movable[nodes]]
            if not len(active):
                continue
            selected = teacher[active]
            if (selected < 0).any() or (selected >= len(regions)).any():
                raise ValueError('teacher leaf outside device')
            def objective(node_logic, device_regions, ids=active, labels=selected):
                logits = self.matching_logits(node_logic, device_regions)
                logits = logits.masked_fill(~compatibility_mask(data, ids, meta), -torch.inf)
                if not torch.isfinite(logits.gather(1, labels[:, None])).all():
                    raise ValueError('teacher violates individual compatibility; derive projected targets first')
                expected = logits.softmax(1).T @ data['demand'][ids]
                per_layer = logits.new_zeros(meta[1].shape)
                per_layer.index_add_(0, meta[0], expected)
                return F.cross_entropy(logits, labels, reduction='sum'), per_layer
            if torch.is_grad_enabled():
                chunk_ce, chunk_expected = checkpoint(objective, logic[active], regions, use_reentrant=False)
            else:
                chunk_ce, chunk_expected = objective(logic[active], regions)
            ce = ce + chunk_ce
            expected_layer = expected_layer + chunk_expected
        ce = ce / movable.sum()
        xy = self.deployment_coordinates(data, logic, regions, teacher)
        span = (data['region_bounds'][teacher][:, [1, 3]] -
                data['region_bounds'][teacher][:, [0, 2]]).clamp_min(1e-6)
        coordinate = F.smooth_l1_loss(xy[movable] / span[movable], teacher_xy[movable] / span[movable])
        fixed = ~movable
        if fixed.any():
            fixed_usage = logic.new_zeros(data['capacity'].shape)
            fixed_usage.index_add_(0, teacher[fixed], data['demand'][fixed])
            expected_layer.index_add_(0, meta[0], fixed_usage)
        overflow = ((expected_layer - meta[1]).clamp_min(0) / meta[1].clamp_min(1.)).square().mean()
        loss = ce + coordinate_weight * coordinate + capacity_weight * overflow
        details = dict(region_ce=float(ce.detach()), coordinate=float(coordinate.detach()),
                       layer_expected_overflow=float(overflow.detach()), supervised=int(movable.sum()))
        return (loss, details) if return_details else loss

    def action_log_probability(self, data, action, *, coordinate_sigma=.2):
        """Score-likelihood surrogate for an evaluated layer/leaf/XY action.

        Hard constrained decoding is not differentiable. These underlying
        joint scores and the deployed XY head receive the feedback gradients;
        no claim of gradients through VPR or exact constrained-policy density.
        """
        if coordinate_sigma <= 0:
            raise ValueError('coordinate_sigma must be positive')
        meta = validate(data)
        logic, regions = self.encode(data)
        assignment = torch.as_tensor(action['assignment'], device=logic.device, dtype=torch.long)
        target_xy = torch.as_tensor(action['xy'], device=logic.device, dtype=logic.dtype)
        if assignment.shape != (len(logic),) or target_xy.shape != (len(logic), 2):
            raise ValueError('feedback action dimensions differ from AP graph')
        if (assignment < 0).any() or (assignment >= len(regions)).any():
            raise ValueError('feedback action region outside device')
        if not torch.isfinite(target_xy).all() or (target_xy < 0).any() or (target_xy >= 1).any():
            raise ValueError('feedback XY must be finite normalized [0,1)')
        movable = data.get('movable', torch.ones(len(logic), device=logic.device, dtype=torch.bool)).bool()
        terms = []
        for start in range(0, len(logic), self.chunk_size):
            nodes = torch.arange(start, min(start + self.chunk_size, len(logic)), device=logic.device)
            nodes = nodes[movable[nodes]]
            if len(nodes):
                def likelihood(node_logic, device_regions, ids=nodes):
                    logits = self.matching_logits(node_logic, device_regions)
                    logits = logits.masked_fill(~compatibility_mask(data, ids, meta), -torch.inf)
                    return logits.log_softmax(1).gather(1, assignment[ids, None]).sum()
                terms.append(checkpoint(likelihood, logic[nodes], regions, use_reentrant=False)
                             if torch.is_grad_enabled() else likelihood(logic[nodes], regions))
        if not terms:
            raise ValueError('no movable action to train')
        xy = self.deployment_coordinates(data, logic, regions, assignment)
        bounds = data['region_bounds'][assignment]
        span = (bounds[:, [1, 3]] - bounds[:, [0, 2]]).clamp_min(1e-6)
        coordinate_ll = -.5 * (((xy[movable] - target_xy[movable]) /
                                (span[movable] * coordinate_sigma)).square()).sum(1).mean()
        result = torch.stack(terms).sum() / movable.sum() + coordinate_ll
        if not torch.isfinite(result):
            raise ValueError('feedback action has invalid/nonfinite likelihood')
        return result

    @torch.no_grad()
    def predict(self, data, *, seed=0, score_noise_std=0., coordinate_noise_std=0.,
                density_passes=2, device_embedding=None, **unused):
        if any(not math.isfinite(v) or v < 0 for v in (score_noise_std, coordinate_noise_std)):
            raise ValueError('candidate noise scales must be finite and nonnegative')
        start = time.perf_counter()
        logic = self.encode_logic(data)
        regions = self.encode_device(data) if device_embedding is None else device_embedding
        query, keys = self.query(logic), self.key(regions)
        encoded = time.perf_counter()
        # Per-node deterministic noise is reproducible across density passes.
        def scores(nodes):
            values = query[nodes] @ keys.T / math.sqrt(self.hidden)
            if score_noise_std:
                generator = torch.Generator(device='cpu').manual_seed(int(seed) + int(nodes[0]) * 1000003)
                values = values + score_noise_std * torch.randn(values.shape, generator=generator).to(values)
            return values
        def coordinates(assignment):
            xy = self.coordinates_from_encoded(data, logic, regions, assignment)
            if coordinate_noise_std:
                generator = torch.Generator(device='cpu').manual_seed(int(seed) ^ 0x5DEECE66D)
                bounds = data['region_bounds'][assignment]
                span = bounds[:, [1, 3]] - bounds[:, [0, 2]]
                low, high = coordinate_interior(bounds)
                noise = torch.randn(xy.shape, generator=generator).to(xy)
                xy = torch.maximum(low, torch.minimum(high, xy + coordinate_noise_std * span * noise))
            return xy
        result = decode(data, scores, density_passes=density_passes, coordinate=coordinates)
        result['phase_seconds'] = dict(encoding=encoded - start,
            decoding=time.perf_counter() - encoded, total=time.perf_counter() - start)
        result['initializer'] = 'gnn'
        return result
