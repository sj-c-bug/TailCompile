"""Bounded-cost soft-density decoder for the existing hierarchical generator.

Weights, hierarchy, per-region matching and the coordinate head are unchanged
from v6.8.  Only inference decoding changes.  A movable AP molecule chooses an
individually resource-compatible leaf; aggregate leaf capacity is a SOFT target,
not a hard legality claim.  Native VPR GP and legalization must still run.

The old decoder sorted every node's candidates, tried resident displacements,
and retried with wider candidate lists.  This decoder instead does a fixed
number of chunked matrix passes, then an argmax.  Memory is O(chunk_size * R),
in addition to embeddings, assignments and R * resource_count bookkeeping.
There is no node-by-site matrix, resident scan or combinatorial repair loop.

This is an intervention that requires VPR evaluation, not a proven speedup or
quality improvement.  Even price-balanced expected demand can overload after
discrete selection; actual overflow is always returned explicitly.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import torch

# Supports both direct script imports and experiments.vtr namespace imports.
VTR_DIRECTORY = Path(__file__).resolve().parents[1]
if str(VTR_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VTR_DIRECTORY))
from tailcompile_v68.model import (  # noqa: E402
    CapacityError, HierarchicalInitializer, SCHEMA as CHECKPOINT_SCHEMA,
    coordinate_interior, to_device,
)

DECODER_SCHEMA = 'tailcompile-v69-soft-density-decoder-1'
MAX_DENSITY_PASSES = 8


def _validate(data, device):
    """Validate geometry/resources before soft decoding, without changing inputs."""
    demand, capacity, bounds = data['demand'], data['capacity'], data['region_bounds']
    if demand.ndim != 2 or capacity.ndim != 2:
        raise ValueError('demand and capacity must be matrices')
    n, r = len(demand), len(capacity)
    if not n or not r or demand.shape[1] != capacity.shape[1]:
        raise ValueError('nonempty matching node/region/resource dimensions required')
    if len(data['cluster_x']) != n or len(data['region_x']) != r or bounds.shape != (r, 4):
        raise ValueError('feature or region-bound dimensions disagree with resources')
    for name in ('demand', 'capacity', 'region_bounds', 'cluster_x', 'region_x',
                 'pool_x', 'logic_weight', 'region_weight', 'logic_criticality'):
        if name not in data:
            continue
        tensor = data[name]
        if tensor.device != device:
            raise ValueError(f'{name} is on {tensor.device}, expected {device}')
        if not torch.isfinite(tensor).all():
            raise ValueError(f'{name} must be finite')
    if (demand < 0).any() or (capacity < 0).any():
        raise ValueError('demand and capacity must be nonnegative')
    if ((bounds < 0) | (bounds > 1)).any():
        raise ValueError('region bounds must use normalized global coordinates')
    coordinate_interior(bounds)  # Includes positive representable-width check.
    width = torch.as_tensor(data.get('channel_width', 100.), device=device)
    if width.numel() != 1 or not torch.isfinite(width).all() or (width < 0).any():
        raise ValueError('channel_width must be one finite nonnegative value')
    movable = data.get('movable', torch.ones(n, device=device, dtype=torch.bool)).bool()
    assignment = data.get('fixed_region', torch.full((n,), -1, device=device,
                                                    dtype=torch.long)).long().clone()
    if movable.shape != (n,) or assignment.shape != (n,):
        raise ValueError('movable and fixed_region must have one entry per molecule')
    if movable.device != device or assignment.device != device:
        raise ValueError('fixed/movable metadata must be on the model device')
    if ((assignment < -1) | (assignment >= r)).any():
        raise ValueError('fixed_region outside device')
    if 'fixed_xy' in data:
        xy = data['fixed_xy']
        if xy.device != device or xy.shape != (n, 2) or not torch.isfinite(xy).all():
            raise ValueError('fixed_xy must be finite Nx2 on the model device')
        # The deployment graph uses [-1, -1] placeholders for movable nodes:
        # these positions are predicted below, not supplied fixed locations.
        # Still validate shape/device/finiteness for the entire tensor, but
        # require normalized coordinates only where positions must be kept.
        fixed_xy = xy[~movable]
        if ((fixed_xy < 0) | (fixed_xy > 1)).any():
            raise ValueError('fixed_xy for nonmovable molecules must use normalized global coordinates')
    elif (~movable).any():
        raise ValueError('fixed_xy is required to preserve nonmovable molecules')
    assignment[movable] = -1
    return movable, assignment


class FastHierarchicalInitializer(HierarchicalInitializer):
    """Drop-in v6.8 checkpoint-compatible generator with soft density decoding.

    Constructor/config/state_dict are inherited unchanged.  The inherited loss
    already trains with an expected-overflow penalty, rather than invoking hard
    capacity decoding.  Existing teacher and real-feedback training can thus
    continue to update exactly these deployed weights.
    """

    @torch.no_grad()
    def predict(self, data, *, topk=16, device_embedding=None, logic_embedding=None,
                score_noise_std=0., coordinate_noise_std=0., seed=0,
                soft_density_passes=2, density_price_weight=1., density_step=1.):
        """Resource-compatible independent assignment with bounded density prices.

        ``topk`` is accepted for v6.8 call compatibility, but no top-k sorting or
        restricted search is used: each chunk examines every device leaf once
        per pass.  ``soft_density_passes=0`` is direct compatible argmax.

        Price passes use expected softmax demand and projected nonnegative
        resource prices.  A final argmax emits one leaf per whole molecule.
        Unknown-region fixed demand is reported, not silently counted as zero
        known demand.  Fixed positions/assignments are never changed.

        Timings synchronize CUDA at phase boundaries and include the complete
        decode wall time in ``total``; nothing here excludes decoding overhead.
        As in v6.8, assignment and usage are CPU tensors; xy stays on the model
        device.  Noisy train-only candidates repeat deterministically on the
        same device/config/seed.  Cross-device bit identity is not promised.
        """
        device = next(self.parameters()).device

        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)

        sync()
        started = time.perf_counter()
        seconds = {name: 0. for name in ('validation', 'encoding_logic',
                   'encoding_device', 'matching', 'soft_density', 'coordinate', 'diagnostics')}

        def timed(name, function):
            sync()
            begin = time.perf_counter()
            result = function()
            sync()
            seconds[name] += time.perf_counter() - begin
            return result

        if (isinstance(soft_density_passes, bool) or
                not isinstance(soft_density_passes, int) or
                not 0 <= soft_density_passes <= MAX_DENSITY_PASSES):
            raise ValueError(f'soft_density_passes must be an integer in [0,{MAX_DENSITY_PASSES}]')
        for name, value in (('score_noise_std', score_noise_std),
                            ('coordinate_noise_std', coordinate_noise_std),
                            ('density_price_weight', density_price_weight),
                            ('density_step', density_step)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if not isinstance(topk, int) or topk < 1:
            raise ValueError('compatibility topk must be a positive integer')
        movable, assignment = timed('validation', lambda: _validate(data, device))
        logic = (timed('encoding_logic', lambda: self.encode_logic(data))
                 if logic_embedding is None else logic_embedding)
        regions = (timed('encoding_device', lambda: self.encode_device(data))
                   if device_embedding is None else device_embedding)
        demand, capacity = data['demand'], data['capacity']

        def validate_embeddings():
            for name, value, count in (('logic', logic, len(demand)),
                                       ('device', regions, len(capacity))):
                if value.device != device or value.shape != (count, self.hidden):
                    raise ValueError(f'{name} embedding shape/device mismatch')
                if not torch.isfinite(value).all():
                    raise FloatingPointError(f'{name} embedding is not finite')
        timed('validation', validate_embeddings)

        # Both projections are computed once, not once for every chunk/pass.
        query, keys = timed('matching', lambda: (self.query(logic), self.key(regions)))
        fixed_known = ~movable & (assignment >= 0)
        fixed_usage = torch.zeros_like(capacity)
        if fixed_known.any():
            fixed_usage.index_add_(0, assignment[fixed_known], demand[fixed_known])
        prices = torch.zeros_like(capacity)
        normalizer = capacity.clamp_min(1.)
        active_nodes = torch.where(movable)[0]

        def logits_for(nodes, generator):
            values = query[nodes] @ keys.T / math.sqrt(self.hidden)
            if not torch.isfinite(values).all():
                raise FloatingPointError('nonfinite matching logits before resource masking')
            if density_price_weight:
                values = values - density_price_weight * (demand[nodes] @ (prices / normalizer).T)
            if score_noise_std:
                # CPU-generated perturbations repeat across passes; changing a
                # price must not simultaneously resample the candidate noise.
                values = values + score_noise_std * torch.randn(
                    values.shape, generator=generator).to(values)
            if not torch.isfinite(values).all():
                raise FloatingPointError('nonfinite adjusted matching logits')
            feasible = torch.ones_like(values, dtype=torch.bool)
            for resource in range(demand.shape[1]):
                feasible &= demand[nodes, resource, None] <= capacity[None, :, resource] + 1e-6
            absent = ~feasible.any(1)
            if absent.any():
                node = int(nodes[torch.where(absent)[0][0]])
                raise CapacityError(f'node {node}: no individually feasible device region')
            return values.masked_fill(~feasible, -torch.inf)

        # Bounded density-price iterations; never retry on aggregate overflow.
        for pass_index in range(soft_density_passes + 1):
            is_final = pass_index == soft_density_passes
            expected = fixed_usage.clone()
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
            for start in range(0, len(active_nodes), self.chunk_size):
                nodes = active_nodes[start:start + self.chunk_size]
                logits = timed('matching', lambda: logits_for(nodes, generator))
                if is_final:
                    # Argmax resolves exact ties to the lower leaf index; no
                    # O(R log R) full sorting is necessary.
                    assignment[nodes] = timed('matching', lambda: logits.argmax(-1))
                else:
                    expected.add_(timed('soft_density', lambda: logits.softmax(-1).T @ demand[nodes]))
            if not is_final:
                def update_prices():
                    # Clamp avoids numerical blow-ups without a hidden retry.
                    return (prices + density_step * (expected - capacity) / normalizer).clamp(0., 100.)
                prices = timed('soft_density', update_prices)

        def coordinates():
            xy = self.coordinates_from_encoded(data, logic, regions, assignment)
            if coordinate_noise_std:
                generator = torch.Generator(device='cpu').manual_seed(int(seed) ^ 0x5DEECE66D)
                bounds = data['region_bounds'][assignment.clamp_min(0)]
                low, high = bounds[:, [0, 2]], bounds[:, [1, 3]]
                safe_low, safe_high = coordinate_interior(bounds)
                noise = torch.randn(xy.shape, generator=generator).to(xy)
                proposal = xy + coordinate_noise_std * (high - low) * noise
                proposal = torch.maximum(safe_low.to(xy), torch.minimum(safe_high.to(xy), proposal))
                xy = torch.where(movable[:, None], proposal, xy)
            if not torch.isfinite(xy).all():
                raise FloatingPointError('coordinate head emitted nonfinite positions')
            if (~movable).any() and not torch.equal(xy[~movable], data['fixed_xy'][~movable]):
                raise AssertionError('fixed coordinates changed during initialization')
            return xy
        xy = timed('coordinate', coordinates)

        def diagnostics():
            valid = assignment >= 0
            usage = torch.zeros(capacity.shape, device=device, dtype=torch.float64)
            usage.index_add_(0, assignment[valid], demand[valid].double())
            caps = capacity.double()
            overflow = (usage - caps).clamp_min(0)
            tolerance = 1e-6 * caps.abs().clamp_min(1.)
            return usage.cpu(), {
                'overflow_by_resource': overflow.sum(0).cpu().tolist(),
                'total_capacity_overflow': float(overflow.sum()),
                'max_capacity_overflow_ratio': float((overflow / caps.clamp_min(1.)).max()),
                'overflow_region_count': int((overflow > tolerance).any(1).sum()),
                'fixed_region_unknown_count': int((~movable & (assignment < 0)).sum()),
                'movable_individual_resource_compatibility_enforced': True,
                'regional_capacity_is_proxy': True,
                'hard_regional_capacity_enforced': False,
                'packing_legality_claimed': False,
            }
        usage, quality = timed('diagnostics', diagnostics)
        cpu_assignment = assignment.cpu()
        sync()
        seconds['total'] = time.perf_counter() - started
        return dict(assignment=cpu_assignment, xy=xy, usage=usage,
                    candidate_region_count=len(capacity), regional_capacity_is_proxy=True,
                    decoder_kind=DECODER_SCHEMA, hard_regional_capacity_enforced=False,
                    soft_density_passes=soft_density_passes,
                    density_price_weight=density_price_weight, density_step=density_step,
                    capacity_diagnostics=quality, phase_seconds=seconds)
