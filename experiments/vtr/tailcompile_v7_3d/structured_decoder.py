"""010 bounded allocation of whole APBlocks to capacity-feasible leaf regions.

This is a five-resource *proxy* certificate, not packing/site legality. Greedy
fragmentation is reported as decoder failure, never as physical infeasibility.
The allocator does not run local search or wait for convergence. Logits are
evaluated in batches, with at most two deterministic ordering attempts.
"""
from __future__ import annotations

import math
import time

import numpy as np
import torch

try:
    from .model import CapacityError, validate, compatibility_mask
except ImportError:
    from model import CapacityError, validate, compatibility_mask

SCHEMA = 'tailcompile-v7-structured-leaf-capacity-010-1'
TOLERANCE = 1e-6


class StructuredCapacityError(CapacityError):
    """A bounded decoder failed; the device may still admit another assignment."""

    def __init__(self, message, *, diagnostics=None):
        super().__init__(message)
        self.diagnostics = diagnostics or {}


def validate_leaf_capacity(data, assignment, *, chunk_size=256):
    """Recompute all region and layer usages; reject a silently overloaded leaf."""
    meta = validate(data)
    raw = torch.as_tensor(assignment, device=data['demand'].device)
    if raw.ndim != 1 or len(raw) != len(data['demand']) or raw.dtype == torch.bool:
        raise ValueError('one integer leaf per whole APBlock required')
    if raw.is_complex() or not torch.isfinite(raw).all() or (raw != raw.floor()).any():
        raise ValueError('leaf assignments must be finite integers')
    leaves = raw.long()
    if (leaves < 0).any() or (leaves >= len(data['capacity'])).any():
        raise ValueError('leaf assignment outside device')
    for start in range(0, len(leaves), chunk_size):
        ids = torch.arange(start, min(start+chunk_size, len(leaves)), device=leaves.device)
        mask = compatibility_mask(data, ids, meta)
        if not mask.gather(1, leaves[ids, None]).all():
            raise StructuredCapacityError('allocation violates an APBlock resource/fixed constraint')
    # CPU float64 sums avoid float32 index_add roundoff on large fractional nets.
    demand = data['demand'].detach().cpu().double()
    capacity = data['capacity'].detach().cpu().double()
    usage = torch.zeros_like(capacity)
    usage.index_add_(0, leaves.cpu(), demand)
    overflow = (usage-capacity).clamp_min(0)
    layer = meta[0][leaves]
    layer_capacity = meta[1].detach().cpu().double()
    layer_usage = torch.zeros_like(layer_capacity)
    layer_usage.index_add_(0, layer.cpu(), demand)
    if (overflow > TOLERANCE).any() or (layer_usage > layer_capacity+TOLERANCE).any():
        raise StructuredCapacityError('allocation exceeds hard leaf/layer-capacity proxy')
    return dict(valid=True, hard_layer_capacity_enforced=True,
                hard_leaf_capacity_enforced=True, packing_legality_claimed=False,
                layer_usage=layer_usage.tolist(), layer_capacity=layer_capacity.tolist(),
                layer_overflow_by_resource=(layer_usage-layer_capacity).clamp_min(0).sum(0).tolist(),
                overflow_by_resource=overflow.sum(0).tolist(),
                overflow_region_count=int((overflow > TOLERANCE).any(1).sum()),
                used_region_count=int(leaves.unique().numel()))


def _numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().double().numpy()
    return np.asarray(value, dtype=np.float64)


@torch.no_grad()
def allocate_regions(data, layer_logits, leaf_logits, *, chunk_size=256,
                     load_weight=.1, max_restarts=2):
    """Allocate atomic APBlocks without leaf or layer proxy overflow.

    ``layer_logits`` is [N,L]. ``leaf_logits`` is [N,R] or a callable taking a
    LongTensor of APBlock ids and returning [batch,R]. Leaf scores are centred
    at the best individually compatible leaf *within* each layer before adding
    layer logits. Equal best-leaf utility therefore does not change merely
    because a layer has more regions. These scores are utilities, not joint
    layer/leaf probabilities or an exact likelihood of the hard allocator.

    Scores combine learned preference and a bounded projected-utilization
    penalty. Compatibility/fixed obligations and resource capacity are hard.
    At most two constrained-first orderings are tried; a failed greedy packing
    raises StructuredCapacityError with diagnostics rather than overloading a
    region, dropping a molecule, silently falling back, or retrying forever.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('positive integer allocation chunk_size required')
    if isinstance(max_restarts, bool) or not isinstance(max_restarts, int) or not 1 <= max_restarts <= 2:
        raise ValueError('max_restarts must be 1 or 2')
    if not math.isfinite(load_weight) or load_weight < 0:
        raise ValueError('nonnegative finite load_weight required')
    started = time.perf_counter()
    meta = validate(data)
    device = data['demand'].device
    n, r, nl = len(data['demand']), len(data['capacity']), len(meta[1])
    if tuple(layer_logits.shape) != (n, nl):
        raise ValueError('layer_logits shape must be [APBlock,layer]')
    if not callable(leaf_logits) and tuple(leaf_logits.shape) != (n, r):
        raise ValueError('leaf_logits must be NxR or logits(nodes)')
    layer_scores = _numpy(layer_logits)
    demand, capacity = _numpy(data['demand']), _numpy(data['capacity'])
    region_layer = meta[0].detach().cpu().numpy()
    if np.any(demand.sum(0) > capacity.sum(0)+TOLERANCE):
        raise StructuredCapacityError('aggregate APBlock demand exceeds device proxy capacity',
                                      diagnostics=dict(reason='aggregate_proxy_capacity', attempts=0))
    counts = np.empty(n, dtype=np.int64)
    score_batches = 0
    for start in range(0, n, chunk_size):
        nodes = torch.arange(start, min(start+chunk_size, n), device=device)
        mask = compatibility_mask(data, nodes, meta).cpu().numpy()
        counts[start:start+len(nodes)] = mask.sum(1)
        for layer in range(nl):
            allowed_layer = mask[:, region_layer == layer].any(1)
            if not np.isfinite(layer_scores[start:start+len(nodes), layer][allowed_layer]).all():
                raise ValueError('nonfinite compatible layer logit')
    scale = np.maximum(capacity.max(0), 1.)
    normalized = demand/scale
    max_pressure, sum_pressure = normalized.max(1), normalized.sum(1)
    ids = np.arange(n)
    orders = [np.lexsort((ids, -max_pressure, counts))]
    if max_restarts == 2:
        orders.append(np.lexsort((ids, -sum_pressure, counts)))
    failures = []
    for attempt, order in enumerate(orders, 1):
        remaining = capacity.copy()
        assignment = np.full(n, -1, dtype=np.int64)
        failed = None
        for start in range(0, n, chunk_size):
            indices = order[start:start+chunk_size]
            nodes = torch.as_tensor(indices.copy(), device=device, dtype=torch.long)
            compatible = compatibility_mask(data, nodes, meta).cpu().numpy()
            scores = _numpy(leaf_logits(nodes) if callable(leaf_logits) else leaf_logits[nodes])
            score_batches += 1
            if scores.shape != (len(indices), r):
                raise ValueError('leaf logits callback returned wrong shape')
            if not np.isfinite(scores[compatible]).all():
                raise ValueError('nonfinite compatible leaf logit')
            scores = np.where(compatible, scores, -np.inf)
            # Conditional best-leaf regret, not a joint probability. A layer
            # with more uniformly scored regions must not lose its layer prior.
            for layer in range(nl):
                cols = np.flatnonzero(region_layer == layer)
                value = scores[:, cols]
                active = compatible[:, cols].any(1)
                if not active.any():
                    continue
                active_value = value[active]
                maximum = active_value.max(1)
                value[active] -= maximum[:, None]
                value[active] += layer_scores[indices[active], layer, None]
                scores[:, cols] = value
            for offset, node in enumerate(indices):
                fits = compatible[offset] & (remaining+TOLERANCE >= demand[node]).all(1)
                if not fits.any():
                    failed = int(node)
                    break
                candidates = np.flatnonzero(fits)
                projected = (capacity[candidates]-remaining[candidates]+demand[node])/np.maximum(capacity[candidates], 1.)
                penalty = (projected*projected).mean(1)
                values = scores[offset, candidates]-load_weight*penalty
                region = int(candidates[int(values.argmax())])
                assignment[node] = region
                remaining[region] -= demand[node]
            if failed is not None:
                break
        if failed is None:
            leaves = torch.as_tensor(assignment, device=device, dtype=torch.long)
            diagnostics = validate_leaf_capacity(data, leaves, chunk_size=chunk_size)
            diagnostics.update(schema=SCHEMA, allocation_attempts=attempt,
                               allowed_allocation_attempts=max_restarts,
                               score_batches=score_batches, score_rows_upper=max_restarts*n,
                               previous_failed_attempts=failures,
                               fallback_used=False, convergence_loop=False)
            usage = torch.as_tensor(capacity-remaining, device=device, dtype=torch.float64)
            layers = meta[0][leaves]
            layer_usage = usage.new_zeros((nl, 5))
            layer_usage.index_add_(0, meta[0], usage)
            return dict(assignment=leaves, layer=layers, usage=usage,
                        layer_usage=layer_usage, capacity_diagnostics=diagnostics,
                        allocation_seconds=time.perf_counter()-started)
        failures.append(dict(attempt=attempt, unassigned_node=failed,
                             assigned_count=int((assignment >= 0).sum())))
    raise StructuredCapacityError('bounded region decoder exhausted ordering attempts; not a physical infeasibility certificate',
        diagnostics=dict(schema=SCHEMA, reason='greedy_fragmentation_or_proxy_capacity',
                         attempts=max_restarts, failures=failures, score_batches=score_batches,
                         fallback_used=False, elapsed_seconds=time.perf_counter()-started))
