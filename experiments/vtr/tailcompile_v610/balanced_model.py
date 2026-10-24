"""Frozen-weight v6.9 inference with bounded repair of *discrete* density.

Softmax expected density may be balanced while its independent argmax puts many
molecules in one leaf.  This module repairs that actual assignment; it does not
train, alter weights, claim packing legality, or run a placement search in VPR.
Each accepted move strictly decreases a dimensionless squared overflow measure.
Matching is chunked and only O(N*K) candidate information is retained.  Candidate
coverage combines learned scores with deterministic stratified compatible leaves
so an exact-score tie does not restrict every molecule to the same first K bins.
Native GP and legalization remain mandatory.
"""
from __future__ import annotations

import math
from pathlib import Path
import sys
import time

import numpy as np
import torch

VTR_DIRECTORY = Path(__file__).resolve().parents[1]
if str(VTR_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(VTR_DIRECTORY))
from tailcompile_v69.fast_model import (  # noqa: E402
    FastHierarchicalInitializer, _validate, coordinate_interior,
)

DECODER_SCHEMA = 'tailcompile-v610-discrete-density-repair-1'
MAX_REPAIR_PASSES = 8
MAX_REPAIR_TOPK = 256


def _normalizer(capacity):
    """Per-region/resource units: rescaling a resource does not alter decisions."""
    scale = capacity.sum(axis=0)
    scale = np.where(scale > 0, scale, 1.)
    return np.where(capacity > 0, capacity, scale[None, :])


def density_statistics(usage, capacity):
    """JSON-safe proxy diagnostics, with no mixed-resource-unit repair objective."""
    usage = np.asarray(usage, dtype=np.float64)
    capacity = np.asarray(capacity, dtype=np.float64)
    overflow = np.maximum(usage - capacity, 0.)
    normalized = overflow / _normalizer(capacity)
    totals = usage.sum(axis=0)
    ordered = np.sort(usage, axis=0)[::-1]
    return dict(
        normalized_squared_overflow=float(np.square(normalized).sum()),
        normalized_squared_overflow_by_resource=np.square(normalized).sum(axis=0).tolist(),
        overflow_by_resource=overflow.sum(axis=0).tolist(),
        overflow_region_count=int(np.any(overflow > 1e-6 * np.maximum(capacity, 1.), axis=1).sum()),
        occupied_region_count=int(np.any(usage > 0, axis=1).sum()),
        top1_demand_fraction_by_resource=np.divide(ordered[0], totals,
            out=np.zeros_like(totals), where=totals > 0).tolist(),
        top3_demand_fraction_by_resource=np.divide(ordered[:3].sum(axis=0), totals,
            out=np.zeros_like(totals), where=totals > 0).tolist(),
        regional_capacity_is_proxy=True, packing_legality_claimed=False,
    )


def repair_assignment(assignment, demand, capacity, nodes, candidate_regions,
                      candidate_scores, centers, *, passes=2, max_moves=None,
                      locality_weight=.1):
    """Deterministic bounded CPU repair of indivisible movable molecules.

    ``nodes`` is the complete movable-node list.  Other assignments, including
    fixed unknown-region -1, are immutable.  Scores only decide between equally
    effective density repairs; locality is a fixed tie/cost preference.  There
    are at most ``passes*N`` visits and ``max_moves`` accepted moves, without
    resident displacement, retry expansion, or a requirement for zero overflow.
    """
    original = np.asarray(assignment, dtype=np.int64)
    a = original.copy()
    d = np.asarray(demand, dtype=np.float64)
    cap = np.asarray(capacity, dtype=np.float64)
    nodes = np.asarray(nodes, dtype=np.int64)
    candidates = np.asarray(candidate_regions, dtype=np.int64)
    scores = np.asarray(candidate_scores, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if d.ndim != 2 or cap.ndim != 2 or d.shape[1] != cap.shape[1]:
        raise ValueError('repair resource dimensions disagree')
    if (a.shape != (len(d),) or candidates.shape != scores.shape or
            candidates.ndim != 2 or len(candidates) != len(nodes) or
            centers.shape != (len(cap), 2)):
        raise ValueError('repair assignment/candidate geometry dimensions disagree')
    if len(np.unique(nodes)) != len(nodes) or np.any((nodes < 0) | (nodes >= len(d))):
        raise ValueError('repair nodes must be unique valid movable indices')
    if np.any((a < -1) | (a >= len(cap))) or np.any(a[nodes] < 0):
        raise ValueError('repair requires valid movable assignments')
    if np.any((candidates < 0) | (candidates >= len(cap))):
        raise ValueError('repair candidate outside region range')
    if not all(np.isfinite(v).all() for v in (d, cap, centers)) or np.isnan(scores).any() or np.isposinf(scores).any():
        raise ValueError('repair inputs must be finite (masked scores may be -inf)')
    if (d < 0).any() or (cap < 0).any():
        raise ValueError('repair resources must be nonnegative')
    if isinstance(passes, bool) or not isinstance(passes, int) or not 0 <= passes <= MAX_REPAIR_PASSES:
        raise ValueError(f'repair_passes must be an integer in [0,{MAX_REPAIR_PASSES}]')
    if max_moves is None:
        max_moves = passes * len(nodes)
    if isinstance(max_moves, bool) or not isinstance(max_moves, int) or max_moves < 0:
        raise ValueError('repair_max_moves must be a nonnegative integer or None')
    if not math.isfinite(locality_weight) or locality_weight < 0:
        raise ValueError('repair_locality_weight must be finite and nonnegative')
    # The outer model validates all individual assignments; keep this public
    # helper safe when used by direct tests or a diagnostic tool as well.
    if np.any(d[nodes] > cap[a[nodes]] + 1e-6):
        raise ValueError('repair starts from individually incompatible assignment')
    usage = np.zeros_like(cap)
    valid = a >= 0
    np.add.at(usage, a[valid], d[valid])
    before = density_statistics(usage, cap)
    initial_usage = usage.copy()
    norm = _normalizer(cap)
    moves = visits = completed_passes = 0
    initial_center = centers[a[nodes]]
    # Large molecules first makes fewer moves necessary.  Resource units are
    # normalized by total supply; node ID provides deterministic final ties.
    totals = cap.sum(axis=0)
    totals = np.where(totals > 0, totals, 1.)
    magnitude = (d[nodes] / totals).max(axis=1) if len(nodes) else np.empty(0)
    order = np.lexsort((nodes, -magnitude))
    reason = 'pass_budget'
    for _ in range(passes):
        pass_moves = 0
        for row in order:
            if moves >= max_moves:
                reason = 'move_budget'
                break
            node = nodes[row]
            source = a[node]
            q = d[node]
            visits += 1
            source_over = np.maximum(usage[source] - cap[source], 0.)
            if not np.any((source_over > 1e-6 * np.maximum(cap[source], 1.)) & (q > 0)):
                continue
            choices = candidates[row]
            compatible = (choices != source) & np.isfinite(scores[row])
            compatible &= np.all(q <= cap[choices] + 1e-6, axis=1)
            if not compatible.any():
                continue
            choices = choices[compatible]
            values = scores[row][compatible]
            source_gain = (np.square(source_over / norm[source]) -
                np.square(np.maximum(usage[source] - q - cap[source], 0.) / norm[source])).sum()
            dest_cost = (np.square(np.maximum(usage[choices] + q - cap[choices], 0.) / norm[choices]) -
                np.square(np.maximum(usage[choices] - cap[choices], 0.) / norm[choices])).sum(axis=1)
            gains = source_gain - dest_cost
            best_gain = float(gains.max())
            tolerance = 1e-12 * max(1., abs(source_gain))
            if best_gain <= tolerance:
                continue
            near_best = gains >= best_gain - tolerance
            choices, values = choices[near_best], values[near_best]
            # Prefer the learned score and short displacement among candidates
            # achieving the same density relief.  No timing/WL guarantee implied.
            cost = -values + locality_weight * np.abs(centers[choices] - initial_center[row]).sum(axis=1)
            selected = np.lexsort((choices, cost))[0]
            destination = int(choices[selected])
            usage[source] -= q
            usage[destination] += q
            a[node] = destination
            moves += 1
            pass_moves += 1
        completed_passes += 1
        if moves >= max_moves:
            break
        if pass_moves == 0:
            reason = 'no_improving_move'
            break
    # Recompute instead of trusting accumulated updates for the acceptance test.
    verified_usage = np.zeros_like(cap)
    valid = a >= 0
    np.add.at(verified_usage, a[valid], d[valid])
    after = density_statistics(verified_usage, cap)
    immutable = np.ones(len(a), dtype=bool)
    immutable[nodes] = False
    invariant = (np.array_equal(a[immutable], original[immutable]) and
                 np.all(d[nodes] <= cap[a[nodes]] + 1e-6))
    improved = after['normalized_squared_overflow'] < before['normalized_squared_overflow'] - 1e-12
    reverted = bool(moves and (not invariant or not improved))
    if reverted:
        a, verified_usage, after = original.copy(), initial_usage, before.copy()
        reason = 'verification_reverted_to_original'
    elif not invariant:
        raise AssertionError('repair violated immutable or resource compatibility invariant')
    return a, verified_usage, dict(before=before, after=after,
        moves_attempted_and_accepted=moves, changed_nodes=int(np.count_nonzero(a != original)),
        node_visits=visits, passes_completed=completed_passes,
        max_moves=max_moves, max_passes=passes,
        reverted_to_original=reverted, termination=reason,
        objective='sum over leaves/resources of squared positive overflow divided by per-resource leaf capacity squared',
        candidate_policy='learned top-k plus deterministic stratified compatible leaves',
        residual_overflow_allowed=True, hard_capacity_enforced=False,
        packing_legality_claimed=False)


class BalancedHierarchicalInitializer(FastHierarchicalInitializer):
    """Same constructor/state_dict as v6.8/v6.9, with inference-only repair."""

    @torch.no_grad()
    def predict(self, data, *, topk=16, device_embedding=None, logic_embedding=None,
                score_noise_std=0., coordinate_noise_std=0., seed=0,
                soft_density_passes=2, density_price_weight=1., density_step=1.,
                repair_passes=2, repair_topk=32, repair_max_moves=None,
                repair_locality_weight=.1):
        if isinstance(repair_topk, bool) or not isinstance(repair_topk, int) or not 1 <= repair_topk <= MAX_REPAIR_TOPK:
            raise ValueError(f'repair_topk must be an integer in [1,{MAX_REPAIR_TOPK}]')
        if isinstance(repair_passes, bool) or not isinstance(repair_passes, int) or not 0 <= repair_passes <= MAX_REPAIR_PASSES:
            raise ValueError(f'repair_passes must be an integer in [0,{MAX_REPAIR_PASSES}]')
        if repair_max_moves is not None and (isinstance(repair_max_moves, bool) or
                not isinstance(repair_max_moves, int) or repair_max_moves < 0):
            raise ValueError('repair_max_moves must be a nonnegative integer or None')
        if not math.isfinite(repair_locality_weight) or repair_locality_weight < 0:
            raise ValueError('repair_locality_weight must be finite and nonnegative')
        device = next(self.parameters()).device
        def sync():
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
        sync()
        begin = time.perf_counter()
        extra = {}
        def timed(name, fn):
            sync()
            start = time.perf_counter()
            result = fn()
            sync()
            extra[name] = time.perf_counter() - start
            return result
        movable, _ = timed('repair_validation', lambda: _validate(data, device))
        logic = (timed('encoding_logic_extra', lambda: self.encode_logic(data))
                 if logic_embedding is None else logic_embedding)
        regions = (timed('encoding_device_extra', lambda: self.encode_device(data))
                   if device_embedding is None else device_embedding)
        result = super().predict(data, topk=topk, device_embedding=regions,
            logic_embedding=logic, score_noise_std=score_noise_std,
            coordinate_noise_std=coordinate_noise_std, seed=seed,
            soft_density_passes=soft_density_passes, density_price_weight=density_price_weight,
            density_step=density_step)
        original = result['assignment'].numpy().copy()
        demand = data['demand'].detach().cpu().double().numpy()
        capacity = data['capacity'].detach().cpu().double().numpy()
        nodes = torch.where(movable)[0]
        centers = (data['region_bounds'][:, [0, 2]] + data['region_bounds'][:, [1, 3]]) / 2

        def candidates():
            k = min(repair_topk, len(regions))
            learned_k = (k + 1) // 2
            spread_k = k - learned_k
            ids, values = [], []
            query, key = self.query(logic), self.key(regions)
            rng = torch.Generator(device='cpu').manual_seed(int(seed))
            for start in range(0, len(nodes), self.chunk_size):
                batch = nodes[start:start+self.chunk_size]
                logits = query[batch] @ key.T / math.sqrt(self.hidden)
                if score_noise_std:
                    logits += score_noise_std * torch.randn(logits.shape, generator=rng).to(logits)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError('nonfinite frozen scores during density repair')
                feasible = torch.ones_like(logits, dtype=torch.bool)
                for resource in range(data['demand'].shape[1]):
                    feasible &= data['demand'][batch, resource, None] <= data['capacity'][None, :, resource] + 1e-6
                masked = logits.masked_fill(~feasible, -torch.inf)
                # topk is deterministic on a fixed device/runtime; cross-device
                # bit identity is not promised (matching v69's contract).
                learned = masked.topk(learned_k, dim=1).indices
                if spread_k:
                    # Build compact ordered feasible ranks without sorting scores.
                    rows, cols = torch.where(feasible)
                    ranks = feasible.long().cumsum(1)[rows, cols] - 1
                    ordered = torch.zeros_like(feasible, dtype=torch.long)
                    ordered[rows, ranks] = cols
                    counts = feasible.sum(1)
                    if (counts == 0).any():
                        raise AssertionError('base decoder accepted an incompatible movable molecule')
                    offset = (batch * 2654435761 + 1013904223) % counts
                    quantiles = (torch.arange(spread_k, device=device)[None, :] * counts[:, None]) // spread_k
                    rank = (offset[:, None] + quantiles) % counts[:, None]
                    spread = ordered.gather(1, rank)
                    chosen = torch.cat((learned, spread), dim=1)
                else:
                    chosen = learned
                ids.append(chosen.cpu())
                values.append(masked.gather(1, chosen).cpu())
            return (torch.cat(ids).numpy(), torch.cat(values).double().numpy()) if len(nodes) else (
                np.empty((0, k), dtype=np.int64), np.empty((0, k), dtype=np.float64))

        # A disabled repair is an exact v69 control and avoids extra matching.
        if repair_passes and repair_max_moves != 0 and result['capacity_diagnostics']['overflow_region_count']:
            candidate_ids, candidate_scores = timed('repair_matching', candidates)
        else:
            candidate_ids = original[nodes.cpu().numpy(), None]
            candidate_scores = np.zeros_like(candidate_ids, dtype=np.float64)
        assignment, usage, repair = timed('discrete_density_repair', lambda: repair_assignment(
            original, demand, capacity, nodes.cpu().numpy(), candidate_ids, candidate_scores,
            centers.detach().cpu().double().numpy(), passes=repair_passes,
            max_moves=repair_max_moves, locality_weight=repair_locality_weight))

        def coordinates():
            final = torch.from_numpy(assignment).to(device)
            xy = self.coordinates_from_encoded(data, logic, regions, final)
            if coordinate_noise_std:
                rng = torch.Generator(device='cpu').manual_seed(int(seed) ^ 0x5DEECE66D)
                bounds = data['region_bounds'][final.clamp_min(0)]
                safe_low, safe_high = coordinate_interior(bounds)
                noise = torch.randn(xy.shape, generator=rng).to(xy)
                proposed = xy + coordinate_noise_std * (bounds[:, [1, 3]] - bounds[:, [0, 2]]) * noise
                proposed = torch.maximum(safe_low.to(xy), torch.minimum(safe_high.to(xy), proposed))
                xy = torch.where(movable[:, None], proposed, xy)
            if not torch.isfinite(xy).all():
                raise FloatingPointError('repaired coordinate head output is not finite')
            if (~movable).any() and not torch.equal(xy[~movable], data['fixed_xy'][~movable]):
                raise AssertionError('density repair moved fixed coordinates')
            return xy
        if repair['changed_nodes']:
            result['xy'] = timed('repair_coordinate', coordinates)
        result['capacity_diagnostics_before'] = dict(result['capacity_diagnostics'])
        result['assignment'] = torch.from_numpy(assignment)
        result['usage'] = torch.from_numpy(usage)
        overflow = np.maximum(usage - capacity, 0.)
        quality = dict(result['capacity_diagnostics'])
        quality.update(overflow_by_resource=overflow.sum(0).tolist(),
            total_capacity_overflow=float(overflow.sum()),
            max_capacity_overflow_ratio=float((overflow / np.maximum(capacity, 1.)).max()),
            overflow_region_count=repair['after']['overflow_region_count'],
            normalized_squared_overflow=repair['after']['normalized_squared_overflow'],
            occupied_region_count=repair['after']['occupied_region_count'])
        result['capacity_diagnostics'] = quality
        result['repair_diagnostics'] = repair
        result['decoder_kind'] = DECODER_SCHEMA
        result['repair_passes'] = repair_passes
        result['repair_topk'] = repair_topk
        result['repair_locality_weight'] = repair_locality_weight
        seconds = dict(result['phase_seconds'])
        seconds['encoding_logic'] += extra.pop('encoding_logic_extra', 0.)
        seconds['encoding_device'] += extra.pop('encoding_device_extra', 0.)
        seconds.update(extra)
        sync()
        seconds['total'] = time.perf_counter() - begin
        result['phase_seconds'] = seconds
        repair['seconds_including_candidate_matching'] = sum(extra.get(k, 0.) for k in (
            'repair_matching', 'discrete_density_repair', 'repair_coordinate'))
        return result
