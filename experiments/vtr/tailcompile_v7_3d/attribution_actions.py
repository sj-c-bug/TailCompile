"""015 exact weak-action decomposition and bounded, matched random controls.

The generator never queries measured QoR. D and J retain the original 011
coordinates byte-for-byte; L/Y project that one J, rather than selecting a
different set of nodes. Capacity checks certify APBlock demand proxies only;
VPR still performs physical packing and legalization.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import math
import time

import numpy as np
import torch

try:
    from .aligned_model import action_digest
    from .aligned_teacher import constraints_digest
    from .model import validate, compatibility_mask
    from .residual_actions import (make_candidates, validate_residual_action,
        SCHEMA, DECODER_SCHEMA, COORDINATE_CONTRACT)
    from .structured_decoder import TOLERANCE
except ImportError:
    from aligned_model import action_digest
    from aligned_teacher import constraints_digest
    from model import validate, compatibility_mask
    from residual_actions import (make_candidates, validate_residual_action,
        SCHEMA, DECODER_SCHEMA, COORDINATE_CONTRACT)
    from structured_decoder import TOLERANCE

METHODS = ('D', 'L', 'Y', 'J', 'R1', 'R2')
RANDOM_SEEDS = (15021, 15022)
GENERATOR = dict(strengths=(.15, .35), max_moves=256, fraction=.02, seed=11301)
MATCH_RTOL = 2e-4
MATCH_ATOL = 2e-7


def _array(value, dtype=None):
    value = torch.as_tensor(value).detach().cpu()
    return value.numpy().astype(dtype, copy=True) if dtype is not None else value.numpy().copy()


def _counts(base, action):
    layers = action['layer'] != base['layer']
    xy = (action['xy'] != base['xy']).any(1)
    return dict(changed_nodes=int((layers | xy).sum()), changed_layers=int(layers.sum()),
                changed_xy=int(xy.sum()), changed_both=int((layers & xy).sum()))


def _stamp(data, base, prototype, assignment, layer, xy, name, arm):
    action = copy.deepcopy(prototype)
    device = data['demand'].device
    action.update(assignment=torch.as_tensor(assignment, device=device, dtype=torch.long).clone(),
        layer=torch.as_tensor(layer, device=device, dtype=torch.long).clone(),
        xy=torch.as_tensor(xy, device=device, dtype=data['demand'].dtype).clone(),
        schema=SCHEMA, decoder_kind=DECODER_SCHEMA, coordinate_contract=COORDINATE_CONTRACT,
        candidate_id=name, initializer='attribution_'+name, residual_arm=arm,
        base_action_sha256=action_digest(base), constraints_sha256=constraints_digest(data),
        tail_weight=0., risk_diagnostics_computed=False)
    action['deployed_action_sha256'] = action_digest(action)
    action['capacity_diagnostics'] = validate_residual_action(data, action, base_action=base, arm=arm)
    action['residual_diagnostics'].update(_counts(base, action))
    return action


def _random_delta(rng, position, low, high, free, length):
    """Finite random directions, then a feasible corner-ray fallback; no shrink."""
    if length == 0:
        return np.zeros(2, dtype=np.float64)
    axes = np.flatnonzero(free)
    if not len(axes):
        return None
    candidates = []
    if len(axes) == 1:
        for sign in rng.permutation(np.array([-1., 1.])):
            delta = np.zeros(2); delta[axes[0]] = sign*length
            candidates.append(delta)
    else:
        for angle in rng.uniform(0., 2*math.pi, 32):
            candidates.append(length*np.array([math.cos(angle), math.sin(angle)]))
        # If a narrow boundary cone defeats the finite angle sample, a ray
        # toward a sufficiently distant corner preserves the requested norm.
        corners = np.array([[low[0], low[1]], [low[0], high[1]],
                            [high[0], low[1]], [high[0], high[1]]])
        for corner in corners[rng.permutation(4)]:
            vector = corner-position
            distance = float(np.linalg.norm(vector))
            if distance >= length and distance > 0:
                candidates.append(vector*(length/distance))
    for delta in candidates:
        point = position+delta
        if (point >= low).all() and (point <= high).all():
            return delta
    return None


def _random_control(data, base, joint, name, seed):
    """Match J's edit templates in random eligible nodes, with truthful misses.

    Random membership is stratified by exact resource demand, original layer
    and free-axis mask. This also matches moved resource mass when all templates
    fit. Two-layer targets necessarily flip to the only other layer; the null
    randomizes membership and XY direction, not an impossible third layer.
    Each source stratum has a bounded random pool. No retry/convergence search.
    """
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    action_dtype = data['demand'].detach().cpu().numpy().dtype
    demand = _array(data['demand'], np.float64)
    capacity = _array(data['capacity'], np.float64)
    bounds = _array(data['region_bounds'], np.float64)
    region_layers = _array(data['region_layer'], np.int64)
    leaves = _array(base['assignment'], np.int64)
    layers = _array(base['layer'], np.int64)
    xy = _array(base['xy'], np.float64)
    original_leaves, original_layers, original_xy = leaves.copy(), layers.copy(), xy.copy()
    joint_layers, joint_xy = _array(joint['layer'], np.int64), _array(joint['xy'], np.float64)
    movable = _array(data.get('movable', torch.ones(len(demand), dtype=torch.bool)), bool)
    fixed = _array(data.get('fixed_xy', torch.full((len(demand), 2), -1.)), np.float64)
    free = fixed < 0
    layer_changed = joint_layers != layers
    xy_changed = (joint_xy != xy).any(1)
    changed = np.flatnonzero(layer_changed | xy_changed)
    requested = _counts(base, joint)
    lengths = np.linalg.norm(joint_xy-xy, axis=1)
    # This mask/metadata is computed once. Candidate compatibility is batched
    # below rather than dispatching a Torch operation for every proposal.
    metadata = validate(data)
    def key(i):
        return (int(original_layers[i]), tuple(demand[i]), tuple(free[i]))
    groups = defaultdict(list)
    for i in np.flatnonzero(movable):
        groups[key(i)].append(int(i))
    needs = Counter(key(int(i)) for i in changed)
    pools = {k: rng.permutation(groups.get(k, []))[:max(64, 8*count)].tolist()
             for k, count in needs.items()}
    nodes = sorted({i for pool in pools.values() for i in pool})
    compatible = {}
    for start in range(0, len(nodes), 256):
        batch = nodes[start:start+256]
        ids = torch.tensor(batch, device=data['demand'].device, dtype=torch.long)
        rows = compatibility_mask(data, ids, metadata).detach().cpu().numpy()
        compatible.update(zip(batch, rows))
    usage = np.zeros_like(capacity)
    np.add.at(usage, leaves, demand)
    layer_capacity = np.zeros((int(region_layers.max())+1, 5))
    np.add.at(layer_capacity, region_layers, capacity)
    layer_usage = np.zeros_like(layer_capacity)
    np.add.at(layer_usage, layers, demand)
    upper = float(torch.nextafter(torch.tensor(1., dtype=data['demand'].dtype),
                                  torch.tensor(0., dtype=data['demand'].dtype)))
    chosen, missing, assignments, norm_errors = set(), [], [], []
    considered = 0
    # Scarce strata first; a fixed random tie order does not consult connectivity.
    shuffled = rng.permutation(changed).tolist()
    ordered = sorted(shuffled, key=lambda i: (len(pools[key(i)]), not bool(layer_changed[i])))
    for template in ordered:
        success = False
        for node in pools[key(template)]:
            if node in chosen:
                continue
            considered += 1
            old_leaf, old_layer = int(leaves[node]), int(layers[node])
            if layer_changed[template]:
                same_xy = ((xy[node] >= bounds[:, [0, 2]]-TOLERANCE) &
                           (xy[node] <= bounds[:, [1, 3]]+TOLERANCE)).all(1)
                fits = (usage+demand[node] <= capacity+TOLERANCE).all(1)
                fits &= (layer_usage[region_layers]+demand[node] <= layer_capacity[region_layers]+TOLERANCE).all(1)
                targets = np.flatnonzero(compatible[node] & same_xy & fits & (region_layers != old_layer))
                targets = rng.permutation(targets)
            else:
                targets = [old_leaf]
            for target in targets:
                low = bounds[target, [0, 2]]
                high = np.minimum(bounds[target, [1, 3]], upper)
                delta = (_random_delta(rng, xy[node], low, high, free[node], float(lengths[template]))
                         if xy_changed[template] else np.zeros(2))
                if delta is None:
                    continue
                # Match after the same precision cast used in the native action.
                point = (xy[node]+delta).astype(action_dtype).astype(np.float64)
                actual_length = float(np.linalg.norm(point-xy[node]))
                actual_xy = bool((point != original_xy[node]).any())
                if actual_xy != bool(xy_changed[template]) or not np.isclose(
                        actual_length, lengths[template], rtol=MATCH_RTOL, atol=MATCH_ATOL):
                    continue
                if ((point < low-TOLERANCE) | (point > high+TOLERANCE)).any():
                    continue
                if (point[~free[node]] != fixed[node, ~free[node]]).any():
                    continue
                leaves[node], layers[node], xy[node] = int(target), int(region_layers[target]), point
                if layer_changed[template]:
                    usage[old_leaf] -= demand[node]; usage[target] += demand[node]
                    layer_usage[old_layer] -= demand[node]; layer_usage[layers[node]] += demand[node]
                chosen.add(node)
                assignments.append(dict(template_node=int(template), random_node=int(node)))
                norm_errors.append(abs(actual_length-float(lengths[template])))
                success = True
                break
            if success:
                break
        if not success:
            missing.append(int(template))
    result = _stamp(data, base, joint, leaves, layers, xy, name, 'J')
    achieved = _counts(base, result)
    requested_mass = demand[changed].sum(0)
    achieved_mass = demand[sorted(chosen)].sum(0) if chosen else np.zeros(5)
    matching = dict(requested=requested, achieved=achieved, requested_templates=len(changed),
        matched_templates=len(assignments), unmatched_template_nodes=missing,
        matched_control=not missing and requested==achieved and bool(np.allclose(requested_mass, achieved_mass)),
        requested_resource_mass=requested_mass.tolist(), achieved_resource_mass=achieved_mass.tolist(),
        xy_l2_norm_tolerance=dict(relative=MATCH_RTOL, absolute=MATCH_ATOL),
        maximum_xy_l2_norm_error=max(norm_errors, default=0.),
        template_assignments=assignments, overlap_with_joint_nodes=len(chosen.intersection(changed.tolist())),
        bounded_pool_nodes=len(nodes), proposals_considered=considered,
        strata='exact demand vector, original layer, free XY axes',
        mismatch_reason='bounded eligible/capacity/geometry search could not match every template' if missing else None)
    result['attribution_diagnostics'] = dict(schema='tailcompile-attribution-action-015-1',
        method=name, role='matched random joint control', random_seed=seed,
        matched_control=matching['matched_control'], matching=matching,
        generation_seconds=time.perf_counter()-started, no_qor_labels=True,
        packing_legality_claimed=False, convergence_loop=False)
    return result


@torch.no_grad()
def make_attribution_candidates(graph, base_action):
    """Return D/L/Y/J/R1/R2; L and Y project the *same* frozen weak J action."""
    if graph['demand'].device.type != 'cpu':
        raise ValueError('015 frozen candidate generation requires CPU tensors')
    bank = make_candidates(graph, base_action, 'J', **GENERATOR)
    identity = bank[0]
    joint = next((a for a in bank if a['candidate_id']=='j_s1'), identity)
    result = dict(D=copy.deepcopy(identity))
    try:
        result['L'] = _stamp(graph, base_action, joint, joint['assignment'], joint['layer'],
                             base_action['xy'], 'L', 'L')
        result['Y'] = _stamp(graph, base_action, joint, base_action['assignment'], base_action['layer'],
                             joint['xy'], 'Y', 'Y')
    except ValueError as exc:
        raise ValueError('015 weak J cannot be exactly decomposed into valid L/Y; no silent repair: '+str(exc)) from exc
    result['J'] = copy.deepcopy(joint)
    for name in ('D', 'L', 'Y', 'J'):
        result[name]['attribution_diagnostics'] = dict(schema='tailcompile-attribution-action-015-1',
            method=name, source_joint_sha256=action_digest(joint),
            source_base_sha256=action_digest(base_action), no_qor_labels=True,
            counts=_counts(base_action, result[name]),
            exact_component_projection=True, packing_legality_claimed=False, convergence_loop=False)
    for name, seed in zip(('R1', 'R2'), RANDOM_SEEDS):
        result[name] = _random_control(graph, base_action, joint, name, seed)
    if tuple(result) != METHODS or action_digest(result['D']) != action_digest(base_action):
        raise AssertionError('015 method order or D identity changed')
    return result
