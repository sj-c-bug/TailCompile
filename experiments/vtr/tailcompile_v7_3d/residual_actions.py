"""011 bounded, resource-feasible edits of a frozen 010 D initialization.

Candidates use only the logical graph, device and D action. Their geometric
and connectivity statistics are features, never measured QoR labels. The new
coordinate contract is normalized leaf containment plus fixed axes: the old
shared-head density-blend reachable interval is not a physical constraint and
is not imposed on a residual coordinate edit. Physical packing remains VPR's
responsibility. No local convergence loop or repeated full decoder is used.
"""
from __future__ import annotations

import math
import numpy as np
import torch

try:
    from .model import validate, compatibility_mask
    from .aligned_model import action_digest
    from .aligned_teacher import constraints_digest
    from .structured_model import validate_structured_action
    from .structured_decoder import validate_leaf_capacity, TOLERANCE
except ImportError:
    from model import validate, compatibility_mask
    from aligned_model import action_digest
    from aligned_teacher import constraints_digest
    from structured_model import validate_structured_action
    from structured_decoder import validate_leaf_capacity, TOLERANCE

SCHEMA = 'tailcompile-v7-residual-action-011-1'
DECODER_SCHEMA = 'bounded-feasible-residual-011-1'
ARMS = ('L', 'Y', 'J')
COORDINATE_CONTRACT = 'normalized leaf containment and fixed axes; no shared-head blend-reach restriction'
FEATURE_NAMES = ('identity', 'strength', 'layer_edit_fraction', 'xy_edit_fraction',
    'any_edit_fraction', 'mean_xy_displacement', 'max_xy_displacement',
    'moved_lut_mass', 'moved_ff_mass', 'moved_memory_mass', 'moved_dsp_mass', 'moved_io_mass',
    'base_xy_length_proxy', 'xy_length_proxy_delta', 'base_critical_xy_proxy',
    'critical_xy_proxy_delta', 'base_cross_layer_proxy', 'cross_layer_proxy_delta',
    'base_critical_cross_layer_proxy', 'critical_cross_layer_proxy_delta',
    'base_peak_leaf_utilization', 'peak_leaf_utilization_delta',
    'base_lut_layer_imbalance', 'lut_layer_imbalance_delta',
    'base_ff_layer_imbalance', 'ff_layer_imbalance_delta',
    'base_memory_layer_imbalance', 'memory_layer_imbalance_delta',
    'base_dsp_layer_imbalance', 'dsp_layer_imbalance_delta',
    'base_io_layer_imbalance', 'io_layer_imbalance_delta')
ACTION_FEATURE_DIM = len(FEATURE_NAMES)


def _cpu(value, dtype=None):
    return torch.as_tensor(value).detach().cpu().to(dtype=dtype) if dtype else torch.as_tensor(value).detach().cpu()


def _core(data, action):
    device = data['demand'].device
    # Validate indices before conversion: silently truncating floats would
    # misrepresent the action whose digest is paired with a VPR reward.
    arrays = {}
    for key in ('assignment', 'layer'):
        raw = torch.as_tensor(action[key], device=device)
        if raw.dtype == torch.bool or raw.is_complex() or not torch.isfinite(raw).all() or (raw != raw.floor()).any():
            raise ValueError('011 action indices must be finite integers')
        arrays[key] = raw.long()
    raw_xy = torch.as_tensor(action['xy'], device=device)
    if raw_xy.is_complex():
        raise ValueError('011 XY must be real')
    arrays['xy'] = raw_xy.to(data['demand'].dtype)
    return arrays


def validate_residual_action(data, action, *, base_action=None, arm=None, chunk_size=256):
    """Recompute all constraints, including the arm's attribution contract.

    A capacity certificate is for five-resource APBlock proxies, not exact
    cluster packing/site legality. Supplying base_action additionally verifies
    D's original contract, the frozen-row obligation and the true edit budget.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size < 1:
        raise ValueError('positive integer chunk_size required')
    meta = validate(data)
    arrays = _core(data, action)
    leaves, layers, xy = [arrays[k] for k in ('assignment', 'layer', 'xy')]
    n, r = len(data['demand']), len(data['capacity'])
    if leaves.shape != (n,) or layers.shape != (n,) or xy.shape != (n, 2):
        raise ValueError('011 action must cover all atomic APBlocks')
    if (leaves < 0).any() or (leaves >= r).any() or not torch.equal(layers, meta[0][leaves]):
        raise ValueError('011 action leaf/layer disagreement or out of range')
    if not torch.isfinite(xy).all() or (xy < 0).any() or (xy >= 1).any():
        raise ValueError('011 XY must be normalized [0,1)')
    bounds = data['region_bounds'][leaves]
    if ((xy < bounds[:, [0, 2]]-TOLERANCE) | (xy > bounds[:, [1, 3]]+TOLERANCE)).any():
        raise ValueError('011 XY outside selected leaf')
    known = meta[4] >= 0
    if not torch.allclose(xy[known], meta[4][known].to(xy), atol=TOLERANCE, rtol=0):
        raise ValueError('011 action moved a fixed XY axis')
    audit = validate_leaf_capacity(data, leaves, chunk_size=chunk_size)
    if action.get('schema') == SCHEMA:
        if action.get('decoder_kind') != DECODER_SCHEMA or action.get('coordinate_contract') != COORDINATE_CONTRACT:
            raise ValueError('011 decoder/coordinate contract mismatch')
        if action.get('constraints_sha256') != constraints_digest(data):
            raise ValueError('011 constraints digest mismatch')
        if action.get('deployed_action_sha256') != action_digest(action):
            raise ValueError('011 deployed action digest mismatch')
    if base_action is not None:
        validate_structured_action(data, base_action, chunk_size=chunk_size)
        base = _core(data, base_action)
        changed_leaf = leaves != base['assignment']
        changed_layer = layers != base['layer']
        changed_xy = (xy != base['xy']).any(1)
        changed = changed_leaf | changed_layer | changed_xy
        movable = data.get('movable', torch.ones(n, device=xy.device, dtype=torch.bool)).bool()
        if changed[~movable].any():
            raise ValueError('011 residual moved an immovable APBlock')
        selected_arm = arm or action.get('residual_arm')
        if selected_arm not in ARMS:
            raise ValueError('011 residual arm must be L, Y or J')
        shared_identity = action.get('candidate_id') == 'identity' and not changed.any()
        if action.get('schema') == SCHEMA and action.get('residual_arm') != selected_arm and not shared_identity:
            raise ValueError('011 recorded residual arm disagrees with requested arm')
        if selected_arm == 'L' and not torch.equal(xy, base['xy']):
            raise ValueError('011 L arm must preserve XY exactly')
        if selected_arm == 'Y' and (changed_leaf.any() or changed_layer.any()):
            raise ValueError('011 Y arm must preserve leaf and layer exactly')
        if selected_arm in ('L', 'J') and not torch.equal(changed_leaf, changed_layer):
            raise ValueError('011 leaf reassignment must be a layer change')
        if action.get('schema') == SCHEMA:
            if action.get('base_action_sha256') != action_digest(base_action):
                raise ValueError('011 base action digest mismatch')
            limit = action.get('residual_diagnostics', {}).get('edit_limit')
            if isinstance(limit, bool) or not isinstance(limit, int) or limit < 0 or int(changed.sum()) > limit:
                raise ValueError('011 residual exceeds bounded edit count')
        audit.update(changed_nodes=int(changed.sum()), changed_layers=int(changed_layer.sum()),
                     changed_xy=int(changed_xy.sum()), immovable_preserved=True,
                     arm_contract_verified=selected_arm)
    audit.update(coordinate_contract=COORDINATE_CONTRACT, convergence_loop=False)
    return audit


def _graph(data, action):
    """One O(N+E) CPU snapshot and weighted connectivity aggregation."""
    n = len(data['demand'])
    xy = _cpu(action['xy'], torch.float64).numpy().copy()
    layers = _cpu(action['layer'], torch.long).numpy().copy()
    leaves = _cpu(action['assignment'], torch.long).numpy().copy()
    src = _cpu(data['logic_src'], torch.long).numpy()
    dst = _cpu(data['logic_dst'], torch.long).numpy()
    weight = _cpu(data['logic_weight'], torch.float64).numpy()
    critical = _cpu(data.get('logic_criticality', torch.zeros(len(src))), torch.float64).numpy().clip(0, 1)
    if (weight < 0).any():
        raise ValueError('011 candidates require nonnegative connection weights')
    weighted = weight*(1+critical)
    degree = np.zeros(n, dtype=np.float64)
    centre = np.zeros((n, 2), dtype=np.float64)
    votes = np.zeros((n, int(data['region_layer'].max())+1), dtype=np.float64)
    np.add.at(degree, src, weighted); np.add.at(degree, dst, weighted)
    np.add.at(centre, src, weighted[:, None]*xy[dst]); np.add.at(centre, dst, weighted[:, None]*xy[src])
    np.add.at(votes, (src, layers[dst]), weighted); np.add.at(votes, (dst, layers[src]), weighted)
    centre = np.where(degree[:, None] > 0, centre/np.maximum(degree[:, None], 1e-30), xy)
    return dict(n=n, xy=xy, leaves=leaves, layers=layers, degree=degree, centre=centre,
                votes=votes, src=src, dst=dst, weight=weight, critical=critical)


def _layer_edit(data, base, graph, limit, seed):
    """At most limit single APBlock moves, checking only bounded incident leaves.

    Resource pressure and incident-neighbour affinity order proposals. They
    propose experiments; no claim is made that a lower proposal proxy means
    better VPR QoR. Unlike a cut-only rule, resource pressure can propose useful
    cross-layer placements even when the original action occupies one layer.
    """
    leaves, layers, xy = base['leaves'].copy(), base['layers'].copy(), base['xy']
    demand = _cpu(data['demand'], torch.float64).numpy()
    capacity = _cpu(data['capacity'], torch.float64).numpy()
    region_layer = _cpu(data['region_layer'], torch.long).numpy()
    layer_capacity = np.zeros((len(graph['votes'][0]), 5))
    np.add.at(layer_capacity, region_layer, capacity)
    usage = np.zeros_like(capacity); np.add.at(usage, leaves, demand)
    layer_usage = np.zeros_like(layer_capacity); np.add.at(layer_usage, layers, demand)
    movable = _cpu(data.get('movable', torch.ones(graph['n'], dtype=torch.bool))).bool().numpy()
    node = np.arange(graph['n'])
    # Hash provides deterministic tie-breaking and distinct finite-strength
    # proposals without depending on global NumPy/Torch random state.
    jitter = ((node.astype(np.int64)*1103515245 + (int(seed) % 2147483647)*12345) % 2147483647)/2147483647
    priority = np.log1p(graph['degree']) + .05*jitter
    order = np.lexsort((node, -priority))
    bounds = _cpu(data['region_bounds'], torch.float64).numpy()
    meta = validate(data)
    changed = []
    considered = 0
    # Examine only a fixed multiple of the sparse edit budget. A fragmented
    # or pinned design yields fewer edits, not an N-node Torch dispatch loop.
    order = order[movable[order]][:max(4*limit, 64)]
    masks = []
    for start in range(0, len(order), 256):
        ids = torch.as_tensor(order[start:start+256].copy(), device=data['demand'].device)
        masks.append(compatibility_mask(data, ids, meta).cpu().numpy())
    compatible_rows = np.concatenate(masks) if masks else np.empty((0, len(capacity)), dtype=bool)
    for row, node_id in enumerate(order):
        if len(changed) >= limit:
            break
        if not movable[node_id]:
            continue
        considered += 1
        compatible = compatible_rows[row]
        same_xy = ((xy[node_id] >= bounds[:, [0, 2]]-TOLERANCE) &
                   (xy[node_id] <= bounds[:, [1, 3]]+TOLERANCE)).all(1)
        old_leaf, old_layer = int(leaves[node_id]), int(layers[node_id])
        fits = (usage+demand[node_id] <= capacity+TOLERANCE).all(1)
        fits &= (layer_usage[region_layer]+demand[node_id] <= layer_capacity[region_layer]+TOLERANCE).all(1)
        choices = np.flatnonzero(compatible & same_xy & fits & (region_layer != old_layer))
        if not len(choices):
            continue
        target_layers = region_layer[choices]
        affinity = graph['votes'][node_id, target_layers]/max(graph['degree'][node_id], 1.)
        projected_layer = (layer_usage[target_layers]+demand[node_id])/np.maximum(layer_capacity[target_layers], 1.)
        projected_leaf = (usage[choices]+demand[node_id])/np.maximum(capacity[choices], 1.)
        score = affinity - .25*np.square(projected_layer).mean(1) - .05*np.square(projected_leaf).mean(1)
        target = int(choices[int(np.argmax(score))])
        layer = int(region_layer[target])
        leaves[node_id], layers[node_id] = target, layer
        usage[old_leaf] -= demand[node_id]; usage[target] += demand[node_id]
        layer_usage[old_layer] -= demand[node_id]; layer_usage[layer] += demand[node_id]
        changed.append(int(node_id))
    return leaves, layers, changed, considered


def _xy_edit(data, graph, leaves, xy, limit, strength, preferred=()):
    bounds = _cpu(data['region_bounds'], torch.float64).numpy()[leaves]
    fixed = _cpu(data.get('fixed_xy', torch.full((graph['n'], 2), -1.)), torch.float64).numpy()
    movable = _cpu(data.get('movable', torch.ones(graph['n'], dtype=torch.bool))).bool().numpy()
    low, high = bounds[:, [0, 2]], bounds[:, [1, 3]]
    # One direct step within the selected leaf. The magnitude is at most
    # strength times that leaf's span on each free coordinate axis.
    desired = np.maximum(low, np.minimum(high, graph['centre']))
    delta = (desired-xy)*strength
    delta = np.maximum(-strength*(high-low), np.minimum(strength*(high-low), delta))
    delta[fixed >= 0] = 0.; delta[~movable] = 0.
    priority = np.abs(delta).sum(1)*(1+np.log1p(graph['degree']))
    order = np.lexsort((np.arange(graph['n']), -priority))
    preferred = list(dict.fromkeys(int(i) for i in preferred))
    preferred_set = set(preferred)
    selected = preferred + [int(i) for i in order if int(i) not in preferred_set]
    selected = np.asarray(selected[:limit], dtype=np.int64)
    result = xy.copy()
    result[selected] += delta[selected]
    # Keep the normalized upper bound strictly below 1 for edge leaves.
    dtype = data['demand'].dtype
    upper = float(torch.nextafter(torch.tensor(1., dtype=dtype), torch.tensor(0., dtype=dtype)))
    result[selected] = np.minimum(result[selected], upper)
    result[fixed >= 0] = fixed[fixed >= 0]
    return result


@torch.no_grad()
def make_candidates(data, base_action, arm, *, strengths=(.15, .35), max_moves=256,
                    fraction=.02, seed=11301):
    """Return identity and at most two unique bounded candidates for one arm.

    L preserves all XY bit-for-bit, Y preserves all regions/layers; J combines
    L then Y with a shared bound on the union of changed APBlocks. More than two
    requested strengths is rejected to keep the experiment budget explicit.
    Unchanged candidates are deduplicated rather than pretending to be new
    training actions. All arrays remain on the graph's device and dtype.
    """
    if arm not in ARMS:
        raise ValueError('011 candidate arm must be L, Y or J')
    if isinstance(max_moves, bool) or not isinstance(max_moves, int) or max_moves < 0:
        raise ValueError('011 max_moves must be a nonnegative integer')
    if not math.isfinite(fraction) or not 0 <= fraction <= 1:
        raise ValueError('011 fraction must lie in [0,1]')
    strengths = tuple(strengths)
    if len(strengths) > 2 or len(set(strengths)) != len(strengths) or any(not math.isfinite(s) or not 0 < s <= 1 for s in strengths):
        raise ValueError('011 requires at most two distinct strengths in (0,1]')
    validate_structured_action(data, base_action)
    graph = _graph(data, base_action)
    n = graph['n']; device = data['demand'].device
    limit = min(max_moves, math.ceil(n*fraction))
    base_sha, constraint_sha = action_digest(base_action), constraints_digest(data)
    result, seen = [], set()
    for index, strength in enumerate((0.,)+strengths):
        core = dict(leaves=graph['leaves'].copy(), layers=graph['layers'].copy(), xy=graph['xy'].copy())
        layer_changed, considered = [], 0
        budget = 0 if index == 0 else max(1, math.ceil(limit*strength/max(strengths))) if limit else 0
        if index and arm in ('L', 'J') and budget:
            core['leaves'], core['layers'], layer_changed, considered = _layer_edit(data, core, graph, budget, seed)
        if index and arm in ('Y', 'J') and budget:
            core['xy'] = _xy_edit(data, graph, core['leaves'], core['xy'], budget, strength, layer_changed)
        action = dict(assignment=torch.as_tensor(core['leaves'], device=device, dtype=torch.long),
            layer=torch.as_tensor(core['layers'], device=device, dtype=torch.long),
            xy=torch.as_tensor(core['xy'], device=device, dtype=data['demand'].dtype),
            schema=SCHEMA, decoder_kind=DECODER_SCHEMA, coordinate_contract=COORDINATE_CONTRACT,
            candidate_id='identity' if index == 0 else arm.lower()+'_s'+str(index),
            residual_arm=arm, candidate_strength=float(strength),
            base_action_sha256=base_sha, constraints_sha256=constraint_sha,
            initializer='residual_'+arm, tail_weight=0., risk_diagnostics_computed=False,
            residual_diagnostics=dict(edit_limit=budget, configured_edit_limit=limit,
                max_moves=max_moves, fraction=float(fraction), proposal_nodes_considered=considered,
                generation_seed=int(seed), convergence_loop=False, packing_legality_claimed=False))
        action['deployed_action_sha256'] = action_digest(action)
        if action['deployed_action_sha256'] in seen:
            continue
        seen.add(action['deployed_action_sha256'])
        audit = validate_residual_action(data, action, base_action=base_action, arm=arm)
        action['capacity_diagnostics'] = audit
        action['residual_diagnostics'].update(changed_nodes=audit['changed_nodes'],
            changed_layers=audit['changed_layers'], changed_xy=audit['changed_xy'])
        result.append(action)
    return result


def _metrics(data, action):
    graph = _graph(data, action)
    src, dst, w, crit = [graph[k] for k in ('src', 'dst', 'weight', 'critical')]
    xy_length = np.abs(graph['xy'][src]-graph['xy'][dst]).sum(1)
    cross = (graph['layers'][src] != graph['layers'][dst]).astype(np.float64)
    denominator, critical_denominator = max(w.sum(), 1.), max((w*crit).sum(), 1.)
    usage = np.zeros_like(_cpu(data['capacity'], torch.float64).numpy())
    demand = _cpu(data['demand'], torch.float64).numpy()
    np.add.at(usage, graph['leaves'], demand)
    cap = _cpu(data['capacity'], torch.float64).numpy()
    peak = float((usage/np.maximum(cap, 1.)).max())
    layer_use = np.zeros((int(data['region_layer'].max())+1, 5))
    np.add.at(layer_use, graph['layers'], demand)
    layer_mass = layer_use/np.maximum(demand.sum(0), 1.)
    imbalance = layer_mass.max(0)-layer_mass.min(0)
    return np.asarray([(xy_length*w).sum()/denominator, (xy_length*w*crit).sum()/critical_denominator,
        (cross*w).sum()/denominator, (cross*w*crit).sum()/critical_denominator, peak]), imbalance


@torch.no_grad()
def action_features(data, base_action, action):
    """Fixed length, finite proposal descriptors; contains no measured label."""
    base, current = _core(data, base_action), _core(data, action)
    layer_changed = current['layer'] != base['layer']
    xy_changed = (current['xy'] != base['xy']).any(1)
    changed = layer_changed | xy_changed | (current['assignment'] != base['assignment'])
    displacement = (current['xy']-base['xy']).abs().sum(1)
    demand = data['demand']
    moved_mass = demand[changed].sum(0)/demand.sum(0).clamp_min(1.)
    before, imbalance_before = _metrics(data, base_action)
    after, imbalance_after = _metrics(data, action)
    features = [float(not changed.any()), float(action.get('candidate_strength', 0.)),
        float(layer_changed.float().mean()), float(xy_changed.float().mean()), float(changed.float().mean()),
        float(displacement.mean()), float(displacement.max()), *moved_mass.cpu().tolist(),
        before[0], after[0]-before[0], before[1], after[1]-before[1],
        before[2], after[2]-before[2], before[3], after[3]-before[3], before[4], after[4]-before[4]]
    for old, new in zip(imbalance_before, imbalance_after):
        features.extend((old, new-old))
    result = torch.tensor(features, dtype=torch.float32, device=demand.device)
    if len(result) != ACTION_FEATURE_DIM or not torch.isfinite(result).all():
        raise ValueError('011 candidate features must be finite and match declared feature schema')
    return result
