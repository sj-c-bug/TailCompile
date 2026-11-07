"""016: bounded current-state descent of an explicit net-level XY wire proxy.

Every accepted proposal strictly decreases the *deployed* coordinate proxy.
This is a discrete, finite candidate search, not a certificate of continuous
optimality, route legality, congestion, or timing. Native GP/APPack still run.
"""
from __future__ import annotations

from collections import Counter
import math
import time

import numpy as np
import torch

try:
    from .residual_actions import validate_residual_action
    from .aligned_model import action_digest
except ImportError:
    from residual_actions import validate_residual_action
    from aligned_model import action_digest


SCHEMA = 'v8-016-explainable-action-1'
DEFAULT_CONFIG = dict(max_evaluations=8192, max_moves=64, max_passes=3,
    max_seconds=20., trust_fraction=.25, max_work_units=20000000,
    gain_absolute=1e-6, gain_relative=1e-9, max_candidate_nodes=256)
OBJECTIVE = ('unweighted original-net incidence HPWL in XY tile units; '
    'cross-layer nets use the minimum sum of layer HPWLs through one actual '
    'directed-access-compatible CHANZ member location; no Z-length/timing term')


class BudgetExceeded(RuntimeError):
    pass


def _config(config):
    result = dict(DEFAULT_CONFIG)
    if config:
        unknown = set(config)-set(result)
        if unknown:
            raise ValueError('016 unknown refinement options: '+str(sorted(unknown)))
        result.update(config)
    for k in ('max_evaluations','max_moves','max_passes','max_work_units','max_candidate_nodes'):
        if isinstance(result[k], bool) or not isinstance(result[k], int) or result[k] < 0:
            raise ValueError('016 nonnegative integer required: '+k)
    for k in ('max_seconds','trust_fraction','gain_absolute','gain_relative'):
        v = result[k]
        if isinstance(v, bool) or not isinstance(v, (int,float)) or not math.isfinite(v) or v < 0:
            raise ValueError('016 finite nonnegative option required: '+k)
    if result['trust_fraction'] > 1:
        raise ValueError('016 trust_fraction must not exceed one original leaf span')
    if result['gain_absolute'] <= 0:
        raise ValueError('016 positive absolute descent tolerance required')
    return result


def _array(value, dtype=None):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _core(data, action):
    """Validate all hard proxy obligations without inheriting old action schema."""
    core = {k: action[k] for k in ('assignment','layer','xy')}
    audit = validate_residual_action(data, core)
    return dict(assignment=_array(core['assignment'],np.int64).copy(),
        layer=_array(core['layer'],np.int64).copy(),
        xy=_array(torch.as_tensor(core['xy']).to(dtype=data['demand'].dtype)).copy()), audit


class _Budget:
    def __init__(self, config, started):
        self.config = config
        self.started = started
        self.work = 0

    def check(self, work=0):
        if self.work+work > self.config['max_work_units']:
            raise BudgetExceeded('work_budget')
        if time.perf_counter()-self.started >= self.config['max_seconds']:
            raise BudgetExceeded('time_budget')
        self.work += work


class _Problem:
    def __init__(self, data, core, summary, budget):
        self.budget = budget
        self.xy, self.leaf, self.layer = core['xy'], core['assignment'], core['layer']
        self.scale = np.asarray(data.get('grid',data.get('teacher_axis_scale',[]))[:2],dtype=float)
        if self.scale.shape != (2,) or not np.isfinite(self.scale).all() or (self.scale <= 0).any():
            raise ValueError('016 explicit positive XY tile scale required')
        self.n = len(self.xy)
        if set(_array(data['region_layer']).tolist()) != {0,1}:
            raise ValueError('016 requires exactly two physical layers')
        if 'nets' not in data:
            raise ValueError('016 requires original net incidence, not expanded message edges')
        self.nets, self.incidence, self.drivers = [], [[] for _ in range(self.n)], []
        directional = {int(r['id']): r for r in data.get('risk_nets',[])}
        for ni, pins in enumerate(data['nets']):
            budget.check(len(pins)+1)
            raw = np.asarray(pins)
            if raw.ndim != 1 or raw.dtype.kind not in 'iu' or (raw < 0).any() or (raw >= self.n).any():
                raise ValueError('016 net incidence must contain in-range integer APBlock ids')
            ids = np.unique(raw).astype(np.int64)
            self.nets.append(ids)
            for i in ids:
                self.incidence[i].append(ni)
            row = directional.get(ni)
            if row is not None and set(row['pins']) != set(ids.tolist()):
                raise ValueError('016 directional net differs from original net incidence')
            drivers = row.get('drivers',[]) if row is not None else []
            self.drivers.append(int(drivers[0]) if len(drivers)==1 and int(drivers[0]) in ids else None)
        self.profiles = []
        if summary is not None:
            grid = summary.get('grid',[])
            if len(grid)!=3 or grid[2]!=2 or not np.array_equal(np.asarray(grid[:2]),self.scale):
                raise ValueError('016 vertical summary must match the two-layer XY grid')
            if 'region_layer' in summary and not np.array_equal(summary['region_layer'],_array(data['region_layer'])):
                raise ValueError('016 vertical summary leaf ordering mismatch')
            for pool in summary.get('pools',[]):
                budget.check()
                # A centroid is not a physical channel entrance. Retain actual
                # positive-capacity member positions and directed access sets.
                points=[]
                for member in pool.get('members',[]):
                    budget.check(1)
                    member_capacity = float(member.get('capacity',0))
                    if not math.isfinite(member_capacity) or member_capacity<0:
                        raise ValueError('016 CHANZ member capacity must be finite and nonnegative')
                    if member_capacity == 0:
                        continue
                    p = np.asarray(member['xy'],dtype=float)
                    if p.shape!=(2,) or not np.isfinite(p).all() or (p<0).any() or (p>=1).any():
                        raise ValueError('016 invalid actual CHANZ member coordinate')
                    points.append(p)
                if not points:
                    continue
                orientations=[]
                for o in pool.get('orientations',[]):
                    sl, tl = int(o['source_layer']),int(o['sink_layer'])
                    if (sl,tl) not in ((0,1),(1,0)):
                        raise ValueError('016 unsupported vertical orientation')
                    orientations.append((sl,tl,frozenset(o['source_regions']),frozenset(o['sink_regions'])))
                if orientations:
                    self.profiles.append((np.unique(np.asarray(points),axis=0)*self.scale,orientations))
        self.portal_cache = {}

    def _entrances(self, source_layer, source_regions, sink_regions):
        key = source_layer, tuple(sorted(source_regions)), tuple(sorted(sink_regions))
        if key not in self.portal_cache:
            points=[]
            for xy, orientations in self.profiles:
                self.budget.check(1)
                if any(sl==source_layer and source_regions<=sr and sink_regions<=tr
                       for sl,tl,sr,tr in orientations):
                    points.append(xy)
            self.portal_cache[key] = np.unique(np.concatenate(points),axis=0) if points else np.empty((0,2))
        return self.portal_cache[key]

    def cost(self, ni):
        ids = self.nets[ni]
        self.budget.check(len(ids)+1)
        if len(ids)<2:
            return 0.
        xy = self.xy[ids]*self.scale
        z = self.layer[ids]
        groups = sorted(set(z.tolist()))
        if len(groups)==1:
            return float(np.ptp(xy,axis=0).sum())
        if groups != [0,1]:
            raise ValueError('016 only two-layer initialization is supported')
        regions = [set(self.leaf[ids[z==layer]].tolist()) for layer in (0,1)]
        driver = self.drivers[ni]
        if driver is not None:
            source = int(self.layer[driver])
            entrances = self._entrances(source,regions[source],regions[1-source])
        else:
            # Unknown drivers require access in both directions at the same
            # actual coordinate; this is conservative, not a pin certificate.
            first = self._entrances(0,regions[0],regions[1])
            second = self._entrances(1,regions[1],regions[0])
            other = set(map(tuple,second.tolist()))
            entrances = np.asarray([p for p in first if tuple(p) in other],dtype=float).reshape(-1,2)
        if not len(entrances):
            return None
        self.budget.check(len(entrances))
        length = np.zeros(len(entrances),dtype=float)
        for layer in (0,1):
            selected = xy[z==layer]
            lo,hi = selected.min(0),selected.max(0)
            length += (hi-lo).sum()+np.maximum(lo-entrances,0).sum(1)+np.maximum(entrances-hi,0).sum(1)
        return float(length.min())

    def costs(self):
        return [self.cost(ni) for ni in range(len(self.nets))]


def objective(data, action, vertical_summary=None, config=None):
    """Independent public recomputation of the explicit proxy and its coverage.

Unrepresented baseline cross-layer nets are returned as None, never zero or
silently dropped under an all-net claim. refine freezes all their endpoints.
"""
    cfg = _config(config)
    started = time.perf_counter()
    core,_ = _core(data,action)
    budget = _Budget(cfg,started)
    budget.check()
    problem = _Problem(data,core,vertical_summary,budget)
    costs = problem.costs()
    missing = [i for i,c in enumerate(costs) if c is None]
    return dict(value=math.fsum(c for c in costs if c is not None),net_costs=costs,
        unsupported_net_ids=missing,represented_net_count=len(costs)-len(missing),
        total_net_count=len(costs),objective_semantics=OBJECTIVE,
        full_net_coverage=not missing,work_units=budget.work,
        direction_unknown_net_count=sum(d is None for d in problem.drivers),
        access_semantics='RRG region reachability upper bound; not a site/pin route certificate')


def refine(data, base_action, vertical_summary=None, config=None):
    """Finite deterministic coordinate descent with identity always admissible.

The search uses only original-net incidence, current geometry and RRG access.
No VPR call, reward input, training, random perturbation, or repeated decoder
occurs here. Stay is the zero-gain choice; resource pressure is a constraint.
"""
    cfg = _config(config)
    started = time.perf_counter()
    base,audit = _core(data,base_action)
    core = {k:v.copy() for k,v in base.items()}
    budget = _Budget(cfg,started)
    diag = dict(config=cfg,objective_semantics=OBJECTIVE,initial_objective=None,
        final_objective=None,history=[],accepted_moves=[],stop_reason=None,
        counters=dict(evaluations=0,accepted_moves=0,passes=0,work_units=0),
        unsupported_net_ids=[],frozen_unsupported_nodes=[],rejections={},
        stationarity_certified=False,physical_legality_claimed=False,
        timing_improvement_claimed=False,full_net_coverage=False,
        total_net_count=len(data.get('nets',[])),represented_net_count=None,
        candidate_rule='local original-leaf trust box; 3x3 XY grid per compatible intersecting leaf; current position included',
        search_rule='first-improving node in current incident-cost order; best feasible option for that node; re-evaluate affected nets after each accepted move',
        global_optimum_claimed=False,pair_swaps=False)
    try:
        budget.check()
        problem = _Problem(data,core,vertical_summary,budget)
        costs = problem.costs()
    except BudgetExceeded as exc:
        diag['stop_reason'] = str(exc)+'_setup'
        return _finish(data,base_action,base,core,audit,diag,budget,started)
    missing = [i for i,c in enumerate(costs) if c is None]
    frozen = {int(n) for ni in missing for n in problem.nets[ni]}
    diag.update(unsupported_net_ids=missing,frozen_unsupported_nodes=sorted(frozen),
        full_net_coverage=not missing,total_net_count=len(costs),represented_net_count=len(costs)-len(missing),
        direction_unknown_net_count=sum(d is None for d in problem.drivers),
        access_semantics='RRG region reachability upper bound; not a site/pin route certificate')
    current = math.fsum(c for c in costs if c is not None)
    diag.update(initial_objective=current,final_objective=current,history=[current])
    demand = _array(data['demand'],float)
    capacity = _array(data['capacity'],float)
    bounds = _array(data['region_bounds'],float)
    region_layer = _array(data['region_layer'],np.int64)
    movable = _array(data.get('movable',np.ones(problem.n,dtype=bool)),bool)
    fixed_xy = _array(data.get('fixed_xy',np.full((problem.n,2),-1.)),float)
    fixed_layer = _array(data.get('fixed_layer',np.full(problem.n,-1)),np.int64)
    fixed_region = _array(data.get('fixed_region',np.full(problem.n,-1)),np.int64)
    legal = _array(data.get('legal_layers',np.ones((problem.n,2),dtype=bool)),bool)
    usage = np.zeros_like(capacity)
    np.add.at(usage,core['assignment'],demand)
    layer_capacity = np.zeros((2,5),dtype=float)
    np.add.at(layer_capacity,region_layer,capacity)
    if 'layer_capacity' in data:
        layer_capacity = _array(data['layer_capacity'],float)
    layer_usage = np.zeros_like(layer_capacity)
    np.add.at(layer_usage,core['layer'],demand)
    base_bounds = bounds[base['assignment']]
    radius = (base_bounds[:,[1,3]]-base_bounds[:,[0,2]])*cfg['trust_fraction']
    low,high = np.maximum(0,base['xy']-radius),np.minimum(np.nextafter(np.float32(1),np.float32(0)),base['xy']+radius)
    reject = Counter()
    eligible = [i for i in range(problem.n) if movable[i] and i not in frozen and problem.incidence[i]]
    # A node's incident objective is an upper bound on reducible local cost.
    # Connectivity degree is never itself a decision to move.
    def order_nodes():
        return sorted(eligible,key=lambda i:(-math.fsum(costs[n] for n in problem.incidence[i]),i))[:cfg['max_candidate_nodes']]
    try:
        for iteration in range(cfg['max_passes']):
            budget.check()
            diag['counters']['passes'] = iteration+1
            accepted = 0
            order = order_nodes()
            budget.check()
            for node in order:
                budget.check()
                if diag['counters']['accepted_moves'] >= cfg['max_moves']:
                    raise BudgetExceeded('move_budget')
                old_leaf,old_layer = int(core['assignment'][node]),int(core['layer'][node])
                old_xy = core['xy'][node].copy()
                affected = problem.incidence[node]
                before = math.fsum(costs[n] for n in affected)
                threshold = max(cfg['gain_absolute'],cfg['gain_relative']*max(1.,before))
                best = None
                # At most a fixed 3x3 grid for each local compatible leaf.
                target_rows = np.flatnonzero((bounds[:,0]<=high[node,0])&(bounds[:,1]>=low[node,0])&
                    (bounds[:,2]<=high[node,1])&(bounds[:,3]>=low[node,1]))
                for target in target_rows:
                    budget.check(1)
                    target = int(target); layer = int(region_layer[target])
                    if (layer>=legal.shape[1] or not legal[node,layer] or
                        fixed_layer[node]>=0 and fixed_layer[node]!=layer or
                        fixed_region[node]>=0 and fixed_region[node]!=target):
                        reject['fixed_or_layer_incompatible']+=1; continue
                    proposed = usage[target]+(demand[node] if target!=old_leaf else 0)
                    proposed_layer = layer_usage[layer]+(demand[node] if layer!=old_layer else 0)
                    if (proposed > capacity[target]+1e-6).any() or (proposed_layer > layer_capacity[layer]+1e-6).any():
                        reject['capacity']+=1; continue
                    lo = np.maximum(low[node],bounds[target,[0,2]])
                    # Interior ownership avoids float rounding outside a leaf.
                    hi = np.minimum(high[node],np.nextafter(bounds[target,[1,3]].astype(core['xy'].dtype),np.array(-np.inf,dtype=core['xy'].dtype)))
                    # Round inward in the deployment dtype before any proposal.
                    # A tolerance here could admit a fixed coordinate exactly
                    # on another leaf's excluded upper boundary.
                    rounded_lo = lo.astype(core['xy'].dtype)
                    rounded_hi = hi.astype(core['xy'].dtype)
                    lo = np.where(rounded_lo<lo,np.nextafter(rounded_lo,np.array(np.inf,dtype=core['xy'].dtype)),rounded_lo)
                    hi = np.where(rounded_hi>hi,np.nextafter(rounded_hi,np.array(-np.inf,dtype=core['xy'].dtype)),rounded_hi)
                    if (lo>hi).any():
                        continue
                    axes=[]
                    for axis in (0,1):
                        if fixed_xy[node,axis]>=0:
                            axes.append([fixed_xy[node,axis]])
                        else:
                            axes.append(sorted(set([float(lo[axis]),float(np.clip(old_xy[axis],lo[axis],hi[axis])),float(hi[axis])])))
                    for x in axes[0]:
                        for y in axes[1]:
                            budget.check()
                            point = np.asarray([x,y],dtype=core['xy'].dtype)
                            if (point<lo).any() or (point>hi).any():
                                reject['fixed_xy_outside_leaf']+=1; continue
                            if target==old_leaf and np.array_equal(point,old_xy):
                                continue
                            if diag['counters']['evaluations']>=cfg['max_evaluations']:
                                raise BudgetExceeded('evaluation_budget')
                            diag['counters']['evaluations']+=1
                            core['assignment'][node],core['layer'][node],core['xy'][node] = target,layer,point
                            try:
                                values = [problem.cost(n) for n in affected]
                            finally:
                                core['assignment'][node],core['layer'][node],core['xy'][node] = old_leaf,old_layer,old_xy
                            if any(c is None for c in values):
                                reject['unsupported_vertical_access']+=1; continue
                            after = math.fsum(values)
                            if before-after>threshold and (best is None or after<best[0]):
                                best = after,target,layer,point.copy(),values
                            else:
                                reject['no_strict_gain']+=1
                if best is not None:
                    after,target,layer,point,values = best
                    old_total = current
                    updates = dict(zip(affected,values))
                    total = math.fsum(updates.get(ni,c) for ni,c in enumerate(costs) if c is not None)
                    if old_total-total <= threshold:
                        reject['global_roundoff_no_gain']+=1; continue
                    core['assignment'][node],core['layer'][node],core['xy'][node] = target,layer,point
                    if target!=old_leaf:
                        usage[old_leaf]-=demand[node];usage[target]+=demand[node]
                    if layer!=old_layer:
                        layer_usage[old_layer]-=demand[node];layer_usage[layer]+=demand[node]
                    for ni,value in updates.items():
                        costs[ni]=value
                    current=total;accepted+=1
                    diag['counters']['accepted_moves']+=1
                    diag['history'].append(current)
                    diag['accepted_moves'].append(dict(nodes=[int(node)],
                        old=dict(assignment=old_leaf,layer=old_layer,xy=old_xy.tolist()),
                        new=dict(assignment=target,layer=layer,xy=point.tolist()),
                        before=old_total,after=current,gain=old_total-current,
                        affected_nets=list(affected),affected_before=before,affected_after=after))
                    diag['final_objective']=current
            if not accepted:
                diag['stop_reason']='no_improving_candidate' if len(order)==len(eligible) else 'candidate_node_budget'
                diag['stationarity_certified']=len(order)==len(eligible)
                break
        else:
            diag['stop_reason']='pass_budget'
    except BudgetExceeded as exc:
        diag['stop_reason']=str(exc)
    diag['rejections']=dict(reject)
    return _finish(data,base_action,base,core,audit,diag,budget,started)


def _finish(data, base_action, base, core, base_audit, diag, budget, started):
    result = {k:torch.as_tensor(v,device=data['demand'].device,
        dtype=(data['demand'].dtype if k=='xy' else torch.long)).clone() for k,v in core.items()}
    audit = validate_residual_action(data,result)
    changed_leaf=core['assignment']!=base['assignment']
    changed_layer=core['layer']!=base['layer']
    changed_xy=np.any(core['xy']!=base['xy'],axis=1)
    changed=changed_leaf|changed_layer|changed_xy
    movable=_array(data.get('movable',np.ones(len(changed),dtype=bool)),bool)
    if changed[~movable].any():
        raise AssertionError('016 modified an immovable APBlock')
    # Single-node moves only; the base-relative trust bound cannot accumulate.
    spans=_array(data['region_bounds'],float)[base['assignment']]
    radius=(spans[:,[1,3]]-spans[:,[0,2]])*diag['config']['trust_fraction']
    if (np.abs(core['xy']-base['xy'])>radius+1e-7).any():
        raise AssertionError('016 exceeded original-coordinate trust box')
    audit.update(changed_nodes=int(changed.sum()),changed_layers=int(changed_layer.sum()),
        changed_xy=int(changed_xy.sum()),immovable_preserved=True)
    diag['counters']['work_units']=budget.work
    diag['elapsed_seconds']=time.perf_counter()-started
    diag['stationarity_scope']='declared finite XY grid and compatible local leaf candidates only'
    result.update(schema=SCHEMA,initializer='E',decoder_kind='finite-monotone-local-refinement-016-1',
        capacity_diagnostics=audit,refinement_diagnostics=diag,
        base_action_sha256=action_digest(base_action))
    result['deployed_action_sha256']=action_digest(result)
    return result
