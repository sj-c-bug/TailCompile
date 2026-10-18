"""2D candidate-dependent demand/supply proxy and capacity-preserving search.

This is a coarse routing proxy, NOT a routability certificate. A pool groups
distinct RR wire nodes with identical directed boundary footprints and segment
types. Switch multiplicity never creates capacity. A long wire remains ONE
shared pool across all boundaries it crosses. Per-net paths share pool demand.
"""
from __future__ import annotations

import gzip
import hashlib
import heapq
import json
import math
import random
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

VTR_COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'
RESOURCES = ('clb', 'memory', 'dsp_top')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def region_at(device, x, y):
    # Interior regions extend to the nearest perimeter for fixed IO terminals.
    return min(range(len(device['regions'])), key=lambda i: (
        max(device['regions'][i]['x_low'] - x, 0, x - device['regions'][i]['x_high']) ** 2
        + max(device['regions'][i]['y_low'] - y, 0, y - device['regions'][i]['y_high']) ** 2, i))


def build_supply(rr_graph, device):
    """Stream XML without retaining millions of rr_edges in memory.

Only wire-node spans crossing a region boundary are counted. Switch-only
crossings and intra-region pin access are outside this proxy's coverage.
No inferred linear channel-width scaling is performed.
"""
    pools = defaultdict(float)
    examples = defaultdict(list)
    segments, node_count, width = {}, 0, None
    grid_max = [-1, -1]
    coordinate_regions = {(x, y): region_at(device, x, y)
                          for x in range(device['grid_width']) for y in range(device['grid_height'])}
    footprints = {}
    opener = gzip.open if str(rr_graph).endswith('.gz') else open
    with opener(rr_graph, 'rb') as stream:
        stack = []
        for event, item in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start':
                stack.append(item)
                continue
            parent = stack[-2] if len(stack) > 1 else None
            tag = item.tag
            if tag == 'channel' and parent is not None and parent.tag == 'channels':
                width = int(item.get('chan_width_max'))
            if tag == 'grid_loc':
                if int(item.get('layer', '0')) != 0:
                    raise ValueError('v4 2D evaluator rejects multi-layer RRG')
                grid_max = [max(grid_max[0], int(item.get('x'))),
                            max(grid_max[1], int(item.get('y')))]
            if tag == 'segment' and parent is not None and parent.tag == 'segments':
                segments[item.get('id')] = item.get('name', item.get('id'))
            if tag == 'node' and parent is not None and parent.tag == 'rr_nodes':
                node_count += 1
                kind = item.get('type')
                if kind == 'CHANZ':
                    raise ValueError('v4 2D evaluator rejects CHANZ')
                if kind in ('CHANX', 'CHANY'):
                    loc = item.find('loc')
                    if int(loc.get('layer', '0')) != 0:
                        raise ValueError('v4 only supports layer=0')
                    xl, xh, yl, yh = [int(loc.get(k)) for k in ('xlow', 'xhigh', 'ylow', 'yhigh')]
                    direction = item.get('direction')
                    geometry = kind, xl, xh, yl, yh, direction
                    if geometry not in footprints:
                        sequence = []
                        for t in range(xl if kind == 'CHANX' else yl,
                                       (xh if kind == 'CHANX' else yh) + 1):
                            rid = coordinate_regions[t if kind == 'CHANX' else xl,
                                                     yl if kind == 'CHANX' else t]
                            if not sequence or rid != sequence[-1]:
                                sequence.append(rid)
                        arcs = list(zip(sequence, sequence[1:]))
                        if direction == 'DEC_DIR':
                            arcs = [(b, a) for a, b in arcs]
                        elif direction == 'BI_DIR':
                            arcs += [(b, a) for a, b in arcs]
                        elif direction != 'INC_DIR':
                            raise ValueError(f'unknown channel direction: {direction}')
                        footprints[geometry] = arcs
                    arcs = footprints[geometry]
                    if arcs:
                        sid = item.find('segment').get('segment_id')
                        key = (kind, direction, segments.get(sid, sid), tuple(sorted(arcs)))
                        capacity = float(item.get('capacity', '1'))
                        if capacity > 0:
                            pools[key] += capacity
                            if len(examples[key]) < 4:
                                examples[key].append(int(item.get('id')))
            # Child loc/segment attributes must survive until their node ends.
            if parent is not None and parent.tag in (
                    'rr_nodes', 'rr_edges', 'grid', 'segments', 'channels'):
                parent.remove(item)
            stack.pop()
    if grid_max != [device['grid_width'] - 1, device['grid_height'] - 1]:
        raise ValueError(f'RRG grid {grid_max} does not match device IR')
    if not pools or not width:
        raise ValueError('no boundary wire supply or channel width found')
    return {
        'schema': 'tailcompile-v4-2d-wire-pools-v1', 'layer_count': 1,
        'rr_graph_sha256': digest(rr_graph), 'channel_width': width,
        'device_ir_content_sha256': hashlib.sha256(json.dumps(device, sort_keys=True).encode()).hexdigest(),
        'rr_node_count': node_count, 'region_count': len(device['regions']),
        'capacity_semantics': 'unique wire-node capacities grouped by shared boundary footprint; no switch-edge count',
        'limitations': ['coarse regional connectivity is not exact RRG reachability',
                       'pin access and switch-only boundary crossings are not modeled',
                       'fractional pool sharing is a proxy, not a legal detailed route'],
        'pools': [{'id': i, 'kind': key[0], 'direction': key[1], 'segment': key[2],
                   'arcs': [list(arc) for arc in key[3]], 'capacity': capacity,
                   'example_rr_node_ids': examples[key]}
                  for i, (key, capacity) in enumerate(sorted(pools.items()))],
    }


def cvar(values, fraction=0.25):
    """Exact empirical upper-tail mean, including fractional endpoint weight."""
    if not 0 < fraction <= 1:
        raise ValueError('tail fraction must be in (0,1]')
    if not values:
        return 0.0
    ordered = sorted(values, reverse=True)
    mass = len(ordered) * fraction
    whole = int(math.floor(mass))
    return (sum(ordered[:whole]) + (mass - whole) *
            (ordered[whole] if whole < len(ordered) else 0.0)) / mass


class RiskEvaluator:
    def __init__(self, logic, device, supply, edges, tail_fraction=0.25):
        self.logic, self.device, self.supply, self.edges = logic, device, supply, edges
        self.tail_fraction = tail_fraction
        if [int(c['id']) for c in logic['clusters']] != list(range(len(logic['clusters']))):
            raise ValueError('cluster IDs must be dense and in order')
        if logic.get('macros') or any(c.get('macro_id') is not None for c in logic['clusters']):
            raise ValueError('explicit macros need a coupled-move implementation')
        self.capacities = [float(p['capacity']) for p in supply['pools']]
        self.arc_pools = defaultdict(list)
        for p in supply['pools']:
            for arc in p['arcs']:
                self.arc_pools[tuple(arc)].append(p['id'])
        self.arc_capacity = {a: sum(self.capacities[i] for i in ids)
                             for a, ids in self.arc_pools.items()}
        self.adj = defaultdict(list)
        self.length = {}
        for (a, b), capacity in self.arc_capacity.items():
            if capacity <= 0:
                continue
            ra, rb = device['regions'][a], device['regions'][b]
            self.length[a, b] = abs(ra['center_x'] - rb['center_x']) + abs(ra['center_y'] - rb['center_y'])
            self.adj[a].append(b)
        self.paths = {}
        for source in range(len(device['regions'])):
            queue, seen = [(0.0, (source,))], set()
            while queue:
                distance, path = heapq.heappop(queue)
                node = path[-1]
                if node in seen:
                    continue
                seen.add(node)
                self.paths[source, node] = tuple(zip(path, path[1:]))
                for target in sorted(self.adj[node]):
                    if target not in seen:
                        # Static deterministic routing: no label or current occupancy input.
                        heapq.heappush(queue, (distance + self.length[node, target]
                                              + 1e-3 / self.arc_capacity[node, target], path + (target,)))
        self.incident = [[] for _ in logic['clusters']]
        for i, edge in enumerate(edges):
            if len(edge.drivers) > 1:
                raise ValueError(f'multi-driver net unsupported: {edge.name}')
            for node in edge.pins:
                self.incident[node].append(i)
        self.route_cache = {}

    def route(self, edge, assignment):
        source = assignment[(edge.drivers or edge.pins[:1])[0]]
        targets = tuple(sorted({assignment[n] for n in (edge.sinks or edge.pins)} - {source}))
        key = source, targets
        if key in self.route_cache:
            return self.route_cache[key]
        arcs = set()
        for target in targets:
            path = self.paths.get((source, target))
            if path is None:
                return None
            arcs.update(path)
        used = {}
        for arc in sorted(arcs):
            for pool in self.arc_pools[arc]:
                # One net reserves a wire once, even across multiple cuts/fanout branches.
                share = self.capacities[pool] / self.arc_capacity[arc]
                used[pool] = max(used.get(pool, 0.0), share)
        result = used, sum(self.length[arc] for arc in arcs)
        self.route_cache[key] = result
        return result

    def legal(self, assignment, fixed=None):
        if len(assignment) != len(self.logic['clusters']):
            return False
        counts = Counter()
        for c, rid in zip(self.logic['clusters'], assignment):
            if not 0 <= rid < len(self.device['regions']):
                return False
            resource = c['resource']
            if resource == 'io':
                if fixed is not None and rid != fixed[c['id']]:
                    return False
                continue
            if resource not in RESOURCES or rid not in c.get('allowed_regions', range(len(self.device['regions']))):
                return False
            counts[rid, resource] += 1
            # Concrete distinct anchors, not fixed proportional fill targets.
            if counts[rid, resource] > len(self.device['regions'][rid]['tile_coordinates'].get(resource, [])):
                return False
        return True

    def state(self, assignment, previous=None, changed=()):
        if not self.legal(assignment):
            return None
        if previous is None:
            loads, routes, wire, timing = [0.0] * len(self.capacities), [None] * len(self.edges), 0.0, 0.0
            affected = range(len(self.edges))
        else:
            loads, routes = previous['loads'].copy(), previous['routes'].copy()
            wire, timing = previous['wire'], previous['timing']
            affected = sorted({i for node in changed for i in self.incident[node]})
        for i in affected:
            edge = self.edges[i]
            old = routes[i]
            if old is not None:
                for pool, value in old[0].items():
                    loads[pool] -= value
                wire -= old[1]
                timing -= old[1] * edge.criticality
            new = self.route(edge, assignment)
            if new is None:
                return None
            routes[i] = new
            for pool, value in new[0].items():
                loads[pool] += value
            wire += new[1]
            timing += new[1] * edge.criticality
        loads = [max(0.0, x) for x in loads]
        ratios = [v / c for v, c in zip(loads, self.capacities)]
        return {'loads': loads, 'routes': routes, 'wire': max(wire, 0.0),
                'timing': max(timing, 0.0), 'cvar': cvar(ratios, self.tail_fraction),
                'mean': sum(ratios) / len(ratios), 'max': max(ratios)}

    def report(self, state):
        if state is None:
            return {'proxy_feasible': False}
        return {'proxy_feasible': True, 'cvar': state['cvar'], 'mean': state['mean'],
                'max': state['max'], 'wire': state['wire'], 'timing': state['timing'],
                'tail_fraction': self.tail_fraction,
                'pools': [{'id': i, 'demand': d, 'capacity': self.capacities[i],
                           'rho': d / self.capacities[i]}
                          for i, d in enumerate(state['loads'])]}


def scales_for(state):
    return {'risk': max(state['cvar'], 1e-9), 'wire': max(state['wire'], 1.0),
            'timing': max(state['timing'], 1.0)}


def matched_scales_for(state):
    """Normalize mean and tail risk to one at the same initial assignment.

    v4 deliberately shared the CVaR denominator across both objectives.  The
    confirmatory experiment needs a fairer ablation: only the aggregation
    operator (mean versus upper-tail CVaR) should differ at initialization.
    """
    return {'tail_risk': max(state['cvar'], 1e-9),
            'mean_risk': max(state['mean'], 1e-9),
            'wire': max(state['wire'], 1.0),
            'timing': max(state['timing'], 1.0)}


def objective(state, scales, mode):
    risk = state['cvar'] if mode == 'tail' else state['mean'] if mode == 'mean' else 0.0
    if mode == 'none':
        risk_term = 0.0
    elif mode == 'tail' and 'tail_risk' in scales:
        risk_scale = scales['tail_risk']
        risk_term = risk / risk_scale
    elif mode == 'mean' and 'mean_risk' in scales:
        risk_scale = scales['mean_risk']
        risk_term = risk / risk_scale
    else:
        # Backward-compatible v4 semantics.
        risk_scale = scales['risk']
        risk_term = risk / risk_scale
    return risk_term + 0.15 * state['wire'] / scales['wire'] + 0.25 * state['timing'] / scales['timing']


def search(evaluator, initial, budget=512, seed=2027, mode='tail', scales=None,
           policy='hotspot', progress=None, max_moved_fraction=None):
    """Fixed objective-evaluation budget; invalid proposals counted separately.

random_restart evaluates random legal perturbations of the ORIGINAL start;
it never takes a VPR label as a selection criterion. All search modes share
the same legalizer, proxy paths and normalization reference.
"""
    if budget < 0 or mode not in ('tail', 'mean', 'none'):
        raise ValueError('invalid search budget or mode')
    rng = random.Random(seed)
    assignment = initial.copy()
    current = evaluator.state(assignment)
    if current is None:
        raise ValueError('initial assignment is infeasible in the coarse proxy')
    original = current
    scales = scales or scales_for(current)
    best = objective(current, scales, mode)
    trace, evaluations, attempts = [], 0, 0
    movable = [c['id'] for c in evaluator.logic['clusters'] if c['resource'] in RESOURCES]
    if max_moved_fraction is not None and not 0 < max_moved_fraction <= 1:
        raise ValueError('max_moved_fraction must be in (0, 1]')
    max_moved = (max(1, math.floor(len(movable) * max_moved_fraction))
                 if max_moved_fraction is not None else None)
    resources = [c['resource'] for c in evaluator.logic['clusters']]
    by_resource = {r: [n for n in movable if resources[n] == r] for r in RESOURCES}
    clock = time.monotonic()
    while movable and evaluations < budget and attempts < max(100, budget * 50):
        attempts += 1
        base = initial if policy == 'random_restart' else assignment
        base_state = original if policy == 'random_restart' else current
        proposal = base.copy()
        changed = set()
        count = rng.randint(1, 8) if policy == 'random_restart' else 1
        for _ in range(count):
            if policy == 'hotspot' and attempts % 4 != 0:
                # Refresh only on acceptance, cache the expensive priority computation.
                if 'priority' not in base_state:
                    rho = [d / c for d, c in zip(base_state['loads'], evaluator.capacities)]
                    pressure = [sum(max((rho[p] for p in base_state['routes'][i][0]), default=0.0)
                                    for i in evaluator.incident[n]) for n in movable]
                    base_state['priority'] = sorted(zip(pressure, movable), reverse=True)[:max(1, len(movable) // 4)]
                node = rng.choice(base_state['priority'])[1]
            else:
                node = rng.choice(movable)
            target = rng.randrange(len(evaluator.device['regions']))
            if rng.random() < 0.5:
                partners = [n for n in by_resource[resources[node]] if proposal[n] != proposal[node]]
                if partners:
                    other = rng.choice(partners)
                    proposal[node], proposal[other] = proposal[other], proposal[node]
                    changed.update((node, other))
            else:
                proposal[node] = target
                changed.add(node)
        changed = {n for n in changed if proposal[n] != base[n]}
        if (not changed or
                (max_moved is not None and
                 sum(proposal[n] != initial[n] for n in movable) > max_moved) or
                not evaluator.legal(proposal, fixed=initial)):
            continue
        candidate = evaluator.state(proposal, base_state, changed)
        evaluations += 1
        if candidate is not None:
            score = objective(candidate, scales, mode)
            if score < best - 1e-10:
                trace.append({'evaluation': evaluations, 'objective_before': best, 'objective_after': score,
                              'changes': [{'cluster': n, 'from': assignment[n], 'to': proposal[n]}
                                          for n in movable if assignment[n] != proposal[n]],
                              'cvar': candidate['cvar'], 'mean': candidate['mean'],
                              'wire': candidate['wire'], 'timing': candidate['timing']})
                assignment, current, best = proposal, candidate, score
        if progress and evaluations % 128 == 0:
            progress({'evaluations': evaluations, 'accepted': len(trace), 'objective': best})
    # Independently recompute accepted state, catching incremental-update errors.
    checked = evaluator.state(assignment)
    for key in ('cvar', 'mean', 'wire', 'timing'):
        if not math.isclose(checked[key], current[key], rel_tol=1e-8, abs_tol=1e-8):
            raise AssertionError(f'incremental {key} mismatch')
    return assignment, {'schema': 'tailcompile-v4-search-v1', 'mode': mode, 'policy': policy,
                        'max_moved_fraction': max_moved_fraction,
                        'budget': budget, 'evaluations': evaluations, 'attempts': attempts,
                        'accepted': len(trace), 'seed': seed, 'normalization': scales,
                        'seconds': time.monotonic() - clock,
                        'stop_reason': 'budget' if evaluations == budget else 'proposal_limit',
                        'initial': evaluator.report(original), 'final': evaluator.report(checked),
                        'trace': trace}
