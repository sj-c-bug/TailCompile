"""Deterministic pre-placement hypergraph TailRisk experiment core.

The cut-pressure formula follows dac_experiment_plan_tailcompile.md. Its
otherwise unspecified early criticality is fixed here to a unit-delay
topological-depth estimate on the mapped BLIF (never placement/routing data).
The recursive balanced cut-net partitioner is deterministic, heuristic, and
not a claim of optimal hypergraph partitioning.
"""

from __future__ import annotations

import math
import re
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

SCALES = (8, 16, 32, 64)
CONSTANTS = {'gnd', 'vcc', 'unconn', 'open', '0', '1', '$false', '$true'}
CLOCK_PORT = re.compile(r'^(clk|clock)(?:$|[_\d\[])', re.IGNORECASE)


@dataclass(frozen=True)
class Primitive:
    kind: str
    model: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    clocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Edge:
    name: str
    pins: tuple[int, ...]
    drivers: tuple[int, ...]
    sinks: tuple[int, ...]
    fanout: int
    criticality: float
    type_factor: float
    weight: float


def logical_lines(path: Path):
    pending = ''
    with path.open(errors='replace') as source:
        for raw in source:
            line = raw.strip()
            if line.endswith('\\'):
                pending += line[:-1] + ' '
                continue
            yield pending + line
            pending = ''
    if pending:
        yield pending


def model_ports(path: Path) -> tuple[str, dict[str, tuple[set[str], set[str]]]]:
    ports: dict[str, tuple[set[str], set[str]]] = {}
    top = ''
    current = ''
    for line in logical_lines(path):
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        if parts[0] == '.model':
            current = parts[1]
            if not top:
                top = current
            ports[current] = (set(), set())
        elif parts[0] == '.inputs' and current:
            ports[current][0].update(parts[1:])
        elif parts[0] == '.outputs' and current:
            ports[current][1].update(parts[1:])
        elif parts[0] == '.end':
            current = ''
    if not top:
        raise ValueError(f'BLIF has no .model: {path}')
    return top, ports


def kind_for(model: str) -> str:
    name = model.lower()
    if 'ram' in name or 'mem' in name:
        return 'bram'
    if any(tag in name for tag in ('mult', 'dsp', 'mac_', 'sop_', 'addition_fp')):
        return 'dsp'
    return 'other'


def parse_blif(path: Path) -> tuple[list[Primitive], dict]:
    top, ports = model_ports(path)
    primitives: list[Primitive] = []
    active = False
    unknown_ports: list[str] = []
    for line in logical_lines(path):
        if not line or line.startswith('#'):
            continue
        parts = line.split()
        op = parts[0]
        if op == '.model':
            active = parts[1] == top
            continue
        if op == '.end' and active:
            break
        if not active:
            continue
        if op == '.names' and len(parts) >= 2:
            if parts[-1] in CONSTANTS and len(parts) == 2:
                continue
            primitives.append(Primitive('lut', '.names', tuple(parts[1:-1]), (parts[-1],)))
        elif op == '.latch' and len(parts) >= 3:
            clocks = (parts[4],) if len(parts) >= 5 and parts[4] not in CONSTANTS else ()
            primitives.append(Primitive('ff', '.latch', (parts[1],), (parts[2],), clocks))
        elif op in ('.subckt', '.gate'):
            model = parts[1]
            if model not in ports:
                raise ValueError(f'unknown subcircuit model {model} in {path}')
            model_in, model_out = ports[model]
            inputs, outputs, clocks = [], [], []
            for term in parts[2:]:
                if '=' not in term:
                    continue
                port, net = term.split('=', 1)
                if net in CONSTANTS:
                    continue
                if port in model_in:
                    if CLOCK_PORT.match(port):
                        clocks.append(net)
                    else:
                        inputs.append(net)
                elif port in model_out:
                    outputs.append(net)
                else:
                    unknown_ports.append(f'{model}.{port}')
            primitives.append(Primitive(kind_for(model), model, tuple(inputs),
                                        tuple(outputs), tuple(clocks)))
    if unknown_ports:
        raise ValueError(f'undeclared subcircuit ports in {path}: {unknown_ports[:12]}')
    # Top-level ports are primitive IO terminals in the physical hypergraph.
    for name in sorted(ports[top][0]):
        if name not in CONSTANTS:
            primitives.append(Primitive('io_in', '.input', (), (name,)))
    for name in sorted(ports[top][1]):
        if name not in CONSTANTS:
            primitives.append(Primitive('io_out', '.output', (name,), ()))
    audit = {'top_model': top, 'model_count': len(ports),
             'primitive_count': len(primitives),
             'primitive_kinds': dict(sorted((kind, sum(p.kind == kind for p in primitives))
                                             for kind in {p.kind for p in primitives}))}
    return primitives, audit


def _combinational(p: Primitive) -> bool:
    return p.kind not in ('ff', 'bram', 'io_in', 'io_out') and not p.clocks


def build_edges(primitives: list[Primitive]) -> tuple[list[Edge], dict]:
    incidence: dict[str, set[int]] = defaultdict(set)
    drivers: dict[str, set[int]] = defaultdict(set)
    sinks: dict[str, set[int]] = defaultdict(set)
    sink_pin_count: dict[str, int] = defaultdict(int)
    clocks = {net for p in primitives for net in p.clocks}
    for node, p in enumerate(primitives):
        for net in p.inputs:
            if net not in CONSTANTS and net not in clocks:
                incidence[net].add(node)
                sinks[net].add(node)
                sink_pin_count[net] += 1
        for net in p.outputs:
            if net not in CONSTANTS and net not in clocks:
                incidence[net].add(node)
                drivers[net].add(node)

    # Unit-delay combinational DAG. Sequential cells are timing boundaries.
    successors = [set() for _ in primitives]
    indegree = [0] * len(primitives)
    for net, ds in drivers.items():
        if len(ds) != 1:
            continue
        driver = next(iter(ds))
        if not _combinational(primitives[driver]):
            continue
        for sink in sinks[net]:
            if sink != driver and _combinational(primitives[sink]) and sink not in successors[driver]:
                successors[driver].add(sink)
                indegree[sink] += 1
    queue = deque(i for i, p in enumerate(primitives) if _combinational(p) and indegree[i] == 0)
    arrival = [0] * len(primitives)
    order = []
    while queue:
        node = queue.popleft()
        order.append(node)
        for sink in successors[node]:
            arrival[sink] = max(arrival[sink], arrival[node] + 1)
            indegree[sink] -= 1
            if indegree[sink] == 0:
                queue.append(sink)
    cycle_nodes = sum(_combinational(p) for p in primitives) - len(order)
    if cycle_nodes:
        raise ValueError(f'{cycle_nodes} combinational primitives are in or downstream of a cycle')
    remaining = [0] * len(primitives)
    for node in reversed(order):
        remaining[node] = max((remaining[sink] + 1 for sink in successors[node]), default=0)

    provisional = []
    max_depth = 1
    for name, pins in incidence.items():
        if len(pins) < 2:
            continue
        ds = tuple(sorted(drivers[name]))
        ss = tuple(sorted(sinks[name]))
        depth = max((arrival[d] + 1 + remaining[s] for d in ds for s in ss if d != s),
                    default=1)
        max_depth = max(max_depth, depth)
        provisional.append((name, tuple(sorted(pins)), ds, ss,
                            sink_pin_count[name], depth))

    edges = []
    for name, pins, ds, ss, fanout, depth in provisional:
        criticality = depth / max_depth
        hetero = any(primitives[node].kind in ('dsp', 'bram') for node in pins)
        factor = 1.25 if hetero else 1.0
        weight = math.log1p(fanout) * (1 + 0.5 * criticality) * factor
        edges.append(Edge(name, pins, ds, ss, fanout, criticality, factor, weight))
    edges.sort(key=lambda e: e.name)
    audit = {'excluded_clock_nets': sorted(clocks),
             'excluded_constant_names': sorted(CONSTANTS),
             'connected_hyperedges': len(edges),
             'unit_delay_max_path_depth': max_depth,
             'combinational_cycle_nodes': cycle_nodes,
             'multi_driver_net_count': sum(len(e.drivers) > 1 for e in edges),
             'undriven_net_count': sum(not e.drivers for e in edges),
             'high_fanout_over_64_count': sum(e.fanout > 64 for e in edges)}
    return edges, audit


def incidence_index(n: int, edges: list[Edge]) -> list[list[int]]:
    result = [[] for _ in range(n)]
    for edge_id, edge in enumerate(edges):
        for node in edge.pins:
            result[node].append(edge_id)
    return result


def adjacency_index(n: int, edges: list[Edge]) -> list[list[int]]:
    # Only initialization uses a star projection; refinement and scoring use
    # the original hyperedges, including large fanout nets.
    result = [set() for _ in range(n)]
    for edge in edges:
        if len(edge.pins) > 64:
            continue
        anchor = edge.pins[0]
        for other in edge.pins[1:]:
            result[anchor].add(other)
            result[other].add(anchor)
    return [sorted(neighbors) for neighbors in result]


def balanced_bisect(nodes: list[int], edges: list[Edge], incident: list[list[int]],
                    neighbors: list[list[int]]) -> tuple[list[int], list[int], dict]:
    allowed = set(nodes)
    target = len(nodes) // 2
    degree = lambda node: sum(edges[e].weight for e in incident[node])
    seed = max(nodes, key=lambda node: (degree(node), -node))
    left = {seed}
    frontier = deque([seed])
    while frontier and len(left) < target:
        node = frontier.popleft()
        for other in neighbors[node]:
            if other in allowed and other not in left:
                left.add(other)
                frontier.append(other)
                if len(left) == target:
                    break
    if len(left) < target:
        for node in nodes:
            if node not in left:
                left.add(node)
                if len(left) == target:
                    break

    local: dict[int, tuple[int, ...]] = {}
    local_incident: dict[int, list[int]] = {node: [] for node in nodes}
    for edge_id in {e for node in nodes for e in incident[node]}:
        pins = tuple(node for node in edges[edge_id].pins if node in allowed)
        if len(pins) >= 2:
            local[edge_id] = pins
            for node in pins:
                local_incident[node].append(edge_id)
    count_left = {edge_id: sum(node in left for node in pins) for edge_id, pins in local.items()}

    def is_cut(edge_id: int, count: int) -> bool:
        return 0 < count < len(local[edge_id])

    def cost() -> float:
        return sum(edges[edge_id].weight for edge_id, count in count_left.items()
                   if is_cut(edge_id, count))

    initial_cost = cost()
    swaps = 0
    for _ in range(12):
        candidates = [[], []]
        for node in nodes:
            side = 0 if node in left else 1
            change = -1 if side == 0 else 1
            gain = 0.0
            for edge_id in local_incident[node]:
                old = count_left[edge_id]
                gain += edges[edge_id].weight * (
                    int(is_cut(edge_id, old)) - int(is_cut(edge_id, old + change)))
            candidates[side].append((gain, node))
        a = sorted(candidates[0], key=lambda pair: (-pair[0], pair[1]))[:24]
        b = sorted(candidates[1], key=lambda pair: (-pair[0], pair[1]))[:24]
        best_gain, best_pair = 1e-9, None
        for _, from_left in a:
            left_edges = set(local_incident[from_left])
            for _, from_right in b:
                gain = 0.0
                for edge_id in left_edges | set(local_incident[from_right]):
                    old = count_left[edge_id]
                    new = old - int(edge_id in left_edges) + int(edge_id in local_incident[from_right])
                    gain += edges[edge_id].weight * (
                        int(is_cut(edge_id, old)) - int(is_cut(edge_id, new)))
                pair = (from_left, from_right)
                if gain > best_gain + 1e-9 or (abs(gain - best_gain) <= 1e-9 and best_pair is not None and pair < best_pair):
                    best_gain, best_pair = gain, pair
        if best_pair is None:
            break
        from_left, from_right = best_pair
        left.remove(from_left)
        left.add(from_right)
        for edge_id in set(local_incident[from_left]) | set(local_incident[from_right]):
            count_left[edge_id] += -int(edge_id in local_incident[from_left]) + int(edge_id in local_incident[from_right])
        swaps += 1
    left_nodes = [node for node in nodes if node in left]
    right_nodes = [node for node in nodes if node not in left]
    if abs(len(left_nodes) - len(right_nodes)) > 1:
        raise AssertionError('unbalanced bisection')
    final_cost = cost()
    if final_cost > initial_cost + 1e-7:
        raise AssertionError('cut objective increased after refinement')
    return left_nodes, right_nodes, {
        'vertices': len(nodes), 'initial_cut_weight': round(initial_cost, 9),
        'final_cut_weight': round(final_cost, 9), 'accepted_swaps': swaps,
        'left_size': len(left_nodes), 'right_size': len(right_nodes),
    }


def recursive_partitions(n: int, edges: list[Edge]) -> tuple[dict[int, list[int]], list[dict]]:
    incident = incidence_index(n, edges)
    neighbors = adjacency_index(n, edges)
    groups = [list(range(n))]
    memberships: dict[int, list[int]] = {}
    trace = []
    while len(groups) < max(SCALES):
        next_groups = []
        for group_id, group in enumerate(groups):
            left, right, record = balanced_bisect(group, edges, incident, neighbors)
            record.update({'parent_scale': len(groups), 'parent_group': group_id})
            trace.append(record)
            next_groups.extend((left, right))
        groups = next_groups
        if len(groups) in SCALES and n // len(groups) >= 16:
            membership = [0] * n
            for group_id, group in enumerate(groups):
                for node in group:
                    membership[node] = group_id
            memberships[len(groups)] = membership
    return memberships, trace


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    at = (len(ordered) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (at - lo)


def rent_exponent(points: list[tuple[int, int]]) -> float | None:
    pairs = [(math.log(size), math.log(cuts)) for size, cuts in points if size > 0 and cuts > 0]
    if len(pairs) < 2:
        return None
    mx = sum(x for x, _ in pairs) / len(pairs)
    my = sum(y for _, y in pairs) / len(pairs)
    denominator = sum((x - mx) ** 2 for x, _ in pairs)
    return sum((x - mx) * (y - my) for x, y in pairs) / denominator if denominator else None


def profile(n: int, edges: list[Edge], memberships: dict[int, list[int]]) -> tuple[dict, list[dict], dict]:
    features = {'primitive_count': n, 'connected_net_count': len(edges),
                'mean_fanout': round(sum(e.fanout for e in edges) / max(len(edges), 1), 6),
                'max_fanout': max((e.fanout for e in edges), default=0)}
    group_rows = []
    rent_points = []
    sample = {}
    for scale, member in sorted(memberships.items()):
        pressure = [0.0] * scale
        cuts = [0] * scale
        sizes = [0] * scale
        for group in member:
            sizes[group] += 1
        for edge in edges:
            touched = {member[node] for node in edge.pins}
            if len(touched) > 1:
                for group in touched:
                    pressure[group] += edge.weight
                    cuts[group] += 1
        median = percentile(pressure, 0.5)
        tail_count = max(1, math.ceil(0.05 * scale))
        cvar = sum(sorted(pressure, reverse=True)[:tail_count]) / tail_count
        prefix = f'k{scale}_'
        values = {
            'median': median, 'mean_cut_pressure': sum(pressure) / scale,
            'p90': percentile(pressure, 0.90), 'p95': percentile(pressure, 0.95),
            'p99': percentile(pressure, 0.99), 'maximum': max(pressure),
            'top5_mean': cvar, 'cvar95': cvar,
            'variance': sum((x - sum(pressure) / scale) ** 2 for x in pressure) / scale,
            'mean_normalized_excess': sum(max(0.0, x / max(median, 1e-12) - 1) for x in pressure) / scale,
            'p95_to_median': percentile(pressure, 0.95) / max(median, 1e-12),
        }
        features.update({prefix + key: round(value, 9) for key, value in values.items()})
        for group in range(scale):
            rent_points.append((sizes[group], cuts[group]))
            group_rows.append({'scale': scale, 'group': group, 'size': sizes[group],
                               'boundary_net_count': cuts[group],
                               'cut_pressure': round(pressure[group], 9),
                               'local_tail_risk': round(pressure[group] / max(median, 1e-12), 9)})
        target = max(range(scale), key=lambda group: (pressure[group], -group))
        contributions = []
        for edge in edges:
            pin_groups = {member[node] for node in edge.pins}
            if target in pin_groups and len(pin_groups) > 1:
                inside = [node for node in edge.pins if member[node] == target]
                outside = [node for node in edge.pins if member[node] != target]
                contributions.append({'net': edge.name, 'weight': round(edge.weight, 9),
                                      'inside_nodes': inside[:3], 'outside_nodes': outside[:3],
                                      'fanout': edge.fanout, 'criticality': round(edge.criticality, 6),
                                      'type_factor': edge.type_factor})
        sample[scale] = {'highest_pressure_group': target,
                         'pressure_from_all_cut_nets': round(sum(x['weight'] for x in contributions), 6),
                         'recorded_pressure': round(pressure[target], 6),
                         'cut_net_count': len(contributions),
                         'top_cut_nets': sorted(contributions, key=lambda x: (-x['weight'], x['net']))[:10]}
    slope = rent_exponent(rent_points)
    features['global_rent_exponent'] = round(slope, 9) if slope is not None else ''
    return features, group_rows, sample
