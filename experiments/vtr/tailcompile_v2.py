#!/usr/bin/env python3
"""Generic pack-aware TailCompile v2 with timing and hierarchical region grids."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import tailcompile_v0 as v0
import tailcompile_v1 as v1
from tailrisk_core import Edge, build_edges, parse_blif, profile


OBJECTIVE_WEIGHTS = {'congestion': 1.0, 'wirelength': 0.15, 'timing': 0.25}
RESOURCE_TYPES = ('clb', 'memory', 'dsp_top')


def allocate_targets(total: int, capacities: list[int]) -> list[int]:
    if total > sum(capacities):
        raise ValueError(f'resource demand {total} exceeds device capacity {sum(capacities)}')
    if total == 0:
        return [0] * len(capacities)
    raw = [total * capacity / sum(capacities) for capacity in capacities]
    targets = [min(capacity, math.floor(value)) for capacity, value in zip(capacities, raw)]
    while sum(targets) < total:
        candidates = [index for index in range(len(capacities)) if targets[index] < capacities[index]]
        chosen = max(candidates, key=lambda index: (raw[index] - targets[index],
                                                    capacities[index] - targets[index], -index))
        targets[chosen] += 1
    return targets


def contract_edges(primitives, atom_to_cluster: dict[str, int]) -> tuple[list[Edge], dict]:
    atom_edges, audit = build_edges(primitives)
    result = []
    for edge in atom_edges:
        mapped = {node: atom_to_cluster[v0.primitive_name(primitives[node])]
                  for node in edge.pins
                  if v0.primitive_name(primitives[node]) in atom_to_cluster}
        pins = tuple(sorted(set(mapped.values())))
        if len(pins) < 2:
            continue
        drivers = tuple(sorted({mapped[node] for node in edge.drivers if node in mapped}))
        sinks = tuple(sorted({mapped[node] for node in edge.sinks if node in mapped}))
        result.append(Edge(edge.name, pins, drivers, sinks, edge.fanout,
                           edge.criticality, edge.type_factor, edge.weight))
    return result, audit


def parse_packed_clusters(path: Path, primitive_names: set[str]) -> tuple[list[dict], dict[str, int]]:
    """Recover atoms at any pb depth, including hierarchical RAM/DSP atoms."""
    root = ET.parse(path).getroot()
    valid_names = primitive_names | {'gnd', 'vcc'}
    clusters, atom_to_cluster = [], {}
    for cluster_id, block in enumerate(root.findall('block')):
        resource = block.get('instance', '').split('[', 1)[0]
        seen, atoms = set(), []
        for child in block.iter('block'):
            name = child.get('name')
            if name in valid_names and name not in seen:
                seen.add(name)
                atoms.append(name)
        if not atoms:
            raise ValueError(f'packed cluster has no mapped atoms: {block.get("instance")}')
        for atom in atoms:
            if atom in atom_to_cluster:
                raise ValueError(f'atom appears in two packed clusters: {atom}')
            atom_to_cluster[atom] = cluster_id
        clusters.append({'id': cluster_id, 'name': block.get('name'),
                         'instance': block.get('instance'), 'resource': resource,
                         'atoms': atoms})
    return clusters, atom_to_cluster


def shape_membership(clusters: list[dict], edges: list[Edge], memberships: dict[int, list[int]],
                     scale: int, device: dict) -> tuple[list[int], dict]:
    member = memberships[scale].copy()
    neighbors = [defaultdict(float) for _ in clusters]
    for edge in edges:
        for index, left in enumerate(edge.pins):
            for right in edge.pins[index + 1:]:
                neighbors[left][right] += edge.weight
                neighbors[right][left] += edge.weight

    target_by_resource, moves = {}, []
    for resource in RESOURCE_TYPES:
        capacities = [int(region['tile_counts'].get(resource, 0)) for region in device['regions']]
        total = sum(cluster['resource'] == resource for cluster in clusters)
        targets = allocate_targets(total, capacities)
        target_by_resource[resource] = targets
        counts = [sum(cluster['resource'] == resource and member[node] == group
                      for node, cluster in enumerate(clusters)) for group in range(scale)]
        while counts != targets:
            sources = [group for group in range(scale) if counts[group] > targets[group]]
            destinations = [group for group in range(scale) if counts[group] < targets[group]]
            if not sources or not destinations:
                raise AssertionError({'resource': resource, 'counts': counts, 'targets': targets})
            best = None
            for source in sources:
                for node, cluster in enumerate(clusters):
                    if member[node] != source or cluster['resource'] != resource:
                        continue
                    source_affinity = sum(value for other, value in neighbors[node].items()
                                          if member[other] == source)
                    for destination in destinations:
                        destination_affinity = sum(value for other, value in neighbors[node].items()
                                                   if member[other] == destination)
                        candidate = (destination_affinity - source_affinity,
                                     -abs(counts[destination] + 1 - targets[destination]),
                                     -node, source, destination, node)
                        if best is None or candidate > best:
                            best = candidate
            if best is None:
                raise AssertionError(f'no movable {resource} cluster')
            source, destination, node = best[-3:]
            member[node] = destination
            counts[source] -= 1
            counts[destination] += 1
            moves.append({'cluster': node, 'resource': resource,
                          'from': source, 'to': destination})

    io_counts = [0] * scale
    for node, cluster in enumerate(clusters):
        if cluster['resource'] != 'io':
            continue
        affinity = [sum(value for other, value in neighbors[node].items()
                        if clusters[other]['resource'] != 'io' and member[other] == group)
                    for group in range(scale)]
        chosen = max(range(scale), key=lambda group: (affinity[group], -io_counts[group], -group))
        member[node] = chosen
        io_counts[chosen] += 1

    for finer_scale, old in list(memberships.items()):
        if finer_scale <= scale:
            continue
        factor = finer_scale // scale
        memberships[finer_scale] = [member[node] * factor + old[node] % factor
                                     for node in range(len(clusters))]
    memberships[scale] = member
    final = {resource: [sum(cluster['resource'] == resource and member[node] == group
                            for node, cluster in enumerate(clusters))
                        for group in range(scale)] for resource in RESOURCE_TYPES}
    return member, {'targets': target_by_resource, 'final': final,
                    'move_count': len(moves), 'moves': moves,
                    'io_clusters_per_group': io_counts}


def build_logic_ir(design: str, blif: Path, packed_net: Path, device: dict, scale: int) -> dict:
    primitives, parser_audit = parse_blif(blif)
    primitive_names = {v0.primitive_name(item) for item in primitives}
    clusters, atom_to_cluster = parse_packed_clusters(packed_net, primitive_names)
    if set(atom_to_cluster) - primitive_names != {'gnd', 'vcc'}:
        raise AssertionError('packed net contains unexpected non-constant atoms')
    absorbed_atoms = sorted(primitive_names - set(atom_to_cluster))
    edges, edge_audit = contract_edges(primitives, atom_to_cluster)
    weights = [0 if cluster['resource'] == 'io' else 1 for cluster in clusters]
    memberships, trace = v1.recursive_cluster_partitions(weights, edges)
    if scale not in memberships:
        raise ValueError(f'unsupported scale {scale}; available {sorted(memberships)}')
    member, shaping = shape_membership(clusters, edges, memberships, scale, device)
    features, group_rows, _ = profile(len(clusters), edges, memberships)
    row_lookup = {(int(row['scale']), int(row['group'])): row for row in group_rows}

    traffic = [[0.0] * scale for _ in range(scale)]
    timing_traffic = [[0.0] * scale for _ in range(scale)]
    for edge in edges:
        touched = sorted({member[node] for node in edge.pins})
        if len(touched) < 2:
            continue
        divisor = len(touched) * (len(touched) - 1) / 2
        for index, left in enumerate(touched):
            for right in touched[index + 1:]:
                value = edge.weight / divisor
                timing_value = value * edge.criticality
                traffic[left][right] += value
                traffic[right][left] += value
                timing_traffic[left][right] += timing_value
                timing_traffic[right][left] += timing_value

    groups = []
    available_scales = sorted(memberships)
    for group in range(scale):
        cluster_ids = [node for node, assigned in enumerate(member) if assigned == group]
        counts = Counter(clusters[node]['resource'] for node in cluster_ids)
        components = [(0.5, float(row_lookup[scale, group]['local_tail_risk']))]
        if scale * 2 in available_scales:
            values = [float(row_lookup[scale * 2, child]['local_tail_risk'])
                      for child in range(2 * group, 2 * group + 2)]
            components.append((0.3, max(values)))
        if scale * 4 in available_scales:
            values = [float(row_lookup[scale * 4, child]['local_tail_risk'])
                      for child in range(4 * group, 4 * group + 4)]
            components.append((0.2, max(values)))
        total_weight = sum(weight for weight, _ in components)
        groups.append({
            'id': group, 'cluster_ids': cluster_ids,
            'cluster_counts': dict(sorted(counts.items())),
            'atom_count': sum(len(clusters[node]['atoms']) for node in cluster_ids),
            'mean_score': float(row_lookup[scale, group]['cut_pressure']),
            'tail_components': [{'weight': weight, 'value': value} for weight, value in components],
            'tail_score': sum(weight * value for weight, value in components) / total_weight,
        })
    for cluster in clusters:
        cluster['group'] = member[cluster['id']]
    return {
        'schema': 'tailcompile-logic-ir-v2', 'design': design, 'scale': scale,
        'mapped_blif': str(blif), 'mapped_blif_sha256': v0.digest(blif),
        'packed_net': str(packed_net), 'packed_net_sha256': v0.digest(packed_net),
        'parser_audit': parser_audit, 'hypergraph_audit': edge_audit,
        'packed_atom_audit': {'mapped_blif_atom_count': len(primitive_names),
                              'packed_atom_count_excluding_constants': len(atom_to_cluster) - 2,
                              'absorbed_or_swept_atom_count': len(absorbed_atoms),
                              'absorbed_or_swept_atom_sample': absorbed_atoms[:32]},
        'partition_trace': trace, 'capacity_shaping': shaping,
        'contracted_global_features': features, 'contracted_edge_count': len(edges),
        'groups': groups, 'traffic_matrix': traffic, 'timing_traffic_matrix': timing_traffic,
        'clusters': clusters, 'cluster_count': len(clusters), 'atom_count': len(atom_to_cluster),
        'resource_totals': dict(sorted(Counter(cluster['resource'] for cluster in clusters).items())),
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = quantile * (len(ordered) - 1)
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else (ordered[low] * (high - position)
                                             + ordered[high] * (position - low))


def binding_objective(assignment: tuple[int, ...], scores: list[float], logic: dict,
                      device: dict) -> dict:
    scale, regions = len(assignment), device['regions']
    overflow_by_resource = {}
    for resource in RESOURCE_TYPES:
        overflow_by_resource[resource] = sum(
            max(0, int(logic['groups'][group]['cluster_counts'].get(resource, 0))
                - int(regions[assignment[group]]['tile_counts'].get(resource, 0))) ** 2
            for group in range(scale))
    overflow = sum(overflow_by_resource.values())

    score_norm = v0.normalize(scores)
    local_supply = []
    for region in regions:
        directional = sum(region[key] for key in
                          ('escape_left', 'escape_right', 'escape_down', 'escape_up'))
        local_supply.append(directional + sum(region['segment_supply'].values()))
    supply_norm = v0.normalize(local_supply)
    risk_supply = sum(score_norm[group] * (1 - supply_norm[assignment[group]])
                      for group in range(scale)) / scale

    cut_supply = {(min(int(row['region_a']), int(row['region_b'])),
                   max(int(row['region_a']), int(row['region_b']))): float(row['count'])
                  for row in device['inter_region_rr_edge_counts_per_channel_width']}
    max_cut = max(cut_supply.values(), default=1.0)
    max_direction = max((region[key] for region in regions for key in
                         ('escape_left', 'escape_right', 'escape_down', 'escape_up')), default=1.0)
    max_segment = max((value for region in regions for value in region['segment_supply'].values()),
                      default=1.0)
    traffic_total = max(sum(sum(row) for row in logic['traffic_matrix']) / 2, 1e-12)
    timing_total = max(sum(sum(row) for row in logic['timing_traffic_matrix']) / 2, 1e-12)
    ratios, wire_cost, timing_cost = [], 0.0, 0.0
    max_distance = max(device['region_grid']['columns'] + device['region_grid']['rows'] - 2, 1)
    for left in range(scale):
        region_left = regions[assignment[left]]
        for right in range(left + 1, scale):
            traffic = logic['traffic_matrix'][left][right]
            if traffic <= 0:
                continue
            region_right = regions[assignment[right]]
            dx = abs(region_left['column'] - region_right['column'])
            dy = abs(region_left['row'] - region_right['row'])
            distance = dx + dy
            wire_cost += traffic / traffic_total * distance / max_distance
            timing_cost += (logic['timing_traffic_matrix'][left][right] / timing_total
                            * distance / max_distance)
            pair = tuple(sorted((assignment[left], assignment[right])))
            pair_supply = cut_supply.get(pair, min(cut_supply.values(), default=1e-12))
            ratios.append((traffic / traffic_total) / max(pair_supply / max_cut, 1e-12))
            horizontal, vertical = traffic * dx / max(distance, 1), traffic * dy / max(distance, 1)
            for source, target in ((region_left, region_right), (region_right, region_left)):
                if horizontal:
                    key = 'escape_right' if target['column'] > source['column'] else 'escape_left'
                    ratios.append((horizontal / traffic_total) / max(source[key] / max_direction, 1e-12))
                if vertical:
                    key = 'escape_up' if target['row'] > source['row'] else 'escape_down'
                    ratios.append((vertical / traffic_total) / max(source[key] / max_direction, 1e-12))
            segment_name = 'L16' if distance > 1 else 'L4'
            segment_supply = (region_left['segment_supply'].get(segment_name, 0)
                              + region_right['segment_supply'].get(segment_name, 0)) / 2
            ratios.append((traffic / traffic_total) / max(segment_supply / max_segment, 1e-12))
    count = max(1, math.ceil(len(ratios) * 0.25))
    cvar = sum(sorted(ratios, reverse=True)[:count]) / count
    congestion = 0.5 * risk_supply + 0.5 * cvar
    scalar = (OBJECTIVE_WEIGHTS['congestion'] * congestion
              + OBJECTIVE_WEIGHTS['wirelength'] * wire_cost
              + OBJECTIVE_WEIGHTS['timing'] * timing_cost)
    return {
        'capacity_overflow_clusters_squared': overflow,
        'overflow_by_resource': overflow_by_resource,
        'risk_supply': risk_supply,
        'rrg_physical_congestion': {'mean': sum(ratios) / max(len(ratios), 1),
                                    'p95': percentile(ratios, 0.95),
                                    'max': max(ratios, default=0.0),
                                    'cvar_top25': cvar, 'sample_count': len(ratios)},
        'congestion_cost': congestion, 'wirelength_proxy': wire_cost,
        'timing_proxy': timing_cost, 'weighted_scalar': scalar,
    }


def solve_binding(logic: dict, device: dict, mode: str) -> dict:
    scale = int(logic['scale'])
    score_key = {'rrg_only': None, 'mean_rrg': 'mean_score', 'tail_rrg': 'tail_score'}[mode]
    scores = ([0.0] * scale if score_key is None
              else [float(group[score_key]) for group in logic['groups']])

    def key(assignment):
        objective = binding_objective(tuple(assignment), scores, logic, device)
        return (objective['capacity_overflow_clusters_squared'],
                objective['weighted_scalar'], tuple(assignment)), objective

    if scale <= 8:
        candidates = []
        for assignment in itertools.permutations(range(scale)):
            candidate_key, objective = key(assignment)
            candidates.append((candidate_key, tuple(assignment), objective))
        _, assignment, objective = min(candidates, key=lambda item: item[0])
        feasible_count = sum(item[2]['capacity_overflow_clusters_squared'] == 0 for item in candidates)
        search = f'exhaustive-{scale}-factorial'
    else:
        assignment = list(range(scale))
        _, objective = key(assignment)
        swaps = 0
        while True:
            best = (objective['capacity_overflow_clusters_squared'], objective['weighted_scalar'])
            best_pair = None
            for left in range(scale):
                for right in range(left + 1, scale):
                    proposal = assignment.copy()
                    proposal[left], proposal[right] = proposal[right], proposal[left]
                    _, candidate = key(proposal)
                    candidate_key = (candidate['capacity_overflow_clusters_squared'],
                                     candidate['weighted_scalar'])
                    if candidate_key < best:
                        best, best_pair = candidate_key, (left, right)
            if best_pair is None:
                break
            left, right = best_pair
            assignment[left], assignment[right] = assignment[right], assignment[left]
            _, objective = key(assignment)
            swaps += 1
        assignment, feasible_count = tuple(assignment), None
        search = f'capacity-first-pair-swap-local-search({swaps}-swaps)'
    return {
        'schema': 'tailcompile-binding-v2', 'mode': mode, 'score_key': score_key,
        'search': search, 'objective_weights': OBJECTIVE_WEIGHTS,
        'objective_policy': 'lexicographic(exact_multi_resource_capacity,weighted_congestion_wire_timing)',
        'assignment': list(assignment), 'objective': objective,
        'feasible_candidate_count': feasible_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--design', required=True)
    parser.add_argument('--blif', type=Path, required=True)
    parser.add_argument('--packed-net', type=Path, required=True)
    parser.add_argument('--rr-graph', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--region-grid', default='2x2', choices=['2x2', '2x4', '4x4'])
    parser.add_argument('--timing-weight', type=float, default=0.25,
                        help='Weight of the early critical-net traffic term (0 disables timing protection).')
    args = parser.parse_args()
    if args.timing_weight < 0:
        parser.error('--timing-weight must be non-negative')
    OBJECTIVE_WEIGHTS['timing'] = args.timing_weight
    columns, rows = map(int, args.region_grid.split('x'))
    scale = columns * rows
    for path in (args.blif, args.packed_net, args.rr_graph):
        if not path.is_file():
            parser.error(f'missing input: {path}')
    args.out.mkdir(parents=True, exist_ok=True)

    v0.SCALE, v0.REGION_COLUMNS, v0.REGION_ROWS = scale, columns, rows
    device = v0.parse_device_ir(args.rr_graph)
    device.update({'schema': 'tailcompile-device-ir-v2', 'design': args.design,
                   'region_grid': {'columns': columns, 'rows': rows, 'count': scale}})
    logic = build_logic_ir(args.design, args.blif, args.packed_net, device, scale)
    (args.out / 'device_ir.json').write_text(json.dumps(device, indent=2) + '\n')
    (args.out / 'logic_ir.json').write_text(json.dumps(logic, indent=2) + '\n')

    methods = {}
    for mode in ('rrg_only', 'mean_rrg', 'tail_rrg'):
        binding = solve_binding(logic, device, mode)
        (args.out / f'{mode}_binding.json').write_text(json.dumps(binding, indent=2) + '\n')
        lowering = v1.lower_fplace(logic, device, binding, args.out / f'{mode}.fplace')
        (args.out / f'{mode}_lowering.json').write_text(json.dumps(lowering, indent=2) + '\n')
        methods[mode] = {'assignment': binding['assignment'], 'objective': binding['objective'],
                         'fplace_sha256': lowering['sha256']}
    evidence_names = ['device_ir.json', 'logic_ir.json']
    for mode in ('rrg_only', 'mean_rrg', 'tail_rrg'):
        evidence_names.extend([f'{mode}_binding.json', f'{mode}_lowering.json', f'{mode}.fplace'])
    summary = {
        'schema': 'tailcompile-v2-build', 'design': args.design,
        'region_grid': args.region_grid, 'objective_weights': OBJECTIVE_WEIGHTS,
        'methods': methods,
        'evidence_sha256': {name: v0.digest(args.out / name) for name in evidence_names},
    }
    (args.out / 'build_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
