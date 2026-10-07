#!/usr/bin/env python3
"""TailCompile v1: pack-aware demand, directional RRG supply, and cluster lowering."""

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
from tailrisk_core import Edge, build_edges, parse_blif, profile


SCALE = 4
REGION_COLUMNS = 2
REGION_ROWS = 2
DEFAULT_BLIF = HERE / 'runs/pilot_v1/sha/full-seed1-w300/sha.pre-vpr.blif'
DEFAULT_PACKED_NET = HERE / 'runs/pilot_v1/sha/full-seed1-w300/sha.net'
DEFAULT_RRG = HERE / 'runs/pilot_v1/sha/route-seed1-w100-rr/sha.rr_graph.xml.gz'
DEFAULT_OUT = HERE / 'tailcompile_v1/sha'


def digest(path: Path) -> str:
    return v0.digest(path)


def parse_packed_clusters(path: Path) -> tuple[list[dict], dict[str, int]]:
    root = ET.parse(path).getroot()
    clusters, atom_to_cluster = [], {}
    for cluster_id, block in enumerate(root.findall('block')):
        instance = block.get('instance', '')
        resource = instance.split('[', 1)[0]
        atoms = [child.get('name') for child in block.iter('block')
                 if not child.findall('block') and child.get('name') not in (None, 'open')]
        if not atoms:
            raise ValueError(f'packed cluster has no atoms: {instance}')
        for atom in atoms:
            if atom in atom_to_cluster:
                raise ValueError(f'atom appears in two packed clusters: {atom}')
            atom_to_cluster[atom] = cluster_id
        clusters.append({'id': cluster_id, 'name': block.get('name'),
                         'instance': instance, 'resource': resource, 'atoms': atoms})
    return clusters, atom_to_cluster


def recursive_cluster_partitions(weights: list[int], edges: list[Edge]) -> tuple[dict[int, list[int]], list[dict]]:
    incident = [[] for _ in weights]
    neighbors = [defaultdict(float) for _ in weights]
    for edge_id, edge in enumerate(edges):
        for node in edge.pins:
            incident[node].append(edge_id)
        if len(edge.pins) <= 64:
            for index, left in enumerate(edge.pins):
                for right in edge.pins[index + 1:]:
                    neighbors[left][right] += edge.weight
                    neighbors[right][left] += edge.weight

    def bisect(nodes: list[int]) -> tuple[list[int], list[int], dict]:
        total = sum(weights[node] for node in nodes)
        target = total / 2
        allowed = set(nodes)
        degree = lambda node: sum(edges[edge_id].weight for edge_id in incident[node])
        seed = max(nodes, key=lambda node: (degree(node), weights[node], -node))
        left, left_weight = {seed}, weights[seed]
        while len(left) < len(nodes) - 1:
            remaining = [node for node in nodes if node not in left]
            fitting = [node for node in remaining if left_weight + weights[node] <= math.ceil(target)]
            if not fitting:
                break
            candidate = max(fitting, key=lambda node: (
                sum(value for other, value in neighbors[node].items() if other in left),
                weights[node], degree(node), -node))
            left.add(candidate)
            left_weight += weights[candidate]
            if left_weight >= math.floor(target) and all(weights[node] > 0 for node in remaining):
                break
        # Attach zero-capacity IO clusters by connectivity without changing CLB balance.
        for node in nodes:
            if node not in left and weights[node] == 0:
                affinity_left = sum(value for other, value in neighbors[node].items() if other in left)
                affinity_right = sum(value for other, value in neighbors[node].items()
                                     if other in allowed and other not in left)
                if affinity_left > affinity_right:
                    left.add(node)
        left_nodes = [node for node in nodes if node in left]
        right_nodes = [node for node in nodes if node not in left]
        if not left_nodes or not right_nodes:
            midpoint = len(nodes) // 2
            left_nodes, right_nodes = nodes[:midpoint], nodes[midpoint:]
            left_weight = sum(weights[node] for node in left_nodes)
        cut = sum(edge.weight for edge in edges
                  if any(node in left for node in edge.pins)
                  and any(node in allowed and node not in left for node in edge.pins))
        return left_nodes, right_nodes, {
            'node_count': len(nodes), 'clb_demand': total,
            'left_clb_demand': left_weight, 'right_clb_demand': total - left_weight,
            'relative_imbalance': abs(total - 2 * left_weight) / max(total, 1),
            'cut_weight': cut,
        }

    groups, memberships, trace = [list(range(len(weights)))], {}, []
    while len(groups) < 32:
        next_groups = []
        for group_id, nodes in enumerate(groups):
            left, right, record = bisect(nodes)
            record.update({'parent_scale': len(groups), 'parent_group': group_id})
            trace.append(record)
            next_groups.extend((left, right))
        groups = next_groups
        if len(groups) in (4, 8, 16, 32):
            member = [0] * len(weights)
            for group_id, nodes in enumerate(groups):
                for node in nodes:
                    member[node] = group_id
            memberships[len(groups)] = member
    return memberships, trace


def build_logic_ir(blif: Path, packed_net: Path) -> dict:
    primitives, parser_audit = parse_blif(blif)
    atom_edges, edge_audit = build_edges(primitives)
    clusters, atom_to_cluster = parse_packed_clusters(packed_net)
    primitive_names = {v0.primitive_name(item) for item in primitives}
    if set(atom_to_cluster) - primitive_names != {'gnd', 'vcc'}:
        raise AssertionError('packed atoms do not match mapped BLIF atoms plus constants')
    if primitive_names - set(atom_to_cluster):
        raise AssertionError('mapped BLIF atom missing from packed netlist')

    contracted_edges = []
    for edge in atom_edges:
        pins = tuple(sorted({atom_to_cluster[v0.primitive_name(primitives[node])] for node in edge.pins}))
        if len(pins) < 2:
            continue
        drivers = tuple(sorted({atom_to_cluster[v0.primitive_name(primitives[node])] for node in edge.drivers}))
        sinks = tuple(sorted({atom_to_cluster[v0.primitive_name(primitives[node])] for node in edge.sinks}))
        contracted_edges.append(Edge(edge.name, pins, drivers, sinks, edge.fanout,
                                     edge.criticality, edge.type_factor, edge.weight))

    weights = [1 if cluster['resource'] == 'clb' else 0 for cluster in clusters]
    memberships, trace = recursive_cluster_partitions(weights, contracted_edges)

    # A rectangular 2x2 split of this heterogeneous architecture has exact CLB
    # capacities [24, 36, 24, 36].  Shape four logical groups to a deterministic
    # feasible demand vector rather than pretending all regions have equal area.
    member = memberships[SCALE]
    target_clb = [24, 32, 24, 31]
    neighbors = [defaultdict(float) for _ in clusters]
    for edge in contracted_edges:
        for index, left in enumerate(edge.pins):
            for right in edge.pins[index + 1:]:
                neighbors[left][right] += edge.weight
                neighbors[right][left] += edge.weight
    counts = [sum(weights[node] for node, group in enumerate(member) if group == group_id)
              for group_id in range(SCALE)]
    moves = []
    while counts != target_clb:
        sources = [group for group in range(SCALE) if counts[group] > target_clb[group]]
        destinations = [group for group in range(SCALE) if counts[group] < target_clb[group]]
        if not sources or not destinations:
            raise AssertionError({'unable_to_shape_cluster_demand': counts, 'target': target_clb})
        best = None
        for source in sources:
            for node, group in enumerate(member):
                if group != source or weights[node] != 1:
                    continue
                source_affinity = sum(value for other, value in neighbors[node].items()
                                      if member[other] == source)
                for destination in destinations:
                    destination_affinity = sum(value for other, value in neighbors[node].items()
                                           if member[other] == destination)
                    candidate = (destination_affinity - source_affinity,
                                 -abs((counts[destination] + 1) - target_clb[destination]),
                                 -node, source, destination, node)
                    if best is None or candidate > best:
                        best = candidate
        if best is None:
            raise AssertionError('no movable CLB cluster found')
        source, destination, node = best[-3:]
        member[node] = destination
        counts[source] -= 1
        counts[destination] += 1
        moves.append({'cluster': node, 'from': source, 'to': destination})

    io_counts = [0] * SCALE
    for node, cluster in enumerate(clusters):
        if cluster['resource'] != 'io':
            continue
        affinity = [sum(value for other, value in neighbors[node].items()
                        if weights[other] == 1 and member[other] == group)
                    for group in range(SCALE)]
        chosen = max(range(SCALE), key=lambda group: (affinity[group], -io_counts[group], -group))
        member[node] = chosen
        io_counts[chosen] += 1

    # Preserve a nested hierarchy after capacity shaping so the multiscale tail
    # children remain children of their final k=4 parent.
    for scale in (8, 16, 32):
        factor = scale // SCALE
        old = memberships[scale]
        memberships[scale] = [member[node] * factor + old[node] % factor
                              for node in range(len(clusters))]
    memberships[SCALE] = member
    features, group_rows, _ = profile(len(clusters), contracted_edges, memberships)
    row_lookup = {(int(row['scale']), int(row['group'])): row for row in group_rows}

    traffic = [[0.0] * SCALE for _ in range(SCALE)]
    for edge in contracted_edges:
        touched = sorted({member[node] for node in edge.pins})
        if len(touched) > 1:
            divisor = len(touched) * (len(touched) - 1) / 2
            for index, left in enumerate(touched):
                for right in touched[index + 1:]:
                    traffic[left][right] += edge.weight / divisor
                    traffic[right][left] += edge.weight / divisor

    groups = []
    for group in range(SCALE):
        cluster_ids = [node for node, assigned in enumerate(member) if assigned == group]
        counts = Counter(clusters[node]['resource'] for node in cluster_ids)
        base = float(row_lookup[4, group]['local_tail_risk'])
        child8 = [float(row_lookup[8, child]['local_tail_risk'])
                  for child in range(2 * group, 2 * group + 2)]
        child16 = [float(row_lookup[16, child]['local_tail_risk'])
                   for child in range(4 * group, 4 * group + 4)]
        child32 = [float(row_lookup[32, child]['local_tail_risk'])
                   for child in range(8 * group, 8 * group + 8)]
        groups.append({
            'id': group, 'cluster_ids': cluster_ids, 'cluster_counts': dict(sorted(counts.items())),
            'atom_count': sum(len(clusters[node]['atoms']) for node in cluster_ids),
            'mean_score': float(row_lookup[4, group]['cut_pressure']),
            'k4_local_tail_risk': base,
            'max_k8_child_tail_risk': max(child8),
            'max_k16_child_tail_risk': max(child16),
            'max_k32_child_tail_risk': max(child32),
            'tail_score': 0.4 * base + 0.2 * max(child8) + 0.2 * max(child16) + 0.2 * max(child32),
        })
    for cluster in clusters:
        cluster['group'] = member[cluster['id']]
    return {
        'schema': 'tailcompile-logic-ir-v1', 'design': 'sha',
        'mapped_blif': str(blif), 'mapped_blif_sha256': digest(blif),
        'packed_net': str(packed_net), 'packed_net_sha256': digest(packed_net),
        'parser_audit': parser_audit, 'hypergraph_audit': edge_audit,
        'partition_trace': trace, 'contracted_global_features': features,
        'capacity_shaping': {'target_clb_clusters': target_clb,
                             'final_clb_clusters': counts, 'moves': moves,
                             'io_clusters_per_group': io_counts},
        'contracted_edge_count': len(contracted_edges), 'groups': groups,
        'traffic_matrix': traffic, 'clusters': clusters,
        'cluster_count': len(clusters), 'atom_count': len(atom_to_cluster),
        'resource_totals': dict(sorted(Counter(cluster['resource'] for cluster in clusters).items())),
    }


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = quantile * (len(ordered) - 1)
    low, high = math.floor(index), math.ceil(index)
    if low == high:
        return ordered[low]
    return ordered[low] * (high - index) + ordered[high] * (index - low)


def binding_objective(assignment: tuple[int, ...], scores: list[float], logic: dict, device: dict) -> dict:
    score_norm = v0.normalize(scores)
    regions = device['regions']
    clb_demand = [int(group['cluster_counts'].get('clb', 0)) for group in logic['groups']]
    clb_capacity = [int(region['tile_counts'].get('clb', 0)) for region in regions]
    overflow = sum(max(0, clb_demand[group] - clb_capacity[assignment[group]]) ** 2
                   for group in range(SCALE))

    local_supply = []
    for region in regions:
        directional = [region[key] for key in ('escape_left', 'escape_right', 'escape_down', 'escape_up')
                       if region[key] > 0]
        segment = region['segment_supply']
        local_supply.append(sum(directional) + segment.get('L4', 0) + segment.get('L16', 0))
    supply_norm = v0.normalize(local_supply)
    risk_supply = sum(score_norm[group] * (1 - supply_norm[assignment[group]])
                      for group in range(SCALE)) / SCALE

    cut_supply = {(min(int(row['region_a']), int(row['region_b'])),
                   max(int(row['region_a']), int(row['region_b']))): float(row['count'])
                  for row in device['inter_region_rr_edge_counts_per_channel_width']}
    max_cut = max(cut_supply.values())
    max_direction = max(region[key] for region in regions
                        for key in ('escape_left', 'escape_right', 'escape_down', 'escape_up'))
    max_segment = max(value for region in regions for value in region['segment_supply'].values())
    total_traffic = max(sum(sum(row) for row in logic['traffic_matrix']) / 2, 1e-12)
    ratios, communication = [], 0.0
    for left in range(SCALE):
        region_left = regions[assignment[left]]
        for right in range(left + 1, SCALE):
            traffic = logic['traffic_matrix'][left][right]
            if traffic <= 0:
                continue
            region_right = regions[assignment[right]]
            distance_x = abs(region_left['column'] - region_right['column'])
            distance_y = abs(region_left['row'] - region_right['row'])
            distance = distance_x + distance_y
            communication += traffic * distance / total_traffic
            pair = tuple(sorted((assignment[left], assignment[right])))
            ratios.append((traffic / total_traffic) / max(cut_supply[pair] / max_cut, 1e-12))

            horizontal = traffic * distance_x / max(distance, 1)
            vertical = traffic * distance_y / max(distance, 1)
            for source, target in ((region_left, region_right), (region_right, region_left)):
                if horizontal:
                    key = 'escape_right' if target['column'] > source['column'] else 'escape_left'
                    ratios.append((horizontal / total_traffic) /
                                  max(source[key] / max_direction, 1e-12))
                if vertical:
                    key = 'escape_up' if target['row'] > source['row'] else 'escape_down'
                    ratios.append((vertical / total_traffic) /
                                  max(source[key] / max_direction, 1e-12))
            segment_name = 'L16' if distance > 1 else 'L4'
            segment_supply = (region_left['segment_supply'].get(segment_name, 0)
                              + region_right['segment_supply'].get(segment_name, 0)) / 2
            ratios.append((traffic / total_traffic) /
                          max(segment_supply / max_segment, 1e-12))

    cutoff = max(1, math.ceil(len(ratios) * 0.25))
    cvar = sum(sorted(ratios, reverse=True)[:cutoff]) / cutoff
    physical = {'mean': sum(ratios) / max(len(ratios), 1), 'p95': percentile(ratios, 0.95),
                'max': max(ratios, default=0.0), 'cvar_top25': cvar, 'sample_count': len(ratios)}
    combined = 0.5 * risk_supply + 0.5 * cvar
    return {'capacity_overflow_clusters_squared': overflow, 'risk_supply': risk_supply,
            'rrg_physical_congestion': physical, 'combined_risk_rrg': combined,
            'communication': communication}


def solve_binding(logic: dict, device: dict, mode: str) -> dict:
    score_key = {'rrg_only': None, 'mean_rrg': 'mean_score',
                 'tail_rrg': 'tail_score'}[mode]
    scores = ([0.0] * SCALE if score_key is None
              else [float(group[score_key]) for group in logic['groups']])
    candidates = []
    for assignment in itertools.permutations(range(SCALE)):
        objective = binding_objective(assignment, scores, logic, device)
        key = (objective['capacity_overflow_clusters_squared'],
               objective['combined_risk_rrg'], objective['communication'], assignment)
        candidates.append((key, assignment, objective))
    _, assignment, objective = min(candidates, key=lambda item: item[0])
    return {
        'schema': 'tailcompile-binding-v1', 'mode': mode, 'score_key': score_key,
        'search': 'exhaustive-4-factorial',
        'objective_policy': 'lexicographic(exact_cluster_capacity,0.5*risk_supply+0.5*rrg_cvar,communication)',
        'assignment': list(assignment), 'objective': objective,
        'feasible_candidate_count': sum(item[2]['capacity_overflow_clusters_squared'] == 0 for item in candidates),
        'candidate_count': len(candidates),
    }


def lower_fplace(logic: dict, device: dict, binding: dict, path: Path) -> dict:
    assignment = binding['assignment']
    region_resource_index = Counter()
    io_index = 0
    lines = ['# TailCompile v1 packed-cluster-aware flat placement',
             '# <atom_name> <x> <y> <layer> <atom_sub_tile>']
    records = []
    io_tiles = device['perimeter_tiles']['io']
    for cluster in sorted(logic['clusters'], key=lambda item: item['id']):
        group, resource = int(cluster['group']), cluster['resource']
        region_id = assignment[group]
        if resource == 'io':
            tile = io_tiles[(io_index // 4) % len(io_tiles)]
            subtile = io_index % 4
            io_index += 1
        else:
            anchors = device['regions'][region_id]['tile_coordinates'].get(resource, [])
            if not anchors:
                raise ValueError(f'region {region_id} has no tiles for {resource}')
            index = region_resource_index[(region_id, resource)]
            if index >= len(anchors):
                raise ValueError(f'exact capacity overflow: region={region_id} resource={resource}')
            tile, subtile = anchors[index], 0
            region_resource_index[(region_id, resource)] += 1
        for atom in cluster['atoms']:
            lines.append(f'{atom} {tile[0]} {tile[1]} 0 {subtile}')
        records.append({'cluster': cluster['id'], 'group': group, 'region': region_id,
                        'resource': resource, 'x': tile[0], 'y': tile[1],
                        'subtile': subtile, 'atom_count': len(cluster['atoms'])})
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    return {'schema': 'tailcompile-lowering-v1', 'fplace': str(path),
            'sha256': digest(path), 'atom_count': sum(row['atom_count'] for row in records),
            'cluster_count': len(records), 'clusters': records}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blif', type=Path, default=DEFAULT_BLIF)
    parser.add_argument('--packed-net', type=Path, default=DEFAULT_PACKED_NET)
    parser.add_argument('--rr-graph', type=Path, default=DEFAULT_RRG)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for path in (args.blif, args.packed_net, args.rr_graph):
        if not path.is_file():
            parser.error(f'missing input: {path}')
    args.out.mkdir(parents=True, exist_ok=True)

    v0.SCALE, v0.REGION_COLUMNS, v0.REGION_ROWS = SCALE, REGION_COLUMNS, REGION_ROWS
    device = v0.parse_device_ir(args.rr_graph)
    device.update({'schema': 'tailcompile-device-ir-v1',
                   'region_grid': {'columns': REGION_COLUMNS, 'rows': REGION_ROWS, 'count': SCALE}})
    logic = build_logic_ir(args.blif, args.packed_net)
    (args.out / 'device_ir.json').write_text(json.dumps(device, indent=2) + '\n')
    (args.out / 'logic_ir.json').write_text(json.dumps(logic, indent=2) + '\n')

    methods = {}
    for mode in ('rrg_only', 'mean_rrg', 'tail_rrg'):
        binding = solve_binding(logic, device, mode)
        binding_path = args.out / f'{mode}_binding.json'
        binding_path.write_text(json.dumps(binding, indent=2) + '\n')
        lowering = lower_fplace(logic, device, binding, args.out / f'{mode}.fplace')
        lowering_path = args.out / f'{mode}_lowering.json'
        lowering_path.write_text(json.dumps(lowering, indent=2) + '\n')
        methods[mode] = {'assignment': binding['assignment'], 'objective': binding['objective'],
                         'fplace_sha256': lowering['sha256']}
    summary = {'schema': 'tailcompile-v1-build', 'design': 'sha', 'methods': methods,
               'evidence_sha256': {path.name: digest(path) for path in args.out.iterdir()
                                   if path.is_file() and path.name != 'build_summary.json'}}
    (args.out / 'build_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
