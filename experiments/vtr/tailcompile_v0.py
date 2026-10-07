"""Build deterministic Logic/Device IR, bindings, and VPR .fplace files.

This is the non-ML TailCompile v0 pilot.  It consumes only a mapped BLIF,
pre-packing molecule constraints, and an architecture RR graph.  No placement,
routing occupancy, or post-route label is read by this program.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import re
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from tailrisk_core import Edge, build_edges, parse_blif, profile

HERE = Path(__file__).resolve().parent
DEFAULT_BLIF = HERE / 'runs/pilot_v1/sha/full-seed1-w300/sha.pre-vpr.blif'
DEFAULT_RRG = HERE / 'runs/pilot_v1/sha/route-seed1-w100-rr/sha.rr_graph.xml.gz'
DEFAULT_MOLECULES = HERE / 'runs/pilot_v1/sha/route-seed1-w100-rr/pre_packing_molecules_and_patterns.echo'
DEFAULT_OUT = HERE / 'tailcompile_v0/sha'
SCALE = 8
REGION_COLUMNS = 2
REGION_ROWS = 4
ATOM_RE = re.compile(r'atom block (.+?)(?: root node)?$')


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def chunks(values: list[int], count: int) -> list[list[int]]:
    if len(values) < count:
        raise ValueError(f'cannot divide {len(values)} coordinates into {count} regions')
    quotient, remainder = divmod(len(values), count)
    result, offset = [], 0
    for index in range(count):
        size = quotient + int(index < remainder)
        result.append(values[offset:offset + size])
        offset += size
    return result


def region_for_coordinate(value: float, bins: list[list[int]]) -> int:
    for index, values in enumerate(bins):
        if min(values) <= value <= max(values):
            return index
    return 0 if value < min(bins[0]) else len(bins) - 1


def parse_device_ir(path: Path) -> dict:
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as stream:
        root = ET.parse(stream).getroot()
    channel = root.find('./channels/channel')
    channel_width = int(channel.attrib['chan_width_max'])
    block_types = {item.attrib['id']: item.attrib['name']
                   for item in root.find('block_types')}
    grid_items = list(root.find('grid'))
    width = max(int(item.attrib['x']) for item in grid_items) + 1
    height = max(int(item.attrib['y']) for item in grid_items) + 1
    interior_x = list(range(1, width - 1))
    interior_y = list(range(1, height - 1))
    x_bins = chunks(interior_x, REGION_COLUMNS)
    y_bins = chunks(interior_y, REGION_ROWS)
    regions = []
    for row in range(REGION_ROWS):
        for column in range(REGION_COLUMNS):
            regions.append({
                'id': row * REGION_COLUMNS + column, 'row': row, 'column': column,
                'x_low': min(x_bins[column]), 'x_high': max(x_bins[column]),
                'y_low': min(y_bins[row]), 'y_high': max(y_bins[row]),
                'center_x': sum(x_bins[column]) / len(x_bins[column]),
                'center_y': sum(y_bins[row]) / len(y_bins[row]),
                'tile_counts': {}, 'tile_coordinates': defaultdict(list),
                'channel_internal': 0.0,
                'escape_left': 0.0, 'escape_right': 0.0,
                'escape_down': 0.0, 'escape_up': 0.0,
                'segment_supply': defaultdict(float),
            })
    perimeter_tiles = defaultdict(list)
    for item in grid_items:
        if int(item.attrib.get('width_offset', 0)) or int(item.attrib.get('height_offset', 0)):
            continue
        x, y = int(item.attrib['x']), int(item.attrib['y'])
        name = block_types[item.attrib['block_type_id']]
        perimeter_tiles[name].append([x, y])
        if x in interior_x and y in interior_y:
            region = regions[region_for_coordinate(y, y_bins) * REGION_COLUMNS
                             + region_for_coordinate(x, x_bins)]
            region['tile_coordinates'][name].append([x, y])

    segments = {item.attrib['id']: item.attrib.get('name', item.attrib['id'])
                for item in root.find('segments')}
    node_regions: dict[int, int] = {}
    rr_nodes = root.find('rr_nodes')
    for node in rr_nodes:
        node_type = node.attrib['type']
        loc = node.find('loc')
        xlow, xhigh = int(loc.attrib['xlow']), int(loc.attrib['xhigh'])
        ylow, yhigh = int(loc.attrib['ylow']), int(loc.attrib['yhigh'])
        center_x, center_y = (xlow + xhigh) / 2, (ylow + yhigh) / 2
        rid = (region_for_coordinate(center_y, y_bins) * REGION_COLUMNS
               + region_for_coordinate(center_x, x_bins))
        node_regions[int(node.attrib['id'])] = rid
        if node_type not in ('CHANX', 'CHANY'):
            continue
        capacity = int(node.attrib.get('capacity', 1)) / channel_width
        span = (xhigh - xlow + 1) if node_type == 'CHANX' else (yhigh - ylow + 1)
        regions[rid]['channel_internal'] += capacity * span
        segment = node.find('segment')
        segment_name = segments.get(segment.attrib['segment_id'], segment.attrib['segment_id'])
        regions[rid]['segment_supply'][segment_name] += capacity * span
        if node_type == 'CHANX':
            row = region_for_coordinate(center_y, y_bins)
            for boundary_index in range(REGION_COLUMNS - 1):
                boundary = max(x_bins[boundary_index]) + 1
                if xlow < boundary <= xhigh:
                    left = regions[row * REGION_COLUMNS + boundary_index]
                    right = regions[row * REGION_COLUMNS + boundary_index + 1]
                    left['escape_right'] += capacity
                    right['escape_left'] += capacity
        else:
            column = region_for_coordinate(center_x, x_bins)
            for boundary_index in range(REGION_ROWS - 1):
                boundary = max(y_bins[boundary_index]) + 1
                if ylow < boundary <= yhigh:
                    lower = regions[boundary_index * REGION_COLUMNS + column]
                    upper = regions[(boundary_index + 1) * REGION_COLUMNS + column]
                    lower['escape_up'] += capacity
                    upper['escape_down'] += capacity

    inter_region_edges = Counter()
    for edge in root.find('rr_edges'):
        source = node_regions[int(edge.attrib['src_node'])]
        sink = node_regions[int(edge.attrib['sink_node'])]
        if source != sink:
            inter_region_edges[tuple(sorted((source, sink)))] += 1

    for region in regions:
        region['tile_counts'] = {name: len(coords)
                                 for name, coords in sorted(region['tile_coordinates'].items())}
        region['tile_coordinates'] = {name: sorted(coords)
                                      for name, coords in sorted(region['tile_coordinates'].items())}
        region['segment_supply'] = dict(sorted(region['segment_supply'].items()))
        region['escape_total'] = sum(region[key] for key in
                                     ('escape_left', 'escape_right', 'escape_down', 'escape_up'))
    return {
        'schema': 'tailcompile-device-ir-v0', 'rr_graph': str(path),
        'rr_graph_sha256': digest(path), 'canonical_channel_width': channel_width,
        'grid_width': width, 'grid_height': height, 'region_grid': [REGION_COLUMNS, REGION_ROWS],
        'routing_supply_is_normalized_by_channel_width': True,
        'block_types': block_types, 'perimeter_tiles': dict(sorted(perimeter_tiles.items())),
        'regions': regions,
        'inter_region_rr_edge_counts_per_channel_width': [
            {'region_a': pair[0], 'region_b': pair[1], 'count': count / channel_width}
            for pair, count in sorted(inter_region_edges.items())],
        'rr_node_count': len(rr_nodes), 'rr_edge_count': len(root.find('rr_edges')),
    }


def parse_molecules(path: Path) -> list[dict]:
    molecules, current = [], None
    for raw in path.read_text(errors='replace').splitlines():
        if raw.startswith('molecule type:'):
            if current is not None:
                molecules.append(current)
            current = {'type': raw.split(':', 1)[1].strip(), 'atoms': []}
        elif current is not None:
            match = ATOM_RE.search(raw.strip())
            if match and match.group(1) != 'empty':
                current['atoms'].append(match.group(1))
    if current is not None:
        molecules.append(current)
    if not molecules or any(not molecule['atoms'] for molecule in molecules):
        raise ValueError(f'invalid molecule echo: {path}')
    return molecules


def primitive_name(primitive) -> str:
    if primitive.kind == 'io_out':
        return 'out:' + primitive.inputs[0]
    if primitive.outputs:
        return primitive.outputs[0]
    raise ValueError(f'primitive has no VPR atom name: {primitive}')


def molecule_weight(molecule: dict, atom_kind: dict[str, str]) -> int:
    """Integer half-FLE proxy used only to balance pre-pack molecules."""
    non_io = [atom for atom in molecule['atoms']
              if atom_kind.get(atom, 'lut') not in ('io_in', 'io_out')]
    if not non_io:
        return 1
    if molecule['type'] == 'chain':
        return 2 * len(non_io)
    if molecule['type'] == 'ble5':
        return 1
    return 2


def weighted_recursive_partitions(weights: list[int], edges: list[Edge]) -> tuple[dict[int, list[int]], list[dict]]:
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
        seed = max(nodes, key=lambda node: (degree(node) / max(weights[node], 1), -node))
        left, left_weight = {seed}, weights[seed]
        while len(left) < len(nodes) - 1:
            remaining = [node for node in nodes if node not in left]
            fitting = [node for node in remaining if left_weight + weights[node] <= target]
            if not fitting:
                candidate = min(remaining, key=lambda node: (abs(target - left_weight - weights[node]), node))
                if abs(target - left_weight) <= abs(target - left_weight - weights[candidate]):
                    break
            else:
                candidate = max(fitting, key=lambda node: (
                    sum(weight for other, weight in neighbors[node].items() if other in left),
                    degree(node) / max(weights[node], 1), -node))
            left.add(candidate)
            left_weight += weights[candidate]
        left_nodes = [node for node in nodes if node in left]
        right_nodes = [node for node in nodes if node not in left]
        if not right_nodes:
            moved = left_nodes.pop()
            right_nodes.append(moved)
            left_weight -= weights[moved]
        cut = sum(edge.weight for edge in edges
                  if any(node in left for node in edge.pins)
                  and any(node in allowed and node not in left for node in edge.pins))
        return left_nodes, right_nodes, {
            'node_count': len(nodes), 'total_weight': total,
            'left_weight': left_weight, 'right_weight': total - left_weight,
            'relative_imbalance': abs(total - 2 * left_weight) / total,
            'cut_weight': cut,
        }

    groups = [list(range(len(weights)))]
    memberships, trace = {}, []
    while len(groups) < 64:
        next_groups = []
        for group_id, group in enumerate(groups):
            left, right, record = bisect(group)
            record.update({'parent_scale': len(groups), 'parent_group': group_id})
            trace.append(record)
            next_groups.extend((left, right))
        groups = next_groups
        if len(groups) in (8, 16, 32, 64):
            membership = [0] * len(weights)
            for group_id, group in enumerate(groups):
                for node in group:
                    membership[node] = group_id
            memberships[len(groups)] = membership
    return memberships, trace


def build_logic_ir(blif: Path, molecule_path: Path) -> dict:
    primitives, parser_audit = parse_blif(blif)
    edges, edge_audit = build_edges(primitives)
    atom_to_node = {primitive_name(primitive): node for node, primitive in enumerate(primitives)}
    if len(atom_to_node) != len(primitives):
        raise AssertionError('VPR atom names derived from BLIF are not unique')
    molecules = parse_molecules(molecule_path)
    molecule_atoms = [atom for molecule in molecules for atom in molecule['atoms']]
    if len(molecule_atoms) != len(set(molecule_atoms)):
        raise AssertionError('an atom appears in multiple pre-pack molecules')
    missing_from_blif = sorted(set(molecule_atoms) - set(atom_to_node))
    missing_from_molecules = sorted(set(atom_to_node) - set(molecule_atoms))
    if missing_from_blif != ['gnd', 'vcc'] or missing_from_molecules:
        raise AssertionError({'only_expected_constants_may_be_extra': missing_from_blif,
                              'parsed_atoms_missing_from_molecules': missing_from_molecules})

    atom_to_molecule = {atom: molecule_id for molecule_id, molecule in enumerate(molecules)
                        for atom in molecule['atoms']}
    atom_kind = {name: primitives[node].kind for name, node in atom_to_node.items()}
    atom_kind.update({'gnd': 'lut', 'vcc': 'lut'})
    contracted_edges = []
    for edge in edges:
        pins = tuple(sorted({atom_to_molecule[primitive_name(primitives[node])] for node in edge.pins}))
        if len(pins) < 2:
            continue
        drivers = tuple(sorted({atom_to_molecule[primitive_name(primitives[node])] for node in edge.drivers}))
        sinks = tuple(sorted({atom_to_molecule[primitive_name(primitives[node])] for node in edge.sinks}))
        contracted_edges.append(Edge(edge.name, pins, drivers, sinks, edge.fanout,
                                     edge.criticality, edge.type_factor, edge.weight))
    weights = [molecule_weight(molecule, atom_kind) for molecule in molecules]
    memberships, trace = weighted_recursive_partitions(weights, contracted_edges)
    features, group_rows, _ = profile(len(molecules), contracted_edges, memberships)
    member = memberships[SCALE]

    row_lookup = {(int(row['scale']), int(row['group'])): row for row in group_rows}
    groups = []
    traffic = [[0.0] * SCALE for _ in range(SCALE)]
    for edge in contracted_edges:
        touched = sorted({member[node] for node in edge.pins})
        if len(touched) > 1:
            divisor = len(touched) * (len(touched) - 1) / 2
            for index, group_a in enumerate(touched):
                for group_b in touched[index + 1:]:
                    traffic[group_a][group_b] += edge.weight / divisor
                    traffic[group_b][group_a] += edge.weight / divisor
    for group in range(SCALE):
        molecule_ids = [node for node, assigned in enumerate(member) if assigned == group]
        atom_names = [atom for molecule_id in molecule_ids for atom in molecules[molecule_id]['atoms']]
        kinds = Counter(atom_kind[atom] for atom in atom_names)
        base = float(row_lookup[8, group]['local_tail_risk'])
        k16_children = [float(row_lookup[16, child]['local_tail_risk'])
                        for child in (2 * group, 2 * group + 1)]
        k32_children = [float(row_lookup[32, child]['local_tail_risk'])
                        for child in range(4 * group, 4 * group + 4)]
        k64_children = [float(row_lookup[64, child]['local_tail_risk'])
                        for child in range(8 * group, 8 * group + 8)]
        groups.append({
            'id': group, 'molecule_count': len(molecule_ids), 'primitive_count': len(atom_names),
            'balance_weight': sum(weights[node] for node in molecule_ids),
            'primitive_kinds': dict(sorted(kinds.items())),
            'mean_score': float(row_lookup[8, group]['cut_pressure']),
            'k8_local_tail_risk': base,
            'max_k16_child_tail_risk': max(k16_children),
            'max_k32_child_tail_risk': max(k32_children),
            'max_k64_child_tail_risk': max(k64_children),
            'tail_score': (0.4 * base + 0.2 * max(k16_children)
                           + 0.2 * max(k32_children) + 0.2 * max(k64_children)),
        })

    atom_records = []
    for molecule_id, molecule in enumerate(molecules):
        assigned_group = member[molecule_id]
        for atom in molecule['atoms']:
            kind = atom_kind[atom]
            atom_records.append({'name': atom, 'kind': kind, 'molecule': molecule_id,
                                 'molecule_type': molecule['type'],
                                 'group': assigned_group})
    return {
        'schema': 'tailcompile-logic-ir-v0', 'design': 'sha', 'mapped_blif': str(blif),
        'mapped_blif_sha256': digest(blif), 'molecule_echo': str(molecule_path),
        'molecule_echo_sha256': digest(molecule_path), 'scale': SCALE,
        'parser_audit': parser_audit, 'hypergraph_audit': edge_audit,
        'partition_trace': trace, 'contracted_global_features': features,
        'contracted_edge_count': len(contracted_edges),
        'groups': groups, 'traffic_matrix': traffic, 'atoms': atom_records,
        'molecule_count': len(molecules),
        'cross_group_molecule_count': 0,
    }


def normalize(values: list[float]) -> list[float]:
    low, high = min(values), max(values)
    return [(value - low) / (high - low) if high > low else 0.0 for value in values]


def binding_objective(assignment: list[int], scores: list[float], logic: dict, device: dict) -> dict:
    score_norm = normalize(scores)
    supplies = [float(region['escape_total']) for region in device['regions']]
    supply_norm = normalize(supplies)
    supply_cost = sum(score_norm[group] * (1 - supply_norm[assignment[group]])
                      for group in range(SCALE)) / SCALE
    total_traffic = sum(sum(row) for row in logic['traffic_matrix']) / 2
    communication = 0.0
    for group_a in range(SCALE):
        region_a = device['regions'][assignment[group_a]]
        for group_b in range(group_a + 1, SCALE):
            region_b = device['regions'][assignment[group_b]]
            distance = abs(region_a['row'] - region_b['row']) + abs(region_a['column'] - region_b['column'])
            communication += (logic['traffic_matrix'][group_a][group_b] * distance
                              / (REGION_COLUMNS + REGION_ROWS - 2))
    communication /= max(total_traffic, 1e-12)
    demand = normalize([float(group['balance_weight']) for group in logic['groups']])
    capacity = normalize([float(region['tile_counts'].get('clb', 0)) for region in device['regions']])
    capacity_cost = sum(max(0.0, demand[group] - capacity[assignment[group]]) ** 2
                        for group in range(SCALE)) / SCALE
    return {'total': communication + supply_cost + 10 * capacity_cost,
            'communication': communication, 'risk_supply': supply_cost,
            'capacity_overflow_proxy': capacity_cost}


def solve_binding(logic: dict, device: dict, mode: str) -> dict:
    score_key = 'mean_score' if mode == 'mean_rrg' else 'tail_score'
    scores = [float(group[score_key]) for group in logic['groups']]
    supplies = [float(region['escape_total']) for region in device['regions']]
    group_order = sorted(range(SCALE), key=lambda group: (-scores[group], group))
    region_order = sorted(range(SCALE), key=lambda region: (-supplies[region], region))
    assignment = [0] * SCALE
    for group, region in zip(group_order, region_order):
        assignment[group] = region
    initial = binding_objective(assignment, scores, logic, device)
    # RRG matching is the purpose of this ablation.  Treat capacity feasibility
    # and risk-to-supply matching as primary objectives; use communication only
    # to break ties.  A scalar sum let the much larger communication term erase
    # the intended difference between mean pressure and tail risk.
    objective_order = ('capacity_overflow_proxy', 'risk_supply', 'communication')

    def objective_tuple(record: dict) -> tuple[float, float, float]:
        return tuple(float(record[key]) for key in objective_order)

    swaps = 0
    while True:
        current = binding_objective(assignment, scores, logic, device)
        best = objective_tuple(current)
        best_pair = None
        for group_a in range(SCALE):
            for group_b in range(group_a + 1, SCALE):
                proposal = assignment.copy()
                proposal[group_a], proposal[group_b] = proposal[group_b], proposal[group_a]
                value = objective_tuple(binding_objective(proposal, scores, logic, device))
                if value < best:
                    best, best_pair = value, (group_a, group_b)
        if best_pair is None:
            break
        group_a, group_b = best_pair
        assignment[group_a], assignment[group_b] = assignment[group_b], assignment[group_a]
        swaps += 1
    final = binding_objective(assignment, scores, logic, device)
    return {'schema': 'tailcompile-binding-v0', 'mode': mode, 'score_key': score_key,
            'objective_policy': 'lexicographic(capacity_overflow_proxy,risk_supply,communication)',
            'objective_order': list(objective_order),
            'assignment': assignment, 'initial_objective': initial,
            'final_objective': final, 'accepted_swaps': swaps,
            'group_rows': [{'group': group, 'region': assignment[group],
                            'score': scores[group],
                            'region_escape_supply': supplies[assignment[group]]}
                           for group in range(SCALE)]}


def nearest_perimeter(center_x: float, center_y: float, coordinates: list[list[int]]) -> list[list[int]]:
    return sorted(coordinates, key=lambda xy: (abs(xy[0] - center_x) + abs(xy[1] - center_y), xy))


def lower_fplace(logic: dict, device: dict, binding: dict, path: Path) -> dict:
    assignments = binding['assignment']
    atoms_by_molecule = defaultdict(list)
    for atom in logic['atoms']:
        atoms_by_molecule[int(atom['molecule'])].append(atom)
    counters = Counter()
    lines = ['# TailCompile v0 atom flat placement',
             '# <atom_name> <x> <y> <layer> <atom_sub_tile>']
    molecule_rows = []
    for molecule_id in sorted(atoms_by_molecule):
        atoms = atoms_by_molecule[molecule_id]
        group = int(atoms[0]['group'])
        region_id = assignments[group]
        region = device['regions'][region_id]
        kinds = {atom['kind'] for atom in atoms}
        if kinds & {'io_in', 'io_out'}:
            resource = 'io'
            anchors = nearest_perimeter(region['center_x'], region['center_y'],
                                        device['perimeter_tiles']['io'])
        elif 'bram' in kinds:
            resource = 'memory'
            anchors = region['tile_coordinates'].get(resource, [])
        elif 'dsp' in kinds:
            resource = 'dsp_top'
            anchors = region['tile_coordinates'].get(resource, [])
        else:
            resource = 'clb'
            anchors = region['tile_coordinates'].get(resource, [])
        if not anchors:
            raise ValueError(f'group {group} region {region_id} has no {resource} anchor')
        key = (region_id, resource)
        anchor = anchors[counters[key] % len(anchors)]
        counters[key] += 1
        for atom in atoms:
            # This VTR commit accepts an undefined sub-tile while parsing, but
            # APPack materializes some molecule roots at sub-tile 0 before its
            # consistency check.  Use 0 for every atom in a molecule so the
            # architecture-derived molecule constraint remains exact.
            lines.append(f'{atom["name"]} {anchor[0]} {anchor[1]} 0 0')
        molecule_rows.append({'molecule': molecule_id, 'group': group, 'region': region_id,
                              'resource': resource, 'x': anchor[0], 'y': anchor[1],
                              'atom_count': len(atoms)})
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    return {'fplace': str(path), 'sha256': digest(path), 'atom_count': len(logic['atoms']),
            'molecule_count': len(molecule_rows), 'molecules': molecule_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--blif', type=Path, default=DEFAULT_BLIF)
    parser.add_argument('--rr-graph', type=Path, default=DEFAULT_RRG)
    parser.add_argument('--molecules', type=Path, default=DEFAULT_MOLECULES)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    for path in (args.blif, args.rr_graph, args.molecules):
        if not path.is_file():
            parser.error(f'missing input: {path}')
    args.out.mkdir(parents=True, exist_ok=True)
    device = parse_device_ir(args.rr_graph)
    logic = build_logic_ir(args.blif, args.molecules)
    (args.out / 'device_ir.json').write_text(json.dumps(device, indent=2) + '\n', encoding='utf-8')
    (args.out / 'logic_ir.json').write_text(json.dumps(logic, indent=2) + '\n', encoding='utf-8')
    summary = {'schema': 'tailcompile-v0-build', 'design': logic['design'], 'methods': {}}
    for mode in ('mean_rrg', 'tail_rrg'):
        binding = solve_binding(logic, device, mode)
        binding_path = args.out / f'{mode}_binding.json'
        binding_path.write_text(json.dumps(binding, indent=2) + '\n', encoding='utf-8')
        lowering = lower_fplace(logic, device, binding, args.out / f'{mode}.fplace')
        lowering_path = args.out / f'{mode}_lowering.json'
        lowering_path.write_text(json.dumps(lowering, indent=2) + '\n', encoding='utf-8')
        summary['methods'][mode] = {'binding': str(binding_path), 'lowering': str(lowering_path),
                                   'objective': binding['final_objective'],
                                   'fplace_sha256': lowering['sha256']}
    summary['evidence_sha256'] = {
        name: digest(args.out / name) for name in ('device_ir.json', 'logic_ir.json',
                                                   'mean_rrg_binding.json', 'tail_rrg_binding.json',
                                                   'mean_rrg.fplace', 'tail_rrg.fplace')}
    (args.out / 'build_summary.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
