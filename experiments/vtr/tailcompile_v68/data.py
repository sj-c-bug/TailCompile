"""Leakage-safe AP-molecule teacher data and streaming fixed-tile device graphs.

No movable placement coordinates enter graph features. AP blocks, not packed
cluster IDs, are the logical nodes. Tile capacities remain packing upper-bound
proxies: only VPR's pack/full legalizer can certify physical feasibility.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
VTR_COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'
SCHEMA = 'tailcompile-v68-ap-teacher-1'
DEMAND_KEYS = ('lut', 'ff', 'memory', 'dsp_top', 'io')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1048576), b''):
            h.update(chunk)
    return h.hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def load_config(path=None):
    value = json.loads(Path(path or HERE / 'cohort.json').read_text(encoding='utf-8'))
    validate_splits(value)
    return value


def validate_splits(config):
    rows = config['designs']
    if not rows:
        raise ValueError('empty cohort')
    seen, families = set(), {}
    previous = set(config.get('previously_used_families', []))
    counts = Counter()
    for row in rows:
        design, family, split = row['design'], row['family'], row['split']
        if design in seen:
            raise ValueError(f'duplicate design: {design}')
        if split not in ('train', 'validation', 'final_test'):
            raise ValueError(f'unknown split: {split}')
        if family in families and families[family] != split:
            raise ValueError(f'design-family leakage: {family}')
        if family in previous and split != 'train':
            raise ValueError(f'previous development family cannot be held out: {family}')
        source = Path(row['source'])
        if source.is_absolute() or '..' in source.parts:
            raise ValueError('benchmark source must stay under the Verilog benchmark root')
        seen.add(design)
        families[family] = split
        counts[split] += 1
    if set(counts) != {'train', 'validation', 'final_test'}:
        raise ValueError('cohort must contain train, validation and final_test')
    return dict(designs=len(rows), families=len(families), split_designs=dict(counts))


def inventory(config, vtr_root):
    """Check source availability, not synthesis/architecture compatibility."""
    root = Path(vtr_root).resolve() / 'vtr_flow/benchmarks/verilog'
    result = []
    for row in config['designs']:
        source = root / row['source']
        present = source.is_file()
        result.append(dict(row, available=present, path=str(source),
                           source_sha256=digest(source) if present else None,
                           runnable_status='not_yet_verified'))
    counts = validate_splits(config)
    counts['design_count'] = counts.pop('designs')
    return dict(schema=SCHEMA, vtr_commit=VTR_COMMIT, **counts,
                available_designs=sum(r['available'] for r in result), designs=result)


def read_ap_snapshot(prefix, structure=False):
    """Read the audited hook's atom rows; reject split atoms and partial exports."""
    prefix = str(prefix)
    size = list(map(int, Path(prefix + '.meta.tsv').read_text().split()))
    if len(size) != 2 or min(size) < 3:
        raise ValueError('invalid AP snapshot grid')
    atoms, blocks = set(), {}
    for line in Path(prefix + '.blocks.tsv').read_text().splitlines():
        fields = line.split('\t')
        if len(fields) not in (8, 9, 10):
            raise ValueError('invalid AP block row schema')
        raw, atom, model, fixed, x, y, z, sub = fields[:8]
        raw, fixed = int(raw), bool(int(fixed))
        loc = [float(x), float(y), float(z), int(sub)]
        if atom in atoms or not all(math.isfinite(v) for v in loc):
            raise ValueError('duplicate/nonfinite AP atom')
        if loc[2] != 0:
            raise ValueError('v6.8 2D data rejects nonzero AP layer')
        if structure and not fixed and loc[:2] != [-1., -1.]:
            raise ValueError('pre-GP structure contains movable coordinates (teacher leakage)')
        b = blocks.setdefault(raw, dict(ap_block_id=raw, atoms=[], models=[],
            model_names=[], molecule_ids=[], fixed=fixed, loc=loc))
        if b['fixed'] != fixed or b['loc'] != loc:
            raise ValueError('incoherent AP block: atom/molecule split')
        b['atoms'].append(atom)
        b['models'].append(model)
        b['model_names'].append(fields[8] if len(fields) > 8 else '')
        b['molecule_ids'].append(fields[9] if len(fields) > 9 else '')
        atoms.add(atom)
    if not blocks:
        raise ValueError('empty AP snapshot')
    ids = sorted(blocks)
    remap = {raw: i for i, raw in enumerate(ids)}
    nets = []
    for line in Path(prefix + '.nets.tsv').read_text().splitlines():
        pins = [int(v) for v in line.split('\t')[1:]]
        if any(v not in remap for v in pins):
            raise ValueError('net references an unknown AP block')
        pins = sorted({remap[v] for v in pins})
        if len(pins) > 1:
            nets.append(pins)
    return dict(size=size, blocks=[blocks[v] for v in ids], nets=nets,
                atom_count=len(atoms), structure=structure)


def atom_resource(model, primitive=None):
    """Classify primitive semantics; never interpret fixed-version model IDs."""
    kind = getattr(primitive, 'kind', '')
    text = (model or getattr(primitive, 'model', '')).lower()
    if kind in ('io_in', 'io_out') or text in ('.input', '.output', 'input', 'output', 'inpad', 'outpad'):
        return 'io'
    if kind == 'ff' or text in ('.latch', 'latch', 'ff', 'dff'):
        return 'ff'
    if kind == 'lut' or text in ('.names', 'names', 'lut'):
        return 'lut'
    if any(word in text for word in ('ram', 'memory', 'rom')):
        return 'memory'
    if any(word in text for word in ('multiply', 'mult_', 'multiplier', 'dsp', 'mac_')):
        return 'dsp_top'
    if text in ('adder', 'add', 'sub', 'carry', 'carry_chain'):
        return 'lut'
    raise ValueError(f'unsupported AP primitive model {model!r}; explicit resource mapping required')


def architecture_model_resources(path):
    root = ET.parse(path).getroot()
    resources = {}
    for top in root.findall('./complexblocklist/pb_type'):
        resource = top.get('name')
        resource = {'clb': None, 'io': 'io', 'memory': 'memory', 'dsp_top': 'dsp_top'}.get(resource)
        if resource:
            for pb in top.iter('pb_type'):
                model = pb.get('blif_model', '').removeprefix('.subckt ').strip()
                if model and model not in ('.names', '.latch'):
                    resources[model] = resource
    return resources


def build_ap_logic(prefix, design, mapped_blif, architecture=None):
    from tailrisk_core import parse_blif
    from tailcompile_v0 import primitive_name
    primitives, audit = parse_blif(Path(mapped_blif))
    by_atom = {primitive_name(p): p for p in primitives}
    model_resources = architecture_model_resources(architecture) if architecture else {}
    graph = read_ap_snapshot(prefix, structure=True)
    clusters = []
    for i, block in enumerate(graph['blocks']):
        raw = Counter(model_resources.get(model) or atom_resource(model, by_atom.get(atom))
                      for atom, model in zip(block['atoms'], block['model_names']))
        hard = [r for r in ('memory', 'dsp_top', 'io') if raw[r]]
        if len(hard) > 1 or (hard and (raw['lut'] or raw['ff'])):
            raise ValueError('mixed hard-resource AP block requires an explicit coupled demand model')
        resource = hard[0] if hard else 'clb'
        # RAM atoms within one VPR grouped AP node share its hard-resource demand.
        # This is an AP demand proxy, not a certificate that a cluster fits a site.
        demand = {key: (int(bool(raw[key])) if key in ('memory', 'dsp_top') else raw[key])
                  for key in DEMAND_KEYS}
        cluster = dict(id=i, ap_block_id=block['ap_block_id'], name=block['atoms'][0],
            atoms=block['atoms'], model_names=block['model_names'], molecule_ids=block['molecule_ids'],
            resource=resource, resource_demand=demand, primitive_counts=dict(raw),
            fixed=block['fixed'], layer=block['loc'][2], subtile=block['loc'][3])
        if block['fixed']:
            cluster['fixed_location'] = block['loc']
        clusters.append(cluster)
    return dict(schema='tailcompile-v68-ap-logic-1', design=design, clusters=clusters,
                nets=graph['nets'], grid_width=graph['size'][0], grid_height=graph['size'][1],
                mapped_blif=str(Path(mapped_blif).resolve()), mapped_blif_sha256=digest(mapped_blif),
                structure_sha256={suffix: digest(str(prefix) + suffix)
                                  for suffix in ('.blocks.tsv', '.nets.tsv', '.meta.tsv')},
                parser_audit=audit, node_semantics='VPR AP blocks, intact molecule/RAM groups',
                capacity_semantics='AP primitive demand proxy; VPR certifies packing legality',
                movable_coordinates_in_features=False)


def parse_teacher(prefix, logic, device, split=None):
    """Match teacher nodes by exact atom sets, never by a presumed cluster ID."""
    if split == 'final_test':
        raise ValueError('final-test teacher labels may not enter model training/selection')
    teacher = read_ap_snapshot(prefix)
    if teacher['size'] != [device['grid_width'], device['grid_height']]:
        raise ValueError('teacher and device dimensions differ')
    by_atoms = {frozenset(b['atoms']): b for b in teacher['blocks']}
    expected = {a for c in logic['clusters'] for a in c['atoms']}
    observed = {a for b in teacher['blocks'] for a in b['atoms']}
    if expected != observed:
        raise ValueError('teacher atom coverage differs from pre-GP structure')
    from tailcompile_v4_risk import region_at
    regions, xy, movable = [], [], []
    for c in logic['clusters']:
        b = by_atoms.get(frozenset(c['atoms']))
        if b is None:
            raise ValueError('teacher splits or merges a pre-GP AP block')
        x, y, layer, sub = b['loc']
        if not (0 <= x < device['grid_width'] and 0 <= y < device['grid_height']):
            raise ValueError('teacher coordinate outside the device')
        if bool(c['fixed']) != bool(b['fixed']):
            raise ValueError('teacher changes fixed block mobility')
        if c['fixed']:
            for axis, coordinate in enumerate((x, y)):
                old = c['fixed_location'][axis]
                if old >= 0 and coordinate != old:
                    raise ValueError('teacher moved a constrained fixed coordinate')
        regions.append(region_at(device, x, y))
        xy.append([x / device['grid_width'], y / device['grid_height']])
        movable.append(not c['fixed'])
    return dict(schema=SCHEMA, teacher_region=regions, teacher_xy=xy, movable=movable,
                atom_coverage=len(expected), coordinate_system='raw native AP x/grid_width and y/grid_height in [0,1); no half-tile shift',
                teacher_sha256=digest(str(prefix) + '.blocks.tsv'))


def stream_device_ir(rr_graph, tile_size=4):
    """Retain only grid/root anchors and block types, never RRG nodes/edges."""
    if tile_size < 1:
        raise ValueError('tile_size must be positive')
    types, roots, maximum = {}, [], [-1, -1]
    channel_width = None
    opener = gzip.open if str(rr_graph).endswith('.gz') else open
    with opener(rr_graph, 'rb') as source:
        stack = []
        for event, item in ET.iterparse(source, events=('start', 'end')):
            if event == 'start':
                stack.append(item)
                continue
            parent = stack[-2] if len(stack) > 1 else None
            if item.tag == 'block_type' and parent is not None and parent.tag == 'block_types':
                types[item.get('id')] = item.get('name')
            if item.tag == 'channel' and parent is not None and parent.tag == 'channels':
                channel_width = int(item.get('chan_width_max'))
            if item.tag == 'grid_loc':
                if int(item.get('layer', '0')) != 0:
                    raise ValueError('v6.8 is 2D; nonzero grid layer')
                x, y = int(item.get('x')), int(item.get('y'))
                maximum = [max(maximum[0], x), max(maximum[1], y)]
                if int(item.get('width_offset', 0)) == 0 and int(item.get('height_offset', 0)) == 0:
                    roots.append((item.get('block_type_id'), x, y))
            if parent is not None and parent.tag in ('rr_nodes', 'rr_edges', 'grid', 'segments', 'channels', 'block_types'):
                parent.remove(item)
            stack.pop()
    width, height = [n + 1 for n in maximum]
    if min(width, height) < 3 or not types or not channel_width:
        raise ValueError('incomplete device grid/types/channel width')
    columns, rows = math.ceil((width - 2) / tile_size), math.ceil((height - 2) / tile_size)
    regions = []
    for row in range(rows):
        for column in range(columns):
            xl, yl = 1 + column * tile_size, 1 + row * tile_size
            xh, yh = min(width - 2, xl + tile_size - 1), min(height - 2, yl + tile_size - 1)
            regions.append(dict(id=len(regions), row=row, column=column,
                x_low=xl, x_high=xh, y_low=yl, y_high=yh, center_x=(xl+xh)/2,
                center_y=(yl+yh)/2, tile_counts={}, tile_coordinates={}, channel_internal=0.,
                escape_left=0., escape_right=0., escape_up=0., escape_down=0., segment_supply={}))
    device = dict(schema='tailcompile-v68-streaming-fixed-tile-device-1',
        rr_graph=str(Path(rr_graph).resolve()), rr_graph_sha256=digest(rr_graph),
        canonical_channel_width=channel_width, grid_width=width, grid_height=height,
        region_grid=dict(columns=columns, rows=rows, count=len(regions)), tile_size=tile_size,
        block_types=types, regions=regions, perimeter_tiles={},
        routing_supply_is_normalized_by_channel_width=True,
        inter_region_rr_edge_counts_per_channel_width=[],
        capacity_semantics='distinct physical root anchors, not independent AP cluster slots')
    seen = set()
    for typ, x, y in roots:
        if (x, y) in seen:
            raise ValueError('duplicate grid root anchor')
        seen.add((x, y))
        name = types[typ]
        if name.upper() == 'EMPTY':
            continue
        device['perimeter_tiles'].setdefault(name, []).append([x, y])
        # Perimeter IO belongs to the nearest tile block, but keeps exact anchors.
        column = max(0, min(columns - 1, (x - 1) // tile_size))
        row = max(0, min(rows - 1, (y - 1) // tile_size))
        region = regions[row * columns + column]
        region['tile_coordinates'].setdefault(name, []).append([x, y])
    for r in regions:
        r['tile_counts'] = {name: len(coords) for name, coords in r['tile_coordinates'].items()}
    return device


def architecture_resource_units(path):
    """Upper bounds per physical CLB; max across mutually exclusive modes."""
    root = ET.parse(path).getroot()  # Architecture is small; RRG is never parsed this way.
    def units(pb, resource):
        model = pb.get('blif_model', '').lower()
        if model in ('.names', '.latch'):
            return Counter({('lut' if model == '.names' else 'ff'): 1})
        if model and resource in ('memory', 'dsp_top', 'io'):
            return Counter({resource: 1})
        alternatives = list(pb.findall('mode')) or [pb]
        results = []
        for alternative in alternatives:
            value = Counter()
            for child in alternative.findall('pb_type'):
                value.update({k: v * int(child.get('num_pb', 1)) for k, v in units(child, resource).items()})
            results.append(value)
        return Counter({k: max((value[k] for value in results), default=0) for k in DEMAND_KEYS})
    values = {}
    for pb in root.findall('./complexblocklist/pb_type'):
        value = dict(units(pb, pb.get('name')))
        if any(value.values()):
            values[pb.get('name')] = value
    # Physical root anchors can have multiple subtiles (especially IO).
    physical = {}
    for tile in root.findall('./tiles/tile'):
        total = Counter()
        for sub in tile.findall('sub_tile'):
            alternatives = [values.get(site.get('pb_type'), {})
                            for site in sub.findall('./equivalent_sites/site')]
            count = int(sub.get('capacity', 1))
            total.update({key: count * max((v.get(key, 0) for v in alternatives), default=0)
                          for key in DEMAND_KEYS})
        physical[tile.get('name')] = dict(total)
    values.update(physical)
    return values


def prepare_graph(logic, device, supply, blif):
    import torch
    from tailcompile_v61 import static_graph
    graph = static_graph(logic, device, Path(blif), supply)
    from tailrisk_core import parse_blif
    from tailcompile_v2 import contract_edges
    primitives, _ = parse_blif(Path(blif))
    edges, _ = contract_edges(primitives, {atom: c['id'] for c in logic['clusters'] for atom in c['atoms']})
    critical, mass = defaultdict(float), defaultdict(float)
    for edge in edges:
        pins = list(edge.pins)
        for source in list(edge.drivers) or pins[:1]:
            for target in list(edge.sinks) or pins[1:]:
                if source != target:
                    for key in ((source, target), (target, source)):
                        critical[key] += edge.weight * edge.criticality
                        mass[key] += edge.weight
    graph['logic_criticality'] = torch.tensor([critical[a, b] / max(mass[a, b], 1e-9)
        for a, b in zip(graph['logic_src'].tolist(), graph['logic_dst'].tolist())], dtype=torch.float32)
    graph['criticality_semantics'] = 'pre-placement unit-delay topological-depth proxy, NOT measured VPR timing'
    graph['channel_width'] = float(supply['channel_width'])
    units = device.get('resource_units', {})
    demand = torch.tensor([[c['resource_demand'][k] for k in DEMAND_KEYS]
                            for c in logic['clusters']], dtype=torch.float32)
    if any(c['resource'] == 'clb' for c in logic['clusters']) and not units:
        raise ValueError('AP graph needs architecture-derived LUT/FF capacity units')
    capacities = []
    for region in device['regions']:
        capacities.append([sum(count * units.get(typ, {}).get(key, 0)
                               for typ, count in region['tile_counts'].items())
                           for key in DEMAND_KEYS])
    fixed = torch.tensor([c['fixed'] for c in logic['clusters']], dtype=torch.bool)
    # Leaf ownership extends to the perimeter, matching region_at for IO.
    # Coordinates remain GP seeds, not claims of resource-compatible sites.
    bounds = torch.tensor([[(0 if r['x_low'] == 1 else r['x_low'] - .5) / device['grid_width'],
                            (device['grid_width'] if r['x_high'] == device['grid_width'] - 2 else r['x_high'] + .5) / device['grid_width'],
                            (0 if r['y_low'] == 1 else r['y_low'] - .5) / device['grid_height'],
                            (device['grid_height'] if r['y_high'] == device['grid_height'] - 2 else r['y_high'] + .5) / device['grid_height']]
                           for r in device['regions']], dtype=torch.float32)
    from tailcompile_v4_risk import region_at
    graph.update(ap_demand=demand, demand=demand,
                 capacity=torch.tensor(capacities, dtype=torch.float32),
                 movable=~fixed, fixed=fixed, region_bounds=bounds,
                 fixed_region=torch.tensor([region_at(device, *c['fixed_location'][:2])
                    if c['fixed'] and min(c['fixed_location'][:2]) >= 0 else -1
                    for c in logic['clusters']], dtype=torch.long),
                 fixed_xy=torch.tensor([[c.get('fixed_location', [-1., -1.])[a] /
                    device['grid_width' if a == 0 else 'grid_height']
                    if c['fixed'] and c['fixed_location'][a] >= 0 else -1.
                    for a in range(2)] for c in logic['clusters']], dtype=torch.float32),
                 nets=logic['nets'])
    return graph


def build_cached_graph(record, out, tile_size=4):
    """Cache the deployment graph by all immutable input hashes and tile size."""
    started = time.perf_counter()
    import torch
    from prepare_tailcompile_v61 import supply_features
    from tailcompile_v4_risk import build_supply
    setup_seconds = time.perf_counter() - started
    if record.get('vtr_commit', VTR_COMMIT) != VTR_COMMIT:
        raise ValueError('VTR cohort mismatch')
    out = Path(out)
    hash_started = time.perf_counter()
    rr_hash = digest(record['rr_graph'])
    architecture_hash = digest(record['architecture']) if record.get('architecture') else None
    device_hash_seconds = time.perf_counter() - hash_started
    hash_started = time.perf_counter()
    signature = dict(schema=SCHEMA, design=record['design'], tile_size=tile_size,
        channel_width=int(record['channel_width']), vtr_commit=VTR_COMMIT,
        mapped_blif_sha256=digest(record['mapped_blif']), rr_graph_sha256=rr_hash,
        structure_sha256={s: digest(str(record['structure_prefix']) + s)
                          for s in ('.blocks.tsv', '.nets.tsv', '.meta.tsv')},
        architecture_sha256=architecture_hash)
    signature['implementation_sha256'] = {p.name: digest(p) for p in
        (Path(__file__), HERE.parent / 'prepare_tailcompile_v61.py',
         HERE.parent / 'tailcompile_v61.py', HERE.parent / 'tailcompile_v3.py',
         HERE.parent / 'tailcompile_v4_risk.py', HERE.parent / 'tailrisk_core.py')}
    logic_hash_seconds = time.perf_counter() - hash_started
    paths = {key: str(out / value) for key, value in
             dict(graph='graph.pt', logic='logic.json', device='device.json', supply='supply.json', audit='graph_audit.json').items()}
    audit_path = out / 'graph_audit.json'
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())
        if previous['signature'] != signature:
            raise ValueError('graph cache inputs changed; use a new cache directory')
        for key in ('graph', 'logic', 'device', 'supply'):
            if not Path(paths[key]).is_file() or digest(paths[key]) != previous['output_sha256'][key]:
                raise ValueError('incomplete/corrupted graph cache')
        return paths
    context_started = time.perf_counter()
    logic = build_ap_logic(record['structure_prefix'], record['design'], record['mapped_blif'], record.get('architecture'))
    logic_seconds = time.perf_counter() - context_started
    context_started = time.perf_counter()
    device = stream_device_ir(record['rr_graph'], tile_size)
    if [logic['grid_width'], logic['grid_height']] != [device['grid_width'], device['grid_height']]:
        raise ValueError('pre-GP structure and RRG grid mismatch')
    supply = build_supply(record['rr_graph'], device)
    if supply['channel_width'] != int(record['channel_width']):
        raise ValueError('actual RRG channel width differs from requested condition')
    supply_features(device, supply)
    if record.get('architecture'):
        device['resource_units'] = architecture_resource_units(record['architecture'])
    device_seconds = time.perf_counter() - context_started
    context_started = time.perf_counter()
    graph = prepare_graph(logic, device, supply, record['mapped_blif'])
    graph_seconds = time.perf_counter() - context_started
    context_started = time.perf_counter()
    save(paths['logic'], logic)
    save(paths['device'], device)
    save(paths['supply'], supply)
    torch.save(graph, paths['graph'])
    serialization_seconds = time.perf_counter() - context_started
    save(audit_path, dict(signature=signature, nodes=len(logic['clusters']), regions=len(device['regions']),
        movable_coordinates_in_features=False, node_semantics=logic['node_semantics'],
        output_sha256={key: digest(paths[key]) for key in ('graph', 'logic', 'device', 'supply')},
        build_wall_seconds=time.perf_counter() - started,
        timing=dict(runtime_module_setup=setup_seconds, logic_provenance=logic_hash_seconds,
                    device_provenance=device_hash_seconds,
                    logic_structure_features=logic_seconds, device_preprocessing=device_seconds,
                    logic_graph_features=graph_seconds, serialization=serialization_seconds),
        runtime_scope='new-design AP structure export and logic feature costs are NOT reusable RRG costs',
        limitations=['AP demand and root capacities are upper-bound proxies, not a packing certificate',
                     'routing pool footprints do not certify pin access or detailed reachability']))
    return paths


def teacher_record(record, prefix, graph_paths):
    """Training/validation targets only; final-test isolation is unconditional."""
    if record['split'] not in ('train', 'validation'):
        raise ValueError('teachers forbidden for final_test')
    logic = json.loads(Path(graph_paths['logic']).read_text())
    device = json.loads(Path(graph_paths['device']).read_text())
    targets = parse_teacher(prefix, logic, device, split=record['split'])
    return dict(design=record['design'], family=record['family'], split=record['split'],
                width=int(record['channel_width']), graph=graph_paths['graph'], **targets)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohort', type=Path, default=HERE / 'cohort.json')
    parser.add_argument('--vtr-root', type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    config = load_config(args.cohort)
    report = inventory(config, args.vtr_root) if args.vtr_root else validate_splits(config)
    if args.out:
        save(args.out, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
