"""3D APBlock structure with teacher coordinates kept outside graph inputs."""
from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tailcompile_v68.data import atom_resource, architecture_model_resources, DEMAND_KEYS
from io_utils import read, write, sha


def read_snapshot(prefix, *, structure=False):
    prefix = str(prefix)
    size = list(map(int, Path(prefix + '.meta.tsv').read_text().split()))
    if len(size) != 3 or min(size) < 1:
        raise ValueError('v7 snapshot requires explicit width height layers')
    blocks, atoms = {}, set()
    for line in Path(prefix + '.blocks.tsv').read_text().splitlines():
        row = line.split('\t')
        if len(row) != 10:
            raise ValueError('v7 AP atom row must contain ten fields')
        raw, atom, model, fixed, x, y, z, sub, model_name, molecule = row
        loc = [float(x), float(y), float(z), int(sub)]
        if atom in atoms or not all(math.isfinite(v) for v in loc):
            raise ValueError('duplicate or nonfinite AP row')
        fixed = bool(int(fixed))
        if structure and not fixed and loc[:2] != [-1., -1.]:
            raise ValueError('movable teacher XY leaked into structure')
        if structure and not fixed and loc[2] not in (-1., 0.):
            raise ValueError('movable teacher layer leaked into structure')
        block = blocks.setdefault(int(raw), dict(ap_block_id=int(raw), atoms=[], model_names=[],
                                                molecule_ids=[], fixed=fixed, loc=loc))
        if block['loc'] != loc or block['fixed'] != fixed:
            raise ValueError('split APBlock or inconsistent atom mobility')
        block['atoms'].append(atom)
        block['model_names'].append(model_name)
        block['molecule_ids'].append(molecule)
        atoms.add(atom)
    if not blocks:
        raise ValueError('empty AP snapshot')
    ids = sorted(blocks)
    remap = {raw: i for i, raw in enumerate(ids)}
    nets = []
    for line in Path(prefix + '.nets.tsv').read_text().splitlines():
        pins = list(dict.fromkeys(remap[int(p)] for p in line.split('\t')[1:]))
        if len(pins) > 1:
            nets.append(pins)
    return dict(size=size, blocks=[blocks[v] for v in ids], nets=nets, atom_count=len(atoms))


def build_logic(prefix, design, blif, architecture, *, family=None, split='train'):
    snapshot = read_snapshot(prefix, structure=True)
    model_resources = architecture_model_resources(architecture)
    clusters = []
    for i, block in enumerate(snapshot['blocks']):
        raw = Counter(model_resources.get(m) or atom_resource(m) for m in block['model_names'])
        hard = [r for r in ('memory', 'dsp_top', 'io') if raw[r]]
        if len(hard) > 1 or (hard and (raw['lut'] or raw['ff'])):
            raise ValueError('mixed hard-resource APBlock needs a coupled demand adapter')
        demand = [int(bool(raw[k])) if k in ('memory', 'dsp_top') else raw[k] for k in DEMAND_KEYS]
        cluster = dict(id=i, ap_block_id=block['ap_block_id'], atoms=block['atoms'],
                       model_names=block['model_names'], molecule_ids=block['molecule_ids'],
                       fixed=block['fixed'], demand=demand,
                       resource=hard[0] if hard else 'clb')
        if block['fixed']:
            cluster['fixed_location'] = block['loc']
        clusters.append(cluster)
    return dict(schema='tailcompile-v7-ap-logic-1', design=design, family=family or design,
                split=split, grid=snapshot['size'], clusters=clusters, nets=snapshot['nets'],
                demand_keys=list(DEMAND_KEYS), blif_sha256=sha(blif), architecture_sha256=sha(architecture),
                movable_coordinates_in_features=False,
                primitive_semantics='AP primitive upper bounds; not a packing certificate')


def quantile(values, p):
    if not values:
        return 0.
    values = sorted(values)
    x = (len(values) - 1) * p
    lo = int(math.floor(x))
    return values[lo] + (values[min(lo + 1, len(values)-1)] - values[lo]) * (x - lo)


def make_graph(logic, device):
    import torch
    if list(logic['grid']) != [device['grid_width'], device['grid_height'], device['layers']]:
        raise ValueError('AP/RRG device grid mismatch')
    demand = torch.tensor([c['demand'] for c in logic['clusters']], dtype=torch.float32)
    capacity = torch.tensor(device['capacity'], dtype=torch.float32)
    region_layer = torch.tensor(device['region_layer'], dtype=torch.long)
    layers = device['layers']
    layer_capacity = torch.zeros((layers, len(DEMAND_KEYS)), dtype=torch.float32)
    layer_capacity.index_add_(0, region_layer, capacity)
    incidence = [[] for _ in logic['clusters']]
    edges = Counter()
    skipped = 0
    for pins in logic['nets']:
        if len(pins) > 128:
            skipped += 1
            continue
        weight = 1. / max(1, len(pins)-1)
        for node in pins:
            incidence[node].append(float(len(pins)-1))
        # Symmetric hyperedge expansion, not an assertion of source/sink timing.
        if len(pins) <= 16:
            for a in pins:
                for b in pins:
                    if a != b:
                        edges[a, b] += weight
        else:
            for a, b in zip(pins, pins[1:] + pins[:1]):
                edges[a, b] += weight
                edges[b, a] += weight
    features, fixed_xy, fixed_layer, fixed_region, legal_layers = [], [], [], [], []
    W, H, _ = logic['grid']
    bounds = torch.tensor(device['region_bounds'], dtype=torch.float32)
    for i, cluster in enumerate(logic['clusters']):
        values = incidence[i]
        mean = sum(values)/len(values) if values else 0.
        tail = sorted(values, reverse=True)[:max(1, math.ceil(len(values)*.05))]
        top = sum(tail)/len(tail) if tail else 0.
        kind = ('clb', 'memory', 'dsp_top', 'io').index(cluster['resource'])
        features.append([float(kind == k) for k in range(4)] +
                        [math.log1p(v) for v in cluster['demand']] +
                        [math.log1p(len(cluster['atoms'])), math.log1p(len(values)),
                         math.log1p(mean), math.log1p(quantile(values, .95)),
                         math.log1p(quantile(values, .99)), math.log1p(top),
                         math.log1p(top/max(mean, 1e-9)), float(cluster['fixed'])])
        loc = cluster.get('fixed_location', [-1., -1., -1., -1])
        xy = [loc[0]/W if loc[0] >= 0 else -1., loc[1]/H if loc[1] >= 0 else -1.]
        z = int(loc[2]) if loc[2] >= 0 else -1
        allowed = []
        for layer in range(layers):
            fits = ((capacity[region_layer == layer] + 1e-6) >= demand[i]).all(1).any().item()
            allowed.append(bool(fits and (z < 0 or layer == z)))
        if not any(allowed):
            raise ValueError(f'APBlock {i} has no individually compatible layer')
        rid = -1
        if min(xy) >= 0 and z >= 0:
            candidates = ((bounds[:, 0] <= xy[0]) & (xy[0] < bounds[:, 1]) &
                          (bounds[:, 2] <= xy[1]) & (xy[1] < bounds[:, 3]) & (region_layer == z)).nonzero()
            if len(candidates) != 1:
                raise ValueError('fixed XYZ does not belong to exactly one leaf')
            rid = int(candidates[0])
        fixed_xy.append(xy)
        fixed_layer.append(z)
        fixed_region.append(rid)
        legal_layers.append(allowed)
    keys = sorted(edges)
    graph = dict(cluster_x=torch.tensor(features, dtype=torch.float32),
        demand=demand, ap_demand=demand, capacity=capacity,
        logic_src=torch.tensor([a for a, b in keys], dtype=torch.long),
        logic_dst=torch.tensor([b for a, b in keys], dtype=torch.long),
        logic_weight=torch.tensor([edges[k] for k in keys], dtype=torch.float32),
        region_layer=region_layer, layer_capacity=layer_capacity,
        fixed=torch.tensor([c['fixed'] for c in logic['clusters']], dtype=torch.bool),
        movable=torch.tensor([not c['fixed'] for c in logic['clusters']], dtype=torch.bool),
        fixed_xy=torch.tensor(fixed_xy, dtype=torch.float32),
        fixed_layer=torch.tensor(fixed_layer, dtype=torch.long),
        fixed_region=torch.tensor(fixed_region, dtype=torch.long),
        legal_layers=torch.tensor(legal_layers, dtype=torch.bool),
        region_bounds=bounds, nets=logic['nets'],
        channel_width=float(device.get('channel_width', 1)), grid=list(logic['grid']),
        teacher_axis_scale=[W, H],
        graph_audit=dict(skipped_global_nets=skipped, fanout_limit=128,
                         logic_edge_semantics='normalized undirected hyperedge incidence, no STA labels',
                         movable_coordinates_in_features=False))
    for name in ('region_x', 'pool_x', 'region_weight'):
        graph[name] = torch.tensor(device[name], dtype=torch.float32)
    for name in ('region_src', 'region_dst', 'pool_ids', 'pool_regions'):
        graph[name] = torch.tensor(device[name], dtype=torch.long)
    graph['pool_x'] = graph['pool_x'].reshape(-1, 8)
    graph['parents'] = [torch.tensor(p, dtype=torch.long) for p in device['parents']]
    return graph


def build_cached_graph(prefix, design, blif, arch, device_path, out, *, family=None, split='train'):
    import torch
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    signature = dict(implementation=sha(__file__), design=design, family=family, split=split,
        blif=sha(blif), arch=sha(arch), device=sha(device_path),
        snapshot={s: sha(str(prefix)+s) for s in ('.blocks.tsv', '.nets.tsv', '.meta.tsv')})
    audit = out/'graph_manifest.json'
    if audit.exists():
        old = read(audit)
        if old['signature'] != signature or sha(out/'graph.pt') != old['graph_sha256']:
            raise ValueError('graph cache changed')
        return old
    logic = build_logic(prefix, design, blif, arch, family=family, split=split)
    graph = make_graph(logic, read(device_path))
    write(out/'logic.json', logic)
    torch.save(graph, out/'graph.pt')
    row = dict(signature=signature, logic=str(out/'logic.json'), graph=str(out/'graph.pt'),
               graph_sha256=sha(out/'graph.pt'), node_count=len(logic['clusters']),
               region_count=len(graph['capacity']), **graph['graph_audit'])
    write(audit, row)
    return row


def raw_teacher(prefix, logic, *, split='train'):
    if split not in ('train', 'validation', 'qualification'):
        raise ValueError('movable teacher exports forbidden for held-out/final designs')
    snapshot = read_snapshot(prefix)
    if snapshot['size'] != logic['grid']:
        raise ValueError('teacher grid changed')
    by_atoms = {frozenset(b['atoms']): b for b in snapshot['blocks']}
    W, H, layers = logic['grid']
    xy, z = [], []
    for block in logic['clusters']:
        teacher = by_atoms.get(frozenset(block['atoms']))
        if teacher is None or bool(teacher['fixed']) != bool(block['fixed']):
            raise ValueError('teacher APBlock/fixed contract differs from structural graph')
        x, y, raw_z, sub = teacher['loc']
        if not (0 <= x < W and 0 <= y < H and 0 <= raw_z <= layers-1):
            raise ValueError('teacher XYZ outside native grid')
        xy.append([x/W, y/H])
        z.append(raw_z)
    if len(by_atoms) != len(logic['clusters']):
        raise ValueError('teacher has unexpected AP blocks')
    return dict(schema='tailcompile-v7-raw-teacher-1', teacher_xy=xy, teacher_raw_z=z,
                teacher_sha256=sha(str(prefix)+'.blocks.tsv'), split=split,
                coordinate_semantics='raw native AP continuous Z, normalized XY; no averaging')
