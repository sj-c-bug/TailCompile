"""Exact-schema graph optimization for repair cohort006; frozen data.py is untouched."""
from __future__ import annotations
from collections import Counter
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from data import build_logic, DEMAND_KEYS, quantile
from io_utils import read, write, sha, install_signal_handlers
from repository_guard import validate_output_roots

def make_graph(logic, device, *, profile=None):
    started = time.perf_counter()
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
    edge_done = time.perf_counter()
    # Exact float32 comparison, once per distinct demand instead of per APBlock.
    unique, inverse = torch.unique(demand, dim=0, return_inverse=True)
    compatible = torch.zeros((len(unique), layers), dtype=torch.bool)
    for layer in range(layers):
        supply = capacity[region_layer == layer] + 1e-6
        # Bound the temporary demand x region x resource tensor.
        chunk = max(1, 1048576 // max(1, len(supply) * demand.shape[1]))
        for lo in range(0, len(unique), chunk):
            compatible[lo:lo+chunk, layer] = (supply[None] >= unique[lo:lo+chunk, None]).all(2).any(1)
    node_compatible = compatible[inverse].tolist()
    compatibility_done = time.perf_counter()
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
        allowed = [fits and (z < 0 or layer == z)
                   for layer, fits in enumerate(node_compatible[i])]
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
    feature_done = time.perf_counter()
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
                         demand_semantics=logic.get('demand_semantics', 'unversioned_manual_fixture'),
                         dsp_mode_units_per_root=(logic.get('dsp_mode_catalog') or {}).get('units_per_root'),
                         movable_coordinates_in_features=False))
    for name in ('region_x', 'pool_x', 'region_weight'):
        graph[name] = torch.tensor(device[name], dtype=torch.float32)
    for name in ('region_src', 'region_dst', 'pool_ids', 'pool_regions'):
        graph[name] = torch.tensor(device[name], dtype=torch.long)
    graph['pool_x'] = graph['pool_x'].reshape(-1, 8)
    graph['parents'] = [torch.tensor(p, dtype=torch.long) for p in device['parents']]
    if profile is not None:
        profile.update(edges_seconds=edge_done-started, compatibility_seconds=compatibility_done-edge_done,
                       features_seconds=feature_done-compatibility_done, tensor_output_seconds=time.perf_counter()-feature_done,
                       nodes=len(demand), unique_demands=len(unique), regions=len(capacity))
    return graph


def semantic_digest(graph):
    """Tensor values and metadata, independent of torch.save ZIP file details."""
    h = hashlib.sha256()
    def visit(value):
        if isinstance(value, torch.Tensor):
            t = value.detach().cpu().contiguous()
            h.update(json.dumps([str(t.dtype), list(t.shape)]).encode())
            h.update(t.numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value):
                h.update(json.dumps(key).encode()); visit(value[key])
        elif isinstance(value, (list, tuple)):
            h.update(str(len(value)).encode())
            for item in value:
                visit(item)
        else:
            h.update(json.dumps(value, sort_keys=True, allow_nan=False).encode())
    visit(graph)
    return h.hexdigest()


def build(request, out):
    """Cold build in an owned subprocess. Caller measures process startup too."""
    from data import make_graph as old_make_graph
    out = Path(out).resolve()
    validate_output_roots((out,))
    out.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    logic = build_logic(request['prefix'], request['design'], request['blif'], request['arch'],
                        family=request['family'], split=request['split'])
    parse_done = time.perf_counter()
    device = read(request['device'])
    device_done = time.perf_counter()
    profile = {}
    graph = (old_make_graph(logic, device) if request.get('engine') == 'legacy' else
             make_graph(logic, device, profile=profile))
    graph_done = time.perf_counter()
    digest = semantic_digest(graph)
    digest_done = time.perf_counter()
    # Torch binaries retain exact schema. Compact JSON removes pretty-print IO
    # only in this new payload; full logical information is preserved.
    (out/'logic.json').write_text(json.dumps(logic, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    torch.save(graph, out/'graph.pt')
    serialize_done = time.perf_counter()
    row = dict(logic=str(out/'logic.json'), graph=str(out/'graph.pt'),
        graph_sha256=sha(out/'graph.pt'), logic_sha256=sha(out/'logic.json'), semantic_sha256=digest,
        node_count=len(logic['clusters']), region_count=len(graph['capacity']),
        profile=dict(logic_parse_seconds=parse_done-start, device_read_seconds=device_done-parse_done,
            make_graph_seconds=graph_done-device_done, semantic_hash_seconds=digest_done-graph_done,
            serialization_seconds=serialize_done-digest_done, **profile),
        **graph['graph_audit'])
    row['profile']['file_hash_seconds'] = time.perf_counter()-serialize_done
    write(out/'graph_manifest.json', row)
    print('TAILCOMPILE_V7_FAST_GRAPH_COMPLETE', flush=True)
    return row


def main():
    install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    build(read(args.request), args.out)


if __name__ == '__main__':
    main()
