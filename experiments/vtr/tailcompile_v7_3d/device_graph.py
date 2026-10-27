"""Fixed-tile, layered device tensors from actual independent RRG wire pools.

The audited legacy streaming parser is used at one physical tile per bin, then
shared pool footprints are contracted to exact tile_size leaves. This avoids
changing the frozen 2D parser or stretching bins to fit the whole device.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from architectures import _module, sha

KEYS = ('lut', 'ff', 'memory', 'dsp_top', 'io')


def resource_units(architecture):
    """Maximum mutually exclusive mode demands per physical root tile."""
    root = ET.parse(architecture).getroot()
    def units(pb, resource):
        model = pb.get('blif_model', '').lower()
        if model in ('.names', '.latch'):
            return Counter({('lut' if model == '.names' else 'ff'): 1})
        if model and resource in ('memory', 'dsp_top', 'io'):
            return Counter({resource: 1})
        alternatives = list(pb.findall('mode')) or [pb]
        values = []
        for alternative in alternatives:
            value = Counter()
            for child in alternative.findall('pb_type'):
                value.update({k: v * int(child.get('num_pb', 1))
                              for k, v in units(child, resource).items()})
            values.append(value)
        return Counter({k: max((v[k] for v in values), default=0) for k in KEYS})
    logical = {pb.get('name'): dict(units(pb, pb.get('name')))
               for pb in root.findall('./complexblocklist/pb_type')}
    physical = {}
    for tile in root.findall('./tiles/tile'):
        total = Counter()
        for sub in tile.findall('sub_tile'):
            alternatives = [logical.get(site.get('pb_type'), {})
                            for site in sub.findall('./equivalent_sites/site')]
            total.update({k: int(sub.get('capacity', 1)) * max((v.get(k, 0) for v in alternatives), default=0)
                          for k in KEYS})
        physical[tile.get('name')] = dict(total)
    return physical


def _grid_size(rr_graph):
    maxima = [-1, -1, -1]
    opener = gzip.open if str(rr_graph).endswith('.gz') else open
    with opener(rr_graph, 'rb') as stream:
        for event, node in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start' and node.tag == 'rr_nodes':
                break
            if event == 'end' and node.tag == 'grid_loc':
                for a, key in enumerate(('x', 'y', 'layer')):
                    maxima[a] = max(maxima[a], int(node.get(key, '0')))
                node.clear()
    if min(maxima) < 0:
        raise ValueError('RRG grid metadata missing before rr_nodes')
    return [v + 1 for v in maxima]


def _vertical_switch_audit(rr_graph):
    """Count actual edge switch classes adjacent to CHANZ, never as capacity."""
    names, vertical, counts = {}, set(), Counter()
    opener = gzip.open if str(rr_graph).endswith('.gz') else open
    with opener(rr_graph, 'rb') as stream:
        stack = []
        for event, item in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start':
                stack.append(item)
                continue
            parent = stack[-2] if len(stack) > 1 else None
            tag = parent.tag if parent is not None else None
            if item.tag == 'switch' and tag == 'switches':
                names[int(item.get('id'))] = item.get('name', 'unnamed')
            elif item.tag == 'node' and tag == 'rr_nodes' and item.get('type') == 'CHANZ':
                vertical.add(int(item.get('id')))
            elif item.tag == 'edge' and tag == 'rr_edges':
                a, b = int(item.get('src_node')), int(item.get('sink_node'))
                if a != b and (a in vertical or b in vertical):
                    sid = int(item.get('switch_id'))
                    counts[names.get(sid, f'switch_id:{sid}')] += 1
            if parent is not None and tag in ('rr_nodes', 'rr_edges', 'grid', 'segments',
                                              'channels', 'block_types', 'switches'):
                parent.remove(item)
            stack.pop()
    return dict(sorted(counts.items()))


def build_device_graph(rr_graph, architecture, tile_size=4, max_seconds=300):
    started = time.perf_counter()
    if not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError('tile_size must be a positive integer')
    w, h, layers = _grid_size(rr_graph)
    # Exact unit bins let the unchanged legacy parser supply complete footprints.
    device, supply = _module('device_supply_3d').build_device_supply(
        rr_graph, architecture, columns=w, rows=h, max_seconds=max_seconds)
    nx, ny = math.ceil(w/tile_size), math.ceil(h/tile_size)
    per_layer = nx*ny
    def region_at(x, y, z):
        return int(z)*per_layer + (int(y)//tile_size)*nx + int(x)//tile_size
    def contract(rid):
        z, rem = divmod(rid, w*h)
        y, x = divmod(rem, w)
        return region_at(x, y, z)
    regions = []
    for z in range(layers):
        for row in range(ny):
            for col in range(nx):
                bounds = [col*tile_size, min((col+1)*tile_size, w),
                          row*tile_size, min((row+1)*tile_size, h)]
                regions.append(dict(id=len(regions), layer=z, column=col, row=row,
                                    bounds=bounds, tile_counts={}, tile_coordinates={}))
    units = resource_units(architecture)
    for unit in device['regions']:
        for typ, coordinates in unit['tile_coordinates'].items():
            for x, y, subtile, z in coordinates:
                region = regions[region_at(x, y, z)]
                region['tile_coordinates'].setdefault(typ, []).append([x, y, subtile, z])
    capacity = []
    for region in regions:
        roots = {typ: len({(p[0], p[1]) for p in coords})
                 for typ, coords in region['tile_coordinates'].items()}
        region['tile_counts'] = roots  # Root count, not an IO subtile count.
        cap = [float(sum(count*units.get(typ, {}).get(key, 0)
                         for typ, count in roots.items())) for key in KEYS]
        region['capacity'] = cap
        capacity.append(cap)
    aggregate = {}
    z_xy = defaultdict(float)
    for pool in supply['pools']:
        if pool['kind'] == 'CHANZ':
            # At unit-bin resolution a vertical footprint has a single XY.
            # Count each shared pool once, regardless of its two BI_DIR arcs.
            point = pool['arcs'][0][0] % (w*h)
            y, x = divmod(point, w)
            z_xy[f'{x},{y}'] += pool['capacity']
        arcs = tuple(sorted({(contract(a), contract(b)) for a, b in pool['arcs']
                             if contract(a) != contract(b)}))
        if not arcs:
            continue
        key = (pool['kind'], pool['direction'], pool['segment'], arcs)
        value = aggregate.setdefault(key, dict(capacity=0., node_count=0, example_rr_node_ids=[]))
        value['capacity'] += pool['capacity']
        value['node_count'] += pool['node_count']
        value['example_rr_node_ids'] = (value['example_rr_node_ids'] + pool['example_rr_node_ids'])[:8]
    pools = [dict(id=i, kind=key[0], direction=key[1], segment=key[2],
                  arcs=[list(a) for a in key[3]], **value)
             for i, (key, value) in enumerate(sorted(aggregate.items()))]
    width = supply['channel_width']
    edges = defaultdict(float)
    pool_ids, pool_regions, pool_x = [], [], []
    directional = [[0.]*6 for _ in regions]
    segment_supply = [[0.]*2 for _ in regions]
    z_classes = defaultdict(float)
    for pool in pools:
        kind, direction, cap = pool['kind'], pool['direction'], pool['capacity']
        axis = ('CHANX', 'CHANY', 'CHANZ').index(kind)
        pool_x.append([math.log1p(cap/width), math.log1p(pool['node_count']/width)] +
                      [float(axis == a) for a in range(3)] +
                      [float(direction == d) for d in ('INC_DIR', 'DEC_DIR', 'BI_DIR')])
        touched = sorted({r for arc in pool['arcs'] for r in arc})
        for rid in touched:
            pool_ids.append(pool['id'])
            pool_regions.append(rid)
            if direction in ('INC_DIR', 'BI_DIR'):
                directional[rid][axis*2] += cap/width
            if direction in ('DEC_DIR', 'BI_DIR'):
                directional[rid][axis*2+1] += cap/width
            if pool['segment'] in ('L4', 'L16'):
                segment_supply[rid][('L4', 'L16').index(pool['segment'])] += cap/width
        for a, b in pool['arcs']:
            edges[a, b] += cap/width
        if kind == 'CHANZ':
            z_classes[f'{direction}|{pool["segment"]}'] += cap
    normalized_bounds = [[r['bounds'][0]/w, r['bounds'][1]/w,
                          r['bounds'][2]/h, r['bounds'][3]/h] for r in regions]
    feature_names = ['center_x', 'center_y', 'layer_fraction'] + [f'log_capacity_{k}' for k in KEYS] + [
        'log_x_inc', 'log_x_dec', 'log_y_inc', 'log_y_dec', 'log_z_inc', 'log_z_dec',
        'log_L4_supply', 'log_L16_supply']
    region_x = []
    for r, bound in zip(regions, normalized_bounds):
        region_x.append([(bound[0]+bound[1])/2, (bound[2]+bound[3])/2,
                         r['layer']/max(1, layers-1)] + [math.log1p(c) for c in r['capacity']] +
                        [math.log1p(s) for s in directional[r['id']]] +
                        [math.log1p(s) for s in segment_supply[r['id']]])
    parents = []
    level_nx, level_ny = nx, ny
    while level_nx > 1 or level_ny > 1:
        px, py = math.ceil(level_nx/2), math.ceil(level_ny/2)
        parents.append([z*px*py+(y//2)*px+x//2 for z in range(layers)
                        for y in range(level_ny) for x in range(level_nx)])
        level_nx, level_ny = px, py
    layer_capacity = [[sum(capacity[r][a] for r in range(z*per_layer, (z+1)*per_layer))
                       for a in range(len(KEYS))] for z in range(layers)]
    z_switch_classes = sorted({link.get('mux') for link in ET.parse(architecture).getroot().findall(
        './scatter_gather_list/sg_pattern/sg_link_list/sg_link')})
    actual_switch_counts = _vertical_switch_audit(rr_graph)
    z_mass = sum(z_xy.values())
    z_radius = sum(cap*((int(key.split(',')[0])-(w-1)/2)**2 +
                       (int(key.split(',')[1])-(h-1)/2)**2)
                   for key, cap in z_xy.items()) / max(z_mass, 1.)
    audit = dict(grid=[w, h, layers], total_resource_capacity=device['total_resource_capacity'],
                 layer_resource_capacity=device['layer_resource_capacity'],
                 z_capacity_by_class=dict(sorted(z_classes.items())),
                 z_switch_classes=z_switch_classes,
                 z_switch_classes_source='unchanged XML sg_link mux declarations',
                 z_actual_adjacent_switch_classes=sorted(actual_switch_counts),
                 z_actual_adjacent_switch_edge_counts=actual_switch_counts,
                 z_capacity_by_xy=dict(sorted(z_xy.items())),
                 z_mean_squared_distance_to_center=z_radius,
                 unique_connected_z_capacity=supply['vertical_connection_audit']['included_capacity'],
                 vertical_connection_audit=supply['vertical_connection_audit'],
                 rr_node_counts=supply['rr_node_counts'], rr_node_count=supply['rr_node_count'],
                 rr_edge_count=supply['rr_edge_count'], capacity_demand_keys=list(KEYS))
    if time.perf_counter() - started > max_seconds:
        raise TimeoutError('complete device graph extraction exceeded its preparation budget')
    return dict(schema='tailcompile-v7-layered-device-1', grid_width=w, grid_height=h,
                layers=layers, grid=dict(width=w, height=h, layers=layers), tile_size=tile_size,
                channel_width=width, region_grid=dict(columns=nx, rows=ny, layers=layers),
                region_x=region_x, region_feature_names=feature_names, regions=regions,
                region_bounds=normalized_bounds, region_layer=[r['layer'] for r in regions],
                region_src=[a for a, b in sorted(edges)], region_dst=[b for a, b in sorted(edges)],
                region_weight=[edges[e] for e in sorted(edges)], parents=parents,
                pool_x=pool_x, pool_ids=pool_ids, pool_regions=pool_regions, pools=pools,
                pool_feature_names=['log_capacity_per_width', 'log_nodes_per_width',
                                    'is_x', 'is_y', 'is_z', 'inc', 'dec', 'bi'],
                capacity=capacity, layer_capacity=layer_capacity, resource_units=units,
                provenance=supply['provenance'], audit=audit,
                extraction_seconds=time.perf_counter()-started,
                limitations=['capacity is an AP primitive upper-bound proxy, not packing legality',
                             'pin access and switch-only crossings are not certified by pooled footprints',
                             'active multi-height/macro tiles require a separately qualified adapter'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rr-graph', type=Path, required=True)
    p.add_argument('--architecture', type=Path, required=True)
    p.add_argument('--tile-size', type=int, default=4)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        p.error('refusing to overwrite a device graph')
    value = build_device_graph(args.rr_graph, args.architecture, args.tile_size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(value, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(dict(regions=len(value['regions']), pools=len(value['pools']),
                          seconds=value['extraction_seconds'], audit=value['audit']), indent=2))


if __name__ == '__main__':
    main()
