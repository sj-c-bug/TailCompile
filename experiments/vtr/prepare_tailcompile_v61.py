"""Re-express existing VPR labels on fixed-tile blocks without new VPR runs."""
import argparse
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import copy
import json
import math
from pathlib import Path
from collections import defaultdict
import torch
from tailcompile_v4_risk import build_supply, digest, region_at, save, RiskEvaluator, VTR_COMMIT
from tailcompile_v2 import contract_edges
from tailrisk_core import parse_blif
from prepare_tailcompile_v6 import actual_hotspot, median_hotspots
from tailcompile_v61 import SCHEMA, static_graph, candidate_graph


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def tile_device(original, size):
    if size < 1:
        raise ValueError('tile size must be positive')
    width, height = original['grid_width'], original['grid_height']
    columns, rows = math.ceil((width - 2) / size), math.ceil((height - 2) / size)
    device = copy.deepcopy(original)
    device['regions'] = []
    device['region_grid'] = dict(columns=columns, rows=rows, count=columns * rows)
    for row in range(rows):
        for col in range(columns):
            xl, yl = 1 + col * size, 1 + row * size
            xh, yh = min(width - 2, xl + size - 1), min(height - 2, yl + size - 1)
            device['regions'].append(dict(id=len(device['regions']), row=row, column=col,
                x_low=xl, x_high=xh, y_low=yl, y_high=yh, center_x=(xl + xh) / 2,
                center_y=(yl + yh) / 2, tile_coordinates={}, tile_counts={},
                channel_internal=0., escape_left=0., escape_right=0., escape_up=0.,
                escape_down=0., segment_supply={}))
    # Original IR contains distinct root anchors only (multi-height offsets excluded).
    seen = set()
    for old in original['regions']:
        for kind, coords in old['tile_coordinates'].items():
            for xy in coords:
                key = kind, *xy
                if key in seen:
                    raise ValueError('duplicate physical anchor')
                seen.add(key)
                r = device['regions'][region_at(device, *xy)]
                r['tile_coordinates'].setdefault(kind, []).append(xy)
    for r in device['regions']:
        r['tile_counts'] = {k: len(v) for k, v in r['tile_coordinates'].items()}
    return device


def supply_features(device, supply):
    # Features deliberately describe boundary pools; no claim of pin access coverage.
    arcs = defaultdict(float)
    for p in supply['pools']:
        cap = p['capacity'] / supply['channel_width']
        for rid in {r for arc in p['arcs'] for r in arc}:
            region = device['regions'][rid]
            region['segment_supply'][p['segment']] = region['segment_supply'].get(p['segment'], 0.) + cap
        for a, b in p['arcs']:
            arcs[a, b] += cap
            ra, rb = device['regions'][a], device['regions'][b]
            direction = ('right' if rb['center_x'] > ra['center_x'] else 'left') if p['kind'] == 'CHANX' else (
                'up' if rb['center_y'] > ra['center_y'] else 'down')
            ra['escape_' + direction] += cap
    device['inter_region_rr_edge_counts_per_channel_width'] = [
        dict(region_a=a, region_b=b, count=c) for (a, b), c in arcs.items()]
    device['v61_feature_semantics'] = 'boundary wire pools; no switch/pin-access reachability guarantee'


def prepare(root, out, tile_size):
    root, out = Path(root).resolve(), Path(out).resolve()
    if (out / 'prepare_report.json').exists():
        previous = load(out / 'prepare_report.json')
        if (previous['source'] != str(root) or previous['tile_size'] != tile_size or
                previous['source_plan_sha256'] != digest(root / 'plan.json') or
                previous['implementation'] != digest(Path(__file__)) or
                previous['bundle_sha256'] != digest(out / 'bundle.pt')):
            raise ValueError('preparation inputs changed; choose new output directory')
        print(json.dumps(dict(stage='reuse_bundle', path=str(out / 'bundle.pt'))), flush=True)
        return
    if out.exists() and any(out.iterdir()):
        raise ValueError('use a new empty preparation directory (immutable tensor bundle)')
    plan = load(root / 'plan.json')
    records = [s for s in plan['runs'] if s.get('candidate')]
    if not records:
        raise ValueError('no candidate trials in source plan')
    contexts, groups = {}, defaultdict(list)
    for spec in records:
        groups[spec['design'], int(spec['width']), spec['method']].append(spec)
    examples, audits = [], []
    for (design, width, method), specs in sorted(groups.items()):
        key = design, width
        folder = root / design / f'w{width}'
        inputs = root / specs[0]['inputs']
        if key not in contexts:
            logic = load(inputs / 'logic_ir.json')
            old = load(inputs / 'device_ir.json')
            if logic.get('macros') or any(c.get('macro_id') is not None for c in logic['clusters']):
                raise ValueError('coupled macro moves are not implemented')
            if tile_size:
                if any('allowed_regions' in c for c in logic['clusters']):
                    raise ValueError('explicit region restrictions require physical remapping')
                device = tile_device(old, tile_size)
                rr = folder / 'supply_run/device.rr_graph.xml'
                if not rr.exists():
                    rr = rr.with_suffix('.xml.gz')
                supply = build_supply(rr, device)
                if supply['rr_graph_sha256'] != load(folder / 'supply.json')['rr_graph_sha256']:
                    raise ValueError('raw RRG differs from archived supply')
                supply_features(device, supply)
            else:
                device, supply = old, load(folder / 'supply.json')
            if supply['channel_width'] != width:
                raise ValueError('RRG width mismatch')
            blif = inputs / f'{design}.pre-vpr.blif'
            primitives, _ = parse_blif(blif)
            mapping = {a: c['id'] for c in logic['clusters'] for a in c['atoms']}
            edges, _ = contract_edges(primitives, mapping)
            evaluator = RiskEvaluator(logic, device, supply, edges)
            base = static_graph(logic, device, blif, supply)
            context_dir = out / design / f'w{width}'
            save(context_dir / 'device.json', device)
            save(context_dir / 'supply.json', supply)
            contexts[key] = dict(base=base, logic=logic, device=device, evaluator=evaluator,
                inputs=str(inputs), folder=str(folder), context_dir=str(context_dir))
        ctx = contexts[key]
        candidate = root / specs[0]['candidate']
        if len({s['seed'] for s in specs}) != len(specs):
            raise ValueError('duplicate placement seed')
        lowering = load(candidate.with_name(candidate.stem + '_lowering.json'))
        clusters = sorted(lowering['clusters'], key=lambda c: c['cluster'])
        if [c['cluster'] for c in clusters] != list(range(len(ctx['logic']['clusters']))):
            raise ValueError('lowering cluster IDs mismatch')
        assignment = [region_at(ctx['device'], c['x'], c['y']) for c in clusters]
        successful, hotspots = [], []
        for s in specs:
            directory = root / s['directory']
            meta, metric = load(directory / 'manifest.json'), load(directory / 'metrics.json')
            if (meta['vtr_commit'] != VTR_COMMIT or meta['seed'] != s['seed'] or
                    meta['route_chan_width'] != width or metric['status'] != meta['vpr_status']):
                raise ValueError('trial identity or VTR version mismatch')
            if metric['manifest_sha256'] != digest(directory / 'manifest.json'):
                raise ValueError('manifest hash mismatch; prepare on original server archive')
            if meta['signature']['input_fplace_sha256'] != digest(candidate):
                raise ValueError('candidate hash mismatch')
            if (meta['signature']['mapped_blif_sha256'] != digest(inputs / f'{design}.pre-vpr.blif') or
                    meta['signature']['input_packed_net_sha256'] != digest(inputs / f'{design}.net')):
                raise ValueError('netlist provenance mismatch')
            if meta['vpr_status'] not in ('success', 'unroutable'):
                raise ValueError('non-routing failure in training labels')
            if meta['vpr_status'] == 'success':
                successful.append(metric)
                hotspots.append(actual_hotspot(directory, ctx['device']))
        import statistics
        target = median_hotspots(hotspots)
        graph = candidate_graph(ctx['base'], ctx['logic'], ctx['device'], ctx['evaluator'], assignment)
        examples.append(dict(design=design, width=width, method=method,
            route=len(successful) / len(specs),
            qor=[statistics.median(m[k] for m in successful) if successful else None
                 for k in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')],
            hotspot=[r['utilization'] for r in target] if target else None,
            graph=graph))
        audits.append(dict(design=design, width=width, method=method, seeds=[s['seed'] for s in specs],
            regions=len(ctx['device']['regions']), candidate_sha256=digest(candidate)))
        print(json.dumps(dict(stage='prepared', design=design, width=width, method=method,
                              regions=len(ctx['device']['regions']))), flush=True)
    portable = {key: {k: v for k, v in ctx.items() if k not in ('base', 'evaluator')}
                for key, ctx in contexts.items()}
    torch.save(dict(schema=SCHEMA, examples=examples, contexts=portable), out / 'bundle.pt')
    save(out / 'prepare_report.json', dict(schema=SCHEMA, source=str(root), tile_size=tile_size,
        source_plan_sha256=digest(root / 'plan.json'), implementation=digest(Path(__file__)),
        bundle_sha256=digest(out / 'bundle.pt'), conditions=audits,
        limits=['pool span connectivity is approximate', 'no pin-access encoding',
                'failed routes have no spatial utilization label']))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--tile-size', type=int, default=8, help='0: existing regions; positive: fixed tile block size')
    a = p.parse_args()
    if a.tile_size < 0:
        p.error('tile size must be nonnegative')
    prepare(a.source, a.out, a.tile_size)
