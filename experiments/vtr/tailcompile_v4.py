#!/usr/bin/env python3
"""Build paired 2D preplacements, holding each initializer and lowerer fixed."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from tailcompile_v4_risk import VTR_COMMIT, RiskEvaluator, digest, region_at, save, scales_for, search
from tailrisk_core import parse_blif
from tailcompile_v2 import contract_edges


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def build(logic_path, device_path, blif, supply_path, det_lowering, gnn_lowering, out,
          budget=512, seed=2027):
    from tailcompile_v3 import lower_fplace
    import torch

    out = Path(out)
    inputs = {k: digest(p) for k, p in (
        ('logic', logic_path), ('device', device_path), ('blif', blif), ('supply', supply_path),
        ('deterministic_start', det_lowering), ('gnn_start', gnn_lowering))}
    inputs.update(budget=budget, search_seed=seed, implementation=digest(Path(__file__)),
                  risk_implementation=digest(Path(__file__).with_name('tailcompile_v4_risk.py')))
    if (out / 'build_summary.json').is_file():
        old = load(out / 'build_summary.json')
        if old['inputs'] != inputs:
            raise ValueError(f'input/config changed; use a new output directory: {out}')
        for row in old['candidates']:
            if digest(out / row['fplace']) != row['fplace_sha256']:
                raise ValueError('candidate modified after build')
        return old
    if out.exists() and any(out.iterdir()):
        raise ValueError(f'incomplete build directory; preserve and choose a new one: {out}')
    logic, device, supply = load(logic_path), load(device_path), load(supply_path)
    if supply.get('provenance', {}).get('vtr_commit') != VTR_COMMIT:
        raise ValueError('production candidates require supply from the server VTR commit')
    if supply['device_ir_content_sha256'] != hashlib.sha256(json.dumps(device, sort_keys=True).encode()).hexdigest():
        raise ValueError('supply and Device IR differ')
    if digest(blif) != logic['mapped_blif_sha256']:
        raise ValueError('BLIF differs from Logic IR')
    primitives, parser_audit = parse_blif(Path(blif))
    mapping = {atom: c['id'] for c in logic['clusters'] for atom in c['atoms']}
    edges, edge_audit = contract_edges(primitives, mapping)
    evaluator = RiskEvaluator(logic, device, supply, edges)
    starts = {}
    for name, path in (('det', det_lowering), ('gnn', gnn_lowering)):
        records = sorted(load(path)['clusters'], key=lambda c: c['cluster'])
        if len(records) != len(logic['clusters']):
            raise ValueError('initializer cluster count mismatch')
        assignment, coordinates = [], []
        for c, rec in zip(logic['clusters'], records):
            if rec['cluster'] != c['id'] or rec['resource'] != c['resource']:
                raise ValueError('initializer cluster identity mismatch')
            rid = region_at(device, rec['x'], rec['y'])
            if c['resource'] != 'io' and rid != rec['region']:
                raise ValueError('initializer physical coordinates contradict its region')
            assignment.append(rid)  # IO regions reflect actual fixed perimeter anchors.
            r = device['regions'][rid]
            xy = rec.get('relative_xy', [
                2 * (rec['x'] - r['center_x']) / max(r['x_high'] - r['x_low'], 1),
                2 * (rec['y'] - r['center_y']) / max(r['y_high'] - r['y_low'], 1)])
            coordinates.append([max(-1.0, min(1.0, float(v))) for v in xy])
        starts[name] = assignment, torch.tensor(coordinates, dtype=torch.float32)
    reference = evaluator.state(starts['det'][0])
    if reference is None:
        raise ValueError('deterministic start is unreachable under the coarse wire-span proxy')
    scales = scales_for(reference)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    start_time = time.monotonic()
    for name, (assignment, coordinates) in starts.items():
        for suffix, mode, policy in (('base', 'tail', 'hotspot'),
                                     ('tail', 'tail', 'hotspot'),
                                     ('mean', 'mean', 'hotspot'),
                                     ('none', 'none', 'hotspot'),
                                     ('random', 'tail', 'random_restart')):
            method = f'{name}_{suffix}'
            print(json.dumps({'stage': 'refine', 'design': logic['design'], 'method': method}), flush=True)
            final, audit = search(evaluator, assignment, 0 if suffix == 'base' else budget,
                                  seed, mode, scales, policy,
                                  progress=lambda row: print(json.dumps({'method': method, **row}), flush=True))
            lower_start = time.monotonic()
            lowering = lower_fplace(logic, device, final, coordinates, out / f'{method}.fplace')
            lower_seconds = time.monotonic() - lower_start
            # Same IO ordering/subtiles as v3. Check the evaluator used those exact locations.
            for rec in lowering['clusters']:
                if region_at(device, rec['x'], rec['y']) != final[rec['cluster']]:
                    raise AssertionError('lowerer and risk evaluator disagree about actual region')
            save(out / f'{method}_lowering.json', lowering)
            save(out / f'{method}_binding.json', {'assignment': final, 'schema': 'tailcompile-v4-binding'})
            save(out / f'{method}_search.json', audit)
            results.append({'method': method, 'fplace': f'{method}.fplace',
                            'fplace_sha256': digest(out / f'{method}.fplace'),
                            'search': f'{method}_search.json',
                            'binding': f'{method}_binding.json', 'lowering': f'{method}_lowering.json',
                            'evaluations': audit['evaluations'], 'accepted': audit['accepted'],
                            'search_seconds': audit['seconds'],
                            'lowering_seconds': lower_seconds,
                            'initial_proxy': {k: v for k, v in audit['initial'].items() if k != 'pools'},
                            'final_proxy': {k: v for k, v in audit['final'].items() if k != 'pools'}})
    summary = {'schema': 'tailcompile-v4-build-v1', 'design': logic['design'],
               'channel_width': supply['channel_width'], 'inputs': inputs,
               'parser_audit': parser_audit, 'edge_audit': edge_audit,
               'demand_policy': 'one unit per net per shared wire pool; fanout paths unioned; clocks/constants excluded by tailrisk_core',
               'driverless_net_count': sum(not e.drivers for e in edges),
               'timing_policy': 'existing pre-placement unit-delay depth proxy; not STA/CPD',
               'normalization_reference': 'same deterministic start for all methods',
               'search_seconds_total': time.monotonic() - start_time, 'candidates': results}
    save(out / 'build_summary.json', summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for flag in ('logic-ir', 'device-ir', 'blif', 'supply', 'det-lowering', 'gnn-lowering', 'out'):
        p.add_argument('--' + flag, type=Path, required=True)
    p.add_argument('--budget', type=int, default=512)
    p.add_argument('--seed', type=int, default=2027)
    a = p.parse_args()
    build(a.logic_ir, a.device_ir, a.blif, a.supply, a.det_lowering, a.gnn_lowering,
          a.out, a.budget, a.seed)


if __name__ == '__main__':
    main()
