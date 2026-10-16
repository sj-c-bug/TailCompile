#!/usr/bin/env python3
"""Build seed-aggregated v6 labels from a completed v5 confirmation campaign."""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from extract_congestion import channels, percentile
from tailcompile_v4_risk import digest, region_at, save


QOR = ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def actual_hotspot(run, device):
    """Aggregate VPR channel occupancy into the same 4x4 physical regions."""
    records = channels(run / 'chanx_occupancy.txt', 'x')
    records += channels(run / 'chany_occupancy.txt', 'y')
    totals = [[[0.0, 0.0] for _ in range(2)] for _ in device['regions']]
    for _, x, y, occupancy, capacity, direction in records:
        rid = region_at(device, x, y)
        axis = 0 if direction == 'x' else 1
        totals[rid][axis][0] += occupancy
        totals[rid][axis][1] += capacity
    region = []
    for axes in totals:
        occupancy = sum(row[0] for row in axes)
        capacity = sum(row[1] for row in axes)
        region.append({
            'utilization': occupancy / capacity if capacity else 0.0,
            'x_utilization': axes[0][0] / axes[0][1] if axes[0][1] else 0.0,
            'y_utilization': axes[1][0] / axes[1][1] if axes[1][1] else 0.0,
        })
    return region


def median_hotspots(values):
    if not values:
        return None
    return [{key: statistics.median(sample[index][key] for sample in values)
             for key in ('utilization', 'x_utilization', 'y_utilization')}
            for index in range(len(values[0]))]


def verify_manifest(metrics, manifest, spec, path):
    """Verify provenance while tolerating Git LF/CRLF checkout conversion."""
    current = digest(path)
    if metrics['manifest_sha256'] == current:
        return None
    semantic = (
        metrics.get('design') == spec['design'] and
        int(metrics.get('width')) == int(spec['width']) and
        metrics.get('method') == spec['method'] and
        int(metrics.get('seed')) == int(spec['seed']) and
        metrics.get('status') == manifest.get('vpr_status') and
        int(manifest.get('seed')) == int(spec['seed']) and
        int(manifest.get('route_chan_width')) == int(spec['width']))
    if not semantic:
        raise ValueError(f'metrics/manifest semantic provenance mismatch: {path.parent}')
    return {
        'kind': 'manifest_byte_hash_changed_but_semantics_match',
        'manifest': str(path),
        'recorded_sha256': metrics['manifest_sha256'],
        'checkout_sha256': current,
        'likely_cause': 'cross-platform LF/CRLF conversion by Git',
    }


def prepare(root, out):
    root, out = Path(root).resolve(), Path(out).resolve()
    summary, plan = load(root / 'summary.json'), load(root / 'plan.json')
    if summary.get('missing') or summary.get('metric_errors'):
        raise ValueError('v5 campaign is incomplete or contains metric errors')
    trials = defaultdict(list)
    provenance_warnings = []
    specs = {(row['design'], int(row['width']), row['method'], int(row['seed'])): row
             for row in plan['runs']}
    for key, spec in specs.items():
        run = root / spec['directory']
        metrics, manifest = load(run / 'metrics.json'), load(run / 'manifest.json')
        warning = verify_manifest(metrics, manifest, spec, run / 'manifest.json')
        if warning:
            provenance_warnings.append(warning)
        if manifest['vpr_status'] not in ('success', 'unroutable'):
            raise ValueError(f'non-scientific VPR failure: {run}')
        trials[key[:3]].append((spec, metrics, run))
    designs = []
    for design in sorted({key[0] for key in trials}):
        input_spec = next(spec for (name, _, _), rows in trials.items() if name == design
                          for spec, _, _ in rows)
        inputs = root / input_spec['inputs']
        logic_path, device_path = inputs / 'logic_ir.json', inputs / 'device_ir.json'
        blif_path = inputs / f'{design}.pre-vpr.blif'
        device = load(device_path)
        conditions = []
        for (name, width, method), rows in sorted(trials.items()):
            if name != design or method == 'default':
                continue
            candidate = root / rows[0][0]['candidate']
            binding = candidate.with_name(candidate.stem + '_binding.json')
            search = candidate.with_name(candidate.stem + '_search.json')
            successes = [metrics for _, metrics, _ in rows if metrics['status'] == 'success']
            hotspots = [actual_hotspot(run, device) for _, metrics, run in rows
                        if metrics['status'] == 'success']
            condition = {
                'design': design, 'width': width, 'method': method,
                'trial_count': len(rows),
                'route_success_rate': len(successes) / len(rows),
                'status_counts': dict(Counter(metrics['status'] for _, metrics, _ in rows)),
                'qor': {metric: statistics.median(row[metric] for row in successes)
                        if successes else None for metric in QOR},
                'actual_hotspot': median_hotspots(hotspots),
                'assignment': load(binding)['assignment'],
                'binding_sha256': digest(binding),
                'candidate_sha256': digest(candidate),
                'search': str(search),
                'supply': str(root / design / f'w{width}' / 'supply.json'),
            }
            conditions.append(condition)
        designs.append({
            'design': design,
            'logic_ir': str(logic_path), 'device_ir': str(device_path),
            'mapped_blif': str(blif_path), 'conditions': conditions,
        })
    payload = {
        'schema': 'tailcompile-v6-spatial-routability-v1',
        'source_v5_root': str(root),
        'source_summary_sha256': digest(root / 'summary.json'),
        'seed_policy': 'placement seeds are nuisance repetitions aggregated before training',
        'comparison': 'same examples, labels, design folds, model capacity and optimization budget',
        'label_policy': {
            'route': 'success fraction across placement seeds',
            'qor': 'median over successful placement seeds',
            'hotspot': 'per-region median VPR channel occupancy/capacity over successful seeds',
        },
        'provenance_warnings': provenance_warnings,
        'designs': designs,
    }
    out.mkdir(parents=True, exist_ok=True)
    save(out / 'dataset.json', payload)
    save(out / 'dataset_summary.json', {
        'design_count': len(designs),
        'condition_count': sum(len(row['conditions']) for row in designs),
        'route_label_histogram': dict(Counter(
            str(condition['route_success_rate']) for row in designs for condition in row['conditions'])),
        'source_v5_root': str(root),
        'provenance_warning_count': len(provenance_warnings),
    })
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v5-root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    payload = prepare(args.v5_root, args.out)
    print(json.dumps({'stage': 'v6_dataset_complete',
                      'design_count': len(payload['designs']),
                      'condition_count': sum(len(row['conditions']) for row in payload['designs'])}))


if __name__ == '__main__':
    main()
