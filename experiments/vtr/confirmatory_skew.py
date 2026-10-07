"""Extract and test the preregistered eight-template fixed-group expansion."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from pathlib import Path

from extract_congestion import channels, percentile

HERE = Path(__file__).resolve().parent
ROOT = HERE / 'synthetic_skew_v1'
RUNS = HERE / 'runs/skew_v1'
SEEDS = (503, 601, 701, 809, 907, 1009, 1103, 1201)
WIDTH = 62


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def region_metrics(archive: Path, design: str) -> tuple[float, float, float]:
    data = (channels(archive / 'chanx_occupancy.txt', 'x')
            + channels(archive / 'chany_occupancy.txt', 'y'))
    route_head = (archive / f'{design}.route').read_text(errors='replace')[:1000]
    match = re.search(r'Array size:\s*(\d+)\s*x\s*(\d+)', route_head)
    if not match:
        raise ValueError(f'array size missing: {archive}')
    nx, ny = map(int, match.groups())
    accum = [[0, 0] for _ in range(64)]
    for _, x, y, occ, cap, _ in data:
        region = min(7, 8 * y // ny) * 8 + min(7, 8 * x // nx)
        accum[region][0] += occ
        accum[region][1] += cap
    values = [occ / cap for occ, cap in accum if cap]
    channel_values = [occ / cap for _, _, _, occ, cap, _ in data]
    return percentile(values, 0.99), max(values), percentile(channel_values, 0.99)


def exact_two_sided_sign_p(positive: int, negative: int) -> float:
    n = positive + negative
    if not n:
        return 1.0
    observed = min(positive, negative)
    probability = sum(math.comb(n, k) for k in range(observed + 1)) / (2 ** n)
    return min(1.0, 2 * probability)


def main() -> None:
    specs = json.loads((ROOT / 'generation_manifest.json').read_text(encoding='utf-8'))
    specs = {(int(row['generator_seed']), row['skew_level']): row for row in specs
             if int(row['generator_seed']) in SEEDS}
    feature_rows = list(csv.DictReader((ROOT / 'fixed_group_features.csv').open(newline='', encoding='utf-8')))
    features = {(int(row['generator_seed']), row['skew_level']): row for row in feature_rows
                if int(row['generator_seed']) in SEEDS}
    labels = []
    for generator_seed in SEEDS:
        for level in ('uniform', 'medium', 'hotspot'):
            spec = specs[generator_seed, level]
            design = spec['design']
            for placement_seed in (1, 2):
                full = RUNS / design / f'full-seed{placement_seed}-w300'
                archive = RUNS / design / f'route-seed{placement_seed}-w{WIDTH}'
                meta = json.loads((archive / 'manifest.json').read_text(encoding='utf-8'))
                full_meta = json.loads((full / 'manifest.json').read_text(encoding='utf-8'))
                expected_place = hashlib.sha256((full / f'{design}.place').read_bytes()).hexdigest()
                log = (archive / 'vpr.out').read_text(errors='replace')
                if (meta['source_sha256'] != spec['blif_sha256'] or
                        meta['placement_sha256'] != expected_place or
                        meta['architecture_sha256'] != full_meta['architecture_sha256']):
                    raise ValueError(f'provenance mismatch: {archive}')
                outcome = ('success' if meta['vpr_status'] == 'success' and 'VPR succeeded' in log else
                           'unroutable' if meta['vpr_status'] == 'unroutable' and
                           'Circuit is unroutable with a channel width factor' in log else 'other_failure')
                if outcome == 'other_failure':
                    raise ValueError(f'non-routing failure: {archive}')
                region_p99 = region_max = channel_p99 = ''
                if outcome == 'success':
                    region_p99, region_max, channel_p99 = region_metrics(archive, design)
                labels.append({'generator_seed': generator_seed, 'skew_level': level,
                               'placement_seed': placement_seed, 'channel_width': WIDTH,
                               'outcome': outcome, 'region_p99': round(region_p99, 9) if region_p99 != '' else '',
                               'region_max': round(region_max, 9) if region_max != '' else '',
                               'channel_p99': round(channel_p99, 9) if channel_p99 != '' else '',
                               'archive': str(archive)})
    write_csv(ROOT / 'confirmatory_route_labels.csv', labels)
    lookup = {(r['generator_seed'], r['skew_level'], r['placement_seed']): r for r in labels}
    pairs = []
    seed_means = []
    for generator_seed in SEEDS:
        deltas = []
        for placement_seed in (1, 2):
            uniform = lookup[generator_seed, 'uniform', placement_seed]
            hotspot = lookup[generator_seed, 'hotspot', placement_seed]
            both = uniform['outcome'] == hotspot['outcome'] == 'success'
            delta = (float(hotspot['region_p99']) - float(uniform['region_p99'])) if both else None
            if delta is not None:
                deltas.append(delta)
            pairs.append({'generator_seed': generator_seed, 'placement_seed': placement_seed,
                          'uniform_outcome': uniform['outcome'], 'hotspot_outcome': hotspot['outcome'],
                          'region_p99_delta': round(delta, 9) if delta is not None else '',
                          'uniform_fixed_k8_mean': features[generator_seed, 'uniform']['k8_mean_cut_pressure'],
                          'hotspot_fixed_k8_mean': features[generator_seed, 'hotspot']['k8_mean_cut_pressure'],
                          'uniform_fixed_k8_p95_to_median': features[generator_seed, 'uniform']['k8_p95_to_median'],
                          'hotspot_fixed_k8_p95_to_median': features[generator_seed, 'hotspot']['k8_p95_to_median']})
        seed_means.append({'generator_seed': generator_seed, 'valid_placement_pairs': len(deltas),
                           'mean_region_p99_delta': round(sum(deltas) / len(deltas), 9) if deltas else ''})
    write_csv(ROOT / 'confirmatory_pairs.csv', pairs)
    write_csv(ROOT / 'confirmatory_seed_means.csv', seed_means)
    values = [float(row['mean_region_p99_delta']) for row in seed_means if row['mean_region_p99_delta'] != '']
    positive = sum(value > 0 for value in values)
    negative = sum(value < 0 for value in values)
    rng = random.Random(20260921)
    bootstrap = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(10000))
    mean_ranges = []
    tail_deltas = []
    for seed in SEEDS:
        means = [float(features[seed, level]['k8_mean_cut_pressure']) for level in ('uniform', 'medium', 'hotspot')]
        mean_ranges.append((max(means) - min(means)) / (sum(means) / 3))
        tail_deltas.append(float(features[seed, 'hotspot']['k8_p95_to_median']) -
                           float(features[seed, 'uniform']['k8_p95_to_median']))
    report = {'generator_seeds': len(SEEDS), 'placement_pairs': len(pairs),
              'all_route_success': all(row['outcome'] == 'success' for row in labels),
              'max_fixed_group_mean_relative_range': max(mean_ranges),
              'mean_hotspot_minus_uniform_tail_ratio': sum(tail_deltas) / len(tail_deltas),
              'valid_generator_seed_means': len(values),
              'positive_seed_means': positive, 'negative_seed_means': negative,
              'mean_region_p99_delta': sum(values) / len(values),
              'bootstrap_95_interval_over_generator_seeds': [bootstrap[250], bootstrap[9749]],
              'exact_two_sided_sign_test_p': exact_two_sided_sign_p(positive, negative),
              'guard': 'Eight generator templates are the independent units; two placement seeds are averaged. '
                       'This validates a fixed-group synthetic mechanism, not automatic partition discovery or '
                       'natural-design generalization.'}
    (ROOT / 'confirmatory_result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
