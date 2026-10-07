"""Extract exact channel-utilization and 8x8 regional labels from VPR outputs.

VPR writes occupancy and capacity for each directed channel coordinate in
chanx_occupancy.txt / chany_occupancy.txt. Failed runs remain in outcomes but
have no congestion score. No placement or routing data enter netlist features.
"""

from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE / 'runs'
OUT = HERE / 'dataset' / 'expanded_v1'
COHORTS = ('pilot_v1', 'expanded_v1')


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    at = (len(values) - 1) * q
    lo = int(at)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (at - lo)


def channels(path: Path, direction: str) -> list[tuple[int, int, int, int, int, str]]:
    result = []
    with path.open(encoding='utf-8') as source:
        next(source)
        for line in source:
            parts = line.split()
            if not parts:
                continue
            if len(parts) != 6:
                raise ValueError(f'unexpected VPR occupancy row: {path}: {line!r}')
            layer, x, y, occupancy, percent, capacity = parts
            layer, x, y = int(layer), int(x), int(y)
            occupancy, capacity = int(occupancy), int(capacity)
            if capacity < 0 or occupancy < 0:
                raise ValueError(f'negative channel data: {path}: {line!r}')
            if capacity:
                if abs(100 * occupancy / capacity - float(percent)) > 0.002:
                    raise ValueError(f'VPR percentage mismatch: {path}: {line!r}')
                result.append((layer, x, y, occupancy, capacity, direction))
            elif occupancy:
                raise ValueError(f'occupied zero-capacity channel: {path}: {line!r}')
    return result


def output_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as stream:
        if not rows:
            return
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    runs = []
    regions = []
    for cohort in COHORTS:
        for manifest in sorted((RUNS / cohort).glob('*/route-seed*-w*/manifest.json')):
            archive = manifest.parent
            meta = json.loads(manifest.read_text(encoding='utf-8'))
            log = (archive / 'vpr.out').read_text(errors='replace')
            status = ('success' if meta['vpr_status'] == 'success' and 'VPR succeeded' in log
                      else 'unroutable' if 'Circuit is unroutable with a channel width factor' in log
                      else 'other_failure')
            width = meta['route_chan_width']
            if width == -1:
                match = re.search(r'Best routing used a channel width factor of\s*(\d+)', log)
                width = int(match.group(1)) if match else ''
            row = {'cohort': cohort, 'design': meta['design'], 'seed': meta['seed'],
                   'run': archive.name, 'requested_width': meta['route_chan_width'],
                   'effective_width': width, 'outcome': status,
                   'architecture_sha256': meta['architecture_sha256'],
                   'mapped_blif_sha256': meta['mapped_blif_sha256'],
                   'placement_sha256': meta['placement_sha256'],
                   'active_channels': '', 'channel_p95': '', 'channel_p99': '',
                   'channel_max': '', 'region_p95': '', 'region_p99': '',
                   'region_max': '', 'region_cvar95': '', 'vpr_logged_max': '',
                   'vpr_max_abs_difference': '', 'archive': str(archive)}
            if status == 'success':
                data = (channels(archive / 'chanx_occupancy.txt', 'x')
                        + channels(archive / 'chany_occupancy.txt', 'y'))
                if not data:
                    raise ValueError(f'no active channel records: {archive}')
                ratios = [occ / cap for _, _, _, occ, cap, _ in data]
                row.update(active_channels=len(data),
                           channel_p95=round(percentile(ratios, 0.95), 9),
                           channel_p99=round(percentile(ratios, 0.99), 9),
                           channel_max=round(max(ratios), 9))
                size = re.search(r'Array size:\s*(\d+)\s*x\s*(\d+)',
                                 (archive / f'{meta["design"]}.route').read_text(errors='replace')[:1000])
                if not size:
                    raise ValueError(f'array size not found: {archive}')
                nx, ny = int(size.group(1)), int(size.group(2))
                accum = [[0, 0, 0] for _ in range(64)]
                for _, x, y, occ, cap, _ in data:
                    gx = min(7, max(0, 8 * x // nx))
                    gy = min(7, max(0, 8 * y // ny))
                    record = accum[gy * 8 + gx]
                    record[0] += occ
                    record[1] += cap
                    record[2] += 1
                regional_values = []
                for region_id, (occ, cap, count) in enumerate(accum):
                    utilization = occ / cap if cap else None
                    regions.append({'cohort': cohort, 'design': meta['design'], 'seed': meta['seed'],
                                    'run': archive.name, 'region_x': region_id % 8,
                                    'region_y': region_id // 8, 'occupancy': occ,
                                    'capacity': cap, 'active_channels': count,
                                    'utilization': round(utilization, 9) if utilization is not None else ''})
                    if utilization is not None:
                        regional_values.append(utilization)
                upper = max(1, math.ceil(0.05 * len(regional_values)))
                row.update(region_p95=round(percentile(regional_values, 0.95), 9),
                           region_p99=round(percentile(regional_values, 0.99), 9),
                           region_max=round(max(regional_values), 9),
                           region_cvar95=round(sum(sorted(regional_values, reverse=True)[:upper]) / upper, 9))
                match = re.search(r'Maximum routing channel utilization:\s*([0-9.]+)', log)
                if match:
                    logged = float(match.group(1))
                    diff = abs(max(ratios) - logged)
                    row['vpr_logged_max'] = logged
                    row['vpr_max_abs_difference'] = round(diff, 9)
                    if diff > 0.011:
                        raise ValueError(f'occupancy max differs from VPR log: {archive}: {diff}')
            runs.append(row)
    output_csv(OUT / 'routing_outcomes.csv', runs)
    output_csv(OUT / 'region_congestion.csv', regions)
    print(f'parsed {len(runs)} route runs, {sum(r["outcome"] == "success" for r in runs)} successes, '
          f'{len(regions)} region rows')


if __name__ == '__main__':
    main()
