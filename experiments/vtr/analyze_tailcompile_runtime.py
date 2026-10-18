#!/usr/bin/env python3
"""Paired VPR runtime/iteration audit of a completed v6.4-style sweep.

Raw VPR logs are optional for a wall-time-only audit, but --require-logs makes
their absence an error. Never interpret a quick routing failure as a speedup.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

from run_tailcompile_v64_wmin import read, tasks, write


PHASE_PATTERNS = {
    'placement_seconds': r'# Analytical Placement took ([\d.]+) seconds',
    'routing_seconds': r'# Routing took ([\d.]+) seconds',
    'vpr_seconds': r'The entire flow of VPR took ([\d.]+) seconds',
    'placement_temperatures': r'Placement number of temperatures:\s*(\d+)',
    'placement_swaps': r'Placement total # of swap attempts:\s*(\d+)',
}


def parse_vpr_log(value):
    """Extract actual work counters, not an inference from elapsed time."""
    result = {}
    for key, pattern in PHASE_PATTERNS.items():
        matches = re.findall(pattern, value)
        result[key] = (int(matches[-1]) if key.startswith('placement_') and
                       key != 'placement_seconds' else float(matches[-1])) if matches else None
    successes = re.findall(r'Successfully routed after (\d+) routing iterations', value)
    if successes:
        result['routing_iterations'] = int(successes[-1])
    elif 'Routing failed.' in value:
        # Read only the router's iteration table, not analytical placement rows.
        sections = re.findall(r'^Iter\s+Time\s+pres\b(.*?)^Routing failed\.',
                              value, flags=re.MULTILINE | re.DOTALL)
        rows = re.findall(r'^\s*(\d+)\s+[\d.]+\s+[\d.]+\s+\d+\s+\d+\b',
                          sections[-1], flags=re.MULTILINE) if sections else []
        result['routing_iterations'] = max(map(int, rows)) if rows else None
    else:
        result['routing_iterations'] = None
    return result


def ratio_summary(pairs, key):
    values = [(a[key], b[key]) for a, b in pairs
              if a.get(key) is not None and b.get(key) is not None]
    if not values:
        return dict(pairs=0, geomean_ratio=None, median_delta=None,
                    faster=0, slower=0, equal=0)
    if any(a <= 0 or b <= 0 for a, b in values):
        raise ValueError(f'nonpositive runtime/iteration value: {key}')
    return dict(pairs=len(values),
        geomean_ratio=math.exp(statistics.mean(math.log(a / b) for a, b in values)),
        median_delta=statistics.median(a - b for a, b in values),
        faster=sum(a < b for a, b in values), slower=sum(a > b for a, b in values),
        equal=sum(a == b for a, b in values))


def analyze(root, require_logs=False):
    root = Path(root)
    plan = read(root / 'plan.json')
    index = {}
    logs = 0
    for task in tasks(plan, root):
        trial = task['directory'] / 'trial.json'
        manifest = task['directory'] / 'manifest.json'
        if not trial.is_file() or not manifest.is_file():
            raise ValueError(f'incomplete sweep; missing trial or manifest: {task["directory"]}')
        row = read(trial)
        raw_manifest = manifest.read_bytes()
        # Git on Windows may turn a synced JSON's LF into CRLF; accept only this
        # byte-for-byte reversible conversion, never arbitrary content changes.
        hashes = {hashlib.sha256(raw_manifest).hexdigest(),
                  hashlib.sha256(raw_manifest.replace(b'\r\n', b'\n')).hexdigest()}
        if row['manifest_sha256'] not in hashes:
            raise ValueError(f'manifest differs from trial record: {manifest}')
        if (row['design'], row['anchor_width'], row['width'], row['method'], row['seed']) != (
                task['case']['design'], task['case']['anchor_width'], task['width'],
                task['method'], task['seed']):
            raise ValueError(f'trial identity mismatch: {trial}')
        if row['status'] not in ('success', 'unroutable'):
            raise ValueError(f'non-routing failure: {trial}')
        log = task['directory'] / 'vpr.out'
        record = dict(row)
        if log.is_file():
            record.update(parse_vpr_log(log.read_text(encoding='utf-8', errors='replace')))
            logs += 1
        elif require_logs:
            raise FileNotFoundError(f'raw VPR log unavailable: {log}')
        key = row['design'], row['anchor_width'], row['width'], row['seed'], row['method']
        index[key] = record
    if len(index) != plan['task_count']:
        raise ValueError('duplicate trial identity')
    conditions = []
    for case in plan['cases']:
        design, anchor = case['design'], case['anchor_width']
        candidate_method = plan.get('candidate_method', 'v63')
        comparisons = {}
        for baseline in ('default', 'gnn_base'):
            grouped = defaultdict(list)
            by_width = defaultdict(lambda: defaultdict(list))
            for width in case['widths']:
                for seed in plan['seeds']:
                    a = index[design, anchor, width, seed, candidate_method]
                    b = index[design, anchor, width, seed, baseline]
                    outcome = ('joint_success' if a['status'] == b['status'] == 'success' else
                        'both_unroutable' if a['status'] == b['status'] == 'unroutable' else
                        'discordant_success')
                    grouped[outcome].append((a, b))
                    by_width[width][outcome].append((a, b))
            keys = ('wall_seconds', 'placement_seconds', 'routing_seconds',
                    'vpr_seconds', 'placement_temperatures', 'placement_swaps',
                    'routing_iterations')
            def summarize(group):
                return dict(count=len(group), metrics={key: ratio_summary(group, key) for key in keys})
            comparisons[baseline] = dict(
                joint_success=summarize(grouped['joint_success']),
                both_unroutable=summarize(grouped['both_unroutable']),
                discordant_success_count=len(grouped['discordant_success']),
                by_width={str(width): {status: summarize(group) for status, group in statuses.items()}
                          for width, statuses in by_width.items()})
        conditions.append(dict(design=design, anchor_width=anchor,
                               selected=case['selected'], candidate_method=candidate_method,
                               comparisons=comparisons))
    selection_path = Path(plan['selection_report'])
    selection = read(selection_path) if selection_path.is_file() else {}
    return dict(schema='tailcompile-runtime-audit-1', task_count=plan['task_count'],
        raw_log_count=logs, log_coverage=logs / plan['task_count'],
        candidate_generation_seconds=selection.get('generation_seconds'),
        conditions=conditions,
        scope='VPR subprocess only: excludes BLIF/packing, fplace generation and data loading',
        caveats=['Wall time from parallel trials can reflect CPU/memory contention.',
                 'Compare jointly successful runs for speed; never reward fast failures.',
                 'Only routing_iterations from raw VPR logs establishes fewer routing iterations.',
                 'A route-iteration count may exist for unsuccessful routing; analyze separately.'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--require-logs', action='store_true')
    args = parser.parse_args()
    report = analyze(args.root, args.require_logs)
    output = args.out or args.root / 'runtime_report.json'
    write(output, report)
    print(f'Runtime report: {output}; raw logs {report["raw_log_count"]}/{report["task_count"]}')


if __name__ == '__main__':
    main()
