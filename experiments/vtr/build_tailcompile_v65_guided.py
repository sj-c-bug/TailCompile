#!/usr/bin/env python3
"""Build frozen, hotspot-guided 2D fplace candidates without VPR labels.

This CPU-only search revises candidate *generation*, not the trained GNN.
It starts from the existing GNN placement and uses the v4 directional RRG
capacity proxy with a strict cluster-move budget. VPR remains the judge.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from prepare_tailcompile_v61 import load
from run_tailcompile_v62 import context, reference_placement
from tailcompile_v3 import lower_fplace
from tailcompile_v4_risk import digest, objective, scales_for, search


CASES = (('conv_layer', 120), ('lenet', 180))
SCHEMA = 'tailcompile-v65-guided-generation-1'


def build(bundle_path, out, budget, restarts, moved_fraction,
          max_wire_ratio, max_timing_ratio):
    bundle_path, out = Path(bundle_path).resolve(), Path(out).resolve()
    signature = dict(schema=SCHEMA, bundle_sha256=digest(bundle_path), budget=budget,
        restarts=restarts, moved_fraction=moved_fraction, max_wire_ratio=max_wire_ratio,
        max_timing_ratio=max_timing_ratio, cases=[list(key) for key in CASES])
    report_path = out / 'selection_report.json'
    if report_path.is_file():
        report = load(report_path)
        if report['signature'] != signature:
            raise ValueError('generation inputs changed; use a new output directory')
        for row in report['selections']:
            if digest(row['selected_fplace']) != row['selected_sha256']:
                raise ValueError('selected placement changed after generation')
        print(json.dumps(dict(stage='reuse_selection', report=str(report_path))), flush=True)
        return report
    if out.exists() and any(out.iterdir()):
        raise ValueError('new generation output directory must be empty')
    bundle = torch.load(bundle_path, map_location='cpu', weights_only=True)
    if not all(key in bundle['contexts'] for key in CASES):
        raise ValueError('v6.1 bundle lacks one of the frozen boundary conditions')
    start = time.monotonic()
    pending = []
    for key in CASES:
        design, width = key
        ctx, evaluator, _ = context(bundle, key)
        baseline, _, initial, xy = reference_placement(ctx)
        initial_state = evaluator.state(initial)
        if initial_state is None:
            raise ValueError(f'baseline assignment is proxy-infeasible: {key}')
        scales = scales_for(initial_state)
        initial_objective = objective(initial_state, scales, 'tail')
        best = None
        attempts = []
        for restart in range(restarts):
            candidate, trace = search(evaluator, initial, budget=budget,
                seed=65000 + restart, mode='tail', scales=scales, policy='hotspot',
                max_moved_fraction=moved_fraction)
            state = evaluator.state(candidate)
            moved = sum(a != b for a, b in zip(candidate, initial))
            score = objective(state, scales, 'tail')
            wire_ratio = state['wire'] / initial_state['wire'] if initial_state['wire'] > 0 else 1.0
            timing_ratio = state['timing'] / initial_state['timing'] if initial_state['timing'] > 0 else 1.0
            eligible = (moved > 0 and score < initial_objective - 1e-8 and
                        wire_ratio <= max_wire_ratio and timing_ratio <= max_timing_ratio)
            audit = dict(restart=restart, seed=65000 + restart, moved_clusters=moved,
                score=score, initial_score=initial_objective, wire_ratio=wire_ratio,
                timing_ratio=timing_ratio, eligible=eligible,
                search_seconds=trace['seconds'], evaluations=trace['evaluations'],
                accepted=trace['accepted'])
            attempts.append(audit)
            print(json.dumps(dict(stage='guided_search', design=design, width=width, **audit)), flush=True)
            if eligible and (best is None or score < best[0]):
                best = score, candidate, audit
        if best is None:
            raise ValueError(f'no proxy-improving, QoR-guarded placement: {key}; '
                             'do not launch a VPR sweep on a no-op')
        pending.append((key, ctx, baseline, best, xy, attempts))
    selections = []
    for key, ctx, baseline, best, xy, attempts in pending:
        design, width = key
        folder = out / design / f'w{width}'
        folder.mkdir(parents=True, exist_ok=False)
        candidate_path = folder / 'guided_tail.fplace'
        lowering = lower_fplace(ctx['logic'], ctx['device'], best[1], xy, candidate_path)
        # The frozen baseline and generated candidate must differ as files and region assignments.
        if digest(candidate_path) == digest(baseline):
            raise ValueError(f'candidate lowering did not change the fplace: {key}')
        (folder / 'lowering.json').write_text(json.dumps(lowering, indent=2) + '\n', encoding='utf-8')
        source = Path(ctx['inputs']).parent.parent
        config = load(source / 'config.json')
        record = next(row for row in config['designs'] if row['design'] == design)
        selections.append(dict(design=design, width=width, selected='guided_tail',
            inputs=str(Path(ctx['inputs']).resolve()), baseline=str(baseline.resolve()),
            baseline_sha256=digest(baseline), selected_fplace=str(candidate_path),
            selected_sha256=digest(candidate_path), device_width=int(record['device_width']),
            full_legalizer=config['ap_full_legalizer'], selected_audit=best[2],
            all_search_attempts=attempts))
    report = dict(signature=signature, generation_seconds=time.monotonic() - start,
                  selections=selections)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(dict(stage='guided_generation_complete', report=str(report_path),
                          seconds=report['generation_seconds'])), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--budget', type=int, default=512)
    parser.add_argument('--restarts', type=int, default=4)
    parser.add_argument('--moved-fraction', type=float, default=0.05)
    parser.add_argument('--max-wire-ratio', type=float, default=1.03)
    parser.add_argument('--max-timing-ratio', type=float, default=1.03)
    args = parser.parse_args()
    if args.budget < 1 or args.restarts < 1 or not 0 < args.moved_fraction <= 0.1:
        parser.error('budget/restarts must be positive; moved fraction must be in (0, 0.1]')
    if not 1 <= args.max_wire_ratio <= 1.1 or not 1 <= args.max_timing_ratio <= 1.1:
        parser.error('proxy QoR ratio guards must be in [1, 1.1]')
    build(args.bundle, args.out, args.budget, args.restarts,
          args.moved_fraction, args.max_wire_ratio, args.max_timing_ratio)


if __name__ == '__main__':
    main()
