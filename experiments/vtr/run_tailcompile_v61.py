"""Generate capacity-preserving learned .fplace refinements and pair with VPR."""
import argparse
import json
import math
import random
from pathlib import Path
from tailcompile_v61 import (SCHEMA, HierarchicalMatcher, candidate_graph, static_graph,
                             swap_proposal, to_device)
import torch
from tailcompile_v4_risk import RiskEvaluator, region_at, save, digest
from tailcompile_v2 import contract_edges
from tailcompile_v3 import lower_fplace
from tailrisk_core import parse_blif
from prepare_tailcompile_v61 import load


def generate(args):
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    if bundle['schema'] != SCHEMA:
        raise ValueError('wrong bundle schema')
    fingerprint = digest(args.bundle)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    plans = []
    for (design, width), ctx in sorted(bundle['contexts'].items()):
        if args.designs and design not in args.designs:
            continue
        checkpoint = args.models / f'holdout_{design}.pt'
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        if (saved['heldout'] != design or design in saved['training_designs'] or
                saved['signature']['bundle_sha256'] != fingerprint):
            raise ValueError('model provenance or held-out separation mismatch')
        work = args.out / design / f'w{width}'
        signature = dict(checkpoint_sha256=digest(checkpoint), bundle_sha256=fingerprint,
            budget=args.budget, seed=args.seed, max_moved_fraction=args.max_moved_fraction,
            implementation=digest(Path(__file__)), schema=SCHEMA)
        if (work / 'selection.json').exists():
            old = load(work / 'selection.json')
            if old['signature'] != signature or digest(work / 'selected.fplace') != old['selected_sha256']:
                raise ValueError('generation inputs changed; use new output directory')
            plans.append(old)
            continue
        if work.exists() and any(work.iterdir()):
            raise ValueError('incomplete generation directory; preserve and choose new output')
        logic, physical = ctx['logic'], ctx['device']
        supply = load(Path(ctx['context_dir']) / 'supply.json')
        inputs = Path(ctx['inputs'])
        blif = inputs / f'{design}.pre-vpr.blif'
        primitives, _ = parse_blif(blif)
        mapping = {a: c['id'] for c in logic['clusters'] for a in c['atoms']}
        edges, _ = contract_edges(primitives, mapping)
        evaluator = RiskEvaluator(logic, physical, supply, edges)
        base = static_graph(logic, physical, blif, supply)
        model = HierarchicalMatcher(hidden=saved['signature']['hidden']).to(device)
        model.load_state_dict(saved['state'])
        model.eval()
        baseline = Path(ctx['folder']) / 'candidates/gnn_base.fplace'
        lower = load(baseline.with_name('gnn_base_lowering.json'))
        records = sorted(lower['clusters'], key=lambda c: c['cluster'])
        reference = [region_at(physical, c['x'], c['y']) for c in records]
        xy = []
        for c, rid in zip(records, reference):
            r = physical['regions'][rid]
            xy.append([2 * (c['x'] - r['center_x']) / max(r['x_high'] - r['x_low'], 1),
                       2 * (c['y'] - r['center_y']) / max(r['y_high'] - r['y_low'], 1)])

        def score(assignment):
            with torch.no_grad():
                output = model(to_device(candidate_graph(base, logic, physical, evaluator, assignment), device))
            # Keep normalized QoR for guarding in log space; no exp overflow.
            wire, timing = [float(output['qor'][i]) * saved['normalization'][i][1] +
                            saved['normalization'][i][0] for i in (2, 3)]
            return dict(score=float(output['score']), wire_log1p=wire, timing_log1p=timing,
                        route_probability=float(torch.sigmoid(output['route_logit'])))

        initial = score(reference)
        best, best_value, trace = list(reference), initial, []
        rng, seen, evaluated = random.Random(args.seed), {tuple(reference)}, 0
        for attempt in range(max(100, args.budget * 50)):
            if evaluated >= args.budget:
                break
            proposal = swap_proposal(best, logic, evaluator, rng, reference, args.max_moved_fraction)
            if proposal is None or tuple(proposal) in seen:
                continue
            seen.add(tuple(proposal))
            if evaluator.state(proposal) is None:
                continue
            value = score(proposal)
            evaluated += 1
            # Predicted guard only: actual CPD/wire still require VPR verification.
            guarded = all(value[k] <= initial[k] + math.log1p(0.05)
                          for k in ('wire_log1p', 'timing_log1p'))
            accepted = guarded and value['score'] > best_value['score'] + 1e-5
            trace.append(dict(evaluation=evaluated, accepted=accepted, guard_pass=guarded, **value))
            if accepted:
                best, best_value = proposal, value
            if evaluated % 16 == 0:
                print(json.dumps(dict(stage='search', design=design, width=width,
                                      evaluated=evaluated, best=best_value['score'])), flush=True)
        work.mkdir(parents=True, exist_ok=True)
        if best == reference:
            import shutil
            shutil.copy2(baseline, work / 'selected.fplace')
            lowered = lower
        else:
            lowered = lower_fplace(logic, physical, best, torch.tensor(xy), work / 'selected.fplace')
            if any(region_at(physical, c['x'], c['y']) != best[c['cluster']] for c in lowered['clusters']):
                raise ValueError('lowering changed assigned region')
        save(work / 'selected_lowering.json', lowered)
        save(work / 'selected_binding.json', dict(assignment=best))
        result = dict(signature=signature, design=design, width=width, inputs=str(inputs),
            baseline=str(baseline), baseline_sha256=digest(baseline), selected=str(work / 'selected.fplace'),
            selected_sha256=digest(work / 'selected.fplace'), evaluated=evaluated,
            moved_clusters=sum(a != b for a, b in zip(best, reference)), initial=initial, final=best_value,
            trace=trace, labels_used_for_selection=False)
        save(work / 'selection.json', result)
        plans.append(result)
    if not plans:
        raise ValueError('no matching designs')
    save(args.out / 'generation_report.json', dict(schema=SCHEMA, candidates=plans))


def vpr(args):
    from run_tailcompile_v4 import Backend
    from v5_vpr_retry import run_one
    from prepare_tailcompile_v3_training import success_metrics
    import os
    backend = Backend(os.environ.get('VTR_ROOT'))
    plan = load(args.out / 'generation_report.json')['candidates']
    if len(set(args.seeds)) != len(args.seeds) or not set(args.seeds).isdisjoint(range(11, 33)):
        raise ValueError('use fresh evaluation seeds, e.g. 41 42')
    rows = []
    for candidate in plan:
        design, width = candidate['design'], candidate['width']
        inputs = Path(candidate['inputs'])
        # Use exact device width and legalizer of the original experiment cohort.
        source = inputs.parent.parent
        config = load(source / 'config.json')
        record = next(r for r in config['designs'] if r['design'] == design)
        for method, fplace in (('default', None), ('gnn_base', Path(candidate['baseline'])),
                               ('v61', Path(candidate['selected']))):
            if fplace and digest(fplace) != candidate['baseline_sha256' if method == 'gnn_base' else 'selected_sha256']:
                raise ValueError('fplace changed after selection')
            for seed in args.seeds:
                directory = args.out / design / f'w{width}/runs/{method}_s{seed}'
                meta = run_one(backend, directory, inputs / f'{design}.pre-vpr.blif',
                    inputs / f'{design}.net', width, int(record['device_width']), seed, fplace,
                    timeout=args.timeout, full_legalizer=config['ap_full_legalizer'])
                row = dict(design=design, width=width, method=method, seed=seed, status=meta['vpr_status'])
                if row['status'] == 'success':
                    row.update(success_metrics(directory, design))
                save(directory / 'v61_metrics.json', row)
                rows.append(row)
                save(args.out / 'vpr_results.json', dict(complete=False, rows=rows))
    save(args.out / 'vpr_results.json', dict(complete=True, rows=rows, comparisons=compare(rows),
        interpretation='paired fresh-seed evaluation; success first, then jointly successful congestion/wire/CPD'))


def compare(rows):
    import statistics
    index = {(r['design'], r['width'], r['seed'], r['method']): r for r in rows}
    results = {}
    for baseline in ('default', 'gnn_base'):
        gain, ratios = [], {k: [] for k in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')}
        for key, row in index.items():
            if key[-1] != 'v61':
                continue
            ref = index[(*key[:3], baseline)]
            gain.append(int(row['status'] == 'success') - int(ref['status'] == 'success'))
            if row['status'] == ref['status'] == 'success':
                for metric in ratios:
                    if row[metric] > 0 and ref[metric] > 0:
                        ratios[metric].append(math.log(row[metric] / ref[metric]))
        results[baseline] = dict(pairs=len(gain), success_rate_gain=statistics.mean(gain),
            joint_success_geomean_ratios={k: math.exp(statistics.mean(v)) if v else None for k, v in ratios.items()},
            joint_success_metric_counts={k: len(v) for k, v in ratios.items()})
    return results


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--models', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--stage', choices=('generate', 'vpr'), default='generate')
    p.add_argument('--designs', nargs='+')
    p.add_argument('--budget', type=int, default=64)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--max-moved-fraction', type=float, default=0.10)
    p.add_argument('--device', default='auto')
    p.add_argument('--seeds', nargs='+', type=int, default=[41, 42])
    p.add_argument('--timeout', type=int, default=1800)
    a = p.parse_args()
    if not 0 < a.max_moved_fraction <= 1 or a.budget < 0:
        p.error('invalid search budget or movement bound')
    if a.stage == 'generate':
        if a.bundle is None or a.models is None:
            p.error('generate requires bundle and models')
        generate(a)
    else:
        vpr(a)
