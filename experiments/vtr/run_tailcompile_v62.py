#!/usr/bin/env python3
"""TailCompile v6.2 intervention-aligned candidate training and evaluation."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import statistics
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from prepare_tailcompile_v6 import actual_hotspot, median_hotspots
from prepare_tailcompile_v61 import load
from prepare_tailcompile_v3_training import success_metrics
from run_tailcompile_v4 import Backend
from run_tailcompile_v61 import recover_interrupted
from tailcompile_v2 import contract_edges
from tailcompile_v3 import lower_fplace
from tailcompile_v4_risk import RiskEvaluator, digest, region_at, save
from tailcompile_v61 import (HierarchicalMatcher, candidate_graph, static_graph,
                             preference, swap_proposal, to_device)
from tailrisk_core import parse_blif
from train_tailcompile_v6 import normalization
from train_tailcompile_v61 import group_loss, grouped
from v5_vpr_retry import run_one


SCHEMA = 'tailcompile-v62-intervention-aligned-1'
PILOT = (('conv_layer', 120), ('eltwise_layer', 120),
         ('lenet', 180), ('lstm', 140))
BUDGETS = (0.01, 0.02, 0.05)
REPLICATES = 2


def context(bundle, key):
    ctx = bundle['contexts'][key]
    logic, device = ctx['logic'], ctx['device']
    supply = load(Path(ctx['context_dir']) / 'supply.json')
    blif = Path(ctx['inputs']) / f'{key[0]}.pre-vpr.blif'
    primitives, _ = parse_blif(blif)
    mapping = {a: c['id'] for c in logic['clusters'] for a in c['atoms']}
    edges, _ = contract_edges(primitives, mapping)
    evaluator = RiskEvaluator(logic, device, supply, edges)
    return ctx, evaluator, static_graph(logic, device, blif, supply)


def reference_placement(ctx):
    baseline = Path(ctx['folder']) / 'candidates/gnn_base.fplace'
    lowering = load(baseline.with_name('gnn_base_lowering.json'))
    records = sorted(lowering['clusters'], key=lambda c: c['cluster'])
    if [c['cluster'] for c in records] != list(range(len(ctx['logic']['clusters']))):
        raise ValueError('baseline lowering cluster IDs mismatch')
    assignment, xy = [], []
    for record in records:
        rid = region_at(ctx['device'], record['x'], record['y'])
        assignment.append(rid)
        region = ctx['device']['regions'][rid]
        xy.append([2 * (record['x'] - region['center_x']) / max(region['x_high'] - region['x_low'], 1),
                   2 * (record['y'] - region['center_y']) / max(region['y_high'] - region['y_low'], 1)])
    return baseline, lowering, assignment, torch.tensor(xy, dtype=torch.float32)


def pool(args):
    bundle = torch.load(args.v61_bundle, map_location='cpu', weights_only=True)
    signature = dict(schema=SCHEMA, bundle_sha256=digest(args.v61_bundle), budgets=list(BUDGETS),
                     replicates=REPLICATES, seed=args.seed)
    report_path = args.out / 'pool_report.json'
    if report_path.exists():
        report = load(report_path)
        if report['signature'] != signature:
            raise ValueError('pool inputs changed; choose a new output directory')
        for row in report['candidates']:
            if digest(row['fplace']) != row['fplace_sha256']:
                raise ValueError('pooled candidate changed')
        print(json.dumps(dict(stage='reuse_pool', candidates=len(report['candidates']))), flush=True)
        return
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError('pool output must be empty')
    rows = []
    for key in sorted(bundle['contexts']):
        design, width = key
        ctx, evaluator, _ = context(bundle, key)
        baseline, lowering, reference, xy = reference_placement(ctx)
        work = args.out / design / f'w{width}'
        work.mkdir(parents=True, exist_ok=True)
        config = load(Path(ctx['inputs']).parent.parent / 'config.json')
        design_record = next(r for r in config['designs'] if r['design'] == design)

        def record_candidate(name, assignment, budget, replicate, attempts):
            path = work / f'{name}.fplace'
            if name == 'noop':
                shutil.copy2(baseline, path)
                lowered = lowering
            else:
                lowered = lower_fplace(ctx['logic'], ctx['device'], assignment, xy, path)
            save(work / f'{name}_binding.json', dict(assignment=assignment))
            save(work / f'{name}_lowering.json', lowered)
            moved = sum(a != b for a, b in zip(assignment, reference))
            rows.append(dict(design=design, width=width, candidate=name, budget=budget,
                replicate=replicate, moved_clusters=moved, attempts=attempts,
                inputs=ctx['inputs'], device_width=design_record['device_width'],
                full_legalizer=config['ap_full_legalizer'],
                fplace=str(path), fplace_sha256=digest(path),
                binding=str(work / f'{name}_binding.json')))

        record_candidate('noop', reference, 0., 0, 0)
        movable = sum(c['resource'] != 'io' for c in ctx['logic']['clusters'])
        seen = {tuple(reference)}
        design_seed = sum((i + 1) * ord(char) for i, char in enumerate(design))
        for budget in BUDGETS:
            for replicate in range(REPLICATES):
                limit = math.floor(movable * budget)
                if limit < 2:
                    raise ValueError(f'{design} has too few movable clusters for budget {budget}')
                target = limit if limit % 2 == 0 else limit - 1
                assignment = None
                attempts = 0
                for retry in range(20):
                    rng = random.Random(args.seed + design_seed * 100003 + int(width) * 1009 +
                                        replicate * 97 + int(budget * 10000) + retry * 7919)
                    trial, trial_attempts = list(reference), 0
                    while (sum(a != b for a, b in zip(trial, reference)) < target and
                           trial_attempts < 10000):
                        proposal = swap_proposal(trial, ctx['logic'], evaluator, rng, reference, budget)
                        trial_attempts += 1
                        if proposal is None:
                            continue
                        before = sum(a != b for a, b in zip(trial, reference))
                        after = sum(a != b for a, b in zip(proposal, reference))
                        if after > before:
                            trial = proposal
                    attempts += trial_attempts
                    if trial != reference and tuple(trial) not in seen:
                        assignment = trial
                        break
                if assignment is None:
                    raise RuntimeError(f'could not generate intervention for {design} w{width} budget={budget}')
                seen.add(tuple(assignment))
                name = f'swap{int(budget * 100):02d}_r{replicate}'
                record_candidate(name, assignment, budget, replicate, attempts)
        print(json.dumps(dict(stage='pool', design=design, width=width,
                              candidates=1 + len(BUDGETS) * REPLICATES)), flush=True)
    save(report_path, dict(signature=signature, candidates=rows))


def parallel_trials(tasks, jobs, timeout, out, result_name):
    backend = Backend(os.environ.get('VTR_ROOT'))

    def execute(task):
        directory = Path(task['directory'])
        recover_interrupted(directory, out)
        inputs = Path(task['inputs'])
        meta = run_one(backend, directory, inputs / f'{task["design"]}.pre-vpr.blif',
            inputs / f'{task["design"]}.net', task['width'], int(task['device_width']),
            task['seed'], Path(task['fplace']) if task.get('fplace') else None,
            timeout=timeout, full_legalizer=task['full_legalizer'])
        row = {k: task[k] for k in ('design', 'width', 'candidate', 'seed')}
        row.update(status=meta['vpr_status'], directory=str(directory))
        if row['status'] == 'success':
            row.update(success_metrics(directory, task['design']))
        save(directory / 'v62_metrics.json', row)
        return row

    rows = []
    print(json.dumps(dict(stage='vpr_schedule', jobs=jobs, tasks=len(tasks))), flush=True)
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(execute, task) for task in tasks]
        for future in as_completed(futures):
            rows.append(future.result())
            save(out / result_name, dict(complete=False, rows=sorted(rows,
                key=lambda r: (r['design'], r['width'], r['candidate'], r['seed']))))
    rows.sort(key=lambda r: (r['design'], r['width'], r['candidate'], r['seed']))
    return rows


def label(args):
    report = load(args.pool / 'pool_report.json')
    candidates = [r for r in report['candidates'] if (r['design'], r['width']) in PILOT]
    tasks = []
    for row in candidates:
        for seed in args.label_seeds:
            tasks.append({**row, 'seed': seed, 'directory': str(args.out / row['design'] /
                f'w{row["width"]}' / row['candidate'] / f's{seed}')})
    rows = parallel_trials(tasks, args.jobs, args.timeout, args.out, 'label_results.json')
    save(args.out / 'label_results.json', dict(complete=True, schema=SCHEMA,
        pool_sha256=digest(args.pool / 'pool_report.json'), seeds=args.label_seeds, rows=rows))


def dataset(args):
    v61 = torch.load(args.v61_bundle, map_location='cpu', weights_only=True)
    labels = load(args.labels / 'label_results.json')
    if not labels.get('complete'):
        raise ValueError('labels incomplete')
    pool_report = load(args.pool / 'pool_report.json')
    pool_index = {(r['design'], r['width'], r['candidate']): r for r in pool_report['candidates']}
    groups = defaultdict(list)
    for row in labels['rows']:
        groups[row['design'], row['width'], row['candidate']].append(row)
    examples = []
    cache = {}
    for key, trials in sorted(groups.items()):
        design, width, candidate = key
        ctx_key = design, width
        if ctx_key not in cache:
            cache[ctx_key] = context(v61, ctx_key)
        ctx, evaluator, base = cache[ctx_key]
        spec = pool_index[key]
        assignment = load(spec['binding'])['assignment']
        successes = [r for r in trials if r['status'] == 'success']
        hotspots = [actual_hotspot(Path(r['directory']), ctx['device']) for r in successes]
        target = median_hotspots(hotspots)
        examples.append(dict(design=design, width=width, method=candidate,
            route=len(successes) / len(trials),
            qor=[statistics.median(r[m] for r in successes) if successes else None
                 for m in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')],
            hotspot=[r['utilization'] for r in target] if target else None,
            graph=candidate_graph(base, ctx['logic'], ctx['device'], evaluator, assignment)))
    args.out.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=SCHEMA, examples=examples, contexts=v61['contexts']), args.out / 'bundle.pt')
    save(args.out / 'dataset_report.json', dict(schema=SCHEMA, examples=len(examples),
        designs=sorted({r['design'] for r in examples}), conditions=sorted([list(k) for k in groups]),
        bundle_sha256=digest(args.out / 'bundle.pt')))


def evaluate_interventions(model, rows, device):
    """Evaluate held-out swap candidates against the v6.2 no-op baseline."""
    records, selections, correct, pairs = [], [], 0., 0
    model.eval()
    with torch.no_grad():
        for group in grouped(rows):
            scores = [model(to_device(row['graph'], device))['score'].item() for row in group]
            chosen = max(range(len(group)), key=lambda i: (round(scores[i], 5), group[i]['method']))
            for i, row in enumerate(group):
                records.append(dict(design=row['design'], width=row['width'], method=row['method'],
                                    score=scores[i], route=row['route'], qor=row['qor']))
                for j in range(i + 1, len(group)):
                    sign = preference(row, group[j])
                    if sign:
                        difference = scores[i] - scores[j]
                        correct += 0.5 if abs(difference) <= 1e-5 else float(sign * difference > 0)
                        pairs += 1
            selected = group[chosen]
            baseline = next((row for row in group if row['method'] == 'noop'), None)
            if baseline is None:
                raise ValueError('v6.2 held-out group lacks noop candidate')
            selections.append(dict(design=selected['design'], width=selected['width'],
                chosen=selected['method'], chosen_route=selected['route'], baseline_route=baseline['route'],
                chosen_qor=selected['qor'], baseline_qor=baseline['qor']))
    return dict(pair_accuracy=correct / pairs if pairs else None, pair_count=pairs,
                selections=selections, records=records)


def train(args):
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    rows, targets = bundle['examples'], sorted({d for d, _ in bundle['contexts']})
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    common_signature = dict(schema=SCHEMA, bundle_sha256=digest(args.bundle), epochs=args.epochs,
                            hidden=args.hidden, members=args.members, seed=args.seed)
    for target in targets:
        train_rows = [r for r in rows if r['design'] != target]
        test_rows = [r for r in rows if r['design'] == target]
        if len({r['design'] for r in train_rows}) < 3:
            raise ValueError(f'not enough independent training designs for {target}')
        norms = normalization(train_rows)
        members = []
        for member in range(args.members):
            seed = args.seed + member
            checkpoint = args.out / f'target_{target}_member{member}.pt'
            torch.manual_seed(seed)
            model = HierarchicalMatcher(hidden=args.hidden).to(device)
            signature = dict(common_signature, target=target, member=member, member_seed=seed,
                             training_designs=sorted({r['design'] for r in train_rows}))
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location=device, weights_only=True)
                if saved.get('signature') != signature:
                    raise ValueError('checkpoint inputs changed; choose a new model directory')
                model.load_state_dict(saved['state']); trace = saved['trace']
            else:
                trace = []
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                rng, groups = random.Random(seed), grouped(train_rows)
                for epoch in range(args.epochs):
                    model.train(); rng.shuffle(groups); losses = []
                    for group in groups:
                        optimizer.zero_grad(set_to_none=True)
                        loss = group_loss(model, group, norms, device)
                        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step(); losses.append(float(loss.detach()))
                    trace.append(dict(epoch=epoch + 1, loss=statistics.mean(losses)))
                    print(json.dumps(dict(stage='train', target=target, member=member, **trace[-1])), flush=True)
                torch.save(dict(signature=signature, target=target,
                    training_designs=signature['training_designs'], state=model.state_dict(),
                    normalization=norms, hidden=args.hidden, seed=seed, trace=trace), checkpoint)
            members.append(dict(member=member, checkpoint_sha256=digest(checkpoint), trace=trace,
                heldout=evaluate_interventions(model, test_rows, device) if test_rows else None))
        reports.append(dict(target=target, training_designs=sorted({r['design'] for r in train_rows}),
                            members=members))
    save(args.out / 'report.json', dict(schema=SCHEMA, members=args.members, targets=reports,
        interpretation='ensemble trained on deployment-distribution swap interventions; target design excluded'))


def infer(model, graph, norm, device):
    with torch.no_grad():
        out = model(to_device(graph, device))
    qor = [float(out['qor'][i]) * norm[i][1] + norm[i][0] for i in range(4)]
    return dict(score=out['score'].item(), route=torch.sigmoid(out['route_logit']).item(), qor=qor)


def conservative_audit(name, moved, values, zero, margin, z, tolerance):
    score = [a['score'] - b['score'] for a, b in zip(values, zero)]
    route = [a['route'] - b['route'] for a, b in zip(values, zero)]
    wire = [a['qor'][2] - b['qor'][2] for a, b in zip(values, zero)]
    timing = [a['qor'][3] - b['qor'][3] for a, b in zip(values, zero)]
    mean, std = statistics.mean(score), statistics.pstdev(score)
    score_lcb = mean - z * std
    route_lcb = statistics.mean(route) - z * statistics.pstdev(route)
    wire_ucb = statistics.mean(wire) + z * statistics.pstdev(wire)
    timing_ucb = statistics.mean(timing) + z * statistics.pstdev(timing)
    ok = (score_lcb >= margin and route_lcb >= 0 and
          wire_ucb <= math.log1p(tolerance) and timing_ucb <= math.log1p(tolerance))
    return dict(candidate=name, eligible=ok, score_mean=mean, score_std=std,
        score_lcb=score_lcb, route_lcb=route_lcb, wire_log_ucb=wire_ucb,
        timing_log_ucb=timing_ucb, moved_clusters=moved)


def select(args):
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    pool_report = load(args.pool / 'pool_report.json')
    signature = dict(schema=SCHEMA, bundle_sha256=digest(args.bundle),
        pool_sha256=digest(args.pool / 'pool_report.json'), members=args.members,
        model_report_sha256=digest(args.models / 'report.json'), min_margin=args.min_margin,
        uncertainty_z=args.uncertainty_z, qor_tolerance=args.qor_tolerance)
    report_path = args.out / 'selection_report.json'
    if report_path.exists():
        old = load(report_path)
        if old.get('signature') != signature:
            raise ValueError('selection inputs changed; choose a new output directory')
        for row in old['selections']:
            if digest(row['selected_fplace']) != row['selected_sha256']:
                raise ValueError('selected fplace changed')
        print(json.dumps(dict(stage='reuse_selection', selections=len(old['selections']))), flush=True)
        return
    candidates = defaultdict(list)
    for row in pool_report['candidates']:
        candidates[row['design'], row['width']].append(row)
    device = torch.device(args.device)
    selections = []
    cache = {}
    for key, rows in sorted(candidates.items()):
        design, width = key
        ctx, evaluator, base = context(bundle, key)
        members = []
        for member in range(args.members):
            saved = torch.load(args.models / f'target_{design}_member{member}.pt',
                               map_location=device, weights_only=True)
            if (saved['target'] != design or design in saved['training_designs'] or
                    saved['signature']['bundle_sha256'] != digest(args.bundle)):
                raise ValueError('target-design leakage')
            model = HierarchicalMatcher(hidden=saved['hidden']).to(device)
            model.load_state_dict(saved['state']); model.eval()
            members.append((model, saved['normalization']))
        predictions = {}
        for row in rows:
            assignment = load(row['binding'])['assignment']
            graph = candidate_graph(base, ctx['logic'], ctx['device'], evaluator, assignment)
            predictions[row['candidate']] = [infer(model, graph, norm, device) for model, norm in members]
        noop = next(r for r in rows if r['candidate'] == 'noop')
        eligible = []
        audits = []
        for row in rows:
            if row['candidate'] == 'noop':
                continue
            values, zero = predictions[row['candidate']], predictions['noop']
            audit = conservative_audit(row['candidate'], row['moved_clusters'], values, zero,
                                       args.min_margin, args.uncertainty_z, args.qor_tolerance)
            audits.append(audit)
            if audit['eligible']:
                eligible.append((audit['score_lcb'], -row['moved_clusters'], row))
        chosen = max(eligible, default=(0, 0, noop))[-1]
        work = args.out / design / f'w{width}'
        work.mkdir(parents=True, exist_ok=True)
        shutil.copy2(chosen['fplace'], work / 'selected.fplace')
        result = dict(design=design, width=width, selected=chosen['candidate'],
            fallback_noop=chosen['candidate'] == 'noop', selected_sha256=digest(work / 'selected.fplace'),
            selected_fplace=str(work / 'selected.fplace'), baseline=noop['fplace'],
            baseline_sha256=noop['fplace_sha256'], inputs=chosen['inputs'],
            device_width=chosen['device_width'], full_legalizer=chosen['full_legalizer'], audits=audits)
        save(work / 'selection.json', result); selections.append(result)
        print(json.dumps(dict(stage='select', design=design, width=width,
                              selected=chosen['candidate'])), flush=True)
    save(args.out / 'selection_report.json', dict(schema=SCHEMA, signature=signature, selections=selections,
        policy=dict(min_margin=args.min_margin, uncertainty_z=args.uncertainty_z,
                    qor_tolerance=args.qor_tolerance)))


def compare(rows, contender='v62'):
    index = {(r['design'], r['width'], r['seed'], r['candidate']): r for r in rows}
    result = {}
    for baseline in ('default', 'gnn_base'):
        gains, logs = [], {m: [] for m in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')}
        for key, row in index.items():
            if key[-1] != contender:
                continue
            ref = index[(*key[:3], baseline)]
            gains.append(int(row['status'] == 'success') - int(ref['status'] == 'success'))
            if row['status'] == ref['status'] == 'success':
                for metric in logs:
                    logs[metric].append(math.log(row[metric] / ref[metric]))
        result[baseline] = dict(pairs=len(gains), net_success_gain=sum(gains),
            success_rate_gain=statistics.mean(gains),
            joint_success_geomean_ratios={m: math.exp(statistics.mean(v)) if v else None
                                           for m, v in logs.items()})
    return result


def evaluation_scope(report, designs, condition_specs):
    """Select whole designs or exact design:width conditions for evaluation."""
    if condition_specs:
        conditions = set()
        for spec in condition_specs:
            try:
                design, width = spec.rsplit(':', 1)
                conditions.add((design, int(width)))
            except (ValueError, TypeError):
                raise ValueError(f'invalid condition {spec!r}; expected DESIGN:WIDTH') from None
        selected = [row for row in report if (row['design'], row['width']) in conditions]
        found = {(row['design'], row['width']) for row in selected}
        if found != conditions:
            raise ValueError(f'evaluation conditions missing from selection report: {sorted(conditions - found)}')
        return selected, conditions
    selected_designs = None if designs == ['all'] else set(designs)
    selected = report if selected_designs is None else [row for row in report
                                                        if row['design'] in selected_designs]
    return selected, None


def evaluate_selected(args):
    report = load(args.selection / 'selection_report.json')['selections']
    report, conditions = evaluation_scope(report, args.designs, args.conditions)
    prior = []
    if args.prior_results:
        prior_report = load(args.prior_results)
        if not prior_report.get('complete'):
            raise ValueError('prior evaluation is incomplete')
        prior = prior_report['rows']
    prior_keys = {(r['design'], r['width'], r['candidate'], r['seed']) for r in prior}
    tasks = []
    for row in report:
        for method, fplace in (('default', None), ('gnn_base', row['baseline']),
                               ('v62', row['selected_fplace'])):
            for seed in args.eval_seeds:
                if (row['design'], row['width'], method, seed) in prior_keys:
                    continue
                if fplace:
                    expected = row['baseline_sha256' if method == 'gnn_base' else 'selected_sha256']
                    if digest(fplace) != expected:
                        raise ValueError('evaluation fplace changed after selection')
                tasks.append(dict(**{k: row[k] for k in ('design', 'width', 'inputs', 'device_width',
                                  'full_legalizer')}, candidate=method, fplace=fplace, seed=seed,
                    directory=str(args.out / row['design'] / f'w{row["width"]}' / method / f's{seed}')))
    new_rows = parallel_trials(tasks, args.jobs, args.timeout, args.out, 'vpr_results.partial.json') if tasks else []
    rows = sorted(prior + new_rows, key=lambda r: (r['design'], r['width'], r['candidate'], r['seed']))
    comparisons = compare(rows)
    pilot = conditions == set(PILOT)
    gate_detail = None
    gate = None
    if pilot:
        ratios = comparisons['default']['joint_success_geomean_ratios']
        bounded = all(ratios[m] is not None and ratios[m] <= 1.03
                      for m in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns'))
        congestion_gain = any(ratios[m] is not None and ratios[m] < 1.
                              for m in ('region_p99', 'channel_p99'))
        gate_detail = dict(no_success_loss_vs_default=comparisons['default']['net_success_gain'] >= 0,
            success_gain_vs_gnn_base=comparisons['gnn_base']['net_success_gain'] >= 1,
            all_joint_success_qor_within_3pct=bounded,
            congestion_metric_improves=congestion_gain)
        gate = all(gate_detail.values())
    save(args.out / 'vpr_results.json', dict(complete=True, schema=SCHEMA, rows=rows,
        comparisons=comparisons, pilot_gate_passed=gate, pilot_gate=gate_detail,
        seeds=args.eval_seeds))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('pool', 'label', 'dataset', 'train', 'select', 'evaluate'))
    p.add_argument('--v61-bundle', type=Path)
    p.add_argument('--bundle', type=Path)
    p.add_argument('--pool', type=Path)
    p.add_argument('--labels', type=Path)
    p.add_argument('--models', type=Path)
    p.add_argument('--selection', type=Path)
    p.add_argument('--prior-results', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--seed', type=int, default=62027)
    p.add_argument('--label-seeds', type=int, nargs='+', default=[51, 52])
    p.add_argument('--eval-seeds', type=int, nargs='+', default=[61, 62])
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--timeout', type=int, default=1800)
    p.add_argument('--epochs', type=int, default=80)
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--members', type=int, default=3)
    p.add_argument('--device', default='cpu')
    p.add_argument('--min-margin', type=float, default=0.01)
    p.add_argument('--uncertainty-z', type=float, default=1.0)
    p.add_argument('--qor-tolerance', type=float, default=0.01)
    p.add_argument('--designs', nargs='+', default=[d for d, _ in PILOT])
    p.add_argument('--conditions', nargs='+', help='exact DESIGN:WIDTH conditions to evaluate')
    a = p.parse_args()
    if a.jobs < 1 or a.members < 2:
        p.error('jobs must be positive and ensemble must have at least two members')
    if (len(set(a.label_seeds)) != len(a.label_seeds) or
            len(set(a.eval_seeds)) != len(a.eval_seeds) or
            set(a.label_seeds) & set(a.eval_seeds)):
        p.error('label and evaluation seeds must each be unique and mutually disjoint')
    required = {'pool': ('v61_bundle',), 'label': ('pool',),
                'dataset': ('v61_bundle', 'pool', 'labels'), 'train': ('bundle',),
                'select': ('bundle', 'pool', 'models'), 'evaluate': ('selection',)}[a.stage]
    if any(getattr(a, name) is None for name in required):
        p.error(f'{a.stage} missing required paths: {required}')
    {'pool': pool, 'label': label, 'dataset': dataset, 'train': train,
     'select': select, 'evaluate': evaluate_selected}[a.stage](a)


if __name__ == '__main__':
    main()
