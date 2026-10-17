#!/usr/bin/env python3
"""Expanded intervention training with an explicit Default placement option."""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import statistics
from collections import defaultdict
from pathlib import Path

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch
from torch import nn

import run_tailcompile_v62 as v62
from prepare_tailcompile_v6 import actual_hotspot, median_hotspots
from prepare_tailcompile_v61 import load
from tailcompile_v4_risk import VTR_COMMIT, digest, save
from tailcompile_v61 import HierarchicalMatcher, candidate_graph, to_device
from train_tailcompile_v6 import normalization
from train_tailcompile_v61 import group_loss, grouped


SCHEMA = 'tailcompile-v63-expanded-interventions-1'
REPLICATES = 4
LABEL_SEEDS = (51, 52, 53)
EVAL_SEEDS = (71, 72, 73)
PROBE = (('conv_layer', 120, 'swap02_r0'),
         ('conv_layer', 120, 'swap02_r1'),
         ('eltwise_layer', 120, 'swap05_r0'))


def conditions(rows):
    return sorted({(r['design'], int(r['width'])) for r in rows})


def expected_count(rows, seeds):
    return len(rows) * len(seeds)


def assert_complete(rows, candidates, seeds):
    expected = {(r['design'], int(r['width']), r['candidate'], int(seed))
                for r in candidates for seed in seeds}
    observed = {(r['design'], int(r['width']), r['candidate'], int(r['seed'])) for r in rows}
    if len(rows) != len(observed) or observed != expected:
        raise ValueError(f'trial coverage differs: expected {len(expected)}, observed {len(observed)}')


def pool(args):
    # Reuse the stable v6.2 lowerer and candidate generator with a new output cohort.
    v62.SCHEMA, v62.REPLICATES = SCHEMA, REPLICATES
    core = args.out / 'swaps'
    v62.pool(argparse.Namespace(v61_bundle=args.v61_bundle, out=core, seed=args.seed))
    core_report = load(core / 'pool_report.json')
    result_path = args.out / 'pool_report.json'
    signature = dict(schema=SCHEMA, core_sha256=digest(core / 'pool_report.json'),
                     default_mode='unbound_vpr', replicates=REPLICATES)
    if result_path.exists():
        old = load(result_path)
        if old['signature'] != signature:
            raise ValueError('expanded pool changed; choose a new output directory')
        return
    rows = list(core_report['candidates'])
    for key in conditions(rows):
        noop = next(r for r in rows if (r['design'], r['width'], r['candidate']) == (*key, 'noop'))
        rows.append(dict(noop, candidate='default', fplace=None, fplace_sha256=None,
                         budget=None, replicate=None, moved_clusters=0))
    save(result_path, dict(signature=signature, candidates=rows))
    print(json.dumps(dict(stage='pool_complete', candidates=len(rows), conditions=len(conditions(rows)))), flush=True)


def probe(args):
    old_pool = load(args.v62_pool)['candidates']
    prior = load(args.v62_pilot)
    if not prior.get('complete') or len(prior['rows']) != 24:
        raise ValueError('v6.2 pilot must have all 24 baseline trials')
    index = {(r['design'], r['width'], r['candidate']): r for r in old_pool}
    tasks = []
    for key in PROBE:
        item = index[key]
        for seed in (61, 62):
            tasks.append(dict(**{k: item[k] for k in ('design', 'width', 'inputs', 'device_width',
                               'full_legalizer', 'fplace')}, candidate=item['candidate'], seed=seed,
                directory=str(args.out / item['design'] / f'w{item["width"]}' /
                              item['candidate'] / f's{seed}')))
    args.out.mkdir(parents=True, exist_ok=True)
    new_rows = v62.parallel_trials(tasks, args.jobs, args.timeout, args.out, 'probe.partial.json')
    comparisons = []
    baseline = {(r['design'], r['width'], r['seed'], r['candidate']): r for r in prior['rows']}
    for row in new_rows:
        ref = baseline[row['design'], row['width'], row['seed'], 'gnn_base']
        comparisons.append(dict(design=row['design'], width=row['width'], candidate=row['candidate'],
            seed=row['seed'], candidate_status=row['status'], baseline_status=ref['status'],
            success_delta=int(row['status'] == 'success') - int(ref['status'] == 'success'),
            candidate_metrics={k: row.get(k) for k in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')},
            baseline_metrics={k: ref.get(k) for k in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')}))
    save(args.out / 'probe_report.json', dict(schema=SCHEMA, complete=True, rows=new_rows,
        comparisons=comparisons, net_success_gain=sum(r['success_delta'] for r in comparisons)))


def label(args):
    report = load(args.pool / 'pool_report.json')
    candidates = report['candidates']
    seeds = tuple(args.label_seeds)
    result_path = args.out / 'label_results.json'
    signature = dict(schema=SCHEMA, pool_sha256=digest(args.pool / 'pool_report.json'), seeds=list(seeds))
    if result_path.exists():
        old = load(result_path)
        if old.get('complete') and old.get('signature') == signature:
            assert_complete(old['rows'], candidates, seeds)
            return
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    reused = 0
    if args.v62_pool and args.v62_labels:
        old_labels = load(args.v62_labels)
        if not old_labels.get('complete'):
            raise ValueError('v6.2 labels are incomplete')
        old_pool = {(r['design'], r['width'], r['candidate']): r
                    for r in load(args.v62_pool)['candidates']}
        new_pool = {(r['design'], r['width'], r['candidate']): r for r in candidates}
        for row in old_labels['rows']:
            key = row['design'], row['width'], row['candidate']
            manifest = Path(row['directory'], 'manifest.json')
            if (row['seed'] in seeds and key in new_pool and key in old_pool and
                    new_pool[key]['fplace_sha256'] == old_pool[key]['fplace_sha256'] and
                    manifest.is_file() and load(manifest)['vtr_commit'] == VTR_COMMIT and
                    load(manifest)['signature']['input_fplace_sha256'] == new_pool[key]['fplace_sha256']):
                rows.append(row)
                reused += 1
    keys = {(r['design'], r['width'], r['candidate'], r['seed']) for r in rows}
    tasks = []
    for item in candidates:
        for seed in seeds:
            key = item['design'], item['width'], item['candidate'], seed
            if key in keys:
                continue
            tasks.append({**item, 'seed': seed, 'directory': str(args.out / item['design'] /
                f'w{item["width"]}' / item['candidate'] / f's{seed}')})
    new = v62.parallel_trials(tasks, args.jobs, args.timeout, args.out, 'label_results.partial.json') if tasks else []
    rows = sorted(rows + new, key=lambda r: (r['design'], r['width'], r['candidate'], r['seed']))
    assert_complete(rows, candidates, seeds)
    save(result_path, dict(signature=signature, complete=True, rows=rows,
                           reused_v62_trials=reused, new_trials=len(new)))


def dataset(args):
    source = load(args.labels / 'label_results.json')
    pool_report = load(args.pool / 'pool_report.json')
    if not source.get('complete'):
        raise ValueError('labels incomplete')
    assert_complete(source['rows'], pool_report['candidates'], source['signature']['seeds'])
    v61 = torch.load(args.v61_bundle, map_location='cpu', weights_only=True)
    index = {(r['design'], r['width'], r['candidate']): r for r in pool_report['candidates']}
    trials = defaultdict(list)
    for row in source['rows']:
        trials[row['design'], row['width'], row['candidate']].append(row)
    examples, cache = [], {}
    for key, values in sorted(trials.items()):
        design, width, method = key
        ctx_key = design, width
        if ctx_key not in cache:
            cache[ctx_key] = v62.context(v61, ctx_key)
        ctx, evaluator, base = cache[ctx_key]
        spec = index[key]
        assignment = load(spec['binding'])['assignment']
        successes = [r for r in values if r['status'] == 'success']
        hotspots = ([actual_hotspot(Path(r['directory']), ctx['device']) for r in successes]
                    if method != 'default' else [])
        target = median_hotspots(hotspots)
        graph = candidate_graph(base, ctx['logic'], ctx['device'], evaluator, assignment)
        graph['is_default'] = method == 'default'
        examples.append(dict(design=design, width=width, method=method,
            route=len(successes) / len(values),
            qor=[statistics.median(r[m] for r in successes) if successes else None
                 for m in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')],
            hotspot=None if method == 'default' else
                    [r['utilization'] for r in target] if target else None,
            graph=graph))
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / 'bundle.pt'
    if destination.exists():
        old = load(args.out / 'dataset_report.json')
        if (old['labels_sha256'] != digest(args.labels / 'label_results.json') or
                old['bundle_sha256'] != digest(destination)):
            raise ValueError('dataset labels changed; choose a new output directory')
        return
    torch.save(dict(schema=SCHEMA, examples=examples, contexts=v61['contexts']), destination)
    save(args.out / 'dataset_report.json', dict(schema=SCHEMA, examples=len(examples),
        designs=sorted({r['design'] for r in examples}), conditions=len(conditions(examples)),
        label_seeds=source['signature']['seeds'], labels_sha256=digest(args.labels / 'label_results.json'),
        bundle_sha256=digest(destination)))


class PlacementChoiceMatcher(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.core = HierarchicalMatcher(hidden=hidden)
        self.default_delta = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 6))

    def forward(self, graph):
        output = self.core(graph)
        if graph.get('is_default', False):
            delta = self.default_delta(output['embedding'])
            output['route_logit'] = output['route_logit'] + delta[0]
            output['qor'] = output['qor'] + delta[1:5]
            output['score'] = output['score'] + delta[5]
        return output


def infer(model, graph, norms, device):
    with torch.no_grad():
        out = model(to_device(graph, device))
    return dict(score=float(out['score']), route=float(torch.sigmoid(out['route_logit'])),
        qor=[float(x) * std + mean for x, (mean, std) in zip(out['qor'], norms)])


def train(args):
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    if bundle['schema'] != SCHEMA:
        raise ValueError('dataset schema mismatch')
    rows = bundle['examples']
    designs = sorted({r['design'] for r in rows})
    if len(designs) < 6 or len(rows) < 150:
        raise ValueError('expanded training needs six designs and at least 150 candidate examples')
    device = torch.device(args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    report = []
    common = dict(schema=SCHEMA, bundle_sha256=digest(args.bundle), epochs=args.epochs,
                  hidden=args.hidden, members=args.members, seed=args.seed)
    for target in designs:
        train_rows = [r for r in rows if r['design'] != target]
        test_rows = [r for r in rows if r['design'] == target]
        norms = normalization(train_rows)
        members = []
        for member in range(args.members):
            seed = args.seed + member
            torch.manual_seed(seed)
            model = PlacementChoiceMatcher(args.hidden).to(device)
            checkpoint = args.out / f'target_{target}_member{member}.pt'
            signature = dict(common, target=target, member=member, training_designs=sorted(
                             {r['design'] for r in train_rows}))
            if checkpoint.exists():
                saved = torch.load(checkpoint, map_location=device, weights_only=True)
                if saved['signature'] != signature:
                    raise ValueError('checkpoint inputs changed; choose a new output directory')
                model.load_state_dict(saved['state'])
                trace = saved['trace']
            else:
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
                rng, groups, trace = random.Random(seed), grouped(train_rows), []
                for epoch in range(args.epochs):
                    model.train()
                    rng.shuffle(groups)
                    losses = []
                    for group in groups:
                        optimizer.zero_grad(set_to_none=True)
                        loss = group_loss(model, group, norms, device)
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                        optimizer.step()
                        losses.append(float(loss.detach()))
                    trace.append(dict(epoch=epoch + 1, loss=statistics.mean(losses)))
                    print(json.dumps(dict(stage='train', target=target, member=member, **trace[-1])), flush=True)
                torch.save(dict(signature=signature, target=target, training_designs=signature['training_designs'],
                                state=model.state_dict(), normalization=norms, trace=trace), checkpoint)
            members.append(dict(member=member, checkpoint_sha256=digest(checkpoint),
                heldout=v62.evaluate_interventions(model, test_rows, device), trace=trace))
        report.append(dict(target=target, training_designs=sorted({r['design'] for r in train_rows}),
                           members=members))
    save(args.out / 'report.json', dict(schema=SCHEMA, targets=report, members=args.members,
        evaluation='leave-one-design-out; target design excluded from model fitting'))


def select(args):
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    pool_report = load(args.pool / 'pool_report.json')
    report_path = args.out / 'selection_report.json'
    signature = dict(schema=SCHEMA, bundle_sha256=digest(args.bundle),
        pool_sha256=digest(args.pool / 'pool_report.json'),
        model_report_sha256=digest(args.models / 'report.json'), members=args.members,
        min_margin=args.min_margin, uncertainty_z=args.uncertainty_z,
        qor_tolerance=args.qor_tolerance)
    if report_path.exists():
        previous = load(report_path)
        if previous['signature'] != signature:
            raise ValueError('selection inputs changed; choose a new output directory')
        for row in previous['selections']:
            if row['selected_fplace'] and digest(row['selected_fplace']) != row['selected_sha256']:
                raise ValueError('selected fplace changed after selection')
        return
    candidates = defaultdict(list)
    for row in pool_report['candidates']:
        candidates[row['design'], row['width']].append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    selections = []
    for key, rows in sorted(candidates.items()):
        design, width = key
        ctx, evaluator, base = v62.context(bundle, key)
        models = []
        for member in range(args.members):
            saved = torch.load(args.models / f'target_{design}_member{member}.pt',
                               map_location=device, weights_only=True)
            if design in saved['training_designs'] or saved['signature']['bundle_sha256'] != digest(args.bundle):
                raise ValueError('target design leaked into training')
            model = PlacementChoiceMatcher(saved['signature']['hidden']).to(device)
            model.load_state_dict(saved['state'])
            model.eval()
            models.append((model, saved['normalization']))
        predictions = {}
        for row in rows:
            assignment = load(row['binding'])['assignment']
            graph = candidate_graph(base, ctx['logic'], ctx['device'], evaluator, assignment)
            graph['is_default'] = row['candidate'] == 'default'
            predictions[row['candidate']] = [infer(model, graph, norm, device) for model, norm in models]
        noop = next(r for r in rows if r['candidate'] == 'noop')
        current = noop
        audits = []
        for row in rows:
            if row['candidate'] == 'noop':
                continue
            audit = v62.conservative_audit(row['candidate'], row['moved_clusters'],
                predictions[row['candidate']], predictions['noop'], args.min_margin,
                args.uncertainty_z, args.qor_tolerance)
            audits.append(audit)
            if audit['eligible'] and (current is noop or
                    audit['score_lcb'] > next(a['score_lcb'] for a in audits
                                               if a['candidate'] == current['candidate'])):
                current = row
        work = args.out / design / f'w{width}'
        work.mkdir(parents=True, exist_ok=True)
        selected_path = None
        selected_sha = None
        if current['fplace']:
            selected_path = work / 'selected.fplace'
            shutil.copy2(current['fplace'], selected_path)
            selected_sha = digest(selected_path)
        result = dict(design=design, width=width, selected=current['candidate'],
            selected_fplace=str(selected_path) if selected_path else None,
            selected_sha256=selected_sha, baseline=noop['fplace'],
            baseline_sha256=noop['fplace_sha256'], inputs=noop['inputs'],
            device_width=noop['device_width'], full_legalizer=noop['full_legalizer'], audits=audits)
        save(work / 'selection.json', result)
        selections.append(result)
        print(json.dumps(dict(stage='select', design=design, width=width,
                              selected=current['candidate'])), flush=True)
    save(report_path, dict(schema=SCHEMA, signature=signature, selections=selections))


def evaluate(args):
    selection = load(args.selection / 'selection_report.json')['selections']
    tasks = []
    for row in selection:
        for method, fplace in (('default', None), ('gnn_base', row['baseline']),
                               ('v63', row['selected_fplace'])):
            for seed in args.eval_seeds:
                if fplace:
                    expected = row['baseline_sha256' if method == 'gnn_base' else 'selected_sha256']
                    if digest(fplace) != expected:
                        raise ValueError('evaluation fplace changed after selection')
                tasks.append(dict(**{k: row[k] for k in ('design', 'width', 'inputs', 'device_width',
                                  'full_legalizer')}, candidate=method, fplace=fplace, seed=seed,
                    directory=str(args.out / row['design'] / f'w{row["width"]}' / method / f's{seed}')))
    args.out.mkdir(parents=True, exist_ok=True)
    rows = v62.parallel_trials(tasks, args.jobs, args.timeout, args.out, 'vpr_results.partial.json')
    comparisons = v62.compare(rows, contender='v63')
    save(args.out / 'vpr_results.json', dict(schema=SCHEMA, complete=True, rows=rows,
        comparisons=comparisons, seeds=args.eval_seeds))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True,
                        choices=('probe', 'pool', 'label', 'dataset', 'train', 'select', 'evaluate'))
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--v61-bundle', type=Path)
    parser.add_argument('--v62-pool', type=Path)
    parser.add_argument('--v62-labels', type=Path)
    parser.add_argument('--v62-pilot', type=Path)
    parser.add_argument('--pool', type=Path)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--models', type=Path)
    parser.add_argument('--selection', type=Path)
    parser.add_argument('--seed', type=int, default=62027)
    parser.add_argument('--label-seeds', type=int, nargs='+', default=list(LABEL_SEEDS))
    parser.add_argument('--eval-seeds', type=int, nargs='+', default=list(EVAL_SEEDS))
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--timeout', type=int, default=1800)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--hidden', type=int, default=48)
    parser.add_argument('--members', type=int, default=3)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--min-margin', type=float, default=0.01)
    parser.add_argument('--uncertainty-z', type=float, default=1.0)
    parser.add_argument('--qor-tolerance', type=float, default=0.01)
    args = parser.parse_args()
    needed = {'probe': ('v62_pool', 'v62_pilot'), 'pool': ('v61_bundle',),
              'label': ('pool',), 'dataset': ('v61_bundle', 'pool', 'labels'),
              'train': ('bundle',), 'select': ('bundle', 'pool', 'models'),
              'evaluate': ('selection',)}[args.stage]
    for name in needed:
        if getattr(args, name) is None:
            parser.error(f'{args.stage} requires --{name.replace("_", "-")}')
    if (set(args.label_seeds) & set(args.eval_seeds) or
            len(set(args.label_seeds)) != len(args.label_seeds) or
            len(set(args.eval_seeds)) != len(args.eval_seeds)):
        parser.error('label and evaluation seeds must be distinct')
    if args.jobs < 1 or args.members < 2:
        parser.error('jobs must be positive and members must be at least two')
    {'probe': probe, 'pool': pool, 'label': label, 'dataset': dataset,
     'train': train, 'select': select, 'evaluate': evaluate}[args.stage](args)


if __name__ == '__main__':
    main()
