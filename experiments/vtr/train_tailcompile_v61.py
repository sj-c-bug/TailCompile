"""Train one candidate selector, with design-held-out selection audits."""
import argparse
import json
import random
import time
import statistics
from collections import defaultdict
from pathlib import Path
from tailcompile_v61 import SCHEMA, HierarchicalMatcher, preference, to_device
import torch
from torch.nn import functional as F
from train_tailcompile_v6 import normalization, targets
from tailcompile_v4_risk import save, digest


def grouped(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['design'], row['width']].append(row)
    return list(groups.values())


def group_loss(model, rows, norms, device):
    outputs, terms = [], []
    for row in rows:
        output = model(to_device(row['graph'], device))
        outputs.append(output)
        route, qor, mask, _, _ = targets(row, norms, device)
        term = F.binary_cross_entropy_with_logits(output['route_logit'], route)
        if mask.any():
            term = term + 0.2 * F.mse_loss(output['qor'][mask], qor[mask])
        if row['hotspot'] is not None:
            term = term + 0.5 * F.mse_loss(output['hotspot'], torch.tensor(row['hotspot'], device=device))
        terms.append(term)
    pairs = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            direction = preference(rows[i], rows[j])
            if direction:
                pairs.append(F.softplus(-direction * (outputs[i]['score'] - outputs[j]['score'])))
    supervised = torch.stack(terms).mean()
    rank = torch.stack(pairs).mean() if pairs else supervised * 0
    return supervised + rank


def evaluate(model, rows, device):
    records, selections, correct, pairs = [], [], 0., 0
    model.eval()
    with torch.no_grad():
        for group in grouped(rows):
            scores = [float(model(to_device(r['graph'], device))['score']) for r in group]
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
            baseline = next(r for r in group if r['method'] == 'gnn_base')
            selections.append(dict(design=selected['design'], width=selected['width'],
                chosen=selected['method'], chosen_route=selected['route'], baseline_route=baseline['route'],
                chosen_qor=selected['qor'], baseline_qor=baseline['qor']))
    return dict(pair_accuracy=correct / pairs if pairs else None, pair_count=pairs,
                selections=selections, records=records)


def run(args):
    if args.epochs < 1:
        raise ValueError('epochs must be positive')
    bundle = torch.load(args.bundle, map_location='cpu', weights_only=True)
    if bundle['schema'] != SCHEMA:
        raise ValueError('bundle schema mismatch')
    rows = bundle['examples']
    designs = sorted({r['design'] for r in rows})
    if len(designs) < 3:
        raise ValueError('need >=3 independent designs for held-out training')
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    signature = dict(bundle_sha256=digest(args.bundle), epochs=args.epochs, seed=args.seed,
                     hidden=args.hidden, schema=SCHEMA,
                     model_source_sha256=digest(Path(__file__).with_name('tailcompile_v61.py')),
                     trainer_sha256=digest(Path(__file__)))
    args.out.mkdir(parents=True, exist_ok=True)
    reports = []
    for heldout in designs:
        checkpoint = args.out / f'holdout_{heldout}.pt'
        train = [r for r in rows if r['design'] != heldout]
        test = [r for r in rows if r['design'] == heldout]
        torch.manual_seed(args.seed)
        model = HierarchicalMatcher(hidden=args.hidden).to(device)
        trace, start = [], time.monotonic()
        if checkpoint.exists():
            saved = torch.load(checkpoint, map_location=device, weights_only=True)
            if saved['signature'] != signature or saved['heldout'] != heldout:
                raise ValueError('checkpoint inputs changed; choose new output directory')
            model.load_state_dict(saved['state'])
            trace = saved['trace']
        else:
            norms = normalization(train)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
            rng, groups = random.Random(args.seed), grouped(train)
            for epoch in range(args.epochs):
                model.train()
                rng.shuffle(groups)
                losses = []
                for group in groups:
                    optimizer.zero_grad(set_to_none=True)
                    loss = group_loss(model, group, norms, device)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite loss')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                trace.append(dict(epoch=epoch + 1, loss=statistics.mean(losses)))
                print(json.dumps(dict(stage='train', heldout=heldout, **trace[-1])), flush=True)
            torch.save(dict(signature=signature, heldout=heldout, training_designs=sorted({r['design'] for r in train}),
                state=model.state_dict(), normalization=norms, trace=trace), checkpoint)
        report = dict(heldout=heldout, seconds=time.monotonic() - start,
                      checkpoint_sha256=digest(checkpoint), trace=trace, **evaluate(model, test, device))
        save(args.out / f'holdout_{heldout}.json', report)
        reports.append(report)
    save(args.out / 'report.json', dict(signature=signature, folds=reports,
        interpretation='development leave-design-out; these six designs are no longer an untouched test set'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=60)
    p.add_argument('--hidden', type=int, default=48)
    p.add_argument('--seed', type=int, default=2027)
    p.add_argument('--device', default='auto')
    run(p.parse_args())
