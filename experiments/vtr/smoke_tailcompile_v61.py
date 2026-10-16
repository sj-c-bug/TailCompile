"""Real-netlist integration smoke. Random weights test plumbing, NOT QoR."""
import argparse
import random
import tempfile
import shutil
from types import SimpleNamespace
from pathlib import Path
from tailcompile_v61 import SCHEMA, HierarchicalMatcher, static_graph, candidate_graph, swap_proposal
import torch
from prepare_tailcompile_v61 import load, tile_device, supply_features
from tailcompile_v4_risk import RiskEvaluator, build_supply, region_at, save, digest
from tailcompile_v2 import contract_edges
from tailcompile_v3 import lower_fplace
from tailrisk_core import parse_blif
from prepare_tailcompile_v6 import actual_hotspot


def run(root, out):
    torch.manual_seed(2027)
    torch.set_num_threads(1)
    logic = load(root / 'inputs/logic_ir.json')
    device = tile_device(load(root / 'inputs/device_ir.json'), 4)
    supply = build_supply(root / 'w90/supply_run/device.rr_graph.xml', device)
    supply_features(device, supply)
    blif = root / 'inputs/sha.pre-vpr.blif'
    primitives, _ = parse_blif(blif)
    edges, _ = contract_edges(primitives, {a: c['id'] for c in logic['clusters'] for a in c['atoms']})
    evaluator = RiskEvaluator(logic, device, supply, edges)
    lower = load(root / 'w90/gnn_initial/candidate_00_seed2027_lowering.json')
    records = sorted(lower['clusters'], key=lambda r: r['cluster'])
    assignment = [region_at(device, c['x'], c['y']) for c in records]
    base = static_graph(logic, device, blif, supply)
    model = HierarchicalMatcher(hidden=12)
    graph = candidate_graph(base, logic, device, evaluator, assignment)
    output = model(graph)
    output['score'].backward()
    original = float(output['score'].detach())
    proposal = None
    rng = random.Random(42)
    for _ in range(200):
        proposal = swap_proposal(assignment, logic, evaluator, rng, assignment, .10)
        if proposal is not None and evaluator.state(proposal) is not None:
            break
    if proposal is None:
        raise AssertionError('no valid test move')
    changed = float(model(candidate_graph(base, logic, device, evaluator, proposal))['score'].detach())
    if original == changed:
        raise AssertionError('candidate scores did not respond to swap')
    out.mkdir(parents=True, exist_ok=True)
    lowering = lower_fplace(logic, device, proposal, torch.zeros(len(assignment), 2), out / 'smoke.fplace')
    assert all(region_at(device, c['x'], c['y']) == proposal[c['cluster']] for c in lowering['clusters'])
    hotspots = actual_hotspot(root / 'w90/supply_run', device)
    assert len(hotspots) == len(device['regions'])
    report = dict(test='untrained real SHA plumbing, not performance evidence',
        regions=len(device['regions']), hierarchy_counts=[len(p) for p in base['parents']],
        clusters=len(assignment), pools=len(supply['pools']), actual_hotspot_count=len(hotspots),
        initial_random_score=original, swapped_random_score=changed,
        output_sha256=digest(out / 'smoke.fplace'),
        model_gradients_finite=all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None),
        passed=True)
    # Exercise production generation and checkpoint-resume using explicitly random weights.
    from run_tailcompile_v61 import generate
    with tempfile.TemporaryDirectory() as temp:
        temp = Path(temp)
        save(temp / 'context/supply.json', supply)
        save(temp / 'origin/candidates/gnn_base_lowering.json', lower)
        shutil.copy2(root / 'w90/gnn_initial/candidate_00_seed2027.fplace',
                     temp / 'origin/candidates/gnn_base.fplace')
        torch.save(dict(schema=SCHEMA, contexts={('sha', 90): dict(logic=logic, device=device,
            context_dir=str(temp / 'context'), inputs=str((root / 'inputs').resolve()),
            folder=str(temp / 'origin'))}), temp / 'bundle.pt')
        (temp / 'models').mkdir()
        torch.save(dict(heldout='sha', training_designs=[], state=model.state_dict(),
            normalization=[(0., 1.)] * 4,
            signature=dict(hidden=12, bundle_sha256=digest(temp / 'bundle.pt'))),
            temp / 'models/holdout_sha.pt')
        args = SimpleNamespace(bundle=temp / 'bundle.pt', models=temp / 'models', out=temp / 'generated',
            device='cpu', designs=None, budget=4, seed=2027, max_moved_fraction=.10)
        generate(args)
        first = digest(temp / 'generated/sha/w90/selected.fplace')
        generate(args)
        assert first == digest(temp / 'generated/sha/w90/selected.fplace')
        report['generation_and_resume_passed'] = True
    save(out / 'smoke_report.json', report)
    print(report)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sha-root', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    run(a.sha_root, a.out)
