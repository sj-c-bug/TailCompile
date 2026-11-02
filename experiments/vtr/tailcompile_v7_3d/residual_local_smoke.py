"""011 engineering smoke: <=8 true labels +10 native comparison runs.

Uses two existing tiny fixtures, not natural-design performance evidence.
The comparison exercises all learned paths without claiming a validation gate.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from io_utils import read, write, freeze, sha, run_trial, vpr_command, prune_derived
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def _json(value):
    if hasattr(value, 'detach'):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {k: _json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json(v) for v in value]
    return value


def run(output, *, timeout=180):
    import torch
    from train_residual import from_source, fit, load_model
    from structured_model import generate as generate_d
    from residual_actions import validate_residual_action
    from aligned_model import action_digest
    from generation import lower_initialization
    from aligned_local_smoke import native_fixed_check
    from layer_observations import collect_layers
    from residual_observations import collect_residual_observations
    from qor_route_metrics import summarize_route
    from train import state_sha256

    output = Path(output).resolve()
    if output.parent != HERE/'runs' or not output.name.startswith('development_011_residual_local'):
        raise ValueError('local smoke requires a distinct ignored 011 local cohort')
    source = HERE/'runs/development_007_layer_local'
    aligned = HERE/'runs/development_008_aligned_local'
    checkpoint = HERE/'runs/development_010_structured_local/models/decoder.pt'
    binary = REPO/'.deps/tailcompile_v7_3d_layer/vtr/build_tailcompile_layer/vpr'
    validate_output_roots((output,), dependency_roots=(source, aligned, checkpoint, binary))
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cases = {c['name']: c for c in read(source/'qualification_inputs.json')}
    names = ('balanced', 'asymmetric')
    records = [r for r in read(aligned/'teacher_records.json')['records']
               if any('/'+name+'/' in r['condition_id'] for name in names)]
    dependencies = [checkpoint, binary, source/'qualification_inputs.json']
    for name in names:
        dependencies.extend(Path(cases[name][k]) for k in ('arch', 'blif'))
    for record in records:
        dependencies.append(Path(record['graph']))
    hashes = {str(p): sha(p) for p in dependencies}
    freeze(output/'protocol.json', dict(schema='residual-local-011-1', pnr_upper=18,
        feedback_seed=11401, comparison_seed=11101, train_epochs=2,
        dependencies=hashes, implementation={p.name: sha(p) for p in HERE.glob('*residual*.py')},
        validation_gate_exercised_in_unit_tests=True, natural_qor_claim=False))
    model = from_source(checkpoint, expected_sha256=sha(checkpoint))
    initial_backbone = state_sha256(model.backbone)
    graphs = {name: torch.load(next(r['graph'] for r in records if '/'+name+'/' in r['condition_id']),
                               weights_only=False, map_location='cpu') for name in names}
    bases, banks = {}, {}
    for name in names:
        bases[name] = generate_d(model.backbone, graphs[name], method='D', diagnostics=False)
        for arm in ('L', 'Y', 'J'):
            _, banks[name, arm] = model.candidate_bank(graphs[name], arm, base_action=bases[name])

    def lower(name, action, method, folder):
        if method != 'D':
            validate_residual_action(graphs[name], action, base_action=bases[name], arm=method)
        logical = read(source/'model_smoke'/name/'logic.json')
        device = read(source/'model_smoke'/name/'device_graph.json')
        gen = lower_initialization(logical, device, action, folder, method=method)
        payload = read(gen['target']); payload.update(_json(action)); write(gen['target'], payload)
        gen['target_sha256'] = sha(gen['target'])
        gen['deployed_action_sha256'] = action_digest(action)
        edits = []; base, target = _json(bases[name]), _json(action)
        for i, block in enumerate(logical['clusters']):
            if base['layer'][i] != target['layer'][i] or base['xy'][i] != target['xy'][i]:
                edits.extend(dict(name=atom, base_layer=base['layer'][i], target_layer=target['layer'][i],
                                  base_xy=base['xy'][i], target_xy=target['xy'][i]) for atom in block['atoms'])
        edit_path = folder/'residual_edits.json'
        write(edit_path, dict(grid=[device.get('grid_width', device.get('grid', {}).get('width')),
                                   device.get('grid_height', device.get('grid', {}).get('height'))], atoms=edits))
        gen.update(residual_edits=str(edit_path), residual_edits_sha256=sha(edit_path))
        write(folder/'generation.json', gen)
        return gen

    def pnr(name, method, seed, gen, target):
        durable = target/'local_trial_report.json'
        if durable.exists():
            cached = read(durable)
            if cached['initializer_sha256'] != (gen['ap_init_sha256'] if gen else None):
                raise ValueError('011 local resume initializer changed')
            return cached
        case = cases[name]
        cmd = vpr_command(binary, case['arch'], case['blif'], case['width'], seed)+case['options']
        cmd += ['--write_flat_place', 'post_route.fplace', '--write_legalized_flat_place',
                'post_legalizer.fplace', '--flat_place_verbosity', '0']
        deps = [binary, case['arch'], case['blif']]+([gen['ap_init']] if gen else [])
        if case.get('constraints'):
            deps.append(case['constraints'])
        env = dict(TC_GP_LAYER_PRIOR='0', TC_GP_EXPORT=target/'post_gp', TC_GP_Z_USAGE=target/'vertical_usage')
        if gen:
            env['TC_GP_INIT'] = gen['ap_init']
        result = run_trial(target/'vpr', cmd, inputs=deps, env=env, timeout=timeout, prune=False)
        work = Path(result['workdir']); log = Path(result['log']).read_text(errors='replace')
        routes = list(work.glob('*.route'))
        row = dict(case=name, method=method, seed=seed, status=result['status'],
            wirelength=result['metrics']['wirelength'], cpd_ns=result['metrics']['cpd_ns'],
            vpr_seconds=result['metrics']['wall_seconds'], phases=result['metrics'].get('phases'),
            initializer_sha256=gen['ap_init_sha256'] if gen else None,
            init_consumed='TAILCOMPILE_V7_GP_INIT_CONSUMED' in log,
            native_gp_continued='TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' in log,
            fixed=native_fixed_check(case['structure'], target/'post_gp'),
            layers=collect_layers(gen['ap_init'] if gen else None, target/'post_gp',
                final_fplace=work/'post_route.fplace', legalized_fplace=work/'post_legalizer.fplace'),
            residual_observations=collect_residual_observations(gen, target, work),
            final_route_usage=summarize_route(routes[0]) if result['status']=='success' and len(routes)==1 else None)
        row['passed'] = (row['status']=='success' and row['fixed']['passed'] and
                         (not gen or row['init_consumed'] and row['native_gp_continued']))
        write(durable, row)
        if result['status'] in ('success', 'unroutable', 'complete'):
            prune_derived(work, protected=deps)
            removed = []
            for p in list(target.glob('post_gp.*'))+[work/'post_route.fplace', work/'post_legalizer.fplace']:
                if p.is_file() and not p.is_symlink() and target in p.resolve().parents:
                    removed.append(dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))); p.unlink()
            write(target/'trace_pruned_report.json', dict(removed=removed, summaries_durable=True))
        print(dict(case=name, method=method, seed=seed, status=row['status']), flush=True)
        return row

    feedback, trials = [], []
    for name in names:
        record = next(r for r in records if '/'+name+'/' in r['condition_id'])
        proposed = [('D', dict(bases[name], candidate_id='identity'))]
        proposed.extend((arm, banks[name, arm][-1]) for arm in ('L', 'Y', 'J') if len(banks[name, arm])>1)
        for arm, action in proposed:
            folder = output/'feedback'/name/arm
            gen = lower(name, action, arm, folder/'initialization')
            row = pnr(name, arm, 11401, gen, folder/'native'); trials.append(row)
            feedback.append(dict(condition_id=record['condition_id'], split='train', arm=arm,
                candidate_id=action['candidate_id'], action=gen['target'], generation=gen, trials=[row]))
    write(output/'feedback.json', feedback)
    training = fit(records, checkpoint, output/'models', feedback, epochs=2, snapshots=(2,),
                   device='cpu', expected_sha256=sha(checkpoint))
    learned = load_model(training['checkpoints'][-1])
    comparison = []
    for name in names:
        for arm in ('B', 'D', 'L', 'Y', 'J'):
            action = (None if arm=='B' else bases[name] if arm=='D' else
                      learned.select_from_bank(graphs[name], arm, bases[name], banks[name, arm]))
            folder = output/'comparison'/name/arm
            gen = lower(name, action, arm, folder/'initialization') if action is not None else None
            comparison.append(pnr(name, arm, 11101, gen, folder/'native'))
    trials.extend(comparison)
    unchanged = hashes == {str(p): sha(p) for p in dependencies}
    report = dict(schema='residual-local-smoke-011-1', passed=unchanged and all(r['passed'] for r in trials),
        actual_pnr=len(trials), pnr_upper=18, source_unchanged=unchanged,
        backbone_unchanged=initial_backbone==state_sha256(learned.backbone), training=training,
        comparison=comparison, candidate_counts={f'{n}/{a}':len(b) for (n,a),b in banks.items()},
        natural_design_claim=False, validation_gate_applied=False, performance_claim=False)
    write(output/'residual_smoke_report.json', report)
    print(dict(passed=report['passed'], pnr=len(trials), feedback_pairs=training['feedback_pairs']), flush=True)
    return report


def verify_workers(output):
    """Replay production raw-structure workers, with no additional VPR runs."""
    output = Path(output).resolve(); validate_output_roots((output,))
    request = read(HERE/'runs/development_009_qor_local/worker_contract/request.json')
    base = HERE/'runs/development_010_structured_local/models/decoder.pt'
    learned = Path(read(output/'models/training_report.json')['checkpoints'][-1])
    request.update(base_checkpoint=str(base), base_checkpoint_sha256=sha(base), generation_seed=11301)
    results = []
    for arm, collecting in (('D', True), ('D', False), ('L', False), ('Y', False), ('J', False)):
        folder = output/'worker_contract'/('collect' if collecting else arm)
        checkpoint = base if arm=='D' else learned
        req = dict(request, method=arm, collect_bank=collecting, enabled=arm!='D',
                   checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint))
        write(folder/'request.json', req)
        done = subprocess.run([sys.executable, str(HERE/'residual_campaign.py'), 'worker',
            '--request', str(folder/'request.json'), '--out', str(folder/'payload')],
            capture_output=True, text=True)
        (folder/'worker.log').write_text(done.stdout+done.stderr, encoding='utf-8')
        if done.returncode:
            raise RuntimeError('011 production worker failed: '+done.stderr[-4000:])
        gen = read(folder/'payload/generation_report.json')
        if collecting:
            if not 1 <= len(gen['candidates']) <= 7:
                raise ValueError('011 production candidate budget violated')
            checks = dict(collecting=True, candidates=len(gen['candidates']))
        else:
            native = read(output/'comparison/asymmetric'/arm/'initialization/generation.json')
            if gen['ap_init_sha256'] != native['ap_init_sha256']:
                raise ValueError('production worker differs from native-consumed smoke initializer: '+arm)
            checks = dict(arm=arm, exact_native_consumed_initializer=True, ap_init_sha256=gen['ap_init_sha256'])
        removed = []
        for name in ('graph.pt', 'logic.json'):
            path = folder/'payload'/name
            if path.is_file() and not path.is_symlink() and folder in path.resolve().parents:
                removed.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))); path.unlink()
        results.append(dict(checks, removed=removed))
    report = dict(passed=True, production_workers=len(results), additional_pnr=0, results=results)
    write(output/'worker_contract_report.json', report)
    print(dict(worker_contract_passed=True, workers=len(results), additional_pnr=0), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'runs/development_011_residual_local')
    parser.add_argument('--timeout', type=int, default=180)
    parser.add_argument('--verify-workers-only', action='store_true')
    args = parser.parse_args()
    result = verify_workers(args.out) if args.verify_workers_only else run(args.out, timeout=args.timeout)
    raise SystemExit(0 if result['passed'] else 1)
