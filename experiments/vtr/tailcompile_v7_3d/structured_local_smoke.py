"""010 bounded local training/true-feedback/P&R on two immutable small fixtures.

At most 24 P&R: 8 candidate labels and 16 four-arm comparisons. These are
engineering cases, not natural-design performance or generalization evidence.
Reports/checkpoints remain in the ignored local cohort.
"""
from __future__ import annotations
import argparse
from pathlib import Path

from io_utils import read, write, freeze, sha, run_trial, vpr_command, prune_derived
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FILES = ('structured_local_smoke.py', 'structured_model.py', 'structured_decoder.py',
         'train_structured.py', 'structured_campaign.py')


def run(source, previous, aligned, output, binary, *, timeout=180):
    import torch
    from train_structured import from_previous, load_model, save_model, train_model
    from structured_model import generate, ACTION_SCHEMA, DECODER_SCHEMA, validate_collected_action
    from train_qor import load_model as load009
    from qor_model import generate as generate009
    from train import state_sha256
    from aligned_teacher import constraints_digest
    from aligned_local_smoke import native_fixed_check
    from generation import lower_initialization
    from layer_observations import collect_layers
    from qor_route_metrics import summarize_route

    source, previous, aligned, output, binary = map(lambda x: Path(x).resolve(),
                                                   (source, previous, aligned, output, binary))
    validate_output_roots((output,), dependency_roots=(source, previous, aligned, binary))
    if any(output == p or output in p.parents or p in output.parents for p in (source, previous, aligned)):
        raise ValueError('010 output must be separate from immutable fixtures')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = {c['name']: c for c in read(source/'qualification_inputs.json')}
    names = ('balanced', 'asymmetric')
    records = [r for r in read(aligned/'teacher_records.json')['records']
               if any('/'+name+'/' in r['condition_id'] for name in names)]
    if len(records) != 2 or any(r['split'] != 'train' for r in records):
        raise ValueError('two small train fixtures required')
    checkpoint = Path(read(previous/'models/bootstrap/bootstrap_report.json')['arms']['X']['checkpoint'])
    inputs = [binary, checkpoint, source/'qualification_inputs.json', aligned/'teacher_records.json']
    for r in records:
        inputs.extend((Path(r['graph']), Path(r['teacher'])))
    for name in names:
        c = cases[name]
        inputs.extend(Path(c[k]) for k in ('arch', 'blif'))
        inputs.extend(source/'model_smoke'/name/f for f in ('logic.json', 'device_graph.json'))
        inputs.extend(Path(c['structure']+s) for s in ('.blocks.tsv', '.nets.tsv', '.meta.tsv'))
        if c.get('constraints'):
            inputs.append(Path(c['constraints']))
    signatures = {str(p): sha(p) for p in inputs}
    freeze(output/'protocol.json', dict(schema='structured-local-010-1',
        pnr_upper=24, feedback_seeds=[10301, 10302], comparison_seeds=[10401, 10402],
        training_epochs=2, training_seed=10300, source_inputs=signatures,
        methods=['B', 'X', 'D', 'N'], tail_weight=0, backend_layer_prior=0,
        implementation={n:sha(HERE/n) for n in FILES},
        natural_design_claim=False))
    base009 = load009(checkpoint)
    decoder = from_previous(checkpoint)
    dpath = Path(save_model(decoder, output/'models/decoder.pt', epoch=0))
    graphs = {name: torch.load(next(r['graph'] for r in records if '/'+name+'/' in r['condition_id']),
                              weights_only=False, map_location='cpu') for name in names}
    generated = {}

    def lower(name, model, method, *, noise=0., seed=0, destination=None):
        graph = graphs[name]
        if method == 'X':
            action = generate009(model, graph, method='X', seed=seed, score_noise_std=noise)
        else:
            action = generate(model, graph, seed=seed, score_noise_std=noise)
        folder = destination or output/'initializers'/name/method
        gen = lower_initialization(read(source/'model_smoke'/name/'logic.json'),
            read(source/'model_smoke'/name/'device_graph.json'), action, folder, method=method)
        if method != 'X':
            payload = read(gen['target'])
            payload.update(schema=ACTION_SCHEMA, decoder_kind=DECODER_SCHEMA,
                shared_xy_sha256=state_sha256(model.shared_xy),
                deployed_action_sha256=action['deployed_action_sha256'],
                constraints_sha256=constraints_digest(graph),
                collection_checkpoint_sha256=sha(dpath) if method != 'N' else sha(trained['checkpoint']))
            write(gen['target'], payload)
            gen.update(target_sha256=sha(gen['target']),
                       shared_xy_sha256=payload['shared_xy_sha256'],
                       constraints_sha256=payload['constraints_sha256'])
            validate_collected_action(graph, payload)
        write(folder/'generation_report.json', gen)
        return gen

    def pnr(name, method, seed, gen, target):
        durable = target/'local_trial_report.json'
        if durable.is_file():
            saved = read(durable)
            if saved['initializer_sha256'] != (gen['ap_init_sha256'] if gen else None):
                raise ValueError('resume initializer changed')
            return saved
        c = cases[name]
        cmd = vpr_command(binary, c['arch'], c['blif'], c['width'], seed)+c['options']
        cmd += ['--write_flat_place', 'post_route.fplace', '--write_legalized_flat_place',
                'post_legalizer.fplace', '--flat_place_verbosity', '0']
        dependencies = [binary, c['arch'], c['blif']]+([gen['ap_init']] if gen else [])
        if c.get('constraints'):
            dependencies.append(c['constraints'])
        env = dict(TC_GP_LAYER_PRIOR='0', TC_GP_EXPORT=target/'post_gp', TC_GP_Z_USAGE=target/'vertical_usage')
        if gen:
            env['TC_GP_INIT'] = gen['ap_init']
        result = run_trial(target/'vpr', cmd, inputs=dependencies, env=env,
            timeout=timeout, prune=False, metadata=dict(stage='structured_local', method=method, seed=seed))
        work = Path(result['workdir'])
        log = Path(result['log']).read_text(errors='replace')
        routes = list(work.glob('*.route'))
        row = dict(case=name, method=method, seed=seed, status=result['status'],
            vpr_seconds=result['metrics']['wall_seconds'], wirelength=result['metrics']['wirelength'],
            cpd_ns=result['metrics']['cpd_ns'], phases=result['metrics'].get('phases'),
            initializer_sha256=gen['ap_init_sha256'] if gen else None,
            init_consumed='TAILCOMPILE_V7_GP_INIT_CONSUMED' in log,
            native_gp_continued='TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' in log,
            fixed=native_fixed_check(c['structure'], target/'post_gp'),
            layers=collect_layers(gen['ap_init'] if gen else None, target/'post_gp',
                final_fplace=work/'post_route.fplace', legalized_fplace=work/'post_legalizer.fplace'),
            final_route_usage=summarize_route(routes[0]) if result['status']=='success' and len(routes)==1 else None)
        row['passed'] = (row['status']=='success' and row['fixed']['passed'] and
                         (not gen or row['init_consumed'] and row['native_gp_continued']))
        write(durable, row)
        if result['status'] in ('success', 'unroutable', 'complete'):
            prune_derived(work, protected=dependencies)
            removed = []
            for p in list(target.glob('post_gp.*'))+[work/'post_route.fplace', work/'post_legalizer.fplace']:
                if p.is_file() and not p.is_symlink() and target.resolve() in p.resolve().parents:
                    removed.append(dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size))
                    p.unlink()
            write(target/'trace_pruned_report.json', dict(removed=removed, summaries_durable=True))
        print(dict(stage=target.parts[-4:], method=method, seed=seed, status=row['status'],
                   WL=row['wirelength'], CPD=row['cpd_ns']), flush=True)
        return row

    entries, trials = [], []
    for name in names:
        record = next(r for r in records if '/'+name+'/' in r['condition_id'])
        seen = set()
        for i, noise in enumerate((0., .25)):
            gen = lower(name, decoder, 'D', noise=noise, seed=10311,
                        destination=output/'feedback'/name/f'candidate{i}'/'initialization')
            if gen['ap_init_sha256'] in seen:
                continue
            seen.add(gen['ap_init_sha256'])
            measured = [pnr(name, 'D'+str(i), seed, gen, output/'feedback'/name/f'candidate{i}'/f's{seed}')
                        for seed in (10301, 10302)]
            trials.extend(measured)
            entries.append(dict(condition_id=record['condition_id'], split='train',
                candidate_id='010-local-'+str(i), action=gen['target'], generation=gen, trials=measured))
    write(output/'feedback.json', entries)
    trained = train_model(records, checkpoint, output/'models/learned', feedback=entries,
                          epochs=2, snapshots=(2,), seed=10300, device='cpu')
    learned = load_model(trained['checkpoint'])
    for name in names:
        generated[name, 'X'] = lower(name, base009, 'X')
        generated[name, 'D'] = lower(name, decoder, 'D')
        generated[name, 'N'] = lower(name, learned, 'N')
    comparisons = []
    for name in names:
        for seed in (10401, 10402):
            for method in ('B', 'X', 'D', 'N'):
                comparisons.append(pnr(name, method, seed, generated.get((name, method)),
                    output/'comparison'/name/f's{seed}'/method))
    trials.extend(comparisons)
    source_unchanged = signatures == {str(p): sha(p) for p in inputs}
    report = dict(schema='structured-local-smoke-010-1', complete=True,
        passed=source_unchanged and all(r['passed'] for r in trials),
        source_unchanged=source_unchanged, actual_pnr=len(trials), pnr_upper=24,
        feedback_actions=len(entries), feedback_pairs=trained.get('feedback_pairs'),
        training=trained, comparison=comparisons,
        tail_weight=0, natural_design_training_completed=False, main104_completed=False,
        performance_claim=False, Wmin_measured=False, evaluation_reports_committed=False)
    write(output/'structured_smoke_report.json', report)
    print(dict(passed=report['passed'], pnr=len(trials), report=str(output/'structured_smoke_report.json')), flush=True)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=HERE/'runs/development_007_layer_local')
    p.add_argument('--previous', type=Path, default=HERE/'runs/development_009_qor_local')
    p.add_argument('--aligned', type=Path, default=HERE/'runs/development_008_aligned_local')
    p.add_argument('--out', type=Path, default=HERE/'runs/development_010_structured_local')
    p.add_argument('--binary', type=Path, default=REPO/'.deps/tailcompile_v7_3d_layer/vtr/build_tailcompile_layer/vpr')
    p.add_argument('--timeout', type=int, default=180)
    a = p.parse_args()
    result = run(a.source, a.previous, a.aligned, a.out, a.binary, timeout=a.timeout)
    raise SystemExit(0 if result['passed'] else 1)
