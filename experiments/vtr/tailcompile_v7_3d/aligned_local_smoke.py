"""008 real tiny consistency smoke, reusing 007 graph/GP evidence read-only.

Trains a small model for 3 shared + 2 paired epochs, then performs exactly two
P&R trials (balanced/asymmetric, one seed, no layer prior). This checks the new
teacher/decoder/risk contract, not natural-design performance or generalization.
"""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path
import time

from io_utils import read, write, freeze, sha, run_trial, vpr_command, prune_derived
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FILES = ('aligned_local_smoke.py', 'aligned_teacher.py', 'aligned_model.py', 'train_aligned_layers.py')


def native_fixed_check(before, after):
    from data import read_snapshot
    original = read_snapshot(before, structure=True)
    placed = read_snapshot(after)
    expected = {a: b['loc'] for b in original['blocks'] if b['fixed'] for a in b['atoms']}
    actual = {a: b['loc'] for b in placed['blocks'] for a in b['atoms'] if a in expected}
    valid = set(expected) == set(actual) and all(
        all(math.isclose(actual[a][axis], value, abs_tol=1e-7, rel_tol=0.)
            for axis, value in enumerate(location[:3]) if value >= 0)
        for a, location in expected.items())
    return dict(fixed_atom_count=len(expected), fixed_atom_coverage=len(actual), passed=valid)


def run(source, output, binary, *, timeout=180):
    import torch
    from aligned_teacher import project_teacher, verify_teacher, deployed_xy_bounds
    from aligned_model import generate
    from train_aligned_layers import train_shared, prepare_arms, train_feedback, load_model
    from generation import lower_initialization
    from supervision import project_teacher as individual_projection
    from data import raw_teacher
    from vertical_risk import hard_risk
    from layer_observations import collect_layers
    from model import validate, compatibility_mask

    source, output, binary = (Path(p).resolve() for p in (source, output, binary))
    validate_output_roots((output,), dependency_roots=(source, binary))
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('007 fixture inputs and 008 outputs must be disjoint')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cases = {r['name']: r for r in read(source/'qualification_inputs.json')}
    selected = ('balanced', 'asymmetric')
    inputs = [binary, source/'qualification_inputs.json']
    for name in selected:
        inputs += [source/'model_smoke'/name/f for f in ('graph.pt', 'logic.json', 'device_graph.json')]
        inputs += [Path(cases[name]['arch']), Path(cases[name]['blif'])]
        for prefix in (cases[name]['structure'], str(source/'deployments'/name/'s2701/I/post_gp')):
            inputs += [Path(prefix+s) for s in ('.blocks.tsv', '.nets.tsv', '.meta.tsv')]
        if cases[name].get('constraints'):
            inputs.append(Path(cases[name]['constraints']))
    source_hashes = {str(p): sha(p) for p in inputs}
    freeze(output/'protocol.json', dict(schema='aligned-local-smoke-008-1', source=str(source),
        input_sha256=source_hashes, implementation={n: sha(HERE/n) for n in FILES},
        shared_epochs=3, candidate_epochs=2, hidden=16, training_seed=3801,
        pnr_seed=3701, pnr_methods=['T-I'], pnr_count=2, timeout=timeout,
        scope='engineering consistency and native consumption; no performance/generalization claim'))

    graphs, logicals, devices, teachers, audits, records = {}, {}, {}, {}, {}, []
    for name in selected:
        folder = output/'teachers'/name
        graph_file = source/'model_smoke'/name/'graph.pt'
        data = torch.load(graph_file, weights_only=False, map_location='cpu')
        logical = read(source/'model_smoke'/name/'logic.json')
        device = read(source/'model_smoke'/name/'device_graph.json')
        raw = raw_teacher(source/'deployments'/name/'s2701/I/post_gp', logical)
        compatible, individual_audit = individual_projection(data, raw)
        aligned, alignment_audit = project_teacher(data, compatible)
        verified = verify_teacher(data, aligned)
        assert verified['valid'] and verified['hard_layer_capacity_satisfied']
        write(folder/'raw_teacher.json', raw)
        write(folder/'individually_compatible_teacher.json', compatible)
        write(folder/'aligned_teacher.json', aligned)
        write(folder/'teacher_alignment_audit.json', dict(individual=individual_audit, aligned=alignment_audit))
        graphs[name], logicals[name], devices[name], teachers[name] = data, logical, device, aligned
        audits[name] = alignment_audit
        records.append(dict(condition_id=f'engineering_fixture/{name}/w320', family='engineering_fixture',
                            split='train', graph=str(graph_file), teacher=str(folder/'aligned_teacher.json')))
    freeze(output/'teacher_records.json', dict(records=records))

    # Explicitly controlled negative example, distinct from native teacher data:
    # both individually compatible RAM APBlocks are requested in the asymmetric
    # layer with only one RAM proxy unit. This is not a fabricated VPR result and
    # is not used as a training record.
    data = graphs['asymmetric']
    controlled = copy.deepcopy(teachers['asymmetric'])
    controlled.pop('schema', None)
    controlled.pop('constraints_sha256', None)
    regions = torch.as_tensor(controlled['teacher_region'], dtype=torch.long)
    xy = torch.as_tensor(controlled['teacher_xy'], dtype=torch.float64)
    meta = validate(data)
    changed_ram = []
    for node in torch.where((data['demand'][:, 2] > 0) & data['movable'].bool())[0].tolist():
        mask = compatibility_mask(data, torch.tensor([node]), meta)[0] & (data['region_layer'] == 0)
        choices = torch.where(mask)[0]
        if len(choices):
            regions[node] = choices[0]
            changed_ram.append(node)
    low, high = deployed_xy_bounds(data, regions, meta)
    xy = torch.maximum(low, torch.minimum(high, xy))
    controlled['teacher_region'] = regions.tolist()
    controlled['teacher_layer'] = data['region_layer'][regions].tolist()
    controlled['teacher_xy'] = xy.tolist()
    controlled_aligned, controlled_audit = project_teacher(data, controlled)
    assert changed_ram and controlled_audit['raw_capacity']['layer_overflow_by_resource'][2] > 0
    assert controlled_audit['aligned_capacity']['hard_layer_capacity_satisfied']
    write(output/'controlled/asymmetric_ram_overflow_source.json', controlled)
    write(output/'controlled/asymmetric_ram_overflow_aligned.json', controlled_aligned)
    write(output/'controlled_memory_overflow_report.json', dict(controlled=True, training_record=False,
        source='actual asymmetric fixture; force individually compatible movable RAM APBlocks into layer0',
        forced_ram_nodes=changed_ram, original_native_teacher_overflow=audits['asymmetric']['raw_capacity']['layer_overflow_by_resource'],
        projection=controlled_audit))

    shared_path = output/'models/shared/training_report.json'
    shared = read(shared_path) if shared_path.exists() else train_shared(
        records, output/'models/shared', epochs=3, hidden=16, seed=3801)
    prepared_path = output/'models/prepared/preparation_report.json'
    prepared = read(prepared_path) if prepared_path.exists() else prepare_arms(
        shared, records, output/'models/prepared', epochs=2, seed=3801)
    # No same-condition runtime preference pool is invented from these two
    # different architectures. Explicitly preserve the no-feedback outcome.
    feedback = train_feedback(records, prepared, output/'models/feedback', [], epochs=2, seed=3801)
    parity, lowered = {}, {}
    for name in selected:
        data = graphs[name]
        checkpoint = prepared['arms']['T']['checkpoint']
        model = load_model(checkpoint)
        model.eval()
        started = time.perf_counter()
        with torch.no_grad():
            _, training_detail = model.supervision_loss(data, teachers[name], tail_weight=.1, shared=False)
            prediction = generate(model, data, method='T', diagnostics=True)
            fast_prediction = generate(model, data, method='T', diagnostics=False)
            direct = hard_risk(data, prediction['layer'], prediction['xy'].to(data['demand']))
        exact_action = training_detail['deployed_action_sha256'] == prediction['deployed_action_sha256'] == fast_prediction['deployed_action_sha256']
        same_risk = math.isclose(training_detail['vertical_tail'], direct['cvar95'], rel_tol=0., abs_tol=1e-7)
        hard_capacity = prediction['capacity_diagnostics']['valid']
        check = dict(training_deployment_action_identical=exact_action,
            risk_forward_equals_deployed_hard_risk=same_risk, training_forward_cvar=training_detail['vertical_tail'],
            deployed_hard_cvar=direct['cvar95'], no_diagnostics_same_action=True,
            no_diagnostics_did_not_compute_risk=not fast_prediction['risk_diagnostics_computed'] and 'hard_vertical_risk' not in fast_prediction,
            deployed_action_sha256=prediction['deployed_action_sha256'],
            layer_capacity=prediction['capacity_diagnostics'], risk_gradient_semantics=training_detail['risk_gradient_semantics'],
            check_seconds=time.perf_counter()-started, physical_packing_certified=False)
        check['passed'] = exact_action and same_risk and hard_capacity and check['no_diagnostics_did_not_compute_risk']
        assert check['passed'], check
        parity[name] = check
        generated = lower_initialization(logicals[name], devices[name], prediction, output/'initializers'/name, method='aligned_T')
        lowered[name] = generated
        write(output/'initializers'/name/'alignment_report.json', check)

    trials = []
    for name in selected:
        case = cases[name]
        target = output/'deployments'/name
        target.mkdir(parents=True, exist_ok=True)
        generated = lowered[name]
        command = vpr_command(binary, case['arch'], case['blif'], case['width'], 3701)+case['options']
        command += ['--write_flat_place', 'post_route.fplace', '--write_legalized_flat_place', 'post_legalizer.fplace',
                    '--flat_place_verbosity', '0']
        dependencies = [binary, case['arch'], case['blif'], generated['ap_init']]
        if case.get('constraints'): dependencies.append(case['constraints'])
        result = run_trial(target/'vpr', command, inputs=dependencies,
            env=dict(TC_GP_INIT=generated['ap_init'], TC_GP_LAYER_PRIOR='0', TC_GP_EXPORT=target/'post_gp',
                     TC_GP_Z_USAGE=target/'vertical_usage'), timeout=timeout, prune=False,
            metadata=dict(stage='aligned_local_consistency', case=name, seed=3701, method='T-I'))
        durable = target/'alignment_pnr_report.json'
        if durable.exists():
            saved = read(durable)
            if saved['initializer_sha256'] != generated['ap_init_sha256']:
                raise ValueError('saved tiny P&R used another initializer')
            trials.append(saved)
            continue
        log = Path(result['log']).read_text(errors='replace')
        consumed = 'TAILCOMPILE_V7_GP_INIT_CONSUMED' in log
        continued = 'TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' in log
        fixed = (native_fixed_check(case['structure'], target/'post_gp')
                 if Path(str(target/'post_gp')+'.blocks.tsv').is_file() else
                 dict(passed=False, reason='post-GP snapshot unavailable'))
        observation = collect_layers(generated['ap_init'], target/'post_gp',
            final_fplace=Path(result['workdir'])/'post_route.fplace',
            legalized_fplace=Path(result['workdir'])/'post_legalizer.fplace')
        trial = dict(case=name, status=result['status'], metrics=result['metrics'],
            initializer_sha256=generated['ap_init_sha256'], init_consumed=consumed,
            native_gp_continued=continued, fixed=fixed, layer_observations=observation,
            passed=result['status']=='success' and consumed and continued and fixed['passed'],
            log=result['log'], vpr_manifest=str(target/'vpr/manifest.json'))
        trials.append(trial)
        write(target/'alignment_pnr_report.json', trial)
        if result['status'] in ('success', 'unroutable', 'complete'):
            prune_derived(result['workdir'], protected=dependencies)
            removed = []
            for path in list(target.glob('post_gp.*')) + [Path(result['workdir'])/f for f in ('post_route.fplace','post_legalizer.fplace')]:
                if path.is_file() and not path.is_symlink() and target.resolve() in path.resolve().parents:
                    removed.append(dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path)))
                    path.unlink()
            write(target/'trace_pruned_report.json', dict(removed=removed, observations_durable=True))
    unchanged = source_hashes == {str(p): sha(p) for p in inputs}
    report = dict(schema='aligned-local-smoke-008-1', complete=True,
        passed=all(r['passed'] for r in parity.values()) and all(r['passed'] for r in trials) and unchanged,
        source_read_only_verified=unchanged, binary_sha256=sha(binary),
        graph_nodes={name: len(data['demand']) for name, data in graphs.items()},
        graph_regions={name: len(data['capacity']) for name, data in graphs.items()},
        teacher_alignment_audits=audits, training_deployment_parity=parity,
        controlled_memory_overflow_audit=controlled_audit,
        parity_scope='pre-GP initializer decoder only; native GP/APPack/SA may change the positions afterwards',
        shared_epochs=3, candidate_epochs=2, shared_training_seconds=shared['training_seconds'],
        prepared_training_seconds={m: prepared['arms'][m]['training_seconds'] for m in ('L','T')},
        reported_weights_changed={m: prepared['arms'][m]['initial_state_sha256'] != prepared['arms'][m]['final_state_sha256'] for m in ('L','T')},
        shared_xy_sha256=prepared['shared_xy_sha256'], feedback_training_performed=False,
        feedback_pairs=feedback['feedback_pairs'], feedback_reason='no same-condition distinct-action runtime preference pool collected',
        pnr_count=len(trials), pnr_success_count=sum(r['status']=='success' for r in trials), trials=trials,
        no_new_backend_build=True, no_new_rr_graph_extraction=True,
        performance_claim=False, natural_design_training_completed=False, Wmin_measured=False)
    write(output/'aligned_smoke_report.json', report)
    print(dict(passed=report['passed'], pnr_success=report['pnr_success_count'], pnr_count=len(trials),
               report=str(output/'aligned_smoke_report.json')), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=HERE/'runs/development_007_layer_local')
    p.add_argument('--out', type=Path, default=HERE/'runs/development_008_aligned_local')
    p.add_argument('--binary', type=Path, default=REPO/'.deps/tailcompile_v7_3d_layer/vtr/build_tailcompile_layer/vpr')
    p.add_argument('--timeout', type=int, default=180)
    a = p.parse_args()
    if not run(a.source, a.out, a.binary, timeout=a.timeout)['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
