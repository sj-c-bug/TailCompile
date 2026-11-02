"""Independent replay of uploaded 009 evidence; no training or P&R.

Uses native manifests and arithmetic seed means, without calling the campaign's
evaluator. Large server-only model/action/graph bytes are not revalidated here.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
METHODS = ('B', 'R', 'Q', 'C', 'X', 'T')
CONTRASTS = (('T', 'B'), ('X', 'B'), ('C', 'B'), ('Q', 'B'),
             ('R', 'B'), ('Q', 'R'), ('C', 'Q'), ('X', 'C'), ('T', 'X'))
METRICS = ('wirelength', 'cpd_ns', 'total_seconds', 'vpr_seconds')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
    return HERE / str(path).split('/experiments/vtr/tailcompile_v7_3d/', 1)[1]


def gm(values):
    values = list(values)
    return math.exp(statistics.mean(math.log(x) for x in values)) if values else None


def contrast(rows, method, baseline, metric):
    lookup = {(r['condition_id'], r['seed'], r['method']): r for r in rows}
    groups = defaultdict(list)
    individual = []
    for r in rows:
        if r['method'] != method:
            continue
        b = lookup[r['condition_id'], r['seed'], baseline]
        if r['status'] == b['status'] == 'success':
            av, bv = metric(r), metric(b)
            assert all(isinstance(x, (int, float)) and math.isfinite(x) and x >= 0 for x in (av, bv))
            groups[r['family'], r['architecture'], r['condition_id']].append((av, bv))
            if bv:
                individual.append(av / bv)
    by_family = defaultdict(list)
    conditions = {}
    for (family, arch, condition), values in sorted(groups.items()):
        am, bm = [statistics.mean(v[i] for v in values) for i in (0, 1)]
        ratio = am / bm if bm else None
        conditions[condition] = dict(method_seed_mean=am, baseline_seed_mean=bm,
                                     ratio=ratio, common_success_seeds=len(values))
        if ratio is not None and ratio > 0:
            by_family[family].append(ratio)
    families = {f: gm(v) for f, v in by_family.items()}
    overall = gm(families.values())
    return dict(ratio=overall, percent_change=None if overall is None else 100*(overall-1),
                family_ratios=families, conditions=conditions,
                individual_seed_wins=sum(x < 1 for x in individual),
                individual_seed_losses=sum(x > 1 for x in individual),
                individual_seed_ties=sum(x == 1 for x in individual))


def review(root):
    root = Path(root)
    rows = read(root/'comparison/report.json')['rows']
    timing = read(root/'comparison/timing_protocol.json')
    expected = {(c, s, m) for c, s in timing['cases'] for m in METHODS}
    keys = [(r['condition_id'], r['seed'], r['method']) for r in rows]
    assert len(keys) == len(set(keys)) == 240 and set(keys) == expected
    assert {r['seed'] for r in rows} == set(range(9001, 9006))
    assert all(r['status'] == 'success' for r in rows)
    assert len({r['condition_id'] for r in rows}) == 8
    protocol = read(root/'qor_protocol.json')
    init_pass = route_wl_pass = native_pass = 0
    binaries, init_hashes, generations = Counter(), {}, {}
    for row in rows:
        mp = local(row['vpr_manifest'])
        manifest = read(mp)
        sig, metrics = manifest['signature'], manifest['metrics']
        assert manifest['returncode'] == 0 and manifest['status'] == row['status']
        for mk, rk in [('wall_seconds', 'vpr_seconds'), ('cpd_ns', 'cpd_ns'), ('wirelength', 'wirelength')]:
            assert metrics[mk] == row[rk]
        cmd = sig['command']
        assert int(cmd[cmd.index('--seed')+1]) == row['seed']
        assert int(cmd[cmd.index('--route_chan_width')+1]) == row['width']
        assert cmd[cmd.index('--num_workers')+1] == '1'
        assert sig['env']['TC_GP_LAYER_PRIOR'] == '0'
        assert sig['affinity'] == row['physical_core']
        assert row['host'] == timing['host'] and row['wave_load'] == len(timing['host']['cores'])
        assert row['cold_total'] and row['total_seconds'] >= row['vpr_seconds']
        identity_path = next(p for p in sig['inputs'] if p.endswith('/binary_identity.json'))
        assert sig['inputs'][identity_path] == digest(local(identity_path))
        identity = read(local(identity_path))
        assert identity['path'] == cmd[0]
        binaries[identity['sha256']] += 1
        native_pass += 1
        if row['method'] != 'B':
            # The frozen pnr() changes status to interface_failure unless both
            # native markers occur. Raw .out files are not part of this upload.
            assert row['phases']['counters']['gp_optimizer_steps_observed'] > 0
            assert sig['inputs'][sig['env']['TC_GP_INIT']] == row['ap_init_sha256']
            gen = read(mp.parents[1]/'generation/payload/generation_report.json')
            generations[row['condition_id'], row['seed'], row['method']] = gen
            assert gen['ap_init_sha256'] == row['ap_init_sha256']
            assert gen['capacity_diagnostics']['valid']
            assert gen['collection_checkpoint_sha256'] == timing['checkpoints'][row['method']]
            assert not gen['risk_diagnostics_computed']
            for stage in ('post_gp', 'post_appack_legalization', 'final'):
                assert row['layer_observations']['stages'][stage]['matched_fraction_of_initial'] == 1
            init_pass += 1
            init_hashes[row['condition_id'], row['seed'], row['method']] = row['ap_init_sha256']
        route = row['final_route_usage']
        assert route['xy_tile_span_sum'] == row['wirelength']
        assert route['xy_coordinate_missing'] == route['z_coordinate_missing'] == 0
        assert route['rows_without_net_header'] == 0
        assert route['unique_chanz_nodes'] == row['unique_chanz_nodes']
        route_wl_pass += 1
    assert dict(binaries) == {protocol['binary_sha256']: 240}
    comparisons = {m+'/'+b: {k: contrast(rows, m, b, lambda r, k=k: r[k]) for k in METRICS}
                   for m, b in CONTRASTS}
    published = read(root/'qor_report.json')
    for item in published['comparisons']:
        for k in METRICS:
            assert math.isclose(comparisons[item['method']+'/'+item['baseline']][k]['ratio'],
                                item['metrics'][k]['ratio'], rel_tol=1e-12)
    same_initializers = {m+'/'+b: {f: sum(init_hashes[c, s, m] == init_hashes[c, s, b]
                                        for c, s in timing['cases'] if c.startswith(f+'/'))
                                 for f in ('spmv', 'mcml')}
                         for m, b in CONTRASTS if b != 'B'}
    phases = {}
    for family in ('spmv', 'mcml'):
        phases[family] = {}
        for method in METHODS:
            subset = [r for r in rows if r['family'] == family and r['method'] == method]
            phases[family][method] = dict(
                means={k: statistics.mean(r[k] for r in subset) for k in
                       ('total_seconds', 'vpr_seconds', 'structure_export_seconds',
                        'generation_process_seconds', 'graph_seconds_within_generator')},
                mean_phase_seconds={k: statistics.mean(r['phases']['seconds'][k] for r in subset) for k in
                                    ('global_placement_seconds', 'full_legalization_seconds',
                                     'detailed_placement_seconds', 'routing_seconds')},
                mean_gp_iterations=statistics.mean(r['phases']['counters']['gp_optimizer_steps_observed'] for r in subset),
                mean_route_iterations=statistics.mean(r['phases']['counters']['routing_iterations_reported_success'] for r in subset))
    route = {m+'/'+b: {k: contrast(rows, m, b, lambda r, k=k: r['final_route_usage'][k]) for k in
                      ('unique_chanz_nodes', 'per_net_unique_chanz_nodes', 'z_hop_sum')}
             for m, b in CONTRASTS}
    vertical_first = {m+'/'+b: contrast(rows, m, b, lambda r: r['vertical_routing']['first']['utilization_cvar95'])
                      for m, b in CONTRASTS}
    deployment = {}
    for family in ('spmv', 'mcml'):
        deployment[family] = {}
        for method in METHODS[1:]:
            selected = [r for r in rows if r['family'] == family and r['method'] == method]
            gs = [generations[r['condition_id'], r['seed'], method] for r in selected]
            deployment[family][method] = dict(
                ap_minority_fraction=statistics.mean(min(g['initial_layer_counts'].get(str(i), 0) for i in (0, 1))/g['molecule_count'] for g in gs),
                initial_atom_minority_fraction=statistics.mean(min(r['layer_observations']['initial']['layer_counts'].get(str(i), 0) for i in (0, 1))/r['layer_observations']['initial']['atom_count'] for r in selected),
                hard_layer_valid=sum(g['capacity_diagnostics']['valid'] for g in gs),
                leaf_proxy_overflow=sum(any(x > 0 for x in g['capacity_diagnostics']['overflow_by_resource']) for g in gs),
                hard_leaf_enforced=sum(g['capacity_diagnostics']['hard_leaf_capacity_enforced'] for g in gs),
                mean_gp_changed_atom_fraction=statistics.mean(r['layer_observations']['stages']['post_gp']['changed_fraction'] for r in selected),
                mean_final_changed_atom_fraction=statistics.mean(r['layer_observations']['stages']['final']['changed_fraction'] for r in selected))
    learning = {}
    for stage, file in [('bootstrap', 'prepared_models_report.json'), ('feedback', 'trained_models_report.json')]:
        reports = read(root/file)['arms']
        learning[stage] = {}
        for arm, report in reports.items():
            steps = [s for e in report['history'] for s in e['steps']]
            assert len(report['history']) == report['epochs'] == (5 if stage == 'bootstrap' else 10)
            assert report['train_condition_count'] == 16 and report['validation_gradients_used'] is False
            assert len({s['condition_id'] for s in steps}) == 16
            assert all(s['condition_id'].split('/')[0] in ('spmv', 'mcml') for s in steps)
            assert all(math.isfinite(s['loss']) for s in steps)
            xy_changed = report['initial_shared_xy_sha256'] != report['shared_xy_sha256']
            assert xy_changed == (arm in ('X', 'T'))
            learning[stage][arm] = dict(
                objective=report['objective'], epochs=report['epochs'], effective_pairs=report['feedback_pairs'],
                train_conditions=16, validation_gradients_used=False, xy_changed=xy_changed,
                first_epoch_feedback_loss=statistics.mean(s['feedback_loss'] for s in report['history'][0]['steps'] if s['supported_feedback_pairs']),
                last_epoch_feedback_loss=statistics.mean(s['feedback_loss'] for s in report['history'][-1]['steps'] if s['supported_feedback_pairs']),
                maximum_recorded_coordinate_gradient_l1=max(s.get('coordinate_gradient_l1', 0) for s in steps),
                cut_weight=report['cut_weight'], tail_weight=report['tail_weight'])
    source_matches, newline_matches, unavailable = [], [], []
    for name, wanted in protocol['implementation'].items():
        raw = (HERE/name).read_bytes()
        if hashlib.sha256(raw).hexdigest() == wanted:
            source_matches.append(name)
        elif hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest() == wanted:
            newline_matches.append(name)
        else:
            raise AssertionError('implementation mismatch: '+name)
    source_checked = []
    for path, wanted in protocol['source_inputs'].items():
        p = local(path)
        if p.is_file():
            assert digest(p) == wanted, path
            source_checked.append(path)
        else:
            unavailable.append(path)
    tasks = {}
    for stage in ('feedback', 'validation', 'comparison'):
        manifests = [read(p) for p in (root/stage).rglob('manifest.json')]
        native = [m for m in manifests if m.get('signature', {}).get('command', [''])[0].endswith('/vpr')
                  and not m['signature'].get('env', {}).get('TC_GP_STRUCTURE_ONLY')]
        tasks[stage] = dict(tasks=len(native), statuses=dict(Counter(m['status'] for m in native)),
                            returncodes=dict(Counter(m['returncode'] for m in native)))
    return dict(schema='tailcompile-qor-independent-review-009-1',
                aggregation='arithmetic same-seed means; equal architecture/family geometric ratios',
                actual_native_tasks=tasks, total_native_tasks=sum(v['tasks'] for v in tasks.values()),
                main=dict(tasks=len(rows), status_counts=dict(Counter(r['status'] for r in rows)),
                          native_manifest_checks=native_pass, saved_initialization_guard_passes=init_pass,
                          saved_route_xy_matches=route_wl_pass, binary_hashes=dict(binaries)),
                comparisons=comparisons, same_initializers=same_initializers,
                phases=phases, routed_vertical_usage=route, first_routing_cvar95=vertical_first,
                deployment=deployment, learning=learning,
                implementation_exact=source_matches, implementation_newline_only=newline_matches,
                source_input_hashes_checked=source_checked, server_only_inputs_not_locally_rehashed=unavailable,
                evidence_sha256={f: digest(root/f) for f in
                                 ('qor_report.json', 'qor_protocol.json', 'comparison/report.json',
                                  'comparison/timing_protocol.json', 'selection.json', 'feedback_report.json',
                                  'prepared_models_report.json', 'trained_models_report.json', 'validation/report.json')},
                Wmin_measured=False, configured_Z_capacity_sweep=False,
                independent_family_test=False,
                conclusion='complete reference-width development comparison; no overall WL/CPD benefit',
                limitations=['large raw server-only tensors, actions and checkpoints not independently rerun',
                             'route summaries rechecked against saved native metrics, original routes pruned',
                             'initialization markers inferred from frozen parser and saved status; raw logs unavailable',
                             'two train-exposed families; no significance/generalization claim',
                             'native WL is XY-only, configured Z supply not inferred from occupancy'])


if __name__ == '__main__':
    root = HERE/'runs/development_009_qor'
    result = review(root)
    out = root/'review_20261102_report.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(report=str(out), tasks=result['actual_native_tasks'],
                          conclusion=result['conclusion']), ensure_ascii=False))
