"""Independently audit uploaded 008 evidence; stdlib only, no new P&R/training.

Raw tensors, teachers and weights are deliberately absent from Git. This checks
their recorded identities, not their unavailable bytes or a new model forward.
"""
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

HERE = Path(__file__).resolve().parent
METHODS = ('B', 'D-I', 'L-I', 'T-I', 'D-S', 'L-S', 'T-S')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local(path):
    return HERE/str(path).split('/experiments/vtr/tailcompile_v7_3d/', 1)[1]


def gm(values):
    return math.exp(statistics.mean(math.log(x) for x in values)) if values else None


def observed_mean(values):
    values = list(values)
    observed = [x for x in values if isinstance(x,(int,float)) and math.isfinite(x)]
    return dict(mean=statistics.mean(observed) if observed else None,
                observed=len(observed),missing=len(values)-len(observed))


def paired(rows, method, baseline='B', *, role='reference', metric='total_seconds', family=None):
    lookup = {(r['condition_id'], r['seed'], r['method']): r for r in rows}
    grouped, pairs = defaultdict(list), []
    for row in rows:
        if row['method'] != method or row['width_role'] != role or family and row['family'] != family:
            continue
        control = lookup[row['condition_id'], row['seed'], baseline]
        if row['status'] != 'success' or control['status'] != 'success':
            continue
        x, y = (metric(row), metric(control)) if callable(metric) else (row.get(metric), control.get(metric))
        if x is None or y is None or min(x, y) <= 0 or not math.isfinite(x/y):
            continue
        ratio = x/y
        pairs.append(ratio)
        grouped[row['family'], row['architecture'], row['condition_id']].append(ratio)
    family_values = defaultdict(list)
    for (fam, arch, cid), values in grouped.items():
        family_values[fam].append(gm(values))
    by_family = {f: gm(v) for f, v in family_values.items()}
    ratio = gm(list(by_family.values()))
    return dict(ratio=ratio, gain=None if ratio is None else 1-ratio, family_ratios=by_family,
        common_success_pairs=len(pairs), wins=sum(x < 1 for x in pairs), losses=sum(x > 1 for x in pairs))


def review(root):
    root = Path(root)
    rows = read(root/'comparison/report.json')['rows']
    timing = read(root/'comparison/timing_protocol.json')
    lookup = {(r['condition_id'], r['seed'], r['method']): r for r in rows}
    expected = {(c, s, m) for c, s in timing['cases'] for m in METHODS}
    assert len(rows) == len(lookup) == 224 and set(lookup) == expected
    native_errors, init_checks, generated_all = [], [], []
    for row in rows:
        manifest = read(local(row['vpr_manifest']))
        for label, a, b in [('status', manifest['status'], row['status']),
                ('wall', manifest['metrics']['wall_seconds'], row['vpr_seconds']),
                ('cpd', manifest['metrics']['cpd_ns'], row['cpd_ns']),
                ('wire', manifest['metrics']['wirelength'], row['wirelength']),
                ('affinity', manifest['signature']['affinity'], row['physical_core'])]:
            if a != b: native_errors.append([row['condition_id'], row['seed'], row['method'], label])
        signature = manifest['signature']
        command = signature['command']
        assert int(command[command.index('--seed')+1]) == row['seed']
        assert int(command[command.index('--route_chan_width')+1]) == row['width']
        assert command[command.index('--num_workers')+1] == '1'
        assert signature['env']['TC_GP_LAYER_PRIOR'] == ('1' if row['method'].endswith('-S') else '0')
        if row['method'] != 'B':
            initial = signature['env']['TC_GP_INIT']
            assert signature['inputs'][initial] == row['ap_init_sha256']
            generated_all.append(read(local(row['vpr_manifest']).parents[1]/'generation/payload/generation_report.json'))
            observations = row['layer_observations']['stages']
            assert all(observations[s]['matched_fraction_of_initial'] == 1 for s in ('post_gp','post_appack_legalization','final'))
        if row['method'].endswith('-I'):
            other = lookup[row['condition_id'], row['seed'], row['method'][0]+'-S']
            init_checks.append(row['ap_init_sha256'] == other['ap_init_sha256'])
        assert row['cold_total'] and row['total_seconds'] >= row['vpr_seconds']
        assert row['host'] == timing['host'] and row['wave_load'] == len(timing['host']['cores'])
    assert not native_errors and all(init_checks)
    metrics = ('total_seconds', 'vpr_seconds', 'cpd_ns', 'wirelength')
    contrasts = [('T-S','B'), ('T-I','B'), ('L-I','B'), ('D-I','B'), ('L-S','B'), ('D-S','B'),
                 ('L-I','D-I'), ('T-I','L-I'), ('T-S','L-S'), ('D-S','D-I'), ('L-S','L-I'), ('T-S','T-I')]
    comparisons = {role: {m+'/'+b: {metric: paired(rows,m,b,role=role,metric=metric) for metric in metrics}
                   for m,b in contrasts} for role in ('reference','stress')}
    original = read(root/'layer_report.json')
    differences = []
    for contrast in original['comparisons']:
        for metric in ('total_seconds','cpd_ns','wirelength'):
            old = contrast['metrics'][metric]['ratio']
            new = paired(rows,contrast['method'],contrast['baseline'],role=contrast['width_role'],metric=metric)['ratio']
            if (old is None) != (new is None) or old is not None and not math.isclose(old,new,rel_tol=1e-12):
                differences.append([contrast['method'],contrast['baseline'],contrast['width_role'],metric])
    assert not differences
    breakdown, generators = {}, {}
    for family in ('spmv','mcml'):
        breakdown[family], generators[family] = {}, {}
        for method in METHODS:
            selected = [r for r in rows if r['family'] == family and r['method'] == method and r['width_role'] == 'reference'
                        and r['status'] == lookup[r['condition_id'],r['seed'],'B']['status'] == 'success']
            breakdown[family][method] = dict(common_success_tasks=len(selected),
                means={k:statistics.mean(r[k] for r in selected) for k in
                       ('total_seconds','vpr_seconds','structure_export_seconds','generation_process_seconds','graph_seconds_within_generator')},
                printed_disjoint_phase_seconds={k:statistics.mean(r['phases']['seconds'][k] for r in selected) for k in
                       ('global_placement_seconds','full_legalization_seconds','detailed_placement_seconds','routing_seconds')},
                mean_gp_cg_iterations=statistics.mean(r['phases']['counters']['gp_cg_iterations'] for r in selected),
                route_iterations=observed_mean(r['phases']['counters']['routing_iterations_reported_success'] for r in selected))
            if method != 'B':
                gs = [read(local(r['vpr_manifest']).parents[1]/'generation/payload/generation_report.json') for r in selected]
                generators[family][method] = dict(
                    mean_minority_ap_fraction=statistics.mean(min(g['initial_layer_counts'].get('0',0),g['initial_layer_counts'].get('1',0))/g['molecule_count'] for g in gs),
                    mean_gp_changed_atom_fraction=statistics.mean(r['layer_observations']['stages']['post_gp']['changed_fraction'] for r in selected),
                    mean_final_changed_atom_fraction=statistics.mean(r['layer_observations']['stages']['final']['changed_fraction'] for r in selected),
                    recorded_action_digests=len({g['deployed_action_sha256'] for g in gs}),
                    risk_diagnostics_off=all(g['risk_diagnostics_computed'] is False for g in gs),
                    hard_layer_capacity_passed=all(g['capacity_diagnostics']['valid'] for g in gs))
    protocol = read(root/'aligned_protocol.json')
    source_ok, newline_only = [], []
    for name, expected_hash in protocol['implementation'].items():
        content = (HERE/name).read_bytes()
        if hashlib.sha256(content).hexdigest() == expected_hash: source_ok.append(name)
        elif hashlib.sha256(content.replace(b'\r\n',b'\n')).hexdigest() == expected_hash: newline_only.append(name)
        else: raise AssertionError('implementation changed: '+name)
    source_inputs_ok, unavailable = [], []
    for path, expected_hash in protocol['source_inputs'].items():
        q = local(path)
        if q.is_file():
            assert sha(q) == expected_hash, path
            source_inputs_ok.append(path)
        else: unavailable.append(path)
    status = {m:{role:dict(Counter(r['status'] for r in rows if r['method']==m and r['width_role']==role))
                for role in ('reference','stress')} for m in METHODS}
    files = ('comparison/report.json','layer_report.json','aligned_protocol.json','queue_plan.json',
             'teacher_alignment_report.json','selection.json','shared_report.json','prepared_models_report.json',
             'trained_models_report.json','feedback_report.json','validation/report.json','comparison/timing_protocol.json')
    tasks = {}
    for stage in ('feedback','validation','comparison'):
        records = [read(p) for p in (root/stage).rglob('manifest.json')]
        native = [r for r in records if r.get('signature',{}).get('command',[None])[0] and
                  r['signature']['command'][0].endswith('/vpr') and not r['signature'].get('env',{}).get('TC_GP_STRUCTURE_ONLY')]
        tasks[stage] = dict(tasks=len(native), statuses=dict(Counter(r['status'] for r in native)))
    risk = {family:{m+'/'+b:{key:paired(rows,m,b,family=family,metric=lambda r,key=key:r['vertical_routing']['first'][key])
            for key in ('utilization_cvar95','member_overuse')} for m,b in [('T-I','L-I'),('T-S','L-S'),('T-S','B')]}
            for family in ('spmv','mcml')}
    teacher = read(root/'teacher_alignment_report.json')
    audits = [r['audit'] for r in teacher['records']]
    assert teacher['complete'] and teacher['all_rechecked'] and len(audits) == 24
    assert all(a['valid'] and a['aligned_capacity']['hard_layer_capacity_satisfied'] for a in audits)
    assert all(not a['raw_inputs_modified'] and max(a['aligned_capacity']['layer_overflow_by_resource']) == 0 for a in audits)
    assert all(a['implementation_sha256'] == sha(HERE/'aligned_teacher.py') for a in audits)
    teacher_summary = dict(conditions=len(audits), raw_inputs_modified=False,
        valid_conditions=sum(a['valid'] for a in audits),
        original_layer_overflow_conditions=sum(max(a['raw_capacity']['layer_overflow_by_resource']) > 0 for a in audits),
        aligned_layer_overflow_conditions=sum(max(a['aligned_capacity']['layer_overflow_by_resource']) > 0 for a in audits),
        changed_layer_labels=sum(a['changed_layer_count'] for a in audits),
        changed_xy_labels=sum(a['changed_xy_count'] for a in audits),
        total_projection_seconds=sum(a['projection_seconds'] for a in audits),
        examples={r['condition_id']:dict(before=r['audit']['raw_capacity']['layer_overflow_by_resource'],
                    after=r['audit']['aligned_capacity']['layer_overflow_by_resource']) for r in teacher['records']
                    if r['condition_id'] in ('spmv/H0Z0/w128','mcml/H0Z0/w224')},
        physical_packing_certified=False,aggregate_leaf_capacity_certified=False,
        raw_derived_teacher_bytes_rechecked_locally=False)
    shared, prepared, trained = [read(root/n) for n in ('shared_report.json','prepared_models_report.json','trained_models_report.json')]
    training_summary = dict(shared_epochs=shared['epochs'],candidate_epochs={m:prepared['arms'][m]['epochs'] for m in ('L','T')},
        feedback_epochs={m:trained['arms'][m]['epochs'] for m in ('L','T')},
        reported_feedback_pairs=trained['feedback_pairs'],
        paired_common_start=prepared['arms']['L']['initial_state_sha256'] == prepared['arms']['T']['initial_state_sha256'],
        shared_xy_hash_consistent=len({shared['shared_xy_sha256'],prepared['shared_xy_sha256'],trained['shared_xy_sha256'],
            *(report['arms'][m]['shared_xy_sha256'] for report in (prepared,trained) for m in ('L','T'))}) == 1,
        reported_weights_changed={phase:{m:report['arms'][m]['initial_state_sha256'] != report['arms'][m]['final_state_sha256']
            for m in ('L','T')} for phase,report in [('candidate',prepared),('feedback',trained)]},
        risk_forward=shared['risk_forward'],risk_gradient=shared['risk_gradient_semantics'],
        validation_gradients_used=any(r['validation_gradients_used'] for report in (prepared,trained) for r in report['arms'].values()),
        raw_checkpoint_bytes_rechecked_locally=False)
    assert training_summary['paired_common_start'] and training_summary['shared_xy_hash_consistent']
    assert not training_summary['validation_gradients_used']
    groups = defaultdict(dict)
    for path in (root/'feedback').rglob('metrics.json'):
        record = read(path)
        if 'condition_id' in record: groups[record['condition_id'],record['method']][record['seed']] = record
    pair_kinds, preference_conditions = Counter(), set()
    for condition in sorted({k[0] for k in groups}):
        actions = [values for (c,m),values in groups.items() if c == condition and m != 'B']
        for i,a in enumerate(actions):
            for b in actions[i+1:]:
                if set(a) != set(b): continue
                sa, sb = [{s for s,r in action.items() if r['status']=='success'} for action in (a,b)]
                if len(sa) != len(sb): pair_kinds['success_count'] += 1
                elif sa and sa == sb and abs(statistics.mean(math.log(a[s]['vpr_seconds']/b[s]['vpr_seconds']) for s in sa)) > 1e-10:
                    pair_kinds['backend_runtime'] += 1
                else: continue
                preference_conditions.add(condition)
    assert sum(pair_kinds.values()) == trained['feedback_pairs']
    training_summary.update(independent_feedback_pair_kinds=dict(pair_kinds),
        independent_feedback_conditions_with_pairs=len(preference_conditions))
    generation_summary = dict(records=len(generated_all),
        decoder_schemas=dict(Counter(g['decoder_kind'] for g in generated_all)),
        all_layer_capacity_passed=all(g['capacity_diagnostics']['valid'] for g in generated_all),
        all_risk_diagnostics_disabled=all(g['risk_diagnostics_computed'] is False for g in generated_all),
        all_shared_xy_hashes_match=all(g['shared_xy_sha256'] == trained['shared_xy_sha256'] for g in generated_all),
        leaf_capacity_overflow_records=sum(max(g['capacity_diagnostics']['overflow_by_resource']) > 0 for g in generated_all),
        hard_leaf_capacity_enforced=False,physical_packing_legality_claimed=False,
        teacher_and_model_forward_replayed_locally=False)
    assert generation_summary['all_layer_capacity_passed'] and generation_summary['all_shared_xy_hashes_match']
    source_protocol = read(local(protocol['source'])/'layer_protocol.json')
    qualification_path = local(protocol['source'])/'qualification/qualification_report.json'
    qualification = read(qualification_path)
    assert qualification['passed'] and sha(qualification_path) == source_protocol['qualification_sha256']
    assert protocol['binary_sha256'] == source_protocol['binary_sha256'] == qualification['binary_sha256'] == read(root/'binary_identity.json')['sha256']
    rss = defaultdict(list)
    for row in rows:
        values = [float(value) for e in row['phases']['evidence'] for value in re.findall(r'max_rss ([\d.]+) MiB',e.get('raw',''))]
        if values: rss[row['family']].append(max(values))
    return dict(schema='independent-aligned-008-review-1',date='2026-11-02',uploaded_commit='d5ff9601b',
        review_source_sha256=sha(__file__),inputs_sha256={name:sha(root/name) for name in files},
        unique_comparison_tasks=len(rows),native_metadata_errors=native_errors,
        paired_initial_sha_checks=len(init_checks),paired_initial_sha_passed=all(init_checks),
        original_aggregate_mismatches=differences,status_counts=status,comparisons=comparisons,
        reference_common_success_breakdown=breakdown,reference_initializer_diagnostics=generators,
        first_routing_actual_risk=risk,native_task_counts=tasks,selection=read(root/'selection.json'),
        teacher_alignment_audit=teacher_summary,training_audit=training_summary,generation_audit=generation_summary,
        execution_checks=dict(exact_source_hashes=source_ok,newline_only=newline_only,
            source_input_hashes_rechecked=len(source_inputs_ok),unavailable_raw_inputs=unavailable,
            declared_per_job_mib=read(root/'execution/compare/protocol.json')['per_job_mib'],
            observed_peak_rss_mib={f:dict(maximum=max(v),median=statistics.median(v),count=len(v)) for f,v in rss.items()},
            system_pressure_or_external_workload_monitoring_available=False,
            reused007_binary_qualification_identity_passed=True,
            fixed_timing_host=timing['host'],fixed_lane_barriers=timing['fixed_lane_barriers']),
        limits=['Two exposed train families and two comparison seeds; no unseen-family generalization',
                'Controlled parallel cold totals; not serial runtime measurements',
                'No Wmin scan; reference/stress points alone do not identify the boundary',
                'Raw teacher/graph/checkpoint bytes are not in Git; no new model forward performed',
                'Phase timers printed by VPR are not asserted to be external wall or CPU seconds'])


if __name__ == '__main__':
    root = HERE/'runs/development_008_aligned'
    result = review(root)
    out = root/'review_20261102_report.json'
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(report=str(out),tasks=result['native_task_counts'],
        primary=result['comparisons']['reference']['T-S/B']),ensure_ascii=False))
