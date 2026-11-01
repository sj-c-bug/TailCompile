"""Read-only independent recomputation of uploaded 007 development evidence.

No Torch, raw netlist, router execution or reuse of layer_campaign.evaluate.
Writes one compact review report; historical outcome files stay immutable.
"""
import argparse
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


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def paired(rows, method, baseline='B', role='reference', metric='total_seconds', family=None):
    lookup = {(r['condition_id'], r['seed'], r['method']): r for r in rows}
    strata = defaultdict(list)
    pairs = []
    for r in rows:
        if r['method'] != method or r['width_role'] != role or family and r['family'] != family:
            continue
        b = lookup[r['condition_id'], r['seed'], baseline]
        if r['status'] != 'success' or b['status'] != 'success':
            continue
        x, y = metric(r) if callable(metric) else r.get(metric), metric(b) if callable(metric) else b.get(metric)
        if x is None or y is None or min(x, y) <= 0:
            continue
        value = x/y
        pairs.append(value)
        strata[r['family'], r['architecture'], r['condition_id']].append(value)
    families = defaultdict(list)
    for (f, arch, cid), values in strata.items():
        families[f].append(gm(values))
    family_ratios = {f: gm(values) for f, values in families.items()}
    ratio = gm(list(family_ratios.values()))
    return dict(ratio=ratio, gain=1-ratio if ratio is not None else None,
                family_ratios=family_ratios, common_success_pairs=len(pairs),
                wins=sum(v < 1 for v in pairs), losses=sum(v > 1 for v in pairs))


def review(root):
    root = Path(root)
    rows = read(root/'comparison/report.json')['rows']
    row_keys = [(r['condition_id'], r['seed'], r['method']) for r in rows]
    assert len(rows) == len(set(row_keys)) == 224
    lookup = dict(zip(row_keys, rows))
    metrics = ('total_seconds', 'vpr_seconds', 'cpd_ns', 'wirelength')
    table = {m: {k: paired(rows, m, metric=k) for k in metrics} for m in METHODS}
    increments = {m+'/'+b: {k: paired(rows, m, b, metric=k) for k in metrics}
                  for m, b in [('L-I', 'D-I'), ('T-I', 'L-I'), ('T-S', 'L-S'),
                               ('D-S', 'D-I'), ('L-S', 'L-I'), ('T-S', 'T-I')]}
    success = {m: {role: dict(Counter(r['status'] for r in rows if r['method'] == m and r['width_role'] == role))
                   for role in ('reference', 'stress')} for m in METHODS}
    files = ('comparison/report.json', 'layer_report.json', 'layer_protocol.json', 'selection.json',
             'trained_models_report.json', 'prepared_models_report.json', 'shared_report.json',
             'validation/report.json', 'comparison/timing_protocol.json', 'feedback_report.json')
    provenance = {name: digest(root/name) for name in files}
    prefix = '/experiments/vtr/tailcompile_v7_3d/runs/'+root.name+'/'
    def local(path):
        return root/str(path).split(prefix, 1)[1]
    native_errors, initial_pairs = [], []
    for r in rows:
        m = read(local(r['vpr_manifest']))
        for name, a, b in [('status', m['status'], r['status']),
                           ('seconds', m['metrics']['wall_seconds'], r['vpr_seconds']),
                           ('cpd', m['metrics']['cpd_ns'], r['cpd_ns']),
                           ('wire', m['metrics']['wirelength'], r['wirelength']),
                           ('core', m['signature']['affinity'], r['physical_core'])]:
            if a != b: native_errors.append([r['condition_id'], r['seed'], r['method'], name])
        if r['method'] != 'B':
            initial = m['signature']['env']['TC_GP_INIT']
            if m['signature']['inputs'][initial] != r['ap_init_sha256']:
                native_errors.append([r['condition_id'], r['seed'], r['method'], 'init_hash'])
        command = m['signature']['command']
        assert int(command[command.index('--seed')+1]) == r['seed']
        assert int(command[command.index('--route_chan_width')+1]) == r['width']
        assert m['signature']['env']['TC_GP_LAYER_PRIOR'] == ('1' if r['method'].endswith('-S') else '0')
        if r['method'].endswith('-I'):
            b = lookup[r['condition_id'], r['seed'], r['method'][0]+'-S']
            initial_pairs.append(r['ap_init_sha256'] == b['ap_init_sha256'])
    assert not native_errors and all(initial_pairs)
    stages = ('global_placement_seconds', 'full_legalization_seconds', 'detailed_placement_seconds', 'routing_seconds')
    breakdown, generators = {}, {}
    for f in ('spmv', 'mcml'):
        breakdown[f] = {}
        for m in METHODS:
            selected = [r for r in rows if r['family'] == f and r['method'] == m and r['width_role'] == 'reference']
            breakdown[f][m] = dict(
                arithmetic_mean_including_failures={k: statistics.mean(r[k] for r in selected) for k in
                    ('total_seconds', 'vpr_seconds', 'structure_export_seconds', 'generation_process_seconds')},
                stage_seconds={k: statistics.mean(r['phases']['seconds'][k] for r in selected) for k in stages},
                cg_iterations=statistics.mean(r['phases']['counters']['gp_cg_iterations'] for r in selected))
        generators[f] = {}
        for m in ('D-I', 'L-I', 'T-I'):
            selected = [r for r in rows if r['family'] == f and r['method'] == m and r['width_role'] == 'reference']
            gs = [read(local(r['vpr_manifest']).parents[1]/'generation/payload/generation_report.json') for r in selected]
            generators[f][m] = dict(
                minority_ap_fraction=statistics.mean(min(g['initial_layer_counts'].get('0', 0), g['initial_layer_counts'].get('1', 0))/g['molecule_count'] for g in gs),
                used_leaves=statistics.mean(g['capacity_diagnostics']['used_region_count'] for g in gs),
                lut_leaf_overflow=statistics.mean(g['capacity_diagnostics']['overflow_by_resource'][0] for g in gs),
                raw_soft_cvar_mean=statistics.mean(g['soft_vertical_risk']['cvar95'] for g in gs),
                raw_hard_cvar_mean=statistics.mean(g['hard_vertical_risk']['cvar95'] for g in gs),
                gp_changed_atom_fraction=statistics.mean(r['layer_observations']['stages']['post_gp']['changed_fraction'] for r in selected),
                final_changed_atom_fraction=statistics.mean(r['layer_observations']['stages']['final']['changed_fraction'] for r in selected),
                encoder_seconds=statistics.mean(g['phase_seconds']['encoding'] for g in gs),
                decoder_seconds=statistics.mean(g['phase_seconds']['decoding'] for g in gs),
                generator_seconds=statistics.mean(g['phase_seconds']['total'] for g in gs))
    real_risk = {}
    for f in ('spmv', 'mcml'):
        real_risk[f] = {m+'/'+b: {k: paired(rows, m, b, family=f,
                       metric=lambda r, k=k: r['vertical_routing']['first'][k])
                       for k in ('utilization_cvar95', 'member_overuse')}
                       for m, b in [('T-I', 'L-I'), ('T-S', 'L-S'), ('T-S', 'B')]}
    original = read(root/'layer_report.json')
    differences = []
    for contrast in original['comparisons']:
        for old_key, new_key in [('total_seconds', 'total_seconds'), ('cpd_ns', 'cpd_ns'), ('wirelength', 'wirelength')]:
            old = contrast['metrics'][old_key]['ratio']
            new = paired(rows, contrast['method'], contrast['baseline'], role=contrast['width_role'], metric=new_key)['ratio']
            if (old is None) != (new is None) or old is not None and not math.isclose(old, new, rel_tol=1e-12):
                differences.append([contrast['method'], contrast['baseline'], contrast['width_role'], old_key])
    assert not differences
    task_counts = {}
    for stage in ('qualification', 'feedback', 'validation', 'comparison'):
        native = []
        for path in (root/stage).rglob('manifest.json'):
            record = read(path)
            signature = record.get('signature', {})
            command = signature.get('command', [])
            if command and command[0].endswith('/vpr') and not signature.get('env', {}).get('TC_GP_STRUCTURE_ONLY'):
                native.append(record)
        task_counts[stage] = dict(tasks=len(native), statuses=dict(Counter(r['status'] for r in native)))
    teachers = {}
    previous = root.parent/'development_005_qualified_learning'
    protocol = read(root/'layer_protocol.json')
    original_signature = read(previous/'models/pretrain/input_signature.json')
    for condition in ('spmv/H0Z0/w128', 'mcml/H0Z0/w224'):
        path = previous/'teachers'/condition/'s511/supervision_audit.json'
        audit = read(path)
        suffix = '/teachers/'+condition+'/s511/projected.json'
        signed = [v for k, v in protocol['teacher_files'].items() if k.endswith(suffix)]
        assert len(signed) == 1 and signed[0] == original_signature['record_payloads'][condition]['teacher']
        teachers[condition] = dict(audit_sha256=digest(path), audit_path=str(path),
            projected_layer_counts=dict(Counter(audit['projected_layer'])),
            changed_layer_count=audit['changed_layer_count'], projected_capacity=audit['projected_capacity'],
            teacher_payload_hash_matches_original_training_signature=True)
    trained = read(root/'trained_models_report.json')
    prepared = read(root/'prepared_models_report.json')
    shared = read(root/'shared_report.json')
    training = dict(shared_epochs=shared['epochs'], candidate_epochs={m: prepared['arms'][m]['epochs'] for m in ('L', 'T')},
        feedback_epochs={m: trained['arms'][m]['epochs'] for m in ('L', 'T')},
        reported_feedback_pairs=trained['feedback_pairs'], tail_normalization=shared['tail_normalization'],
        paired_common_start=prepared['arms']['L']['initial_state_sha256'] == prepared['arms']['T']['initial_state_sha256'],
        frozen_xy_hash_consistent=len({shared['shared_xy_sha256'], prepared['shared_xy_sha256'], trained['shared_xy_sha256']}) == 1,
        reported_weights_changed={phase: {m: report['arms'][m]['initial_state_sha256'] != report['arms'][m]['final_state_sha256']
                                 for m in ('L', 'T')} for phase, report in [('candidate', prepared), ('feedback', trained)]})
    feedback_groups = defaultdict(dict)
    for path in (root/'feedback').rglob('metrics.json'):
        row = read(path)
        if 'condition_id' in row:
            feedback_groups[row['condition_id'], row['method']][row['seed']] = row
    pair_kinds, candidate_results = Counter(), Counter()
    feedback_conditions = set()
    for condition in sorted({k[0] for k in feedback_groups}):
        actions = [(m, v) for (c, m), v in feedback_groups.items() if c == condition and m != 'B']
        control = feedback_groups[condition, 'B']
        for method, action in actions:
            if set(action) == set(control) and all(action[s]['status'] == control[s]['status'] == 'success' for s in action):
                ratio = gm([action[s]['vpr_seconds']/control[s]['vpr_seconds'] for s in action])
                candidate_results['backend_improves' if ratio < 1 else 'backend_regresses'] += 1
        for i, (_, a) in enumerate(actions):
            for _, b in actions[i+1:]:
                if set(a) != set(b): continue
                sa = {s for s, row in a.items() if row['status'] == 'success'}
                sb = {s for s, row in b.items() if row['status'] == 'success'}
                if len(sa) != len(sb):
                    pair_kinds['success_count'] += 1
                elif sa and sa == sb and abs(statistics.mean(math.log(a[s]['vpr_seconds']/b[s]['vpr_seconds']) for s in sa)) > 1e-10:
                    pair_kinds['backend_runtime'] += 1
                else:
                    continue
                feedback_conditions.add(condition)
    assert sum(pair_kinds.values()) == trained['feedback_pairs'] == 40
    training['independent_feedback_pair_kinds'] = dict(pair_kinds)
    training['independent_feedback_conditions_with_pairs'] = len(feedback_conditions)
    training['common_success_candidate_backend_results'] = dict(candidate_results)
    implementation, newline_only = [], []
    for name, expected in protocol['implementation'].items():
        content = (HERE/name).read_bytes()
        if hashlib.sha256(content).hexdigest() == expected:
            implementation.append(name)
        elif hashlib.sha256(content.replace(b'\r\n', b'\n')).hexdigest() == expected:
            newline_only.append(name)
        else:
            raise AssertionError('execution source changed: '+name)
    qualification = read(root/'qualification/qualification_report.json')
    assert protocol['binary_sha256'] == qualification['binary_sha256'] == read(root/'binary_identity.json')['sha256']
    assert digest(root/'qualification/qualification_report.json') == protocol['qualification_sha256']
    timing = read(root/'comparison/timing_protocol.json')
    assert all(r['host'] == timing['host'] and r['wave_load'] == 16 for r in rows)
    rss = defaultdict(list)
    for row in rows:
        values = [float(m) for e in row['phases']['evidence'] for m in re.findall(r'max_rss ([\d.]+) MiB', e.get('raw', ''))]
        if values: rss[row['family']].append(max(values))
    resource = dict(declared_per_job_mib=read(root/'execution/compare/protocol.json')['per_job_mib'],
        observed_peak_rss_mib={f: dict(count=len(v), minimum=min(v), maximum=max(v), median=statistics.median(v),
                                     over_declared=sum(x > 4096 for x in v)) for f, v in rss.items()},
        system_pressure_or_external_workload_monitoring_available=False)
    return dict(schema='independent-layer-007-review-1', date='2026-11-01', uploaded_commit='8b5d4e319',
        inputs_sha256=provenance, review_source_sha256=digest(__file__), unique_comparison_tasks=len(rows),
        native_metadata_errors=native_errors, paired_initial_sha_checks=len(initial_pairs), paired_initial_sha_passed=True,
        original_aggregate_mismatches=differences, success=success, reference_vs_default=table,
        reference_incremental= increments, reference_breakdown=breakdown, reference_initializer_diagnostics=generators,
        real_first_routing_risk=real_risk, native_task_counts=task_counts, teacher_constraint_audits=teachers,
        training_report_checks=training,
        execution_checks=dict(exact_source_hashes=implementation, newline_only_source_differences=newline_only,
            binary_sha256=protocol['binary_sha256'], timing_host=timing['host'], memory=resource,
            method_order_positions={m: dict(Counter(r['order'].index(m) for r in rows if r['method'] == m)) for m in METHODS}),
        selection=read(root/'selection.json'), feedback=read(root/'feedback_report.json'),
        qualifications=read(root/'qualification/qualification_report.json')['passed'],
        limitations=['Two exposed train families, two backend seeds; no unseen-family generalization',
                     'Cold process measurements in a fixed 16-lane environment; no serial runtime claim',
                     'No Wmin scan; stress panel largely saturated with unroutable cases',
                     'Raw stdout, RR graphs, weights and new backend manifest unavailable locally',
                     'Capacity proxies and atom relocation statistics are not physical packing certificates',
                     'Last routing-iteration occupancy may precede restoration of the best route'],
        conclusion='No cold end-to-end acceleration over Default; small incremental GNN gains do not establish vertical proxy validity')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=HERE/'runs/development_007_layer')
    args = p.parse_args()
    result = review(args.root)
    output = args.root/'review_20261101_report.json'
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(output=str(output), primary=result['reference_vs_default']['T-S'],
                         metadata_errors=len(result['native_metadata_errors'])), indent=2))


if __name__ == '__main__':
    main()
