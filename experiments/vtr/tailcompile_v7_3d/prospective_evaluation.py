"""Paired QoR and cross-seed opportunity in the prospective 014 joint pilot."""
from collections import Counter, defaultdict
from itertools import product
import math
from statistics import mean

from structured_campaign import contrast


CANDIDATES = ('D', 'j_s1', 'j_s2')  # Stable tie priority, original initialization first.
NATIVE = ('Default',) + CANDIDATES
SCORERS = ('E012', 'G012', 'E013')
SEEDS = (14101, 14102, 14103)
ARCHES = ('H0Z0', 'H0Z1', 'H1Z0', 'H1Z1')


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check_prediction(condition, pred):
    if pred['condition_id'] != condition['id'] or not _digest(pred['device_signature']):
        raise ValueError('014 frozen prediction identity mismatch')
    if set(pred['candidates']) != set(CANDIDATES) or set(pred['choices']) != set(SCORERS):
        raise ValueError('014 incomplete frozen joint candidate/scorer panel')
    for record in pred['candidates'].values():
        if not all(_digest(record[key]) for key in ('deployed_action_sha256', 'ap_init_sha256')):
            raise ValueError('014 invalid frozen candidate signature')
    for name, choice in pred['choices'].items():
        candidate = pred['candidates'].get(choice['method'])
        if candidate is None or any(candidate[key] != choice[key] for key in
                                    ('deployed_action_sha256', 'ap_init_sha256')):
            raise ValueError('014 frozen scorer choice differs from measured candidate: '+name)
        ids, scores = choice['candidate_ids'], choice['scores']
        if (not ids or ids[0] != 'D' or len(set(ids)) != len(ids) or set(ids)-set(CANDIDATES)
                or len(ids) != len(scores) or not all(_finite(score) for score in scores)
                or ids[max(range(len(ids)), key=lambda i: scores[i])] != choice['method']):
            raise ValueError('014 frozen choice disagrees with candidate scores: '+name)
        if not _finite(choice['scoring_seconds']) or choice['scoring_seconds'] < 0:
            raise ValueError('014 invalid measured scoring time')
    if not _finite(pred['generation_seconds']) or pred['generation_seconds'] < 0:
        raise ValueError('014 invalid measured candidate generation time')


def _index(rows, conditions, predictions):
    by_id = {c['id']: c for c in conditions}
    families = {c['family'] for c in conditions}
    pairs = {(c['family'], c['architecture']) for c in conditions}
    if (len(conditions) != 8 or len(by_id) != 8 or len(families) != 2
            or pairs != set(product(families, ARCHES)) or set(predictions) != set(by_id)):
        raise ValueError('014 requires the complete frozen two-family/four-architecture panel')
    for family in families:
        cases = [c for c in conditions if c['family'] == family]
        if len({(c['design'], c['width']) for c in cases}) != 1:
            raise ValueError('014 mixed designs or reference widths within family')
    for cid, c in by_id.items():
        _check_prediction(c, predictions[cid])
    indexed, execution, manifests = {}, {}, {}
    for r in rows:
        cid, seed, method = r['condition_id'], r['seed'], r['method']
        key = cid, seed, method
        if key in indexed or cid not in by_id or seed not in SEEDS or method not in NATIVE:
            raise ValueError('014 duplicate/unexpected main trial: '+str(key))
        if any(r[k] != by_id[cid][k] for k in ('family', 'design', 'architecture', 'width')):
            raise ValueError('014 trial differs from qualified condition: '+str(key))
        pred = predictions[cid]
        if pred['device_signature'] != r['device_signature']:
            raise ValueError('014 native device differs from frozen prediction: '+str(key))
        expected = pred['candidates'].get(method, {})
        if any(r[k] != expected.get(k) for k in ('deployed_action_sha256', 'ap_init_sha256')):
            raise ValueError('014 consumed native initializer differs from frozen candidate: '+str(key))
        signature = r['execution_signature']
        if not _digest(signature) or execution.setdefault((cid, seed), signature) != signature:
            raise ValueError('014 unmatched native backend/netlist/RRG/seed signature: '+str(key))
        if r['status'] not in ('success', 'unroutable') and not (
                r['status'] == 'packing_or_placement_failure' and r.get('confirmed_candidate_infeasibility') is True):
            raise ValueError('014 infrastructure/interface failure cannot be a scientific outcome: '+str(key))
        if r['status'] == 'success' and not all(_finite(r[k]) and r[k] > 0 for k in ('wirelength', 'cpd_ns')):
            raise ValueError('014 successful route lacks positive WL/CPD: '+str(key))
        if r['status'] in ('success', 'unroutable') and (
                r['input_consumed'] != (method != 'Default') or method != 'Default' and not r['native_solver_continued']):
            raise ValueError('014 missing native initializer consumer evidence: '+str(key))
        if not _finite(r['vpr_seconds']) or r['vpr_seconds'] < 0:
            raise ValueError('014 invalid native runtime: '+str(key))
        evidence = (cid, seed, signature, r['ap_init_sha256'], r['status'], r.get('wirelength'),
                    r.get('cpd_ns'), r['vpr_seconds'], r.get('unique_chanz_nodes'))
        if not r['vpr_manifest'] or manifests.setdefault(r['vpr_manifest'], evidence) != evidence:
            raise ValueError('014 shared native manifest has conflicting inputs or results: '+str(key))
        indexed[key] = r
    if set(indexed) != set(product(by_id, SEEDS, NATIVE)):
        raise ValueError('014 incomplete native matrix')
    return indexed


def _best(indexed, cid, selection_seeds):
    """Use successful counts first; tied candidates need a common QoR population."""
    success = {m: [s for s in selection_seeds if indexed[cid, s, m]['status'] == 'success']
               for m in CANDIDATES}
    top = max(map(len, success.values()))
    contenders = [m for m in CANDIDATES if len(success[m]) == top]
    common = sorted(set.intersection(*(set(success[m]) for m in contenders)))
    scores = {}
    for m in contenders:
        if common:
            scores[m] = .5*sum(math.log(mean(indexed[cid, s, m][k] for s in common))
                               for k in ('wirelength', 'cpd_ns'))
    method = min(contenders, key=scores.get) if scores else contenders[0]
    return dict(method=method, selection_seeds=list(selection_seeds), success_counts={m: len(v) for m,v in success.items()},
                contenders=contenders, common_success_seeds=common, log_qor=scores,
                no_common_qor=not common)


def _coverage(rows):
    return dict(planned=len(rows), successes=sum(r['status'] == 'success' for r in rows),
                status_counts=dict(Counter(r['status'] for r in rows)))


def evaluate(rows, conditions, predictions):
    """Evaluate complete native runs and attribute measured QoR to frozen selections."""
    predicted = predictions['conditions']
    indexed = _index(rows, conditions, predicted)
    derived, oracle, heldout, selections = [], {}, {}, defaultdict(dict)
    costs = {}
    for c in conditions:
        cid = c['id']
        pred = predicted[cid]
        oracle[cid] = _best(indexed, cid, SEEDS)
        heldout[cid] = {}
        costs[cid] = dict(candidate_generation_seconds=pred['generation_seconds'],
                          prediction_stage_seconds=pred['prediction_worker_internal_seconds'],
                          scoring_seconds={m: pred['choices'][m]['scoring_seconds'] for m in SCORERS},
                          measured_native_seconds={})
        for seed in SEEDS:
            heldout[cid][str(seed)] = _best(indexed, cid, [s for s in SEEDS if s != seed])
            choices = {m: m for m in NATIVE}
            choices.update({m: pred['choices'][m]['method'] for m in SCORERS})
            choices.update(oracle=oracle[cid]['method'], leave_seed_out=heldout[cid][str(seed)]['method'])
            selections[cid][str(seed)] = choices
            costs[cid]['measured_native_seconds'][str(seed)] = {m: indexed[cid, seed, m]['vpr_seconds'] for m in NATIVE}
            for name, picked in choices.items():
                r = indexed[cid, seed, picked]
                route = r.get('final_route_usage') or {'unique_chanz_nodes': r.get('unique_chanz_nodes')}
                derived.append(dict(condition_id=cid, family=c['family'], architecture=c['architecture'],
                    seed=seed, method=name, status=r['status'], wirelength=r.get('wirelength'), cpd_ns=r.get('cpd_ns'),
                    final_route_usage=route, vpr_seconds=None, total_seconds=None))
    by_method = defaultdict(list)
    for r in derived:
        by_method[r['method']].append(r)
    summary = {method: dict(_coverage(values), by_family={
        family: _coverage([r for r in values if r['family'] == family]) for family in sorted({r['family'] for r in values})},
        by_condition={cid: {str(r['seed']): r['status'] for r in values if r['condition_id'] == cid}
                      for cid in predicted})
        for method, values in by_method.items()}

    comparisons = {}
    controls = ('D', 'j_s1', 'Default')
    # Report the two fixed strengths, the three frozen selectors and both pool diagnostics.
    for method in (*CANDIDATES, *SCORERS, 'oracle', 'leave_seed_out'):
        for base in controls:
            if method == base:
                continue
            result = contrast(derived, method, base)
            result['metrics'].pop('total_seconds')
            result['metrics'].pop('vpr_seconds')
            for metric in result['metrics'].values():
                ratio = metric['ratio']
                metric['percent_change'] = 100*(ratio-1) if ratio is not None else None
            compared = result['metrics']['wirelength']['conditions']
            result['conditions_without_common_success'] = sorted(set(predicted)-set(compared))
            result['qor_coverage'] = dict(conditions=len(compared), planned_conditions=len(conditions),
                families=sorted(result['metrics']['wirelength']['family_ratios']))
            result['planned_seed_pairs'] = len(conditions)*len(SEEDS)
            result['method_successes'] = summary[method]['successes']
            result['baseline_successes'] = summary[base]['successes']
            result['identical_native_input_pairs'] = sum(
                indexed[c['id'], seed, selections[c['id']][str(seed)][method]]['ap_init_sha256'] ==
                indexed[c['id'], seed, selections[c['id']][str(seed)][base]]['ap_init_sha256']
                for c, seed in product(conditions, SEEDS))
            comparisons[method+'/'+base] = result
    return dict(schema='tailcompile-prospective-joint-evaluation-014-1', complete=True,
                logical_native_trials=len(rows), unique_native_manifests=len({r['vpr_manifest'] for r in rows}),
                families=sorted({c['family'] for c in conditions}), seeds=list(SEEDS), primary='E012',
                success_by_method=summary, comparisons=comparisons, selections=dict(selections),
                pool_oracle=oracle, leave_one_seed_out=heldout,
                measured_costs=costs, cost_semantics='Candidate generation excludes scoring. Prediction-stage seconds include all three heads. Native runtime is reported per measured input. Aliases reference the same measurement.',
                oracle_is_deployable=False, cross_seed_selection_is_diagnostic=True,
                wirelength_units='XY tile-span; final CHANZ usage reported separately', Wmin_measured=False,
                independent_family_count=len({c['family'] for c in conditions}), final_blind_test=False)
