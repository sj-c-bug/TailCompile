"""Frozen 015 factorial attribution with independent-family uncertainty.

This module reads compact native results only. It performs no action selection,
training, filesystem access or inference. Failed scientific trials remain in the
coverage denominator; QoR estimates explicitly condition on common success.
"""
from collections import Counter, defaultdict
from itertools import product
import math
import random
from statistics import mean


ARCHES = ('H0Z0', 'H0Z1', 'H1Z0', 'H1Z1')
SEEDS = (15101, 15102, 15103)
CANDIDATES = ('D', 'L', 'Y', 'J', 'R1', 'R2')
METHODS = ('Default',) + CANDIDATES
FACTORIAL = ('D', 'L', 'Y', 'J')
METRICS = ('wirelength', 'cpd_ns', 'final_unique_chanz_nodes')
BOOTSTRAP_SEED = 15015


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _gmean(values):
    values = list(values)
    if not values:
        return None
    if min(values) < 0:
        raise ValueError('015 geometric means require nonnegative values')
    return 0. if 0 in values else math.exp(mean(math.log(v) for v in values))


def _stratum(condition):
    record = condition.get('record', {})
    value = condition.get('training_exposure', record.get('training_exposure',
        condition.get('exposure', record.get('exposure', 'unspecified'))))
    if not isinstance(value, str) or not value:
        raise ValueError('015 training exposure must be an explicit string when supplied')
    return value


def _index(rows, conditions, predictions, seeds, minimum_families):
    if not isinstance(minimum_families, int) or isinstance(minimum_families, bool) or minimum_families < 1:
        raise ValueError('015 minimum_families must be positive')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('015 backend seeds must be unique integers')
    by_id = {c['id']: c for c in conditions}
    families = {c['family'] for c in conditions}
    if (len(by_id) != len(conditions) or len(families) < minimum_families or
            len(conditions) != len(families)*len(ARCHES) or
            {(c['family'], c['architecture']) for c in conditions} != set(product(families, ARCHES))):
        raise ValueError('015 requires independent families with their complete four-architecture panel')
    designs, mapped = {}, {}
    for family in families:
        panel = [c for c in conditions if c['family'] == family]
        if len({(c['design'], c['width']) for c in panel}) != 1:
            raise ValueError('015 family mixes designs or reference widths')
        if len({c.get('split') for c in panel}) != 1 or len({_stratum(c) for c in panel}) != 1:
            raise ValueError('015 family leakage between splits or exposure strata')
        for c in panel:
            if type(c['width']) is not int or c['width'] <= 0 or c.get('width_role', 'reference') != 'reference':
                raise ValueError('015 requires one positive reference width per design')
            if designs.setdefault(c['design'], family) != family:
                raise ValueError('015 the same design cannot count as two independent families')
            digest = c.get('record', {}).get('mapped_blif_sha256')
            if digest and (not _digest(digest) or mapped.setdefault(digest, family) != family):
                raise ValueError('015 mapped design is duplicated across independent families')
    predicted = predictions['conditions']
    if set(predicted) != set(by_id):
        raise ValueError('015 incomplete prediction panel')
    for cid, p in predicted.items():
        if p.get('condition_id', cid) != cid or not _digest(p['device_signature']):
            raise ValueError('015 frozen prediction device/condition mismatch')
        if set(p['candidates']) != set(CANDIDATES):
            raise ValueError('015 frozen candidate panel must contain D/L/Y/J/R1/R2')
        for method, candidate in p['candidates'].items():
            if not all(_digest(candidate.get(k)) for k in ('ap_init_sha256', 'deployed_action_sha256')):
                raise ValueError('015 invalid frozen initializer signature')
            if method.startswith('R') and type(candidate.get('diagnostics', {}).get('matched_control')) is not bool:
                raise ValueError('015 random control requires an explicit matching diagnostic')
    indexed, signatures, manifests = {}, {}, {}
    for row in rows:
        cid, seed, method = row['condition_id'], row['seed'], row['method']
        key = cid, seed, method
        if key in indexed or cid not in by_id or seed not in seeds or method not in METHODS:
            raise ValueError('015 duplicate or unexpected native trial: '+str(key))
        if any(row[k] != by_id[cid][k] for k in ('family', 'design', 'architecture', 'width')):
            raise ValueError('015 native row differs from frozen condition: '+str(key))
        p = predicted[cid]
        if row['device_signature'] != p['device_signature']:
            raise ValueError('015 native device differs from frozen prediction: '+str(key))
        expected = p['candidates'].get(method, {})
        if any(row.get(k) != expected.get(k) for k in ('ap_init_sha256', 'deployed_action_sha256')):
            raise ValueError('015 consumed initializer differs from frozen action: '+str(key))
        signature = row['execution_signature']
        if not _digest(signature) or signatures.setdefault((cid, seed), signature) != signature:
            raise ValueError('015 unmatched backend/netlist/RRG/seed execution: '+str(key))
        status = row['status']
        if status not in ('success', 'unroutable') and not (
                status == 'packing_or_placement_failure' and row.get('confirmed_candidate_infeasibility') is True):
            raise ValueError('015 infrastructure/interface failure is not a scientific outcome: '+str(key))
        if status in ('success', 'unroutable') and (
                row['input_consumed'] != (method != 'Default') or
                method != 'Default' and not row['native_solver_continued']):
            raise ValueError('015 missing native initializer consumer evidence: '+str(key))
        if status == 'success' and not all(_finite(row.get(k)) and row[k] > 0 for k in ('wirelength', 'cpd_ns')):
            raise ValueError('015 successful trial lacks positive WL/CPD: '+str(key))
        if not _finite(row.get('vpr_seconds')) or row['vpr_seconds'] < 0:
            raise ValueError('015 invalid native runtime: '+str(key))
        chanz = _value(row, 'final_unique_chanz_nodes')
        if chanz is not None and (not _finite(chanz) or chanz < 0):
            raise ValueError('015 invalid routed CHANZ usage: '+str(key))
        evidence = (cid, seed, signature, row.get('ap_init_sha256'), status,
                    row.get('wirelength'), row.get('cpd_ns'), row['vpr_seconds'], chanz)
        manifest = row.get('vpr_manifest')
        if not manifest or manifests.setdefault(manifest, evidence) != evidence:
            raise ValueError('015 shared native manifest has conflicting evidence: '+str(key))
        indexed[key] = row
    if set(indexed) != set(product(by_id, seeds, METHODS)):
        raise ValueError('015 incomplete native matrix; scientific failures must remain as rows')
    return indexed, by_id, predicted


def _value(row, metric):
    if metric == 'final_unique_chanz_nodes':
        route = row.get('final_route_usage') or {}
        return route.get('unique_chanz_nodes', row.get('unique_chanz_nodes'))
    return row.get(metric)


def _coverage(rows):
    rows = list(rows)
    return dict(planned=len(rows), successes=sum(r['status'] == 'success' for r in rows),
                status_counts=dict(sorted(Counter(r['status'] for r in rows).items())))


def _bootstrap(values, samples):
    if len(values) < 2 or not samples:
        return None
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    draws = sorted(_gmean(rng.choices(values, k=n)) for _ in range(samples))
    def quantile(p):
        index = (len(draws)-1)*p
        low = int(index)
        return draws[low] + (draws[min(low+1, len(draws)-1)]-draws[low])*(index-low)
    low, high = quantile(.025), quantile(.975)
    return dict(ratio=[low, high], percent_change=[100*(low-1),100*(high-1)],
                samples=samples, seed=BOOTSTRAP_SEED, unit='independent family',
                interpretation='descriptive family-block bootstrap; small panels do not establish significance')


def _aggregate(values, by_id, samples):
    """Primary pooled estimate uses complete four-architecture family blocks."""
    all_families = sorted({c['family'] for c in by_id.values()})
    grouped = defaultdict(list)
    for cid, value in values.items():
        grouped[by_id[cid]['family']].append((by_id[cid]['architecture'], value))
    family_ratios, available, coverage, differences = {}, {}, {}, {}
    for family in all_families:
        entries = grouped[family]
        ratios = [v['ratio'] for _, v in entries if v['ratio'] is not None]
        diffs = [v['absolute_difference'] for _, v in entries if v.get('absolute_difference') is not None]
        available[family] = _gmean(ratios)
        coverage[family] = dict(planned_architectures=len(ARCHES), observed_architectures=len(entries),
                                ratio_architectures=len(ratios), complete=len(ratios)==len(ARCHES))
        if len(ratios) == len(ARCHES):
            family_ratios[family] = _gmean(ratios)
        if diffs:
            differences[family] = mean(diffs)
    ratio = _gmean(family_ratios.values())
    leave_one = {family:_gmean(v for key,v in family_ratios.items() if key != family)
                 for family in family_ratios}
    strata = {}
    for label in sorted({_stratum(c) for c in by_id.values()}):
        families = {c['family'] for c in by_id.values() if _stratum(c)==label}
        subset = {f:v for f,v in family_ratios.items() if f in families}
        value = _gmean(subset.values())
        strata[label] = dict(ratio=value, percent_change=None if value is None else 100*(value-1),
                             effective_family_count=len(subset), planned_family_count=len(families))
    return dict(ratio=ratio, percent_change=None if ratio is None else 100*(ratio-1),
        family_ratios=family_ratios, family_available_architecture_ratios=available,
        family_coverage=coverage, effective_family_count=len(family_ratios),
        planned_family_count=len(all_families), complete_architecture_families=sorted(family_ratios),
        incomplete_architecture_families=sorted(set(all_families)-set(family_ratios)),
        mean_absolute_difference=mean(differences.values()) if differences else None,
        family_absolute_differences=differences, leave_one_family_out=leave_one,
        leave_one_family_out_percent_change={f:None if v is None else 100*(v-1) for f,v in leave_one.items()},
        family_bootstrap_95=_bootstrap(list(family_ratios.values()), samples),
        training_exposure_strata=strata, conditions=values,
        aggregation='seed arithmetic means; architecture geometric mean; independent families equal geometric weight; primary estimate requires all four architectures per family')


def _comparison(indexed, by_id, predicted, seeds, method, baseline, samples):
    required = tuple(m for m in METHODS if m in set(FACTORIAL) | {method, baseline})
    randoms = {m for m in (method, baseline) if m.startswith('R')}
    unmatched = sorted(cid for cid,p in predicted.items() if any(
        not p['candidates'][m]['diagnostics']['matched_control'] for m in randoms))
    common = {cid:[s for s in seeds if all(indexed[cid,s,m]['status']=='success' for m in required)]
              for cid in by_id if cid not in unmatched}
    metrics = {}
    for metric in METRICS:
        values = {}
        for cid, success in common.items():
            usable = [s for s in success if all(_value(indexed[cid,s,m],metric) is not None
                                               for m in (method,baseline))]
            if not usable:
                continue
            numerator = mean(_value(indexed[cid,s,method], metric) for s in usable)
            denominator = mean(_value(indexed[cid,s,baseline], metric) for s in usable)
            ratio = numerator/denominator if denominator > 0 else None
            values[cid] = dict(common_success_seeds=usable, method_seed_mean=numerator,
                               baseline_seed_mean=denominator, ratio=ratio,
                               absolute_difference=numerator-denominator)
        metrics[metric] = _aggregate(values, by_id, samples)
    successes = sum(len(s) for s in common.values())
    return dict(method=method, baseline=baseline, metrics=metrics,
        required_common_success_methods=list(required), common_success_seed_pairs=successes,
        planned_seed_pairs=len(by_id)*len(seeds), conditions_without_common_success=sorted(
            cid for cid,seeds_ in common.items() if not seeds_),
        unmatched_random_control_conditions=unmatched, random_match_required=sorted(randoms),
        method_coverage=_coverage(indexed[cid,s,method] for cid,s in product(by_id,seeds)),
        baseline_coverage=_coverage(indexed[cid,s,baseline] for cid,s in product(by_id,seeds)),
        qor_population='conditional on success of all D/L/Y/J and both compared methods; random controls additionally require a matched budget',
        interpretation='report WL and CPD separately; a delay increase does not automatically reject a wirelength tradeoff')


def _interaction(indexed, by_id, seeds, samples):
    metrics = {}
    for metric in ('wirelength','cpd_ns'):
        values = {}
        for cid in by_id:
            common = [s for s in seeds if all(indexed[cid,s,m]['status']=='success' for m in FACTORIAL)]
            if not common:
                continue
            means = {m:mean(_value(indexed[cid,s,m],metric) for s in common) for m in FACTORIAL}
            effect = math.log(means['J'])-math.log(means['L'])-math.log(means['Y'])+math.log(means['D'])
            values[cid] = dict(common_success_seeds=common, method_seed_means=means,
                log_interaction=effect, ratio=math.exp(effect), absolute_difference=None)
        result = _aggregate(values, by_id, samples)
        result['mean_log_interaction'] = None if result['ratio'] is None else math.log(result['ratio'])
        metrics[metric] = result
    return dict(metrics=metrics, formula='log(J)-log(L)-log(Y)+log(D)',
                interpretation='negative log interaction means the joint edit improves this metric beyond the multiplicative combination of isolated edits; this does not identify a circuit-independent physical cause')


def _mechanism_stage(rows, stage):
    """Aggregate sparse observations using each field's native denominator."""
    statuses, observation_statuses = Counter(), Counter()
    matched, layered, declared = [], [], []
    sums = dict(layer_target_retained_fraction=0., mean_normalized_xy_l1_to_target=0.,
                mean_normalized_xy_l1_to_base=0.)
    denominators = dict.fromkeys(sums, 0)
    missing_metric_atoms = dict.fromkeys(sums, 0)
    for row in rows:
        observation = row.get('residual_observations')
        if not observation:
            observation_statuses['missing_observation'] += 1
            statuses['missing_observation'] += 1
            continue
        observation_statuses[observation.get('status', 'available')] += 1
        expected = observation.get('edited_atoms')
        if expected is not None:
            if type(expected) is not int or expected < 0:
                raise ValueError('015 mechanism edited_atoms must be a nonnegative integer')
            declared.append(expected)
        value = observation.get('stages', {}).get(stage)
        if value is None:
            statuses['not_applicable' if observation.get('status')=='not_applicable' else 'missing_stage'] += 1
            continue
        statuses[value['status']] += 1
        counts = {}
        for field, values in (('matched_atoms',matched),('matched_layer_edited_atoms',layered)):
            count = value.get(field)
            if count is not None:
                if type(count) is not int or count < 0:
                    raise ValueError('015 mechanism denominator must be a nonnegative integer')
                values.append(count)
                counts[field] = count
        if (counts.get('matched_layer_edited_atoms',0) > counts.get('matched_atoms',0) or
                expected is not None and counts.get('matched_atoms',0) > expected):
            raise ValueError('015 mechanism matched counts exceed their native population')
        for field in sums:
            weight = counts.get('matched_layer_edited_atoms' if field=='layer_target_retained_fraction' else 'matched_atoms')
            metric = value.get(field)
            if metric is not None and (not _finite(metric) or metric < 0 or
                    field=='layer_target_retained_fraction' and metric > 1):
                raise ValueError('015 invalid mechanism observation metric')
            if weight is None or weight == 0:
                continue
            if metric is None:
                missing_metric_atoms[field] += weight
            else:
                sums[field] += metric*weight
                denominators[field] += weight
    return dict(trials=len(rows), status_counts=dict(sorted(statuses.items())),
        observation_status_counts=dict(sorted(observation_statuses.items())),
        declared_edited_atoms=sum(declared) if declared else None,
        matched_atoms=sum(matched) if matched else None,
        matched_layer_edited_atoms=sum(layered) if layered else None,
        known_count_trials=dict(declared_edited_atoms=len(declared),matched_atoms=len(matched),
                                matched_layer_edited_atoms=len(layered)),
        **{field:sums[field]/denominators[field] if denominators[field] else None for field in sums},
        metric_denominators=denominators, missing_metric_atoms=missing_metric_atoms)


def _mechanism(rows, families):
    stages = ('post_gp','post_appack_legalization','final')
    by_method = {}
    for method in METHODS:
        selected = [r for r in rows if r['method']==method]
        by_method[method] = dict(stages={s:_mechanism_stage(selected,s) for s in stages},
            by_family={f:dict(stages={s:_mechanism_stage([r for r in selected if r['family']==f],s)
                                      for s in stages}) for f in families})
    return dict(by_method=by_method, stages=list(stages),
        aggregation='descriptive atom-weighted observations, pooled over conditions and backend seeds; not family-equal QoR estimates',
        denominator_semantics='retention uses matched layer-edited atoms with a reported fraction; XY distances use matched edited atoms with the corresponding reported mean; missing evidence contributes no distance or retention value',
        atom_observations_are_not_independent_samples=True,
        retention_is_not_a_success_metric=True,
        interpretation='stage retention describes how the backend uses edits; retention alone does not establish a performance improvement or its cause')


def evaluate(rows, conditions, predictions, seeds=SEEDS, *, minimum_families=6, bootstrap_samples=1000):
    """Audit a complete frozen panel and return compact, JSON-safe attribution."""
    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 2000:
        raise ValueError('015 family bootstrap samples must be between zero and 2000')
    seeds = tuple(seeds)
    indexed, by_id, predicted = _index(rows, conditions, predictions, seeds, minimum_families)
    families = sorted({c['family'] for c in conditions})
    coverage = {m:dict(_coverage(r for r in rows if r['method']==m),
        by_family={f:_coverage(r for r in rows if r['method']==m and r['family']==f) for f in families},
        by_training_exposure={label:_coverage(r for r in rows if r['method']==m and
            _stratum(by_id[r['condition_id']])==label) for label in sorted({_stratum(c) for c in conditions})})
        for m in METHODS}
    pairs = list(dict.fromkeys([(m,b) for m in CANDIDATES for b in ('D','Default') if m!=b]
                              + [('J',b) for b in ('L','Y','R1','R2')]))
    comparisons = {m+'/'+b:_comparison(indexed,by_id,predicted,seeds,m,b,bootstrap_samples) for m,b in pairs}
    return dict(schema='tailcompile-attribution-evaluation-015-1', complete=True,
        logical_native_trials=len(rows), unique_native_manifests=len({r['vpr_manifest'] for r in rows}),
        methods=list(METHODS), families=families, independent_family_count=len(families), seeds=list(seeds),
        success_by_method=coverage, comparisons=comparisons,
        factorial_interaction=_interaction(indexed,by_id,seeds,bootstrap_samples),
        mechanism_observations=_mechanism(rows,families),
        matched_random_conditions={m:[cid for cid,p in predicted.items() if p['candidates'][m]['diagnostics']['matched_control']]
                                   for m in ('R1','R2')},
        primary_metrics=['wirelength','cpd_ns'], secondary_metrics=['final_unique_chanz_nodes'],
        training_exposure_semantics='existing training-exposed families are development controls, not held-out generalization',
        random_controls_semantics='both fixed random proposals are reported; no best-of-random or post-route candidate selection',
        interpretation='averages can support a tradeoff despite individual regressions; family sensitivity and uncertainty must accompany pooled means',
        cost_semantics='shared/reused native timings and parallel collection are not a controlled end-to-end speed benchmark',
        wirelength_units='XY tile-span; routed CHANZ usage reported separately',
        Wmin_measured=False, final_blind_test=False, model_selection_performed=False)
