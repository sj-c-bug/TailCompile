"""016 four-arm QoR comparison; proxy descent and physical QoR are separate evidence."""
from collections import Counter
from itertools import product
from statistics import mean

from attribution_evaluation import (_aggregate, _coverage, _digest, _finite,
                                    _mechanism_stage, _stratum, _value)

METHODS = ('Default', 'D', 'J', 'E')
SEEDS = (16101, 16102, 16103)
ARCHES = ('H0Z0', 'H0Z1', 'H1Z0', 'H1Z1')
METRICS = ('wirelength', 'cpd_ns', 'final_unique_chanz_nodes')


def _index(rows, conditions, predictions, seeds, minimum_families):
    if type(minimum_families) is not int or minimum_families < 1:
        raise ValueError('016 minimum_families must be positive')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('016 seeds must be distinct integers')
    by_id = {c['id']: c for c in conditions}
    families = {c['family'] for c in conditions}
    if (len(by_id) != len(conditions) or len(families) < minimum_families or
        len(conditions) != len(families)*4 or
        {(c['family'], c['architecture']) for c in conditions} != set(product(families, ARCHES))):
        raise ValueError('016 requires complete four-architecture family blocks')
    designs, hashes = {}, {}
    for f in families:
        panel = [c for c in conditions if c['family'] == f]
        if (len({(c['design'], c['width']) for c in panel}) != 1 or
            len({_stratum(c) for c in panel}) != 1 or len({c.get('split') for c in panel}) != 1):
            raise ValueError('016 inconsistent family design/width/exposure')
        for c in panel:
            if type(c['width']) is not int or c['width'] <= 0 or c.get('width_role', 'reference') != 'reference':
                raise ValueError('016 positive reference width required')
            if designs.setdefault(c['design'], f) != f:
                raise ValueError('016 duplicate design across families')
            digest = c.get('record', {}).get('mapped_blif_sha256')
            if digest and (not _digest(digest) or hashes.setdefault(digest, f) != f):
                raise ValueError('016 duplicated/invalid mapped netlist identity')
    predicted = predictions['conditions']
    if set(predicted) != set(by_id):
        raise ValueError('016 incomplete frozen predictions')
    for cid, p in predicted.items():
        if p.get('condition_id', cid) != cid or not _digest(p['device_signature']):
            raise ValueError('016 prediction device/condition mismatch')
        if set(p['candidates']) != {'D', 'J', 'E'}:
            raise ValueError('016 candidate panel must be D/J/E')
        for c in p['candidates'].values():
            if not all(_digest(c.get(k)) for k in ('ap_init_sha256', 'deployed_action_sha256')):
                raise ValueError('016 invalid initializer digest')
    indexed, signatures, manifests = {}, {}, set()
    for row in rows:
        cid, seed, method = row['condition_id'], row['seed'], row['method']
        key = cid, seed, method
        if key in indexed or cid not in by_id or seed not in seeds or method not in METHODS:
            raise ValueError('016 duplicate/unexpected native row')
        if any(row[k] != by_id[cid][k] for k in ('design', 'family', 'architecture', 'width')):
            raise ValueError('016 row differs from frozen condition')
        if row['device_signature'] != predicted[cid]['device_signature']:
            raise ValueError('016 device differs from frozen prediction')
        expected = predicted[cid]['candidates'].get(method, {})
        if any(row.get(k) != expected.get(k) for k in ('ap_init_sha256', 'deployed_action_sha256')):
            raise ValueError('016 consumed initializer differs from prediction')
        signature = row['execution_signature']
        if not _digest(signature) or signatures.setdefault((cid, seed), signature) != signature:
            raise ValueError('016 unpaired backend/seed/device signature')
        status = row['status']
        if status not in ('success', 'unroutable') and not (
            status == 'packing_or_placement_failure' and row.get('confirmed_candidate_infeasibility') is True):
            raise ValueError('016 infrastructure failure cannot be treated as QoR')
        if status in ('success', 'unroutable') and (
            row['input_consumed'] != (method != 'Default') or
            method != 'Default' and not row['native_solver_continued']):
            raise ValueError('016 missing native consumer/continuation evidence')
        if status == 'success' and not all(_finite(row.get(k)) and row[k] > 0 for k in ('wirelength', 'cpd_ns')):
            raise ValueError('016 invalid successful WL/CPD')
        if not _finite(row.get('vpr_seconds')) or row['vpr_seconds'] < 0:
            raise ValueError('016 invalid runtime')
        z = _value(row, 'final_unique_chanz_nodes')
        if z is not None and (not _finite(z) or z < 0):
            raise ValueError('016 invalid routed CHANZ count')
        manifest = row.get('vpr_manifest')
        if not manifest or manifest in manifests:
            raise ValueError('016 every trial must have its own native manifest')
        manifests.add(manifest)
        indexed[key] = row
    if set(indexed) != set(product(by_id, seeds, METHODS)):
        raise ValueError('016 incomplete matrix; retain all scientific failures')
    return indexed, by_id, predicted


def _comparison(indexed, by_id, seeds, method, baseline, samples, required=None):
    required = tuple(required or (method, baseline))
    common = {cid: [s for s in seeds if all(indexed[cid, s, m]['status'] == 'success' for m in required)]
              for cid in by_id}
    metrics = {}
    for metric in METRICS:
        values = {}
        for cid, successes in common.items():
            usable = [s for s in successes if all(_value(indexed[cid, s, m], metric) is not None for m in (method, baseline))]
            if not usable:
                continue
            a = mean(_value(indexed[cid, s, method], metric) for s in usable)
            b = mean(_value(indexed[cid, s, baseline], metric) for s in usable)
            values[cid] = dict(common_success_seeds=usable, method_seed_mean=a,
                baseline_seed_mean=b, ratio=a/b if b > 0 else None, absolute_difference=a-b)
        metrics[metric] = _aggregate(values, by_id, samples)
    return dict(method=method, baseline=baseline, required_common_success_methods=list(required),
        common_success_seed_pairs=sum(map(len, common.values())), planned_seed_pairs=len(by_id)*len(seeds),
        conditions_without_common_success=[c for c, s in common.items() if not s], metrics=metrics,
        method_coverage=_coverage(indexed[c,s,method] for c,s in product(by_id,seeds)),
        baseline_coverage=_coverage(indexed[c,s,baseline] for c,s in product(by_id,seeds)),
        qor_population='pairwise common success; failures retained separately in complete denominators')


def evaluate(rows, conditions, predictions, seeds=SEEDS, *, minimum_families=6, bootstrap_samples=1000):
    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 2000:
        raise ValueError('016 bootstrap_samples must be in [0,2000]')
    seeds = tuple(seeds)
    indexed, by_id, predicted = _index(rows, conditions, predictions, seeds, minimum_families)
    families = sorted({c['family'] for c in conditions})
    pairs = (('D','Default'), ('J','Default'), ('E','Default'), ('J','D'), ('E','D'), ('E','J'))
    comparisons = {m+'/'+b: _comparison(indexed, by_id, seeds, m, b, bootstrap_samples) for m,b in pairs}
    # Same population table supports attribution; standalone D/Default never depends on E/J failures.
    common_table = {m+'/'+b: _comparison(indexed, by_id, seeds, m, b, bootstrap_samples, METHODS) for m,b in pairs}
    stages = ('post_gp', 'post_appack_legalization', 'final')
    diagnostics = {cid:p['candidates']['E'].get('diagnostics', {}) for cid,p in predicted.items()}
    proxy_to_qor = {}
    for cid, d in diagnostics.items():
        initial, final = d.get('initial_objective'), d.get('final_objective')
        wl = comparisons['E/D']['metrics']['wirelength']['conditions'].get(cid, {})
        cpd = comparisons['E/D']['metrics']['cpd_ns']['conditions'].get(cid, {})
        if _finite(initial) and _finite(final) and final > initial + 1e-7:
            raise ValueError('016 refinement receipt reports an increasing objective')
        proxy_to_qor[cid] = dict(initial_objective=initial, final_objective=final,
            proxy_gain=initial-final if _finite(initial) and _finite(final) else None,
            full_net_coverage=d.get('full_net_coverage'), stop_reason=d.get('stop_reason'),
            counters=d.get('counters'),
            common_success_seeds=wl.get('common_success_seeds', []),
            routed_WL_percent_change=None if wl.get('ratio') is None else 100*(wl['ratio']-1),
            routed_CPD_percent_change=None if cpd.get('ratio') is None else 100*(cpd['ratio']-1),
            interpretation='negative routed change is improvement; positive proxy gain alone does not establish it')
    return dict(schema='tailcompile-explainable-evaluation-016-1', complete=True,
        logical_native_trials=len(rows), methods=list(METHODS), seeds=list(seeds), families=families,
        success_by_method={m: dict(_coverage(r for r in rows if r['method']==m),
            by_family={f:_coverage(r for r in rows if r['method']==m and r['family']==f) for f in families}) for m in METHODS},
        comparisons=comparisons, all_four_common_success_comparisons=common_table,
        refinement_diagnostics=diagnostics,
        proxy_to_qor_by_condition=proxy_to_qor,
        refinement_stop_reasons=dict(Counter(d.get('stop_reason', 'missing') for d in diagnostics.values())),
        mechanism={m:{s:_mechanism_stage([r for r in rows if r['method']==m],s) for s in stages} for m in METHODS},
        Wmin_measured=False, new_training=False, metric_priority=['wirelength','cpd_ns'],
        interpretation='proxy descent is not a claim of final VPR QoR; report WL and CPD separately with success coverage; six exposed development families are not a blind test; runtime is secondary')
