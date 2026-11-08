"""017 2x2 objective/search QoR comparison; proxy descent and physical QoR are separate evidence."""
from collections import Counter
from itertools import product
from statistics import mean
import math

from attribution_evaluation import (_aggregate, _coverage, _digest, _finite,
                                    _mechanism_stage, _stratum, _value)

METHODS = ('Default', 'D', 'W', 'WS', 'T', 'TS')
SEEDS = (17101, 17102, 17103)
ARCHES = ('H0Z0', 'H0Z1', 'H1Z0', 'H1Z1')
METRICS = ('wirelength', 'cpd_ns', 'final_unique_chanz_nodes')
FACTORIAL = ('W', 'WS', 'T', 'TS')


def _index(rows, conditions, predictions, seeds, minimum_families):
    if type(minimum_families) is not int or minimum_families < 1:
        raise ValueError('017 minimum_families must be positive')
    if not seeds or len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError('017 seeds must be distinct integers')
    by_id = {c['id']: c for c in conditions}
    families = {c['family'] for c in conditions}
    if (len(by_id) != len(conditions) or len(families) < minimum_families or
        len(conditions) != len(families)*4 or
        {(c['family'], c['architecture']) for c in conditions} != set(product(families, ARCHES))):
        raise ValueError('017 requires complete four-architecture family blocks')
    designs, hashes = {}, {}
    for f in families:
        panel = [c for c in conditions if c['family'] == f]
        if (len({(c['design'], c['width']) for c in panel}) != 1 or
            len({_stratum(c) for c in panel}) != 1 or len({c.get('split') for c in panel}) != 1):
            raise ValueError('017 inconsistent family design/width/exposure')
        for c in panel:
            if type(c['width']) is not int or c['width'] <= 0 or c.get('width_role', 'reference') != 'reference':
                raise ValueError('017 positive reference width required')
            if designs.setdefault(c['design'], f) != f:
                raise ValueError('017 duplicate design across families')
            digest = c.get('record', {}).get('mapped_blif_sha256')
            if digest and (not _digest(digest) or hashes.setdefault(digest, f) != f):
                raise ValueError('017 duplicated/invalid mapped netlist identity')
    predicted = predictions['conditions']
    if set(predicted) != set(by_id):
        raise ValueError('017 incomplete frozen predictions')
    for cid, p in predicted.items():
        if p.get('condition_id', cid) != cid or not _digest(p['device_signature']):
            raise ValueError('017 prediction device/condition mismatch')
        if set(p['candidates']) != {'D', 'W', 'WS', 'T', 'TS'}:
            raise ValueError('017 candidate panel must be D/W/WS/T/TS')
        for c in p['candidates'].values():
            if not all(_digest(c.get(k)) for k in ('ap_init_sha256', 'deployed_action_sha256')):
                raise ValueError('017 invalid initializer digest')
        normalizers = None
        for arm, mode, swaps in (('W','wl',False),('WS','wl',True),('T','joint',False),('TS','joint',True)):
            d=p['candidates'][arm].get('diagnostics',{})
            norms=d.get('normalizers',{})
            if (d.get('setup_complete') is not True or d.get('objective_mode')!=mode or
                d.get('pair_swaps') is not swaps or
                not all(_finite(norms.get(k)) and norms[k]>0 for k in ('wirelength','timing'))):
                raise ValueError('017 missing complete setup or wrong factorial objective/move contract')
            if normalizers is None: normalizers=norms
            elif norms!=normalizers: raise ValueError('017 factorial arms must share frozen D normalizers')
    indexed, signatures, manifests = {}, {}, set()
    for row in rows:
        cid, seed, method = row['condition_id'], row['seed'], row['method']
        key = cid, seed, method
        if key in indexed or cid not in by_id or seed not in seeds or method not in METHODS:
            raise ValueError('017 duplicate/unexpected native row')
        if any(row[k] != by_id[cid][k] for k in ('design', 'family', 'architecture', 'width')):
            raise ValueError('017 row differs from frozen condition')
        if row['device_signature'] != predicted[cid]['device_signature']:
            raise ValueError('017 device differs from frozen prediction')
        expected = predicted[cid]['candidates'].get(method, {})
        if any(row.get(k) != expected.get(k) for k in ('ap_init_sha256', 'deployed_action_sha256')):
            raise ValueError('017 consumed initializer differs from prediction')
        signature = row['execution_signature']
        if not _digest(signature) or signatures.setdefault((cid, seed), signature) != signature:
            raise ValueError('017 unpaired backend/seed/device signature')
        status = row['status']
        if status not in ('success', 'unroutable') and not (
            status == 'packing_or_placement_failure' and row.get('confirmed_candidate_infeasibility') is True):
            raise ValueError('017 infrastructure failure cannot be treated as QoR')
        if status in ('success', 'unroutable') and (
            row['input_consumed'] != (method != 'Default') or
            method != 'Default' and not row['native_solver_continued']):
            raise ValueError('017 missing native consumer/continuation evidence')
        if status == 'success' and not all(_finite(row.get(k)) and row[k] > 0 for k in ('wirelength', 'cpd_ns')):
            raise ValueError('017 invalid successful WL/CPD')
        if not _finite(row.get('vpr_seconds')) or row['vpr_seconds'] < 0:
            raise ValueError('017 invalid runtime')
        z = _value(row, 'final_unique_chanz_nodes')
        if z is not None and (not _finite(z) or z < 0):
            raise ValueError('017 invalid routed CHANZ count')
        manifest = row.get('vpr_manifest')
        if not manifest or manifest in manifests:
            raise ValueError('017 every trial must have its own native manifest')
        manifests.add(manifest)
        indexed[key] = row
    if set(indexed) != set(product(by_id, seeds, METHODS)):
        raise ValueError('017 incomplete matrix; retain all scientific failures')
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
        qor_population=('pairwise common success' if set(required)=={method,baseline} else
                        'common success of all declared required methods')+
                       '; failures retained separately in complete denominators')


def _joint_outcomes(comparison):
    wl = comparison['metrics']['wirelength']
    timing = comparison['metrics']['cpd_ns']
    cases = {}
    for cid in wl['conditions']:
        w, t = wl['conditions'][cid]['ratio'], timing['conditions'][cid]['ratio']
        label = ('both_improve' if w < 1 and t < 1 else
                 'both_regress' if w > 1 and t > 1 else
                 'both_equal' if w == 1 and t == 1 else 'tradeoff_or_one_equal')
        cases[cid] = dict(wirelength_ratio=w, cpd_ratio=t, outcome=label)
    return dict(average_both_improve=wl['ratio'] is not None and timing['ratio'] is not None and
                wl['ratio'] < 1 and timing['ratio'] < 1,
        case_counts=dict(Counter(c['outcome'] for c in cases.values())), cases=cases,
        interpretation='averaged routed WL and CPD must each improve for a joint-gain claim; individual cases may regress')


def _interaction(indexed, by_id, seeds, samples):
    result = {}
    for metric in METRICS:
        values = {}
        for cid in by_id:
            common = [s for s in seeds if all(indexed[cid,s,m]['status']=='success' and
                      _value(indexed[cid,s,m],metric) is not None for m in FACTORIAL)]
            if not common:
                continue
            means = {m:mean(_value(indexed[cid,s,m],metric) for s in common) for m in FACTORIAL}
            if min(means.values()) <= 0:
                continue
            effect = math.log(means['TS'])-math.log(means['T'])-math.log(means['WS'])+math.log(means['W'])
            values[cid] = dict(common_success_seeds=common, method_seed_means=means,
                ratio=math.exp(effect), log_interaction=effect, absolute_difference=None)
        result[metric] = _aggregate(values,by_id,samples)
    return dict(metrics=result, formula='log(TS)-log(T)-log(WS)+log(W)',
        required_common_success_methods=list(FACTORIAL),
        interpretation='negative means the combined change exceeds the multiplicative isolated effects; not a physical-causation proof')


def evaluate(rows, conditions, predictions, seeds=SEEDS, *, minimum_families=6, bootstrap_samples=1000):
    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 2000:
        raise ValueError('017 bootstrap_samples must be in [0,2000]')
    seeds = tuple(seeds)
    indexed, by_id, predicted = _index(rows,conditions,predictions,seeds,minimum_families)
    families = sorted({c['family'] for c in conditions})
    pairs = tuple(dict.fromkeys([(m,b) for m in METHODS[1:] for b in ('Default','D') if m!=b]
                  + [('T','W'),('WS','W'),('TS','T'),('TS','WS'),('TS','W')]))
    comparisons = {m+'/'+b:_comparison(indexed,by_id,seeds,m,b,bootstrap_samples) for m,b in pairs}
    common = {m+'/'+b:_comparison(indexed,by_id,seeds,m,b,bootstrap_samples,FACTORIAL)
              for m,b in (('T','W'),('WS','W'),('TS','T'),('TS','WS'),('TS','W'))}
    stages = ('post_gp','post_appack_legalization','final')
    diagnostics = {cid:{m:p['candidates'][m].get('diagnostics',{}) for m in FACTORIAL}
                   for cid,p in predicted.items()}
    proxy_to_qor = {}
    for cid, arms in diagnostics.items():
        proxy_to_qor[cid] = {}
        for m,d in arms.items():
            initial,final = d.get('initial_objective'),d.get('final_objective')
            if _finite(initial) and _finite(final) and final > initial + 1e-7:
                raise ValueError('017 increasing optimization objective in receipt')
            w = comparisons[m+'/D']['metrics']['wirelength']['conditions'].get(cid,{})
            t = comparisons[m+'/D']['metrics']['cpd_ns']['conditions'].get(cid,{})
            proxy_to_qor[cid][m] = dict(initial_objective=initial,final_objective=final,
                proxy_gain=initial-final if _finite(initial) and _finite(final) else None,
                initial_wirelength=d.get('initial_wirelength'),final_wirelength=d.get('final_wirelength'),
                initial_timing=d.get('initial_timing'),final_timing=d.get('final_timing'),
                stop_reason=d.get('stop_reason'),counters=d.get('counters'),
                common_success_seeds=w.get('common_success_seeds',[]),
                routed_WL_percent_change=None if w.get('ratio') is None else 100*(w['ratio']-1),
                routed_CPD_percent_change=None if t.get('ratio') is None else 100*(t['ratio']-1))
    return dict(schema='tailcompile-factorial-evaluation-017-1',complete=True,
        logical_native_trials=len(rows),methods=list(METHODS),seeds=list(seeds),families=families,
        factors=dict(objective={'W':'wl','WS':'wl','T':'joint','TS':'joint'},
                     swaps={'W':False,'WS':True,'T':False,'TS':True}),
        success_by_method={m:dict(_coverage(r for r in rows if r['method']==m),
            by_family={f:_coverage(r for r in rows if r['method']==m and r['family']==f) for f in families}) for m in METHODS},
        comparisons=comparisons, factorial_common_success_comparisons=common,
        interaction=_interaction(indexed,by_id,seeds,bootstrap_samples),
        joint_outcomes={pair:_joint_outcomes(c) for pair,c in comparisons.items()},
        refinement_diagnostics=diagnostics,proxy_to_qor_by_condition=proxy_to_qor,
        refinement_stop_reasons={m:dict(Counter(d[m].get('stop_reason','missing') for d in diagnostics.values())) for m in FACTORIAL},
        mechanism={m:{s:_mechanism_stage([r for r in rows if r['method']==m],s) for s in stages} for m in METHODS},
        Wmin_measured=False,new_training=False,connection_topology='fixed Hecate SB only',
        metric_priority=['wirelength','cpd_ns'],
        interpretation='joint objective is an interconnect-delay/criticality proxy, not STA or routed CPD; actual routed metrics and failure coverage determine QoR; exposed development families are not a blind test')

