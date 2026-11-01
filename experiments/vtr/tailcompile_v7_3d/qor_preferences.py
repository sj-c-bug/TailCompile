"""009 train-only preferences and seed-mean QoR evaluation (stdlib only).

Runtime is a matched training control and a reported cost. It is never used as
the primary outcome of the 009 experiment. Historical 007/008 code is immutable.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

POLICY = 'qor-first-20261102'
METHODS = ('B', 'R', 'Q', 'C', 'X', 'T')
LEARNED = METHODS[1:]
NORMAL_FAILURE = {'unroutable', 'packing_or_placement_failure', 'decode_failure',
                  'timeout', 'resource_budget_exceeded'}
EPS = 1e-10


def _positive(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value > 0


def _metric(row, name):
    aliases = {'runtime': ('vpr_seconds', 'wall_seconds', 'vpr_wall_seconds', 'backend_seconds'),
               'wirelength': ('wirelength',), 'cpd_ns': ('cpd_ns',)}
    for key in aliases.get(name, (name,)):
        value = row.get(key, row.get('metrics', {}).get(key))
        if _positive(value):
            return float(value)
    return None


def _payload(action, base=None):
    if isinstance(action, dict):
        return action
    path = Path(action)
    if base is not None and not path.is_absolute():
        path = Path(base)/path
    return json.loads(path.read_text(encoding='utf-8'))


def _xy_identity(entry, base=None):
    generation = entry.get('generation') or {}
    value = generation.get('shared_xy_sha256')
    if value is not None:
        return value
    return _payload(entry['action'], base).get('shared_xy_sha256')


def _layout_key(entry, base=None):
    """Fingerprint only the decisions frozen-coordinate actors can change."""
    payload = _payload(entry['action'], base)
    if 'layer' not in payload or 'assignment' not in payload:
        raise ValueError('complete layer/assignment action required')
    value = [payload['layer'], payload['assignment']]
    return hashlib.sha256(json.dumps(value, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def _pair_delta(a, b, objective):
    """Positive means a is better; failure runtimes never enter this function."""
    sa = {s for s, row in a['by_seed'].items() if row['status'] == 'success'}
    sb = {s for s, row in b['by_seed'].items() if row['status'] == 'success'}
    if len(sa) != len(sb):
        return (1. if len(sa) > len(sb) else -1.), 'success_count', None
    if not sa or sa != sb:
        return None
    if objective == 'runtime':
        values = [(_metric(a['by_seed'][s], 'runtime'), _metric(b['by_seed'][s], 'runtime')) for s in sa]
        if any(x is None or y is None for x, y in values):
            return None
        delta = statistics.mean(math.log(y/x) for x, y in values)
        return delta, 'runtime_log_ratio', None
    components = {}
    for metric in ('wirelength', 'cpd_ns'):
        av = [_metric(a['by_seed'][s], metric) for s in sa]
        bv = [_metric(b['by_seed'][s], metric) for s in sa]
        if any(x is None for x in av+bv):
            return None
        components[metric] = math.log(statistics.mean(bv)/statistics.mean(av))
    return .5*sum(components.values()), 'qor_log_mean_ratio', components


def build_preferences(feedback_rows, *, objective='qor', scales=None,
                      allowed_xy_sha256=None, base=None):
    """Build immutable action references and bounded train-only pair weights.

    The caller filters legacy entries without a ``split`` to the declared train
    conditions; an explicit validation/test entry is rejected. When ``scales``
    is omitted, both objective scales are fitted before actor-support filtering,
    so R/Q/C and X/T receive the same scale for the same collection. Train-only
    provenance/constraint validation of action bytes is also required by trainer.
    """
    if objective not in ('qor', 'runtime'):
        raise ValueError('objective must be qor or runtime')
    grouped, audit = defaultdict(list), []
    seen = set()
    for entry in feedback_rows or []:
        if entry.get('split', 'train') != 'train':
            raise ValueError('preferences and scale fitting require train-only entries')
        if entry.get('action') is None:
            continue
        condition = entry.get('condition_id')
        if not isinstance(condition, str) or not condition:
            raise ValueError('nonempty exact condition_id required')
        name = entry.get('candidate_id')
        if name is not None:
            reference = entry['action'] if isinstance(entry['action'], str) else hashlib.sha256(
                json.dumps(entry['action'], sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()
            key = condition, name, entry.get('collection_id'), reference
            if key in seen:
                raise ValueError('duplicate candidate within condition')
            seen.add(key)
        rows = entry.get('trials') or []
        seeds = [row.get('seed') for row in rows]
        reason = None
        if not rows or any(not isinstance(s, int) or isinstance(s, bool) for s in seeds) or len(set(seeds)) != len(seeds):
            reason = 'empty_or_duplicate_trial_seeds'
        elif any(row.get('status') not in NORMAL_FAILURE | {'success'} for row in rows):
            reason = 'infrastructure_or_unknown_status'
        elif any(row['status'] == 'success' and
                 any(_metric(row, metric) is None for metric in
                     (('wirelength', 'cpd_ns') if objective == 'qor' else ('runtime',))) for row in rows):
            reason = 'missing_measured_'+objective+'_metric'
        item_audit = dict(condition_id=condition, candidate_id=name, eligible=reason is None,
                          reason=reason, split=entry.get('split'), legacy_split_filtered_by_caller='split' not in entry)
        audit.append(item_audit)
        if reason is None:
            grouped[condition].append(dict(entry=entry, by_seed={row['seed']: row for row in rows}, audit=item_audit))
    raw = {kind: [] for kind in ('qor', 'runtime')}
    for actions in grouped.values():
        for i, a in enumerate(actions):
            for b in actions[i+1:]:
                if set(a['by_seed']) != set(b['by_seed']):
                    continue
                for kind in raw:
                    result = _pair_delta(a, b, kind)
                    if result and result[1] != 'success_count' and abs(result[0]) > EPS:
                        raw[kind].append(abs(result[0]))
    if scales is None:
        scales = dict(schema='tailcompile-qor-preference-scales-009-1',
            qor=statistics.median(raw['qor']) if raw['qor'] else 1.,
            runtime=statistics.median(raw['runtime']) if raw['runtime'] else 1.,
            fit_train_only=True, source='bootstrap legacy train feedback; frozen before new collection',
            fitted_pair_counts={key: len(values) for key, values in raw.items()},
            zero_signal_fallback={key: not values for key, values in raw.items()})
    else:
        scales = copy.deepcopy(scales)
        if any(not _positive(scales.get(kind)) for kind in ('qor', 'runtime')):
            raise ValueError('positive finite frozen qor/runtime scales required')
        if scales.get('fit_train_only') is False:
            raise ValueError('non-training scales forbidden')
    support_filtered = defaultdict(list)
    for condition, actions in grouped.items():
        for item in actions:
            if allowed_xy_sha256 is not None:
                try:
                    identity = _xy_identity(item['entry'], base)
                    item['layout_key'] = _layout_key(item['entry'], base)
                except (OSError, ValueError, TypeError, KeyError):
                    item['audit'].update(eligible=False, reason='missing_or_invalid_action_support_identity')
                    continue
                if identity != allowed_xy_sha256:
                    item['audit'].update(eligible=False, reason='outside_frozen_xy_support')
                    continue
            support_filtered[condition].append(item)
    pairs_by_condition, counts = {}, Counter()
    for condition in grouped:
        actions = support_filtered[condition]
        pairs = []
        for i, a in enumerate(actions):
            for b in actions[i+1:]:
                if set(a['by_seed']) != set(b['by_seed']):
                    counts['different_seed_sets'] += 1
                    continue
                if allowed_xy_sha256 is not None and a['layout_key'] == b['layout_key']:
                    counts['same_layer_leaf_no_frozen_actor_signal'] += 1
                    continue
                result = _pair_delta(a, b, objective)
                if not result:
                    counts['no_comparable_success_population'] += 1
                    continue
                delta, kind, components = result
                if abs(delta) <= EPS:
                    counts['tied_objective'] += 1
                    continue
                better, worse = (a, b) if delta > 0 else (b, a)
                weight = 1. if kind == 'success_count' else min(5., abs(delta)/scales[objective])
                pairs.append(dict(better=better['entry']['action'], worse=worse['entry']['action'],
                    better_candidate_id=better['entry'].get('candidate_id'), worse_candidate_id=worse['entry'].get('candidate_id'),
                    weight=weight, kind=kind, raw_delta=abs(delta),
                    components_for_a_over_b=None if components is None else {k: -v for k, v in components.items()},
                    shared_trial_seeds=sorted(a['by_seed'])))
                counts[kind] += 1
        pairs_by_condition[condition] = pairs
    return dict(schema='tailcompile-qor-preferences-009-1', policy=POLICY, objective=objective,
        pairs_by_condition=pairs_by_condition, audit=audit, scales=scales,
        pair_counts=dict(counts), total_pairs=sum(map(len, pairs_by_condition.values())),
        allowed_xy_sha256=allowed_xy_sha256, weight_cap=5., success_weight=1.,
        feedback_coefficient=.2, actor_eligibility='same frozen XY/decoder' if allowed_xy_sha256 else 'all constraint-validated collection policies')


def _gm(values):
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def _aggregate(condition_rows, metric):
    by_arch = defaultdict(list)
    for row in condition_rows:
        value = row['metrics'][metric]['ratio']
        if value is not None:
            by_arch[row['family'], row['architecture']].append(value)
    arch = {key: _gm(values) for key, values in by_arch.items()}
    by_family = defaultdict(list)
    for (family, architecture), value in arch.items():
        by_family[family].append(value)
    families = {family: _gm(values) for family, values in by_family.items()}
    ratio = _gm(list(families.values()))
    bootstrap = []
    if families:
        rng = random.Random(9301)
        values = list(families.values())
        bootstrap = sorted(_gm([rng.choice(values) for _ in values]) for _ in range(1024))
    return dict(ratio=ratio, gain=None if ratio is None else 1-ratio, average_positive=None if ratio is None else ratio < 1,
        family_ratios=families, architecture_ratios={f'{f}/{a}': value for (f, a), value in arch.items()},
        valid_conditions=sum(row['metrics'][metric]['ratio'] is not None for row in condition_rows),
        family_count=len(families), descriptive_family_bootstrap_gain95=None if not bootstrap else [1-bootstrap[998], 1-bootstrap[25]],
        small_family_interval_is_not_generalization_proof=True)


def _classification(wl, cpd):
    if wl is None or cpd is None:
        return 'insufficient_qor'
    if wl <= 1 and cpd <= 1:
        return 'average_joint_improvement' if min(wl, cpd) < 1 else 'no_average_change'
    if wl >= 1 and cpd >= 1:
        return 'average_joint_regression'
    return 'qor_tradeoff'


def evaluate(rows, *, methods=METHODS, expected=None):
    """Arithmetic seed means first; equal architecture and family geometric ratios."""
    rows = list(rows)
    keys = [(r['condition_id'], r['seed'], r['method']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate measured task')
    if any(r['method'] not in methods for r in rows):
        raise ValueError('unexpected method')
    lookup = dict(zip(keys, rows))
    expected = None if expected is None else set(tuple(k) for k in expected)
    expected_seeds = defaultdict(set)
    for condition, seed, method in expected if expected is not None else keys:
        expected_seeds[condition].add(seed)
    contrasts = [('T', 'B'), ('X', 'B'), ('C', 'B'), ('Q', 'B'), ('R', 'B'),
                 ('Q', 'R'), ('C', 'Q'), ('X', 'C'), ('T', 'X')]
    metric_names = ('wirelength', 'cpd_ns', 'total_seconds', 'vpr_seconds')
    comparisons = []
    for method, baseline in contrasts:
        if method not in methods or baseline not in methods:
            continue
        roles = sorted({r.get('width_role', 'reference') for r in rows if r['method'] in (method, baseline)})
        for role in roles:
            selected = [r for r in rows if r['method'] == method and r.get('width_role', 'reference') == role]
            conditions = sorted({r['condition_id'] for r in selected} | {r['condition_id'] for r in rows
                if r['method'] == baseline and r.get('width_role', 'reference') == role})
            condition_rows, status = [], Counter()
            for condition in conditions:
                seeds = sorted(expected_seeds[condition])
                both, success = [], []
                exemplar = next(r for r in rows if r['condition_id'] == condition)
                for seed in seeds:
                    a, b = lookup.get((condition, seed, method)), lookup.get((condition, seed, baseline))
                    status['method_'+(a['status'] if a else 'missing')] += 1
                    status['baseline_'+(b['status'] if b else 'missing')] += 1
                    if a is not None and b is not None:
                        both.append((a, b))
                        if a['status'] == b['status'] == 'success':
                            success.append((a, b))
                metric_rows = {}
                for metric in metric_names:
                    a_values, b_values = [], []
                    for a, b in success:
                        a_values.append(_metric(a, metric)); b_values.append(_metric(b, metric))
                    valid = bool(success) and all(v is not None for v in a_values+b_values)
                    av, bv = (statistics.mean(a_values), statistics.mean(b_values)) if valid else (None, None)
                    metric_rows[metric] = dict(method_seed_mean=av, baseline_seed_mean=bv, ratio=av/bv if valid else None)
                wl, cp = [metric_rows[m]['ratio'] for m in ('wirelength', 'cpd_ns')]
                condition_rows.append(dict(condition_id=condition, family=exemplar['family'], architecture=exemplar['architecture'],
                    planned_seeds=seeds, observed_both=len(both), common_success_seeds=[a['seed'] for a, b in success],
                    common_success_count=len(success), full_success_coverage=bool(seeds) and len(success) == len(seeds),
                    metrics=metric_rows, classification=_classification(wl, cp),
                    q=None if wl is None or cp is None else .5*(math.log(wl)+math.log(cp))))
            aggregate = {metric: _aggregate(condition_rows, metric) for metric in metric_names}
            populations = [{r['condition_id'] for r in condition_rows if r['metrics'][metric]['ratio'] is not None}
                           for metric in ('wirelength', 'cpd_ns')]
            consistent = bool(populations[0]) and populations[0] == populations[1]
            wl, cp = [aggregate[m]['ratio'] if consistent else None for m in ('wirelength', 'cpd_ns')]
            full = bool(condition_rows) and all(r['full_success_coverage'] for r in condition_rows)
            complete = expected is not None and set(keys) == expected
            qor_complete = consistent and len(populations[0]) == len(condition_rows)
            comparisons.append(dict(method=method, baseline=baseline, width_role=role,
                observed_method_tasks=len(selected), planned_paired_tasks=sum(len(r['planned_seeds']) for r in condition_rows),
                jointly_successful=sum(r['common_success_count'] for r in condition_rows), status_counts=dict(status),
                estimate_population=('all planned paired seeds' if complete else 'all observed paired seeds; expected matrix incomplete or unprovided')
                    if full else 'conditional common-success seed subset',
                full_success_coverage=full, consistent_qor_population=consistent, full_qor_metric_coverage=qor_complete, metrics=aggregate,
                classification=_classification(wl, cp), joint_q=None if wl is None or cp is None else .5*(math.log(wl)+math.log(cp)),
                full_population_joint_improvement_claim=complete and full and qor_complete and _classification(wl, cp) == 'average_joint_improvement',
                conditions=condition_rows,
                qor_condition_wins=sum(r['q'] < 0 for r in condition_rows if r['q'] is not None),
                qor_condition_losses=sum(r['q'] > 0 for r in condition_rows if r['q'] is not None)))
    present = set(keys)
    return dict(schema='tailcompile-qor-evaluation-009-1', policy=POLICY,
        complete=None if expected is None else present == expected, completed=len(rows),
        missing=None if expected is None else sorted(expected-present),
        unexpected=None if expected is None else sorted(present-expected),
        status_counts=dict(Counter(r['status'] for r in rows)), comparisons=comparisons,
        primary_comparison=['T', 'B', 'reference'], primary_metrics=['wirelength', 'cpd_ns'],
        seed_aggregation='arithmetic mean on the same successful seeds before metric ratios',
        architecture_and_family_aggregation='equal geometric means', runtime_is_cost=True,
        Wmin_measured=False, generalization_claim=False, per_case_qor_veto=False)


def choose_epoch(rows, epochs=(5, 10), *, methods=LEARNED):
    """Choose one epoch from X/T success coverage, then common-population QoR."""
    rows = list(rows)
    keys = [(r['epoch'], r['method'], r['condition_id'], r['seed']) for r in rows]
    if len(keys) != len(set(keys)):
        raise ValueError('duplicate checkpoint validation task')
    lookup = dict(zip(keys, rows))
    controls = {(r['condition_id'], r['seed']) for r in rows if r['method'] == 'B' and r['epoch'] == 0}
    expected = {(epoch, method, c, s) for epoch in epochs for method in methods for c, s in controls}
    expected |= {(0, 'B', c, s) for c, s in controls}
    if not controls or set(keys) != expected or not {'X', 'T'} <= set(methods):
        raise ValueError('checkpoint selection incomplete or unexpected; missing is not a loss')
    if any(r['status'] == 'success' and any(_metric(r, m) is None for m in ('wirelength','cpd_ns')) for r in rows):
        raise ValueError('checkpoint selection has missing QoR metrics on successful task')
    common = [(c, s) for c, s in sorted(controls) if lookup[0, 'B', c, s]['status'] == 'success'
        and all(lookup[epoch, method, c, s]['status'] == 'success' for epoch in epochs for method in ('X', 'T'))]
    scores = []
    for epoch in epochs:
        chosen_rows = [lookup[0, 'B', c, s] for c, s in common]
        chosen_rows += [lookup[epoch, method, c, s] for c, s in common for method in ('X', 'T')]
        evaluation = evaluate(chosen_rows, methods=('B', 'X', 'T'))
        qs = [r['joint_q'] for r in evaluation['comparisons'] if r['baseline'] == 'B' and r['method'] in ('X', 'T')]
        score = statistics.mean(qs) if len(qs) == 2 and all(q is not None for q in qs) else None
        scores.append(dict(epoch=epoch,
            successes=sum(lookup[epoch, method, c, s]['status'] == 'success' for method in ('X', 'T') for c, s in controls),
            common_success_q=score, common_success_qor_ratio=None if score is None else math.exp(score),
            common_success_tasks=len(common)))
    best = min(scores, key=lambda r: (-r['successes'], float('inf') if r['common_success_q'] is None else r['common_success_q'], r['epoch']))
    return dict(schema='tailcompile-qor-selection-009-1', policy=POLICY, epoch=best['epoch'], scores=scores,
        common_success_conditions=len(common), common_success_design_conditions=len({c for c, s in common}),
        performance_selection_completed=best['common_success_q'] is not None,
        selection='shared epoch for all arms; X/T success count then common-population mean QoR q; earliest tie',
        runtime_used_for_selection=False,
        no_common_success_fallback='maximum X/T success coverage, then earlier epoch' if not common else None)
