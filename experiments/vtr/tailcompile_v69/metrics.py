"""Frozen-grid width and runtime-first acceptance metrics (standard library only).

Public API:
  aggregate_widths(rows, seeds=FIVE_SEEDS, widths_by_design={name: [widths]}, ...)
  compare_runtime(rows, expected_conditions=[{design, width, seed}], ...)
  choose_on_validation(runtime_report, split='validation')

Rows have design/method/width/seed/status. Runtime rows additionally provide
vpr_seconds (wall_seconds is accepted), generation_seconds, wirelength and cpd
(final_cpd_ns is accepted). Numerical fields may instead be in metrics.
generation_seconds MUST be supplied, including explicit zero for Default; an
unknown initializer cost must never become a free initializer. Rows must come
from one frozen measurement protocol, not mixed serial/concurrent experiments.

Width means are arithmetic means of exactly five per-seed observed minima on
the same frozen grid. They are NOT W90, minima over seeds, or a proof of global
minimum width. Unroutable is a known routing outcome; timeouts/infra/missing are
unknown, never silently converted to an unroutable label. All ratios below are
candidate / baseline (lower is better). Thresholds are provisional policy,
not statistically established tolerances or observed experimental gains.
"""
from __future__ import annotations

from collections import Counter
import math
import statistics


KNOWN_OUTCOMES = frozenset(('success', 'unroutable'))
DEFAULT_BUDGETS = {
    'width_regression': 0.03,
    'wirelength_regression': 0.03,
    'cpd_regression': 0.03,
    'min_total_time_reduction': 0.05,
}


def _number(value, *, zero=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or (value < 0 if zero else value <= 0):
        return None
    return float(value)


def _metric(row, *names, zero=False):
    """Invalid present values cannot silently fall back to another time scope."""
    nested = row.get('metrics') or {}
    for source in (row, nested):
        for name in names:
            if name in source:
                return _number(source[name], zero=zero)
    return None


def _key(row, *, method=True):
    width = row['width']
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError('width must be a positive integer')
    result = (str(row['design']), width, row['seed'])
    return result + (str(row['method']),) if method else result


def _index(rows):
    result = {}
    for row in rows:
        key = _key(row)
        if key in result:
            raise ValueError(f'duplicate trial: {key}; do not mix protocols/attempts')
        result[key] = row
    return result


def _geomean(values):
    values = list(values)
    return math.exp(statistics.mean(map(math.log, values))) if values else None


def _budgets(values):
    result = dict(DEFAULT_BUDGETS)
    if values:
        if set(values) - set(result):
            raise ValueError('unknown budget names')
        result.update(values)
    for name, value in result.items():
        if _number(value, zero=True) is None:
            raise ValueError(f'invalid nonnegative budget: {name}')
    if result['min_total_time_reduction'] >= 1:
        raise ValueError('min_total_time_reduction must be less than one')
    return result


def aggregate_widths(rows, *, seeds, widths_by_design,
                     methods=('default', 'fast')):
    """Retain every planned cell; do not average only the successful seeds."""
    seeds, methods = list(seeds), list(methods)
    if len(seeds) != 5 or len(set(seeds)) != 5:
        raise ValueError('exactly five distinct frozen seeds are required')
    if not methods or len(set(methods)) != len(methods):
        raise ValueError('methods must be nonempty and distinct')
    grids = {}
    for design, widths in widths_by_design.items():
        grid = list(widths)
        if not grid or any(_number(w) is None or not isinstance(w, int) for w in grid):
            raise ValueError('each design needs a nonempty positive integer width grid')
        if len(set(grid)) != len(grid):
            raise ValueError('duplicate width in frozen grid')
        grids[str(design)] = sorted(grid)
    if not grids:
        raise ValueError('empty design plan')
    index = _index(rows)
    expected = {(d, w, s, m) for d, grid in grids.items()
                for w in grid for s in seeds for m in methods}
    if set(index) - expected:
        raise ValueError('scan contains trials outside the frozen plan')
    groups = []
    for design, grid in sorted(grids.items()):
        for method in methods:
            observations = []
            for seed in seeds:
                statuses = {w: str(index.get((design, w, seed, method), {}).get('status', 'missing'))
                            for w in grid}
                successful = [w for w in grid if statuses[w] == 'success']
                minimum = min(successful) if successful else None
                unknown = [w for w in grid if statuses[w] not in KNOWN_OUTCOMES]
                reversals = [w for w in grid if minimum is not None and w > minimum
                             and statuses[w] == 'unroutable']
                observations.append(dict(
                    seed=seed, observed_minimum_width=minimum,
                    left_censored=minimum == grid[0],
                    right_censored=minimum is None and not unknown,
                    no_observed_success=minimum is None,
                    unknown=bool(unknown), unknown_widths=unknown,
                    statuses=[dict(width=w, status=statuses[w]) for w in grid],
                    nonmonotone=bool(reversals), failures_above_first_success=reversals,
                    observed_grid_minimum_known=minimum is not None and not unknown,
                    exact_global_minimum_proven=False))
            complete = all(not r['unknown'] for r in observations)
            eligible = all(r['observed_grid_minimum_known'] for r in observations)
            groups.append(dict(
                design=design, method=method, width_grid=grid, per_seed=observations,
                complete=complete, five_seed_mean_eligible=eligible,
                five_seed_mean_observed_wmin=(statistics.mean(r['observed_minimum_width']
                                               for r in observations) if eligible else None),
                left_censored_seed_count=sum(r['left_censored'] for r in observations),
                right_censored_seed_count=sum(r['right_censored'] for r in observations),
                unknown_seed_count=sum(r['unknown'] for r in observations),
                nonmonotone_seed_count=sum(r['nonmonotone'] for r in observations)))
    return dict(schema='tailcompile-v69-width-1', seeds=seeds, methods=methods,
                complete=all(g['complete'] for g in groups), groups=groups,
                definition='Arithmetic mean of exactly five observed frozen-grid minima; not W90.',
                censoring_policy='Unknown grid outcomes suppress the mean; left-censored means are explicitly observed-grid only.',
                exact_global_minimum_proven=False)


def _width_guard(width_report, design, baseline, candidate, budget):
    if not width_report:
        return dict(known=False, passed=False, ratio=None, reason='width report missing')
    groups = {(g['design'], g['method']): g for g in width_report['groups']}
    a, b = groups.get((design, baseline)), groups.get((design, candidate))
    known = bool(a and b and a.get('five_seed_mean_eligible') and
                 b.get('five_seed_mean_eligible') and a['width_grid'] == b['width_grid'])
    ma = _number(a.get('five_seed_mean_observed_wmin')) if a else None
    mb = _number(b.get('five_seed_mean_observed_wmin')) if b else None
    known = known and ma is not None and mb is not None
    ratio = mb / ma if known else None
    return dict(known=known, passed=known and ratio <= 1 + budget + 1e-12,
                baseline_mean=ma, candidate_mean=mb, ratio=ratio,
                left_censored=bool(a and b and (a['left_censored_seed_count'] or
                                               b['left_censored_seed_count'])),
                interpretation='Observed frozen-grid means, not physical minimum proof')


def _summarize_pairs(pairs, width_guards, budgets):
    known = all(p['outcomes_known'] for p in pairs)
    joint = [p for p in pairs if p['joint_success']]
    valid = [p for p in joint if p['metrics_complete']]
    metric_complete = bool(joint) and len(valid) == len(joint)
    ratios = {key: _geomean(p['ratios'][key] for p in valid)
              for key in ('vpr', 'total', 'cpd', 'wirelength')}
    # Gates use design means; worst-pair ratios are separately visible, never hidden.
    checks = dict(
        all_outcomes_known=bool(pairs) and known,
        no_success_loss=known and not any(p['success_loss'] for p in pairs),
        joint_success_metrics_complete=metric_complete,
        width_within_budget=bool(width_guards) and all(g['passed'] for g in width_guards),
        cpd_within_budget=metric_complete and ratios['cpd'] <= 1 + budgets['cpd_regression'] + 1e-12,
        wirelength_within_budget=metric_complete and ratios['wirelength'] <= 1 + budgets['wirelength_regression'] + 1e-12,
        total_runtime_reduction=metric_complete and ratios['total'] <= 1 - budgets['min_total_time_reduction'] + 1e-12)
    return dict(
        expected_pairs=len(pairs), joint_success_pairs=len(joint),
        measured_joint_success_pairs=len(valid),
        joint_success_coverage=len(joint) / len(pairs) if pairs else 0.,
        unknown_outcome_pairs=sum(not p['outcomes_known'] for p in pairs),
        success_loss_pairs=sum(p['success_loss'] for p in pairs),
        success_gain_pairs=sum(p['success_gain'] for p in pairs),
        baseline_success_count=sum(p['baseline_status'] == 'success' for p in pairs),
        candidate_success_count=sum(p['candidate_status'] == 'success' for p in pairs),
        candidate_over_baseline_geomean=ratios,
        vpr_speedup=1 / ratios['vpr'] if ratios['vpr'] else None,
        generation_inclusive_speedup=1 / ratios['total'] if ratios['total'] else None,
        total_time_reduction=1 - ratios['total'] if ratios['total'] is not None else None,
        worst_pair_ratios={key: max((p['ratios'][key] for p in valid), default=None)
                           for key in ratios},
        checks=checks, passed=all(checks.values()),
        conditional_runtime_warning=('Runtime/QoR ratios use joint-success pairs only; '
            'coverage and all success losses are reported separately. No timing benefit '
            'is asserted for failed, timed-out, missing, or newly rescued pairs.'))


def compare_runtime(rows, *, expected_conditions, baseline='default', candidate='fast',
                    budgets=None, width_report=None):
    """Same design/width/seed pairs; measured VPR + initializer cost, not throughput.

    The caller must freeze hardware/affinity/concurrency/order and initializer cost
    scope. Pass distinct protocols separately. Default may use generation_seconds=0.
    Small regressions are permitted in each design's joint-success geometric-mean
    CPD/WL and five-seed arithmetic-mean width. Runtime is aggregated with equal
    weight per design AFTER within-design aggregation; a design need not itself
    meet the speedup target. Every design must meet quality/evidence guards.
    Outcome unknowns and invalid numeric metrics always block gates.
    """
    if baseline == candidate:
        raise ValueError('baseline and candidate must differ')
    budgets = _budgets(budgets)
    conditions = [_key(c, method=False) for c in expected_conditions]
    if not conditions or len(set(conditions)) != len(conditions):
        raise ValueError('expected conditions must be nonempty and unique')
    index = _index(rows)
    expected = {(*c, m) for c in conditions for m in (baseline, candidate)}
    if set(index) - expected:
        raise ValueError('timing rows outside the frozen paired conditions')
    pairs = []
    for design, width, seed in conditions:
        a, b = (index.get((design, width, seed, method), {})
                for method in (baseline, candidate))
        sa, sb = str(a.get('status', 'missing')), str(b.get('status', 'missing'))
        joint = sa == sb == 'success'
        pair = dict(design=design, width=width, seed=seed,
                    baseline_status=sa, candidate_status=sb,
                    outcomes_known=sa in KNOWN_OUTCOMES and sb in KNOWN_OUTCOMES,
                    joint_success=joint, success_loss=sa == 'success' and sb == 'unroutable',
                    success_gain=sa == 'unroutable' and sb == 'success',
                    metrics_complete=False, ratios=None)
        times = []
        for row in (a, b):
            vpr = _metric(row, 'vpr_seconds', 'wall_seconds')
            generation = _metric(row, 'generation_seconds', zero=True)
            times.append(dict(vpr=vpr, generation=generation,
                              total=vpr + generation if vpr is not None and generation is not None else None,
                              cpd=_metric(row, 'cpd', 'final_cpd_ns'),
                              wirelength=_metric(row, 'wirelength')))
        pair.update(baseline_cost=times[0], candidate_cost=times[1])
        if joint and all(value is not None for t in times for value in t.values()):
            pair['metrics_complete'] = True
            pair['ratios'] = {k: times[1][k] / times[0][k]
                              for k in ('vpr', 'total', 'cpd', 'wirelength')}
        pairs.append(pair)
    by_design = []
    guards = []
    for design in sorted({c[0] for c in conditions}):
        guard = _width_guard(width_report, design, baseline, candidate, budgets['width_regression'])
        guards.append(guard)
        summary = _summarize_pairs([p for p in pairs if p['design'] == design], [guard], budgets)
        ratio = summary['candidate_over_baseline_geomean']['total']
        by_design.append(dict(design=design, width_guard=guard,
            quality_and_evidence_passed=all(value for key, value in summary['checks'].items()
                                            if key != 'total_runtime_reduction'),
            runtime_target_met=summary['checks']['total_runtime_reduction'],
            runtime_regressed=ratio > 1 + 1e-12 if ratio is not None else None,
            **summary))
    aggregate = _summarize_pairs(pairs, guards, budgets)
    aggregate['condition_weighted_candidate_over_baseline_geomean'] = aggregate['candidate_over_baseline_geomean']
    # A design with many measured widths or surviving pairs must not dominate
    # the main runtime conclusion. Missing whole-design metrics are not dropped.
    ratios = {}
    for key in ('vpr', 'total', 'cpd', 'wirelength'):
        values = [d['candidate_over_baseline_geomean'][key] for d in by_design]
        ratios[key] = _geomean(values) if all(v is not None for v in values) else None
    aggregate.update(candidate_over_baseline_geomean=ratios,
        weighting='Equal design weights after within-design joint-success pair geometric means',
        design_count=len(by_design),
        designs_meeting_runtime_target=sum(d['runtime_target_met'] for d in by_design),
        designs_with_runtime_regression=sum(d['runtime_regressed'] is True for d in by_design),
        vpr_speedup=1 / ratios['vpr'] if ratios['vpr'] else None,
        generation_inclusive_speedup=1 / ratios['total'] if ratios['total'] else None,
        total_time_reduction=1 - ratios['total'] if ratios['total'] is not None else None)
    metric_complete = all(d['checks']['joint_success_metrics_complete'] for d in by_design)
    aggregate['checks'].update(
        joint_success_metrics_complete=metric_complete,
        cpd_within_budget=all(d['checks']['cpd_within_budget'] for d in by_design),
        wirelength_within_budget=all(d['checks']['wirelength_within_budget'] for d in by_design),
        total_runtime_reduction=metric_complete and ratios['total'] is not None and
            ratios['total'] <= 1 - budgets['min_total_time_reduction'] + 1e-12,
        all_design_quality_and_evidence_guards_pass=all(d['quality_and_evidence_passed'] for d in by_design))
    aggregate['passed'] = all(aggregate['checks'].values())
    return dict(schema='tailcompile-v69-runtime-first-1', baseline=baseline, candidate=candidate,
                complete=len(index) == len(expected) and bool(width_report and width_report.get('complete')),
                evidence_complete=all(d['checks']['all_outcomes_known'] and
                                      d['checks']['joint_success_metrics_complete'] and
                                      d['width_guard']['known'] for d in by_design),
                budget_status='provisional policy; freeze before inspecting final-test outcomes',
                budgets=budgets, ratio_direction='candidate / baseline; lower is better',
                cost_scope='measured VPR + explicitly supplied generation_seconds; reusable RRG costs reported separately',
                status_counts=dict(Counter(p[f'{side}_status'] for p in pairs for side in ('baseline', 'candidate'))),
                pairs=pairs, by_design=by_design, aggregate=aggregate,
                passed=aggregate['passed'])


def choose_on_validation(report, *, split):
    """Choose one global policy on development/validation, never on final tests.

    This is not a per-test-design oracle selector. Freeze its result before tests;
    final-test gate outcomes describe results and must not change deployment.
    """
    if split not in ('train', 'validation'):
        raise ValueError('selection is permitted only on train/validation, not final-test labels')
    return dict(split=split, selected_method=report['candidate'] if report['passed'] else report['baseline'],
                candidate_accepted=report['passed'], global_policy=True,
                reason='runtime-first gates passed' if report['passed'] else 'one or more gates incomplete/failed')
