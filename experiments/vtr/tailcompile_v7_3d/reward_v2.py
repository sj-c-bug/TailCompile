"""Non-saturating, constrained candidate preference; never imputes failure time.

This policy is for a new cohort. The frozen train.py policy is kept as the
matched control. Acceptance still uses measured cold totals and original QoR
limits; a learning preference is never an acceptance verdict.
"""
from __future__ import annotations
import itertools
import math
from train import INFRASTRUCTURE, NORMAL_FAILURE, _metric

POLICY = 'guarded-log-runtime-preference-2'


def outcome_reward(candidate_trials, default_trials, *, quality_limit=1.03):
    if (len({r['seed'] for r in candidate_trials}) != len(candidate_trials) or
            len({r['seed'] for r in default_trials}) != len(default_trials)):
        raise ValueError('duplicate feedback seed')
    candidate = {r['seed']: r for r in candidate_trials}
    default = {r['seed']: r for r in default_trials}
    if not candidate or candidate.keys() != default.keys():
        return dict(eligible=False, reason='unmatched_seed_set')
    observations, logs, excesses = [], [], []
    gains = losses = 0
    for seed in sorted(candidate):
        a, b = candidate[seed], default[seed]
        sa, sb = a.get('status'), b.get('status')
        normal = NORMAL_FAILURE | {'success'}
        if sa in INFRASTRUCTURE or sb in INFRASTRUCTURE or sa not in normal or sb not in normal:
            return dict(eligible=False, reason='infrastructure_or_unknown_outcome')
        obs = dict(seed=seed, status=sa, default_status=sb, log_runtime_advantage=None,
                   cpd_ratio=None, wirelength_ratio=None)
        if sa == sb == 'success':
            t, base = _metric(a, 'total_seconds'), _metric(b, 'total_seconds')
            if t is None or base is None or min(t, base) <= 0:
                return dict(eligible=False, reason='missing_measured_total_runtime')
            ratios = []
            for key, aliases in (('cpd_ratio', ('cpd_ns', 'cpd')),
                                 ('wirelength_ratio', ('wirelength', 'total_wirelength'))):
                aq, bq = _metric(a, *aliases), _metric(b, *aliases)
                if aq is None or bq is None or min(aq, bq) <= 0:
                    return dict(eligible=False, reason='missing_measured_quality')
                obs[key] = aq/bq
                ratios.append(aq/bq)
            # Log subtraction avoids overflow of base/t. No early clipping.
            value = math.log(base) - math.log(t)
            obs['log_runtime_advantage'] = value
            logs.append(value)
            excesses.append(max(0., *(math.log(v/quality_limit) for v in ratios)))
        elif sa == 'success':
            gains += 1
        elif sb == 'success':
            losses += 1
        observations.append(obs)
    if not logs and not gains and not losses:
        return dict(eligible=False, reason='no_comparable_success_or_success_change')
    violation = max(excesses, default=0.)
    # Categorical safety loss stays adverse, but ranking below retains severity.
    raw = sum(logs)/len(logs) if logs else None
    signed = (-1. if losses or violation > 0 else 1. if gains else
              math.tanh(raw / math.log(1.25)))
    return dict(eligible=True, policy=POLICY, advantage=signed,
                raw_log_runtime_advantage=raw, success_gains=gains, success_losses=losses,
                success_count=sum(r['status'] == 'success' for r in candidate.values()),
                quality_log_excess=violation, quality_protection=violation <= 0,
                success_protection=losses == 0, observations=observations,
                comparable_seed_count=len(logs), failure_runtime_imputed=False)


def preference(a, b, *, quality_tolerance=1e-6, runtime_deadband=math.log(1.02)):
    """Return +1/-1/0, with deterministic priority and a 2% runtime deadband.

    Avoid comparing times from different successful seed subsets. Success-loss
    protection precedes QoR, then success count, then constraint severity or
    measured runtime. Improving a violating candidate does NOT mark it safe.
    """
    if not a.get('eligible') or not b.get('eligible'):
        return 0
    if [(r['seed'],r['default_status']) for r in a['observations']] != [(r['seed'],r['default_status']) for r in b['observations']]:
        return 0
    def compare(x, y):
        return int(x > y) - int(x < y)
    for key in ('success_protection', 'quality_protection', 'success_count'):
        order = compare(a[key], b[key])
        if order:
            return order
    if not a['quality_protection']:
        delta = b['quality_log_excess'] - a['quality_log_excess']
        if abs(delta) > quality_tolerance:
            return 1 if delta > 0 else -1
    by_a = {r['seed']: r for r in a['observations']}
    by_b = {r['seed']: r for r in b['observations']}
    if by_a.keys() != by_b.keys():
        return 0
    # Require identical measured runtime populations; no failure interpolation.
    keys_a = {s for s, r in by_a.items() if r['log_runtime_advantage'] is not None}
    keys_b = {s for s, r in by_b.items() if r['log_runtime_advantage'] is not None}
    if not keys_a or keys_a != keys_b:
        return 0
    delta = sum(by_a[s]['log_runtime_advantage'] - by_b[s]['log_runtime_advantage']
                for s in keys_a)/len(keys_a)
    return (1 if delta > 0 else -1) if abs(delta) > runtime_deadband else 0


def pairs(actions):
    result = []
    for i, j in itertools.combinations(range(len(actions)), 2):
        order = preference(actions[i], actions[j])
        if order:
            result.append((i, j) if order > 0 else (j, i))
    return result


def feedback_loss(model, data, actions, *, coordinate_sigma=.2, preference_weight=.5):
    import torch
    usable = [r for r in actions if r.get('eligible')]
    if not usable:
        raise ValueError('no eligible feedback actions')
    ll = torch.stack([model.action_log_probability(data, r['action'], coordinate_sigma=coordinate_sigma)
                      for r in usable])
    reward = ll.new_tensor([r['advantage'] for r in usable])
    if not torch.isfinite(reward).all():
        raise ValueError('nonfinite signed reward')
    signed = -(reward * ll).mean()
    ranking = pairs(usable)
    preference_loss = (torch.stack([torch.nn.functional.softplus(-(ll[i]-ll[j]))
                                   for i, j in ranking]).mean() if ranking else ll.new_zeros(()))
    return signed + preference_weight*preference_loss, dict(
        signed_action_loss=float(signed.detach()), pairwise_loss=float(preference_loss.detach()),
        preference_pairs=len(ranking), positive_actions=int((reward > 0).sum()),
        negative_actions=int((reward < 0).sum()), actual_generator_updated=True,
        policy=POLICY, runtime_pair_deadband_ratio=1.02)
