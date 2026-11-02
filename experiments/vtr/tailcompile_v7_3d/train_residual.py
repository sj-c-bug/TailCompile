"""011 train only small residual-ranking heads from immutable measured VPR QoR."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

try:
    from .residual_model import ResidualSelector, SCHEMA, OBJECTIVE, ARMS
    from .residual_actions import validate_residual_action
    from .structured_model import StructuredInitializer
    from .train_structured import load_model as load_d
    from .aligned_model import action_digest
    from .aligned_teacher import constraints_digest
    from .train import TrainingInputCache, state_sha256
    from .repository_guard import validate_output_roots
    from .io_utils import raise_if_cancelled
except ImportError:
    from residual_model import ResidualSelector, SCHEMA, OBJECTIVE, ARMS
    from residual_actions import validate_residual_action
    from structured_model import StructuredInitializer
    from train_structured import load_model as load_d
    from aligned_model import action_digest
    from aligned_teacher import constraints_digest
    from train import TrainingInputCache, state_sha256
    from repository_guard import validate_output_roots
    from io_utils import raise_if_cancelled

NORMAL_FAILURE = {'unroutable', 'packing_or_placement_failure', 'decode_failure'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def from_source(path, *, device='cpu', expected_sha256=None, seed=11301, **config):
    digest = sha(path)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError('011 frozen D checkpoint identity changed')
    backbone = load_d(path, device='cpu')
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = ResidualSelector(backbone, seed=seed, source_checkpoint_sha256=digest, **config)
    return model.ranking_to(device)


def save_model(model, path, *, epoch=0):
    path = Path(path); validate_output_roots((path,)); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=SCHEMA, objective_version=OBJECTIVE, config=model.config,
                    backbone_config=model.backbone.config, state_dict=model.state_dict(),
                    backbone_sha256=state_sha256(model.backbone), state_sha256=state_sha256(model),
                    source_checkpoint_sha256=model.source_checkpoint_sha256, epoch=epoch,
                    training_target='measured VPR pair ranking only', tail_weight=0.), path)
    return str(path.resolve())


def load_model(path, *, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('schema') != SCHEMA or payload.get('objective_version') != OBJECTIVE:
        raise ValueError('011 residual checkpoint schema/objective mismatch')
    if payload.get('tail_weight') != 0:
        raise ValueError('011 does not use TailRisk')
    model = ResidualSelector(StructuredInitializer(**payload['backbone_config']),
        source_checkpoint_sha256=payload['source_checkpoint_sha256'], **payload['config'])
    model.load_state_dict(payload['state_dict'])
    if state_sha256(model) != payload['state_sha256'] or state_sha256(model.backbone) != payload['backbone_sha256']:
        raise ValueError('011 residual checkpoint hash mismatch')
    return model.ranking_to(device)


def _group_entries(entries):
    """Accept grouped trials or flat rows; never discard infrastructure failures."""
    result, flat = [], {}
    for entry in entries or []:
        if 'trials' in entry:
            result.append(entry)
        else:
            key = (entry.get('condition_id'), entry.get('arm'), entry.get('candidate_id'))
            if key not in flat:
                flat[key] = {k: v for k, v in entry.items() if k not in ('seed', 'status', 'wirelength', 'cpd_ns')}
                flat[key]['trials'] = []
            else:
                old, new = flat[key].get('action'), entry.get('action')
                old_identity = action_digest(old) if isinstance(old, dict) else str(old)
                new_identity = action_digest(new) if isinstance(new, dict) else str(new)
                if old_identity != new_identity:
                    raise ValueError('011 flat trial rows changed immutable candidate action')
            flat[key]['trials'].append({k: entry.get(k) for k in
                ('seed', 'status', 'wirelength', 'cpd_ns', 'confirmed_candidate_infeasibility')})
    return result+list(flat.values())


def _trial_map(entry):
    rows = entry.get('trials') or []
    if not rows:
        raise ValueError('011 candidate has no measured trials')
    seeds = [r.get('seed') for r in rows]
    if any(isinstance(s, bool) or not isinstance(s, int) for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError('011 duplicate/invalid trial seeds')
    for row in rows:
        if row.get('status') not in NORMAL_FAILURE | {'success'}:
            raise ValueError('011 infrastructure/unknown failures cannot become ranking labels')
        if row['status']=='packing_or_placement_failure' and row.get('confirmed_candidate_infeasibility') is not True:
            raise ValueError('011 unconfirmed packing/placement failures may be backend bugs, not ranking labels')
        if row['status'] == 'success':
            for metric in ('wirelength', 'cpd_ns'):
                value = row.get(metric)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                    raise ValueError('011 successful trial missing positive measured QoR')
    return {r['seed']: r for r in rows}


def pair_delta(a, b):
    """Positive means a better; seed arithmetic means precede log QoR ratios."""
    if set(a) != set(b):
        return None
    sa, sb = [{s for s, row in side.items() if row['status'] == 'success'} for side in (a, b)]
    if len(sa) != len(sb):
        return (1. if len(sa) > len(sb) else -1.), 'success_count'
    if not sa or sa != sb:
        return None
    delta = .5*sum(math.log(statistics.mean(b[s][m] for s in sa)/
                           statistics.mean(a[s][m] for s in sa)) for m in ('wirelength', 'cpd_ns'))
    return (delta, 'qor_log_seed_mean_ratio') if abs(delta) > 1e-10 else None


def prepare_feedback(model, records, entries, cache):
    records_by_id = {r['condition_id']: r for r in records}
    measured = defaultdict(dict); audit = []
    banks, tensors = {}, {}
    for cid, record in records_by_id.items():
        graph = cache.graph(record['graph'])
        context = model.context(graph)
        base = None
        for arm in ARMS:
            base, candidates = model.candidate_bank(graph, arm, base_action=base)
            banks[cid, arm] = (base, candidates)
            tensors[cid, arm] = model.candidate_features(graph, candidates, base, context=context)
    for entry in _group_entries(entries):
        cid, arm, name = entry.get('condition_id'), entry.get('arm'), entry.get('candidate_id')
        if entry.get('split') != 'train' or cid not in records_by_id:
            raise ValueError('011 feedback outside explicit train-only records')
        if arm not in ARMS+('D',):
            raise ValueError('011 feedback arm must be D/L/Y/J')
        if arm == 'D' and name != 'identity':
            raise ValueError('011 D feedback must measure unchanged identity')
        graph = cache.graph(records_by_id[cid]['graph']); action = cache.label(entry['action'])
        digest = action_digest(action)
        if arm != 'D' and action.get('residual_arm') != arm:
            raise ValueError('011 measured residual arm attribution mismatch')
        if action.get('deployed_action_sha256') != digest:
            raise ValueError('011 measured action digest mismatch')
        generation = entry.get('generation') or {}
        for key in ('deployed_action_sha256', 'constraints_sha256'):
            if generation.get(key) is not None and generation[key] != (digest if key == 'deployed_action_sha256' else constraints_digest(graph)):
                raise ValueError('011 collection metadata mismatch: '+key)
        if generation.get('target_sha256') and isinstance(entry['action'], (str, Path)):
            path = Path(entry['action'])
            if cache.base is not None and not path.is_absolute(): path = Path(cache.base)/path
            if sha(path) != generation['target_sha256']:
                raise ValueError('011 measured action bytes changed')
        checkpoint = action.get('source_checkpoint_sha256', action.get('collection_checkpoint_sha256'))
        if checkpoint is not None and checkpoint != model.source_checkpoint_sha256:
            raise ValueError('011 feedback came from a different frozen D checkpoint')
        trial_map = _trial_map(entry)
        target_arms = ARMS if arm == 'D' else (arm,)
        for target_arm in target_arms:
            base, candidates = banks[cid, target_arm]
            matching = [i for i, candidate in enumerate(candidates)
                        if candidate['candidate_id'] == name and action_digest(candidate) == digest]
            if len(matching) != 1:
                raise ValueError('011 measured action is outside regenerated finite candidate bank')
            # One measured D identity is shared between all three banks, but
            # its provenance arm remains the arm that serialized it. The
            # exact unchanged action digest proves identity in each bank.
            validate_residual_action(graph, action, base_action=base,
                                     arm=action.get('residual_arm', target_arm))
            key = cid, target_arm
            if name in measured[key]:
                # D is shared; repeated identical identity is accepted only if
                # every measured trial is exactly the same evidence.
                if measured[key][name]['trials'] != trial_map:
                    raise ValueError('011 duplicate candidate with inconsistent measurements')
            else:
                measured[key][name] = dict(index=matching[0], trials=trial_map, digest=digest)
        audit.append(dict(condition_id=cid, arm=arm, candidate_id=name, action_sha256=digest,
                          measured_seeds=sorted(trial_map), unchanged_regeneration_verified=True))
    raw, counts = defaultdict(list), Counter()
    for key, actions in measured.items():
        if 'identity' not in actions:
            raise ValueError('011 measured bank needs a D identity control')
        expected_seeds = set(actions['identity']['trials'])
        if any(set(action['trials']) != expected_seeds for action in actions.values()):
            raise ValueError('011 candidates require the same complete measured seed set as D')
        for a, b in itertools.combinations(actions.values(), 2):
            result = pair_delta(a['trials'], b['trials'])
            if result is None:
                counts['no_comparable_nonzero_preference'] += 1; continue
            delta, kind = result
            better, worse = (a, b) if delta > 0 else (b, a)
            raw[key].append(dict(better=better['index'], worse=worse['index'], raw_delta=abs(delta), kind=kind))
            counts[kind] += 1
    values = [p['raw_delta'] for pairs in raw.values() for p in pairs if p['kind'] != 'success_count']
    scale = statistics.median(values) if values else 1.
    for pairs in raw.values():
        for pair in pairs:
            pair['weight'] = 1. if pair['kind'] == 'success_count' else min(5., pair['raw_delta']/scale)
    return dict(features=tensors, pairs=raw, audit=audit, counts=dict(counts), scale=scale, measured=measured,
                total_pairs=sum(map(len, raw.values())), banks=banks)


def candidate_pool_diagnostic(prepared, model):
    """Training-pool ceiling versus what the head chose, never test performance."""
    conditions = []
    with torch.no_grad():
        for key, actions in sorted(prepared['measured'].items()):
            control = actions['identity']['trials']
            success_sets = {name: {s for s, row in value['trials'].items() if row['status']=='success'}
                            for name, value in actions.items()}
            common = set.intersection(*success_sets.values()) if success_sets else set()
            candidates = []
            for name, action in actions.items():
                row = dict(candidate_id=name, successful_seeds=sorted(success_sets[name]),
                           success_count=len(success_sets[name]), common_success_q_vs_D=None,
                           common_success_WL_ratio=None, common_success_CPD_ratio=None,
                           pairwise_comparable_preference_vs_D=None)
                result = pair_delta(action['trials'], control)
                if result is not None:
                    row['pairwise_comparable_preference_vs_D'] = dict(
                        delta_positive_means_better=result[0], kind=result[1])
                if common:
                    ratios = {metric: statistics.mean(action['trials'][s][metric] for s in common)/
                              statistics.mean(control[s][metric] for s in common)
                              for metric in ('wirelength','cpd_ns')}
                    row.update(common_success_q_vs_D=.5*sum(math.log(v) for v in ratios.values()),
                               common_success_WL_ratio=ratios['wirelength'],
                               common_success_CPD_ratio=ratios['cpd_ns'])
                candidates.append(row)
            best = min(candidates, key=lambda r: (-r['success_count'],
                r['common_success_q_vs_D'] if r['common_success_q_vs_D'] is not None else 0.,
                r['candidate_id'] != 'identity', r['candidate_id']))
            _, bank = prepared['banks'][key]
            scores = model.score_features(prepared['features'][key], key[1])
            selected = bank[int(scores.argmax())]['candidate_id']
            conditions.append(dict(condition_id=key[0], arm=key[1], candidates=candidates,
                common_success_seeds=sorted(common), offline_best_candidate_id=best['candidate_id'],
                offline_ranking_rule='success count, then QoR on all-candidate common success seeds; identity tie',
                selected_candidate_id=selected, selected_candidate_measured=selected in actions,
                selected_is_offline_best=selected==best['candidate_id'],
                any_comparable_qor_improvement_vs_D=any(
                    r['pairwise_comparable_preference_vs_D'] is not None and
                    r['pairwise_comparable_preference_vs_D']['kind']=='qor_log_seed_mean_ratio' and
                    r['pairwise_comparable_preference_vs_D']['delta_positive_means_better']>0
                    for r in candidates)))
    return dict(population='current measured training candidate pool only',
                offline_best_is_not_deployable_performance=True,
                inference_uses_measured_qor=False, conditions=conditions,
                conditions_with_qor_improvement=sum(r['any_comparable_qor_improvement_vs_D'] for r in conditions),
                effective_pairs_per_arm={arm: sum(len(v) for k,v in prepared['pairs'].items() if k[1]==arm)
                                         for arm in ARMS})


def fit(records, source_checkpoint, output, feedback=None, *, epochs=5, snapshots=(2, 5),
        seed=11301, learning_rate=1e-3, device='cpu', base=None, expected_sha256=None):
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('011 requires explicit train-only records')
    if len({r['condition_id'] for r in records}) != len(records):
        raise ValueError('011 duplicate training condition')
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 5:
        raise ValueError('011 epochs must be bounded in [1,5]')
    snapshots = sorted(set(snapshots))
    if not snapshots or min(snapshots) < 1 or max(snapshots) > epochs or not 0 < learning_rate <= .1:
        raise ValueError('011 invalid snapshots/learning-rate budget')
    output = Path(output); validate_output_roots((output,))
    if (output/'training_report.json').exists():
        raise ValueError('011 training output already completed')
    model = from_source(source_checkpoint, device=device, expected_sha256=expected_sha256, seed=seed)
    cache = TrainingInputCache(base=base, device='cpu', label_entries=32)
    started = time.perf_counter()
    prepared = prepare_feedback(model, records, feedback, cache)
    if not prepared['total_pairs']:
        _write(output/'zero_signal_report.json',dict(schema='tailcompile-v7-residual-zero-signal-011-1',
            training_completed=False, actual_qor_feedback_trained=False,
            reason='no effective measured QoR preferences; validation/test not authorized by this fit',
            source_checkpoint_sha256=model.source_checkpoint_sha256,
            feedback_eligibility=prepared['audit'],pair_counts=prepared['counts'],
            candidate_pool=candidate_pool_diagnostic(prepared,model)))
        raise ValueError('011 no effective measured QoR preferences; training/selection must stop')
    initial, frozen = state_sha256(model), state_sha256(model.backbone)
    optimizer = torch.optim.Adam(model.heads.parameters(), lr=learning_rate)
    rng = random.Random(seed); history, checkpoints = [], []
    for epoch in range(1, epochs+1):
        model.train(); keys = list(prepared['pairs']); rng.shuffle(keys); steps = []
        for key in keys:
            raise_if_cancelled()
            optimizer.zero_grad(set_to_none=True)
            scores = model.score_features(prepared['features'][key], key[1])
            pairs = prepared['pairs'][key]
            loss = sum(pair['weight']*F.softplus(-(scores[pair['better']]-scores[pair['worse']]))
                       for pair in pairs)/len(pairs)
            if not torch.isfinite(loss): raise FloatingPointError('011 nonfinite measured preference loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.heads.parameters()):
                raise FloatingPointError('011 nonfinite ranking gradient')
            gradient = sum(float(p.grad.abs().sum()) for p in model.heads[key[1]].parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.heads.parameters(), 5.)
            optimizer.step()
            steps.append(dict(condition_id=key[0], arm=key[1], loss=float(loss.detach()),
                              supported_feedback_pairs=len(pairs), ranking_gradient_l1=gradient))
        history.append(dict(epoch=epoch, average_loss=statistics.mean(s['loss'] for s in steps), steps=steps))
        if epoch in snapshots:
            checkpoints.append(save_model(model, output/f'epoch_{epoch:04d}.pt', epoch=epoch))
    model.eval(); ranks = {arm: dict(correct=0, ties=0, pairs=0) for arm in ARMS}
    with torch.no_grad():
        for key, pairs in prepared['pairs'].items():
            scores = model.score_features(prepared['features'][key], key[1])
            for pair in pairs:
                difference = float(scores[pair['better']]-scores[pair['worse']])
                ranks[key[1]]['correct'] += int(difference > 0)
                ranks[key[1]]['ties'] += int(difference == 0)
                ranks[key[1]]['pairs'] += 1
    for rank in ranks.values(): rank['accuracy'] = rank['correct']/rank['pairs'] if rank['pairs'] else None
    if frozen != state_sha256(model.backbone) or any(p.grad is not None for p in model.backbone.parameters()):
        raise ValueError('011 frozen D backbone changed during residual training')
    if sha(source_checkpoint) != model.source_checkpoint_sha256:
        raise ValueError('011 frozen D checkpoint bytes changed during training')
    report = dict(schema='tailcompile-v7-residual-training-011-1', objective_version=OBJECTIVE,
        checkpoint=checkpoints[-1], checkpoints=checkpoints, epochs=epochs,
        source_checkpoint=str(source_checkpoint), source_checkpoint_sha256=model.source_checkpoint_sha256,
        initial_state_sha256=initial, final_state_sha256=state_sha256(model), backbone_sha256=frozen,
        backbone_unchanged=True, backbone_trained=False, selector_trained=True,
        actual_qor_feedback_trained=True, feedback_pairs=prepared['total_pairs'], pair_counts=prepared['counts'],
        pair_scale=dict(source='current011 train-only measured candidates', qor=prepared['scale'], weight_cap=5.),
        feedback_eligibility=prepared['audit'], training_rank=ranks,
        candidate_pool=candidate_pool_diagnostic(prepared, model),
        rank_population='measured training pairs; not validation or generalization',
        train_condition_count=len(records), validation_gradients_used=False,
        teacher_labels_used=False, geometry_proxy_used_as_target=False, tail_weight=0., tail_evaluated=False,
        candidate_features_trainable=False, head_context='frozen D GNN logical/device mean and std + candidate descriptors',
        action_space='fixed finite feasible residual bank including unchanged D identity',
        backbone_execution_device='cpu', ranking_device=str(device),
        history=history, training_seconds=time.perf_counter()-started, input_cache=cache.audit())
    _write(output/'training_report.json', report)
    return report


train = fit
train_model = fit
