"""Bounded paired training for 007; old 005/006 trainers stay immutable.

Shared pretraining -> paired five-epoch candidate preparation -> one feedback
round. Raw VPR time preferences have no per-case 3% quality gate or -1 clipping.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

try:
    from .layer_model import ExplicitLayerInitializer, SCHEMA, generate
    from .train import TrainingInputCache, state_sha256, INFRASTRUCTURE, NORMAL_FAILURE
    from .io_utils import raise_if_cancelled
    from .repository_guard import validate_output_roots
except ImportError:
    from layer_model import ExplicitLayerInitializer, SCHEMA, generate
    from train import TrainingInputCache, state_sha256, INFRASTRUCTURE, NORMAL_FAILURE
    from io_utils import raise_if_cancelled
    from repository_guard import validate_output_roots


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _check_records(records):
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('nonempty explicitly train-only records required')
    if len({r['condition_id'] for r in records}) != len(records):
        raise ValueError('duplicate training condition')


def save_model(model, path, *, epoch, arm):
    torch.save(dict(schema=SCHEMA, config=model.config, state_dict=model.state_dict(),
        shared_xy_frozen=model.shared_xy_frozen, shared_xy_sha256=state_sha256(model.shared_xy),
        epoch=epoch, arm=arm, generator_sha256=state_sha256(model)), path)
    return str(Path(path).resolve())


def load_model(path, *, device='cpu'):
    value = torch.load(path, map_location=device, weights_only=False)
    if value.get('schema') != SCHEMA:
        raise ValueError('explicit layer checkpoint schema mismatch')
    model = ExplicitLayerInitializer(**value['config']).to(device)
    model.load_state_dict(value['state_dict'])
    if state_sha256(model) != value.get('generator_sha256'):
        raise ValueError('explicit layer generator checkpoint hash mismatch')
    if state_sha256(model.shared_xy) != value['shared_xy_sha256']:
        raise ValueError('shared XY checkpoint hash mismatch')
    if value.get('shared_xy_frozen'):
        model.freeze_shared_xy()
    return model


def _metric_time(row):
    for name in ('vpr_seconds', 'wall_seconds', 'vpr_wall_seconds', 'backend_seconds'):
        value = row.get(name, row.get('metrics', {}).get(name))
        if value is not None and math.isfinite(float(value)) and float(value) > 0:
            return float(value)
    return None


def feedback_preferences(entries):
    """Comparable action pairs, grouped by exact condition and seed set.

Status changes are categorical preferences, never fabricated failure runtimes.
Two successful actions on the same seeds use raw log backend-time ratios.
CPD/WL remain in the original entries but are not used as rejection thresholds.
"""
    grouped, audit = {}, []
    for entry in entries or []:
        if entry.get('action') is None:
            continue  # Default is a control, not an action likelihood.
        rows = entry.get('trials', [])
        seeds = [int(r['seed']) for r in rows]
        why = None
        if not rows or len(set(seeds)) != len(seeds):
            why = 'empty_or_duplicate_trial_seeds'
        elif any(r.get('status') in INFRASTRUCTURE or r.get('status') not in NORMAL_FAILURE | {'success'} for r in rows):
            why = 'infrastructure_or_unknown_status'
        elif any(r.get('status') == 'success' and _metric_time(r) is None for r in rows):
            why = 'missing_measured_backend_time'
        audit.append(dict(condition_id=entry['condition_id'], candidate_id=entry.get('candidate_id'),
                          eligible=why is None, reason=why))
        if why is None:
            item = dict(entry, by_seed={int(r['seed']): r for r in rows})
            grouped.setdefault(entry['condition_id'], []).append(item)
    result = {}
    for condition, actions in grouped.items():
        pairs = []
        for i, a in enumerate(actions):
            for b in actions[i+1:]:
                if set(a['by_seed']) != set(b['by_seed']):
                    continue
                sa = {s for s, r in a['by_seed'].items() if r['status'] == 'success'}
                sb = {s for s, r in b['by_seed'].items() if r['status'] == 'success'}
                if len(sa) != len(sb):
                    better, worse = (a, b) if len(sa) > len(sb) else (b, a)
                    delta, kind = 1., 'success_count'
                elif sa and sa == sb:
                    delta = sum(math.log(_metric_time(b['by_seed'][s])/_metric_time(a['by_seed'][s])) for s in sa)/len(sa)
                    if abs(delta) <= 1e-10:
                        continue
                    better, worse = (a, b) if delta > 0 else (b, a)
                    delta, kind = abs(delta), 'raw_log_backend_runtime'
                else:
                    continue
                pairs.append(dict(better=better['action'], worse=worse['action'], weight=delta, kind=kind))
        result[condition] = pairs
    return result, audit


def feedback_loss(model, data, pairs, cache):
    prob = model.layer_distribution(data)[1]
    loss = prob.new_zeros(())
    for pair in pairs:
        better = cache.label(pair['better'])
        worse = cache.label(pair['worse'])
        delta = model.action_log_probability(data, better, probability=prob) - model.action_log_probability(data, worse, probability=prob)
        loss = loss + float(pair['weight'])*F.softplus(-delta)
    return loss/max(1, len(pairs))


def _fit(records, output, *, model, epochs, arm, shared=False, tail_weight=0.,
         feedback=None, learning_rate=1e-3, seed=2801, device='cpu', base=None,
         snapshots=None):
    _check_records(records)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('positive integer epochs required')
    output = Path(output); validate_output_roots((output,)); output.mkdir(parents=True, exist_ok=True)
    random.seed(seed); torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    cache = TrainingInputCache(base=base, device=device)
    pairs, audit = feedback_preferences(feedback)
    unknown = set(pairs)-{r['condition_id'] for r in records}
    if unknown:
        raise ValueError('feedback conditions outside training records')
    initial = state_sha256(model)
    xy_hash = state_sha256(model.shared_xy)
    history, paths = [], []
    snapshots = sorted(set(snapshots or [epochs]))
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        raise_if_cancelled(); model.train()
        order = list(range(len(records))); random.shuffle(order)
        steps, feedback_steps = [], 0
        for index in order:
            record = records[index]
            data, teacher = cache.graph(record['graph']), cache.label(record['teacher'])
            optimizer.zero_grad(set_to_none=True)
            loss, diagnostic = model.supervision_loss(data, teacher, tail_weight=tail_weight, shared=shared)
            selected = pairs.get(record['condition_id'], [])
            if selected:
                preference = feedback_loss(model, data, selected, cache)
                loss = loss + .2*preference
                diagnostic['feedback_loss'] = float(preference.detach()); feedback_steps += 1
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite layer training loss')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            # Keep per-epoch evidence compact. Full pool arrays belong to the
            # generated action diagnostics, not every training step in Git.
            if 'vertical_risk' in diagnostic:
                diagnostic['vertical_risk'] = {k: v for k, v in diagnostic['vertical_risk'].items()
                    if k not in ('pool_demand', 'pool_pressure', 'pool_capacity')}
            steps.append(dict(loss=float(loss.detach()), **diagnostic))
        if model.shared_xy_frozen and state_sha256(model.shared_xy) != xy_hash:
            raise AssertionError('frozen conditional XY changed during layer training')
        history.append(dict(epoch=epoch, average_loss=sum(v['loss'] for v in steps)/len(steps),
                            feedback_conditions=feedback_steps, steps=steps))
        if epoch in snapshots:
            paths.append(save_model(model, output/f'epoch_{epoch:04d}.pt', epoch=epoch, arm=arm))
    report = dict(schema='tailcompile-v7-layer-training-1', arm=arm, epochs=epochs,
        checkpoint=paths[-1], checkpoints=paths, config=model.config, history=history,
        initial_state_sha256=initial, final_state_sha256=state_sha256(model),
        shared_xy_sha256=state_sha256(model.shared_xy), shared_xy_frozen=model.shared_xy_frozen,
        feedback_pairs=sum(map(len, pairs.values())), feedback_audit=audit,
        feedback_kind='raw_matched_backend_runtime_preference_no_per_case_quality_gate',
        actual_generator_trained=True, train_condition_count=len(records), validation_gradients_used=False,
        tail_weight=tail_weight, vpr_or_capacity_decoder_differentiated=False,
        training_seconds=time.perf_counter()-started, input_cache=cache.audit())
    _write(output/'training_report.json', report)
    return report


def train_shared(records, output, *, epochs=20, hidden=64, chunk_size=256,
                 learning_rate=1e-3, seed=2801, device='cpu', base=None):
    _check_records(records)
    cache = TrainingInputCache(base=base)
    data = cache.graph(records[0]['graph'])
    widths = dict(cluster_width=data['cluster_x'].shape[1], region_width=data['region_x'].shape[1],
                  pool_width=data['pool_x'].shape[1])
    del data
    positives, scale_audit = [], []
    for record in records:
        graph = cache.graph(record['graph'])
        if 'vertical_risk_device' not in graph:
            scale_audit.append(dict(condition_id=record['condition_id'], available=False))
            continue
        try:
            from .vertical_risk import hard_risk
        except ImportError:
            from vertical_risk import hard_risk
        teacher = cache.label(record['teacher'])
        layer = graph['region_layer'][torch.as_tensor(teacher['teacher_region'], dtype=torch.long)]
        detail = hard_risk(graph, layer, torch.as_tensor(teacher['teacher_xy'], dtype=torch.float32))
        value = float(detail['cvar95'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('nonfinite or negative train teacher Tail statistic')
        if value > 0: positives.append(value)
        scale_audit.append(dict(condition_id=record['condition_id'], available=True, teacher_cvar95=value))
    tail_scale = statistics.median(positives) if positives else 1.
    random.seed(seed); torch.manual_seed(seed)
    model = ExplicitLayerInitializer(**widths, hidden=hidden, chunk_size=chunk_size, vertical_tail_scale=tail_scale)
    report = _fit(records, output, model=model, epochs=epochs, arm='shared', shared=True,
                  learning_rate=learning_rate, seed=seed, device=device, base=base)
    # Deployment checkpoint always records the complete frozen XY contract.
    model.freeze_shared_xy()
    report['checkpoint'] = save_model(model, Path(output)/'shared_frozen.pt', epoch=epochs, arm='shared')
    report['checkpoints'] = [report['checkpoint']]
    report['shared_xy_frozen'] = True
    report['tail_normalization'] = dict(scale=tail_scale, source='positive train teacher CVaR median',
        zero_or_unavailable_fallback=not positives, validation_or_test_used=False, conditions=scale_audit)
    _write(Path(output)/'training_report.json', report)
    return report


def prepare_arms(shared, records, output, *, epochs=5, tail_weight=.1,
                 learning_rate=1e-3, seed=2801, device='cpu', base=None):
    checkpoint = shared['checkpoint'] if isinstance(shared, dict) else shared
    result = dict(schema='tailcompile-v7-layer-paired-preparation-1', arms={})
    for arm in ('L', 'T'):
        model = load_model(checkpoint, device=device)
        if not model.shared_xy_frozen:
            raise ValueError('paired preparation requires frozen XY')
        result['arms'][arm] = _fit(records, Path(output)/arm, model=model, epochs=epochs, arm=arm,
            tail_weight=tail_weight if arm == 'T' else 0., learning_rate=learning_rate,
            seed=seed, device=device, base=base)
    a, b = result['arms']['L'], result['arms']['T']
    if a['initial_state_sha256'] != b['initial_state_sha256'] or a['shared_xy_sha256'] != b['shared_xy_sha256']:
        raise AssertionError('paired preparation lacks common start/shared XY')
    result['shared_xy_sha256'] = a['shared_xy_sha256']
    _write(Path(output)/'preparation_report.json', result)
    return result


def train_feedback(records, prepared, output, feedback, *, epochs=10, tail_weight=.1,
                   learning_rate=1e-3, seed=2801, device='cpu', base=None):
    _check_records(records)
    pairs, audit = feedback_preferences(feedback)
    output = Path(output); validate_output_roots((output,)); output.mkdir(parents=True, exist_ok=True)
    result = dict(schema='tailcompile-v7-layer-paired-feedback-1', arms={},
                  feedback_pairs=sum(map(len, pairs.values())), feedback_audit=audit)
    for arm in ('L', 'T'):
        checkpoint = prepared['arms'][arm]['checkpoint']
        if not result['feedback_pairs']:
            result['arms'][arm] = dict(checkpoint=checkpoint, checkpoints=[checkpoint],
                actual_generator_trained=False, epochs=0, reason='no_comparable_preferences',
                shared_xy_sha256=prepared['arms'][arm]['shared_xy_sha256'])
            continue
        model = load_model(checkpoint, device=device)
        result['arms'][arm] = _fit(records, output/arm, model=model, epochs=epochs, arm=arm,
            tail_weight=tail_weight if arm == 'T' else 0., feedback=feedback,
            learning_rate=learning_rate, seed=seed, device=device, base=base,
            snapshots=sorted({max(1, epochs//2), epochs}))
    result['shared_xy_sha256'] = prepared['shared_xy_sha256']
    _write(output/'feedback_training_report.json', result)
    return result
