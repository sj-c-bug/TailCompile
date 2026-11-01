"""008 bounded training with checked feasible targets and deployed-action risk.

Pure caches and signed-outcome preferences are reused from 007. No historical
training function is patched or redirected. 007 checkpoints/actions are not
silently adopted by this distinct schema.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import random
import statistics
import time
import torch
from torch.nn import functional as F

try:
    from .aligned_model import AlignedLayerInitializer, SCHEMA, DECODER_SCHEMA, RISK_GRADIENT, generate
    from .aligned_teacher import verify_teacher, constraints_digest
    from .train import TrainingInputCache, state_sha256
    from .train_layers import feedback_preferences
    from .io_utils import raise_if_cancelled
    from .repository_guard import validate_output_roots
except ImportError:
    from aligned_model import AlignedLayerInitializer, SCHEMA, DECODER_SCHEMA, RISK_GRADIENT, generate
    from aligned_teacher import verify_teacher, constraints_digest
    from train import TrainingInputCache, state_sha256
    from train_layers import feedback_preferences
    from io_utils import raise_if_cancelled
    from repository_guard import validate_output_roots

ACTION_SCHEMA = 'tailcompile-v7-aligned-action-008-1'


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def _check_records(records):
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('nonempty explicitly train-only aligned records required')
    if len({r['condition_id'] for r in records}) != len(records):
        raise ValueError('duplicate aligned training condition')


def save_model(model, path, *, epoch, arm):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=SCHEMA, config=model.config, state_dict=model.state_dict(),
        shared_xy_frozen=model.shared_xy_frozen, shared_xy_sha256=state_sha256(model.shared_xy),
        epoch=epoch, arm=arm, generator_sha256=state_sha256(model),
        decoder_schema=DECODER_SCHEMA, risk_gradient_semantics=RISK_GRADIENT,
        aligned_teacher_required=True), path)
    return str(path.resolve())


def load_model(path, *, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('schema') != SCHEMA or saved.get('decoder_schema') != DECODER_SCHEMA:
        raise ValueError('aligned checkpoint schema/decoder mismatch; do not load 007 weights as 008')
    model = AlignedLayerInitializer(**saved['config']).to(device)
    model.load_state_dict(saved['state_dict'])
    if state_sha256(model) != saved.get('generator_sha256'):
        raise ValueError('aligned generator checkpoint hash mismatch')
    if state_sha256(model.shared_xy) != saved.get('shared_xy_sha256'):
        raise ValueError('aligned shared XY checkpoint hash mismatch')
    if saved.get('shared_xy_frozen'): model.freeze_shared_xy()
    return model


def _check_feedback_action(action, expected_xy_hash):
    if (action.get('schema') != ACTION_SCHEMA or action.get('decoder_kind') != DECODER_SCHEMA or
            action.get('shared_xy_sha256') != expected_xy_hash):
        raise ValueError('feedback must be freshly evaluated 008 actions from this frozen XY module')


def feedback_loss(model, data, pairs, cache, expected_xy_hash):
    probability = model.layer_distribution(data)[1]
    loss = probability.new_zeros(())
    for pair in pairs:
        better, worse = cache.label(pair['better']), cache.label(pair['worse'])
        _check_feedback_action(better, expected_xy_hash); _check_feedback_action(worse, expected_xy_hash)
        signature = constraints_digest(data)
        if any(action.get('constraints_sha256') != signature for action in (better, worse)):
            raise ValueError('feedback action belongs to different graph constraints')
        delta = model.action_log_probability(data, better, probability=probability)-model.action_log_probability(data, worse, probability=probability)
        loss = loss+float(pair['weight'])*F.softplus(-delta)
    return loss/max(1, len(pairs))


def _fit(records, output, *, model, epochs, arm, shared=False, tail_weight=0., feedback=None,
         learning_rate=1e-3, seed=2801, device='cpu', base=None, snapshots=None):
    _check_records(records)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or epochs < 1:
        raise ValueError('positive integer aligned epochs required')
    output = Path(output); validate_output_roots((output,)); output.mkdir(parents=True, exist_ok=True)
    random.seed(seed); torch.manual_seed(seed)
    model = model.to(device)
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    cache = TrainingInputCache(base=base, device=device)
    pairs, feedback_audit = feedback_preferences(feedback)
    if set(pairs)-{r['condition_id'] for r in records}:
        raise ValueError('feedback outside aligned train records')
    # Validate action provenance even when a label happens not to form a pair.
    xy_hash = state_sha256(model.shared_xy)
    for entry in feedback or []:
        if entry.get('action') is not None:
            _check_feedback_action(cache.label(entry['action']), xy_hash)
    initial = state_sha256(model)
    history, paths = [], []
    snapshots = sorted(set(snapshots or [epochs]))
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        raise_if_cancelled(); model.train()
        order = list(range(len(records))); random.shuffle(order)
        steps, feedback_steps = [], 0
        for i in order:
            record = records[i]
            data, target = cache.graph(record['graph']), cache.label(record['teacher'])
            # The model independently checks the target's actual layer/leaf/XY
            # feasibility; an old audit flag or file name cannot bypass it.
            optimizer.zero_grad(set_to_none=True)
            loss, details = model.supervision_loss(data, target, shared=shared, tail_weight=tail_weight)
            preferences = pairs.get(record['condition_id'], [])
            if preferences:
                preference = feedback_loss(model, data, preferences, cache, xy_hash)
                loss = loss+.2*preference
                details['feedback_loss'] = float(preference.detach()); feedback_steps += 1
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite aligned training objective')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            if 'vertical_risk' in details:
                details['vertical_risk'] = {k: v for k, v in details['vertical_risk'].items()
                    if k not in ('pool_demand', 'pool_pressure', 'pool_capacity')}
            steps.append(dict(loss=float(loss.detach()), **details))
        if model.shared_xy_frozen and state_sha256(model.shared_xy) != xy_hash:
            raise AssertionError('aligned layer update changed the frozen XY module')
        history.append(dict(epoch=epoch, average_loss=statistics.mean(s['loss'] for s in steps),
                            feedback_conditions=feedback_steps, steps=steps))
        if epoch in snapshots:
            paths.append(save_model(model, output/f'epoch_{epoch:04d}.pt', epoch=epoch, arm=arm))
    report = dict(schema='tailcompile-v7-aligned-training-008-1', arm=arm, epochs=epochs,
        checkpoint=paths[-1], checkpoints=paths, config=model.config, history=history,
        initial_state_sha256=initial, final_state_sha256=state_sha256(model),
        shared_xy_sha256=state_sha256(model.shared_xy), shared_xy_frozen=model.shared_xy_frozen,
        feedback_pairs=sum(map(len, pairs.values())), feedback_audit=feedback_audit,
        actual_generator_trained=True, train_condition_count=len(records), validation_gradients_used=False,
        tail_weight=tail_weight, aligned_teacher_required=True, decoder_schema=DECODER_SCHEMA,
        risk_forward='actual capacity-decoded layer and finite-density XY', risk_gradient_semantics=RISK_GRADIENT,
        vpr_or_capacity_decoder_differentiated=False, training_seconds=time.perf_counter()-started,
        inference_risk_diagnostics_default=False, input_cache=cache.audit())
    _write(output/'training_report.json', report)
    return report


def train_shared(records, output, *, epochs=20, hidden=64, chunk_size=256, density_passes=2,
                 resource_ce_fraction=.5, learning_rate=1e-3, seed=2801, device='cpu', base=None):
    _check_records(records)
    cache = TrainingInputCache(base=base)
    first = cache.graph(records[0]['graph'])
    widths = dict(cluster_width=first['cluster_x'].shape[1], region_width=first['region_x'].shape[1],
                  pool_width=first['pool_x'].shape[1])
    positives, scale_audit = [], []
    for record in records:
        data, target = cache.graph(record['graph']), cache.label(record['teacher'])
        teacher_audit = verify_teacher(data, target)
        if 'vertical_risk_device' not in data:
            scale_audit.append(dict(condition_id=record['condition_id'], available=False, teacher_valid=teacher_audit['valid']))
            continue
        try:
            from .vertical_risk import hard_risk
        except ImportError:
            from vertical_risk import hard_risk
        layer = torch.as_tensor(target['teacher_layer'], dtype=torch.long)
        detail = hard_risk(data, layer, torch.as_tensor(target['teacher_xy'], dtype=torch.float32))
        value = float(detail['cvar95'])
        if not math.isfinite(value) or value < 0:
            raise ValueError('invalid aligned train teacher CVaR')
        if value > 0: positives.append(value)
        scale_audit.append(dict(condition_id=record['condition_id'], available=True, teacher_valid=teacher_audit['valid'], teacher_cvar95=value))
    scale = statistics.median(positives) if positives else 1.
    random.seed(seed); torch.manual_seed(seed)
    model = AlignedLayerInitializer(**widths, hidden=hidden, chunk_size=chunk_size,
        vertical_tail_scale=scale, density_passes=density_passes, resource_ce_fraction=resource_ce_fraction)
    report = _fit(records, output, model=model, epochs=epochs, arm='shared', shared=True,
                  learning_rate=learning_rate, seed=seed, device=device, base=base)
    model.freeze_shared_xy()
    report['checkpoint'] = save_model(model, Path(output)/'shared_frozen.pt', epoch=epochs, arm='shared')
    report['checkpoints'] = [report['checkpoint']]; report['shared_xy_frozen'] = True
    report['tail_normalization'] = dict(scale=scale, source='positive aligned train teacher CVaR median',
        zero_or_unavailable_fallback=not positives, validation_or_test_used=False, conditions=scale_audit)
    _write(Path(output)/'training_report.json', report)
    return report


def prepare_arms(shared, records, output, *, epochs=5, tail_weight=.1, learning_rate=1e-3,
                 seed=2801, device='cpu', base=None):
    checkpoint = shared['checkpoint'] if isinstance(shared, dict) else shared
    result = dict(schema='tailcompile-v7-aligned-preparation-008-1', arms={})
    for arm in ('L', 'T'):
        model = load_model(checkpoint, device=device)
        if not model.shared_xy_frozen: raise ValueError('aligned paired preparation requires frozen XY')
        result['arms'][arm] = _fit(records, Path(output)/arm, model=model, epochs=epochs, arm=arm,
            tail_weight=tail_weight if arm == 'T' else 0., learning_rate=learning_rate,
            seed=seed, device=device, base=base)
    a, b = result['arms']['L'], result['arms']['T']
    if a['initial_state_sha256'] != b['initial_state_sha256'] or a['shared_xy_sha256'] != b['shared_xy_sha256']:
        raise AssertionError('aligned paired models lack common initial state/XY')
    result['shared_xy_sha256'] = a['shared_xy_sha256']
    _write(Path(output)/'preparation_report.json', result)
    return result


def train_feedback(records, prepared, output, feedback, *, epochs=10, tail_weight=.1,
                   learning_rate=1e-3, seed=2801, device='cpu', base=None):
    _check_records(records)
    output = Path(output); validate_output_roots((output,)); output.mkdir(parents=True, exist_ok=True)
    cache = TrainingInputCache(base=base)
    for entry in feedback or []:
        if entry.get('action') is not None:
            _check_feedback_action(cache.label(entry['action']), prepared['shared_xy_sha256'])
    pairs, audit = feedback_preferences(feedback)
    result = dict(schema='tailcompile-v7-aligned-feedback-008-1', arms={},
                  feedback_pairs=sum(map(len, pairs.values())), feedback_audit=audit)
    for arm in ('L', 'T'):
        checkpoint = prepared['arms'][arm]['checkpoint']
        if not result['feedback_pairs']:
            result['arms'][arm] = dict(checkpoint=checkpoint, checkpoints=[checkpoint], actual_generator_trained=False,
                epochs=0, reason='no_comparable_preferences', shared_xy_sha256=prepared['shared_xy_sha256'])
            continue
        model = load_model(checkpoint, device=device)
        result['arms'][arm] = _fit(records, output/arm, model=model, epochs=epochs, arm=arm,
            tail_weight=tail_weight if arm == 'T' else 0., feedback=feedback,
            learning_rate=learning_rate, seed=seed, device=device, base=base,
            snapshots=sorted({max(1, epochs//2), epochs}))
    result['shared_xy_sha256'] = prepared['shared_xy_sha256']
    _write(output/'feedback_training_report.json', result)
    return result
