"""010 bounded supervised/geometry training with genuine deployable QoR pairs.

Old actions are never projected while retaining their route labels. They either
pass current action/leaf constraints unchanged or are explicitly excluded.
Teacher projection is supervised geometry only and carries no measured reward.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

try:
    from .structured_model import (StructuredInitializer, SCHEMA, DECODER_SCHEMA,
        ACTION_SCHEMA, OBJECTIVE, generate, project_teacher, validate_structured_action, action_digest)
    from .structured_decoder import StructuredCapacityError
    from .qor_preferences import build_preferences
    from .train_qor import load_model as load009
    from .train import TrainingInputCache, state_sha256
    from .repository_guard import validate_output_roots
    from .io_utils import raise_if_cancelled
except ImportError:
    from structured_model import (StructuredInitializer, SCHEMA, DECODER_SCHEMA,
        ACTION_SCHEMA, OBJECTIVE, generate, project_teacher, validate_structured_action, action_digest)
    from structured_decoder import StructuredCapacityError
    from qor_preferences import build_preferences
    from train_qor import load_model as load009
    from train import TrainingInputCache, state_sha256
    from repository_guard import validate_output_roots
    from io_utils import raise_if_cancelled


def _write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def save_model(model, path, *, epoch=0):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(schema=SCHEMA, decoder_schema=DECODER_SCHEMA, config=model.config,
                    state_dict=model.state_dict(), generator_sha256=state_sha256(model),
                    shared_xy_sha256=state_sha256(model.shared_xy), epoch=epoch,
                    objective_version=OBJECTIVE, tail_weight=0.), path)
    return str(path.resolve())


def load_model(path, *, device='cpu'):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get('schema') != SCHEMA or payload.get('decoder_schema') != DECODER_SCHEMA:
        raise ValueError('010 checkpoint schema mismatch; use explicit from009 migration')
    if payload.get('objective_version') != OBJECTIVE or payload.get('tail_weight') != 0:
        raise ValueError('010 checkpoint objective mismatch')
    model = StructuredInitializer(**payload['config']).to(device)
    model.load_state_dict(payload['state_dict'])
    if state_sha256(model) != payload['generator_sha256'] or state_sha256(model.shared_xy) != payload['shared_xy_sha256']:
        raise ValueError('010 checkpoint hash mismatch')
    return model


def from009(path, *, device='cpu'):
    source = load009(path, device='cpu')
    if source.arm != 'X' or source.shared_xy_frozen:
        raise ValueError('010 source must be the selected trainable 009 X checkpoint')
    # D control and N training start byte-identical even across worker processes.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(10301)
        model = StructuredInitializer(**source.config)
    missing, unexpected = model.load_state_dict(source.state_dict(), strict=False)
    if unexpected or any(not k.startswith(('layer_context.', 'xy_context.')) for k in missing):
        raise ValueError('009 to 010 migration changed an inherited parameter')
    return model.to(device)


def from_previous(path, *, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    return load_model(path, device=device) if payload.get('schema') == SCHEMA else from009(path, device=device)


def _records(records):
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('010 requires explicitly train-only records')
    if len({r['condition_id'] for r in records}) != len(records):
        raise ValueError('duplicate 010 condition')


def _eligible(records, entries, cache):
    by_condition = {r['condition_id']: r for r in records}
    eligible, audits = [], []
    for entry in entries or []:
        if entry.get('split', 'train') != 'train' or entry['condition_id'] not in by_condition:
            raise ValueError('010 feedback outside train conditions')
        if entry.get('action') is None:
            continue
        data = cache.graph(by_condition[entry['condition_id']]['graph'])
        action = cache.label(entry['action'])
        generation = entry.get('generation') or {}
        for key in ('shared_xy_sha256', 'constraints_sha256', 'deployed_action_sha256'):
            recorded = action_digest(action) if key == 'deployed_action_sha256' and key not in action else action.get(key)
            if key in generation and generation[key] != recorded:
                raise ValueError('010 feedback collection metadata mismatch: '+key)
        if generation.get('target_sha256') and isinstance(entry['action'], (str, Path)):
            p = Path(entry['action'])
            if not p.is_absolute() and cache.base is not None:
                p = Path(cache.base)/p
            if hashlib.sha256(p.read_bytes()).hexdigest() != generation['target_sha256']:
                raise ValueError('010 feedback action bytes changed')
        try:
            validate_structured_action(data, action, collected=True)
        except StructuredCapacityError as exc:
            audits.append(dict(condition_id=entry['condition_id'], candidate_id=entry.get('candidate_id'),
                               eligible=False, reason='old_action_not_010_leaf_feasible', detail=str(exc),
                               old_qor_attached_to_projection=False))
            continue
        eligible.append(entry)
        audits.append(dict(condition_id=entry['condition_id'], candidate_id=entry.get('candidate_id'),
                           eligible=True, reason=None, old_qor_attached_to_projection=False))
    return eligible, audits


def _feedback(model, data, pairs, cache):
    logits = model.layer_distribution(data)[0]
    logprob = F.log_softmax(logits, 1)
    encoded = model.encode_shared(data)
    scores = {}
    def score(reference):
        key = str(reference) if isinstance(reference, (str, Path)) else id(reference)
        if key not in scores:
            action = cache.label(reference)
            scores[key] = model.action_score(data, action, log_probability=logprob,
                                             encoded=encoded, validated=True)
        return scores[key]
    loss = logits.new_zeros(())
    correct = tied = 0
    for pair in pairs:
        difference = score(pair['better'])-score(pair['worse'])
        loss = loss+float(pair['weight'])*F.softplus(-difference)
        correct += int(float(difference.detach()) > 0)
        tied += int(float(difference.detach()) == 0)
    return loss/max(1, len(pairs)), dict(correct=correct, ties=tied, pairs=len(pairs))


def fit(records, source_checkpoint, output, feedback=None, *, epochs=5, snapshots=(2, 5),
        seed=10301, learning_rate=1e-3, device='cpu', base=None):
    _records(records)
    if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
        raise ValueError('010 epochs must be bounded in [1,20]')
    snapshots = sorted(set(snapshots))
    if not snapshots or min(snapshots) < 1 or max(snapshots) > epochs:
        raise ValueError('010 checkpoint snapshots outside epoch budget')
    if not 0 < learning_rate <= .1:
        raise ValueError('010 learning rate outside (0,.1]')
    output = Path(output); validate_output_roots((output,)); output.mkdir(parents=True, exist_ok=True)
    if (output/'training_report.json').exists():
        raise ValueError('010 training output already completed; use new attempt')
    random.seed(seed); torch.manual_seed(seed)
    model = from_previous(source_checkpoint, device=device)
    cache = TrainingInputCache(base=base, device=device, label_entries=32)
    entries, audits = _eligible(records, feedback, cache)
    preferences = build_preferences(entries, objective='qor', base=base)
    preferences['scales'].update(
        schema='tailcompile-structured-preference-scales-010-1',
        source='010 current eligible train feedback pool; fitted before this fit',
        eligible_action_count=len(entries),
        eligible_condition_count=len({entry['condition_id'] for entry in entries}))
    pairs = preferences['pairs_by_condition']
    projected, projection_audits = {}, []
    for record in records:
        data, source = cache.graph(record['graph']), cache.label(record['teacher'])
        teacher = project_teacher(data, source, chunk_size=model.chunk_size)
        projected[record['condition_id']] = teacher
        file = output/'projected_teachers'/record['condition_id']/'teacher.json'
        _write(file, teacher)
        projection_audits.append(dict(condition_id=record['condition_id'], path=str(file.resolve()),
                                      target_sha256=teacher['target_sha256'], **teacher.get('projection', {})))
    initial, initial_xy = state_sha256(model), state_sha256(model.shared_xy)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    history, checkpoints = [], []
    started = time.perf_counter()
    for epoch in range(1, epochs+1):
        model.train(); order = list(range(len(records))); random.shuffle(order); steps = []
        for index in order:
            raise_if_cancelled()
            record = records[index]; condition = record['condition_id']; data = cache.graph(record['graph'])
            optimizer.zero_grad(set_to_none=True)
            loss, detail = model.supervision_loss(data, projected[condition])
            active = pairs.get(condition, [])
            if active:
                preference, ranks = _feedback(model, data, active, cache)
                loss = loss+.2*preference
                detail.update(feedback_loss=float(preference.detach()), preupdate_training_rank=ranks)
            if not torch.isfinite(loss):
                raise FloatingPointError('nonfinite 010 training loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('nonfinite 010 training gradient')
            detail['coordinate_gradient_l1'] = sum(float(p.grad.abs().sum()) for p in model.shared_xy.coordinate_head.parameters() if p.grad is not None)
            detail['layer_gradient_l1'] = sum(float(p.grad.abs().sum()) for name, p in model.named_parameters() if name.startswith(('layer_network.', 'layer_query.', 'layer_context.')) and p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            steps.append(dict(condition_id=condition, loss=float(loss.detach()), supported_feedback_pairs=len(active), **detail))
        history.append(dict(epoch=epoch, average_loss=statistics.mean(s['loss'] for s in steps), steps=steps))
        if epoch in snapshots:
            checkpoints.append(save_model(model, output/f'epoch_{epoch:04d}.pt', epoch=epoch))
    final_rank = dict(correct=0, ties=0, pairs=0)
    model.eval()
    with torch.no_grad():
        for record in records:
            active = pairs.get(record['condition_id'], [])
            if active:
                _, ranks = _feedback(model, cache.graph(record['graph']), active, cache)
                for key in final_rank: final_rank[key] += ranks[key]
    final_rank['accuracy'] = final_rank['correct']/final_rank['pairs'] if final_rank['pairs'] else None
    report = dict(schema='tailcompile-v7-structured-training-010-1', objective_version=OBJECTIVE,
                  checkpoint=checkpoints[-1], checkpoints=checkpoints, epochs=epochs,
                  source_checkpoint=str(source_checkpoint), initial_state_sha256=initial,
                  final_state_sha256=state_sha256(model), initial_shared_xy_sha256=initial_xy,
                  shared_xy_sha256=state_sha256(model.shared_xy), xy_trained=True,
                  actual_generator_trained=True, actual_qor_feedback_trained=preferences['total_pairs'] > 0,
                  feedback_pairs=preferences['total_pairs'], feedback_eligibility=audits,
                  feedback_audit=preferences['audit'], scales=preferences['scales'],
                  training_rank=final_rank, rank_population='training supported pairs; not validation or generalization',
                  train_condition_count=len(records), validation_gradients_used=False,
                  teacher_projection=projection_audits, history=history, tail_weight=0., tail_evaluated=False,
                  training_seconds=time.perf_counter()-started, input_cache=cache.audit(),
                  inference='bounded hard five-resource leaf allocation; native GP/APPack still required')
    _write(output/'training_report.json', report)
    return report


train = fit
train_model = fit
migrate_from009 = from009
