"""Matched three-epoch control/treatment; reuses frozen model, cache and labels.

The local loop is based on frozen train.py. No monkey-patching or changes to the
005 trainer: policy is explicit and both arms begin at the same pretrained state.
"""
from __future__ import annotations
import json
from pathlib import Path
import random
import time
import torch
import train as legacy
import reward_v2 as refined
from train import (TrainingInputCache, state_sha256, Hierarchical3DInitializer, SCHEMA)
from io_utils import raise_if_cancelled
from repository_guard import validate_output_roots

def train(records, output, *, epochs=3, hidden=64, learning_rate=1e-3,
          seed=601, checkpoint=None, feedback=None, feedback_weight=.2,
          device='cpu', base=None, cpu_graph_cache=2, device_graph_cache=2,
          label_cache=4, policy='guarded'):
    """Train a teacher-anchored generator; return checkpoints pending VPR selection.

    records: [{graph: <dict/path>, teacher: <dict/path>, split:'train',
               condition_id:<stable signature>, family:<family>}].
    feedback: [{condition_id, action:{assignment,xy}, trials, default_trials}].
    """
    if policy not in ('clipped', 'guarded'):
        raise ValueError('unknown repair reward policy')
    reward_function = legacy.outcome_reward if policy == 'clipped' else refined.outcome_reward
    loss_function = legacy.feedback_loss if policy == 'clipped' else refined.feedback_loss
    if epochs < 1 or len(records) < 1:
        raise ValueError('positive epochs and nonempty train records required')
    raise_if_cancelled()
    if any(record.get('split') != 'train' for record in records):
        raise ValueError('only explicit train records may receive gradients')
    if len({record['condition_id'] for record in records}) != len(records):
        raise ValueError('duplicate condition_id in teacher records')
    random.seed(seed); torch.manual_seed(seed)
    output = Path(output)
    validate_output_roots((output,))
    output.mkdir(parents=True, exist_ok=True)
    cache = TrainingInputCache(base=base, device=device, cpu_graph_entries=cpu_graph_cache,
                               device_graph_entries=device_graph_cache, label_entries=label_cache)
    first = cache.graph(records[0]['graph'])
    config = dict(cluster_width=first['cluster_x'].shape[1], region_width=first['region_x'].shape[1],
                  hidden=hidden, chunk_size=256, pool_width=first['pool_x'].shape[1], demand_width=5)
    del first
    model = Hierarchical3DInitializer(**config).to(device)
    if checkpoint is not None:
        saved = torch.load(checkpoint, map_location=device, weights_only=False)
        if saved.get('schema') != SCHEMA or saved.get('config') != config:
            raise ValueError('3D checkpoint schema/config mismatch')
        model.load_state_dict(saved['state_dict'])
    initial_state_sha256 = state_sha256(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    feedback_by_condition, reward_audit, default_signatures = {}, [], {}
    known = {record['condition_id'] for record in records}
    for entry in feedback or []:
        if entry.get('condition_id') not in known:
            raise ValueError('feedback not linked to an explicit train condition')
        condition = entry['condition_id']
        signature = legacy.digest(entry['default_trials'])
        if default_signatures.setdefault(condition, signature) != signature:
            raise ValueError('candidate actions use different Default controls')
        reward = reward_function(entry['trials'], entry['default_trials'])
        action = entry.get('action')
        if action is None or (isinstance(action, dict) and not {'assignment', 'xy'} <= action.keys()):
            reward = dict(eligible=False, reason='no_complete_deployment_action',
                          observed_outcome_statuses=[row.get('status') for row in entry['trials']],
                          outcome_reward_before_action_mask=reward)
        reward_audit.append(dict(condition_id=entry['condition_id'], candidate_id=entry.get('candidate_id'), **reward))
        if reward['eligible']:
            feedback_by_condition.setdefault(entry['condition_id'], []).append(
                dict(action=entry['action'], **reward))
    pair_count = sum(len(refined.pairs(items)) for items in feedback_by_condition.values()) if policy == 'guarded' else None
    if policy == 'guarded' and not pair_count:
        raise ValueError('no guarded candidate preferences; inspect feedback before training')
    snapshots = sorted({1, epochs})
    history, paths = [], []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        raise_if_cancelled()
        model.train()
        order = list(range(len(records))); random.shuffle(order)
        totals = []
        feedback_details = []
        for index in order:
            raise_if_cancelled()
            record = records[index]
            data, teacher = cache.graph(record['graph']), cache.label(record['teacher'])
            optimizer.zero_grad(set_to_none=True)
            teacher_loss, diagnostic = model.supervision_loss(data, teacher, return_details=True)
            total = teacher_loss
            actions = feedback_by_condition.get(record['condition_id'], [])
            if actions:
                step_actions = [dict(item, action=cache.label(item['action'])) for item in actions]
                intervention, detail = loss_function(model, data, step_actions)
                total = total + feedback_weight * intervention
                feedback_details.append(dict(condition_id=record['condition_id'], **detail))
            if not torch.isfinite(total):
                raise FloatingPointError('nonfinite generator training objective')
            total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.); optimizer.step()
            totals.append(float(total.detach()))
            # Do not retain the current graph/labels in the training records.
            # The bounded caches alone own reusable input payloads between steps.
            if actions:
                del step_actions
            del data, teacher
        history.append(dict(epoch=epoch, teacher_anchored_training_loss=sum(totals) / len(totals),
                            feedback_conditions=len(feedback_details), feedback=feedback_details))
        if epoch in snapshots:
            path = output / f'epoch_{epoch:04d}.pt'
            torch.save(dict(schema=SCHEMA, config=config, state_dict=model.state_dict(), epoch=epoch,
                            seed=seed, selection_status='pending_real_vpr_validation'), path)
            paths.append(str(path.resolve()))
    final_state_sha256 = state_sha256(model)
    report = dict(schema='tailcompile-v7-3d-generator-training-1', training_implementation='repair-006', reward_policy=policy, initial_preference_pairs=pair_count,
        training_seed=seed,
        train_condition_count=len(records), family_count=len({r.get('family') for r in records}),
        epochs=epochs, feedback_weight=feedback_weight, feedback_audit=reward_audit,
        eligible_feedback_actions=sum(row['eligible'] for row in reward_audit),
        positive_feedback_actions=sum(row.get('advantage', 0) > 0 for row in reward_audit),
        negative_feedback_actions=sum(row.get('advantage', 0) < 0 for row in reward_audit),
        effective_nonzero_feedback_actions=sum(abs(row.get('advantage', 0)) > 1e-12 for row in reward_audit),
        feedback_objective='matched_control_'+policy+'_plus_teacher_anchor',
        vpr_or_hard_decoder_differentiated=False,
        history=history, checkpoints=paths, selected_checkpoint=None,
        checkpoint_selection='pending_real_vpr_validation_not_teacher_loss',
        measured_training_seconds=time.perf_counter() - started,
        actual_generator_trained=True, validation_gradients_used=False,
        initial_state_sha256=initial_state_sha256, final_state_sha256=final_state_sha256,
        weights_changed=initial_state_sha256 != final_state_sha256,
        training_input_cache=cache.audit(),
        architecture_transfer='new_3d_model_no_automatic_2d_weight_claim')
    (output / 'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report
