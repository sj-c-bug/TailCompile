"""Train the deployed 3D generator using teachers and signed VPR feedback.

Teacher pretraining emits up to three candidates, never a teacher-loss-selected
"best" checkpoint. The pipeline must evaluate these on real validation VPR and
use supervision.select_checkpoint. Feedback keeps positive and negative actions;
only outcomes with a comparable budget/condition and observed status contribute.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
import copy
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch

try:
    from .io_utils import install_signal_handlers, raise_if_cancelled
    from .repository_guard import validate_output_roots
except ImportError:
    from io_utils import install_signal_handlers, raise_if_cancelled
    from repository_guard import validate_output_roots

try:
    from .model import Hierarchical3DInitializer, SCHEMA, to_device
except ImportError:
    from model import Hierarchical3DInitializer, SCHEMA, to_device

FLOAT_KEYS = ('cluster_x', 'region_x', 'pool_x', 'logic_weight', 'region_weight',
              'logic_criticality', 'region_bounds', 'demand', 'capacity',
              'fixed_xy', 'layer_capacity')
LONG_KEYS = ('logic_src', 'logic_dst', 'region_src', 'region_dst', 'pool_ids',
             'pool_regions', 'region_layer', 'fixed_region', 'fixed_layer')
BOOL_KEYS = ('movable', 'legal_layers')
INFRASTRUCTURE = {'infrastructure_failure', 'infrastructure_error', 'oom_unknown',
                  'unsupported', 'unknown', 'not_run'}
NORMAL_FAILURE = {'unroutable', 'packing_or_placement_failure', 'decode_failure',
                  'timeout', 'resource_budget_exceeded'}


def read_json(value, base=None):
    if isinstance(value, (dict, list)):
        return copy.deepcopy(value)
    path = Path(value)
    if base is not None and not path.is_absolute():
        path = Path(base) / path
    return json.loads(path.read_text(encoding='utf-8'))


def load_graph(value, *, base=None, device='cpu'):
    if isinstance(value, (str, Path)) and Path(value).suffix in ('.pt', '.pth'):
        path = Path(value)
        if base is not None and not path.is_absolute():
            path = Path(base) / path
        graph = torch.load(path, map_location='cpu', weights_only=False)
    else:
        graph = read_json(value, base)
    for key in FLOAT_KEYS:
        if key in graph:
            graph[key] = torch.as_tensor(graph[key], dtype=torch.float32)
    for key in LONG_KEYS:
        if key in graph:
            graph[key] = torch.as_tensor(graph[key], dtype=torch.long)
    for key in BOOL_KEYS:
        if key in graph:
            graph[key] = torch.as_tensor(graph[key], dtype=torch.bool)
    graph['parents'] = [torch.as_tensor(parent, dtype=torch.long) for parent in graph.get('parents', [])]
    for key, width in (('pool_x', 8), ('region_bounds', 4), ('demand', 5), ('capacity', 5)):
        if graph[key].numel() == 0:
            graph[key] = graph[key].reshape(0, width)
    return to_device(graph, device)


def state_sha256(model):
    """Stable tensor-byte digest, independent of torch.save archive metadata."""
    result = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        result.update(name.encode('utf-8'))
        result.update(str(value.dtype).encode('ascii'))
        result.update(json.dumps(list(value.shape)).encode('ascii'))
        result.update(value.view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


class TrainingInputCache:
    """Bounded graph/target LRU; records sharing a graph do not duplicate it.

    Raw teachers are projected before training and stored as derived targets.
    The cache only reads those immutable labels; it never runs projection.
    CPU training uses only the CPU graph cache. CUDA training keeps at most two
    graph payloads on device, plus the current step's autograd intermediates.
    """
    def __init__(self, *, base=None, device='cpu', cpu_graph_entries=2,
                 device_graph_entries=2, label_entries=4):
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (cpu_graph_entries, device_graph_entries, label_entries)):
            raise ValueError('training cache limits must be positive integers')
        if device_graph_entries > 2:
            raise ValueError('at most two device graphs may be resident')
        self.base, self.device = base, torch.device(device)
        self.cpu_limit, self.device_limit, self.label_limit = cpu_graph_entries, device_graph_entries, label_entries
        self.cpu_graphs, self.device_graphs, self.labels = OrderedDict(), OrderedDict(), OrderedDict()
        self.source_signatures = {}
        self.stats = dict(graph_loads=0, graph_disk_loads=0, cpu_graph_hits=0, device_graph_hits=0,
                          cpu_graph_evictions=0, device_graph_evictions=0,
                          label_loads=0, label_hits=0, label_evictions=0,
                          max_cpu_graphs=0, max_device_graphs=0, max_labels=0,
                          teacher_projection_calls=0)

    def _key(self, value):
        if isinstance(value, (str, Path)):
            path = Path(value)
            if self.base is not None and not path.is_absolute():
                path = Path(self.base) / path
            path = path.resolve()
            stat = path.stat()
            key = ('file', str(path))
            signature = (stat.st_size, stat.st_mtime_ns)
            if key in self.source_signatures and self.source_signatures[key] != signature:
                raise ValueError(f'training input changed during the run: {path}')
            self.source_signatures[key] = signature
            return key
        return ('memory', id(value))

    def _remember(self, cache, key, value, limit, *, kind):
        cache[key] = value
        cache.move_to_end(key)
        while len(cache) > limit:
            cache.popitem(last=False)
            self.stats[kind + '_evictions'] += 1
        self.stats['max_' + {'cpu_graph': 'cpu_graphs', 'device_graph': 'device_graphs',
                             'label': 'labels'}[kind]] = max(
            self.stats['max_' + {'cpu_graph': 'cpu_graphs', 'device_graph': 'device_graphs',
                                 'label': 'labels'}[kind]], len(cache))
        return value

    def graph(self, value):
        key = self._key(value)
        if self.device.type != 'cpu' and key in self.device_graphs:
            self.stats['device_graph_hits'] += 1
            self.device_graphs.move_to_end(key)
            return self.device_graphs[key]
        if key in self.cpu_graphs:
            self.stats['cpu_graph_hits'] += 1
            self.cpu_graphs.move_to_end(key)
            graph = self.cpu_graphs[key]
        else:
            if len(self.cpu_graphs) >= self.cpu_limit:
                self.cpu_graphs.popitem(last=False)
                self.stats['cpu_graph_evictions'] += 1
            graph = load_graph(value, base=self.base, device='cpu')
            self.stats['graph_loads'] += 1
            self.stats['graph_disk_loads'] += int(isinstance(value, (str, Path)))
            self._remember(self.cpu_graphs, key, graph, self.cpu_limit, kind='cpu_graph')
        if self.device.type == 'cpu':
            return graph
        # Evict before allocating another device payload, not after an OOM.
        if len(self.device_graphs) >= self.device_limit:
            self.device_graphs.popitem(last=False)
            self.stats['device_graph_evictions'] += 1
        graph = to_device(graph, self.device)
        return self._remember(self.device_graphs, key, graph, self.device_limit, kind='device_graph')

    def label(self, value):
        key = self._key(value)
        if key in self.labels:
            self.stats['label_hits'] += 1
            self.labels.move_to_end(key)
            return self.labels[key]
        label = read_json(value, self.base)
        self.stats['label_loads'] += 1
        return self._remember(self.labels, key, label, self.label_limit, kind='label')

    def audit(self):
        return dict(self.stats, cpu_graph_limit=self.cpu_limit,
                    device_graph_limit=self.device_limit, label_limit=self.label_limit,
                    graph_loading='per_step_bounded_lru',
                    labels='preprojected_immutable_cpu_targets')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def _metric(trial, *keys):
    for key in keys:
        value = trial.get(key)
        if value is not None and math.isfinite(float(value)):
            return float(value)
    return None


def outcome_reward(candidate_trials, default_trials, *, quality_limit=1.03):
    """Reward is observed, censored-safe direction; not fabricated failure time.

    Matched successful trials produce clipped log-runtime advantage. A genuine
    success loss or quality-protection violation gives categorical -1, success
    gain gives +1. Two failures provide no runtime comparison. Infrastructure
    and missing seeds mask the action instead of becoming adverse labels.
    """
    candidate = {int(row['seed']): row for row in candidate_trials}
    baseline = {int(row['seed']): row for row in default_trials}
    if not candidate or set(candidate) != set(baseline):
        return dict(eligible=False, reason='unmatched_seed_set')
    values, reasons = [], []
    for seed in sorted(candidate):
        trial, default = candidate[seed], baseline[seed]
        status, base_status = trial.get('status'), default.get('status')
        if status in INFRASTRUCTURE or base_status in INFRASTRUCTURE:
            return dict(eligible=False, reason='infrastructure_or_unknown_outcome')
        if status not in NORMAL_FAILURE | {'success'} or base_status not in NORMAL_FAILURE | {'success'}:
            return dict(eligible=False, reason='unrecognized_outcome')
        if status == 'success' and base_status != 'success':
            values.append(1.); reasons.append('success_gain'); continue
        if status != 'success' and base_status == 'success':
            values.append(-1.); reasons.append('success_loss'); continue
        if status != 'success':
            continue
        t = _metric(trial, 'total_seconds', 'total_runtime', 'total_runtime_seconds')
        b = _metric(default, 'total_seconds', 'total_runtime', 'total_runtime_seconds')
        if t is None or b is None or t <= 0 or b <= 0:
            return dict(eligible=False, reason='missing_measured_total_runtime')
        for keys in (('cpd_ns', 'cpd'), ('wirelength', 'total_wirelength')):
            a_quality, b_quality = _metric(trial, *keys), _metric(default, *keys)
            if a_quality is None or b_quality is None or b_quality <= 0:
                return dict(eligible=False, reason='missing_measured_quality')
            if a_quality / b_quality > quality_limit:
                values.append(-1.); reasons.append('quality_protection_violation'); break
        else:
            values.append(max(-1., min(1., math.log(b / t) / math.log(1.25))))
            reasons.append('matched_success_runtime_advantage')
    if not values:
        return dict(eligible=False, reason='no_comparable_success_or_success_change')
    return dict(eligible=True, advantage=sum(values) / len(values),
                comparable_seed_count=len(values), reasons=reasons,
                failure_runtime_imputed=False)


def feedback_loss(model, data, actions, *, coordinate_sigma=.2, preference_weight=.5):
    """Signed action likelihood plus pairwise preference update the generator.

    Each action contains assignment, xy, and an observed signed advantage. Poor
    candidates lower their own likelihood; they are not silently discarded.
    Pairwise terms compare positive and negative actions under the same graph.
    """
    usable = [item for item in actions if item.get('eligible', True)]
    if not usable:
        raise ValueError('no eligible feedback actions')
    likelihoods, advantages = [], []
    for item in usable:
        value = float(item['advantage'])
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('feedback advantage must be finite and clipped to [-1,1]')
        likelihoods.append(model.action_log_probability(data, item['action'], coordinate_sigma=coordinate_sigma))
        advantages.append(value)
    ll = torch.stack(likelihoods)
    reward = ll.new_tensor(advantages)
    signed = -(reward.detach() * ll).mean()
    preference = ll.new_zeros(())
    pairs = 0
    for better in range(len(usable)):
        for worse in range(len(usable)):
            delta = advantages[better] - advantages[worse]
            if delta > 1e-6:
                preference = preference + delta * torch.nn.functional.softplus(-(ll[better] - ll[worse]))
                pairs += 1
    if pairs:
        preference = preference / pairs
    loss = signed + preference_weight * preference
    return loss, dict(signed_action_loss=float(signed.detach()),
                      pairwise_loss=float(preference.detach()), preference_pairs=pairs,
                      positive_actions=sum(v > 0 for v in advantages),
                      negative_actions=sum(v < 0 for v in advantages),
                      zero_actions=sum(v == 0 for v in advantages),
                      actual_generator_updated=True)


def train(records, output, *, epochs=20, hidden=64, learning_rate=1e-3,
          seed=601, checkpoint=None, feedback=None, feedback_weight=.2,
          device='cpu', base=None, cpu_graph_cache=2, device_graph_cache=2,
          label_cache=4):
    """Train a teacher-anchored generator; return checkpoints pending VPR selection.

    records: [{graph: <dict/path>, teacher: <dict/path>, split:'train',
               condition_id:<stable signature>, family:<family>}].
    feedback: [{condition_id, action:{assignment,xy}, trials, default_trials}].
    """
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
    feedback_by_condition, reward_audit = {}, []
    known = {record['condition_id'] for record in records}
    for entry in feedback or []:
        if entry.get('condition_id') not in known:
            raise ValueError('feedback not linked to an explicit train condition')
        reward = outcome_reward(entry['trials'], entry['default_trials'])
        action = entry.get('action')
        if action is None or (isinstance(action, dict) and not {'assignment', 'xy'} <= action.keys()):
            reward = dict(eligible=False, reason='no_complete_deployment_action',
                          observed_outcome_statuses=[row.get('status') for row in entry['trials']],
                          outcome_reward_before_action_mask=reward)
        reward_audit.append(dict(condition_id=entry['condition_id'], candidate_id=entry.get('candidate_id'), **reward))
        if reward['eligible']:
            feedback_by_condition.setdefault(entry['condition_id'], []).append(
                dict(action=entry['action'], **reward))
    snapshots = sorted({1, max(1, (epochs + 1) // 2), epochs})
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
                intervention, detail = feedback_loss(model, data, step_actions)
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
    report = dict(schema='tailcompile-v7-3d-generator-training-1', training_implementation=2,
        training_seed=seed,
        train_condition_count=len(records), family_count=len({r.get('family') for r in records}),
        epochs=epochs, feedback_weight=feedback_weight, feedback_audit=reward_audit,
        eligible_feedback_actions=sum(row['eligible'] for row in reward_audit),
        positive_feedback_actions=sum(row.get('advantage', 0) > 0 for row in reward_audit),
        negative_feedback_actions=sum(row.get('advantage', 0) < 0 for row in reward_audit),
        effective_nonzero_feedback_actions=sum(abs(row.get('advantage', 0)) > 1e-12 for row in reward_audit),
        feedback_objective='signed_joint_score_likelihood_and_pairwise_preference_plus_teacher_anchor',
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


def main():
    install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--feedback', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--hidden', type=int, default=64)
    parser.add_argument('--seed', type=int, default=601)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--cpu-graph-cache', type=int, default=2)
    parser.add_argument('--device-graph-cache', type=int, default=2)
    parser.add_argument('--label-cache', type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    manifest = read_json(args.records)
    records = manifest.get('records', []) if isinstance(manifest, dict) else manifest
    feedback = read_json(args.feedback) if args.feedback else None
    if isinstance(feedback, dict):
        feedback = feedback.get('records', [])
    report = train(records, args.out, epochs=args.epochs, hidden=args.hidden,
                   seed=args.seed, checkpoint=args.checkpoint, feedback=feedback,
                   device=args.device, base=args.records.parent,
                   cpu_graph_cache=args.cpu_graph_cache, device_graph_cache=args.device_graph_cache,
                   label_cache=args.label_cache)
    print(json.dumps({key: report[key] for key in ('train_condition_count', 'epochs',
        'eligible_feedback_actions', 'positive_feedback_actions', 'negative_feedback_actions',
        'checkpoints', 'checkpoint_selection')}, indent=2))


if __name__ == '__main__':
    main()
