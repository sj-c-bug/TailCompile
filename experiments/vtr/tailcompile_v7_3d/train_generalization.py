"""013 frozen-feature ranking with bounded capacity, L2 and fold-only scales."""
from __future__ import annotations

from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

try:
    from .generalization_model import GeneralizationModel, SCHEMA, OBJECTIVE
    from .generalization_spec import config_by_id
    from .train_edit_context import validate_cache, evaluate_cache, _pairs, _digest, _write, sha
    from .train import state_sha256
    from .residual_model import ARMS
    from .repository_guard import validate_output_roots
    from .io_utils import raise_if_cancelled
except ImportError:
    from generalization_model import GeneralizationModel, SCHEMA, OBJECTIVE
    from generalization_spec import config_by_id
    from train_edit_context import validate_cache, evaluate_cache, _pairs, _digest, _write, sha
    from train import state_sha256
    from residual_model import ARMS
    from repository_guard import validate_output_roots
    from io_utils import raise_if_cancelled


def save_model(model, path, *, epoch=0, train_ids=(), calibration=None, factor_config=None):
    path = Path(path)
    validate_output_roots((path,))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(schema=SCHEMA, objective_version=OBJECTIVE, config=model.config,
                   factor_config=factor_config, state_dict=model.state_dict(),
                   state_sha256=state_sha256(model), epoch=epoch,
                   train_ids=sorted(train_ids), calibration=calibration,
                   normalization=model.normalization_metadata())
    payload['payload_sha256'] = _digest(payload)
    torch.save(payload, path)
    return str(path.resolve())


def load_model(path, *, device='cpu'):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    if payload.get('payload_sha256') != _digest({k: v for k, v in payload.items() if k != 'payload_sha256'}):
        raise ValueError('013 checkpoint metadata/state hash mismatch')
    if payload.get('schema') != SCHEMA or payload.get('objective_version') != OBJECTIVE:
        raise ValueError('013 checkpoint schema/objective mismatch')
    model = GeneralizationModel(**payload['config'])
    model.load_state_dict(payload['state_dict'])
    if state_sha256(model) != payload['state_sha256'] or model.normalization_metadata() != payload['normalization']:
        raise ValueError('013 checkpoint state/normalization mismatch')
    factors = payload['factor_config']
    if factors is not None:
        if factors != config_by_id(factors['id']) or any(
                factors[key] != model.config[key] for key in ('mode', 'hidden', 'normalization')):
            raise ValueError('013 checkpoint factor configuration mismatch')
    if model.config['normalizer_train_ids'] != payload['train_ids']:
        raise ValueError('013 checkpoint normalizer/train fold mismatch')
    model.training_metadata = dict(train_ids=payload['train_ids'], epoch=payload['epoch'],
                                   calibration=payload['calibration'], factor_config=factors,
                                   normalization=payload['normalization'])
    return model.to(device)


def fit_cache(cache, train_ids, output, *, config, epochs=80, snapshots=(5, 20, 80),
              seed=12301, learning_rate=.001, device='cpu'):
    validate_cache(cache)
    if not isinstance(config, dict) or config != config_by_id(config.get('id')):
        raise ValueError('013 configuration must match a frozen factor control')
    ids = set(train_ids)
    known = {bank['condition_id'] for bank in cache['banks']}
    if not ids or not ids <= known or isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 80:
        raise ValueError('013 known nonempty train IDs and bounded epochs [1,80] required')
    snapshots = sorted(set(snapshots))
    if not snapshots or min(snapshots) < 1 or max(snapshots) > epochs or snapshots[-1] != epochs or not 0 < learning_rate <= .1:
        raise ValueError('013 snapshots must include final epoch and respect training budget')
    output = Path(output)
    validate_output_roots((output,))
    if (output/'training_report.json').exists():
        raise ValueError('013 training output already completed')
    banks = [bank for bank in cache['banks'] if bank['condition_id'] in ids]
    pairs = [_pairs(bank) for bank in banks]
    scales = [pair['raw_delta'] for arr in pairs for pair in arr if pair['kind'] != 'success_count']
    scale = statistics.median(scales) if scales else 1.
    if not any(pairs):
        _write(output/'zero_signal_report.json', dict(schema='generalization-zero-signal-013-1',
               training_completed=False, train_ids=sorted(ids), factor_config=config, train_pairs=0,
               cache_sha256=cache['cache_sha256'], new_pnr=0, reason='no effective train-fold preferences'))
        raise ValueError('013 no effective train-fold preferences; stop without checkpoint')
    for arr in pairs:
        for pair in arr:
            pair['weight'] = 1. if pair['kind'] == 'success_count' else min(5., pair['raw_delta']/scale)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = GeneralizationModel(cache['global_dim'], cache['local_dim'],
                    mode=config['mode'], hidden=config['hidden'], normalization=config['normalization'],
                    source_checkpoint_sha256=cache['source_checkpoint_sha256'],
                    backbone_sha256=cache['backbone_sha256'], cache_sha256=cache['cache_sha256'])
    normalization = model.fit_normalizer(cache, ids)
    initial = state_sha256(model)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=config['weight_decay'])
    rng = random.Random(seed)
    history, checkpoints = [], []
    started = time.perf_counter()
    calibration = dict(qor_scale=scale, weight_cap=5., source='train IDs only; fitted after fold split',
                       train_ids=sorted(ids), holdout_labels_used=False)
    for epoch in range(1, epochs+1):
        model.train()
        order = [i for i, arr in enumerate(pairs) if arr]
        rng.shuffle(order)
        losses, gradient = [], 0.
        for i in order:
            raise_if_cancelled()
            optimizer.zero_grad(set_to_none=True)
            bank = banks[i]
            scores = model.score_features(bank['global_features'], bank['local_features'], bank['arm'])
            loss = sum(pair['weight']*F.softplus(-(scores[pair['better']]-scores[pair['worse']]))
                       for pair in pairs[i])/len(pairs[i])
            if not torch.isfinite(loss):
                raise FloatingPointError('013 nonfinite ranking loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('013 nonfinite head gradient')
            gradient += sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch, loss=statistics.mean(losses), gradient_l1=gradient, updates=len(order)))
        if epoch in snapshots:
            checkpoints.append(save_model(model, output/f'epoch_{epoch:04d}.pt', epoch=epoch,
                               train_ids=ids, calibration=calibration, factor_config=config))
    evaluation = evaluate_cache(model, cache, ids)
    report = dict(schema='tailcompile-v7-generalization-fit-013-1', mode=config['mode'],
                  factor_config=config, epochs=epochs, seed=seed,
                  checkpoint=checkpoints[-1], checkpoints=checkpoints,
                  checkpoint_sha256={path: sha(path) for path in checkpoints},
                  train_ids=sorted(ids), train_condition_count=len(ids), train_pairs=sum(map(len, pairs)),
                  train_pairs_by_arm={arm: sum(len(p) for b, p in zip(banks, pairs) if b['arm'] == arm) for arm in ARMS},
                  initial_state_sha256=initial, final_state_sha256=state_sha256(model),
                  parameter_count=sum(p.numel() for p in model.parameters()),
                  global_dim=cache['global_dim'], local_dim=cache['local_dim'],
                  calibration=calibration, normalization=normalization,
                  optimizer=dict(name='Adam', learning_rate=learning_rate, weight_decay=config['weight_decay'],
                                 penalty='coupled L2 on head parameters only'),
                  source_checkpoint_sha256=cache['source_checkpoint_sha256'],
                  backbone_sha256=cache['backbone_sha256'], cache_sha256=cache['cache_sha256'],
                  backbone_trained=False, training_uses_cached_frozen_features=True,
                  validation_labels_used_for_optimization=False, teacher_targets_used=False,
                  tail_weight=0., new_pnr=0, snapshots_for_fit_diagnostics_only=True,
                  test_checkpoint_epoch_fixed=epochs, training_evaluation=evaluation,
                  history=history, training_seconds=time.perf_counter()-started, device=str(device))
    _write(output/'training_report.json', report)
    return report
