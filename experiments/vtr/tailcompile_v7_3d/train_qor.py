"""009 matched-budget runtime/QoR training of the deployed layer/XY actor."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path
import random
import statistics
import time

import torch
from torch.nn import functional as F

try:
    from .qor_model import QorInitializer, ARMS, SCHEMA, DECODER_SCHEMA, ACTION_SCHEMA, RISK_GRADIENT, generate, validate_collected_action
    from .qor_preferences import build_preferences
    from .train import TrainingInputCache, state_sha256
    from .train_aligned_layers import load_model as load008
    from .repository_guard import validate_output_roots
    from .io_utils import raise_if_cancelled
except ImportError:
    from qor_model import QorInitializer, ARMS, SCHEMA, DECODER_SCHEMA, ACTION_SCHEMA, RISK_GRADIENT, generate, validate_collected_action
    from qor_preferences import build_preferences
    from train import TrainingInputCache, state_sha256
    from train_aligned_layers import load_model as load008
    from repository_guard import validate_output_roots
    from io_utils import raise_if_cancelled


def _write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _check_records(records):
    if not records or any(r.get('split') != 'train' for r in records):
        raise ValueError('009 requires nonempty explicitly train-only records')
    if len({r['condition_id'] for r in records}) != len(records):
        raise ValueError('duplicate 009 training condition')


def save_model(model,path,*,epoch):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema=SCHEMA,config=model.config,state_dict=model.state_dict(),
        arm=model.arm,epoch=epoch,generator_sha256=state_sha256(model),
        shared_xy_sha256=state_sha256(model.shared_xy),shared_xy_frozen=model.shared_xy_frozen,
        decoder_schema=DECODER_SCHEMA,objective_version='qor-first-20261102',
        risk_gradient_semantics=RISK_GRADIENT),path)
    return str(path.resolve())


def load_model(path,*,device='cpu'):
    payload=torch.load(path,map_location=device,weights_only=False)
    if payload.get('schema') != SCHEMA or payload.get('decoder_schema') != DECODER_SCHEMA:
        raise ValueError('009 checkpoint schema/decoder mismatch; migrate 008 only at bootstrap')
    model=QorInitializer(**payload['config']).to(device)
    model.load_state_dict(payload['state_dict'])
    if payload.get('arm') != model.arm or payload.get('shared_xy_frozen') != model.shared_xy_frozen:
        raise ValueError('009 checkpoint arm/trainability mismatch')
    if state_sha256(model) != payload.get('generator_sha256') or state_sha256(model.shared_xy) != payload.get('shared_xy_sha256'):
        raise ValueError('009 checkpoint state hash mismatch')
    return model


def from008(path,arm,*,device='cpu'):
    source=load008(path,device=device)
    if not source.shared_xy_frozen:
        raise ValueError('009 common source must be an explicitly frozen 008 candidate')
    model=QorInitializer(**source.config,arm=arm).to(device)
    model.load_state_dict(source.state_dict(),strict=True)
    return model


def _validate_entries(records,entries,cache):
    by={r['condition_id']:r for r in records}
    for entry in entries or []:
        if entry.get('split','train') != 'train' or entry['condition_id'] not in by:
            raise ValueError('feedback outside 009 train records')
        if entry.get('action') is None: continue
        data=cache.graph(by[entry['condition_id']]['graph'])
        action=cache.label(entry['action'])
        validate_collected_action(data,action)
        generation=entry.get('generation') or {}
        for name in ('shared_xy_sha256','constraints_sha256'):
            if name in generation and generation[name] != action.get(name):
                raise ValueError('collection metadata/action identity mismatch: '+name)
        if generation.get('target_sha256') is not None and isinstance(entry['action'],(str,Path)):
            path=Path(entry['action'])
            if cache.base is not None and not path.is_absolute(): path=Path(cache.base)/path
            if hashlib.sha256(path.read_bytes()).hexdigest() != generation['target_sha256']:
                raise ValueError('collected action payload hash mismatch')


def feedback_loss(model,data,pairs,cache,*,constant_cache=None,condition_id=None):
    """Full actor feedback. Frozen actor constants are computed once per action."""
    log_probability=F.log_softmax(model.layer_distribution(data)[0],dim=1)
    encoded=None if model.shared_xy_frozen else model.shared_xy.encode(data)
    scores={}
    def score(reference):
        key=('path',str(reference)) if isinstance(reference,(str,Path)) else ('memory',id(reference))
        if key not in scores:
            action=cache.label(reference)
            constant=None
            if model.shared_xy_frozen:
                stored=(condition_id,key)
                if constant_cache is not None and stored in constant_cache:
                    constant=constant_cache[stored]
                else:
                    with torch.no_grad():
                        parts=model.conditional_components(data,action)
                        constant=parts['leaf_log_probability']-parts['xy_squared_error']/(2*.25**2)
                    if constant_cache is not None: constant_cache[stored]=constant
            scores[key]=model.action_score(data,action,log_probability=log_probability,encoded=encoded,
                validated=True,constant=constant)
        return scores[key]
    loss=log_probability.new_zeros(())
    for pair in pairs:
        loss=loss+float(pair['weight'])*F.softplus(-(score(pair['better'])-score(pair['worse'])))
    return loss/max(1,len(pairs))


def _fit(records,output,*,model,feedback,scales,frozen_xy_sha256,epochs,seed=9301,
         learning_rate=1e-3,device='cpu',base=None,snapshots=None,require_pairs=False):
    _check_records(records)
    if isinstance(epochs,bool) or not isinstance(epochs,int) or epochs < 1:
        raise ValueError('positive integer QoR epochs required')
    output=Path(output); validate_output_roots((output,)); output.mkdir(parents=True,exist_ok=True)
    objective='runtime' if model.arm=='R' else 'qor'
    allowed=frozen_xy_sha256 if model.shared_xy_frozen else None
    preferences=build_preferences(feedback,objective=objective,scales=scales,allowed_xy_sha256=allowed,base=base)
    pairs=preferences['pairs_by_condition']
    if set(pairs)-{r['condition_id'] for r in records}: raise ValueError('preferences outside train records')
    cache=TrainingInputCache(base=base,device=device,label_entries=16)
    _validate_entries(records,feedback,cache)
    initial=state_sha256(model); initial_xy=state_sha256(model.shared_xy)
    if model.shared_xy_frozen and initial_xy != frozen_xy_sha256:
        raise ValueError('frozen actor XY does not match common 008 source')
    pair_count=sum(map(len,pairs.values()))
    if require_pairs and pair_count == 0:
        checkpoint=save_model(model,output/'unchanged.pt',epoch=0)
        report=dict(schema='tailcompile-v7-qor-training-009-1',arm=model.arm,objective=objective,
            epochs=0,checkpoint=checkpoint,checkpoints=[checkpoint],actual_generator_trained=False,
            reason='no_comparable_supported_preferences',feedback_pairs=0,feedback_audit=preferences['audit'],
            initial_state_sha256=initial,final_state_sha256=initial,shared_xy_sha256=initial_xy,
            initial_shared_xy_sha256=initial_xy,shared_xy_frozen=model.shared_xy_frozen,
            validation_gradients_used=False,scales=scales,history=[])
        _write(output/'training_report.json',report)
        return report
    random.seed(seed); torch.manual_seed(seed)
    model=model.to(device)
    optimizer=torch.optim.Adam((p for p in model.parameters() if p.requires_grad),lr=learning_rate)
    snapshots=sorted(set(snapshots or [epochs]))
    if min(snapshots)<1 or max(snapshots)>epochs: raise ValueError('checkpoint epochs outside budget')
    history,paths=[],[]; constants={}; started=time.perf_counter()
    for epoch in range(1,epochs+1):
        raise_if_cancelled(); model.train(); order=list(range(len(records))); random.shuffle(order)
        steps=[]
        for index in order:
            r=records[index]; data,target=cache.graph(r['graph']),cache.label(r['teacher'])
            optimizer.zero_grad(set_to_none=True)
            loss,detail=model.supervision_loss(data,target)
            active=pairs.get(r['condition_id'],[])
            if active:
                preference=feedback_loss(model,data,active,cache,constant_cache=constants,condition_id=r['condition_id'])
                loss=loss+.2*preference; detail['feedback_loss']=float(preference.detach())
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite QoR objective')
            loss.backward()
            coordinate=[p.grad for p in model.shared_xy.coordinate_head.parameters() if p.grad is not None]
            detail['coordinate_gradient_l1']=sum(float(g.abs().sum()) for g in coordinate)
            if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError('nonfinite QoR gradient')
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            if 'vertical_risk' in detail:
                detail['vertical_risk']={k:v for k,v in detail['vertical_risk'].items() if k not in ('pool_demand','pool_pressure','pool_capacity')}
            steps.append(dict(condition_id=r['condition_id'],loss=float(loss.detach()),supported_feedback_pairs=len(active),**detail))
        if model.shared_xy_frozen and state_sha256(model.shared_xy)!=frozen_xy_sha256:
            raise AssertionError('frozen actor XY changed')
        history.append(dict(epoch=epoch,average_loss=statistics.mean(s['loss'] for s in steps),steps=steps))
        if epoch in snapshots: paths.append(save_model(model,output/f'epoch_{epoch:04d}.pt',epoch=epoch))
    report=dict(schema='tailcompile-v7-qor-training-009-1',arm=model.arm,objective=objective,epochs=epochs,
        checkpoint=paths[-1],checkpoints=paths,config=model.config,history=history,
        initial_state_sha256=initial,final_state_sha256=state_sha256(model),
        initial_shared_xy_sha256=initial_xy,shared_xy_sha256=state_sha256(model.shared_xy),
        shared_xy_frozen=model.shared_xy_frozen,actual_generator_trained=True,
        feedback_pairs=pair_count,feedback_audit=preferences['audit'],scales=scales,
        objective_version='qor-first-20261102',train_condition_count=len(records),
        validation_gradients_used=False,training_seconds=time.perf_counter()-started,input_cache=cache.audit(),
        cut_weight=.1 if model.arm in ('R','Q') else 0.,tail_weight=.1 if model.arm=='T' else 0.,
        actor_score='mean logP(layer)+mean logP(leaf|layer)-normalized XY MSE/(2*.25**2)',
        coordinate_feedback_differentiable=not model.shared_xy_frozen,
        risk_forward='actual hard decoded layer and finite-density XY',risk_gradient_semantics=RISK_GRADIENT,
        inference_risk_diagnostics_default=False)
    _write(output/'training_report.json',report)
    return report


def train_bootstrap(records,source_checkpoint,output,feedback,*,epochs=5,seed=9301,
                    learning_rate=1e-3,device='cpu',base=None):
    _check_records(records)
    # Both scale estimates use only the supplied signed train pool before any
    # new actor is optimized; frozen and trainable actors share these constants.
    preferences=build_preferences(feedback,objective='qor',base=base)
    scales=preferences['scales']
    source=load008(source_checkpoint)
    frozen=state_sha256(source.shared_xy)
    output=Path(output); validate_output_roots((output,)); output.mkdir(parents=True,exist_ok=True)
    report=dict(schema='tailcompile-v7-qor-bootstrap-009-1',arms={},frozen_xy_sha256=frozen,
        scales=scales,source_checkpoint=str(source_checkpoint),objective_version='qor-first-20261102')
    for arm in ARMS:
        model=from008(source_checkpoint,arm,device=device)
        report['arms'][arm]=_fit(records,output/arm,model=model,feedback=feedback,scales=scales,
            frozen_xy_sha256=frozen,epochs=epochs,seed=seed,learning_rate=learning_rate,device=device,base=base)
    if len({r['initial_state_sha256'] for r in report['arms'].values()})!=1:
        raise AssertionError('five QoR arms do not share initial weights')
    _write(output/'bootstrap_report.json',report)
    return report


def train_feedback(records,prepared,output,feedback,*,epochs=10,seed=9301,
                   learning_rate=1e-3,device='cpu',base=None):
    _check_records(records)
    if prepared.get('schema')!='tailcompile-v7-qor-bootstrap-009-1':
        raise ValueError('009 feedback requires signed 009 bootstrap report')
    output=Path(output); validate_output_roots((output,)); output.mkdir(parents=True,exist_ok=True)
    report=dict(schema='tailcompile-v7-qor-feedback-009-1',arms={},
        frozen_xy_sha256=prepared['frozen_xy_sha256'],scales=prepared['scales'],objective_version='qor-first-20261102')
    for arm in ARMS:
        model=load_model(prepared['arms'][arm]['checkpoint'],device=device)
        report['arms'][arm]=_fit(records,output/arm,model=model,feedback=feedback,scales=prepared['scales'],
            frozen_xy_sha256=prepared['frozen_xy_sha256'],epochs=epochs,seed=seed,
            learning_rate=learning_rate,device=device,base=base,snapshots=sorted({max(1,epochs//2),epochs}),require_pairs=True)
    _write(output/'feedback_training_report.json',report)
    return report
