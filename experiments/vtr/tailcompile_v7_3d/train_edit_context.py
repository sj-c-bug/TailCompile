"""012 signed cached features and train-only calibrated finite-action ranking.

Preparation revalidates original 011 measured actions; it never routes them or
attaches old labels to changed coordinates. Grouped folds choose train IDs
before scales, optimization or checkpoint creation. Evaluation selects candidates
from features before reading their held-out labels.
"""
from __future__ import annotations

from collections import Counter
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
    from .edit_context_model import EditContextModel, feature_blocks, SCHEMA, OBJECTIVE
    from .edit_context_features import FEATURE_SCHEMA
    from .residual_model import ARMS
    from .train_residual import from_source, prepare_feedback, pair_delta, _trial_map
    from .train import TrainingInputCache, state_sha256
    from .repository_guard import validate_output_roots
    from .io_utils import raise_if_cancelled
except ImportError:
    from edit_context_model import EditContextModel, feature_blocks, SCHEMA, OBJECTIVE
    from edit_context_features import FEATURE_SCHEMA
    from residual_model import ARMS
    from train_residual import from_source, prepare_feedback, pair_delta, _trial_map
    from train import TrainingInputCache, state_sha256
    from repository_guard import validate_output_roots
    from io_utils import raise_if_cancelled

CACHE_SCHEMA='tailcompile-v7-edit-feature-cache-012-1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write(path,value):
    path=Path(path);validate_output_roots((path,));path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def _digest(value):
    digest=hashlib.sha256()
    def add(v):
        if torch.is_tensor(v):
            v=v.detach().cpu().contiguous()
            digest.update(b'T'+str(v.dtype).encode()+repr(tuple(v.shape)).encode()+v.numpy().tobytes())
        elif isinstance(v,dict):
            digest.update(b'{')
            for key in sorted(v,key=lambda k:str(k)):
                add(str(key));add(v[key])
            digest.update(b'}')
        elif isinstance(v,(list,tuple)):
            digest.update(b'[')
            for item in v:add(item)
            digest.update(b']')
        else:digest.update(json.dumps(v,sort_keys=True,allow_nan=False).encode()+b'\0')
    add(value)
    return digest.hexdigest()


def cache_digest(cache):
    return _digest({k:v for k,v in cache.items() if k!='cache_sha256'})


def validate_cache(cache):
    if cache.get('schema')!=CACHE_SCHEMA or cache.get('feature_schema')!=FEATURE_SCHEMA or \
        cache.get('cache_sha256')!=cache_digest(cache):
        raise ValueError('012 feature cache schema/content hash mismatch')
    keys=[]
    for bank in cache['banks']:
        key=(bank['condition_id'],bank['arm']);keys.append(key)
        if bank['arm'] not in ARMS or bank['candidate_ids'][0]!='identity':
            raise ValueError('012 cache bank must contain unchanged identity first')
        k=len(bank['candidate_ids'])
        if len(set(bank['candidate_ids']))!=k or len(bank['digests'])!=k or len(bank['trials'])!=k:
            raise ValueError('012 candidate label coverage mismatch')
        if bank['global_features'].shape!=(k,cache['global_dim']) or bank['local_features'].shape!=(k,cache['local_dim']):
            raise ValueError('012 cached feature dimensions mismatch')
        if not torch.isfinite(bank['global_features']).all() or not torch.isfinite(bank['local_features']).all():
            raise ValueError('012 cached features nonfinite')
        if torch.count_nonzero(bank['local_features'][0]):
            raise ValueError('012 identity local features must be neutral zero')
        seeds=set(bank['trials'][0])
        if any(set(trials)!=seeds for trials in bank['trials']):
            raise ValueError('012 cached candidates need complete matched measured seeds')
        for trials in bank['trials']:
            if set(_trial_map(dict(trials=list(trials.values()))))!=set(trials):
                raise ValueError('012 cached trial map keys disagree with measured seeds')
    if not keys or len(set(keys))!=len(keys):
        raise ValueError('012 empty/duplicate feature bank')
    return cache


def prepare_cache(records,source_checkpoint,feedback,*,expected_sha256=None,base=None):
    if not records or any(r.get('split')!='train' for r in records) or \
        len({r['condition_id'] for r in records})!=len(records):
        raise ValueError('012 requires unique explicit train-source records')
    model=from_source(source_checkpoint,expected_sha256=expected_sha256)
    cache=TrainingInputCache(base=base,device='cpu',label_entries=32)
    ids={r['condition_id'] for r in records}
    if any(e.get('condition_id') not in ids or e.get('split')!='train' for e in feedback):
        raise ValueError('012 feedback outside frozen source training records')
    banks=[];audits=[];inputs={str(Path(source_checkpoint).resolve()):sha(source_checkpoint)}
    start=time.perf_counter()
    for record in records:
        raise_if_cancelled()
        cid=record['condition_id'];data=cache.graph(record['graph'])
        if isinstance(record['graph'],(str,Path)):
            graph_path=Path(record['graph'])
            if base is not None and not graph_path.is_absolute():graph_path=Path(base)/graph_path
            inputs[str(graph_path.resolve())]=sha(graph_path)
        else:inputs['memory-graph:'+cid]=_digest(record['graph'])
        entries=[e for e in feedback if e['condition_id']==cid]
        prepared=prepare_feedback(model,[record],entries,cache)
        audits.extend(prepared['audit'])
        with torch.no_grad():encoded=model.backbone.encode_shared(data)
        for entry in entries:
            if isinstance(entry.get('action'),(str,Path)):
                path=Path(entry['action'])
                if base is not None and not path.is_absolute():path=Path(base)/path
                inputs[str(path.resolve())]=sha(path)
        # Only original measured actions are eligible for offline scoring.
        for arm in ARMS:
            key=cid,arm
            if key not in prepared['measured']:
                raise ValueError('012 source condition has no measured identity control')
            base_action,full=prepared['banks'][key]
            measured=prepared['measured'][key]
            chosen=[(i,a) for i,a in enumerate(full) if a['candidate_id'] in measured]
            actions=[a for i,a in chosen]
            glob,local=feature_blocks(model.backbone,data,base_action,actions,encoded=encoded)
            # Exact old global features should remain byte-identical.
            assert torch.equal(glob,prepared['features'][key][[i for i,a in chosen]])
            trials=[]
            for i,action in chosen:
                rows=measured[action['candidate_id']]['trials']
                trials.append({s:{k:r[k] for k in ('seed','status','wirelength','cpd_ns','confirmed_candidate_infeasibility') if k in r}
                               for s,r in rows.items()})
            family=record.get('family') or cid.split('/')[0]
            architecture=record.get('architecture') or cid.split('/')[-2]
            banks.append(dict(condition_id=cid,family=family,architecture=architecture,arm=arm,
                global_features=glob,local_features=local,candidate_ids=[a['candidate_id'] for a in actions],
                digests=[a['deployed_action_sha256'] for a in actions],trials=trials,
                full_candidate_ids=[a['candidate_id'] for a in full],
                full_candidate_digests=[a['deployed_action_sha256'] for a in full],
                measured_candidate_count=len(actions),generated_candidate_count=len(full)))
    if sha(source_checkpoint)!=model.source_checkpoint_sha256:
        raise ValueError('012 source checkpoint bytes changed while preparing features')
    result=dict(schema=CACHE_SCHEMA,feature_schema=FEATURE_SCHEMA,banks=banks,
        global_dim=banks[0]['global_features'].shape[1],local_dim=banks[0]['local_features'].shape[1],
        source_checkpoint_sha256=model.source_checkpoint_sha256,backbone_sha256=state_sha256(model.backbone),
        source_inputs=inputs,feedback_sha256=_digest(feedback),feedback_eligibility=audits,
        preparation_seconds=time.perf_counter()-start,candidate_generator='frozen011',
        graph_or_teacher_labels_used_as_features=False,new_pnr=0)
    result['cache_sha256']=cache_digest(result)
    return validate_cache(result)


def save_cache(cache,path):
    validate_cache(cache);path=Path(path);validate_output_roots((path,));path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(cache,path)
    return str(path.resolve())


def load_cache(path):
    return validate_cache(torch.load(path,map_location='cpu',weights_only=False))


def save_model(model,path,*,epoch=0,train_ids=(),calibration=None):
    path=Path(path);validate_output_roots((path,));path.parent.mkdir(parents=True,exist_ok=True)
    payload=dict(schema=SCHEMA,objective_version=OBJECTIVE,config=model.config,
        state_dict=model.state_dict(),state_sha256=state_sha256(model),epoch=epoch,
        train_ids=sorted(train_ids),calibration=calibration)
    payload['payload_sha256']=_digest(payload)
    torch.save(payload,path)
    return str(path.resolve())


def load_model(path,*,device='cpu'):
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload.get('payload_sha256')!=_digest({k:v for k,v in payload.items() if k!='payload_sha256'}):
        raise ValueError('012 checkpoint metadata/state hash mismatch')
    if payload.get('schema')!=SCHEMA or payload.get('objective_version')!=OBJECTIVE:
        raise ValueError('012 checkpoint schema/objective mismatch')
    model=EditContextModel(**payload['config']);model.load_state_dict(payload['state_dict'])
    if state_sha256(model)!=payload['state_sha256']:
        raise ValueError('012 checkpoint state hash mismatch')
    model.training_metadata=dict(train_ids=payload['train_ids'],epoch=payload['epoch'],calibration=payload['calibration'])
    return model.to(device)


def _pairs(bank,*,seed_subset=None):
    pairs=[]
    trials=bank['trials']
    if seed_subset is not None:
        trials=[{s:r for s,r in t.items() if s in seed_subset} for t in trials]
    for i,j in itertools.combinations(range(len(trials)),2):
        delta=pair_delta(trials[i],trials[j])
        if delta is not None:
            value,kind=delta
            pairs.append(dict(better=i if value>0 else j,worse=j if value>0 else i,
                              raw_delta=abs(value),kind=kind))
    return pairs


def select_indices(model,cache,condition_ids):
    """No trial/QoR label inspection; identity wins score ties."""
    validate_cache(cache)
    if model.config['cache_sha256']!=cache['cache_sha256']:
        raise ValueError('012 evaluation cache differs from trained feature source')
    ids=set(condition_ids);known={b['condition_id'] for b in cache['banks']}
    if not ids or not ids<=known:
        raise ValueError('012 unknown/empty evaluation IDs')
    result=[];model.eval()
    with torch.no_grad():
        for bank in cache['banks']:
            if bank['condition_id'] not in ids:continue
            scores=model.score_features(bank['global_features'],bank['local_features'],bank['arm'])
            if not torch.isfinite(scores).all():raise FloatingPointError('012 nonfinite candidate scores')
            index=int(scores.argmax())
            result.append(dict(condition_id=bank['condition_id'],family=bank['family'],architecture=bank['architecture'],
                arm=bank['arm'],selected_index=index,selected_candidate_id=bank['candidate_ids'][index],
                selected_digest=bank['digests'][index],candidate_ids=bank['candidate_ids'],scores=scores.cpu().tolist()))
    return result


def evaluate_cache(model,cache,condition_ids,*,seed_subset=None):
    condition_ids=tuple(condition_ids)
    predictions=select_indices(model,cache,condition_ids)
    lookup={(b['condition_id'],b['arm']):b for b in cache['banks']}
    ranks={a:dict(correct=0,ties=0,pairs=0) for a in ARMS};rows=[]
    for pred in predictions:
        bank=lookup[pred['condition_id'],pred['arm']]
        pairs=_pairs(bank,seed_subset=seed_subset);scores=pred['scores'];rank=ranks[bank['arm']]
        for pair in pairs:
            delta=scores[pair['better']]-scores[pair['worse']]
            rank['correct']+=int(delta>0);rank['ties']+=int(delta==0);rank['pairs']+=1
        trials=bank['trials']
        if seed_subset is not None:trials=[{s:r for s,r in t.items() if s in seed_subset} for t in trials]
        success=[{s for s,r in t.items() if r['status']=='success'} for t in trials]
        common=set.intersection(*success) if success else set()
        q=[];metrics=[]
        for i,t in enumerate(trials):
            values={m:statistics.mean(t[s][m] for s in common) if common else None for m in ('wirelength','cpd_ns')}
            metrics.append(values)
            q.append(.5*sum(math.log(values[m]/metrics[0][m]) for m in values) if common else None)
        best=min(range(len(trials)),key=lambda i:(-len(success[i]),q[i] if q[i] is not None else 0.,i!=0,i))
        selected=pred['selected_index']
        rows.append(dict(pred,common_success_seeds=sorted(common),selected_success_count=len(success[selected]),
            D_success_count=len(success[0]),selected_q_vs_D=q[selected],
            selected_WL=metrics[selected]['wirelength'],selected_CPD=metrics[selected]['cpd_ns'],
            D_WL=metrics[0]['wirelength'],D_CPD=metrics[0]['cpd_ns'],offline_best_candidate_id=bank['candidate_ids'][best],
            oracle_q=q[best],q_regret=None if not common else q[selected]-q[best],
            oracle_success_count=len(success[best]),selected_is_offline_best=selected==best,
            pair_count=len(pairs),generated_candidate_count=bank['generated_candidate_count'],
            measured_candidate_count=len(trials)))
    for rank in ranks.values():rank['accuracy']=rank['correct']/rank['pairs'] if rank['pairs'] else None
    return dict(rows=rows,ranking=ranks,condition_ids=sorted(set(condition_ids)),
        identity_selections=sum(r['selected_candidate_id']=='identity' for r in rows),
        candidate_selections=len(rows),selection_used_labels=False,
        label_population='reused011 measured feedback pool; not new P&R or independent benchmark validation',
        oracle_is_deployable_performance=False,seed_subset=sorted(seed_subset) if seed_subset is not None else None)


def fit_cache(cache,train_ids,output,*,mode='G',epochs=80,snapshots=(5,20,80),seed=12301,
              learning_rate=1e-3,device='cpu'):
    validate_cache(cache);ids=set(train_ids);known={b['condition_id'] for b in cache['banks']}
    if not ids or not ids<=known or isinstance(epochs,bool) or not isinstance(epochs,int) or not 1<=epochs<=80:
        raise ValueError('012 nonempty known train IDs and bounded epochs [1,80] required')
    snapshots=sorted(set(snapshots))
    if not snapshots or min(snapshots)<1 or max(snapshots)>epochs or not 0<learning_rate<=.1:
        raise ValueError('012 invalid snapshots/learning-rate budget')
    output=Path(output);validate_output_roots((output,))
    if (output/'training_report.json').exists():raise ValueError('012 training output already completed')
    banks=[b for b in cache['banks'] if b['condition_id'] in ids]
    pairs=[_pairs(b) for b in banks]
    scales=[p['raw_delta'] for arr in pairs for p in arr if p['kind']!='success_count']
    scale=statistics.median(scales) if scales else 1.
    if not any(pairs):
        _write(output/'zero_signal_report.json',dict(schema='edit-context-zero-signal-012-1',
            training_completed=False,train_ids=sorted(ids),mode=mode,train_pairs=0,
            cache_sha256=cache['cache_sha256'],new_pnr=0,
            reason='no effective train-fold preferences; no checkpoint created'))
        raise ValueError('012 no effective train-fold preferences; stop without checkpoint')
    for arr in pairs:
        for p in arr:p['weight']=1. if p['kind']=='success_count' else min(5.,p['raw_delta']/scale)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model=EditContextModel(cache['global_dim'],cache['local_dim'],mode=mode,
            source_checkpoint_sha256=cache['source_checkpoint_sha256'],backbone_sha256=cache['backbone_sha256'],
            cache_sha256=cache['cache_sha256'])
    initial=state_sha256(model);model=model.to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)
    rng=random.Random(seed);history=[];checkpoints=[];started=time.perf_counter()
    calibration=dict(qor_scale=scale,weight_cap=5.,source='train IDs only; fitted after fold split',
        train_ids=sorted(ids),holdout_labels_used=False)
    for epoch in range(1,epochs+1):
        model.train();order=[i for i,a in enumerate(pairs) if a];rng.shuffle(order);losses=[];gradient=0.
        for i in order:
            raise_if_cancelled();optimizer.zero_grad(set_to_none=True);bank=banks[i]
            scores=model.score_features(bank['global_features'],bank['local_features'],bank['arm'])
            loss=sum(p['weight']*F.softplus(-(scores[p['better']]-scores[p['worse']])) for p in pairs[i])/len(pairs[i])
            if not torch.isfinite(loss):raise FloatingPointError('012 nonfinite ranking loss')
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('012 nonfinite head gradient')
            gradient+=sum(float(p.grad.abs().sum()) for p in model.parameters() if p.grad is not None)
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch,loss=statistics.mean(losses),gradient_l1=gradient,updates=len(order)))
        if epoch in snapshots:
            checkpoints.append(save_model(model,output/f'epoch_{epoch:04d}.pt',epoch=epoch,
                train_ids=ids,calibration=calibration))
    evaluation=evaluate_cache(model,cache,ids)
    report=dict(schema='tailcompile-v7-edit-context-fit-012-1',mode=mode,epochs=epochs,seed=seed,
        checkpoint=checkpoints[-1],checkpoints=checkpoints,checkpoint_sha256={p:sha(p) for p in checkpoints},
        train_ids=sorted(ids),train_condition_count=len(ids),train_pairs=sum(map(len,pairs)),
        train_pairs_by_arm={a:sum(len(p) for b,p in zip(banks,pairs) if b['arm']==a) for a in ARMS},
        initial_state_sha256=initial,final_state_sha256=state_sha256(model),
        parameter_count=sum(p.numel() for p in model.parameters()),global_dim=cache['global_dim'],local_dim=cache['local_dim'],
        calibration=calibration,source_checkpoint_sha256=cache['source_checkpoint_sha256'],
        backbone_sha256=cache['backbone_sha256'],cache_sha256=cache['cache_sha256'],
        backbone_trained=False,training_uses_cached_frozen_features=True,
        validation_labels_used_for_optimization=False,teacher_targets_used=False,tail_weight=0.,new_pnr=0,
        snapshots_for_fit_diagnostics_only=True,test_checkpoint_epoch_fixed=epochs,
        training_evaluation=evaluation,history=history,training_seconds=time.perf_counter()-started,device=str(device))
    _write(output/'training_report.json',report)
    return report
