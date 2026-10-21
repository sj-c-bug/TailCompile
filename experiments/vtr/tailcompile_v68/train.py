"""GPU-only supervised/improvement-distillation training of the deployed model.

No VPR is launched here. Pretraining uses native-GP teachers; fine-tuning uses
only train-family initializers whose *actual warm-start VPR* labels beat the
paired Default under the frozen CPD/wirelength guards. The scorer/generator
confusion is prevented by checkpoint schema and recorded weight fingerprints.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import random
import time
from collections import OrderedDict

os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import torch

from common import emit, fingerprint, freeze, read, sha, write
from model import HierarchicalInitializer, SCHEMA, to_device
from supervision import SUPERVISION_SCHEMA, project_teacher


def weights_sha(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def atomic_checkpoint(path,value):
    temporary=path.with_suffix('.pt.tmp');torch.save(value,temporary);temporary.replace(path)


def validate_dataset(dataset):
    families={}
    for e in dataset['examples']:
        if e['split'] not in ('train','validation'):
            raise ValueError('Final-test example entered generator training')
        old=families.setdefault(e['family'],e['split'])
        if old!=e['split']:
            raise ValueError('Family leakage between train/validation')
        if sha(e['graph'])!=e['graph_sha256']:
            raise ValueError('Training graph changed')
    if not any(e['split']=='train' for e in dataset['examples']):
        raise ValueError('No train families')
    return families


def rows_for_training(dataset,feedback=None):
    rows=list(dataset['examples'])
    if feedback is None:
        return rows
    replacements={}
    for row in feedback['selected']:
        if row['split']!='train':
            raise ValueError('VPR intervention feedback must be train-family only')
        if sha(row['target'])!=row['target_sha256']:
            raise ValueError('Selected feedback initializer target changed')
        key=row['design'],row['width']
        if key in replacements:
            raise ValueError('Duplicate feedback target')
        replacements[key]=row
    result=[]
    for e in rows:
        replacement=replacements.get((e['design'],e['width']))
        if replacement:
            if replacement['family']!=e['family'] or e['split']!='train':
                raise ValueError('Feedback context does not match original training split')
            e=dict(e,target=read(replacement['target']),weight=float(e.get('weight',1))*2.,
                   target_kind='real_vpr_selected_initializer')
        result.append(e)
    known={(e['design'],e['width']) for e in rows if e['split']=='train'}
    if set(replacements)-known:
        raise ValueError('Feedback target does not match an original training condition')
    return result


def train(a):
    if a.device=='auto':
        a.device='cuda' if torch.cuda.is_available() else 'cpu'
    device=torch.device(a.device)
    if device.type=='cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.set_num_threads(a.threads)
    random.seed(a.seed);torch.manual_seed(a.seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(a.seed)
    dataset=read(a.dataset);families=validate_dataset(dataset)
    if dataset.get('tile_size',a.tile_size)!=a.tile_size:
        raise ValueError('Training tile_size differs from the frozen prepared dataset')
    feedback=read(a.feedback) if a.feedback else None
    if feedback is not None and not a.initialize:
        raise ValueError('Fine-tuning must initialize the pretrained generator')
    if feedback is not None and not feedback.get('all_route_outcomes_known',True):
        raise ValueError('Feedback contains censored or infrastructure outcomes; finish explicit retries first')
    if feedback is not None and feedback['generator_checkpoint_sha256']!=sha(a.initialize):
        raise ValueError('Feedback labels were collected using a different generator checkpoint')
    rows=rows_for_training(dataset,feedback)
    train_rows=[r for r in rows if r['split']=='train']
    validation=[r for r in rows if r['split']=='validation']
    if not validation and not a.allow_no_validation:
        raise ValueError('Validation families required; --allow-no-validation is a smoke-test-only override')
    a.out.mkdir(parents=True,exist_ok=True)
    signature=dict(schema=SCHEMA,supervision_schema=SUPERVISION_SCHEMA,
        dataset_sha256=sha(a.dataset),
        feedback_sha256=sha(a.feedback) if a.feedback else None,
        initial_checkpoint_sha256=sha(a.initialize) if a.initialize else None,
        epochs=a.epochs,hidden=a.hidden,chunk_size=a.chunk_size,lr=a.lr,seed=a.seed,
        code={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('model.py'),
                                    Path(__file__).with_name('supervision.py')]})
    freeze(a.out/'training_signature.json',signature)
    if (a.out/'training_report.json').exists():
        report=read(a.out/'training_report.json')
        if report['complete'] and sha(a.out/'generator.pt')==report['checkpoint_sha256']:
            completed_audit_path=a.out/'supervision_audit.json'
            if not completed_audit_path.exists():
                raise ValueError('Completed training is missing its supervision audit')
            completed_audit=read(completed_audit_path)
            if (completed_audit.get('schema')!=SUPERVISION_SCHEMA or not completed_audit.get('complete')
                    or fingerprint(completed_audit['conditions'])!=report['supervision_audit']['conditions_sha256']):
                raise ValueError('Completed training supervision audit changed or is incomplete')
            emit('reuse_complete_training',path=str(a.out/'training_report.json'));return
    # Cache CPU graph objects only. Graphs move to GPU one at a time, rather than
    # retaining the complete 20-30 family dataset on an 80GB accelerator.
    graphs=OrderedDict()
    def cpu_graph(row):
        if row['graph'] not in graphs:
            graphs[row['graph']]=torch.load(row['graph'],map_location='cpu',weights_only=True)
            while len(graphs)>a.graph_cache:
                graphs.popitem(last=False)
        graphs.move_to_end(row['graph'])
        return graphs[row['graph']]
    def graph(row):
        return to_device(cpu_graph(row),device)
    # Only dimensions are needed for construction; do not keep the first whole
    # GPU graph alive while later examples are on the accelerator.
    first=cpu_graph(train_rows[0])
    if a.initialize:
        initial=torch.load(a.initialize,map_location=device,weights_only=True)
        if initial.get('schema')!=SCHEMA:
            raise ValueError('Initialization checkpoint is not the deployed v6.8 generator')
        model=HierarchicalInitializer(**initial['model_config']).to(device)
        model.load_state_dict(initial['model_state'])
    else:
        model=HierarchicalInitializer(first['cluster_x'].shape[1],first['region_x'].shape[1],
            hidden=a.hidden,chunk_size=a.chunk_size,demand_width=first['demand'].shape[1]).to(device)
    del first
    initial_weights=weights_sha(model.state_dict())
    optimizer=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=1e-4)
    start_epoch=1;history=[];best=math.inf
    latest=a.out/'latest.pt'
    if latest.exists():
        saved=torch.load(latest,map_location=device,weights_only=True)
        if saved['signature']!=signature:
            raise ValueError('Training resume identity mismatch')
        model.load_state_dict(saved['model_state']);optimizer.load_state_dict(saved['optimizer'])
        start_epoch=saved['epoch']+1;history=saved['history'];best=saved['best_loss']
        initial_weights=saved['initial_weights_sha256']
        torch.set_rng_state(saved['torch_rng'].cpu())
        if device.type=='cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        emit('training_resumed',epoch=start_epoch)
    emit('training_start',role='actual_coordinate_generator',device=str(device),
        device_name=torch.cuda.get_device_name(device) if device.type=='cuda' else 'CPU',
        train_families=sorted(k for k,v in families.items() if v=='train'),
        validation_families=sorted(k for k,v in families.items() if v=='validation'),
        training_examples=len(train_rows),validation_examples=len(validation),
        real_feedback_targets=len(feedback['selected']) if feedback else 0)
    # Teachers are supervisory outputs, never graph features.  Prepare them
    # lazily on first use, on CPU, without modifying frozen inputs or keeping a
    # second dataset of device graphs in accelerator memory.
    projected_targets={}
    audit_path=a.out/'supervision_audit.json'
    condition_audits={}
    if audit_path.exists():
        previous_audit=read(audit_path)
        if previous_audit.get('schema')!=SUPERVISION_SCHEMA:
            raise ValueError('Supervision audit schema changed; use a new training output directory')
        condition_audits={(r['design'],r['width'],r['split']):r
                          for r in previous_audit.get('conditions',[])}
    def save_supervision_audit(complete=False):
        conditions=[condition_audits[k] for k in sorted(condition_audits)]
        expected={(r['design'],r['width'],r['split']) for r in rows}
        if complete and set(condition_audits)!=expected:
            raise ValueError('Cannot complete training without an audit for every supervision condition')
        counts=('movable_nodes','conflicted_nodes','reassigned_nodes','boundary_adjusted_nodes')
        summary={name:sum(r[name] for r in conditions) for name in counts}
        summary['max_displacement']=max((r['max_displacement'] for r in conditions),default=0.)
        summary['mean_displacement']=(sum(r['mean_displacement']*r['movable_nodes']
            for r in conditions)/max(summary['movable_nodes'],1))
        resources={}
        for condition in conditions:
            for resource,count in condition['conflict_resources'].items():
                resources[resource]=resources.get(resource,0)+count
        summary['conflict_resources']=resources
        value=dict(schema=SUPERVISION_SCHEMA,complete=complete,partial=not complete,
            audited_conditions=len(conditions),expected_conditions=len(rows),
            summary=summary,conditions=conditions,conditions_sha256=fingerprint(conditions),
            raw_dataset_sha256=signature['dataset_sha256'],
            note='CPU-only target projection; raw targets, graph features and inference capacity constraints are unchanged.')
        write(audit_path,value)
        return value
    save_supervision_audit()
    def loss_for(row,details=False,phase='train'):
        try:
            key=(row['design'],row['width'],row['split'])
            if key not in projected_targets:
                # Physical dimensions are supervision-only distance metadata,
                # not teacher coordinates or extra model input features.
                projection_data=dict(cpu_graph(row))
                device_path=Path(row['graph']).with_name('device.json')
                if device_path.exists():
                    device_metadata=read(device_path)
                    if 'grid_width' in device_metadata and 'grid_height' in device_metadata:
                        projection_data['teacher_axis_scale']=[device_metadata['grid_width'],
                                                               device_metadata['grid_height']]
                target,audit=project_teacher(projection_data,row['target'],chunk_size=a.chunk_size)
                projected_targets[key]=target
                condition_audits[key]=dict(design=row['design'],width=row['width'],
                    split=row['split'],**{k:v for k,v in audit.items() if k!='schema'})
                save_supervision_audit()
                emit('teacher_projection',design=row['design'],width=row['width'],
                    split=row['split'],**audit)
            data=graph(row);target=projected_targets[key]
            return model.loss(data,torch.tensor(target['teacher_region'],device=device),
                torch.tensor(target['teacher_xy'],dtype=torch.float32,device=device),
                torch.tensor(target['movable'],dtype=torch.bool,device=device),return_details=details,
                geometry_teacher_xy=torch.tensor(row['target']['teacher_xy'],dtype=torch.float32,device=device))
        except Exception as error:
            emit('training_example_error',design=row['design'],width=row['width'],
                 split=row['split'],phase=phase,error=str(error))
            raise RuntimeError(f'Generator {phase} failed: {row["design"]}/w{row["width"]} '
                               f'({row["split"]}): {error}') from error
    began=time.monotonic()
    for epoch in range(start_epoch,a.epochs+1):
        model.train();order=list(train_rows);random.Random(a.seed+epoch).shuffle(order)
        total=0.;components=[]
        for row in order:
            optimizer.zero_grad(set_to_none=True)
            loss,details=loss_for(row,True)
            weighted=loss*float(row.get('weight',1.))
            if not torch.isfinite(weighted):
                raise RuntimeError(f'Nonfinite generator loss: {row["design"]}/w{row["width"]}')
            weighted.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);optimizer.step()
            total+=float(loss.detach());components.append(details)
        model.eval()
        with torch.no_grad():
            validation_loss=statistics_mean([float(loss_for(r,phase='validation')) for r in validation]) if validation else total/len(order)
        row=dict(epoch=epoch,train_loss=total/len(order),validation_loss=validation_loss,
            elapsed_seconds=round(time.monotonic()-began,2),
            mean_components={k:statistics_mean([d[k] for d in components]) for k in components[0]})
        history.append(row);emit('epoch',**row)
        epoch_audit=save_supervision_audit()
        state=dict(schema=SCHEMA,model_config=model.config,model_state=model.state_dict(),
            supervision_schema=SUPERVISION_SCHEMA,supervision_summary=epoch_audit['summary'],
            supervision_audit_sha256=epoch_audit['conditions_sha256'],
            role='native_gp_initializer_generator',tile_size=a.tile_size,
            training_families=sorted(k for k,v in families.items() if v=='train'),
            validation_families=sorted(k for k,v in families.items() if v=='validation'),
            signature=signature,epoch=epoch)
        if validation_loss<best:
            best=validation_loss;atomic_checkpoint(a.out/'generator.pt',state)
        # Both optimizer and RNG state survive an interrupted GPU instance.
        atomic_checkpoint(latest,dict(state,optimizer=optimizer.state_dict(),history=history,
            best_loss=best,initial_weights_sha256=initial_weights,torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type=='cuda' else None))
        write(a.out/'progress.json',dict(complete=False,last_epoch=epoch,best_loss=best,history=history))
    final=torch.load(a.out/'generator.pt',map_location='cpu',weights_only=True)
    final_weights=weights_sha(final['model_state'])
    supervision_audit=save_supervision_audit(complete=True)
    report=dict(complete=True,schema=SCHEMA,supervision_schema=SUPERVISION_SCHEMA,
        supervision_audit=dict(path=str(audit_path),summary=supervision_audit['summary'],
                               conditions_sha256=supervision_audit['conditions_sha256']),
        role='actual_deployed_generator',
        epochs_completed=a.epochs,best_epoch=final['epoch'],best_validation_loss=best,
        history=history,checkpoint_sha256=sha(a.out/'generator.pt'),
        initial_weights_sha256=initial_weights,final_weights_sha256=final_weights,
        weights_changed=initial_weights!=final_weights,
        training_families=final['training_families'],validation_families=final['validation_families'],
        final_test_examples=0,real_feedback_targets=len(feedback['selected']) if feedback else 0,
        measurement='Teacher reconstruction is not evidence of routed W_min improvement; evaluate native VPR separately.')
    write(a.out/'training_report.json',report);emit('training_complete',**{k:report[k] for k in
        ('epochs_completed','best_epoch','weights_changed','real_feedback_targets')})


def statistics_mean(values):
    return sum(values)/len(values)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--initialize',type=Path);p.add_argument('--feedback',type=Path)
    p.add_argument('--device',default='auto');p.add_argument('--epochs',type=int,default=150)
    p.add_argument('--hidden',type=int,default=64);p.add_argument('--chunk-size',type=int,default=512)
    p.add_argument('--tile-size',type=int,default=4);p.add_argument('--threads',type=int,default=2)
    p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--seed',type=int,default=6801)
    p.add_argument('--graph-cache',type=int,default=2,help='Bound CPU tensor cache; do not retain all design graphs')
    p.add_argument('--allow-no-validation',action='store_true')
    a=p.parse_args()
    if min(a.epochs,a.hidden,a.chunk_size,a.tile_size,a.threads,a.graph_cache)<1 or a.lr<=0:
        p.error('Invalid positive training budget')
    train(a)
