"""012 offline edit-context ablation using the unchanged measured 011 bank.

No VPR, route parsing or new candidate collection. Grouped holdouts exclude
scorer labels only: the frozen D backbone has already seen these design families.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import itertools
import json
import math
import os
from pathlib import Path
import queue
import signal
import statistics
import sys
import time

from campaign import campaign_lock, resource_budget
from io_utils import read, write, freeze, sha, install_signal_handlers, run_trial as run_fit_process, raise_if_cancelled
from repository_guard import validate_output_roots
from residual_campaign import CODE as RESIDUAL_CODE, scientific_result

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
MODES = ('G','E')
ARMS = ('L','Y','J')
TRAIN_SEEDS = (12301,12302)
EPOCHS = 80
SNAPSHOTS = (5,20,80)
STAGES = ('plan','check','prepare','train','report')
CODE = tuple(dict.fromkeys(RESIDUAL_CODE+('edit_context_features.py','edit_context_model.py',
    'train_edit_context.py','edit_context_campaign.py','edit_context_cache_import.py',
    'run_edit_context_campaign.sh')))


def execution_options(device, jobs=None, fit_timeout=1800):
    jobs=(1 if device.startswith('cuda') else 4) if jobs is None else jobs
    if isinstance(jobs,bool) or not isinstance(jobs,int) or jobs<1 or \
        isinstance(fit_timeout,bool) or not isinstance(fit_timeout,int) or fit_timeout<1:
        raise ValueError('012 jobs and fit timeout must be positive integers')
    if device!='cpu' and not (device=='cuda' or device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('012 device must be cpu, cuda or cuda:N')
    if device.startswith('cuda') and jobs!=1:
        raise ValueError('012 CUDA uses one isolated fit process; use CPU for parallel fits')
    return dict(jobs=jobs,fit_timeout=fit_timeout,threads_per_worker=1,
                process_isolation=True)


def parallel_fit_map(tasks, *, jobs, run_task, on_complete=None, cancel_running=None):
    """Bounded submissions; no Torch model or RNG is shared between fit processes.

    Callbacks run in the controller. Completion order cannot reorder the final
    matrix. On failure the caller cancels its own subprocesses before shutdown.
    """
    if isinstance(jobs,bool) or not isinstance(jobs,int) or jobs<1:
        raise ValueError('positive parallel fit jobs required')
    tasks=list(tasks);results=[None]*len(tasks);cursor=0;pending={}
    executor=ThreadPoolExecutor(max_workers=jobs)
    try:
        while cursor<len(tasks) and len(pending)<jobs:
            pending[executor.submit(run_task,tasks[cursor])]=cursor;cursor+=1
        while pending:
            raise_if_cancelled()
            done,_=wait(pending,timeout=.2,return_when=FIRST_COMPLETED)
            # Observe all failures before scheduling another task in this batch.
            values=[(f,pending[f],f.result()) for f in sorted(done,key=lambda f:pending[f])]
            for future,index,value in values:
                del pending[future];results[index]=value
                if on_complete is not None:on_complete(index,value)
            while cursor<len(tasks) and len(pending)<jobs:
                pending[executor.submit(run_task,tasks[cursor])]=cursor;cursor+=1
    except BaseException:
        for future in pending:future.cancel()
        if cancel_running is not None:cancel_running()
        raise
    finally:
        executor.shutdown(wait=True,cancel_futures=True)
    return results


def _stop_fit_processes():
    # This controller exits after a failure; a resumed CLI starts with a fresh
    # cancellation registry. The frozen helper kills only its owned groups.
    from io_utils import _request_stop
    _request_stop(signal.SIGTERM,None)


def _verify_fit_request(request):
    path=Path(request['cache']).resolve()
    if HERE/'runs' not in path.parents or sha(path)!=request['cache_file_sha256']:
        raise ValueError('012 worker cache path/bytes changed')
    for name,wanted in request['implementation'].items():
        if name not in CODE or sha(HERE/name)!=wanted:
            raise ValueError('012 worker implementation changed: '+name)
    if set(request['implementation'])!=set(CODE):
        raise ValueError('012 worker implementation signature incomplete')
    execution_options(request['device'],1,request.get('fit_timeout',1800))
    split=request['split']
    if request['mode'] not in MODES or request['seed'] not in TRAIN_SEEDS or \
        split['heldout'] and set(split['train_ids']) & set(split['evaluation_ids']):
        raise ValueError('012 worker mode/seed/fold mismatch')


def fit_worker(request,out):
    """Exactly one signed fit, launched as a fresh Python process."""
    out=owned(out);_verify_fit_request(request)
    import torch
    from train_edit_context import fit_cache,load_cache,load_model,evaluate_cache
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    cache=load_cache(request['cache'])
    if cache['cache_sha256']!=request['cache_sha256']:
        raise ValueError('012 worker cache content changed')
    split=request['split'];mode=request['mode'];seed=request['seed']
    if split not in splits(cache['banks']):raise ValueError('012 worker uses an unfrozen fold')
    out.mkdir(parents=True,exist_ok=True)
    training=fit_cache(cache,split['train_ids'],out,mode=mode,epochs=EPOCHS,
        snapshots=SNAPSHOTS,seed=seed,learning_rate=.001,device=request['device'])
    if training['train_pairs']<=0:raise ValueError('012 no effective train-fold preferences')
    for path,wanted in training['checkpoint_sha256'].items():
        if sha(path)!=wanted:raise ValueError('012 trained checkpoint changed before evaluation')
    final=training['checkpoints'][-1]
    if not str(final).endswith('epoch_0080.pt'):raise ValueError('012 heldout epoch must be frozen80')
    model=load_model(final,device=request['device'])
    if set(model.training_metadata['train_ids'])!=set(split['train_ids']) or \
        model.training_metadata['epoch']!=EPOCHS or model.config['mode']!=mode:
        raise ValueError('012 model trained on an unexpected fold')
    evaluation=evaluate_cache(model,cache,split['evaluation_ids'])
    seed_evaluation={str(s):evaluate_cache(model,cache,split['evaluation_ids'],seed_subset={s})
                     for s in (11401,11402)}
    fit_curves=[]
    if not split['heldout']:
        for epoch,path in zip(SNAPSHOTS,training['checkpoints']):
            snapshot=load_model(path,device=request['device'])
            fit_curves.append(dict(epoch=epoch,training_evaluation=evaluate_cache(snapshot,cache,split['train_ids'])))
    _verify_fit_request(request)
    value=dict(split=split,mode=mode,seed=seed,training=training,evaluation=evaluation,
        seed_evaluation=seed_evaluation,full_fit_curves=fit_curves,
        epoch_selection_used_holdout=False,new_pnr=0,worker_pid=os.getpid(),
        worker_threads=torch.get_num_threads(),cache_file_sha256=request['cache_file_sha256'])
    freeze(out/'fit_report.json',value)
    print('TAILCOMPILE_EDIT_CONTEXT_FIT_COMPLETE',flush=True)
    return value


def plan():
    return dict(schema='edit-context-campaign-012-1',modes=dict(G='global features plus zero local tail',
        E='same head with edited-node, neighbor and before/after region features'),arms=list(ARMS),
        head_width=32,learning_rate=.001,epochs=EPOCHS,snapshots=list(SNAPSHOTS),training_seeds=list(TRAIN_SEEDS),
        split_groups=dict(full_fit_diagnostic=1,scorer_leave_family_out=2,scorer_leave_architecture_out=4),
        offline_fits=28,new_pnr=0,new_candidate_generation=False,source011_actions_and_labels_unchanged=True,
        source_D_backbone_frozen=True,frozen_backbone_family_exposure=True,
        holdout_claim='scorer labels held out; frozen backbone was trained on these families previously',
        selection_epoch=80,holdout_used_for_epoch_or_hyperparameter_selection=False,
        success_rule='reported separately; failed trials never receive fabricated WL/CPD',
        metric_rule='same measured seed arithmetic means, then equal architecture/family geometric ratios',
        prediction_measured_population='known 011 candidate bank; not a new P&R or unseen benchmark test',
        offline_best_is_deployable=False,train_fold_only_pair_scale=True,
        execution=dict(cpu_default_jobs=4,cuda_default_jobs=1,threads_per_fit=1,
                       independent_processes=True,shared_read_only_cache=True),
        runtime_role='offline cost only',Wmin_measured=False,independent_family_test=False)


def owned(path):
    path=Path(path).resolve();runs=HERE/'runs'
    if runs not in path.parents or not path.relative_to(runs).parts[0].startswith('development_012'):
        raise ValueError('012 writes require a separate development_012 cohort inside v7 runs')
    validate_output_roots((path,))
    return path


def source_path(path):
    path=Path(path).resolve()
    if HERE/'runs' not in path.parents:
        raise ValueError('012 source must be a canonical v7 runs cohort')
    return path


def _pin(path,pins,expected=None):
    path=Path(path).resolve();value=sha(path)
    if expected is not None and value!=expected:
        raise ValueError('012 frozen source changed: '+str(path))
    pins[str(path)]=value
    return value


def inspect_source(source,*,verify=True):
    """Read-only preflight. Missing large server inputs are reported explicitly."""
    source=source_path(source)
    names=('residual_protocol.json','prepared_report.json','feedback.json','feedback_report.json',
           'comparison/report.json')
    missing=[str(source/n) for n in names if not (source/n).is_file()]
    if missing:return dict(ready=False,missing=missing,plan=plan())
    protocol=read(source/'residual_protocol.json');prepared=read(source/'prepared_report.json')
    feedback=read(source/'feedback.json');summary=read(source/'feedback_report.json')
    rows=read(source/'comparison/report.json')['rows']
    ids={a['condition_id'] for a in feedback}
    expected_main={(c,s,m) for c,s,m in itertools.product(ids,(11101,11102),('B','D','L','Y','J'))}
    if len(rows)!=80 or {(r['condition_id'],r['seed'],r['method']) for r in rows}!=expected_main:
        raise ValueError('012 requires completed01180-run main matrix')
    if len(ids)!=8 or len(feedback)!=56 or summary['actions']!=56 or summary['native_runs']!=112:
        raise ValueError('012 requires the complete frozen01156-action/112-trial bank')
    checkpoint=Path(prepared['checkpoint']);old008=Path(protocol['source008'])
    records_path=old008/'teacher_records.json'
    missing += [str(p) for p in (checkpoint,records_path) if not p.is_file()]
    if missing:return dict(ready=False,missing=missing,plan=plan())
    records=[dict(r,architecture=r['condition_id'].split('/')[1])
             for r in read(records_path)['records'] if r['condition_id'] in ids]
    if len(records)!=8 or {r['condition_id'] for r in records}!=ids or any(r['split']!='train' for r in records):
        raise ValueError('012 records must exactly match the original011 train conditions')
    for record in records:
        if not Path(record['graph']).is_file():missing.append(record['graph'])
    for action in feedback:
        if not isinstance(action.get('action'),str) or not Path(action['action']).is_file():
            missing.append(str(action.get('action')))
        for trial in action.get('trials',[]):
            if not Path(trial['vpr_manifest']).is_file():missing.append(trial['vpr_manifest'])
    if missing:return dict(ready=False,missing=sorted(set(missing)),plan=plan())
    pins={}
    if verify:
        for name,wanted in protocol['implementation'].items():_pin(HERE/name,pins,wanted)
        for name in names:_pin(source/name,pins)
        _pin(checkpoint,pins,prepared['checkpoint_sha256'])
        if protocol['base_checkpoint_sha256']!=prepared['checkpoint_sha256']:
            raise ValueError('011 D identities disagree')
        _pin(records_path,pins,protocol['source_inputs'][str(records_path.resolve())])
        for record in records:_pin(record['graph'],pins,protocol['source_inputs'][record['graph']])
        expected={(c,a,k) for c,a,k in itertools.product(ids,ARMS,(1,2))}
        actual=set();native_count=0
        identities=[]
        for action in feedback:
            arm=action['arm'];candidate=action['candidate_id'];gen=action['generation']
            if arm=='D':
                if candidate!='identity':raise ValueError('011 D control must be identity')
                identities.append(action['condition_id'])
            else:
                if arm not in ARMS or candidate not in (arm.lower()+'_s1',arm.lower()+'_s2'):
                    raise ValueError('012 measured bank is outside frozen011 finite actions')
                actual.add((action['condition_id'],arm,int(candidate[-1])))
            _pin(action['action'],pins,gen['target_sha256'])
            if {r['seed'] for r in action['trials']}!={11401,11402} or len(action['trials'])!=2:
                raise ValueError('012 requires the original complete two-seed feedback labels')
            for trial in action['trials']:
                manifest=read(trial['vpr_manifest']);signature=manifest['signature'];metrics=manifest['metrics']
                if not scientific_result(trial) or manifest['status']!=trial['status']:
                    raise ValueError('012 source trial is incomplete/infrastructure failure')
                expected_code={'success':0,'unroutable':2,'packing_or_placement_failure':1}.get(trial['status'])
                if expected_code is None or manifest['returncode']!=expected_code:
                    raise ValueError('012 source native return code disagrees with scientific status')
                cmd=signature['command']
                if int(cmd[cmd.index('--seed')+1])!=trial['seed'] or \
                    int(cmd[cmd.index('--route_chan_width')+1])!=trial['width'] or trial['condition_id']!=action['condition_id']:
                    raise ValueError('012 measured native seed/width/condition identity changed')
                binary_path=next(path for path in signature['inputs'] if Path(path).name=='binary_identity.json')
                _pin(binary_path,pins,signature['inputs'][binary_path])
                if read(binary_path)['sha256']!=protocol['binary_sha256']:
                    raise ValueError('012 recorded native binary differs from frozen011 backend')
                for k in ('wirelength','cpd_ns'):
                    if metrics[k]!=trial[k]:raise ValueError('012 trial labels differ from native manifest')
                if signature['inputs'][signature['env']['TC_GP_INIT']]!=gen['ap_init_sha256'] or \
                    trial['ap_init_sha256']!=gen['ap_init_sha256']:
                    raise ValueError('012 measured action differs from native-consumed initializer')
                _pin(trial['vpr_manifest'],pins);native_count+=1
        if actual!=expected or len(identities)!=8 or set(identities)!=ids or native_count!=112:
            raise ValueError('012 incomplete measured candidate support')
    return dict(ready=True,missing=[],source=str(source),checkpoint=str(checkpoint.resolve()),
        checkpoint_sha256=prepared['checkpoint_sha256'],records=records,feedback=feedback,
        source_inputs=pins,plan=plan())


def splits(banks):
    meta={b['condition_id']:(b['family'],b['architecture']) for b in banks}
    ids=sorted(meta)
    if len(ids)!=8 or len({f for f,a in meta.values()})!=2 or len({a for f,a in meta.values()})!=4:
        raise ValueError('012 requires the frozen two-family by four-architecture panel')
    result=[dict(id='full_fit',group='full_fit_diagnostic',train_ids=ids,evaluation_ids=ids,
                 heldout=False)]
    for group,index in (('scorer_leave_family_out',0),('scorer_leave_architecture_out',1)):
        for value in sorted({item[index] for item in meta.values()}):
            test=[c for c in ids if meta[c][index]==value]
            result.append(dict(id=group+'__'+value,group=group,heldout_value=value,
                train_ids=[c for c in ids if c not in test],evaluation_ids=test,heldout=True))
    return result


def aggregate(rows):
    """One prediction per condition/arm in each mode, seed and split group."""
    keys=[(r['condition_id'],r['arm']) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('012 duplicate heldout condition prediction')
    result={}
    for arm in ARMS:
        arr=[r for r in rows if r['arm']==arm];by_family=defaultdict(list);metrics={}
        for key in ('WL','CPD'):
            by_family.clear()
            for row in arr:
                a,b=row['selected_'+key],row['D_'+key]
                if a is not None and b is not None and a>0 and b>0:
                    by_family[row['family']].append(a/b)
            family={f:math.exp(statistics.mean(math.log(v) for v in values)) for f,values in by_family.items()}
            ratio=math.exp(statistics.mean(math.log(v) for v in family.values())) if family else None
            metrics[key]=dict(ratio=ratio,percent_change=None if ratio is None else (ratio-1)*100,
                family_ratios=family,conditions_with_qor=sum(len(v) for v in by_family.values()))
        result[arm]=dict(conditions=len(arr),identity_selections=sum(r['selected_candidate_id']=='identity' for r in arr),
            selected_successes=sum(r['selected_success_count'] for r in arr),D_successes=sum(r['D_success_count'] for r in arr),
            selected_is_offline_best=sum(r['selected_is_offline_best'] for r in arr),metrics=metrics,
            mean_q_regret=statistics.mean(r['q_regret'] for r in arr if r['q_regret'] is not None)
                if any(r['q_regret'] is not None for r in arr) else None)
    return result


def validate_paired_fits(fits):
    """Feature visibility is the only intended G/E training difference."""
    grouped=defaultdict(dict)
    for fit in fits:
        key=(fit['split']['id'],fit['seed']);mode=fit['mode']
        if mode not in MODES or mode in grouped[key]:raise ValueError('012 duplicate/unknown paired control')
        if fit['training']['mode']!=mode:raise ValueError('012 fit mode differs from training evidence')
        grouped[key][mode]=fit
    fields=('initial_state_sha256','parameter_count','train_ids','train_pairs','train_pairs_by_arm',
            'calibration','source_checkpoint_sha256','backbone_sha256','cache_sha256',
            'global_dim','local_dim','epochs','seed','device')
    for key,pair in grouped.items():
        if set(pair)!=set(MODES):raise ValueError('012 missing G/E paired control')
        if pair['G']['split']!=pair['E']['split']:raise ValueError('012 paired fold definitions differ')
        for name in fields:
            if pair['G']['training'][name]!=pair['E']['training'][name]:
                raise ValueError('012 unmatched G/E training control: '+name)
    return dict(paired_controls_verified=True,pairs=len(grouped),matched_fields=list(fields))


def feature_contrast(local_rows,global_rows):
    """Compare E-selected versus G-selected actions on the same heldout bank."""
    lookup={(r['condition_id'],r['arm']):r for r in global_rows}
    if len(lookup)!=len(global_rows) or {(r['condition_id'],r['arm']) for r in local_rows}!=set(lookup):
        raise ValueError('012 G/E prediction populations differ')
    paired=[]
    for row in local_rows:
        control=lookup[row['condition_id'],row['arm']]
        paired.append(dict(row,D_WL=control['selected_WL'],D_CPD=control['selected_CPD'],
            D_success_count=control['selected_success_count']))
    out=aggregate(paired)
    for arm in out:
        out[arm]['baseline']='G selected candidate'
        out[arm]['method']='E selected candidate'
        out[arm]['G_selected_successes']=out[arm].pop('D_successes')
        out[arm]['E_selected_successes']=out[arm].pop('selected_successes')
        out[arm]['same_candidate_choices']=sum(row['selected_digest']==lookup[row['condition_id'],arm]['selected_digest']
            for row in local_rows if row['arm']==arm)
        out[arm].pop('mean_q_regret')
        out[arm].pop('identity_selections')
        out[arm].pop('selected_is_offline_best')
    return out


class Campaign:
    def __init__(self,args):self.args,self.out=args,owned(args.out)

    def setup(self):
        self.source=inspect_source(self.args.source)
        if not self.source['ready']:raise ValueError('012 server inputs missing: '+str(self.source['missing']))
        if self.args.source==self.out or self.args.source in self.out.parents or self.out in self.args.source.parents:
            raise ValueError('012 source and output must be disjoint')
        origin=getattr(self.args,'cache_from',None)
        if origin and (origin==self.out or origin in self.out.parents or self.out in origin.parents):
            raise ValueError('012 imported cache and output must be disjoint')
        freeze(self.out/'edit_context_protocol.json',dict(plan=plan(),source=self.source['source'],
            source_inputs=self.source['source_inputs'],base_checkpoint=self.source['checkpoint'],
            base_checkpoint_sha256=self.source['checkpoint_sha256'],implementation={n:sha(HERE/n) for n in CODE},
            cache_from=str(self.args.cache_from) if getattr(self.args,'cache_from',None) else None,
            execution=execution_options(self.args.device,getattr(self.args,'jobs',None),
                                        getattr(self.args,'fit_timeout',1800))))
        freeze(self.out/'execution'/self.args.stage/'protocol.json',dict(device=self.args.device,
            **execution_options(self.args.device,getattr(self.args,'jobs',None),getattr(self.args,'fit_timeout',1800))))

    def prepare(self):
        from train_edit_context import prepare_cache,save_cache,load_cache
        path=self.out/'feature_cache.pt';report=self.out/'prepared_report.json'
        if report.exists():
            p=read(report)
            path=Path(p['cache'])
            if sha(path)!=p['cache_file_sha256']:raise ValueError('012 prepared cache file changed')
            load_cache(path);return
        if getattr(self.args,'cache_from',None):
            from edit_context_cache_import import import_cache
            imported=import_cache(self.source,self.args.cache_from,
                                  current_implementation={n:sha(HERE/n) for n in CODE})
            freeze(report,dict(imported,complete=True,conditions=8,new_pnr=0,
                source_actions_unchanged=True,splits=splits(load_cache(imported['cache'])['banks'])))
            return
        started=time.perf_counter()
        cache=prepare_cache(self.source['records'],self.source['checkpoint'],self.source['feedback'],
            expected_sha256=self.source['checkpoint_sha256'])
        save_cache(cache,path)
        freeze(report,dict(complete=True,cache=str(path),cache_file_sha256=sha(path),cache_sha256=cache['cache_sha256'],
            conditions=8,banks=len(cache['banks']),splits=splits(cache['banks']),global_dim=cache['global_dim'],
            local_dim=cache['local_dim'],cache_file_bytes=path.stat().st_size,
            feature_cache_seconds=time.perf_counter()-started,new_pnr=0,source_actions_unchanged=True))

    def cache(self):
        from train_edit_context import load_cache
        prepared=read(self.out/'prepared_report.json');path=Path(prepared['cache'])
        if sha(path)!=prepared['cache_file_sha256']:raise ValueError('012 prepared feature cache changed')
        cache=load_cache(path)
        if cache['cache_sha256']!=prepared['cache_sha256']:raise ValueError('012 cache content pin changed')
        return cache

    def train(self):
        cache=self.cache()
        options=execution_options(self.args.device,getattr(self.args,'jobs',None),getattr(self.args,'fit_timeout',1800))
        budget=resource_budget(min(options['jobs'],28),1024,reserve_mib=1024)
        jobs=budget['effective'];cores=queue.Queue()
        for core in budget['physical_cores']:cores.put(core)
        implementation={n:sha(HERE/n) for n in CODE}
        prepared=read(self.out/'prepared_report.json')
        tasks=list(itertools.product(splits(cache['banks']),MODES,TRAIN_SEEDS))
        freeze(self.out/'execution/train/parallel_protocol.json',dict(
            **options,fit_count=len(tasks),
            matrix=[[s['id'],m,seed] for s,m,seed in tasks],
            cache=prepared['cache'],cache_file_sha256=prepared['cache_file_sha256']))
        # Current free RAM and CPU availability are observations, not immutable
        # experimental inputs. A normal resume must not fail when they change.
        observations=self.out/'execution/train/parallel_runs'
        observation=observations/f'attempt_{len(list(observations.glob("attempt_*")))+1:04d}'/'metrics.json'
        write(observation,dict(effective_jobs=jobs,budget=budget,pid=os.getpid(),
            started_unix=time.time(),host=list(os.uname()) if hasattr(os,'uname') else None))
        print(json.dumps(dict(event='edit_context_parallel_start',requested_jobs=options['jobs'],
            effective_jobs=jobs,threads_per_fit=1,planned=len(tasks))),flush=True)

        def verify_value(value,split,mode,seed):
            training=value['training']
            if value['split']!=split or value['mode']!=mode or value['seed']!=seed or \
                training['mode']!=mode or training['seed']!=seed or training['epochs']!=EPOCHS or \
                training['device']!=self.args.device or training['cache_sha256']!=cache['cache_sha256'] or \
                training['source_checkpoint_sha256']!=cache['source_checkpoint_sha256'] or \
                training['train_ids']!=sorted(split['train_ids']) or value['worker_threads']!=1:
                raise ValueError('012 completed fit differs from its frozen request')
            for path,wanted in training['checkpoint_sha256'].items():
                if sha(path)!=wanted:raise ValueError('012 cached trained checkpoint changed')

        def run_task(task):
            raise_if_cancelled()
            split,mode,seed=task
            folder=self.out/'models'/split['id']/mode/f's{seed}'
            durable=folder/'fit_report.json'
            if durable.exists():
                value=read(durable)
                verify_value(value,split,mode,seed)
            else:
                request=dict(cache=prepared['cache'],cache_file_sha256=prepared['cache_file_sha256'],
                    cache_sha256=cache['cache_sha256'],split=split,mode=mode,seed=seed,
                    device=self.args.device,fit_timeout=options['fit_timeout'],implementation=implementation)
                existing=sorted(p for p in folder.glob('attempt_*') if p.is_dir())
                value=None
                for attempt in existing:
                    if (attempt/'fit_report.json').is_file() and (attempt/'worker/manifest.json').is_file() and \
                        read(attempt/'worker/manifest.json')['status']=='complete':
                        if read(attempt/'request.json')!=request:raise ValueError('012 recovered fit request changed')
                        value=read(attempt/'fit_report.json');verify_value(value,split,mode,seed)
                        value['worker_manifest']=str(attempt/'worker/manifest.json');break
                if value is None:
                    if len(existing)>=2:raise RuntimeError('012 bounded offline fit retry budget exhausted')
                    attempt=folder/f'attempt_{len(existing)+1:04d}'
                    freeze(attempt/'request.json',request)
                    core=cores.get()
                    try:
                        result=run_fit_process(attempt/'worker',[sys.executable,HERE/'edit_context_campaign.py',
                            'fit-worker','--request',attempt/'request.json','--out',attempt],
                            inputs=[attempt/'request.json',prepared['cache']]+[HERE/n for n in CODE],
                            affinity=core,timeout=options['fit_timeout'],
                            marker='TAILCOMPILE_EDIT_CONTEXT_FIT_COMPLETE',prune=False)
                    finally:cores.put(core)
                    if result['status']!='complete':
                        raise RuntimeError('012 isolated fit failed; resume completed fits after diagnosis: '+result['log'])
                    value=read(attempt/'fit_report.json');verify_value(value,split,mode,seed)
                    value['worker_manifest']=str(attempt/'worker/manifest.json')
                freeze(durable,value)
            return dict(path=str(durable),sha256=sha(durable),split_id=split['id'],mode=mode,seed=seed)

        progress={}
        def completed(index,value):
            progress[index]=value
            write(self.out/'train_progress_report.json',dict(completed=len(progress),planned=28,
                effective_jobs=jobs,fits=[progress[i] for i in sorted(progress)]))
            print(json.dumps(dict(event='edit_context_fit_complete',split=value['split_id'],mode=value['mode'],seed=value['seed'],
                completed=len(progress),planned=28)),flush=True)
        fits=parallel_fit_map(tasks,jobs=jobs,run_task=run_task,
            on_complete=completed,cancel_running=_stop_fit_processes)
        summary=self.out/'trained_report.json'
        first_jobs=read(summary)['effective_jobs'] if summary.exists() else jobs
        freeze(summary,dict(complete=len(fits)==28,planned=28,fits=fits,new_pnr=0,
            no_holdout_checkpoint_selection=True,frozen_backbone_family_exposure=True,
            effective_jobs=first_jobs,isolated_processes=True,threads_per_fit=1))

    def report(self):
        records=read(self.out/'trained_report.json');fits=[];raw_fits=[];grouped=defaultdict(list);grouped_seed=defaultdict(list)
        if len(records['fits'])!=28 or not records['complete']:raise ValueError('012 offline fit matrix incomplete')
        expected={(s['id'],m,seed) for s,m,seed in itertools.product(
            read(self.out/'prepared_report.json')['splits'],MODES,TRAIN_SEEDS)}
        actual={(r['split_id'],r['mode'],r['seed']) for r in records['fits']}
        if actual!=expected:raise ValueError('012 offline fit matrix has missing/unexpected conditions')
        for entry in records['fits']:
            if sha(entry['path'])!=entry['sha256']:raise ValueError('012 fitted evidence changed')
            fit=read(entry['path']);split=fit['split'];evaluation=fit['evaluation']
            if (entry['split_id'],entry['mode'],entry['seed'])!=(split['id'],fit['mode'],fit['seed']):
                raise ValueError('012 fit index differs from measured training evidence')
            raw_fits.append(fit)
            if split['heldout'] and set(split['train_ids']) & set(split['evaluation_ids']):
                raise ValueError('012 heldout scorer label leakage')
            if fit['training']['calibration']['train_ids']!=sorted(split['train_ids']):
                raise ValueError('012 fold scale used unexpected labels')
            grouped[split['group'],fit['mode'],fit['seed']].extend(evaluation['rows'])
            predictions={(r['condition_id'],r['arm']):r['selected_digest'] for r in evaluation['rows']}
            for vpr_seed,value in fit['seed_evaluation'].items():
                if {(r['condition_id'],r['arm']):r['selected_digest'] for r in value['rows']}!=predictions:
                    raise ValueError('012 historical seed evaluation changed model-selected actions')
                grouped_seed[split['group'],fit['mode'],fit['seed'],vpr_seed].extend(value['rows'])
            fits.append(dict(split=split,mode=fit['mode'],seed=fit['seed'],ranking=evaluation['ranking'],
                identity_selections=evaluation['identity_selections'],candidate_selections=evaluation['candidate_selections'],
                rows=evaluation['rows'],train_pairs=fit['training']['train_pairs'],
                parameter_count=fit['training']['parameter_count'],training_seconds=fit['training']['training_seconds'],
                full_fit_curves=fit['full_fit_curves']))
        summaries={f'{group}/{mode}/s{seed}':aggregate(rows) for (group,mode,seed),rows in grouped.items()}
        seed_summaries={f'{group}/{mode}/s{seed}/historical_vpr_s{vpr_seed}':aggregate(rows)
                        for (group,mode,seed,vpr_seed),rows in grouped_seed.items()}
        contrasts={f'{group}/s{seed}':feature_contrast(grouped[group,'E',seed],grouped[group,'G',seed])
                   for group,mode,seed in grouped if mode=='E'}
        report=dict(schema='edit-context-evaluation-012-1',complete=True,offline_fits=28,new_pnr=0,
            summaries=summaries,E_over_G_contrasts=contrasts,historical_vpr_seed_summaries=seed_summaries,
            historical_vpr_seed_predictions_identical=True,training_random_seeds_are_not_new_design_cases=True,
            fits=fits,head_epochs=80,source_actions_and_labels_unchanged=True,
            group_holdout_is_scorer_labels_only=True,frozen_backbone_family_exposure=True,
            reused_measurements_not_new_implementation_pnr=True,oracle_is_deployable=False,
            holdout_used_for_checkpoint_selection=False,independent_family_test=False,Wmin_measured=False)
        report['paired_training_controls']=validate_paired_fits(raw_fits)
        write(self.out/'edit_context_report.json',report)
        print(json.dumps(dict(complete=True,offline_fits=28,new_pnr=0,report=str(self.out/'edit_context_report.json'))),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('stage',choices=STAGES+('fit-worker',))
    parser.add_argument('--source',type=Path,default=HERE/'runs/development_011_residual')
    parser.add_argument('--out',type=Path,default=HERE/'runs/development_012_edit_context_parallel')
    parser.add_argument('--device',default='cpu');parser.add_argument('--jobs',type=int)
    parser.add_argument('--fit-timeout',type=int,default=1800)
    parser.add_argument('--cache-from',type=Path);parser.add_argument('--request',type=Path)
    args=parser.parse_args()
    try:args.jobs=execution_options(args.device,args.jobs,args.fit_timeout)['jobs']
    except ValueError as exc:parser.error(str(exc))
    args.source=source_path(args.source);args.out=owned(args.out);install_signal_handlers()
    if args.cache_from:args.cache_from=source_path(args.cache_from)
    if args.stage=='fit-worker':
        if args.request is None:parser.error('fit-worker requires --request')
        fit_worker(read(args.request),args.out);return
    if args.stage=='plan':print(json.dumps(plan(),indent=2));return
    if args.stage=='check':
        result=inspect_source(args.source)
        print(json.dumps({k:v for k,v in result.items() if k not in ('records','feedback','source_inputs')},indent=2))
        raise SystemExit(0 if result['ready'] else 2)
    with campaign_lock(args.out,'edit-context-'+args.stage):
        runner=Campaign(args);runner.setup();getattr(runner,args.stage)()


if __name__=='__main__':main()
