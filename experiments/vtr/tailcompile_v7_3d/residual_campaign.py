"""011: learn finite, feasible residual edits of frozen 010 D using measured QoR.

Separate train/validation/test seeds; validation can disable each edit arm.
All arms retain native GP/APPack/SA/routing. No teacher imitation or TailRisk.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time

from structured_campaign import CODE as STRUCTURED_CODE, contrast, source008, verify_checkpoint
from qor_campaign import QorCampaign, panel, legacy, check as check008
from campaign import campaign_lock, resource_budget
from io_utils import read, write, freeze, sha, run_trial, install_signal_handlers, vpr_command, prune_derived
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
METHODS = ('B', 'D', 'L', 'Y', 'J')
ARMS = METHODS[2:]
EPOCHS = (2, 5)
FEEDBACK_SEEDS = (11401, 11402)
VALIDATION_SEEDS = (11201, 11202)
TEST_SEEDS = (11101, 11102)
SCIENTIFIC_STATUSES = frozenset(('success','unroutable','packing_or_placement_failure','decode_failure'))
STAGES = ('plan', 'check', 'prepare', 'feedback', 'train', 'validate', 'compare', 'report')
CODE = tuple(dict.fromkeys(STRUCTURED_CODE + ('residual_campaign.py', 'residual_actions.py',
    'residual_model.py', 'train_residual.py', 'residual_observations.py', 'run_residual_campaign.sh')))


def queue_plan():
    return dict(schema='residual-campaign-011-1', methods=list(METHODS),
        factors=dict(B='native Default', D='frozen 010 D GNN plus capacity decoder',
                     L='learned choice of bounded layer edits', Y='learned choice of bounded XY edits',
                     J='learned choice of joint layer and XY edits'),
        policy='small candidate-ranking head over frozen GNN and bounded feasible graph-conditioned edits',
        primary_comparisons=[['L','D'],['Y','D'],['J','D']],
        contrasts=[['L','D'],['Y','D'],['J','D'],['J','L'],['J','Y'],['D','B'],['J','B']],
        train_families=['spmv','mcml'], active_validation_families=['blob_merge','diffeq'],
        reference_widths=dict(spmv=128,mcml=224), training_seed=11301,
        feedback_seeds=list(FEEDBACK_SEEDS), validation_seeds=list(VALIDATION_SEEDS), comparison_seeds=list(TEST_SEEDS),
        candidates_per_arm_upper=3, unique_candidates_per_condition_upper=7,
        feedback_runs_upper=112, validation_runs=112, comparison_runs=80, research_runs_upper=304,
        training_epochs=5, checkpoint_epochs=list(EPOCHS), source010_D_read_only=True,
        backbone_trainable=False, teacher_targets_used=False, projected_rewards_reused=False,
        tail_weight=0.,backend_layer_prior=0,new_backend_builds=0,new_rrg_extractions=0,
        gate='per arm: no lower validation success count than D and strictly negative common-population WL/CPD q; ties use D',
        runtime_role='cost only',Wmin_measured=False,independent_family_test=False,
        metric_rule='arithmetic same-successful-seed means, then equal architecture/family geometric ratios',
        WL_units='native nonglobal CHANX/CHANY lengths only; final CHANZ separately')


def owned(path):
    path=Path(path).resolve(); runs=HERE/'runs'
    if runs not in path.parents or not path.relative_to(runs).parts[0].startswith('development_011'):
        raise ValueError('011 writes require a separate development_011 cohort inside v7 runs')
    validate_output_roots((path,))
    return path


def previous009(source010):
    recorded=Path(read(Path(source010)/'structured_protocol.json')['source009'])
    result=HERE/'runs'/recorded.name
    if recorded.exists() and recorded.resolve()!=result.resolve():
        raise ValueError('010 source009 must belong to this canonical repository')
    return result


def completed_source(source010):
    report=read(Path(source010)/'comparison/report.json');rows=report.get('rows',[])
    keys={(r['condition_id'],r['seed'],r['method']) for r in rows}
    conditions={r['condition_id'] for r in rows}
    expected={(c,s,m) for c,s,m in itertools.product(conditions,(10101,10102),('B','X','D','N'))}
    if len(conditions)!=8 or len(rows)!=64 or len(keys)!=64 or keys!=expected or report.get('completed')!=64:
        raise ValueError('011 requires the completed frozen010 comparison matrix')
    return report


def check(args):
    required=('structured_protocol.json','prepared_report.json','comparison/report.json')
    missing=[str(args.source/n) for n in required if not (args.source/n).is_file()]
    if missing:return dict(ready=False,missing=missing,plan=queue_plan())
    completed_source(args.source)
    old009=previous009(args.source); old008=source008(old009)
    from types import SimpleNamespace
    result=check008(SimpleNamespace(source=old008))
    prepared=read(args.source/'prepared_report.json')
    checkpoint=Path(prepared['checkpoint'])
    if not checkpoint.is_file():result['missing'].append(str(checkpoint))
    else:verify_checkpoint(checkpoint,prepared['checkpoint_sha256'])
    result.update(ready=not result['missing'],plan=queue_plan(),source009=str(old009))
    return result


def choose_policies(rows, *, condition_ids=None):
    """Gate per arm on validation only, sharing successful seeds across epochs.

    Failed runs remain in success counts, never become zero-valued QoR. Both
    epochs use exactly the same D/successful candidate population for ranking.
    """
    if any(not scientific_result(r) for r in rows):raise ValueError('011 selection cannot use infrastructure/interface failures as bad candidates')
    lookup={(r['condition_id'],r['seed'],r['method'],r['epoch']):r for r in rows}
    if len(lookup)!=len(rows):raise ValueError('duplicate 011 validation task')
    controls={(r['condition_id'],r['seed']) for r in rows if r['method']=='D' and r['epoch']==0}
    ids=set(condition_ids) if condition_ids is not None else {c for c,s in controls}
    expected={(c,s,'D',0) for c,s in itertools.product(ids,VALIDATION_SEEDS)}
    expected|={(c,s,m,e) for c,s,m,e in itertools.product(ids,VALIDATION_SEEDS,ARMS,EPOCHS)}
    if len(ids)!=8 or set(lookup)!=expected:raise ValueError('011 validation requires eight conditions, two seeds, all three arms and both epochs')
    baseline_success=sum(lookup[c,s,'D',0]['status']=='success' for c,s in controls)
    result={}
    for arm in ARMS:
        common={(c,s) for c,s in controls if all(lookup[c,s,m,e]['status']=='success'
            for m,e in (('D',0),(arm,2),(arm,5)))}
        scores=[]
        for epoch in EPOCHS:
            subset=[lookup[c,s,'D',0] for c,s in common]+[lookup[c,s,arm,epoch] for c,s in common]
            comparison=contrast(subset,arm,'D') if common else None
            q=comparison['joint_q'] if comparison else None
            successes=sum(lookup[c,s,arm,epoch]['status']=='success' for c,s in controls)
            eligible=successes>=baseline_success and q is not None and math.isfinite(q) and q<0
            scores.append(dict(epoch=epoch,successes=successes,common_success_pairs=len(common),joint_q=q,
                eligible=eligible,comparison=comparison))
        eligible=[s for s in scores if s['eligible']]
        best=min(eligible,key=lambda s:(-s['successes'],s['joint_q'],s['epoch'])) if eligible else None
        result[arm]=dict(enabled=best is not None,epoch=best['epoch'] if best else 0,
            scores=scores,baseline_successes=baseline_success,
            reason='validation_qor_supported' if best else 'validation_did_not_support_replacement; deploy unchanged D')
    return dict(arms=result,validation_only=True,test_seeds_used=False,runtime_used_for_selection=False,
        gate=queue_plan()['gate'],performance_selection_completed=True)


def scientific_result(row):
    return row.get('status') in SCIENTIFIC_STATUSES and (
        row['status']!='packing_or_placement_failure' or row.get('confirmed_candidate_infeasibility') is True)


def classify_native_failure(status,returncode,text):
    """The older generic packing status also covers backend assertions/crashes."""
    confirmed=(status=='packing_or_placement_failure' and returncode==1 and
        'Failed to find device which satisfies resource requirements' in text and
        not any(token in text.lower() for token in ('assertion','segmentation fault','sigsegv','sigabrt','internal error')))
    if status=='packing_or_placement_failure' and not confirmed:status='infrastructure_failure'
    return status,confirmed


def candidate_diagnostics(actions):
    """Measured training-only opportunity; an oracle is never a deployed arm."""
    controls={a['condition_id']:a for a in actions if a['arm']=='D' and a['candidate_id']=='identity'}
    rows=[]
    for action in actions:
        if action['arm']=='D':continue
        control=controls.get(action['condition_id'])
        if control is None:raise ValueError('011 measured candidate lacks a D control')
        a={t['seed']:t for t in action['trials']};b={t['seed']:t for t in control['trials']}
        if set(a)!=set(b):raise ValueError('011 candidate/control measured seeds differ')
        successes=[sum(t['status']=='success' for t in side.values()) for side in (a,b)]
        common=[s for s in a if a[s]['status']==b[s]['status']=='success']
        q=None
        if common:
            q=.5*sum(math.log(statistics.mean(a[s][m] for s in common)/statistics.mean(b[s][m] for s in common))
                for m in ('wirelength','cpd_ns'))
        rows.append(dict(condition_id=action['condition_id'],arm=action['arm'],candidate_id=action['candidate_id'],
            candidate_successes=successes[0],D_successes=successes[1],common_success_seeds=common,joint_q=q,
            higher_success_count=successes[0]>successes[1],
            lower_common_qor_without_success_loss=successes[0]>=successes[1] and q is not None and q<0))
    return dict(rows=rows,train_only=True,oracle_is_deployable=False,
        better_candidates_by_arm={arm:sum(r['arm']==arm and (r['higher_success_count'] or r['lower_common_qor_without_success_loss']) for r in rows) for arm in ARMS},
        all_negative_results_retained=True)


def worker(request,out):
    verify_checkpoint(request['checkpoint'],request.get('checkpoint_sha256'))
    verify_checkpoint(request['base_checkpoint'],request.get('base_checkpoint_sha256'))
    import torch
    from data import build_logic
    from fast_graph import make_graph
    from layer_features import augment_graph
    from generation import lower_initialization
    from aligned_teacher import constraints_digest
    from aligned_model import action_digest
    from train import state_sha256
    from model import CapacityError
    from train_structured import load_model as load_base
    from train_residual import from_source, load_model
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    validate_output_roots((out,));out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter();r=request
    logical=build_logic(r['prefix'],r['design'],r['blif'],r['architecture'],family=r['family'],split=r['split'])
    device=read(r['device'])
    graph=augment_graph(make_graph(logical,device),logical,device,read(r['vertical_summary']),r['blif'],r['prefix'])
    torch.save(graph,out/'graph.pt');write(out/'logic.json',logical)
    result=dict(graph=str(out/'graph.pt'),logic=str(out/'logic.json'),graph_sha256=sha(out/'graph.pt'),
        logic_sha256=sha(out/'logic.json'),graph_seconds=time.perf_counter()-start)
    collecting=r.get('collect_bank',False)
    if collecting:
        selector=from_source(r['base_checkpoint'],expected_sha256=r['base_checkpoint_sha256'])
        backbone=selector.backbone
    elif r['method']=='D' or not r.get('enabled',True):
        backbone=load_base(r['base_checkpoint'],device='cpu').eval();selector=None
    else:
        selector=load_model(r['checkpoint'],device='cpu');backbone=selector.backbone
        if selector.source_checkpoint_sha256!=r['base_checkpoint_sha256']:
            raise ValueError('011 selector uses a different frozen D backbone')
    try:
        with torch.no_grad():
            if collecting:
                predictions=[];identity_seen=False;base=None
                for arm in ARMS:
                    base,bank=selector.candidate_bank(graph,arm,base_action=base)
                    for prediction in bank:
                        # Identity is the shared measured D control. Equal
                        # nonidentity actions in different arms retain their
                        # own IDs, constraints and measured training support.
                        identity=prediction['candidate_id']=='identity'
                        if identity and identity_seen:continue
                        identity_seen|=identity
                        predictions.append((arm,prediction))
            else:
                if selector is None:
                    base=backbone.decode_action(graph,method='D');prediction=base
                else:
                    base,bank=selector.candidate_bank(graph,r['method'])
                    prediction=selector.select_from_bank(graph,r['method'],base,bank,enabled=r.get('enabled',True))
                predictions=[(r['method'],prediction)]
    except CapacityError as exc:
        result.update(status='decode_failure',reason=str(exc),packing_infeasibility_claimed=False)
    else:
        outputs=[]
        for index,(arm,prediction) in enumerate(predictions):
            directory=out/('candidates/'+str(index) if collecting else 'initialization')
            lowered=lower_initialization(logical,device,prediction,directory,method=arm)
            changed=(torch.as_tensor(prediction['layer'])!=torch.as_tensor(base['layer'])) | (
                torch.as_tensor(prediction['xy'])!=torch.as_tensor(base['xy'])).any(1)
            grid=device.get('grid',{})
            width=device.get('grid_width',grid.get('width'));height=device.get('grid_height',grid.get('height'))
            edits=[]
            for node in changed.nonzero().flatten().tolist():
                for atom in logical['clusters'][node]['atoms']:
                    edits.append(dict(name=atom,base_layer=int(base['layer'][node]),target_layer=int(prediction['layer'][node]),
                        base_xy=torch.as_tensor(base['xy'][node]).tolist(),target_xy=torch.as_tensor(prediction['xy'][node]).tolist()))
            edit_path=directory/'residual_edits.json';write(edit_path,dict(grid=[width,height],atoms=edits))
            lowered.update(residual_edits=str(edit_path),residual_edits_sha256=sha(edit_path))
            action=read(lowered['target'])
            # Keep bounded-edit provenance and metadata; tensors remain in payloads only.
            def plain(value):
                if hasattr(value,'detach'):return value.detach().cpu().tolist()
                if isinstance(value,dict):return {k:plain(v) for k,v in value.items()}
                if isinstance(value,(list,tuple)):return [plain(v) for v in value]
                return value
            action.update(plain({k:v for k,v in prediction.items() if k not in action}))
            action.update(constraints_sha256=constraints_digest(graph),collection_checkpoint_sha256=r['checkpoint_sha256'],
                base_checkpoint_sha256=r['base_checkpoint_sha256'],collection_arm=arm,
                deployed_action_sha256=action_digest(prediction))
            write(lowered['target'],action)
            lowered.update(target_sha256=sha(lowered['target']),deployed_action_sha256=action_digest(prediction),
                candidate_id=prediction.get('candidate_id','identity'),collection_arm=arm,
                shared_xy_sha256=state_sha256(backbone.shared_xy),base_checkpoint_sha256=r['base_checkpoint_sha256'],
                collection_checkpoint_sha256=r['checkpoint_sha256'],risk_diagnostics_computed=False)
            write(directory/'generation.json',lowered);outputs.append(lowered)
        if collecting:result['candidates']=outputs
        else:result.update(outputs[0])
    verify_checkpoint(r['checkpoint'],r['checkpoint_sha256']);verify_checkpoint(r['base_checkpoint'],r['base_checkpoint_sha256'])
    result['worker_internal_seconds']=time.perf_counter()-start
    write(out/'generation_report.json',result);print('TAILCOMPILE_RESIDUAL_WORKER_COMPLETE',flush=True)
    return result


class ResidualCampaign(QorCampaign):
    def __init__(self,args):
        self.args,self.out=args,owned(args.out)
        self.source010=args.source;self.source009=previous009(args.source);self.source008=source008(self.source009)
        self.conditions=panel(self.source008);self.train=[c for c in self.conditions if c['split']=='train']
        self.reference=[c for c in self.train if c['width_role']=='reference'];self.validation=[c for c in self.conditions if c['split']=='validation']
        self.previous=legacy(self.source008);self.budget=resource_budget(args.jobs,args.per_job_mib)
        self.binary=None;self._devices={}

    def setup(self):
        from install_layer_hook import verify
        for source in (self.source010,self.source009,self.source008,self.previous):
            if source==self.out or source in self.out.parents or self.out in source.parents:raise ValueError('011 output and all source cohorts must be disjoint')
        completed_source(self.source010)
        p=read(self.source010/'structured_protocol.json');prepared=read(self.source010/'prepared_report.json')
        self.base_checkpoint=Path(prepared['checkpoint']);self.base_sha=prepared['checkpoint_sha256']
        verify_checkpoint(self.base_checkpoint,self.base_sha)
        self.args.backend=Path(p['backend']).resolve();verify(self.args.backend)
        self.binary=Path(read(self.args.backend/'backend_manifest.json')['binary'])
        if sha(self.binary)!=p['binary_sha256']:raise ValueError('frozen 010 backend changed')
        for name,wanted in p['implementation'].items():
            if sha(HERE/name)!=wanted:raise ValueError('frozen 010 implementation changed: '+name)
        for path,wanted in p['source_inputs'].items():
            if sha(path)!=wanted:raise ValueError('frozen 010 source input changed: '+path)
        aligned=read(self.source008/'aligned_protocol.json')
        legacy_inputs={}
        for name in ('layer_protocol.json','borrowed/conditions.json','borrowed/qualified_source_manifest.json'):
            path=self.previous/name
            if sha(path)!=aligned['source_inputs'].get(str(path.resolve())):
                raise ValueError('frozen qualified legacy metadata changed: '+name)
            legacy_inputs[str(path)]=sha(path)
        qualified=self.previous/'qualification/qualification_report.json';qualification=read(qualified)
        if not qualification.get('passed') or qualification['binary_sha256']!=p['binary_sha256'] or \
            sha(qualified)!=read(self.previous/'layer_protocol.json')['qualification_sha256']:
            raise ValueError('frozen backend qualification mismatch')
        legacy_inputs[str(qualified)]=sha(qualified)
        self.catalog=read(self.previous/'borrowed/qualified_source_manifest.json')
        old_reference=Path(self.catalog['source'])/'binary_identity.json'
        if sha(old_reference)!=self.catalog['files'][str(old_reference.resolve())]:raise ValueError('original binary identity changed')
        self.old_binary_hash=read(old_reference)['sha256']
        self.teacher_rows={r['condition_id']:r for r in read(self.source008/'teacher_records.json')['records']}
        inputs={str(self.source010/n):sha(self.source010/n) for n in ('structured_protocol.json','prepared_report.json','comparison/report.json')}
        inputs.update(legacy_inputs)
        for source,name in ((self.source008,'aligned_protocol.json'),(self.source008,'teacher_records.json')):
            path=source/name;inputs[str(path)]=sha(path)
        inputs[str(self.base_checkpoint)]=self.base_sha
        for c in self.conditions:
            row=self.teacher_rows[c['id']]
            if row['split']!=c['split'] or row['family']!=c['family']:raise ValueError('borrowed graph split changed')
            if sha(row['graph'])!=p['source_inputs'].get(row['graph']):raise ValueError('frozen 010 graph changed')
            inputs[row['graph']]=sha(row['graph'])
            dev=self.device(c)
            for path in [c['record']['blif'],dev['architecture'],dev['device'],dev['vertical_summary'],
                *(c['record']['structure_prefix']+s for s in ('.blocks.tsv','.nets.tsv','.meta.tsv'))]:
                inputs[str(Path(path).resolve())]=sha(path)
        freeze(self.out/'residual_protocol.json',dict(plan=queue_plan(),source010=str(self.source010),source009=str(self.source009),
            source008=str(self.source008),base_checkpoint=str(self.base_checkpoint),base_checkpoint_sha256=self.base_sha,
            source_inputs=inputs,backend=str(self.args.backend),binary_sha256=p['binary_sha256'],
            implementation={n:sha(HERE/n) for n in CODE},timeout=self.args.timeout))
        freeze(self.out/'queue_plan.json',queue_plan());self.binary_reference=self.out/'binary_identity.json'
        freeze(self.binary_reference,dict(path=str(self.binary),sha256=p['binary_sha256']))
        st=self.binary.stat();self.binary_stat=(st.st_size,st.st_mtime_ns,st.st_ino)
        freeze(self.out/'execution'/self.args.stage/'protocol.json',dict(device=self.args.device,jobs=self.args.jobs,
            per_job_mib=self.args.per_job_mib,timeout=self.args.timeout,physical_cores=self.budget['physical_cores'],effective_lanes=self.budget['effective']))

    def prepare(self):
        freeze(self.out/'training_records.json',dict(records=[dict(condition_id=c['id'],split=c['split'],family=c['family'],
            width_role=c['width_role'],graph=self.teacher_rows[c['id']]['graph']) for c in self.conditions]))
        freeze(self.out/'prepared_report.json',dict(complete=True,checkpoint=str(self.base_checkpoint),checkpoint_sha256=self.base_sha,
            train_conditions=len(self.train),reference_conditions=len(self.reference),active_validation_conditions=len(self.validation),
            frozen_backbone=True,teacher_targets_used=False,source010_N_used=False))

    def request(self,c,checkpoint=None,method='D',prefix=None,noise=0.,collect_bank=False):
        prepared=read(self.out/'prepared_report.json');base=prepared['checkpoint'];base_sha=prepared['checkpoint_sha256']
        verify_checkpoint(base,base_sha);checkpoint=str(checkpoint or base)
        enabled=method in ARMS and checkpoint!=base
        if enabled:expected=read(self.out/'trained_report.json').get('checkpoint_sha256',{}).get(checkpoint)
        else:expected=base_sha
        verify_checkpoint(checkpoint,expected)
        return dict(self.device(c),prefix=prefix or c['record']['structure_prefix'],design=c['design'],family=c['family'],split=c['split'],
            blif=c['record']['blif'],checkpoint=checkpoint,checkpoint_sha256=expected,base_checkpoint=base,
            base_checkpoint_sha256=base_sha,method=method,enabled=enabled,collect_bank=collect_bank,generation_seed=11301)

    def generate(self,c,target,checkpoint=None,method='D',prefix=None,noise=0.,core=None,collect_bank=False):
        self.guard();request=self.request(c,checkpoint,method,prefix,noise,collect_bank)
        freeze(target/'request.json',request)
        raw_inputs=[request[k] for k in ('blif','architecture','device','vertical_summary')]
        raw_inputs.extend(request['prefix']+s for s in ('.blocks.tsv','.nets.tsv','.meta.tsv'))
        result=run_trial(target/'worker',[sys.executable,HERE/'residual_campaign.py','worker','--request',target/'request.json',
            '--out',target/'payload'],inputs=[target/'request.json',request['checkpoint'],request['base_checkpoint']]+raw_inputs+[HERE/n for n in CODE],
            timeout=self.args.timeout,affinity=core,marker='TAILCOMPILE_RESIDUAL_WORKER_COMPLETE',prune=False)
        if result['status']!='complete':raise RuntimeError('011 worker failed (not a routability label): '+result['log'])
        value=read(target/'payload/generation_report.json');value['generation_process_seconds']=result['metrics']['wall_seconds']
        return value

    def pnr(self,c,seed,target,method,generated=None,core=None):
        self.guard(); dev=self.device(c)
        frozen_device=read(Path(dev['vertical_summary']).parent/'device_preparation_report.json')
        stat=Path(dev['rr_graph']).stat()
        if frozen_device['rr_stat'] != [stat.st_size,stat.st_mtime_ns,stat.st_ino]:
            raise ValueError('frozen RRG changed after preparation')
        env=dict(TC_GP_LAYER_PRIOR='0',TC_GP_EXPORT=target/'post_gp',TC_GP_Z_USAGE=target/'vertical_usage')
        inputs=[self.binary_reference,dev['architecture'],c['record']['blif'],dev['device_manifest']]
        if generated is not None:
            env['TC_GP_INIT']=generated['ap_init']; inputs.append(generated['ap_init'])
        result=run_trial(target/'vpr',vpr_command(self.binary,dev['architecture'],c['record']['blif'],c['width'],seed,
            rr_graph=dev['rr_graph'])+['--device','pilot','--write_flat_place','post_route.fplace',
            '--write_legalized_flat_place','post_legalizer.fplace','--flat_place_verbosity','0'],
            inputs=inputs,env=env,affinity=core,timeout=self.args.timeout,prune=False)
        if (target/'metrics.json').exists():
            cached=read(target/'metrics.json')
            if (cached['condition_id'],cached['seed'],cached['method']) != (c['id'],seed,method):
                raise ValueError('cached native result identity changed')
            return cached
        status=result['status']; text=Path(result['log']).read_text(errors='replace')
        status,confirmed_infeasibility=classify_native_failure(status,result.get('returncode'),text)
        if generated and status in ('success','unroutable') and ('TAILCOMPILE_V7_GP_INIT_CONSUMED' not in text or
            'TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' not in text): status='interface_failure'
        row=dict(condition_id=c['id'],design=c['design'],family=c['family'],split=c['split'],architecture=c['architecture'],
            width=c['width'],width_role=c['width_role'],method=method,seed=seed,status=status,
            confirmed_candidate_infeasibility=confirmed_infeasibility,
            vpr_seconds=result['metrics']['wall_seconds'],cpd_ns=result['metrics']['cpd_ns'],wirelength=result['metrics']['wirelength'],
            phases=result['metrics']['phases'],unique_chanz_nodes=result['metrics']['unique_chanz_nodes'],
            vpr_manifest=str(target/'vpr/manifest.json'),ap_init_sha256=generated.get('ap_init_sha256') if generated else None)
        from vertical_risk import route_pool_summary
        from layer_observations import collect_layers
        from qor_route_metrics import summarize_route
        row['vertical_routing']={s:route_pool_summary(read(dev['vertical_summary']),path) for s in ('first','last')
                                 if (path:=target/f'vertical_usage.{s}.tsv').is_file()}
        row['last_vertical_snapshot_semantics']='last completed iteration, may precede best-route restoration'
        row['layer_observations']=collect_layers(generated['ap_init'] if generated else None,target/'post_gp',
            final_fplace=Path(result['workdir'])/'post_route.fplace',legalized_fplace=Path(result['workdir'])/'post_legalizer.fplace')
        routes=list(Path(result['workdir']).glob('*.route'))
        row['final_route_usage']=summarize_route(routes[0]) if status=='success' and len(routes)==1 else None
        row['wirelength_units']='nonglobal CHANX/CHANY node lengths, CHANZ excluded; not physical micrometres'
        from residual_observations import collect_residual_observations
        row['residual_observations']=collect_residual_observations(generated,target,Path(result['workdir']))
        write(target/'metrics.json',row)
        if status in ('success','unroutable','complete') or confirmed_infeasibility:
            prune_derived(result['workdir'],protected=inputs)
            removed=[]
            for trace in list(target.glob('post_gp.*'))+[Path(result['workdir'])/n for n in ('post_route.fplace','post_legalizer.fplace')]:
                if trace.is_file() and not trace.is_symlink() and target.resolve() in trace.resolve().parents:
                    removed.append(dict(path=str(trace),sha256=sha(trace),bytes=trace.stat().st_size)); trace.unlink()
            write(target/'trace_pruned_report.json',dict(removed=removed,observations_and_route_usage_durable=True))
        return row

    def feedback(self):
        if (self.out/'feedback_report.json').exists():return
        prepared=read(self.out/'prepared_report.json');verify_checkpoint(prepared['checkpoint'],prepared['checkpoint_sha256'])
        cores=self.budget['physical_cores'][:min(self.budget['effective'],len(self.reference))]
        freeze(self.out/'feedback/timing_protocol.json',dict(cores=cores,seeds=list(FEEDBACK_SEEDS),each_lane_sequential=True))
        def lane(item):
            i,core=item;actions=[];failures=[]
            for c in self.reference[i::len(cores)]:
                folder=self.out/'feedback'/c['id']
                generated=self.generate(c,folder/'generation',prepared['checkpoint'],'D',core=core,collect_bank=True)
                if 'candidates' not in generated:
                    failures.append(dict(condition_id=c['id'],generation=generated));self.prune_feedback_graph(folder);continue
                for index,gen in enumerate(generated['candidates']):
                    arm='D' if gen['candidate_id']=='identity' else gen['collection_arm']
                    trials=[self.pnr(c,s,folder/f'candidate_{index}'/f's{s}',arm,gen,core) for s in FEEDBACK_SEEDS]
                    if any(not scientific_result(t) for t in trials):
                        raise RuntimeError('011 feedback infrastructure/interface failure; inspect native manifest')
                    actions.append(dict(condition_id=c['id'],family=c['family'],split='train',arm=arm,
                        candidate_id=gen['candidate_id'],action=gen['target'],generation=gen,trials=trials))
                self.prune_feedback_graph(folder)
            return actions,failures
        with ThreadPoolExecutor(max_workers=len(cores)) as pool:groups=list(pool.map(lane,enumerate(cores)))
        actions=[a for g in groups for a in g[0]];failures=[a for g in groups for a in g[1]]
        freeze(self.out/'feedback.json',actions)
        freeze(self.out/'feedback_report.json',dict(actions=len(actions),native_runs=sum(len(a['trials']) for a in actions),
            budget_upper=112,decode_failures=failures,old_labels_or_projected_rewards_reused=False,
            candidate_diversity={c['id']:sum(a['condition_id']==c['id'] for a in actions) for c in self.reference},
            measured_opportunity=candidate_diagnostics(actions)))

    def train_models(self):
        if (self.out/'trained_report.json').exists():return
        from train_residual import fit
        prepared=read(self.out/'prepared_report.json');verify_checkpoint(prepared['checkpoint'],prepared['checkpoint_sha256'])
        records=[r for r in read(self.out/'training_records.json')['records'] if r['split']=='train' and r['width_role']=='reference']
        result=fit(records,prepared['checkpoint'],self.fresh_attempt(self.out/'models/trained'),feedback=read(self.out/'feedback.json'),
            epochs=5,snapshots=EPOCHS,seed=11301,device=self.args.device,expected_sha256=prepared['checkpoint_sha256'])
        result['checkpoint_sha256']={str(p):sha(p) for p in result['checkpoints']}
        freeze(self.out/'trained_report.json',result)
        if result.get('effective_pairs',result.get('feedback_pairs',0))<=0:raise RuntimeError('011 has no effective real QoR preference pairs; stop before validation')

    def validate(self):
        trained=read(self.out/'trained_report.json')
        if trained.get('effective_pairs',trained.get('feedback_pairs',0))<=0:raise ValueError('011 validation requires effective real QoR preference pairs')
        checkpoints=trained['checkpoints']
        if len(checkpoints)!=2:raise ValueError('011 requires epoch2 and epoch5 snapshots')
        for p in checkpoints:verify_checkpoint(p,trained.get('checkpoint_sha256',{}).get(str(p)))
        prepared=read(self.out/'prepared_report.json');base=verify_checkpoint(prepared['checkpoint'],prepared['checkpoint_sha256'])
        core=self.budget['physical_cores'][0];rows=[]
        freeze(self.out/'validation/timing_protocol.json',dict(core=core,sequential=True,seeds=list(VALIDATION_SEEDS),
            checkpoints={str(p):sha(p) for p in checkpoints}))
        for c,s in itertools.product(self.validation,VALIDATION_SEEDS):
            rows.append(dict(self.cold(c,s,'D',base,self.out/'validation'/c['id']/f's{s}'/'D',core),epoch=0))
            for epoch,checkpoint,arm in itertools.product(EPOCHS,checkpoints,ARMS):
                if checkpoints[EPOCHS.index(epoch)]!=checkpoint:continue
                rows.append(dict(self.cold(c,s,arm,checkpoint,self.out/'validation'/c['id']/f's{s}'/f'{arm}_e{epoch}',core),epoch=epoch))
        if any(not scientific_result(r) for r in rows):raise RuntimeError('011 validation infrastructure/interface failure; no gate selection')
        selection=choose_policies(rows,condition_ids=[c['id'] for c in self.validation])
        for arm,value in selection['arms'].items():
            checkpoint=checkpoints[EPOCHS.index(value['epoch'])] if value['enabled'] else base
            value.update(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint))
        freeze(self.out/'selection.json',selection);write(self.out/'validation/report.json',dict(rows=rows,selection=selection))

    def compare(self):
        selection=read(self.out/'selection.json');trained=read(self.out/'trained_report.json')
        prepared=read(self.out/'prepared_report.json');base=verify_checkpoint(prepared['checkpoint'],prepared['checkpoint_sha256'])
        checkpoints={'D':base}
        for arm,value in selection['arms'].items():
            checkpoint=verify_checkpoint(value['checkpoint'],value['checkpoint_sha256'])
            if value['enabled'] and trained['checkpoint_sha256'].get(checkpoint)!=value['checkpoint_sha256']:
                raise ValueError('selected011 model was not a frozen trained snapshot')
            if not value['enabled'] and checkpoint!=base:raise ValueError('disabled011 arm must deploy exact D')
            checkpoints[arm]=checkpoint
        cases=list(itertools.product(self.reference,TEST_SEEDS));available=self.budget['physical_cores'][:self.budget['effective']]
        lanes=max(n for n in range(1,len(available)+1) if len(cases)%n==0);cores=available[:lanes]
        host=dict(uname=list(os.uname()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cores=cores)
        freeze(self.out/'comparison/timing_protocol.json',dict(host=host,cases=[[c['id'],s] for c,s in cases],methods=list(METHODS),
            seed_rotation='case index modulo five',checkpoints={m:sha(p) for m,p in checkpoints.items()},
            validation_selection_sha256=sha(self.out/'selection.json'),fixed_lane_barriers=True))
        rows=[]
        for start in range(0,len(cases),lanes):
            wave=self.out/'comparison'/f'wave_{start//lanes:04d}'
            signature=dict(host=host,cases=[[c['id'],s] for c,s in cases[start:start+lanes]],
                timing_protocol_sha256=sha(self.out/'comparison/timing_protocol.json'))
            if (wave/'completed.json').exists():
                cached=read(wave/'completed.json')
                if cached['signature']!=signature:raise ValueError('011 timing host/input changed; use new cohort')
                rows.extend(cached['rows']);continue
            if len(list(wave.glob('attempt_*')))>=2:raise RuntimeError('011 whole-wave retry budget exhausted')
            attempt=self.fresh_attempt(wave);barrier=threading.Barrier(lanes)
            def lane(item):
                i,(c,s)=item;core=cores[i];os.sched_setaffinity(0,{core})
                offset=(start+i)%len(METHODS);order=METHODS[offset:]+METHODS[:offset];result=[]
                try:
                    for m in order:
                        barrier.wait(timeout=self.args.timeout*4)
                        row=self.cold(c,s,m,checkpoints.get(m),attempt/f'lane_{i:03d}'/c['id']/f's{s}'/m,core)
                        result.append(dict(row,order=list(order),wave_load=lanes,host=host,
                            validation_enabled=selection['arms'][m]['enabled'] if m in ARMS else None))
                        barrier.wait(timeout=self.args.timeout*4)
                except BaseException:barrier.abort();raise
                return result
            with ThreadPoolExecutor(max_workers=lanes) as pool:groups=list(pool.map(lane,enumerate(cases[start:start+lanes])))
            wave_rows=[r for group in groups for r in group]
            freeze(wave/'completed.json',dict(signature=signature,rows=wave_rows,actual_attempt=str(attempt),
                previous_incomplete_attempts=len(list(wave.glob('attempt_*')))-1))
            rows.extend(wave_rows);write(self.out/'comparison/report.json',dict(planned=80,completed=len(rows),rows=rows))

    def report(self):
        rows=read(self.out/'comparison/report.json')['rows'];expected={(c['id'],s,m) for c,s,m in itertools.product(self.reference,TEST_SEEDS,METHODS)}
        keys=[(r['condition_id'],r['seed'],r['method']) for r in rows]
        if len(keys)!=len(set(keys)):raise ValueError('duplicate011 native task')
        result=dict(schema='residual-evaluation-011-1',complete=set(keys)==expected,completed=len(rows),
            missing=sorted(expected-set(keys)),unexpected=sorted(set(keys)-expected),status_counts=dict(Counter(r['status'] for r in rows)),
            status_by_method={m:dict(Counter(r['status'] for r in rows if r['method']==m)) for m in METHODS},
            comparisons=[contrast(rows,m,b) for m,b in queue_plan()['contrasts']],
            validation_selection=read(self.out/'selection.json'),tail_weight=0.,runtime_is_cost=True,
            Wmin_measured=False,independent_family_test=False,
            candidate_oracle_role='training diagnostics only, not deployable performance')
        write(self.out/'residual_report.json',result)
        print('011 comparison complete' if result['complete'] else '011 comparison incomplete',result['status_counts'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES+('worker',))
    p.add_argument('--source',type=Path,default=HERE/'runs/development_010_structured')
    p.add_argument('--out',type=Path,default=HERE/'runs/development_011_residual')
    p.add_argument('--request',type=Path);p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--per-job-mib',type=int,default=8192);p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--device',default='cpu');args=p.parse_args()
    if min(args.jobs,args.per_job_mib,args.timeout)<1:p.error('positive resource limits required')
    install_signal_handlers();args.source=args.source.resolve();args.out=owned(args.out)
    if args.stage=='worker':worker(read(args.request),args.out);return
    if args.stage=='plan':print(json.dumps(queue_plan(),indent=2));return
    if args.stage=='check':
        result=check(args);print(json.dumps(result,indent=2));raise SystemExit(0 if result['ready'] else 2)
    with campaign_lock(args.out,'residual-'+args.stage):
        runner=ResidualCampaign(args);runner.setup();getattr(runner,'train_models' if args.stage=='train' else args.stage)()


if __name__=='__main__':main()
