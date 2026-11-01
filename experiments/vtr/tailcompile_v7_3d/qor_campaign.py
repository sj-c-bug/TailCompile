"""009 QoR-first train/feedback/validation/comparison; immutable 008 inputs."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import itertools
import os
from pathlib import Path
import sys
import threading
import time

from aligned_campaign import AlignedCampaign, CODE as ALIGNED_CODE
from campaign import campaign_lock, identity, resource_budget
from io_utils import read, write, freeze, sha, run_trial, vpr_command, install_signal_handlers, prune_derived
from repository_guard import validate_output_roots
from qor_preferences import evaluate, choose_epoch

HERE = Path(__file__).resolve().parent
METHODS = ('B', 'R', 'Q', 'C', 'X', 'T')
ARMS = METHODS[1:]
STAGES = ('plan','check','prepare','bootstrap','feedback','train','validate','compare','report')
CODE = tuple(dict.fromkeys(ALIGNED_CODE + ('qor_campaign.py','qor_preferences.py','qor_model.py','train_qor.py',
                                        'run_qor_campaign.sh','qor_route_metrics.py')))


def queue_plan():
    return dict(schema='qor-campaign-009-1', objective_version='qor-first-20261102',
        methods=list(METHODS), primary_comparison=['T','B'], auxiliary_comparison=['X','B'],
        contrasts=[['Q','R'],['C','Q'],['X','C'],['T','X']],
        train_families=['spmv','mcml'], active_validation_families=['blob_merge','diffeq'],
        qor_weights=dict(wirelength=.5,cpd_ns=.5), runtime_role='cost only',
        factors={m:dict(objective='runtime' if m=='R' else 'qor',cut_weight=.1 if m in ('R','Q') else 0.,
                       xy_trainable=m in ('X','T'),tail_weight=.1 if m=='T' else 0.) for m in ARMS},
        backend_layer_prior=0, reference_widths=dict(spmv=128,mcml=224),
        feedback_seeds=[9101,9102], validation_seeds=[9201,9202], comparison_seeds=list(range(9001,9006)),
        training_seed=9301, bootstrap_epochs=5, feedback_epochs=10, checkpoint_epochs=[5,10],
        structure_export_seed=2701, feedback_runs_upper=96, validation_runs_upper=176,comparison_runs=240,
        research_runs_upper=512,local_engineering_runs_upper=4,Wmin_measured=False,
        generalization_claim=False, original008_read_only=True, new_backend_builds=0,new_rrg_extractions=0,
        aggregation='same successful seeds arithmetic mean per condition, ratio, architecture and family equal geometric mean',
        support_boundary='R/Q/C frozen-XY actions only; X/T all validated collection policies',
        WL_units='pinned VPR nonglobal CHANX/CHANY node lengths only; CHANZ excluded, reported separately; not physical micrometres')


def owned(path):
    path=Path(path).resolve(); runs=HERE/'runs'
    if runs not in path.parents or not path.relative_to(runs).parts[0].startswith('development_009'):
        raise ValueError('009 output requires a separate development_009 cohort inside v7 runs')
    validate_output_roots((path,))
    return path


def legacy(source):
    recorded=Path(read(Path(source)/'aligned_protocol.json')['source'])
    # Cohorts retain the canonical repository-relative identity across mounts.
    result=HERE/'runs'/recorded.name
    if recorded.exists() and recorded.resolve() != result.resolve():
        raise ValueError('008 source007 must resolve to this repository cohort')
    return result


def panel(source):
    source=Path(source)
    if not read(source/'layer_report.json').get('complete'):
        raise ValueError('009 requires completed immutable008 comparison evidence')
    previous=legacy(source)
    records=read(previous/'borrowed/conditions.json')['records']
    selected=[r for r in records if r['family'] in ('spmv','mcml') or
              r['family'] in ('blob_merge','diffeq') and r['width_role']=='reference']
    if len(selected)!=24 or len({r['id'] for r in selected})!=24:
        raise ValueError('expected sixteen train and eight active validation inputs')
    for r in selected:
        expected='train' if r['family'] in ('spmv','mcml') else 'validation'
        if r['split'] != expected: raise ValueError('family/split changed')
        if expected=='train' and r['width_role']=='reference' and r['width'] != queue_plan()['reference_widths'][r['family']]:
            raise ValueError('public reference width changed')
    return selected


def check(args):
    missing=[]
    required=('aligned_protocol.json','layer_report.json','teacher_records.json','feedback.json',
              'teacher_alignment_report.json','prepared_models_report.json')
    for name in required:
        if not (args.source/name).is_file(): missing.append(str(args.source/name))
    if missing: return dict(ready=False,missing=missing,plan=queue_plan())
    conditions=panel(args.source); previous=legacy(args.source)
    source_protocol=read(args.source/'aligned_protocol.json')
    backend=Path(source_protocol['backend'])
    for path in (backend/'backend_manifest.json',previous/'qualification/qualification_report.json'):
        if not path.is_file(): missing.append(str(path))
    teachers={r['condition_id']:r for r in read(args.source/'teacher_records.json')['records']}
    checkpoint=Path(read(args.source/'prepared_models_report.json')['arms']['L']['checkpoint'])
    if not checkpoint.is_file(): missing.append(str(checkpoint))
    for c in conditions:
        for path in (Path(teachers[c['id']]['graph']),Path(teachers[c['id']]['teacher']),
                     Path(c['record']['blif']),*(Path(c['record']['structure_prefix']+e) for e in ('.blocks.tsv','.nets.tsv','.meta.tsv'))):
            if not path.is_file(): missing.append(str(path))
    for entry in read(args.source/'feedback.json'):
        if entry.get('action') and not Path(entry['action']).is_file(): missing.append(entry['action'])
    catalog=read(previous/'borrowed/qualified_source_manifest.json')
    old_reference=Path(catalog['source'])/'binary_identity.json'
    if not old_reference.is_file(): missing.append(str(old_reference))
    else:
        old_hash=read(old_reference)['sha256']
        for c in conditions:
            grid=c['record']['grid']; arch=Path(catalog['architectures'][f'{grid[0]}x{grid[1]}/{c["architecture"]}'])
            if not arch.is_file(): missing.append(str(arch)); continue
            key=identity(dict(architecture=sha(arch),binary=old_hash,width=c['width']))[:20]
            folder=Path(catalog['devices'][key])
            for path in (folder/'manifest.json',folder/'rr_graph.xml',folder/'device_graph.json',previous/'vertical_devices'/key/'summary.json'):
                if not path.is_file(): missing.append(str(path))
    return dict(ready=not missing,missing=sorted(set(missing)),conditions=len(conditions),plan=queue_plan())


def worker(request,out):
    import torch
    from data import build_logic
    from fast_graph import make_graph
    from layer_features import augment_graph
    from generation import lower_initialization
    from train_qor import load_model
    from qor_model import generate
    from train import state_sha256
    from aligned_teacher import constraints_digest
    from model import CapacityError
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    validate_output_roots((out,)); out.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); r=request
    logical=build_logic(r['prefix'],r['design'],r['blif'],r['architecture'],family=r['family'],split=r['split'])
    device=read(r['device'])
    graph=augment_graph(make_graph(logical,device),logical,device,read(r['vertical_summary']),r['blif'],r['prefix'])
    torch.save(graph,out/'graph.pt'); write(out/'logic.json',logical)
    result=dict(graph=str(out/'graph.pt'),logic=str(out/'logic.json'),graph_sha256=sha(out/'graph.pt'),
                logic_sha256=sha(out/'logic.json'),graph_seconds=time.perf_counter()-start)
    model=load_model(r['checkpoint'],device='cpu')
    try:
        with torch.no_grad():
            prediction=generate(model,graph,method=r['method'],seed=r.get('generation_seed',9301),diagnostics=False)
    except CapacityError as exc:
        result.update(status='decode_failure',reason=str(exc),packing_infeasibility_claimed=False)
    else:
        lowered=lower_initialization(logical,device,prediction,out/'initialization',method=r['method'])
        action=read(lowered['target'])
        action.update(schema='tailcompile-v7-qor-action-009-1',decoder_kind=prediction['decoder_kind'],
            shared_xy_sha256=state_sha256(model.shared_xy),constraints_sha256=constraints_digest(graph),
            collection_checkpoint_sha256=sha(r['checkpoint']),collection_arm=r['method'])
        write(lowered['target'],action)
        lowered.update(target_sha256=sha(lowered['target']),shared_xy_sha256=action['shared_xy_sha256'],
            constraints_sha256=action['constraints_sha256'],collection_checkpoint_sha256=action['collection_checkpoint_sha256'],
            deployed_action_sha256=prediction['deployed_action_sha256'],risk_diagnostics_computed=prediction['risk_diagnostics_computed'])
        write(out/'initialization/generation.json',lowered); result.update(lowered)
    result['worker_internal_seconds']=time.perf_counter()-start
    write(out/'generation_report.json',result); print('TAILCOMPILE_QOR_WORKER_COMPLETE',flush=True)
    return result


class QorCampaign(AlignedCampaign):
    def __init__(self,args):
        self.args,self.out=args,owned(args.out)
        self.conditions=panel(args.source)
        self.train=[c for c in self.conditions if c['split']=='train']
        self.reference=[c for c in self.train if c['width_role']=='reference']
        self.validation=[c for c in self.conditions if c['split']=='validation']
        self.budget=resource_budget(args.jobs,args.per_job_mib); self.binary=None; self._devices={}
        self.previous=legacy(args.source)

    def setup(self):
        from install_layer_hook import verify
        if self.args.source==self.out or self.args.source in self.out.parents or self.out in self.args.source.parents:
            raise ValueError('source008 and new009 must be disjoint')
        p=read(self.args.source/'aligned_protocol.json'); self.args.backend=Path(p['backend']).resolve()
        for name in ('layer_protocol.json','borrowed/conditions.json','borrowed/qualified_source_manifest.json'):
            path=self.previous/name
            if sha(path)!=p['source_inputs'].get(str(path.resolve())):
                raise ValueError('frozen source007 metadata changed: '+name)
        verify(self.args.backend); manifest=read(self.args.backend/'backend_manifest.json')
        self.binary=Path(manifest['binary'])
        if sha(self.binary)!=manifest['binary_sha256'] or manifest['binary_sha256']!=p['binary_sha256']:
            raise ValueError('borrowed008 binary changed')
        old=read(self.previous/'layer_protocol.json'); qpath=self.previous/'qualification/qualification_report.json'
        q=read(qpath)
        if not q.get('passed') or q['binary_sha256']!=p['binary_sha256'] or sha(qpath)!=old['qualification_sha256']:
            raise ValueError('borrowed binary qualification mismatch')
        self.catalog=read(self.previous/'borrowed/qualified_source_manifest.json')
        old_reference=Path(self.catalog['source'])/'binary_identity.json'
        if sha(old_reference)!=self.catalog['files'][str(old_reference.resolve())]: raise ValueError('old binary identity changed')
        self.old_binary_hash=read(old_reference)['sha256']
        self.teacher_rows={r['condition_id']:r for r in read(self.args.source/'teacher_records.json')['records']}
        self.source_model_report=read(self.args.source/'prepared_models_report.json')['arms']['L']
        self.source_checkpoint=Path(self.source_model_report['checkpoint'])
        inputs={str(self.args.source/n):sha(self.args.source/n) for n in
                ('aligned_protocol.json','layer_report.json','teacher_records.json','feedback.json','teacher_alignment_report.json','prepared_models_report.json')}
        inputs[str(self.source_checkpoint)]=sha(self.source_checkpoint)
        for c in self.conditions:
            row=self.teacher_rows[c['id']]
            if row['split']!=c['split'] or row['family']!=c['family']: raise ValueError('teacher split mismatch')
            if sha(row['graph'])!=p['source_inputs'].get(row['graph']): raise ValueError('borrowed graph changed')
            inputs[row['graph']]=sha(row['graph']); inputs[row['teacher']]=sha(row['teacher'])
        # All old inputs and implementation remain read-only; runtime configuration is per-stage.
        for name,expected in p['implementation'].items():
            if sha(HERE/name)!=expected: raise ValueError('frozen008 implementation changed: '+name)
        freeze(self.out/'qor_protocol.json',dict(plan=queue_plan(),source=str(self.args.source),legacy_source=str(self.previous),
            source_inputs=inputs,source_checkpoint=str(self.source_checkpoint),binary_sha256=p['binary_sha256'],
            backend=str(self.args.backend),implementation={n:sha(HERE/n) for n in CODE},timeout=self.args.timeout,
            new_backend_builds=0,new_rrg_extractions=0))
        freeze(self.out/'queue_plan.json',queue_plan())
        self.binary_reference=self.out/'binary_identity.json'
        freeze(self.binary_reference,dict(path=str(self.binary),sha256=p['binary_sha256']))
        st=self.binary.stat(); self.binary_stat=(st.st_size,st.st_mtime_ns,st.st_ino)
        freeze(self.out/'execution'/self.args.stage/'protocol.json',dict(device=self.args.device,jobs=self.args.jobs,
            per_job_mib=self.args.per_job_mib,timeout=self.args.timeout,physical_cores=self.budget['physical_cores'],
            effective_lanes=self.budget['effective']))

    def device(self,c):
        if c['id'] in self._devices: return self._devices[c['id']]
        grid=c['record']['grid']; arch=Path(self.catalog['architectures'][f'{grid[0]}x{grid[1]}/{c["architecture"]}'])
        if sha(arch)!=self.catalog['files'][str(arch.resolve())]: raise ValueError('borrowed architecture changed')
        key=identity(dict(architecture=sha(arch),binary=self.old_binary_hash,width=c['width']))[:20]
        folder=Path(self.catalog['devices'][key]); cost=read(self.previous/'vertical_devices'/key/'device_preparation_report.json')
        if sha(folder/'manifest.json')!=self.catalog['files'][str((folder/'manifest.json').resolve())]:
            raise ValueError('borrowed device manifest changed')
        summary=self.previous/'vertical_devices'/key/'summary.json'
        if sha(summary)!=cost['summary_sha256']: raise ValueError('borrowed vertical summary changed')
        value=dict(architecture=str(arch),rr_graph=str(folder/'rr_graph.xml'),device=str(folder/'device_graph.json'),
                   device_manifest=str(folder/'manifest.json'),vertical_summary=str(summary))
        self._devices[c['id']]=value
        return value

    def request(self,c,checkpoint=None,method='T',prefix=None,noise=0.):
        return dict(self.device(c),prefix=prefix or c['record']['structure_prefix'],design=c['design'],family=c['family'],
                    split=c['split'],blif=c['record']['blif'],checkpoint=str(checkpoint),method=method,generation_seed=9301)

    def generate(self,c,target,checkpoint=None,method='T',prefix=None,noise=0.,core=None):
        self.guard(); freeze(target/'request.json',self.request(c,checkpoint,method,prefix,noise))
        result=run_trial(target/'worker',[sys.executable,HERE/'qor_campaign.py','worker','--request',target/'request.json',
            '--out',target/'payload'],inputs=[target/'request.json']+[HERE/n for n in CODE],timeout=self.args.timeout,
            affinity=core,marker='TAILCOMPILE_QOR_WORKER_COMPLETE',prune=False)
        if result['status']!='complete': raise RuntimeError('009 generator worker failed: '+result['log'])
        value=read(target/'payload/generation_report.json'); value['generation_process_seconds']=result['metrics']['wall_seconds']
        return value

    @staticmethod
    def prune_feedback_graph(folder):
        """Training reuses signed source graphs; candidate actions stay intact."""
        removed=[]
        for name in ('graph.pt','logic.json'):
            path=folder/'generation/payload'/name
            if path.is_file() and not path.is_symlink() and folder.resolve() in path.resolve().parents:
                removed.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)));path.unlink()
        write(folder/'payload_pruned.json',dict(files=removed,removed_bytes=sum(r['bytes'] for r in removed),
            source_training_graphs_and_actions_preserved=True))

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
        if generated and status in ('success','unroutable') and ('TAILCOMPILE_V7_GP_INIT_CONSUMED' not in text or
            'TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' not in text): status='interface_failure'
        row=dict(condition_id=c['id'],design=c['design'],family=c['family'],split=c['split'],architecture=c['architecture'],
            width=c['width'],width_role=c['width_role'],method=method,seed=seed,status=status,
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
        write(target/'metrics.json',row)
        if status in ('success','unroutable','complete'):
            prune_derived(result['workdir'],protected=inputs)
            removed=[]
            for trace in list(target.glob('post_gp.*'))+[Path(result['workdir'])/n for n in ('post_route.fplace','post_legalizer.fplace')]:
                if trace.is_file() and not trace.is_symlink() and target.resolve() in trace.resolve().parents:
                    removed.append(dict(path=str(trace),sha256=sha(trace),bytes=trace.stat().st_size)); trace.unlink()
            write(target/'trace_pruned_report.json',dict(removed=removed,observations_and_route_usage_durable=True))
        return row

    def prepare(self):
        import torch
        from aligned_teacher import verify_teacher,_digest
        from train_aligned_layers import load_model
        from train import state_sha256
        torch.set_num_threads(1); rows=[]; audits=[]
        for c in self.conditions:
            row=self.teacher_rows[c['id']]; graph=torch.load(row['graph'],weights_only=False,map_location='cpu')
            target=read(row['teacher']); check=verify_teacher(graph,target)
            source_audit=read(self.args.source/'aligned_teachers'/c['id']/'projection_audit.json')
            if source_audit['derived_sha256']!=_digest(target): raise ValueError('derived008 teacher identity changed')
            rows.append(dict(row)); audits.append(dict(condition_id=c['id'],split=c['split'],valid=check['valid']))
        model=load_model(self.source_checkpoint)
        if state_sha256(model)!=self.source_model_report['final_state_sha256']:
            raise ValueError('source008 prepared model weights changed')
        freeze(self.out/'teacher_records.json',dict(records=rows))
        freeze(self.out/'source_feedback.json',read(self.args.source/'feedback.json'))
        freeze(self.out/'source_model_report.json',dict(checkpoint=str(self.source_checkpoint),sha256=sha(self.source_checkpoint),
            state_sha256=state_sha256(model),shared_xy_sha256=state_sha256(model.shared_xy)))
        write(self.out/'prepared_report.json',dict(complete=True,conditions=len(rows),train_conditions=len(self.train),
            active_validation_conditions=len(self.validation),teacher_audits=audits,source_inputs_unchanged=True))

    def bootstrap(self):
        from train_qor import train_bootstrap
        if (self.out/'prepared_models_report.json').exists(): return
        records=[r for r in read(self.out/'teacher_records.json')['records'] if r['split']=='train']
        result=train_bootstrap(records,self.source_checkpoint,self.fresh_attempt(self.out/'models/bootstrap'),
            read(self.out/'source_feedback.json'),epochs=5,seed=9301,device=self.args.device)
        freeze(self.out/'prepared_models_report.json',result)

    def feedback(self):
        prepared=read(self.out/'prepared_models_report.json'); actions=[]
        cores=self.budget['physical_cores'][:self.budget['effective']]
        for c in self.reference: self.device(c)
        def job(item):
            i,c=item; core=cores[i]; os.sched_setaffinity(0,{core}); pool=[]; seen={}
            for arm in ARMS:
                folder=self.out/'feedback'/c['id']/arm
                generated=self.generate(c,folder/'generation',prepared['arms'][arm]['checkpoint'],arm,core=core)
                if not generated.get('ap_init'):
                    pool.append(dict(candidate_id=arm,condition_id=c['id'],action=None,trials=[],generation=generated)); continue
                digest=generated['ap_init_sha256']
                if digest in seen:
                    write(folder/'duplicate_report.json',dict(alias_of=seen[digest],ap_init_sha256=digest))
                    self.prune_feedback_graph(folder); continue
                seen[digest]=arm
                trials=[self.pnr(c,s,folder/f's{s}',arm,generated,core) for s in (9101,9102)]
                pool.append(dict(candidate_id='009-'+arm,condition_id=c['id'],action=generated['target'],trials=trials,generation=generated))
                self.prune_feedback_graph(folder)
            controls=[self.pnr(c,s,self.out/'feedback'/c['id']/'B'/f's{s}','B',core=core) for s in (9101,9102)]
            for a in pool: a['default_trials']=controls
            return pool
        for start in range(0,len(self.reference),len(cores)):
            with ThreadPoolExecutor(max_workers=len(cores)) as executor:
                groups=list(executor.map(job,enumerate(self.reference[start:start+len(cores)])))
            actions.extend(a for group in groups for a in group)
            write(self.out/'feedback_progress_report.json',dict(completed_conditions=min(start+len(cores),len(self.reference)),actions=len(actions)))
        write(self.out/'feedback.json',actions)
        write(self.out/'feedback_report.json',dict(actions=len(actions),planned_pnr_upper=96,
            measured_pnr=sum(len(a['trials']) for a in actions)+2*len(self.reference),
            successful=sum(t['status']=='success' for a in actions for t in a['trials'])))

    def train_models(self):
        from train_qor import train_feedback
        if (self.out/'trained_models_report.json').exists(): return
        records=[r for r in read(self.out/'teacher_records.json')['records'] if r['split']=='train']
        feedback=read(self.out/'source_feedback.json')+read(self.out/'feedback.json')
        result=train_feedback(records,read(self.out/'prepared_models_report.json'),self.fresh_attempt(self.out/'models/feedback'),
            feedback,epochs=10,seed=9301,device=self.args.device)
        freeze(self.out/'trained_models_report.json',result)

    def validate(self):
        arms=read(self.out/'trained_models_report.json')['arms']; epochs=(5,10)
        checkpoints={m:arms[m]['checkpoints'] for m in ARMS}; core=self.budget['physical_cores'][0]
        if any(len(v)!=2 for v in checkpoints.values()): raise ValueError('009 validation requires epoch5/10 snapshots')
        freeze(self.out/'validation/timing_protocol.json',dict(core=core,sequential=True,uname=list(os.uname()),
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            checkpoints={m:{str(p):sha(p) for p in paths} for m,paths in checkpoints.items()}))
        rows=[]
        for c,s in itertools.product(self.validation,(9201,9202)):
            rows.append(dict(self.cold(c,s,'B',None,self.out/'validation'/c['id']/f's{s}'/'B',core),epoch=0))
            for index,e in enumerate(epochs):
                for m in ARMS:
                    rows.append(dict(self.cold(c,s,m,checkpoints[m][index],self.out/'validation'/c['id']/f's{s}'/f'{m}_e{e}',core),epoch=e))
        selection=choose_epoch(rows,epochs=epochs); index=list(epochs).index(selection['epoch'])
        selection['checkpoints']={m:checkpoints[m][index] for m in ARMS}
        selection['checkpoint_sha256']={m:sha(p) for m,p in selection['checkpoints'].items()}
        freeze(self.out/'selection.json',selection)
        write(self.out/'validation/report.json',dict(rows=rows,selection=selection))

    def compare(self):
        checkpoints=read(self.out/'selection.json')['checkpoints']
        cases=list(itertools.product(self.reference,range(9001,9006)))
        available=self.budget['physical_cores'][:self.budget['effective']]
        lanes=max(n for n in range(1,len(available)+1) if len(cases)%n==0); cores=available[:lanes]
        host=dict(uname=list(os.uname()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cores=cores)
        freeze(self.out/'comparison/timing_protocol.json',dict(host=host,cases=[[c['id'],s] for c,s in cases],methods=list(METHODS),
            seed_rotation='case index modulo six',checkpoints={m:sha(p) for m,p in checkpoints.items()},fixed_lane_barriers=True))
        rows=[]
        for start in range(0,len(cases),lanes):
            wave=self.out/'comparison'/f'wave_{start//lanes:04d}'
            signature=dict(host=host,cases=[[c['id'],s] for c,s in cases[start:start+lanes]],
                timing_protocol_sha256=sha(self.out/'comparison/timing_protocol.json'))
            completed=wave/'completed.json'
            if completed.exists():
                cached=read(completed)
                if cached['signature']!=signature: raise ValueError('timing host/input signature changed; use new cohort')
                rows.extend(cached['rows']); continue
            if len(list(wave.glob('attempt_*')))>=2: raise RuntimeError('whole timing-wave retry budget exhausted')
            attempt=self.fresh_attempt(wave); barrier=threading.Barrier(lanes)
            def lane(item):
                i,(c,s)=item; core=cores[i]; os.sched_setaffinity(0,{core})
                offset=(start+i)%len(METHODS); order=METHODS[offset:]+METHODS[:offset]; result=[]
                try:
                    for m in order:
                        barrier.wait(timeout=self.args.timeout*4)
                        row=self.cold(c,s,m,checkpoints.get(m),attempt/f'lane_{i:03d}'/c['id']/f's{s}'/m,core)
                        result.append(dict(row,order=list(order),wave_load=lanes,host=host))
                        barrier.wait(timeout=self.args.timeout*4)
                except BaseException:
                    barrier.abort(); raise
                return result
            with ThreadPoolExecutor(max_workers=lanes) as pool:
                groups=list(pool.map(lane,enumerate(cases[start:start+lanes])))
            wave_rows=[r for group in groups for r in group]
            freeze(completed,dict(signature=signature,rows=wave_rows,actual_attempt=str(attempt),
                previous_incomplete_attempts=len(list(wave.glob('attempt_*')))-1,previous_incomplete_costs_retained=True))
            rows.extend(wave_rows)
            write(self.out/'comparison/report.json',dict(planned=240,completed=len(rows),rows=rows))

    def report(self):
        rows=read(self.out/'comparison/report.json')['rows']
        expected=[(c['id'],s,m) for c,s,m in itertools.product(self.reference,range(9001,9006),METHODS)]
        result=evaluate(rows,expected=expected)
        primary=next(r for r in result['comparisons'] if (r['method'],r['baseline'],r['width_role'])==('T','B','reference'))
        if not result['complete']: result['conclusion']='incomplete_comparison'
        elif any(r['status'] in ('infrastructure_failure','interface_failure','interrupted') for r in rows):
            result['conclusion']='infrastructure_or_interface_requires_diagnosis'
        elif primary['joint_q'] is None: result['conclusion']='insufficient_successful_qor_pairs'
        else:
            result['conclusion']=('' if primary['full_success_coverage'] and primary['full_qor_metric_coverage'] else 'conditional_')+primary['classification']
        result.update(actual_comparison_vpr_tasks=sum(r.get('vpr_manifest') is not None for r in rows),
            comparison_decode_or_preparation_failures=sum(r.get('vpr_manifest') is None for r in rows))
        write(self.out/'qor_report.json',result)
        print(result.get('conclusion'),result.get('status_counts'),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=STAGES+('worker',))
    p.add_argument('--source',type=Path,default=HERE/'runs/development_008_aligned')
    p.add_argument('--out',type=Path,default=HERE/'runs/development_009_qor')
    p.add_argument('--request',type=Path); p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--per-job-mib',type=int,default=8192); p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--device',default='cpu'); args=p.parse_args()
    if min(args.jobs,args.per_job_mib,args.timeout)<1: p.error('positive resource limits required')
    install_signal_handlers(); args.source=args.source.resolve(); args.out=owned(args.out)
    if args.stage=='worker': worker(read(args.request),args.out); return
    if args.stage=='plan': print(__import__('json').dumps(queue_plan(),indent=2)); return
    if args.stage=='check':
        result=check(args); print(__import__('json').dumps(result,indent=2)); raise SystemExit(0 if result['ready'] else 2)
    with campaign_lock(args.out,'qor-'+args.stage):
        campaign=QorCampaign(args); campaign.setup()
        getattr(campaign,'train_models' if args.stage=='train' else args.stage)()


if __name__=='__main__': main()
