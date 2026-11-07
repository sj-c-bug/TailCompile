"""016 four-arm test of bounded explainable net-level refinement."""
import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import os
from pathlib import Path
import shutil
import sys
import threading

from campaign import campaign_lock, resource_budget, identity
from io_utils import read, write, freeze, sha, run_trial, install_signal_handlers, raise_if_cancelled
from repository_guard import validate_output_roots
from architectures import COMMIT
from explainable_native import METHODS, SEEDS, pinned, device_signature, run_native

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STAGES = ('prepare', 'predict', 'compare', 'report')
CODE = tuple(sorted(p.name for p in HERE.iterdir() if p.suffix in ('.py','.inc','.sh')))


def repository_path(value):
    """Accept canonical/mount aliases of historical paths, never a sibling repo."""
    path = Path(value)
    if path.is_file():
        resolved = path.resolve()
        if REPO not in resolved.parents:
            raise ValueError('016 historical payload outside canonical checkout: '+str(path))
        return resolved
    text = str(value).replace('\\','/')
    if '/TailCompile/' in text:
        path = REPO/text.split('/TailCompile/',1)[1]
    elif not path.is_absolute():
        path = REPO/path
    path = path.resolve()
    if REPO not in path.parents:
        raise ValueError('016 historical payload escapes checkout')
    return path


def source_model():
    source = HERE/'runs/development_012_edit_context_parallel/edit_context_protocol.json'
    protocol = read(source)
    checkpoint = repository_path(protocol['base_checkpoint'])
    if not checkpoint.is_relative_to(HERE/'runs') or not checkpoint.relative_to(HERE/'runs').parts[0].startswith('development_010'):
        raise ValueError('016 D must use the frozen010 checkpoint')
    pinned(checkpoint, protocol['base_checkpoint_sha256'])
    return dict(path=str(checkpoint), sha256=protocol['base_checkpoint_sha256'],
                source_protocol=str(source), source_protocol_sha256=sha(source))


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('plan','check',*STAGES,'all'))
    p.add_argument('--out',type=Path,default=HERE/'runs/development_016_explainable')
    p.add_argument('--manifest',type=Path,default=HERE/'attribution_panel_015.json')
    p.add_argument('--backend-manifest',type=Path,default=REPO/'.deps/tailcompile_v7_3d_layer/vtr/backend_manifest.json')
    p.add_argument('--qualification-report',type=Path,default=HERE/'runs/development_007_layer/qualification/qualification_report.json')
    p.add_argument('--jobs',type=int,default=16)
    p.add_argument('--per-job-mib',type=int,default=8192)
    p.add_argument('--rrg-jobs',type=int,default=2)
    p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--worker-timeout',type=int,default=180)
    p.add_argument('--refine-max-seconds',type=float,default=20.)
    p.add_argument('--refine-hard-timeout',type=int,default=60)
    args = p.parse_args(argv)
    args.out = args.out.resolve()
    if args.out.parent != HERE/'runs' or not args.out.name.startswith('development_016_'):
        p.error('016 output must use a direct development_016_ cohort under v7_3d/runs')
    if min(args.jobs,args.per_job_mib,args.rrg_jobs,args.timeout,args.worker_timeout,args.refine_hard_timeout)<1:
        p.error('016 counts/budgets must be positive')
    if not 0 < args.refine_max_seconds <= 120 or not args.refine_max_seconds < args.refine_hard_timeout < args.worker_timeout <= 600:
        p.error('016 requires 0 < refine soft seconds < hard timeout < worker timeout <= 600; soft <=120')
    args.scratch = REPO/'vtr_work/tailcompile_v7_3d/v8_016'/args.out.name
    return args


def checked_backend(args):
    backend = read(args.backend_manifest)
    binary = Path(backend['binary']).resolve()
    if backend['vtr_commit'] != COMMIT:
        raise ValueError('016 requires pinned40f8 backend')
    pinned(binary,backend['binary_sha256'])
    proof = read(args.qualification_report)
    if not proof.get('passed') or not proof.get('complete') or proof['binary_sha256'] != backend['binary_sha256']:
        raise ValueError('016 requires existing passing XYZ/backend qualification')
    return binary,backend['binary_sha256']


class Controller:
    def __init__(self,args):
        self.args=args; self.out=args.out
        self.binary,self.binary_hash=checked_backend(args)
        self.binary_stat=self.stamp(self.binary)
        validate_output_roots((args.out,args.scratch),dependency_roots=(self.binary.parent,))
        self.budget=resource_budget(args.jobs,args.per_job_mib)
        self.rrg_slots=threading.Semaphore(min(args.rrg_jobs,self.budget['effective']))
        self.binary_reference=self.out/'binary_identity.json'
        freeze(self.binary_reference,dict(binary=str(self.binary),sha256=self.binary_hash,
            backend_manifest_sha256=sha(args.backend_manifest),commit=COMMIT))
        self.source_stats={}

    @staticmethod
    def stamp(path):
        stat=Path(path).stat()
        return stat.st_size,stat.st_mtime_ns,stat.st_ino

    def guard(self):
        raise_if_cancelled()
        if self.stamp(self.binary)!=self.binary_stat:
            raise ValueError('016 executable changed during run')
        for path,stamp in self.source_stats.items():
            if self.stamp(path)!=stamp:
                raise ValueError('016 frozen input changed during run: '+path)
        if shutil.disk_usage(self.out).free < 8*1024**3:
            raise OSError('016 requires 8GiB headroom; dispatch stopped')

    def parallel(self,items,function):
        items=iter(items); results=[]; pending=set()
        with ThreadPoolExecutor(max_workers=self.budget['effective']) as pool:
            def submit():
                try: item=next(items)
                except StopIteration: return False
                self.guard(); pending.add(pool.submit(function,item)); return True
            for _ in range(self.budget['effective']):
                if not submit(): break
            while pending:
                finished,_=wait(pending,return_when=FIRST_COMPLETED)
                pending.difference_update(finished)
                for task in finished: results.append(task.result())
                for _ in finished:
                    if not submit(): break
        return results


def checked_panel(args):
    from attribution_panel import load_panel
    print('V8_016_CHECK_INPUT_PAYLOAD_START',flush=True)
    panel=load_panel(args.manifest,repo=REPO,verify_payload=True)
    print('V8_016_CHECK_INPUT_PAYLOAD_COMPLETE families='+str(panel['family_count']),flush=True)
    return panel


def verify_panel(controller,panel):
    for c in panel['conditions']:
        proof=read(c['consumer_manifest'])
        signature=proof['signature']; command=signature['command']
        if proof['status']!='success' or proof['returncode']!=0:
            raise ValueError('016 source condition lacks successful native consumer')
        # Frozen trials pin a compact binary_identity file rather than the
        # executable itself. Follow that exact receipt, not the old path name.
        refs=[]
        for path,digest in signature['inputs'].items():
            if Path(path).name=='binary_identity.json':
                file=repository_path(path); pinned(file,digest); refs.append(read(file))
        if not refs or any(r.get('sha256')!=controller.binary_hash for r in refs):
            raise ValueError('016 source consumer used another backend')
        fields=c['fields']
        if (repository_path(command[1])!=Path(fields['arch']).resolve() or
            repository_path(command[2])!=Path(fields['blif']).resolve() or
            int(command[command.index('--route_chan_width')+1])!=c['width'] or
            repository_path(command[command.index('--read_rr_graph')+1])!=Path(fields['rr_graph']).resolve()):
            raise ValueError('016 source consumer device/netlist mismatch')
        device_signature(c,controller.binary_hash)
    for path,digest in panel['sources'].items():
        file=repository_path(path); pinned(file,digest)
        controller.source_stats[str(file)]=controller.stamp(file)
    for path in panel.get('payload_files',{}):
        file=Path(path)
        controller.source_stats[str(file)]=controller.stamp(file)


def refinement_config(args):
    from explainable_refinement import DEFAULT_CONFIG
    return dict(DEFAULT_CONFIG, max_seconds=args.refine_max_seconds)


def prepare(controller,panel,model):
    frozen=dict(schema='v8-016-attribution-1', panel=panel, model=model, methods=METHODS,seeds=SEEDS,
        binary=str(controller.binary),binary_sha256=controller.binary_hash, backend_manifest_sha256=sha(controller.args.backend_manifest),
        qualification_sha256=sha(controller.args.qualification_report),
        code={name:sha(HERE/name) for name in CODE}, timeout=controller.args.timeout,
        inherited_code={str(p.relative_to(REPO)):sha(p) for p in
            (HERE.parent/'tailcompile_v68/model.py',HERE.parent/'tailcompile_v68/data.py',
             HERE.parent/'tailcompile_v69/phase_metrics.py',HERE.parent/'tailrisk_core.py')},
        jobs=controller.args.jobs,per_job_mib=controller.args.per_job_mib,
        rrg_jobs=controller.args.rrg_jobs, max_native_runs=len(panel['conditions'])*len(SEEDS)*len(METHODS),
        worker_timeout=controller.args.worker_timeout, refinement_config=refinement_config(controller.args),
        refinement_hard_timeout=controller.args.refine_hard_timeout,
        initialization_rule='positive net-level portal-aware HPWL gain only; stay always available',
        termination_claim='budget termination is not a local optimum; native VPR still continues',
        old_weak_generator=dict(arm='J',strengths=[.15,.35],max_moves=256,fraction=.02,seed=11301),
        new_training=False, Wmin_measured=False, metric_priority=['wirelength','cpd_ns'],
        simultaneous_improvement_required=False, selection_after_QoR=False,python=sys.version)
    # Round-trip tuples to the persisted JSON representation before comparison.
    freeze(controller.out/'explainable_protocol.json',json.loads(json.dumps(frozen)))
    freeze(controller.out/'qualified_panel.json',dict(conditions=panel['conditions'],sources=panel['sources']))
    write(controller.out/'resources.json',dict(cpu=controller.budget,rrg_jobs=controller.args.rrg_jobs))
    for name in CODE:
        file=HERE/name
        controller.source_stats[str(file)]=controller.stamp(file)


def prepare_vertical_summary(condition,target,*,max_seconds,expected_device_signature,source_root=None):
    """Reuse only a receipt-pinned 015 cache, read-only; otherwise derive anew.

    The old selection lock pins the global report, which pins this condition's
    request, which pins the complete cache bytes. prepare_device additionally
    verifies today's implementation, full RRG digest and semantic device hash.
    A matching embedded signature alone never authorizes an old payload.
    """
    import hashlib
    from vertical_risk import prepare_device
    target=Path(target).resolve()
    validate_output_roots((target,))
    if HERE/'runs' not in target.parents or not target.relative_to(HERE/'runs').parts[0].startswith('development_016_'):
        raise ValueError('016 vertical cache must stay in its owning cohort')
    receipt_path=target.with_name('vertical_cache_report.json')
    fields=condition['fields']; device=read(fields['device'])
    def checked_json(path,digest):
        raw=Path(path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=digest:
            raise ValueError('015 cache receipt mismatch: '+str(path))
        return json.loads(raw)
    # An interrupted copy/derivation can finish the disposable cache before
    # publishing its receipt. No worker request can yet pin it at this point.
    # Recreate only this owned derived file; never change the old source.
    if target.is_file() and not receipt_path.exists():
        target.unlink()
        print('V8_016_VERTICAL_CACHE_RECOVER_UNPUBLISHED condition='+condition['id'],flush=True)
    if target.exists():
        receipt=read(receipt_path)
        pinned(target,receipt['target_sha256'])
        if receipt.get('device_signature')!=expected_device_signature:
            raise ValueError('016 resumed vertical cache belongs to another device')
        prepare_device(fields['rr_graph'],device,target,max_seconds=max_seconds)
        print('V8_016_VERTICAL_CACHE_RESUME condition='+condition['id'],flush=True)
        return receipt
    root=Path(source_root or HERE/'runs/development_015_attribution').resolve()
    if root.parent!=HERE/'runs' or not root.name.startswith('development_015_'):
        raise ValueError('016 cache source must be a read-only015 cohort')
    copied=False
    reason=None
    source={}
    try:
        selection_path=root/'predictions/selection.json'
        selection=read(selection_path)
        if selection.get('pre_measurement') is not True:
            raise ValueError('015 source has no frozen pre-measurement selection lock')
        report_path=root/'predictions/prediction_report.json'
        old=checked_json(report_path,selection['predictions_sha256'])
        row=old['conditions'][condition['id']]
        folder=(root/'predictions'/condition['id']).resolve()
        if root/'predictions' not in folder.parents:
            raise ValueError('015 condition cache escapes prediction directory')
        request_path=folder/'request.json'
        request=checked_json(request_path,row['request_sha256'])
        if row['device_signature']!=expected_device_signature or request['device_signature']!=expected_device_signature:
            raise ValueError('015 source uses another device signature')
        cache=repository_path(request['vertical_summary'])
        if cache!=folder/'vertical_summary.json':
            raise ValueError('015 cache does not belong to this condition')
        matches=[digest for path,digest in request['inputs'].items() if repository_path(path)==cache]
        if len(matches)!=1:
            raise ValueError('015 request does not uniquely pin its vertical cache')
        raw=cache.read_bytes()
        if hashlib.sha256(raw).hexdigest()!=matches[0]:
            raise ValueError('015 cache content differs from frozen request')
        # Exact bytes preserve the old pinned cache digest. Writes are owned016.
        target.parent.mkdir(parents=True,exist_ok=True)
        staged=target.with_name(target.name+'.reuse.tmp')
        try:
            staged.write_bytes(raw)
            staged.replace(target)
        finally:
            if staged.is_file(): staged.unlink()
        copied=True
        prepare_device(fields['rr_graph'],device,target,max_seconds=max_seconds)
        source=dict(source_cohort=str(root),source_cache=str(cache),source_cache_sha256=matches[0],
            source_request=str(request_path),source_request_sha256=row['request_sha256'],
            source_prediction_report=str(report_path),source_prediction_report_sha256=selection['predictions_sha256'],
            source_selection_sha256=sha(selection_path))
    except (OSError,ValueError,KeyError,TypeError) as exc:
        reason=type(exc).__name__+': '+str(exc)
        if copied:
            target.unlink()
        print('V8_016_VERTICAL_CACHE_REBUILD condition='+condition['id']+' reason='+reason,flush=True)
        prepare_device(fields['rr_graph'],device,target,max_seconds=max_seconds)
    receipt=dict(schema='v8-016-vertical-cache-1',mode='reused_015' if copied and reason is None else 'rebuilt',
        condition_id=condition['id'],device_signature=expected_device_signature,target_sha256=sha(target),
        current_signature_verified=True,reuse_rejected_reason=reason,**source)
    freeze(receipt_path,receipt)
    print('V8_016_VERTICAL_CACHE_READY condition='+condition['id']+' mode='+receipt['mode'],flush=True)
    return receipt


def predict(controller,panel,model):
    locked=controller.out/'predictions/prediction_report.json'
    if locked.exists():
        return locked_predictions(controller,panel,model)
    if (controller.out/'comparison').exists():
        raise ValueError('016 cannot generate predictions after scientific measurements')
    def worker(c):
        folder=controller.out/'predictions'/c['id']; folder.mkdir(parents=True,exist_ok=True)
        f=c['fields']; summary=folder/'vertical_summary.json'
        print('V8_016_VERTICAL_SUMMARY_START condition='+c['id'],flush=True)
        with controller.rrg_slots:
            cache_provenance=prepare_vertical_summary(c,summary,max_seconds=controller.args.timeout,
                expected_device_signature=device_signature(c,controller.binary_hash))
        print('V8_016_VERTICAL_SUMMARY_COMPLETE condition='+c['id'],flush=True)
        inputs={str(Path(f[k]).resolve()):sha(f[k]) for k in ('arch','blif','device','device_manifest')}
        for suffix in ('.blocks.tsv','.nets.tsv','.meta.tsv','.pins.tsv'):
            path=Path(c['record']['structure_prefix']+suffix)
            if path.is_file(): inputs[str(path.resolve())]=sha(path)
        for path in (Path(model['path']),summary,summary.with_name('vertical_cache_report.json')): inputs[str(path.resolve())]=sha(path)
        req=dict(out=str(controller.out),condition=c,model=model,inputs=inputs,
                 refinement_config=refinement_config(controller.args),refinement_hard_timeout=controller.args.refine_hard_timeout,
                 code={name:sha(HERE/name) for name in CODE},
                 vertical_summary=str(summary),vertical_cache_provenance=cache_provenance,device_signature=device_signature(c,controller.binary_hash))
        freeze(folder/'request.json',req)
        trial=run_trial(folder/'worker',[sys.executable,HERE/'explainable_native.py','worker',
            folder/'request.json',folder/'payload'],inputs=[folder/'request.json',*inputs],
            timeout=controller.args.worker_timeout,marker='TAILCOMPILE_V8_016_PREDICTION_COMPLETE',prune=False)
        if trial['status']!='complete':
            raise RuntimeError('016 prediction failed: '+trial['log'])
        value=read(folder/'payload/prediction_report.json')
        value.update(request_sha256=sha(folder/'request.json'),payload_sha256=sha(folder/'payload/prediction_report.json'))
        return c['id'],value
    values=dict(controller.parallel(panel['conditions'],worker))
    report=dict(schema='v8-016-predictions-1',model=model,conditions=values)
    freeze(locked,report)
    freeze(controller.out/'predictions/selection.json',dict(predictions_sha256=sha(locked),
        conditions_sha256=identity(panel['conditions']),pre_measurement=True))
    return report


def locked_predictions(controller,panel,model):
    from explainable_native import validate_candidate
    path=controller.out/'predictions/prediction_report.json'; lock=read(path.parent/'selection.json')
    if lock['predictions_sha256']!=sha(path) or lock['conditions_sha256']!=identity(panel['conditions']):
        raise ValueError('016 prediction/condition lock mismatch')
    pred=read(path)
    if pred['model']!=model or set(pred['conditions'])!={c['id'] for c in panel['conditions']}:
        raise ValueError('016 frozen prediction source mismatch')
    for c in panel['conditions']:
        folder=path.parent/c['id']; value=pred['conditions'][c['id']]
        req=read(pinned(folder/'request.json',value['request_sha256']))
        payload=read(pinned(folder/'payload/prediction_report.json',value['payload_sha256']))
        if (value['candidates']!=payload['candidates'] or req['condition']!=c or req['model']!=model or
            req['refinement_config']!=refinement_config(controller.args) or
            req['refinement_hard_timeout']!=controller.args.refine_hard_timeout or
            req['code']!={name:sha(HERE/name) for name in CODE} or
            value['device_signature']!=device_signature(c,controller.binary_hash)):
            raise ValueError('016 prediction differs from original worker inputs')
        for source,digest in req['inputs'].items(): pinned(source,digest)
        for candidate in value['candidates'].values(): validate_candidate(candidate,controller.out)
    return pred


def compare(controller,panel,model):
    pred=locked_predictions(controller,panel,model)
    lock=controller.out/'predictions/prediction_report.json'
    freeze(controller.out/'comparison/protocol.json',dict(predictions_sha256=sha(lock),methods=METHODS,seeds=SEEDS))
    def measure(task):
        c,seed=task; values=[]
        # Fixed rotation balances queue position without outcome-dependent order.
        start=(seed-SEEDS[0])%len(METHODS); order=METHODS[start:]+METHODS[:start]
        for method in order:
            folder=controller.out/'comparison'/c['id']/f's{seed}'/method
            values.append(run_native(controller,c,seed,folder,method,pred['conditions'][c['id']]['candidates'].get(method)))
        return values
    tasks=[(c,s) for c in panel['conditions'] for s in SEEDS]
    rows=[r for batch in controller.parallel(tasks,measure) for r in batch]
    rows.sort(key=lambda r:(r['condition_id'],r['seed'],METHODS.index(r['method'])))
    freeze(controller.out/'comparison/trials_report.json',dict(complete=True,rows=rows,predictions_sha256=sha(lock)))


def verify_measurements(out,rows,conditions=None,predictions=None,protocol=None):
    indexed={c['id']:c for c in conditions or []}
    for row in rows:
        folder=out/'comparison'/row['condition_id']/f's{row["seed"]}'/row['method']
        if out/'comparison' not in folder.resolve().parents:
            raise ValueError('016 invalid measurement location')
        native=read(pinned(folder/'vpr/manifest.json',row['vpr_manifest_sha256']))
        if read(folder/'metrics.json')!=row:
            raise ValueError('016 summary differs from individual metrics')
        sig=native['signature']; cmd=sig['command']
        init=sig['env'].get('TC_GP_INIT')
        if (int(cmd[cmd.index('--seed')+1])!=row['seed'] or
            int(cmd[cmd.index('--route_chan_width')+1])!=row['width'] or
            sig['inputs'].get(init)!=row['ap_init_sha256'] or
            sig['metadata']!={'method':row['method'],'execution_signature':row['execution_signature']} or
            sig['env'].get('TC_GP_LAYER_PRIOR')!='0' or sig['env'].get('TC_GP_EXPORT')!='post_gp' or
            bool(init)!=(row['method']!='Default') or
            any(native['metrics'][k]!=row[k] for k in ('wirelength','cpd_ns','unique_chanz_nodes')) or
            native['metrics']['wall_seconds']!=row['vpr_seconds'] or native['status']!=row['status'] or
            (row['status']=='success' and native['returncode']!=0) or
            (row['status']=='unroutable' and native['returncode']!=2)):
            raise ValueError('016 native seed/initializer/QoR mismatch')
        if indexed:
            c=indexed[row['condition_id']]; fields=c['fields']
            expected=device_signature(c,protocol['binary_sha256'])
            candidate=predictions['conditions'][c['id']]['candidates'].get(row['method'])
            if (row['device_signature']!=expected or row['ap_init_sha256']!=(candidate or {}).get('ap_init_sha256') or
                init!=(candidate or {}).get('ap_init') or Path(cmd[0]).resolve()!=Path(protocol['binary']).resolve() or
                any(Path(actual).resolve()!=Path(fields[k]).resolve() for k,actual in
                    (('arch',cmd[1]),('blif',cmd[2]),('rr_graph',cmd[cmd.index('--read_rr_graph')+1]))) or
                sig['timeout']!=protocol['timeout']):
                raise ValueError('016 native command/device differs from frozen protocol')
            execution=identity(dict(device_signature=expected,command=cmd,condition_id=c['id'],
                native_env=dict(TC_GP_LAYER_PRIOR='0',TC_GP_EXPORT='post_gp'),timeout=protocol['timeout']))
            receipt=out/'binary_identity.json'
            input_pins={str(repository_path(k)):v for k,v in sig['inputs'].items()}
            if (execution!=row['execution_signature'] or input_pins.get(str(receipt))!=sha(receipt) or
                read(receipt)['sha256']!=protocol['binary_sha256']):
                raise ValueError('016 native executable/command signature mismatch')


def report_inputs(out):
    protocol=read(out/'explainable_protocol.json')
    panel=read(out/'qualified_panel.json')
    path=out/'predictions/prediction_report.json'; predictions=read(path)
    lock=read(path.parent/'selection.json')
    raw=read(out/'comparison/trials_report.json')
    if (not raw['complete'] or raw['predictions_sha256']!=sha(path) or lock['predictions_sha256']!=sha(path) or
        lock['conditions_sha256']!=identity(panel['conditions']) or
        protocol['panel']['conditions']!=panel['conditions'] or protocol['model']!=predictions['model'] or
        tuple(protocol['seeds'])!=SEEDS or tuple(protocol['methods'])!=METHODS):
        raise ValueError('016 report must use locked predictions and frozen protocol')
    verify_measurements(out,raw['rows'],panel['conditions'],predictions,protocol)
    return raw,panel,predictions


def main(argv=None):
    args=arguments(argv)
    if args.stage=='plan':
        from attribution_panel import load_panel
        panel=load_panel(args.manifest,repo=REPO,verify_payload=False)
        designs=[dict(design=d['design'],family=d['family'],grid_xy=d.get('grid'),
            widths=sorted({c['width'] for c in panel['conditions'] if c['design']==d['design']}),
            exposure=d.get('exposure')) for d in panel['designs']]
        print(json.dumps(dict(designs=designs,family_count=panel['family_count'],methods=METHODS,
            seeds=SEEDS,max_native_runs=len(panel['conditions'])*len(SEEDS)*len(METHODS),
            training=False,backend_rebuild=False,Wmin=False,
            refine_max_seconds=args.refine_max_seconds,refine_hard_timeout=args.refine_hard_timeout,worker_timeout=args.worker_timeout),indent=2)); return
    install_signal_handlers()
    if args.stage=='report':
        with campaign_lock(args.out,'v8_016_report'):
            from explainable_evaluation import evaluate
            raw,panel,predictions=report_inputs(args.out)
            write(args.out/'explainable_report.json',evaluate(raw['rows'],panel['conditions'],predictions))
        return
    import torch
    panel=checked_panel(args); model=source_model()
    if args.stage=='check':
        binary,digest=checked_backend(args)
        print(json.dumps(dict(ready=True,family_count=panel['family_count'],binary=str(binary),
            binary_sha256=digest,model_sha256=model['sha256'],torch_version=str(torch.__version__)),indent=2)); return
    with campaign_lock(args.out,'v8_016_'+args.stage):
        controller=Controller(args); verify_panel(controller,panel); prepare(controller,panel,model)
        for stage in STAGES if args.stage=='all' else (args.stage,):
            print('V8_016_STAGE_START stage='+stage,flush=True)
            if stage=='prepare': pass
            elif stage=='predict': predict(controller,panel,model)
            elif stage=='compare': compare(controller,panel,model)
            else:
                from explainable_evaluation import evaluate
                locked_predictions(controller,panel,model)
                raw,frozen_panel,pred=report_inputs(args.out)
                write(args.out/'explainable_report.json',evaluate(raw['rows'],frozen_panel['conditions'],pred))
            print('V8_016_STAGE_COMPLETE stage='+stage,flush=True)
        print('V8_016_GROUP_COMPLETE group='+args.stage,flush=True)


if __name__=='__main__':
    main()
