"""Runtime-oriented initializer experiment; reuse v68 data, never overwrite it."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import statistics
import sys
import threading
import time

from bridge import (HERE, V68, REPO, emit, execute, fingerprint, freeze, read,
    recover, sha, write, legacy_experiment as e, legacy_timing as pt,
    check_free_space, log_storage_failure, StoragePressureError, parallel, prune_trial)
from phase_metrics import parse_phase_metrics
from metrics import aggregate_widths, compare_runtime, DEFAULT_BUDGETS


def bounded_contexts(contexts, split):
    """Two widths chosen only from old Default calibration, never GNN outcomes."""
    chosen=[]
    for design in sorted({c['design'] for c in contexts if c['split']==split}):
        group=sorted((c for c in contexts if c['design']==design),key=lambda c:c['width'])
        reference=group[0]['reference_width']
        near=next((c for c in group if c['width']>=reference),group[-1])
        for c in (near,group[-1]):
            if c not in chosen:
                chosen.append(c)
    return chosen


def prepare(a):
    a.source=a.source.resolve();a.out=a.out.resolve()
    a.out.relative_to(REPO/'experiments/vtr/tailcompile_v69/runs')
    if a.source==a.out:
        raise ValueError('v69 must have its own output directory')
    if len(a.seeds)!=5 or len(set(a.seeds))!=5:
        raise ValueError('Exactly FIVE distinct evaluation seeds are required')
    if not a.feedback_seeds or len(set(a.feedback_seeds))!=len(a.feedback_seeds):
        raise ValueError('Feedback seeds must be nonempty and unique')
    old=read(a.source/'protocol.json')
    forbidden=set(sum((old.get(k,[]) for k in
        ('calibration_seeds','teacher_seeds','feedback_seeds','evaluation_seeds')),[]))
    if set(a.seeds)&set(a.feedback_seeds) or (set(a.seeds)|set(a.feedback_seeds))&forbidden:
        raise ValueError('Use disjoint new feedback/evaluation seeds, not the old teacher/test seeds')
    contexts=read(a.source/'contexts.json')['contexts']
    keys=set()
    for c in contexts:
        if not re.fullmatch(r'[A-Za-z0-9_.-]+',c['design']) or c['design'] in ('.','..'):
            raise ValueError('Unsafe design name')
        key=c['design'],c['width']
        if key in keys:
            raise ValueError('Duplicate source context')
        keys.add(key)
    checkpoint=(a.initial_checkpoint or a.source/'models/finetune/generator.pt').resolve()
    protocol=dict(schema='tailcompile-v69-runtime-1',source=str(a.source),
        source_protocol_sha256=sha(a.source/'protocol.json'),
        source_contexts_sha256=sha(a.source/'contexts.json'),
        source_dataset_sha256=sha(a.source/'dataset.json'),identity=old['identity'],
        initial_checkpoint=str(checkpoint),initial_checkpoint_sha256=sha(checkpoint),
        options=dict(soft_density_passes=a.soft_density_passes,density_price_weight=a.density_price_weight,density_step=1.),
        budgets=dict(width_regression=a.width_budget,cpd_regression=a.cpd_budget,
            wirelength_regression=a.wire_budget,min_total_time_reduction=a.time_gain),
        seeds=a.seeds,feedback_seeds=a.feedback_seeds,candidates=a.candidates,
        timeout=a.timeout,methods=['default','fast'],
        budgets_status='Provisional configurable engineering tradeoff, frozen before outcomes',
        width_metric='Arithmetic mean of five per-seed minima on the frozen observed width grid',
        runtime_metric='Full worker wall minus reusable RRG encoding; includes Python/model/I/O/inference/decoding/VPR',
        timing_selection='Default calibration reference width and highest frozen width; no joint-success filtering',
        reuse_note='Same old final-test families were already inspected in v68: this is a follow-up, not a new independent holdout',
        code={p.name:sha(p) for p in (HERE/'run.py',HERE/'generate.py',HERE/'fast_model.py',
            HERE/'timed_worker.py',HERE/'metrics.py',HERE/'phase_metrics.py',HERE/'bridge.py',
            V68/'model.py',V68/'common.py',V68/'parallel_timing.py',V68/'storage.py',
            V68/'experiment.py',V68/'lifecycle.py',V68/'generation.py')})
    freeze(a.out/'protocol.json',protocol)
    freeze(a.out/'contexts.json',dict(contexts=contexts))
    emit('prepared',out=str(a.out),input_copies=0,feedback_conditions=len(bounded_contexts(contexts,'train')),
        evaluation_conditions=sum(c['split']=='final_test' for c in contexts),budgets=protocol['budgets'])
    return protocol,contexts


def load(a, vpr=False):
    p=read(a.out/'protocol.json');source=Path(p['source'])
    for filename,key in (('contexts.json','source_contexts_sha256'),('protocol.json','source_protocol_sha256')):
        if sha(source/filename)!=p[key]:
            raise ValueError('Frozen v68 source changed; do not mix cohorts')
    contexts=read(a.out/'contexts.json')['contexts']
    if contexts!=read(source/'contexts.json')['contexts']:
        raise ValueError('v69 context list changed')
    if vpr:
        checked={}
        for c in contexts:
            path=c['mapped_blif']
            if path not in checked:
                checked[path]=sha(path)
            if checked[path]!=c['mapped_blif_sha256']:
                raise ValueError(f'Canonical mapped input changed: {path}')
    for filename,digest in p['code'].items():
        path=HERE/filename if (HERE/filename).exists() else V68/filename
        if sha(path)!=digest:
            raise ValueError(f'Frozen implementation changed: {filename}; use a new v69 cohort')
    a.timeout=p['timeout']
    if vpr:
        e.identity(a)
        if a.identity!=p['identity']:
            raise ValueError('VPR binary/architecture/hook differs from reused v68 data')
    return p,contexts


def generate_all(a, contexts, checkpoint, root, options, candidates=1):
    from generate import signature, validate_generation, worker
    tasks=[];results={}
    for c in contexts:
        for k in range(candidates):
            folder=root/c['design']/f'w{c["width"]}'/f'c{k}'
            saved=validate_generation(folder,signature(c,checkpoint,options,k))
            if saved:
                results[c['design'],c['width'],k]=saved
            else:
                tasks.append((c,str(checkpoint),str(folder),options,k))
    emit('generation_start',reused=len(results),pending=len(tasks),jobs=a.generation_jobs)
    if not tasks:
        return results
    from generation import _terminate_pool
    pool=cf.ProcessPoolExecutor(max_workers=min(a.generation_jobs,len(tasks)),mp_context=mp.get_context('spawn'))
    old_handler=signal.getsignal(signal.SIGTERM)
    def stop(signum, frame):
        raise KeyboardInterrupt('Stopping own generation workers; completed candidates retained')
    signal.signal(signal.SIGTERM,stop)
    try:
        futures={pool.submit(worker,t):(t[0]['design'],t[0]['width'],t[4]) for t in tasks}
        pending=set(futures)
        while pending:
            done,pending=cf.wait(pending,timeout=5,return_when=cf.FIRST_COMPLETED)
            for f in done:
                results[futures[f]]=f.result()
            progress=dict(complete=not pending,completed=len(results),expected=len(contexts)*candidates,pending=len(pending))
            write(root/'generation_progress.json',progress);emit('generation_progress',**progress)
        pool.shutdown()
    except BaseException:
        _terminate_pool(pool)
        raise
    finally:
        signal.signal(signal.SIGTERM,old_handler)
    return results


def parse_row(result, directory, context, seed, method):
    text=(directory/'vpr.out').read_text(encoding='utf-8',errors='replace')
    if method=='fast' and result['status'] in ('success','unroutable') and (
            'TAILCOMPILE_V68_GP_INIT_CONSUMED' not in text or
            'TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1' not in text or
            'skipping the Global Placement' in text):
        raise RuntimeError(f'Native GP did not consume initialization: {directory}')
    def metric(pattern):
        values=re.findall(pattern,text)
        value=float(values[-1]) if values else None
        return value if value is None or math.isfinite(value) else None
    row=dict(result,design=context['design'],family=context['family'],split=context['split'],
        width=context['width'],seed=seed,method=method,
        cpd=metric(r'Final critical path delay \(least slack\):\s*([\d.eE+-]+)'),
        wirelength=metric(r'Total wirelength:\s*([\d.eE+-]+)'),phase_metrics=parse_phase_metrics(text))
    write(directory/'metrics.json',row)
    if log_storage_failure(text,result['returncode']):
        raise StoragePressureError(f'Disk quota/full at {directory}; stop dispatch and retain evidence')
    return row


def scan(a,p,contexts,checkpoint,generated):
    freeze(a.out/'evaluation_plan.json',dict(checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
        contexts=[dict(design=c['design'],width=c['width']) for c in contexts],seeds=p['seeds'],methods=p['methods']))
    def one(task):
        c,seed,method=task
        directory=a.out/'evaluation/runs'/c['design']/f'w{c["width"]}'/f's{seed}'/method
        g=generated[c['design'],c['width'],0]
        init=g['ap_init'] if method=='fast' else None
        check_free_space(a.out)
        result=execute(e.vpr_command(a,c['mapped_blif'],c['width'],seed,c['device_width']),directory,
            dict(a.identity,stage='route',protocol_sha256=fingerprint(p),design=c['design'],width=c['width'],
                seed=seed,method=method,blif_sha256=sha(c['mapped_blif']),initializer_sha256=sha(init) if init else None),
            env={'TC_GP_INIT':init} if init else None,timeout=a.timeout,heartbeat=a.heartbeat,
            retry_failed=a.retry_failed,retry_timeouts=a.retry_timeouts)
        row=parse_row(result,directory,c,seed,method)
        prune_trial(directory,a.out)
        return row
    tasks=[(c,s,m) for c in contexts for s in p['seeds'] for m in
           (p['methods'] if s%2 else p['methods'][::-1])]
    parallel(one,tasks,a.jobs)
    return scan_report(a)


def scan_report(a):
    p,contexts=load(a)
    plan=read(a.out/'evaluation_plan.json');rows=[]
    widths={}
    for c in plan['contexts']:
        widths.setdefault(c['design'],[]).append(c['width'])
        for seed in plan['seeds']:
            for method in p['methods']:
                path=a.out/'evaluation/runs'/c['design']/f'w{c["width"]}'/f's{seed}'/method/'metrics.json'
                if path.exists():
                    rows.append(read(path))
    result=aggregate_widths(rows,seeds=plan['seeds'],widths_by_design=widths,methods=p['methods'])
    write(a.out/'evaluation_report.json',result)
    emit('width_report',path=str(a.out/'evaluation_report.json'),recorded_trials=len(rows))
    return result


def make_waves(units,cores):
    waves=[]
    for start in range(0,len(units),len(cores)):
        lanes=[dict(u,lane=i,cpu=cores[i]['cpu'],order=['default','fast'] if (start+i)%2==0 else ['fast','default'])
               for i,u in enumerate(units[start:start+len(cores)])]
        waves.append(dict(index=len(waves),effective_lanes=len(lanes),lanes=lanes))
    return waves


def timed_trial(a,p,context,generation,checkpoint,lane,method,directory,barrier,protocol):
    command=[str(x) for x in e.vpr_command(a,context['mapped_blif'],context['width'],lane['seed'],context['device_width'])]
    spec=dict(cpu=lane['cpu'],context=context,method=method,checkpoint=str(checkpoint),
        candidate=lane['candidate'],options=p['options'],expected_ap_sha256=generation['ap_init_sha256'],vpr_command=command)
    freeze(directory/'spec.json',spec)
    check_free_space(a.out);barrier.wait(timeout=60)
    result=execute([sys.executable,HERE/'timed_worker.py','--spec',directory/'spec.json'],directory,
        dict(a.identity,stage='route',protocol_sha256=fingerprint(protocol),spec_sha256=fingerprint(spec)),
        timeout=a.timeout,heartbeat=a.heartbeat)
    row=parse_row(result,directory,context,lane['seed'],method)
    row['candidate']=lane['candidate'];row['timing_protocol']=fingerprint(protocol)
    row['cpu']=lane['cpu'];row['runtime_scope']=p['runtime_metric']
    cost=read(directory/'cost.json') if (directory/'cost.json').exists() else None
    affinity=read(directory/'affinity.json') if (directory/'affinity.json').exists() else None
    if affinity is None or affinity['allowed_cpus']!=[lane['cpu']] or set(affinity['thread_env'].values())!={'1'}:
        raise ValueError(f'Failed CPU/thread isolation: {directory}')
    row['cost']=cost
    if cost:
        # For Default its full subprocess wall is the comparison baseline.
        # For Fast charge all process setup, graph/model I/O, actual inference,
        # decode and lowering. Only separately measured reusable RRG encoding
        # is subtracted. No throughput amortization or cached logic discount.
        row['vpr_seconds']=cost['vpr_seconds']
        row['generation_seconds']=max(0.,result['wall_seconds']-cost['vpr_seconds']-cost['rrg_encoding_seconds'])
        row['setup_cost_semantics']='Default: worker setup only; Fast: setup + model/I/O/inference/decode/lowering, minus reusable RRG encoding'
        row['total_seconds_including_rrg_encoding']=result['wall_seconds']
    else:
        row['vpr_seconds']=None;row['generation_seconds']=None
    write(directory/'metrics.json',row)
    return row


def timed_pairs(a,p,contexts,checkpoint,generated,root,seeds,candidates=1):
    units=[dict(design=c['design'],width=c['width'],seed=s,candidate=k)
           for c in contexts for k in range(candidates) for s in seeds]
    hw=pt.hardware(a.timing_jobs,a.cpu_list,len(units))
    protocol=dict(schema='tailcompile-v69-controlled-pairs-1',hardware=hw,
        parent_protocol_sha256=fingerprint(p),checkpoint_sha256=sha(checkpoint),
        timeout=a.timeout,seeds=seeds,units=units,
        generation_sha256={f'{d}/w{w}/c{k}':fingerprint(g) for (d,w,k),g in generated.items()},
        scope=p['runtime_metric'],policy='same physical core, AB/BA, half-wave barriers; entire partial waves rerun')
    freeze(root/'protocol.json',protocol)
    waves=make_waves(units,hw['selected_cores']);freeze(root/'schedule.json',dict(waves=waves))
    cases={(c['design'],c['width']):c for c in contexts};all_rows=[]
    for wave in waves:
        old=pt.completed_wave(root,wave)
        if old and not pt.retry_wave(a,old):
            all_rows.extend(old['results'].values());emit('reuse_wave',wave=wave['index']);continue
        pt.archive_wave(root,wave,a.confirm_remote_stopped)
        folder=pt.wave_folder(root,wave);results={};evidence={};before=pt.snapshot();begin=time.monotonic()
        for half in (0,1):
            barrier=threading.Barrier(len(wave['lanes']));errors=[]
            emit('timing_half',wave=wave['index'],waves=len(waves),half=half,lanes=len(wave['lanes']))
            def one(lane):
                directory=folder/f'half{half}'/f'lane{lane["lane"]:02d}'
                c=cases[lane['design'],lane['width']]
                try:
                    row=timed_trial(a,p,c,generated[c['design'],c['width'],lane['candidate']],checkpoint,
                        lane,lane['order'][half],directory,barrier,protocol)
                    return lane,row,directory
                except BaseException:
                    barrier.abort();raise
            with cf.ThreadPoolExecutor(max_workers=len(wave['lanes'])) as pool:
                for future in cf.as_completed([pool.submit(one,l) for l in wave['lanes']]):
                    try:
                        lane,row,directory=future.result();results[f'{half}:{lane["lane"]}']=row
                        for name in ('trial.json','metrics.json','affinity.json','spec.json'):
                            evidence[str((directory/name).relative_to(folder))]=sha(directory/name)
                        if (directory/'cost.json').exists():
                            evidence[str((directory/'cost.json').relative_to(folder))]=sha(directory/'cost.json')
                    except Exception as ex:
                        errors.append(str(ex));emit('timing_error',error=str(ex))
            for lane in wave['lanes']:
                directory=folder/f'half{half}'/f'lane{lane["lane"]:02d}'
                if (directory/'metrics.json').exists():
                    prune_trial(directory,a.out)  # All measured lanes drained first.
            if errors:
                raise RuntimeError(f'Incomplete wave retained, recover/resume as a unit: {errors[0]}')
        record=dict(complete=True,wave=wave,results=results,evidence_sha256=evidence,
            wall_seconds=time.monotonic()-begin,load_before=before,load_after=pt.snapshot())
        write(folder/'wave.json',record);all_rows.extend(results.values())
        emit('wave_complete',wave=wave['index'],waves=len(waves),trials=len(results))
    write(root/'rows.json',dict(complete=True,rows=all_rows,protocol_sha256=fingerprint(protocol)))
    return all_rows


def feedback(a):
    p,contexts=load(a,vpr=True);contexts=bounded_contexts(contexts,'train')
    checkpoint=Path(p['initial_checkpoint'])
    if sha(checkpoint)!=p['initial_checkpoint_sha256']:
        raise ValueError('Feedback checkpoint changed')
    root=a.out/'feedback'
    generated=generate_all(a,contexts,checkpoint,root/'generation',p['options'],p['candidates'])
    rows=timed_pairs(a,p,contexts,checkpoint,generated,root/'runs/timing'/a.run_name,p['feedback_seeds'],p['candidates'])
    selected=[];diagnostics=[];budgets=p['budgets']
    for c in contexts:
        choices=[]
        for k in range(p['candidates']):
            group=[r for r in rows if (r['design'],r['width'],r['candidate'])==(c['design'],c['width'],k)]
            index={(r['seed'],r['method']):r for r in group};reason='accepted';ratios={key:[] for key in ('total','cpd','wirelength')}
            for s in p['feedback_seeds']:
                b,g=index.get((s,'default')),index.get((s,'fast'))
                if not b or not g or b['status']!='success' or g['status']!='success':
                    reason='requires_paired_success_for_runtime_teacher';break
                values=[b.get('vpr_seconds'),g.get('vpr_seconds'),g.get('generation_seconds'),b.get('cpd'),g.get('cpd'),b.get('wirelength'),g.get('wirelength'),b.get('generation_seconds')]
                if any(v is None or not math.isfinite(v) for v in values) or min(values[:2]+values[3:7])<=0:
                    reason='missing_measured_cost_or_qor';break
                ratios['total'].append((g['vpr_seconds']+g['generation_seconds'])/(b['vpr_seconds']+b['generation_seconds']))
                for key in ('cpd','wirelength'):ratios[key].append(g[key]/b[key])
            means={key:math.exp(statistics.mean(map(math.log,v))) if v else None for key,v in ratios.items()}
            if reason=='accepted' and (max(ratios['cpd'])>1+budgets['cpd_regression'] or max(ratios['wirelength'])>1+budgets['wirelength_regression']):reason='qor_budget'
            if reason=='accepted' and means['total']>1-budgets['min_total_time_reduction']:reason='no_end_to_end_time_gain'
            quality=dict(acceptable=reason=='accepted',reason=reason,geomean_ratios=means,
                measurement='controlled same-core runtime including fresh initialization cost',
                width_guard='Not measurable per condition; five-seed mean-width guard is applied in evaluation')
            diagnostics.append(dict(design=c['design'],width=c['width'],candidate=k,**quality))
            if quality['acceptable']:choices.append((means['total'],k,quality))
        if choices:
            _,k,quality=min(choices,key=lambda x:(x[0],x[1]));g=generated[c['design'],c['width'],k]
            selected.append(dict(design=c['design'],family=c['family'],split='train',width=c['width'],
                candidate=k,target=g['target'],target_sha256=g['target_sha256'],quality=quality))
    report=dict(schema='tailcompile-v69-runtime-feedback-1',complete=True,selected=selected,diagnostics=diagnostics,
        all_route_outcomes_known=all(r['status'] in ('success','unroutable') for r in rows),
        generator_checkpoint_sha256=sha(checkpoint),final_test_examples=0,paired_trials=len(rows),
        timing_run=a.run_name,budgets=budgets,
        scope='Train-only hindsight distillation targets; not a deployed oracle or final-test selection')
    write(root/'feedback_dataset.json',report);emit('feedback_complete',selected=len(selected),trials=len(rows))


def evaluate(a):
    p,contexts=load(a,vpr=True);contexts=[c for c in contexts if c['split']=='final_test']
    checkpoint=(a.checkpoint or a.out/'models/runtime/generator.pt').resolve()
    if not checkpoint.is_file():
        raise ValueError('Run runtime finetune first, or explicitly --checkpoint for a frozen-checkpoint fast-decoder diagnostic')
    generated=generate_all(a,contexts,checkpoint,a.out/'evaluation/generation',p['options'])
    scan(a,p,contexts,checkpoint,generated)


def timing(a):
    p,contexts=load(a,vpr=True);contexts=bounded_contexts(contexts,'final_test')
    plan=read(a.out/'evaluation_plan.json');checkpoint=Path(plan['checkpoint'])
    if sha(checkpoint)!=plan['checkpoint_sha256']:
        raise ValueError('Evaluation checkpoint changed')
    generated=generate_all(a,contexts,checkpoint,a.out/'evaluation/generation',p['options'])
    root=a.out/'timing'/a.run_name
    rows=timed_pairs(a,p,contexts,checkpoint,generated,root,p['seeds'])
    width_report=scan_report(a)
    expected=[dict(design=c['design'],width=c['width'],seed=s) for c in contexts for s in p['seeds']]
    report=compare_runtime(rows,expected_conditions=expected,budgets=p['budgets'],width_report=width_report)
    report.update(schema='tailcompile-v69-runtime-report-1',timing_run=a.run_name,
        runtime_scope=p['runtime_metric'],protocol=read(root/'protocol.json'),
        final_test_caveat=p['reuse_note'],raw_rows=rows,serial_and_scan_timings_included=False,
        reusable_rrg_preprocessing_excluded=True)
    write(root/'runtime_report.json',report)
    emit('runtime_report',path=str(root/'runtime_report.json'))


def status(a):
    result={}
    for key,name in [('prepared','protocol.json'),('feedback','feedback/feedback_dataset.json'),
                     ('finetune','models/runtime/training_report.json'),('evaluate','evaluation_report.json'),
                     ('timing',f'timing/{a.run_name}/runtime_report.json')]:
        path=a.out/name
        if path.exists():
            value=read(path);result[key]=dict(exists=True,complete=value.get('complete'),path=str(path))
            if key=='prepared':result[key]['complete']=True
            if key=='feedback':
                result[key]['selected_targets']=len(value.get('selected',[]))
                result[key]['all_route_outcomes_known']=value.get('all_route_outcomes_known',False)
            if key=='timing':
                result[key]['evidence_complete']=value.get('evidence_complete',False)
                result[key]['passed']=value.get('passed',False)
        else:result[key]=dict(exists=False)
    for phase,root in [('feedback',a.out/'feedback/runs/timing'/a.run_name),('timing',a.out/'timing'/a.run_name)]:
        if (root/'schedule.json').exists():
            waves=read(root/'schedule.json')['waves']
            result[phase]['waves_expected']=len(waves)
            result[phase]['waves_complete']=sum(pt.completed_wave(root,w) is not None for w in waves)
    emit('status',**result)


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','feedback','evaluate','timing','report','status','recover'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--initial-checkpoint',type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--vtr-root',dest='vtr',type=Path,default=os.environ.get('VTR_ROOT'))
    p.add_argument('--jobs',type=int,default=16);p.add_argument('--generation-jobs',type=int,default=16)
    p.add_argument('--timing-jobs',type=int,default=16);p.add_argument('--cpu-list')
    p.add_argument('--run-name',default='paired_001');p.add_argument('--timeout',type=float,default=7200)
    p.add_argument('--heartbeat',type=float,default=60)
    p.add_argument('--seeds',type=int,nargs='+',default=[371,372,373,374,375])
    p.add_argument('--feedback-seeds',type=int,nargs='+',default=[351,352])
    p.add_argument('--candidates',type=int,default=4)
    p.add_argument('--soft-density-passes',type=int,default=2);p.add_argument('--density-price-weight',type=float,default=1.)
    p.add_argument('--width-budget',type=float,default=.03);p.add_argument('--wire-budget',type=float,default=.03)
    p.add_argument('--cpd-budget',type=float,default=.03);p.add_argument('--time-gain',type=float,default=.05)
    p.add_argument('--retry-failed',action='store_true');p.add_argument('--retry-timeouts',action='store_true')
    p.add_argument('--confirm-remote-stopped',action='store_true')
    a=p.parse_args(argv);a.out=a.out.resolve()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_name):p.error('Unsafe run name')
    if min(a.jobs,a.generation_jobs,a.timing_jobs,a.candidates,a.timeout,a.heartbeat)<=0:p.error('Budgets must be positive')
    if not 0<=a.soft_density_passes<=8 or not math.isfinite(a.density_price_weight) or a.density_price_weight<0:p.error('Invalid soft density options (passes must be 0..8)')
    if not all(math.isfinite(v) and 0<=v<1 for v in (a.width_budget,a.wire_budget,a.cpd_budget,a.time_gain)):p.error('Tradeoff budgets must be finite fractions in [0,1)')
    if a.phase=='prepare' and not a.source:p.error('prepare requires --source')
    return a


if __name__=='__main__':
    a=parse_args()
    if a.phase=='recover':
        pt.assert_workers_stopped(a.out,a.confirm_remote_stopped)
        recover(a.out,a.confirm_remote_stopped)
    else:{'prepare':prepare,'feedback':feedback,'evaluate':evaluate,'timing':timing,'report':scan_report,'status':status}[a.phase](a)
