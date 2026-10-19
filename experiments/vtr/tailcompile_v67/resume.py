"""Resumable ABCD orchestration; frozen placement algorithms stay unchanged.

Old trial.json files are preserved. Their timeout classification is recorded in
outcome.json. New retries move the entire previous attempt into _attempts/.
The original algorithm/binary/input identity is still strictly checked.
"""
import argparse
from collections import Counter
import concurrent.futures as cf
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import threading
import time
import uuid

import four_routes as f

HERE=Path(__file__).resolve().parent
TERMINAL={'success','unroutable','timeout'}
SKIPPED={'not_run_after_scan_timeout','not_run_after_scan_error'}
STOP=threading.Event()


def emit(stage,**details):
    print(json.dumps(dict(stage=stage,**details)),flush=True)


def normalized(row):
    result=dict(row)
    if row.get('process_returncode')==124:
        result['original_status']=row['status'];result['status']='timeout'
    return result


def signature(a,identity,c,seed,method,timeout=None):
    return dict(identity=identity,sources=c['sources'],seed=seed,width=c['width'],
        device_width=c['device_width'],method=method,timeout=a.timeout if timeout is None else timeout,
        fplace=c['fplace_sha256'] if method in ('B_gnn','D_gnn_refine') else None)


def validate_old(a,identity,c,seed,method,row):
    old=row['signature']
    if old!=signature(a,identity,c,seed,method,old['timeout']):
        raise ValueError('algorithm, binary, architecture, netlist or candidate changed; use a new cohort')
    if old['timeout']>a.timeout:
        raise ValueError('cannot reuse a larger-budget trial under a smaller budget')


def process_alive(folder):
    path=folder/'process.json'
    if not path.exists():return False
    proc=f.read(path);p=Path('/proc')/str(proc['pid'])/'stat'
    try:return p.read_text().rsplit(')',1)[1].split()[19]==proc['start_ticks']
    except (OSError,KeyError,IndexError):return False


def archive(a,folder):
    relative=folder.resolve().relative_to(a.out.resolve())
    if relative.parts[0] not in ('runs','timing') or folder.is_symlink():
        raise ValueError('refuse to archive outside trial directories')
    if process_alive(folder):raise RuntimeError(f'prior process is still alive: {folder}')
    target=a.out/'_attempts'/relative/(time.strftime('%Y%m%d_%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    target.parent.mkdir(parents=True,exist_ok=True)
    folder.rename(target)
    emit('attempt_archived',source=str(folder),destination=str(target))


def outcome(folder,row,**metadata):
    value=dict(row,**metadata)
    if (folder/'trial.json').exists():value['trial_sha256']=f.sha(folder/'trial.json')
    f.write(folder/'outcome.json',value)
    return value


def stage_for(line,current):
    if 'TAILCOMPILE_WORKER_STAGE ' in line:
        try:return 'worker:'+json.loads(line.split('TAILCOMPILE_WORKER_STAGE ',1)[1])['stage']
        except (ValueError,KeyError):pass
    for marker,stage in [('TAILCOMPILE_GP_REPLACED','refinement_applied'),
        ('TAILCOMPILE_GP_EXPORTED','coarse_snapshot_exported'),('## AP Global Placer','global_placement'),
        ('# Analytical Placement','analytical_placement'),('Begin packing','packing'),
        ('Routing circuit','routing'),('VPR succeeded','vpr_complete')]:
        if marker in line:return stage
    return current


def run_process(a,backend,identity,c,seed,method,folder):
    folder.mkdir(parents=True,exist_ok=True)
    sig=signature(a,identity,c,seed,method)
    f.write(folder/'started.json',sig)
    shutil.copy2(c['packed'],folder/'input.net')
    cmd=f.command(backend,c,seed,method,folder)
    env=f.environment(folder,method,c['width']);env['PYTHONUNBUFFERED']='1'
    if method in f.METHODS[2:]:
        env['TC_GP_WORKER']=shlex.join([os.sys.executable,'-u',str(HERE/'observe_refine.py'),
            '--prefix',str(folder/'coarse'),'--output',str(folder/'refined.fplace'),'--width',str(c['width'])])
    # libc line buffering flushes existing C++ hook/GP messages without rebuilding VPR.
    stdbuf=shutil.which('stdbuf')
    launched=[stdbuf,'-oL','-eL',*cmd] if stdbuf else cmd
    if getattr(a,'pinned_cpu',None) is not None:
        launched=[os.sys.executable,str(HERE/'affinity_exec.py'),'--cpu',str(a.pinned_cpu),
                  '--audit',str(folder/'launch_affinity.json'),'--',*launched]
    if getattr(a,'launch_barrier',None) is not None:a.launch_barrier.wait(timeout=120)
    started_unix=time.time()
    start=time.monotonic();current='launch';offset=0;pending='';last_heartbeat=start;history=[]
    def event(stage,**details):
        item=dict(stage=stage,elapsed_seconds=time.monotonic()-start,**details)
        history.append(item)
        with (folder/'stages.jsonl').open('a') as events:
            events.write(json.dumps(item)+'\n');events.flush()
        emit('trial_stage',method=method,design=c['design'],width=c['width'],seed=seed,progress=item)
    event(current,line_buffering=bool(stdbuf))
    with (folder/'vpr.out').open('w') as log:
        proc=subprocess.Popen(launched,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        try:
            ticks=(Path('/proc')/str(proc.pid)/'stat').read_text().rsplit(')',1)[1].split()[19]
            f.write(folder/'process.json',dict(pid=proc.pid,start_ticks=ticks))
        except OSError:pass
        reason=None
        try:
            with (folder/'vpr.out').open(errors='replace') as reader:
                while True:
                    reader.seek(offset);chunk=reader.read();offset=reader.tell()
                    lines=(pending+chunk).split('\n');pending=lines.pop()
                    for line in lines:
                        stage=stage_for(line,current)
                        if stage!=current:current=stage;event(current)
                    code=proc.poll()
                    if code is not None:break
                    now=time.monotonic()
                    if STOP.is_set() or now-start>=a.timeout:
                        reason='interrupted' if STOP.is_set() else 'timeout'
                        try:os.killpg(proc.pid,signal.SIGKILL)
                        except ProcessLookupError:pass
                        proc.wait();code=130 if reason=='interrupted' else 124;break
                    if now-last_heartbeat>=a.heartbeat:
                        event('heartbeat',last_observed_stage=current);last_heartbeat=now
                    time.sleep(.25)
        except BaseException:
            try:os.killpg(proc.pid,signal.SIGKILL)
            except ProcessLookupError:pass
            proc.wait();raise
    elapsed=time.monotonic()-start
    text=(folder/'vpr.out').read_text(errors='replace')
    # Read the final flushed tail as well, so completed fast workers retain their stage.
    for line in text.splitlines():current=stage_for(line,current)
    status=reason or ('timeout' if code==124 else f.classify_vpr_result(code,text))
    selection=f.read(folder/'refined.selection.json') if (folder/'refined.selection.json').exists() else None
    if status not in ('timeout','interrupted') and method in f.METHODS[2:] and (
            'TAILCOMPILE_GP_REPLACED' not in text or selection is None):status='other_failure'
    def metric(pattern):
        values=re.findall(pattern,text);return float(values[-1]) if values else None
    result=dict(signature=sig,design=c['design'],width=c['width'],seed=seed,method=method,
        started_unix=started_unix,finished_unix=time.time(),
        status=status,process_returncode=code,wall_seconds=elapsed,command=cmd,launched_command=launched,
        phases=f.parse_vpr_log(text),local_refinement=selection,refinement_already_in_wall_time=True,
        last_observed_stage=current,stage_history=history,
        orchestration=dict(resume_sha256=f.sha(Path(__file__)),observer_sha256=f.sha(HERE/'observe_refine.py')),
        wirelength=metric(r'Total wirelength:\s*([\d.eE+-]+)'),
        cpd=metric(r'Final critical path delay \(least slack\):\s*([\d.eE+-]+)'),
        hook_env={k:v for k,v in env.items() if k.startswith('TC_GP_')})
    if getattr(a,'pinned_cpu',None) is not None:
        audit=folder/'launch_affinity.json'
        result['cpu_affinity']=f.read(audit) if audit.exists() else None
        if result['cpu_affinity'] is None or result['cpu_affinity']['allowed_cpus']!=[a.pinned_cpu]:
            result['status']='other_failure'
        events=folder/'worker.events.jsonl'
        starts=[json.loads(line) for line in events.read_text().splitlines() if json.loads(line).get('stage')=='worker_start'] if events.exists() else []
        if method in f.METHODS[2:] and result['status'] in ('success','unroutable'):
            result['worker_cpu_affinity']=starts[0].get('allowed_cpus') if starts else None
            if result['worker_cpu_affinity']!=[a.pinned_cpu]:result['status']='other_failure'
    f.write(folder/'trial.json',result)
    event('trial_finished',status=result['status'],last_observed_stage=current)
    return outcome(folder,result,reused=False)


def run_trial(a,backend,identity,task,timing=False):
    c,seed,method,folder=task;trial=folder/'trial.json'
    if trial.exists():
        raw=f.read(trial);validate_old(a,identity,c,seed,method,raw);row=normalized(raw)
        retry=row['status']=='interrupted' or (row['status']=='timeout' and a.retry_timeouts) or (
               row['status'] not in TERMINAL and a.retry_failed)
        if not retry:
            emit('reused',method=method,design=c['design'],width=c['width'],seed=seed,status=row['status'])
            return outcome(folder,row,reused=True)
        archive(a,folder)
    elif folder.exists() and any(folder.iterdir()):
        # A skipped timing condition has no trial to archive or execute by default.
        skip=folder/'outcome.json'
        if skip.exists() and f.read(skip).get('status') in SKIPPED:
            pass
        else:
            started=folder/'started.json'
            if started.exists():
                validate_old(a,identity,c,seed,method,dict(signature=f.read(started)))
            archive(a,folder)
    if timing and not a.time_scan_timeouts:
        scan=a.out/'runs'/c['design']/f'w{c["width"]}'/f's{seed}'/method/'trial.json'
        if scan.exists():
            raw=f.read(scan);validate_old(a,identity,c,seed,method,raw);row=normalized(raw)
            if row['status'] not in ('success','unroutable'):
                folder.mkdir(parents=True,exist_ok=True)
                return outcome(folder,dict(design=c['design'],width=c['width'],seed=seed,method=method,
                    status='not_run_after_scan_timeout' if row['status']=='timeout' else 'not_run_after_scan_error',
                    wall_seconds=None,wirelength=None,cpd=None,source_trial=str(scan),source_sha256=f.sha(scan),
                    source_trial_relative=scan.relative_to(a.out).as_posix(),
                    scan_status=row['status'],scan_wall_seconds=row['wall_seconds']),reused=False)
    if STOP.is_set():raise RuntimeError('interrupted before launch; resume later')
    return run_process(a,backend,identity,c,seed,method,folder)


def execute(a,timing=False):
    if os.name!='posix':raise RuntimeError('VPR execution requires Linux/WSL')
    plan=f.read(a.out/'plan.json')
    if plan['signature']['code']!=f.direct.implementation():raise ValueError('generator changed; use a new cohort')
    backend=f.direct.DirectBackend(os.environ['VTR_ROOT']);hook=f.verify(Path(backend.root))
    identity=dict(binary=backend.binary_hash,arch=backend.arch_hash,hook=hook,code=f.code_identity(),threads=1)
    path=a.out/'abcd_protocol.json'
    protocol=f.read(path) if path.exists() else dict(identity=identity,timeout=a.timeout,plan_sha256=f.sha(a.out/'plan.json'))
    if protocol['identity']!=identity or protocol['plan_sha256']!=f.sha(a.out/'plan.json'):
        raise ValueError('frozen algorithm/binary/inputs changed; cannot resume this cohort')
    if a.timeout<protocol['timeout']:raise ValueError('cannot lower the frozen timeout budget')
    f.freeze(path,protocol)
    for c in plan['cases']:
        for path,digest in [(c['blif'],c['sources']['mapped_blif']),(c['packed'],c['sources']['packed_net']),
                            (c['fplace'],c['fplace_sha256'])]:
            if f.sha(path)!=digest:raise ValueError(f'frozen input changed: {path}')
    session=a.out/'resume_sessions'/(time.strftime('%Y%m%d_%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8]+'.json')
    f.write(session,dict(phase='timing' if timing else 'run',runner_sha256=f.sha(Path(__file__)),
        observer_sha256=f.sha(HERE/'observe_refine.py'),timeout=a.timeout,jobs=1 if timing else a.jobs,
        retry_timeouts=a.retry_timeouts,retry_failed=a.retry_failed,time_scan_timeouts=a.time_scan_timeouts,
        instrumentation='line-buffered VPR and worker stage events; unchanged placement algorithm'))
    selection=timing_plan(a,plan) if timing else None
    selected={tuple(row) for row in selection['selected_tasks']} if selection else None
    pending=[t for t in f.tasks(a,plan,timing) if not timing or task_key(t) in selected]
    counts=Counter();errors=[]
    emit('phase_plan',phase='timing' if timing else 'run',selected_task_count=len(pending),
         execution_jobs=1 if timing else a.jobs)
    # Validate the first C/D paths. A timeout is recorded and does not abort.
    preflight=[]
    for method in f.METHODS[2:]:
        task=next((t for t in pending if t[2]==method),None)
        if task is None:continue
        result=run_trial(a,backend,identity,task,timing);counts[result['status']]+=1;preflight.append(task[3])
        if result['status']=='other_failure':raise RuntimeError(f'preflight infrastructure failure: {task[3]}/vpr.out')
    with cf.ThreadPoolExecutor(max_workers=1 if timing else a.jobs) as executor:
        futures={executor.submit(run_trial,a,backend,identity,t,timing):t for t in pending if t[3] not in preflight}
        for future in cf.as_completed(futures):
            try:counts[future.result()['status']]+=1
            except Exception as error:
                errors.append(str(error));emit('task_error',error=str(error))
    f.write(session,dict(f.read(session),status_counts=dict(counts),errors=errors))
    emit('phase_finished',phase='timing' if timing else 'run',status_counts=dict(counts),errors=len(errors))
    if errors:raise RuntimeError(f'{len(errors)} validation/interruption errors; retained completed results: {errors[0]}')


def observation(folder):
    trial=folder/'trial.json'
    if trial.exists():return normalized(f.read(trial))
    path=folder/'outcome.json'
    if path.exists():
        row=f.read(path)
        if row.get('source_trial'):
            source=Path(row['source_trial'])
            if row.get('source_trial_relative'):
                source=folder.parents[4]/row['source_trial_relative']
            if not source.is_file() or f.sha(source)!=row['source_sha256']:
                return dict(status='missing',wall_seconds=None)
        return row
    return dict(status='missing',wall_seconds=None)


def width_bounds(grid,rows):
    statuses={r['width']:r['status'] for r in rows}
    unknown=[w for w in grid if statuses[w] not in ('success','unroutable')]
    success=[w for w in grid if statuses[w]=='success']
    observed=min(success,default=None)
    if not unknown:return dict(f.width_result(grid,rows),observed_min_success_width=observed,unknown_widths=[])
    unresolved=observed is None or any(w<observed for w in unknown)
    return dict(w_min_grid=None if unresolved else observed,
        observed_min_success_width=observed,unknown_widths=unknown,
        possible_min_on_grid=min(unknown+success),upper_bound_on_grid=observed,
        censor='unresolved_timeout_or_missing' if unresolved else ('left' if observed==min(grid) else 'none'),
        monotonicity_violations=sum(statuses[w]=='unroutable' for w in grid if observed is not None and w>observed))


def counts(a,plan):
    result={}
    for phase in ('runs','timing'):
        result[phase]=dict(Counter(observation(t[3])['status'] for t in f.tasks(a,plan,phase=='timing')))
    return result


def task_key(task):
    c,seed,method,_=task
    return c['design'],c['width'],seed,method


def scan_report(a,write=True):
    """Summarize the complete width experiment without starting timing/VPR."""
    plan=f.read(a.out/'plan.json');widths=[];comparisons=[];status=Counter()
    for design in sorted({c['design'] for c in plan['cases']}):
        cases=sorted([c for c in plan['cases'] if c['design']==design],key=lambda c:c['width'])
        grid=[c['width'] for c in cases];by_method={}
        for method in f.METHODS:
            per_seed={};rates={}
            for seed in plan['seeds']:
                rows=[dict(observation(a.out/'runs'/design/f'w{c["width"]}'/f's{seed}'/method),width=c['width']) for c in cases]
                status.update(row['status'] for row in rows)
                per_seed[str(seed)]=width_bounds(grid,rows)
            for c in cases:
                observed=Counter(observation(a.out/'runs'/design/f'w{c["width"]}'/f's{s}'/method)['status'] for s in plan['seeds'])
                n=len(plan['seeds']);unknown=n-observed['success']-observed['unroutable']
                rates[c['width']]=dict(success_lower=observed['success']/n,
                    success_upper=(observed['success']+unknown)/n,status_counts=dict(observed))
            observed=min((w for w,r in rates.items() if r['success_lower']>=.8),default=None)
            possible=min((w for w,r in rates.items() if r['success_upper']>=.8),default=None)
            item=dict(design=design,method=method,per_seed=per_seed,success_rate_bounds=rates,
                w80=observed if observed==possible else None,observed_w80=observed,possible_w80=possible)
            widths.append(item);by_method[method]=item
        for candidate,baseline in f.PAIRS:
            pairs=[]
            for seed in plan['seeds']:
                b=by_method[baseline]['per_seed'][str(seed)];c=by_method[candidate]['per_seed'][str(seed)]
                wb,wc=b['w_min_grid'],c['w_min_grid']
                delta=wc-wb if wb is not None and wc is not None else None
                unresolved=any(x['censor']=='unresolved_timeout_or_missing' for x in (b,c))
                outcome='unresolved' if unresolved else 'censored' if delta is None else (
                    'improved_on_grid' if delta<0 else 'regressed_on_grid' if delta>0 else 'equal_on_grid')
                pairs.append(dict(seed=seed,baseline=wb,candidate=wc,delta_grid=delta,outcome=outcome,
                    exact_within_grid=b['censor']==c['censor']=='none',
                    baseline_censor=b['censor'],candidate_censor=c['censor']))
            comparisons.append(dict(design=design,candidate=candidate,baseline=baseline,pairs=pairs,
                outcomes=dict(Counter(p['outcome'] for p in pairs))))
    result=dict(scan_complete=not(set(status)-TERMINAL),status_counts=dict(status),
        all_route_outcomes_known=not(set(status)-{'success','unroutable'}),
        width_results=widths,wmin_comparisons=comparisons,
        timing_required=False,note='Timeouts are unknown; censored grid minima are not exact physical Wmin.')
    if write:
        f.write(a.out/'scan_report.json',result)
        emit('scan_report_written',scan_complete=result['scan_complete'],status_counts=dict(status))
    return result


def timing_plan(a,plan=None):
    """Freeze a small timing subset using routing outcomes only, never runtime."""
    plan=plan or f.read(a.out/'plan.json')
    seed_count=getattr(a,'timing_seed_count',3);width_count=getattr(a,'timing_width_count',1)
    seeds=plan['seeds'][:seed_count]
    scan={};evidence=[]
    for task in f.tasks(a,plan,False):
        key=task_key(task);folder=task[3];row=observation(folder);scan[key]=row['status']
        path=folder/'trial.json'
        if not path.exists():path=folder/'outcome.json'
        evidence.append([list(key),f.sha(path) if path.exists() else None,row['status']])
    if any(s not in TERMINAL for s in scan.values()):
        raise ValueError('Timing selection requires completed scan records; generate scan-report to inspect missing/error cases')
    import hashlib
    signature=dict(plan_sha256=f.sha(a.out/'plan.json'),scan_sha256=hashlib.sha256(
        json.dumps(evidence,sort_keys=True).encode()).hexdigest(),seed_count=seed_count,width_count=width_count,
        policy='first fixed seeds; lowest joint-success widths per design/comparison/seed; runtime is never read')
    path=a.out/'timing_selection.json'
    if path.exists():
        old=f.read(path)
        if old['signature']==signature:return old
        if not getattr(a,'replan_timing',False):
            raise ValueError('scan/timing scope changed; set TAILCOMPILE_V67_REPLAN_TIMING=1 to archive and rebuild selection')
        history=a.out/'timing_selection_history'/(time.strftime('%Y%m%d_%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8]+'.json')
        f.write(history,old)
    chosen=set();comparisons=[]
    for design in sorted({c['design'] for c in plan['cases']}):
        widths=sorted(c['width'] for c in plan['cases'] if c['design']==design)
        for candidate,baseline in f.PAIRS:
            pairs=[];unavailable=[]
            for seed in seeds:
                eligible=[w for w in widths if scan[(design,w,seed,candidate)]==scan[(design,w,seed,baseline)]=='success']
                if not eligible:unavailable.append(seed)
                for w in eligible[:width_count]:
                    pairs.append(dict(width=w,seed=seed))
                    chosen.update([(design,w,seed,candidate),(design,w,seed,baseline)])
            comparisons.append(dict(design=design,candidate=candidate,baseline=baseline,pairs=pairs,
                no_joint_success_seeds=unavailable))
    result=dict(signature=signature,fixed_seeds=seeds,comparisons=comparisons,
        selected_tasks=[list(k) for k in sorted(chosen)],selected_task_count=len(chosen),
        maximum_tasks_before_dedup=len({c['design'] for c in plan['cases']})*len(f.PAIRS)*len(seeds)*width_count*2,
        caveat='Runtime conclusions apply only to selected joint-success boundary conditions, not all widths/designs.')
    f.write(path,result)
    emit('timing_selection_written',selected_task_count=len(chosen),fixed_seeds=seeds)
    return result


def timing_progress(a,plan,selection):
    selected={tuple(row) for row in selection['selected_tasks']}
    status=Counter();outside=Counter()
    for task in f.tasks(a,plan,True):
        observed=observation(task[3])['status']
        (status if task_key(task) in selected else outside).update([observed])
    return dict(expected=len(selected),status_counts=dict(status),
        accounted=sum(n for s,n in status.items() if s in TERMINAL),
        remaining=sum(n for s,n in status.items() if s not in TERMINAL),
        existing_records_outside_scope=sum(n for s,n in outside.items() if s!='missing'))


def request_stop(a):
    """Gracefully stop only resume.py controllers for this exact output cohort."""
    if os.name!='posix':raise RuntimeError('stop requires Linux/WSL')
    matched=[]
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit() or int(entry.name)==os.getpid():continue
        try:
            args=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
            if '--out' not in args:continue
            if Path(args[args.index('--out')+1]).resolve()!=a.out.resolve():continue
            scripts=[i for i,arg in enumerate(args) if any(arg.replace('\\','/').endswith('/tailcompile_v67/'+name)
                for name in ('resume.py','parallel_runtime.py'))]
            if not scripts or args[scripts[0]+1] not in ('run','timing'):continue
            os.kill(int(entry.name),signal.SIGTERM);matched.append(int(entry.name))
        except (OSError,ValueError,IndexError):continue
    deadline=time.monotonic()+30
    while any((Path('/proc')/str(pid)).exists() for pid in matched) and time.monotonic()<deadline:
        time.sleep(.25)
    emit('stop_requested',controller_pids=matched,out=str(a.out),
         still_active=[pid for pid in matched if (Path('/proc')/str(pid)).exists()],
         note='SIGTERM closes active trials; completed records retained. Restart after output lock is released.')


def report(a):
    plan=f.read(a.out/'plan.json');widths=[];comparisons=[];summaries=[]
    scan=scan_report(a);selection=timing_plan(a,plan)
    def load(phase,c,s,m):
        return dict(observation(a.out/phase/c['design']/f'w{c["width"]}'/f's{s}'/m),width=c['width'])
    for design in sorted({c['design'] for c in plan['cases']}):
        cases=sorted([c for c in plan['cases'] if c['design']==design],key=lambda c:c['width'])
        grid=[c['width'] for c in cases]
        for method in f.METHODS:
            per_seed={str(s):width_bounds(grid,[load('runs',c,s,method) for c in cases]) for s in plan['seeds']}
            rates={}
            for c in cases:
                status=Counter(load('runs',c,s,method)['status'] for s in plan['seeds']);n=len(plan['seeds'])
                unknown=n-status['success']-status['unroutable']
                rates[c['width']]=dict(success_lower=status['success']/n,success_upper=(status['success']+unknown)/n,
                    status_counts=dict(status))
            observed=min((w for w,r in rates.items() if r['success_lower']>=.8),default=None)
            possible=min((w for w,r in rates.items() if r['success_upper']>=.8),default=None)
            widths.append(dict(design=design,method=method,per_seed=per_seed,success_rate_bounds=rates,
                w80=observed if observed==possible else None,observed_w80=observed,possible_w80=possible))
            for phase in ('runs','timing'):
                rows=[load(phase,c,s,method) for c in cases for s in plan['seeds']]
                summaries.append(dict(design=design,method=method,phase=phase,status_counts=dict(Counter(r['status'] for r in rows)),
                    timeout_budget_counts=dict(Counter(str(r.get('signature',{}).get('timeout','not_measured')) for r in rows)),
                    instrumentation_counts=dict(Counter('observed_runner' if r.get('orchestration') else
                        'skipped' if r['status'] in SKIPPED else 'legacy_runner' for r in rows)),
                    timeout_wall_seconds=[r['wall_seconds'] for r in rows if r['status']=='timeout'],
                    worker_stage_at_timeout=[r.get('last_observed_stage','unknown_legacy_buffered_log') for r in rows if r['status']=='timeout']))
        for candidate,baseline in f.PAIRS:
            pairs=[];excluded=Counter()
            selected_pairs=next(p for p in selection['comparisons'] if p['design']==design and
                p['candidate']==candidate and p['baseline']==baseline)
            selected_keys={(p['width'],p['seed']) for p in selected_pairs['pairs']}
            for c in cases:
                timing=f.read(c['generation'])['timing']
                for s in plan['seeds']:
                    if (c['width'],s) not in selected_keys:continue
                    b=load('timing',c,s,baseline);r=load('timing',c,s,candidate)
                    if b['status']==r['status']=='success':
                        pairs.append(dict(width=c['width'],seed=s,
                            speedup={scope:f.total(b,timing,scope)/f.total(r,timing,scope) for scope in f.SCOPES},
                            wirelength_ratio=r['wirelength']/b['wirelength'] if r.get('wirelength') and b.get('wirelength') else None,
                            cpd_ratio=r['cpd']/b['cpd'] if r.get('cpd') and b.get('cpd') else None))
                    else:excluded[f'{b["status"]} / {r["status"]}']+=1
            comparisons.append(dict(design=design,candidate=candidate,baseline=baseline,joint_success_pairs=len(pairs),
                selected_pair_count=len(selected_keys),no_joint_success_seeds=selected_pairs['no_joint_success_seeds'],
                excluded_pairs=dict(excluded),pairs=pairs,speedup={scope:math.exp(statistics.mean(math.log(r['speedup'][scope]) for r in pairs))
                if pairs else None for scope in f.SCOPES},interpretation='conditional on both succeeding; inspect excluded/timeout counts'))
    status=counts(a,plan)
    progress=timing_progress(a,plan,selection)
    accounted=scan['scan_complete'] and progress['remaining']==0
    f.write(a.out/'resume_evaluation_report.json',dict(complete=accounted,phase_status_counts=status,
        timing_selection_progress=progress,timing_selection=selection,wmin_comparisons=scan['wmin_comparisons'],
        all_scan_route_outcomes_known=not(set(status['runs'])-{'success','unroutable'}),
        all_timing_executed=not(set(status['timing'])-TERMINAL),
        width_results=widths,runtime_comparisons=comparisons,trial_summaries=summaries,
        counted_components=f.SCOPES,full_cold_from_raw_rrg=None,
        timeout_semantics='budget exhausted; not unroutable. Affected Wmin unresolved; no speedup from skipped/failed trials',
        refinement_cost='already inside VPR wall',speedup_direction='baseline/candidate; >1 faster'))
    emit('report_written',complete=accounted,phase_status_counts=status,timing_selection_progress=progress)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['generate','run','scan-report','timing-plan','timing','report','status','stop'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--prior-weight',type=float,default=.35);p.add_argument('--seeds',nargs='+',type=int,default=[251,252,253,254,255])
    p.add_argument('--widths',nargs='+',type=int);p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--heartbeat',type=int,default=120)
    p.add_argument('--retry-timeouts',action='store_true');p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--time-scan-timeouts',action='store_true')
    p.add_argument('--timing-seed-count',type=int,default=3)
    p.add_argument('--timing-width-count',type=int,default=1)
    p.add_argument('--replan-timing',action='store_true')
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if min(a.jobs,a.timeout,a.heartbeat,a.timing_seed_count,a.timing_width_count)<1:p.error('counts/budgets must be positive')
    if len(a.seeds)<3 or len(set(a.seeds))!=len(a.seeds):p.error('need >=3 distinct seeds')
    if a.widths and (len(a.widths)<3 or a.widths!=sorted(set(a.widths)) or min(a.widths)<=0):p.error('invalid width grid')
    if a.phase in ('run','timing'):
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:STOP.set())
        execute(a,a.phase=='timing')
    elif a.phase=='generate':f.direct.generate(a)
    elif a.phase=='scan-report':scan_report(a)
    elif a.phase=='timing-plan':
        plan=f.read(a.out/'plan.json');selected=timing_plan(a,plan)
        emit('timing_plan',**timing_progress(a,plan,selected))
    elif a.phase=='report':report(a)
    elif a.phase=='stop':request_stop(a)
    else:
        plan=f.read(a.out/'plan.json') if (a.out/'plan.json').exists() else None
        status=counts(a,plan) if plan else {}
        path=a.out/'resume_evaluation_report.json'
        report=f.read(path) if path.exists() else None
        selection_path=a.out/'timing_selection.json'
        selection=f.read(selection_path) if selection_path.exists() else None
        progress=timing_progress(a,plan,selection) if plan and selection else None
        emit('status',expected_per_phase=len(plan['cases'])*len(plan['seeds'])*4 if plan else None,
             expected_selected_timing=progress['expected'] if progress else None,
             phase_status_counts=status,report_exists=report is not None,
             scan_report_exists=(a.out/'scan_report.json').exists(),timing_selection_progress=progress,
             complete=bool(report and report['complete'] and progress and progress['remaining']==0 and
                report.get('timing_selection',{}).get('signature')==selection['signature'] and
                not(set(status['runs'])-TERMINAL)),
             all_scan_route_outcomes_known=bool(plan and not(set(status['runs'])-{'success','unroutable'})))


if __name__=='__main__':main()
