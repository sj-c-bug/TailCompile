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
        status=status,process_returncode=code,wall_seconds=elapsed,command=cmd,launched_command=launched,
        phases=f.parse_vpr_log(text),local_refinement=selection,refinement_already_in_wall_time=True,
        last_observed_stage=current,stage_history=history,
        orchestration=dict(resume_sha256=f.sha(Path(__file__)),observer_sha256=f.sha(HERE/'observe_refine.py')),
        wirelength=metric(r'Total wirelength:\s*([\d.eE+-]+)'),
        cpd=metric(r'Final critical path delay \(least slack\):\s*([\d.eE+-]+)'),
        hook_env={k:v for k,v in env.items() if k.startswith('TC_GP_')})
    f.write(folder/'trial.json',result)
    event('trial_finished',status=status,last_observed_stage=current)
    return outcome(folder,result,reused=False)


def run_trial(a,backend,identity,task,timing=False):
    c,seed,method,folder=task;trial=folder/'trial.json'
    if trial.exists():
        raw=f.read(trial);validate_old(a,identity,c,seed,method,raw);row=normalized(raw)
        retry=(row['status']=='timeout' and a.retry_timeouts) or (
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
    pending=list(f.tasks(a,plan,timing));counts=Counter();errors=[]
    # Validate the first C/D paths. A timeout is recorded and does not abort.
    preflight=[]
    for method in f.METHODS[2:]:
        task=next(t for t in pending if t[2]==method)
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


def report(a):
    plan=f.read(a.out/'plan.json');widths=[];comparisons=[];summaries=[]
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
            for c in cases:
                timing=f.read(c['generation'])['timing']
                for s in plan['seeds']:
                    b=load('timing',c,s,baseline);r=load('timing',c,s,candidate)
                    if b['status']==r['status']=='success':
                        pairs.append(dict(width=c['width'],seed=s,
                            speedup={scope:f.total(b,timing,scope)/f.total(r,timing,scope) for scope in f.SCOPES},
                            wirelength_ratio=r['wirelength']/b['wirelength'] if r.get('wirelength') and b.get('wirelength') else None,
                            cpd_ratio=r['cpd']/b['cpd'] if r.get('cpd') and b.get('cpd') else None))
                    else:excluded[f'{b["status"]} / {r["status"]}']+=1
            comparisons.append(dict(design=design,candidate=candidate,baseline=baseline,joint_success_pairs=len(pairs),
                excluded_pairs=dict(excluded),pairs=pairs,speedup={scope:math.exp(statistics.mean(math.log(r['speedup'][scope]) for r in pairs))
                if pairs else None for scope in f.SCOPES},interpretation='conditional on both succeeding; inspect excluded/timeout counts'))
    status=counts(a,plan)
    accounted=all(not(set(c)-TERMINAL-SKIPPED) for c in status.values())
    f.write(a.out/'resume_evaluation_report.json',dict(complete=accounted,phase_status_counts=status,
        all_scan_route_outcomes_known=not(set(status['runs'])-{'success','unroutable'}),
        all_timing_executed=not(set(status['timing'])-TERMINAL),
        width_results=widths,runtime_comparisons=comparisons,trial_summaries=summaries,
        counted_components=f.SCOPES,full_cold_from_raw_rrg=None,
        timeout_semantics='budget exhausted; not unroutable. Affected Wmin unresolved; no speedup from skipped/failed trials',
        refinement_cost='already inside VPR wall',speedup_direction='baseline/candidate; >1 faster'))
    emit('report_written',complete=accounted,phase_status_counts=status)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['generate','run','timing','report','status'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True);p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--device',default='cpu')
    p.add_argument('--prior-weight',type=float,default=.35);p.add_argument('--seeds',nargs='+',type=int,default=[251,252,253,254,255])
    p.add_argument('--widths',nargs='+',type=int);p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--heartbeat',type=int,default=120)
    p.add_argument('--retry-timeouts',action='store_true');p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--time-scan-timeouts',action='store_true')
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if min(a.jobs,a.timeout,a.heartbeat)<1:p.error('jobs, timeout and heartbeat must be positive')
    if len(a.seeds)<3 or len(set(a.seeds))!=len(a.seeds):p.error('need >=3 distinct seeds')
    if a.widths and (len(a.widths)<3 or a.widths!=sorted(set(a.widths)) or min(a.widths)<=0):p.error('invalid width grid')
    if a.phase in ('run','timing'):
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:STOP.set())
        execute(a,a.phase=='timing')
    elif a.phase=='generate':f.direct.generate(a)
    elif a.phase=='report':report(a)
    else:
        plan=f.read(a.out/'plan.json') if (a.out/'plan.json').exists() else None
        status=counts(a,plan) if plan else {}
        path=a.out/'resume_evaluation_report.json'
        report=f.read(path) if path.exists() else None
        emit('status',expected_per_phase=len(plan['cases'])*len(plan['seeds'])*4 if plan else None,
             phase_status_counts=status,report_exists=report is not None,
             complete=bool(report and report['complete'] and report['phase_status_counts']==status),
             all_scan_route_outcomes_known=bool(plan and not(set(status['runs'])-{'success','unroutable'})))


if __name__=='__main__':main()
