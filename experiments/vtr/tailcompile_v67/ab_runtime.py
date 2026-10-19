"""Independent A/B-only timing, without changing the frozen ABCD experiment.

Uses the same VPR runner, affinity and paired wave protocol. Does NOT reuse
wall times from mixed ABCD waves (in particular disk-quota-failed waves).
Existing netlists, generated .fplace and frozen routing-only timing selection
are reused. No GNN inference, local refinement or new routing-width search.
"""
import argparse
from collections import Counter
import concurrent.futures as cf
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import threading
import time
from types import SimpleNamespace

import parallel_runtime as p

r=p.r


def ab_selection(selection):
    comparisons=[c for c in selection['comparisons']
                 if (c['baseline'],c['candidate'])==('A_default','B_gnn')]
    if len({c['design'] for c in comparisons})!=len(comparisons):
        raise ValueError('duplicate A/B comparison for one design')
    tasks=sorted({(c['design'],pair['width'],pair['seed'],method)
                  for c in comparisons for pair in c['pairs']
                  for method in ('A_default','B_gnn')})
    if not tasks:raise ValueError('no joint-success A/B pairs in frozen timing selection')
    return dict(selection,comparisons=comparisons,selected_tasks=[list(t) for t in tasks],
        selected_task_count=len(tasks),maximum_tasks_before_dedup=len(tasks),
        comparison_scope='A_default vs B_gnn only; same routing-only chosen pairs')


def prepare(a):
    if os.name!='posix':raise RuntimeError('A/B execution requires Linux/WSL')
    if not a.run_name.startswith('ab_'):raise ValueError('use an independent run name starting with ab_')
    plan=r.f.read(a.out/'plan.json')
    selection=ab_selection(r.timing_plan(a,plan))
    if plan['signature']['code']!=r.f.direct.implementation():raise ValueError('generator changed')
    backend=r.f.direct.DirectBackend(os.environ['VTR_ROOT'])
    identity=dict(binary=backend.binary_hash,arch=backend.arch_hash,
        hook=r.f.verify(Path(backend.root)),code=r.f.code_identity(),threads=1)
    original=r.f.read(a.out/'abcd_protocol.json')
    if original['identity']!=identity or original['plan_sha256']!=r.f.sha(a.out/'plan.json'):
        raise ValueError('original scan algorithm/binary/inputs changed')
    hw=p.hardware(a.concurrency,p.parse_cpus(a.cpu_list) if a.cpu_list else None)
    packages={}
    for name in ('numpy','networkx'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    protocol=dict(schema='tailcompile-ab-only-parallel-v1',identity=identity,hardware=hw,
        python_executable=str(Path(os.sys.executable).resolve()),python_version=platform.python_version(),packages=packages,
        generation_reports_sha256={f'{c["design"]}/w{c["width"]}':r.f.sha(c['generation']) for c in plan['cases']},
        concurrency=a.concurrency,timeout=a.timeout,code=dict(p.code_hashes(),ab_runtime=r.f.sha(Path(__file__))),
        input_plan_sha256=r.f.sha(a.out/'plan.json'),selection_sha256=r.f.sha(a.out/'timing_selection.json'),
        generation_cost='Existing generation.json observations, separately charged; no inference rerun',
        policy='A/B only; paired same-core AB/BA; synchronized half-waves; incomplete wave rerun as a unit',
        previous_runtime_reused=False,local_refinement=False)
    folder=p.run_dir(a)
    r.f.freeze(folder/'protocol.json',protocol)
    schedule=dict(selection=selection,waves=p.wave_schedule(selection,hw['selected_cores']))
    r.f.freeze(folder/'schedule.json',schedule)
    for c in plan['cases']:
        for path,digest in [(c['blif'],c['sources']['mapped_blif']),
                            (c['packed'],c['sources']['packed_net']),(c['fplace'],c['fplace_sha256'])]:
            if r.f.sha(path)!=digest:raise ValueError(f'input changed: {path}')
    return folder,protocol,schedule,plan,backend


def execute(a):
    folder,protocol,schedule,plan,backend=prepare(a)
    cases={(c['design'],c['width']):c for c in plan['cases']}
    started=time.time()
    for wave in schedule['waves']:
        if r.STOP.is_set():break
        if p.completed_wave(folder,wave):
            r.emit('parallel_wave_reused',wave=wave['index']);continue
        path=p.wave_folder(folder,wave)
        if path.exists():p.archive_wave(folder,path)
        path.mkdir(parents=True,exist_ok=True)
        wave_start=time.time();results={};hashes={};errors=[]
        load_before=dict(cpu_stat=p.read_text('/sys/fs/cgroup/cpu.stat'),load_average=list(os.getloadavg()),
            cpu_frequency_khz={str(lane['cpu']):p.read_text(f'/sys/devices/system/cpu/cpu{lane["cpu"]}/cpufreq/scaling_cur_freq')
                               for lane in wave['lanes']})
        for half in (0,1):
            if r.STOP.is_set():break
            barrier=threading.Barrier(wave['effective_lanes'])
            def one(lane):
                c=cases[lane['design'],lane['width']];method=lane['methods'][half]
                directory=path/f'half{half}'/f'lane{lane["lane"]:02d}'
                args=SimpleNamespace(**dict(vars(a),pinned_cpu=lane['cpu'],launch_barrier=barrier))
                try:return lane,r.run_process(args,backend,protocol['identity'],c,lane['seed'],method,directory)
                except BaseException:
                    barrier.abort();raise
            with cf.ThreadPoolExecutor(max_workers=wave['effective_lanes']) as executor:
                futures=[executor.submit(one,lane) for lane in wave['lanes']]
                for future in cf.as_completed(futures):
                    try:
                        lane,row=future.result();key=f'{half}:{lane["lane"]}'
                        results[key]=row
                        trial=path/f'half{half}'/f'lane{lane["lane"]:02d}'/'trial.json'
                        hashes[trial.relative_to(folder).as_posix()]=r.f.sha(trial)
                        r.emit('ab_trial_finished',wave=wave['index'],half=half,lane=lane['lane'],
                            method=row['method'],status=row['status'],finished_in_wave=len(results),
                            expected_in_wave=2*wave['effective_lanes'])
                    except Exception as error:errors.append(repr(error))
            if errors or any(row['status'] not in r.TERMINAL for row in results.values()):break
        complete=len(results)==2*wave['effective_lanes'] and all(row['status'] in r.TERMINAL for row in results.values())
        data=dict(complete=complete,wave=wave,started_unix=wave_start,finished_unix=time.time(),
            load_before=load_before,cpu_stat_after=p.read_text('/sys/fs/cgroup/cpu.stat'),
            wall_seconds=time.time()-wave_start,results=results,trial_hashes=hashes,errors=errors,
            memory_available=p.read_text('/proc/meminfo'),cpu_pressure=p.read_text('/proc/pressure/cpu'),
            memory_pressure=p.read_text('/proc/pressure/memory'))
        r.f.write(path/'wave.json',data)
        r.emit('parallel_wave_finished',wave=wave['index'],complete=complete,effective_lanes=wave['effective_lanes'])
        if not complete:
            r.f.write(folder/'last_session.json',dict(started_unix=started,finished_unix=time.time(),stopped=r.STOP.is_set()))
            raise RuntimeError(f'incomplete A/B wave: {path}; inspect trial.json/vpr.out; check storage quota before retry')
    r.f.write(folder/'last_session.json',dict(started_unix=started,finished_unix=time.time(),stopped=r.STOP.is_set()))
    if r.STOP.is_set():raise RuntimeError('A/B timing stopped; completed waves retained')


def status(a):
    folder=p.run_dir(a);schedule=r.f.read(folder/'schedule.json')
    counts=Counter();active=0;complete=0
    for wave in schedule['waves']:
        complete+=p.completed_wave(folder,wave) is not None
        for half in (0,1):
            for lane in wave['lanes']:
                directory=p.wave_folder(folder,wave)/f'half{half}'/f'lane{lane["lane"]:02d}'
                if (directory/'trial.json').exists():
                    try:counts[r.f.read(directory/'trial.json')['status']]+=1
                    except (OSError,ValueError,KeyError):counts['invalid_trial_json']+=1
                elif r.process_alive(directory):counts['running']+=1
                elif (directory/'started.json').exists():counts['started_without_result']+=1
                else:counts['missing']+=1
                active+=r.process_alive(directory)
    result=dict(run_name=a.run_name,waves_complete=complete,waves_expected=len(schedule['waves']),
        trial_status_counts=dict(counts),trials_expected=sum(2*w['effective_lanes'] for w in schedule['waves']),
        active_processes=active,report_exists=(folder/'parallel_report.json').exists(),
        complete=complete==len(schedule['waves']))
    r.emit('ab_status',**result)
    return result


def report(a):
    # Existing report code keeps effective-load strata separate and charges
    # generation costs according to the original explicit scope definitions.
    p.report(a)
    path=p.run_dir(a)/'parallel_report.json';result=r.f.read(path)
    result.update(experiment='A_default_vs_B_gnn_only',local_refinement=False,
        inference_and_generation_reused=True,old_mixed_runtime_reused=False,
        interpretation='A/B runtime test, not evidence of acceleration until measured; fixed selected joint-success widths, not a new W_min scan')
    r.f.write(path,result)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['plan','run','report','status','evaluate'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--run-name',default='ab_c8_001')
    parser.add_argument('--concurrency',type=int,default=8);parser.add_argument('--cpu-list')
    parser.add_argument('--timeout',type=int,default=7200);parser.add_argument('--heartbeat',type=int,default=120)
    a=parser.parse_args();a.out=a.out.resolve();a.replan_timing=False
    if not a.run_name.startswith('ab_'):parser.error('--run-name must start with ab_ (do not overwrite c16_001)')
    if min(a.concurrency,a.timeout,a.heartbeat)<1:parser.error('positive concurrency/time budgets required')
    selection=r.f.read(a.out/'timing_selection.json')
    a.timing_seed_count=selection['signature']['seed_count']
    a.timing_width_count=selection['signature']['width_count']
    if a.phase=='status':status(a);return
    if os.name!='posix':parser.error('execution/reporting requires Linux/WSL')
    import fcntl
    # Respect the original pipeline lock: do not time alongside an ABCD run.
    with (a.out/'.lock').open('a') as lock:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('another v6.7 phase is active; stop it before A/B timing')
        if a.phase=='report':report(a);return
        if a.phase=='plan':
            folder,protocol,schedule,_,_=prepare(a)
            r.emit('ab_plan',path=str(folder),concurrency=a.concurrency,
                cpus=[c['cpu'] for c in protocol['hardware']['selected_cores']],
                pairs=sum(w['effective_lanes'] for w in schedule['waves']),
                trials=sum(2*w['effective_lanes'] for w in schedule['waves']),waves=len(schedule['waves']),
                filesystem_free_GiB=shutil.disk_usage(folder).free/1024**3,
                quota_checked=False,warning='Filesystem free space does not establish remaining user/project/inode quota')
            return
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:r.STOP.set())
        execute(a)
        if a.phase=='evaluate':report(a)


if __name__=='__main__':main()
