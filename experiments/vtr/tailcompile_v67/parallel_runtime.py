"""Pinned, paired, synchronized parallel timing; independent from serial data.

Each lane measures two routes of one comparison on the SAME reported physical
core. Half-wave barriers and alternating AB/BA order reduce scheduling bias.
Memory/cache/frequency effects remain: reports identify measured load/host.
"""
import argparse
from collections import Counter,defaultdict
import concurrent.futures as cf
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import signal
import statistics
import threading
import time
from types import SimpleNamespace
import uuid

import resume as r

HERE=Path(__file__).resolve().parent


def read_text(path,default=None):
    try:return Path(path).read_text().strip()
    except OSError:return default


def parse_cpus(value):
    cpus=set()
    for token in value.split(','):
        if '-' in token:
            low,high=map(int,token.split('-'));cpus.update(range(low,high+1))
        else:cpus.add(int(token))
    return sorted(cpus)


def topology(allowed=None,sysroot=Path('/sys/devices/system/cpu')):
    allowed=sorted(os.sched_getaffinity(0) if allowed is None else allowed)
    rows=[]
    for cpu in allowed:
        base=sysroot/f'cpu{cpu}'
        package=read_text(base/'topology/physical_package_id');core=read_text(base/'topology/core_id')
        if package is None or core is None:raise RuntimeError('missing CPU topology; cannot claim physical-core isolation')
        nodes=sorted(base.glob('node[0-9]*'))
        rows.append(dict(cpu=cpu,package=int(package),core=int(core),
            numa=int(nodes[0].name[4:]) if nodes else None,
            siblings=read_text(base/'topology/thread_siblings_list'),
            l3_id=read_text(base/'cache/index3/id'),governor=read_text(base/'cpufreq/scaling_governor')))
    return rows


def select_cores(rows,n,manual=None,quota=None):
    allowed={row['cpu']:row for row in rows}
    if manual is not None:
        if not set(manual)<=set(allowed):raise ValueError('requested CPUs are outside this process affinity/cpuset')
        selected=[allowed[c] for c in manual]
        if len(selected)!=n:raise ValueError('CPU list must contain exactly concurrency distinct CPUs')
    else:
        selected=[];seen=set()
        for row in sorted(rows,key=lambda x:(x['numa'] if x['numa'] is not None else -1,x['package'],x['core'],x['cpu'])):
            key=row['package'],row['core']
            if key not in seen:seen.add(key);selected.append(row)
        selected=selected[:n]
    if len(selected)<n:raise ValueError(f'only {len(selected)} distinct reported physical cores available; reduce concurrency')
    if len({(x['package'],x['core']) for x in selected})!=n:raise ValueError('CPU list includes SMT siblings of the same physical core')
    if quota is not None and n>quota:raise ValueError(f'concurrency {n} exceeds cgroup quota {quota:g} CPU equivalents')
    return selected


def quota_info(root=Path('/sys/fs/cgroup')):
    root=root.resolve()
    values=[]
    # Container-root limits plus the current process cgroup when mounted at host root.
    paths={root}
    for line in (read_text('/proc/self/cgroup','') or '').splitlines():
        if line.startswith('0::'):
            candidate=(root/line[3:].lstrip('/')).resolve()
            if candidate.is_relative_to(root):paths.add(candidate)
    for path in list(paths):
        if path.is_relative_to(root):paths.update([p for p in path.parents if p.is_relative_to(root)])
    for path in paths:
        raw=read_text(path/'cpu.max')
        if raw and raw.split()[0]!='max':
            q,period=map(int,raw.split());values.append(q/period)
    return min(values) if values else None


def hardware(n,manual=None):
    rows=topology();quota=quota_info();cores=select_cores(rows,n,manual,quota)
    model=next((line.split(':',1)[1].strip() for line in (read_text('/proc/cpuinfo','') or '').splitlines()
                if line.startswith('model name')),'unknown')
    return dict(host=platform.node(),boot_id=read_text('/proc/sys/kernel/random/boot_id'),
        cpu_model=model,platform=platform.platform(),allowed_logical_cpus=sorted(os.sched_getaffinity(0)),
        reported_physical_cores=len({(x['package'],x['core']) for x in rows}),quota_cpu_equivalents=quota,
        selected_cores=cores,memory_limit=read_text('/sys/fs/cgroup/memory.max'),
        note='Reported OS/VM topology; binding does not reserve cache, memory bandwidth or the host.')


def code_hashes():
    return {p.name:r.f.sha(p) for p in (Path(__file__),HERE/'resume.py',HERE/'affinity_exec.py',HERE/'observe_refine.py')}


def run_dir(a):
    if not a.run_name or any(ch not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for ch in a.run_name):
        raise ValueError('run name must contain only letters, digits, underscores and hyphens')
    return a.out/'timing_parallel'/a.run_name


def wave_schedule(selection,cores):
    units=[]
    for comparison_index,comp in enumerate(selection['comparisons']):
        for pair in comp['pairs']:
            seed_index=selection['fixed_seeds'].index(pair['seed'])
            methods=[comp['baseline'],comp['candidate']]
            if (comparison_index+seed_index)%2:methods.reverse()
            units.append(dict(design=comp['design'],width=pair['width'],seed=pair['seed'],
                candidate=comp['candidate'],baseline=comp['baseline'],methods=methods,
                comparison_index=comparison_index))
    waves=[];n=len(cores)
    for start in range(0,len(units),n):
        index=len(waves);lanes=[]
        for lane,unit in enumerate(units[start:start+n]):
            lanes.append(dict(unit,lane=lane,cpu=cores[(index+lane)%n]['cpu']))
        waves.append(dict(index=index,effective_lanes=len(lanes),lanes=lanes))
    return waves


def prepare(a):
    if os.name!='posix':raise RuntimeError('parallel timing requires Linux/WSL')
    plan=r.f.read(a.out/'plan.json');selection=r.timing_plan(a,plan)
    if plan['signature']['code']!=r.f.direct.implementation():raise ValueError('generator changed')
    backend=r.f.direct.DirectBackend(os.environ['VTR_ROOT']);hook=r.f.verify(Path(backend.root))
    identity=dict(binary=backend.binary_hash,arch=backend.arch_hash,hook=hook,code=r.f.code_identity(),threads=1)
    original=r.f.read(a.out/'abcd_protocol.json')
    if original['identity']!=identity or original['plan_sha256']!=r.f.sha(a.out/'plan.json'):
        raise ValueError('scan algorithm/binary/inputs changed; cannot compare in this cohort')
    hw=hardware(a.concurrency,parse_cpus(a.cpu_list) if a.cpu_list else None)
    # Frequency/memory snapshots vary and belong in session logs, not frozen identity.
    packages={}
    for name in ('numpy','networkx'):
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    protocol=dict(schema='tailcompile-controlled-parallel-v1',identity=identity,hardware=hw,
        python_executable=str(Path(os.sys.executable).resolve()),python_version=platform.python_version(),packages=packages,
        generation_reports_sha256={f'{c["design"]}/w{c["width"]}':r.f.sha(c['generation']) for c in plan['cases']},
        concurrency=a.concurrency,timeout=a.timeout,code=code_hashes(),
        input_plan_sha256=r.f.sha(a.out/'plan.json'),selection_sha256=r.f.sha(a.out/'timing_selection.json'),
        generation_cost='Existing generation.json CPU/GPU observations; not remeasured under concurrent VPR load',
        policy='paired same-core AB/BA; synchronized half-waves; incomplete wave rerun as a unit')
    folder=run_dir(a);r.f.freeze(folder/'protocol.json',protocol)
    schedule=dict(selection=selection,waves=wave_schedule(selection,hw['selected_cores']))
    r.f.freeze(folder/'schedule.json',schedule)
    for c in plan['cases']:
        for path,digest in [(c['blif'],c['sources']['mapped_blif']),
                            (c['packed'],c['sources']['packed_net']),(c['fplace'],c['fplace_sha256'])]:
            if r.f.sha(path)!=digest:raise ValueError(f'input changed: {path}')
    return folder,protocol,schedule,plan,backend


def wave_folder(folder,wave):return folder/'waves'/f'w{wave["index"]:04d}'


def completed_wave(folder,wave):
    path=wave_folder(folder,wave)/'wave.json'
    if not path.exists():return None
    row=r.f.read(path)
    if not row.get('complete'):return None
    for trial,digest in row['trial_hashes'].items():
        p=folder/trial
        if not p.is_file() or r.f.sha(p)!=digest:raise ValueError('completed wave evidence changed')
    return row


def archive_wave(folder,path):
    path.resolve().relative_to((folder/'waves').resolve())
    if path.is_symlink():raise ValueError('refuse symlink wave directory')
    for process in path.rglob('process.json'):
        if r.process_alive(process.parent):raise RuntimeError(f'previous child still active: {process}')
    dest=folder/'_attempts'/(path.name+'_'+time.strftime('%Y%m%d_%H%M%S',time.gmtime())+'_'+uuid.uuid4().hex[:8])
    dest.parent.mkdir(parents=True,exist_ok=True);path.rename(dest)
    r.emit('parallel_wave_archived',destination=str(dest))


def execute(a):
    folder,protocol,schedule,plan,backend=prepare(a)
    cases={(c['design'],c['width']):c for c in plan['cases']};started=time.time();summaries=[]
    for wave in schedule['waves']:
        if r.STOP.is_set():break
        old=completed_wave(folder,wave)
        if old:
            r.emit('parallel_wave_reused',wave=wave['index']);summaries.append(old);continue
        path=wave_folder(folder,wave)
        if path.exists():archive_wave(folder,path)
        path.mkdir(parents=True,exist_ok=True)
        wave_start=time.time();results={};hashes={};errors=[]
        load_before=dict(cpu_stat=read_text('/sys/fs/cgroup/cpu.stat'),load_average=list(os.getloadavg()),
            cpu_frequency_khz={str(lane['cpu']):read_text(f'/sys/devices/system/cpu/cpu{lane["cpu"]}/cpufreq/scaling_cur_freq')
                               for lane in wave['lanes']})
        # Wave tails may have fewer lanes; their reports stay in a separate load stratum.
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
                    except Exception as error:errors.append(repr(error))
            if errors:break
        complete=len(results)==2*wave['effective_lanes'] and all(row['status'] in r.TERMINAL for row in results.values())
        data=dict(complete=complete,wave=wave,started_unix=wave_start,finished_unix=time.time(),
            load_before=load_before,cpu_stat_after=read_text('/sys/fs/cgroup/cpu.stat'),
            wall_seconds=time.time()-wave_start,results=results,trial_hashes=hashes,errors=errors,
            memory_available=read_text('/proc/meminfo'),cpu_pressure=read_text('/proc/pressure/cpu'),
            memory_pressure=read_text('/proc/pressure/memory'))
        r.f.write(path/'wave.json',data);summaries.append(data)
        r.emit('parallel_wave_finished',wave=wave['index'],complete=complete,effective_lanes=wave['effective_lanes'])
        if not complete:
            if r.STOP.is_set():break
            raise RuntimeError(f'incomplete wave: {path}; diagnose then resume (prior attempt archived)')
    r.f.write(folder/'last_session.json',dict(started_unix=started,finished_unix=time.time(),
        host=platform.node(),waves_seen=len(summaries),stopped=r.STOP.is_set()))
    if r.STOP.is_set():raise RuntimeError('parallel timing stopped; completed waves retained')


def load_stats(rows):
    if not rows:return dict(peak_active_trials=0,mean_active_trials=0,start_skew_seconds=0)
    events=sorted((row['started_unix']+offset,delta) for row in rows
                  for offset,delta in [(0,1),(row['wall_seconds'],-1)])
    active=peak=0
    for _,delta in events:active+=delta;peak=max(peak,active)
    starts=[row['started_unix'] for row in rows]
    span=max(row['started_unix']+row['wall_seconds'] for row in rows)-min(starts)
    return dict(peak_active_trials=peak,mean_active_trials=sum(row['wall_seconds'] for row in rows)/span if span else 0,
                start_skew_seconds=max(starts)-min(starts))


def report(a):
    folder=run_dir(a);protocol=r.f.read(folder/'protocol.json');schedule=r.f.read(folder/'schedule.json')
    plan=r.f.read(a.out/'plan.json');cases={(c['design'],c['width']):c for c in plan['cases']}
    groups=defaultdict(list);status=Counter();waves_complete=0;measured=0.;skews=[];wave_rows=[];loads=[]
    for wave in schedule['waves']:
        data=completed_wave(folder,wave)
        if data is None:continue
        waves_complete+=1;measured+=data['wall_seconds']
        for half in (0,1):
            starts=[data['results'][f'{half}:{lane["lane"]}']['started_unix'] for lane in wave['lanes']]
            skews.append(max(starts)-min(starts))
            loads.append(dict(wave=wave['index'],half=half,effective_lanes=wave['effective_lanes'],
                **load_stats([data['results'][f'{half}:{lane["lane"]}'] for lane in wave['lanes']])))
        for lane in wave['lanes']:
            values={data['results'][f'{h}:{lane["lane"]}']['method']:data['results'][f'{h}:{lane["lane"]}'] for h in (0,1)}
            baseline=values[lane['baseline']];candidate=values[lane['candidate']]
            status.update([baseline['status'],candidate['status']])
            c=cases[lane['design'],lane['width']]
            # Portable fallback for server paths after Git synchronization.
            generation=Path(c['generation'])
            if not generation.exists():generation=a.out/'generation'/lane['design']/f'w{lane["width"]}'/'generation.json'
            expected=protocol.get('generation_reports_sha256',{}).get(f'{lane["design"]}/w{lane["width"]}')
            if expected is not None and r.f.sha(generation)!=expected:raise ValueError('generation timing evidence changed')
            t=r.f.read(generation)['timing']
            paired=baseline['status']==candidate['status']=='success'
            item=dict(wave=wave['index'],design=lane['design'],width=lane['width'],seed=lane['seed'],
                cpu=lane['cpu'],order=lane['methods'],baseline_status=baseline['status'],candidate_status=candidate['status'],
                baseline_wall_seconds=baseline['wall_seconds'],candidate_wall_seconds=candidate['wall_seconds'],
                vpr_wall_speedup=baseline['wall_seconds']/candidate['wall_seconds'] if paired else None,
                scope_speedup={scope:r.f.total(baseline,t,scope)/r.f.total(candidate,t,scope) if paired else None for scope in r.f.SCOPES},
                wirelength_ratio=candidate['wirelength']/baseline['wirelength'] if paired and candidate.get('wirelength') and baseline.get('wirelength') else None,
                cpd_ratio=candidate['cpd']/baseline['cpd'] if paired and candidate.get('cpd') and baseline.get('cpd') else None)
            # Never pool a partially filled tail-wave with the requested full concurrency.
            groups[(lane['design'],lane['candidate'],lane['baseline'],wave['effective_lanes'])].append(item)
        wave_rows.append(dict(index=wave['index'],effective_lanes=wave['effective_lanes'],wall_seconds=data['wall_seconds']))
    comparisons=[]
    for (design,candidate,baseline,load),rows in sorted(groups.items()):
        valid=[row for row in rows if row['vpr_wall_speedup'] is not None]
        def geomean(values):return math.exp(statistics.mean(math.log(v) for v in values)) if values else None
        comparisons.append(dict(design=design,candidate=candidate,baseline=baseline,effective_lanes=load,
            pair_count=len(rows),joint_success_pairs=len(valid),excluded_pairs=len(rows)-len(valid),pairs=rows,
            order_counts=dict(Counter('baseline_first' if row['order'][0]==baseline else 'candidate_first' for row in rows)),
            vpr_wall_speedup=geomean([row['vpr_wall_speedup'] for row in valid]),
            qor_candidate_over_baseline={key:geomean([row[key] for row in valid if row[key] is not None and row[key]>0])
                                         for key in ('wirelength_ratio','cpd_ratio')},
            scope_speedup={scope:geomean([row['scope_speedup'][scope] for row in valid]) for scope in r.f.SCOPES}))
    result=dict(complete=waves_complete==len(schedule['waves']),protocol=protocol,
        waves_complete=waves_complete,waves_expected=len(schedule['waves']),status_counts=dict(status),
        comparisons=comparisons,waves=wave_rows,half_wave_load=loads,max_half_wave_start_skew_seconds=max(skews,default=None),
        throughput=dict(measured_completed_wave_seconds=measured,completed_trials=sum(status.values()),
            trials_per_hour=3600*sum(status.values())/measured if measured else None,
            note='sum of completed wave wall times; not elapsed calendar time across stopped sessions'),
        generation_timing_source='pre-existing generation observations, separately charged; not measured under parallel load',
        serial_results_included=False,speedup_direction='baseline/candidate; >1 faster',
        caveat='Concurrent wall-time comparison under this wave protocol, not isolated single-task latency. Partial wave loads reported separately.')
    r.f.write(folder/'parallel_report.json',result);r.emit('parallel_report',complete=result['complete'],
        waves_complete=waves_complete,waves_expected=len(schedule['waves']),status_counts=dict(status))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['plan','run','report','status'])
    p.add_argument('--out',type=Path,required=True);p.add_argument('--run-name',default='c4_001')
    p.add_argument('--concurrency',type=int,default=4);p.add_argument('--cpu-list')
    p.add_argument('--timeout',type=int,default=7200);p.add_argument('--heartbeat',type=int,default=120)
    p.add_argument('--timing-seed-count',type=int,default=3);p.add_argument('--timing-width-count',type=int,default=1)
    a=p.parse_args();a.out=a.out.resolve();a.replan_timing=False
    if min(a.concurrency,a.timeout,a.heartbeat,a.timing_seed_count,a.timing_width_count)<1:p.error('positive concurrency and budgets required')
    if a.phase=='plan':
        folder,protocol,schedule,_,_=prepare(a)
        r.emit('parallel_plan',path=str(folder),concurrency=a.concurrency,
            cpus=[c['cpu'] for c in protocol['hardware']['selected_cores']],waves=len(schedule['waves']),
            available_reported_physical_cores=protocol['hardware']['reported_physical_cores'],
            quota_cpu_equivalents=protocol['hardware']['quota_cpu_equivalents'],memory_limit=protocol['hardware']['memory_limit'],
            trials=sum(2*w['effective_lanes'] for w in schedule['waves']))
    elif a.phase=='run':
        for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,lambda *_:r.STOP.set())
        execute(a)
    elif a.phase=='report':report(a)
    else:
        folder=run_dir(a);schedule=r.f.read(folder/'schedule.json')
        complete=sum(completed_wave(folder,w) is not None for w in schedule['waves'])
        r.emit('parallel_status',run_name=a.run_name,waves_complete=complete,waves_expected=len(schedule['waves']),
            remaining_waves=len(schedule['waves'])-complete,report_exists=(folder/'parallel_report.json').exists())


if __name__=='__main__':main()
