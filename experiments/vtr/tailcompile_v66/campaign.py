"""CPU experiment orchestration. No candidate routing labels enter selection.

All paths in compact records are relative to this campaign. Keep CPU threads
per VPR fixed; jobs is independent process concurrency, not VPR thread count.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from core import read, write, sha, snapshot, positions, candidates, flat, retention
from install_hook import verify, COMMIT
from run_tailcompile_v4 import classify_vpr_result
from analyze_tailcompile_runtime import parse_vpr_log

def emit(stage, **kwargs):
    print(json.dumps(dict(stage=stage, **kwargs)), flush=True)

def frozen(path, value):
    if path.exists() and read(path) != value:
        raise ValueError(f'inputs/protocol changed: {path}; choose a new output directory')
    if not path.exists():
        write(path, value)

def code_identity():
    return {name:sha(HERE/name) for name in ['core.py','campaign.py','learn.py','worker.py','build_pool.py','gp_hook.inc']}

def flat_coordinates(path):
    if not path.is_file(): return None
    return {f[0]:tuple(map(float,f[1:5])) for line in path.read_text().splitlines()
            if line and not line.startswith('#') and len(f:=line.split())>=5}

def parallel(fn, items, jobs):
    # executor.map queues tasks but only jobs processes execute. Every result is persisted.
    failures, results = [], []
    with cf.ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = {executor.submit(fn, item): item for item in items}
        for future in cf.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(str(error))
                emit('task_failed', error=str(error))
    if failures:
        raise RuntimeError(f'{len(failures)} tasks failed; successful tasks retained. First: {failures[0]}')
    return results

def environment(a):
    a.vtr = Path(os.environ['VTR_ROOT']).resolve()
    a.binary = a.vtr / 'vpr/vpr'
    a.arch = a.vtr / 'vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml'
    hook = verify(a.vtr)
    check=subprocess.run([str(a.binary),'--help'],capture_output=True,text=True,timeout=30)
    if check.returncode:
        raise RuntimeError(f'VPR cannot start (check shared libraries): {check.stderr[-2000:]}')
    # A real GP smoke is still required: source verification alone cannot prove binary provenance.
    a.identity = dict(commit=COMMIT, hook_sha256=hook, binary_sha256=sha(a.binary), arch_sha256=sha(a.arch))
    if (a.out/'protocol.json').exists():
        protocol=read(a.out/'protocol.json')
        if protocol['identity']!=a.identity or protocol['code']!=code_identity():
            raise ValueError('frozen executable/source changed; use a new experiment cohort')
        if protocol['vpr_threads']!=a.vpr_threads or protocol['replicas']!=a.replicas:
            raise ValueError('thread/action budget changed; use original settings')

def registry(a):
    cohort = read(a.cohort)
    designs = cohort['designs']
    if a.designs:
        designs = [d for d in designs if d['design'] in a.designs]
        if set(a.designs) != {d['design'] for d in designs}:
            raise ValueError('unknown requested designs')
    families = {}
    for d in designs:
        if d['family'] in families and families[d['family']] != d['split']:
            raise ValueError('family split leakage')
        families[d['family']] = d['split']
    frozen(a.out/'cohort.json', dict(schema=cohort['schema'], designs=designs,
                                    exploratory_subset=bool(a.designs)))
    return designs

def prepare(a):
    designs = registry(a)
    def one(d):
        folder = a.out/'inputs'/d['design']
        source = a.vtr/'vtr_flow/benchmarks/verilog'/d['source']
        if not source.is_file():
            raise FileNotFoundError(source)
        signature = dict(a.identity, source_sha256=sha(source),
                         include_sha256=sha(source.parent/'hard_block_include.v') if d['source'].startswith('koios/') else None)
        frozen(folder/'input_signature.json', signature)
        blif = folder/(d['design']+'.pre-vpr.blif')
        if blif.is_file() and (folder/'prepared.json').is_file():
            if read(folder/'prepared.json')['blif_sha256'] != sha(blif):
                raise ValueError('mapped netlist changed')
            return
        work = folder/'flow'
        work.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(a.vtr/'vtr_flow/scripts/run_vtr_flow.py'), str(source), str(a.arch),
                   '-temp_dir', str(work), '-ending_stage', 'abc', '-timeout', str(a.synthesis_timeout)]
        if d['source'].startswith('koios/'):
            command += ['-include', str(source.parent/'hard_block_include.v')]
        with (folder/'synthesis.log').open('w') as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=a.synthesis_timeout+120)
        # ABC-stage output is *.abc.blif, not the VPR postprocessor name.
        generated = work/(d['design']+'.abc.blif')
        if result.returncode or not generated.is_file():
            raise RuntimeError(f'synthesis failed: {folder}/synthesis.log')
        shutil.copy2(generated, blif)
        write(folder/'prepared.json', dict(blif_sha256=sha(blif), command=command, source=d['source']))
        emit('prepared', design=d['design'])
    parallel(one, designs, a.prepare_jobs)

def run(a, design, width, seed, directory, replacement=None, expected_gp=None, deploy=False):
    directory.mkdir(parents=True, exist_ok=True)
    blif = a.out/'inputs'/design/(design+'.pre-vpr.blif')
    signature = dict(a.identity, blif_sha256=sha(blif), width=width, seed=seed,
                     replacement_sha256=sha(replacement) if replacement else None,
                     threads=a.vpr_threads, timeout=a.timeout, full_legalizer='appack',
                     expected_gp=expected_gp)
    if deploy:
        signature['model_sha256']=sha(a.out/'model/ranker.pt')
        signature['worker_sha256']=sha(HERE/'worker.py')
    record = directory/'trial.json'
    if record.is_file():
        old = read(record)
        if old['signature'] != signature:
            raise ValueError(f'run signature changed: {directory}')
        if old['status'] not in ('success','unroutable'):
            raise ValueError(f'prior infrastructure failure: {directory}; use recover after diagnosis')
        return old
    # An interruption is never silently overwritten. The recover phase quarantines it.
    if (directory/'started.json').exists():
        raise ValueError(f'interrupted run: {directory}; use recover after stopping workers')
    started=dict(pid=os.getpid(), host=os.uname().nodename, signature=signature)
    write(directory/'started.json', started)
    env = dict(os.environ, TC_GP_EXPORT=str(directory/'gp'), OMP_NUM_THREADS=str(a.vpr_threads))
    env.pop('TC_GP_REPLACE', None)
    env.pop('TC_GP_WORKER', None)
    if replacement:
        env['TC_GP_REPLACE'] = str(replacement)
    if deploy:
        env['TC_GP_REPLACE']=str(directory/'deployed.fplace')
        env['TC_GP_WORKER']=shlex.join([sys.executable,str(HERE/'worker.py'),'--prefix',str(directory/'gp'),
            '--model',str(a.out/'model/ranker.pt'),'--output',str(directory/'deployed.fplace'),
            '--width',str(width),'--replicas',str(a.replicas)])
    command = [str(a.binary), str(a.arch), str(blif), '--analytical_place', '--route', '--analysis',
               '--ap_full_legalizer', 'appack', '--ap_detailed_placer', 'annealer',
               '--seed', str(seed), '--route_chan_width', str(width), '--num_workers', str(a.vpr_threads),
               '--write_flat_place', str(directory/'post_route.fplace'),
               '--write_legalized_flat_place', str(directory/'post_legalizer.fplace')]
    start = time.monotonic()
    with (directory/'vpr.out').open('w') as log:
        process=subprocess.Popen(command,cwd=directory,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        write(directory/'started.json',dict(started,child_pid=process.pid))
        try:
            code=process.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGTERM)
            try: process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid,signal.SIGKILL);process.wait()
            code = 124
    elapsed = time.monotonic()-start
    text = (directory/'vpr.out').read_text(errors='replace')
    status = classify_vpr_result(code, text)
    prefix = directory/'gp'
    gp_hash = sha(str(prefix)+'.blocks.tsv') if Path(str(prefix)+'.blocks.tsv').is_file() else None
    if not gp_hash or 'TAILCOMPILE_GP_EXPORTED' not in text:
        status = 'other_failure'
    if (replacement or deploy) and ('TAILCOMPILE_GP_REPLACED' not in text or gp_hash != expected_gp):
        status = 'other_failure'
    def metric(pattern):
        values = re.findall(pattern, text)
        return float(values[-1]) if values else None
    result = dict(signature=signature, status=status, returncode=code, wall_seconds=elapsed,
                  gp_sha256=gp_hash, command=command, phases=parse_vpr_log(text),
                  cpd=metric(r'Final critical path delay \(least slack\):\s*([\d.eE+-]+)'),
                  wirelength=metric(r'Total wirelength:\s*([\d.eE+-]+)'),
                  complete= status in ('success','unroutable'))
    write(record,result)
    emit('vpr', design=design, width=width, seed=seed, status=status, seconds=elapsed, path=str(directory))
    if not result['complete']:
        raise RuntimeError(f'VPR/hook failure ({code}): {directory}/vpr.out')
    return result

def screen(a):
    designs = read(a.out/'cohort.json')['designs']
    frozen(a.out/'protocol.json', dict(identity=a.identity, code=code_identity(), vpr_threads=a.vpr_threads,
        screen_widths=a.screen_widths, screen_seeds=[201,202], train_seeds=[211,212,213],
        test_seeds=[231,232,233,234,235], width_step=10, half_band=30, replicas=a.replicas))
    tasks = [(d,w,s) for d in designs for w in a.screen_widths for s in [201,202]]
    parallel(lambda t: run(a,t[0]['design'],t[1],t[2],a.out/'screen'/t[0]['design']/f'w{t[1]}'/f's{t[2]}'),tasks,a.jobs)
    plans = []
    for d in designs:
        passed = [w for w in a.screen_widths if any(read(a.out/'screen'/d['design']/f'w{w}'/f's{s}'/'trial.json')['status']=='success' for s in [201,202])]
        center = min(passed) if passed else max(a.screen_widths)
        plans.append(dict(d, widths=list(range(max(10,center-30),center+31,10)),
                          screen_right_censored=not bool(passed),
                          screen_left_censored=bool(passed and center==min(a.screen_widths))))
    frozen(a.out/'plan.json', dict(designs=plans, protocol_sha256=sha(a.out/'protocol.json')))

def contexts(a, split=None):
    for d in read(a.out/'plan.json')['designs']:
        if split and d['split'] not in split:
            continue
        seeds = [231,232,233,234,235] if d['split']=='test' else [211,212,213]
        for w in d['widths']:
            for seed in seeds:
                yield d,w,seed,a.out/'contexts'/d['design']/f'w{w}'/f's{seed}'

def pool(a):
    def one(t):
        d,w,seed,folder=t
        default = run(a,d['design'],w,seed,folder/'default')
        done = folder/'pool.json'
        if done.exists():
            if read(done)['gp_sha256'] != default['gp_sha256']:
                raise ValueError('pool GP signature changed')
            return
        subprocess.run([sys.executable,str(HERE/'build_pool.py'),'--folder',str(folder),
                        '--width',str(w),'--replicas',str(a.replicas)],check=True)
        emit('pool',design=d['design'],width=w,seed=seed,distinct=read(done)['unique_fplaces'])
    parallel(one,list(contexts(a)),a.jobs)

def label(a, testing=False):
    selected=read(a.out/'selection.json') if testing else None
    if testing and selected['model_sha256'] != sha(a.out/'model/ranker.pt'):
        raise ValueError('model changed after selection freeze')
    tasks=[]
    for d,w,seed,folder in contexts(a, {'test'} if testing else {'train','validation'}):
        p=read(folder/'pool.json')
        for action in p['actions']:
            tasks.append((d,w,seed,folder,action,p['gp_sha256']))
    def one(t):
        d,w,seed,folder,action,gp_hash=t
        name=action['name']
        replacement=folder/(name+'.fplace')
        if sha(replacement)!=action['sha256']:
            raise ValueError('candidate changed')
        result=run(a,d['design'],w,seed,folder/name,replacement,gp_hash)
        graph=read(folder/'graph.json')
        xy=np.load(folder/(name+'.npy'))
        checks={phase:retention(folder/'default'/phase,folder/name/phase,graph,xy)
                for phase in ['post_legalizer.fplace','post_route.fplace']}
        if name=='replay':
            original=flat_coordinates(folder/'default/post_route.fplace')
            checks['exact_post_route_replay'] = original is not None and original==flat_coordinates(folder/name/'post_route.fplace')
            checks['same_status']=read(folder/'default/trial.json')['status']==result['status']
        write(folder/name/'retention.json',checks)
        if name=='replay' and not (checks['exact_post_route_replay'] and checks['same_status']):
            raise RuntimeError(f'replay gate failed: {folder}; do not interpret action results')
    parallel(one,[t for t in tasks if t[4]['name']=='replay'],a.jobs)
    parallel(one,[t for t in tasks if t[4]['name']!='replay'],a.jobs)
    write(a.out/('evaluation_complete.json' if testing else 'labels_complete.json'),dict(trials=len(tasks),complete=True))

def deploy(a):
    selection=read(a.out/'selection.json')
    if selection['model_sha256']!=sha(a.out/'model/ranker.pt'):
        raise ValueError('model changed after freeze')
    choices={r['context']:r['action'] for r in selection['rows']}
    def one(t):
        d,w,s,folder=t
        gp=read(folder/'pool.json')['gp_sha256']
        result=run(a,d['design'],w,s,folder/'deployed',expected_gp=gp,deploy=True)
        actual=read(folder/'deployed/deployed.selection.json')['action']
        if actual!=choices[str(folder.relative_to(a.out))]:
            raise ValueError('deployment differs from frozen selection; inspect numeric/feature drift')
        return result
    parallel(one,list(contexts(a,{'test'})),a.jobs)
    write(a.out/'deployment_complete.json',dict(complete=True))

def timing(a):
    # Dedicated CPU timing, one task at a time, alternating order within paired trials.
    # Avoid attributing different concurrent workload interference to the method.
    choices={r['context']:r['action'] for r in read(a.out/'selection.json')['rows']}
    for d,w,s,folder in contexts(a,{'test'}):
        methods=['default','selected'] if s%2 else ['selected','default']
        for method in methods:
            directory=folder/'timing'/method
            run(a,d['design'],w,s,directory,expected_gp=read(folder/'pool.json')['gp_sha256'],deploy=method=='selected')
            if method=='selected' and read(directory/'deployed.selection.json')['action']!=choices[str(folder.relative_to(a.out))]:
                raise ValueError('timed deployment does not match frozen action')
    write(a.out/'timing_complete.json',dict(complete=True,concurrent_vpr_processes=1,alternating_order=True,
        caveat='Run on a reserved CPU instance; unrelated external jobs remain uncontrolled.'))

def recover(a):
    # Shell holds campaign lock. Remote-host incomplete runs need explicit user attestation.
    if not a.workers_stopped:
        raise ValueError('stop all old workers, then pass --workers-stopped; never recover an active run')
    targets=[]
    for path in a.out.rglob('started.json'):
        if '_quarantine' in path.parts: continue
        record=path.with_name('trial.json')
        if record.exists() and read(record).get('complete'): continue
        old=read(path)
        if old['host']==os.uname().nodename and any(Path(f'/proc/{pid}').exists() for pid in [old['pid'],old.get('child_pid',-1)]):
            raise ValueError(f'worker PID still exists: {old}')
        targets.append(path.parent)
    dest=a.out/'_quarantine'/time.strftime('%Y%m%d_%H%M%S')
    for folder in targets:
        resolved=folder.resolve()
        resolved.relative_to(a.out.resolve())
        target=dest/folder.relative_to(a.out)
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(folder),str(target))
    emit('recovered',count=len(targets),archive=str(dest))

def report(a):
    selection=read(a.out/'selection.json')
    selected={r['context']:r['action'] for r in selection['rows']}
    rows=[]
    replay=[]
    for d,w,seed,folder in contexts(a,{'test'}):
        p=read(folder/'pool.json')
        trials={name:read(folder/name/'trial.json') for name in ['default']+[x['name'] for x in p['actions']]}
        if not all(r['complete'] for r in trials.values()): raise ValueError('incomplete evaluation')
        replay.append(read(folder/'replay/retention.json'))
        key=str(folder.relative_to(a.out))
        pick=selected[key]
        # Oracle is test-label hindsight, never the deployable selector result.
        oracle=min(trials,key=lambda name:(trials[name]['status']!='success',trials[name]['wirelength'] or float('inf')))
        for method,name in [('default','default'),('replay','replay'),('selected',pick),('oracle_diagnostic',oracle)]:
            r=(read(folder/'timing'/method/'trial.json') if method in ('selected','default') else trials[name])
            # Actual synchronous deployment includes Python startup, candidate pool,
            # feature extraction, model load/inference and full VPR flow. No anchor-route credit.
            if not r['complete']: raise ValueError('deployment incomplete')
            cost=r['wall_seconds']
            rows.append(dict(design=d['design'],family=d['family'],width=w,seed=seed,method=method,action=name,
                             status=r['status'],runtime=cost,wirelength=r['wirelength'],cpd=r['cpd']))
    widths=[]
    for d in read(a.out/'plan.json')['designs']:
        if d['split']!='test': continue
        for method in ['default','replay','selected','oracle_diagnostic']:
            sub=[r for r in rows if r['design']==d['design'] and r['method']==method]
            byseed=[]
            for seed in sorted({r['seed'] for r in sub}):
                wins=[r['width'] for r in sub if r['seed']==seed and r['status']=='success']
                minimum=min(wins) if wins else None
                byseed.append(dict(seed=seed,sampled_wmin=minimum,left_censored=minimum==min(d['widths']),right_censored=minimum is None))
            rates={w:np.mean([r['status']=='success' for r in sub if r['width']==w]) for w in d['widths']}
            good=[w for w,v in rates.items() if v>=.8]
            widths.append(dict(design=d['design'],method=method,w80=min(good) if good else None,seeds=byseed,
                               success_rates={str(w):float(v) for w,v in rates.items()}))
    paired=[]
    for design in sorted({r['design'] for r in rows}):
        data=[r for r in rows if r['design']==design]
        baseline={(r['width'],r['seed']):r for r in data if r['method']=='default'}
        for method in ['replay','selected']:
            group=[(r,baseline[r['width'],r['seed']]) for r in data if r['method']==method]
            common=[(r,b) for r,b in group if r['status']==b['status']=='success']
            ratios={metric:math.exp(np.mean([math.log(r[metric]/b[metric]) for r,b in common if r[metric] and b[metric]]))
                    if any(r[metric] and b[metric] for r,b in common) else None for metric in ['runtime','wirelength','cpd']}
            paired.append(dict(design=design,method=method,pairs=len(group),joint_success=len(common),geomean_ratios=ratios))
    write(a.out/'evaluation_report.json',dict(complete=True,rows=rows,width_results=widths,paired=paired,
        replay_checks=replay,oracle_is_test_label_hindsight=True,grid_resolution=10,
        runtime_is_measured_single_process_with_worker=True,
        oracle_runtime_is_not_deployable=True,
        conclusion='Inspect per-family W80 and censoring before claiming benefit; no result is a guaranteed optimum.'))
    emit('report_complete',test_families=len({r['family'] for r in rows}),trials=len(rows))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['prepare','screen','pool','label','evaluate','deploy','timing','report','status','recover'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cohort',type=Path,default=HERE/'cohort.json')
    p.add_argument('--designs',nargs='+')
    p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--prepare-jobs',type=int,default=2)
    p.add_argument('--vpr-threads',type=int,default=1)
    p.add_argument('--timeout',type=int,default=7200)
    p.add_argument('--synthesis-timeout',type=int,default=14400)
    p.add_argument('--replicas',type=int,default=2)
    p.add_argument('--screen-widths',type=int,nargs='+',default=[40,80,120,160,200,240,280,320])
    p.add_argument('--workers-stopped',action='store_true')
    a=p.parse_args()
    a.out=a.out.resolve(); a.out.mkdir(parents=True,exist_ok=True)
    if a.phase=='status':
        trials=list(a.out.rglob('trial.json'))
        counts={}
        for path in trials:
            if '_quarantine' in path.parts: continue
            status=read(path)['status']; counts[status]=counts.get(status,0)+1
        emit('status',statuses=counts,labels_complete=(a.out/'labels_complete.json').exists(),
             trained=(a.out/'model/training_report.json').exists(),selected=(a.out/'selection.json').exists(),
             evaluation_complete=(a.out/'evaluation_complete.json').exists(),deployment_complete=(a.out/'deployment_complete.json').exists(),
             timing_complete=(a.out/'timing_complete.json').exists(),report_complete=(a.out/'evaluation_report.json').exists())
        return
    if a.phase not in ('report','recover'): environment(a)
    if a.phase=='evaluate': label(a,testing=True)
    else: globals()[a.phase](a)

if __name__=='__main__': main()
