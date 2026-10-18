"""A Default / B GNN / C Default+refine / D GNN+refine paired experiment.

The same hook runs after the GP-or-read_flat_place branch, before legalization.
Per-child environments avoid races between parallel trials. Worker time is
already inside VPR wall time and must NEVER be added a second time.
"""
import argparse
import concurrent.futures as cf
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import statistics
import subprocess
import sys
import time

import direct
from direct import read, write, sha, freeze, SCOPES, overhead, width_result
from run_tailcompile_v4 import classify_vpr_result
from analyze_tailcompile_runtime import parse_vpr_log

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'tailcompile_v66'))
from install_hook import verify

METHODS = ('A_default', 'B_gnn', 'C_default_refine', 'D_gnn_refine')
PAIRS = [('B_gnn','A_default'), ('C_default_refine','A_default'),
         ('D_gnn_refine','B_gnn'), ('D_gnn_refine','C_default_refine'),
         ('D_gnn_refine','A_default')]


def code_identity():
    files = [HERE/'four_routes.py', HERE/'refine.py', HERE/'direct.py',
             HERE.parent/'tailcompile_v3.py', HERE.parent/'run_tailcompile_v4.py',
             HERE.parent/'tailcompile_v66/core.py', HERE.parent/'tailcompile_v66/gp_hook.inc']
    return {str(p.relative_to(HERE.parent)): sha(p) for p in files}


def environment(folder, method, width):
    env = os.environ.copy()
    for k in ('TC_GP_EXPORT', 'TC_GP_REPLACE', 'TC_GP_WORKER'):
        env.pop(k, None)
    for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        env[k] = '1'
    if method in METHODS[2:]:
        env['TC_GP_EXPORT'] = str(folder/'coarse')
        env['TC_GP_REPLACE'] = str(folder/'refined.fplace')
        env['TC_GP_WORKER'] = shlex.join([sys.executable, str(HERE/'refine.py'),
            '--prefix', str(folder/'coarse'), '--output', str(folder/'refined.fplace'),
            '--width', str(width)])
    return env


def command(backend, c, seed, method, folder):
    cmd = [backend.binary, backend.arch, c['blif'], '--analytical_place', '--route', '--analysis',
           '--ap_full_legalizer', 'appack', '--ap_detailed_placer', 'annealer',
           '--seed', str(seed), '--route_chan_width', str(c['width']),
           '--device_width', str(c['device_width']), '--num_workers', '1',
           '--net_file', str(folder/'input.net'), '--flat_place_verbosity', '1',
           '--write_flat_place', str(folder/'post_route.fplace'),
           '--write_legalized_flat_place', str(folder/'post_legalizer.fplace')]
    if method in ('B_gnn','D_gnn_refine'):
        cmd += ['--read_flat_place', c['fplace']]
    return cmd


def run_trial(a, backend, identity, task):
    c, seed, method, folder = task
    sig = dict(identity=identity, sources=c['sources'], seed=seed, width=c['width'],
               device_width=c['device_width'], method=method, timeout=a.timeout,
               fplace= c['fplace_sha256'] if method in ('B_gnn','D_gnn_refine') else None)
    trial = folder/'trial.json'
    if trial.exists():
        old = read(trial)
        if old['signature'] != sig or old['status'] not in ('success','unroutable'):
            raise ValueError(f'changed/failed trial: {folder}; preserve it for diagnosis')
        return old
    if folder.exists() and any(folder.iterdir()):
        raise ValueError(f'incomplete trial: {folder}; preserve it for diagnosis')
    folder.mkdir(parents=True, exist_ok=True)
    write(folder/'started.json', sig)
    shutil.copy2(c['packed'], folder/'input.net')
    cmd = command(backend,c,seed,method,folder)
    env = environment(folder,method,c['width'])
    start = time.perf_counter()
    with (folder/'vpr.out').open('w') as log:
        proc = subprocess.Popen(cmd,cwd=folder,env=env,stdout=log,stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=a.timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = 124
    elapsed = time.perf_counter()-start
    text = (folder/'vpr.out').read_text(errors='replace')
    status = classify_vpr_result(code,text)
    selection = None
    if method in METHODS[2:]:
        p = folder/'refined.selection.json'
        if p.exists(): selection = read(p)
        if 'TAILCOMPILE_GP_REPLACED' not in text or selection is None:
            status = 'other_failure'  # stale binary / worker failure is NOT a routing failure
    def metric(pattern):
        values = re.findall(pattern,text)
        return float(values[-1]) if values else None
    result = dict(signature=sig,design=c['design'],width=c['width'],seed=seed,method=method,
        status=status,process_returncode=code,wall_seconds=elapsed,command=cmd,
        phases=parse_vpr_log(text),local_refinement=selection,
        refinement_already_in_wall_time=True,
        wirelength=metric(r'Total wirelength:\s*([\d.eE+-]+)'),
        cpd=metric(r'Final critical path delay \(least slack\):\s*([\d.eE+-]+)'),
        hook_env={k:v for k,v in env.items() if k.startswith('TC_GP_')})
    write(trial,result)
    print(f'{method} {c["design"]} w{c["width"]} s{seed}: {status} {elapsed:.3f}s',flush=True)
    if status not in ('success','unroutable'):
        raise RuntimeError(f'VPR/worker failed ({code}): {folder}/vpr.out')
    return result


def tasks(a, plan, timing):
    for c in plan['cases']:
        for seed in plan['seeds']:
            offset = seed % 4
            order = METHODS[offset:]+METHODS[:offset]
            for method in order:
                yield c,seed,method,a.out/('timing' if timing else 'runs')/c['design']/f'w{c["width"]}'/f's{seed}'/method


def execute(a, timing=False):
    if os.name != 'posix': raise RuntimeError('VPR execution requires Linux/WSL')
    plan = read(a.out/'plan.json')
    if plan['signature']['code'] != direct.implementation(): raise ValueError('generator code changed')
    backend = direct.DirectBackend(os.environ['VTR_ROOT'])
    hook = verify(Path(backend.root))
    identity = dict(binary=backend.binary_hash,arch=backend.arch_hash,hook=hook,
                    code=code_identity(),threads=1)
    freeze(a.out/'abcd_protocol.json',dict(identity=identity,timeout=a.timeout,plan_sha256=sha(a.out/'plan.json')))
    for c in plan['cases']:
        for path, digest in [(c['blif'],c['sources']['mapped_blif']),
                             (c['packed'],c['sources']['packed_net']),(c['fplace'],c['fplace_sha256'])]:
            if sha(path) != digest: raise ValueError(f'frozen input changed: {path}')
    pending = list(tasks(a,plan,timing))
    # Preflight C AND D sequentially: an unrebuilt hook must fail before batch launch.
    for method in METHODS[2:]:
        run_trial(a,backend,identity,next(t for t in pending if t[2]==method))
    failures=[]
    with cf.ThreadPoolExecutor(max_workers=1 if timing else a.jobs) as executor:
        futures=[executor.submit(run_trial,a,backend,identity,t) for t in pending]
        for future in cf.as_completed(futures):
            try: future.result()
            except Exception as e:
                failures.append(str(e));print(str(e),flush=True)
    if failures: raise RuntimeError(f'{len(failures)} failed trials. First: {failures[0]}')


def total(row, timing, scope):
    # Worker startup, graph construction, search and lowering are inside VPR wall.
    return row['wall_seconds'] + (overhead(timing,scope) if row['method'] in ('B_gnn','D_gnn_refine') else 0)


def report(a):
    plan=read(a.out/'plan.json'); widths=[]; comparisons=[]
    def load(phase,c,seed,method):
        row=read(a.out/phase/c['design']/f'w{c["width"]}'/f's{seed}'/method/'trial.json')
        if row['status'] not in ('success','unroutable'): raise ValueError('infrastructure failure is not a label')
        return row
    for design in sorted({c['design'] for c in plan['cases']}):
        cases=[c for c in plan['cases'] if c['design']==design]
        grid=sorted(c['width'] for c in cases)
        for method in METHODS:
            per_seed={str(s):width_result(grid,[load('runs',c,s,method) for c in cases]) for s in plan['seeds']}
            rates={c['width']:sum(load('runs',c,s,method)['status']=='success' for s in plan['seeds'])/len(plan['seeds']) for c in cases}
            widths.append(dict(design=design,method=method,per_seed=per_seed,success_rates=rates,
                               w80=min((w for w,r in rates.items() if r>=.8),default=None)))
        for candidate,baseline in PAIRS:
            rows=[]
            for c in cases:
                t=read(c['generation'])['timing']
                for s in plan['seeds']:
                    b=load('timing',c,s,baseline);r=load('timing',c,s,candidate)
                    if b['status']==r['status']=='success':
                        rows.append(dict(width=c['width'],seed=s,
                            speedup={scope:total(b,t,scope)/total(r,t,scope) for scope in SCOPES},
                            wirelength_ratio=r['wirelength']/b['wirelength'] if r['wirelength'] and b['wirelength'] else None,
                            cpd_ratio=r['cpd']/b['cpd'] if r['cpd'] and b['cpd'] else None))
            comparisons.append(dict(design=design,candidate=candidate,baseline=baseline,
                joint_success_pairs=len(rows),pairs=rows,speedup={scope:
                math.exp(statistics.mean(math.log(r['speedup'][scope]) for r in rows)) if rows else None for scope in SCOPES}))
    write(a.out/'evaluation_report.json',dict(complete=True,methods=METHODS,width_results=widths,
        runtime_comparisons=comparisons,counted_components=SCOPES,
        refinement_cost='Included in C/D VPR wall; never added twice',full_cold_from_raw_rrg=None,
        speedup_direction='baseline / candidate; >1 faster',
        policy='Frozen label-free RUDY-tail + HPWL local search, NOT trained v66 selector'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['generate','run','timing','report','status'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,required=True)
    p.add_argument('--screen',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--prior-weight',type=float,default=.35)
    p.add_argument('--seeds',type=int,nargs='+',default=[251,252,253,254,255])
    p.add_argument('--widths',type=int,nargs='+');p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--timeout',type=int,default=7200)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if a.jobs<1 or a.timeout<=0: p.error('jobs/timeout must be positive')
    if len(a.seeds)<3 or len(set(a.seeds))!=len(a.seeds):p.error('need >=3 distinct seeds')
    if a.widths and (len(a.widths)<3 or a.widths!=sorted(set(a.widths)) or min(a.widths)<=0):p.error('need >=3 sorted positive widths')
    if a.phase=='generate': direct.generate(a)
    elif a.phase in ('run','timing'):execute(a,a.phase=='timing')
    elif a.phase=='report':report(a)
    else:
        plan=read(a.out/'plan.json') if (a.out/'plan.json').exists() else None
        expected=len(plan['cases'])*len(plan['seeds'])*4 if plan else None
        counts={phase:sum(read(p).get('status') in ('success','unroutable') for p in (a.out/phase).rglob('trial.json')) for phase in ('runs','timing')}
        print(dict(expected_per_phase=expected,valid_trials=counts,
                   complete=expected is not None and all(n==expected for n in counts.values()) and (a.out/'evaluation_report.json').exists()))


if __name__=='__main__':main()
