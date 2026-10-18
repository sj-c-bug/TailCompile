"""GNN-only vs Default, with explicit reusable-device timing accounting.

No random candidates, swaps, post-GP worker or candidate selection. Uses the
original trained TwoTowerPlacementGNN and native VPR --read_flat_place.
"""
import argparse
import concurrent.futures as cf
import json
import math
import os
import platform
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
from run_tailcompile_v64_wmin import read,write,sha,width_result
from run_tailcompile_v4 import Backend,run_one
from tailcompile_v4_risk import VTR_COMMIT
from analyze_tailcompile_runtime import parse_vpr_log

SCOPES={
    'warm_inference_lowering': ['tensor_transfer','logic_forward','matching_and_coordinate','capacity_decode','fplace_lowering'],
    'warm_operational': ['logic_ir_load','graph_features','tensor_transfer','logic_forward','matching_and_coordinate','capacity_decode','fplace_lowering'],
    'cold_from_prepared_ir': ['model_load','logic_ir_load','device_ir_load','graph_features','tensor_transfer','rrg_encoding',
                              'logic_forward','matching_and_coordinate','capacity_decode','fplace_lowering'],
}

def overhead(timing,scope):
    return sum(timing[k] for k in SCOPES[scope])

def freeze(path,data):
    if path.exists() and read(path)!=data: raise ValueError(f'changed inputs: {path}; use a new output directory')
    if not path.exists(): write(path,data)

def implementation():
    return {str(p.name):sha(p) for p in [Path(__file__),HERE.parent/'tailcompile_v3.py',
                                       HERE.parent/'run_tailcompile_v4.py']}

def generate(a):
    import torch
    from tailcompile_v3 import TwoTowerPlacementGNN,graph_inputs,add_partition_prior,min_cost_decode,lower_fplace
    torch.set_num_threads(1)
    device=torch.device(a.device)
    def measured(fn):
        if device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();value=fn()
        if device.type=='cuda':torch.cuda.synchronize()
        return value,time.perf_counter()-start
    checkpoint,load_seconds=measured(lambda:torch.load(a.checkpoint,map_location=device,weights_only=True))
    if 'model_state' not in checkpoint:raise ValueError('requires original generator checkpoint, not a v6 candidate scorer')
    conditioned=bool(checkpoint.get('conditioned_on_channel_width',False))
    config=read(a.inputs)
    if config['vtr_commit']!=VTR_COMMIT:raise ValueError('input VTR cohort mismatch')
    signature=dict(inputs_sha256=sha(a.inputs),checkpoint_sha256=sha(a.checkpoint),code=implementation(),
                   device=str(device),torch=torch.__version__,prior_weight=a.prior_weight,seeds=a.seeds,
                   screen_root=str(a.screen.resolve()),grid=a.widths)
    freeze(a.out/'signature.json',signature)
    cases=[]
    # In-memory cache key includes checkpoint, device IR and width conditioning.
    cache={}
    for record in config['designs']:
        design=record['design']
        if design in checkpoint.get('training_designs',[]) or design in checkpoint.get('validation_designs',[]):
            raise ValueError(f'{design} participated in checkpoint development; use a separately declared development cohort')
        if a.widths:
            widths=a.widths
        else:
            wins=[]
            for m in (a.screen/design).glob('w*/default_s*/manifest.json'):
                r=read(m)
                if r.get('vpr_status')=='success':wins.append(int(r['route_chan_width']))
            if not wins:raise ValueError(f'no completed Default screen for {design}; provide --widths explicitly')
            center=min(wins);widths=list(range(max(10,center-30),center+31,10))
        for width in widths:
            folder=a.out/'generation'/design/f'w{width}';folder.mkdir(parents=True,exist_ok=True)
            manifest=folder/'generation.json'
            sources={key:sha(Path(record[key])) for key in ['logic_ir','device_ir','mapped_blif','packed_net']}
            context_signature=dict(signature,sources=sources,width=width,design=design)
            if manifest.exists():
                result=read(manifest)
                if result['signature']!=context_signature or sha(folder/'gnn.fplace')!=result['fplace_sha256']:
                    raise ValueError('generation provenance changed')
            else:
                t={}
                logic,t['logic_ir_load']=measured(lambda:read(record['logic_ir']))
                physical,t['device_ir_load']=measured(lambda:read(record['device_ir']))
                if logic['mapped_blif_sha256']!=sources['mapped_blif'] or logic['packed_net_sha256']!=sources['packed_net']:
                    raise ValueError('IR and fixed netlist differ')
                if physical['region_grid']['count']!=16 or logic['scale']!=16:
                    raise ValueError('original generator baseline requires original matching 4x4 IR, not tile8 scorer IR')
                data,t['graph_features']=measured(lambda:graph_inputs(logic,physical,Path(record['mapped_blif']),width if conditioned else None))
                data,t['tensor_transfer']=measured(lambda:{k:v.to(device) if torch.is_tensor(v) else v for k,v in data.items()})
                def initialize():
                    model=TwoTowerPlacementGNN(data['cluster_x'].shape[1],data['region_x'].shape[1],checkpoint['hidden_width']).to(device)
                    model.load_state_dict(checkpoint['model_state']);model.eval();return model
                model,initialize_seconds=measured(initialize)
                t['model_load']=load_seconds+initialize_seconds
                cache_key=(sources['device_ir'],sha(a.checkpoint),width if conditioned else None,str(device))
                with torch.no_grad():
                    if cache_key not in cache:
                        def encode():
                            region=model.rrg_tower(data['region_x'],data['region_src'],data['region_dst'],data['region_weight'])
                            return model.region_projection(region)
                        region,seconds=measured(encode);cache[cache_key]=(region,seconds)
                        hit=False
                    else:hit=True
                    region,t['rrg_encoding']=cache[cache_key]
                    embedding,t['logic_forward']=measured(lambda:model.logic_tower(data['cluster_x'],data['logic_src'],data['logic_dst'],data['logic_weight']))
                    def heads():
                        scores=model.logic_projection(embedding) @ region.T / math.sqrt(embedding.shape[1])
                        xy=model.coordinate_head(embedding)
                        return add_partition_prior(scores,xy,logic,physical,a.prior_weight),xy
                    (scores,xy),t['matching_and_coordinate']=measured(heads)
                    (assignment,audit),t['capacity_decode']=measured(lambda:min_cost_decode(scores,logic,physical))
                    lower,t['fplace_lowering']=measured(lambda:lower_fplace(logic,physical,assignment,xy,folder/'gnn.fplace'))
                result=dict(signature=context_signature,timing=t,fplace_sha256=lower['sha256'],rrg_cache_hit=hit,
                    hardware=dict(host=platform.node(),platform=platform.platform(),torch_threads=torch.get_num_threads(),
                                  inference_device=torch.cuda.get_device_name(device) if device.type=='cuda' else 'cpu'),
                    raw_rrg_to_ir_seconds=None,packing_seconds=None,process_startup_seconds=None,
                    scope_seconds={scope:overhead(t,scope) for scope in SCOPES},
                    operations='one trained forward + partition prior + capacity decode + lowering; no local search',
                    rrg_cache_key=list(cache_key))
                write(manifest,result)
                print(json.dumps(dict(stage='generated',design=design,width=width,timing=t)),flush=True)
            cases.append(dict(design=design,width=width,blif=record['mapped_blif'],packed=record['packed_net'],
                sources=sources,device_width=int(record['device_width']),fplace=str(folder/'gnn.fplace'),
                fplace_sha256=result['fplace_sha256'],generation=str(manifest)))
    freeze(a.out/'plan.json',dict(signature=signature,cases=cases,seeds=a.seeds,
        note='Existing six development benchmarks; no claim of untouched-family validation. Same seeds and widths for both methods.'))

class DirectBackend(Backend):
    def __init__(self,root):
        self.root=str(Path(root).resolve());self.distro=None
        if self.call(['git','-C',self.root,'rev-parse','HEAD']).stdout.strip()!=VTR_COMMIT:raise ValueError('VTR version mismatch')
        changes=self.call(['git','-C',self.root,'diff','--name-only','--ignore-submodules','HEAD']).stdout.strip()
        if changes:
            sys.path.insert(0,str(HERE.parent/'tailcompile_v66'))
            from install_hook import verify
            verify(Path(root)) # accept only the exact audited hook, inactive below
        self.binary=self.root+'/vpr/vpr'
        self.arch=self.root+'/vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml'
        self.binary_hash=sha(self.binary);self.arch_hash=sha(self.arch)
        self.call([self.binary,'--help'])

    def command(self,args,cwd=None):
        command=super().command(args,cwd)
        if command[0]==getattr(self,'binary',None) and '--analytical_place' in command:command+=['--num_workers','1']
        return command

def tasks(a,plan,timing=False):
    for case in plan['cases']:
        for seed in plan['seeds']:
            for method in (['default','gnn_only'] if seed%2 else ['gnn_only','default']):
                directory=a.out/('timing' if timing else 'runs')/case['design']/f'w{case["width"]}'/f's{seed}'/method
                yield case,seed,method,directory

def execute(a,timing=False):
    for key in ['TC_GP_EXPORT','TC_GP_REPLACE','TC_GP_WORKER']:os.environ.pop(key,None)
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
    plan=read(a.out/'plan.json')
    if plan['signature']['code']!=implementation():raise ValueError('implementation changed; use new output')
    backend=DirectBackend(os.environ['VTR_ROOT'])
    freeze(a.out/'vpr_identity.json',dict(commit=VTR_COMMIT,binary=backend.binary_hash,arch=backend.arch_hash,threads=1,timeout=a.timeout))
    for c in plan['cases']:
        if sha(c['blif'])!=c['sources']['mapped_blif'] or sha(c['packed'])!=c['sources']['packed_net'] or sha(c['fplace'])!=c['fplace_sha256']:
            raise ValueError('frozen inputs changed')
    def one(task):
        c,seed,method,folder=task
        r=run_one(backend,folder,Path(c['blif']),Path(c['packed']),c['width'],c['device_width'],seed,
                  Path(c['fplace']) if method=='gnn_only' else None,timeout=a.timeout,full_legalizer='appack')
        text=(folder/'vpr.out').read_text(errors='replace')
        def metric(pattern):
            values=re.findall(pattern,text);return float(values[-1]) if values else None
        write(folder/'trial.json',dict(design=c['design'],width=c['width'],seed=seed,method=method,status=r['vpr_status'],
            wall_seconds=r['wall_seconds'],phases=parse_vpr_log(text),
            wirelength=metric(r'Total wirelength:\s*([\d.eE+-]+)'),
            cpd=metric(r'Final critical path delay \(least slack\):\s*([\d.eE+-]+)'),
            actual_num_workers=1,manifest_sha256=sha(folder/'manifest.json')))
    # Timing intentionally serial, alternating within each pair. Throughput runs parallel.
    with cf.ThreadPoolExecutor(max_workers=1 if timing else a.jobs) as executor:
        futures=[executor.submit(one,t) for t in tasks(a,plan,timing)]
        failures=[]
        for future in cf.as_completed(futures):
            try:future.result()
            except Exception as error:
                failures.append(str(error));print(json.dumps(dict(stage='failed',error=str(error))),flush=True)
    if failures:raise RuntimeError(f'{len(failures)} failures; completed trials retained. First: {failures[0]}')

def report(a):
    plan=read(a.out/'plan.json');rows=[];widths=[];ratios=[]
    for design in sorted({c['design'] for c in plan['cases']}):
        cases=[c for c in plan['cases'] if c['design']==design]
        grid=sorted(c['width'] for c in cases)
        for method in ['default','gnn_only']:
            per_seed={}
            for seed in plan['seeds']:
                samples=[read(a.out/'runs'/design/f'w{w}'/f's{seed}'/method/'trial.json') for w in grid]
                per_seed[str(seed)]=width_result(grid,samples)
            rates={w:sum(read(a.out/'runs'/design/f'w{w}'/f's{s}'/method/'trial.json')['status']=='success' for s in plan['seeds'])/len(plan['seeds']) for w in grid}
            acceptable=[w for w in grid if rates[w]>=.8]
            widths.append(dict(design=design,method=method,per_seed=per_seed,w80=min(acceptable) if acceptable else None,success_rates=rates))
        for c in cases:
            timing=read(c['generation'])['timing']
            for seed in plan['seeds']:
                path=a.out/'timing'/design/f'w{c["width"]}'/f's{seed}'
                baseline=read(path/'default/trial.json');gnn=read(path/'gnn_only/trial.json')
                row=dict(design=design,width=c['width'],seed=seed,default=baseline,gnn_only=gnn,
                    components=timing,scope_totals={scope:gnn['wall_seconds']+overhead(timing,scope) for scope in SCOPES},
                    full_cold_from_raw_rrg=None)
                rows.append(row)
        paired=[r for r in rows if r['design']==design and r['default']['status']==r['gnn_only']['status']=='success']
        ratios.append(dict(design=design,joint_success_pairs=len(paired),speedup={scope:
            math.exp(statistics.mean(math.log(r['default']['wall_seconds']/r['scope_totals'][scope]) for r in paired)) if paired else None for scope in SCOPES},
            qor_gnn_over_default={key:math.exp(statistics.mean(math.log(r['gnn_only'][key]/r['default'][key]) for r in paired if r['gnn_only'][key] and r['default'][key]))
                if any(r['gnn_only'][key] and r['default'][key] for r in paired) else None for key in ['wirelength','cpd']}))
    write(a.out/'evaluation_report.json',dict(complete=True,width_results=widths,runtime_comparisons=ratios,paired_rows=rows,
        counted_components=SCOPES,rrg_preprocessing_not_measured=True,speedup_direction='Default / method; >1 faster',
        caveat='Cold-from-prepared-IR is NOT cold from raw RRG/RTL. Warm primary excludes feature preparation. No success claim from failed-run speed.'))

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['generate','run','timing','report','status'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--inputs',type=Path,default=HERE.parent/'tailcompile_v5/runs/holdout_001/inputs/holdout_inputs.json')
    p.add_argument('--screen',type=Path,default=HERE.parent/'tailcompile_v5/runs/holdout_001/screen/runs')
    p.add_argument('--checkpoint',type=Path,default=HERE.parent/'tailcompile_v3_training/model_width_conditioned/ranker.pt')
    p.add_argument('--device',default='cpu')
    p.add_argument('--prior-weight',type=float,default=.35)
    p.add_argument('--seeds',type=int,nargs='+',default=[251,252,253,254,255])
    p.add_argument('--widths',type=int,nargs='+')
    p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--timeout',type=int,default=7200)
    a=p.parse_args();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    if a.widths and (len(a.widths)<3 or a.widths!=sorted(set(a.widths)) or min(a.widths)<=0):p.error('width grid must have >=3 sorted positive distinct widths')
    if len(a.seeds)<3 or len(a.seeds)!=len(set(a.seeds)):p.error('need >=3 distinct seeds')
    if a.phase in ['run','timing']:execute(a,a.phase=='timing')
    elif a.phase=='status':
        print(json.dumps(dict(plan=(a.out/'plan.json').exists(),trials=len(list((a.out/'runs').rglob('trial.json'))),
            timing_trials=len(list((a.out/'timing').rglob('trial.json'))),complete=(a.out/'evaluation_report.json').exists())))
    else:globals()[a.phase](a)

if __name__=='__main__':main()
