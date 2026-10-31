"""At most 12 small P&R trials for 007 prior/zero-prior/native continuation.

Structure-only exports are recorded separately. Optional learned init replaces
the asymmetric fixture init, without adding deployment trials.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil
import xml.etree.ElementTree as ET
from architectures import SOURCE_ARCH, _module
from data import read_snapshot
from io_utils import freeze, read, run_trial, sha, vpr_command, write, prune_derived
from qualify import make_fixed_constraints
from repository_guard import validate_output_roots


def prepare_inputs(vtr_root, out):
    vtr_root, out = Path(vtr_root).resolve(), Path(out).resolve()
    validate_output_roots((out,), dependency_roots=(vtr_root,))
    cases = []
    folder=out/'inputs/single'
    folder.mkdir(parents=True,exist_ok=True)
    for src,name in ((vtr_root/'vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml','arch.xml'),
                     (vtr_root/'vtr_flow/benchmarks/blif/4/5xp1.blif','tiny.blif')):
        if not (folder/name).exists(): shutil.copy2(src,folder/name)
    cases.append(dict(name='single',arch=str(folder/'arch.xml'),blif=str(folder/'tiny.blif'),width=100,layers=1,
                      options=['--device_width','14']))
    fixtures=_module('heterogeneity_fixtures')
    for name,kind in (('balanced','aligned'),('asymmetric','asymmetric')):
        folder=out/'inputs'/name
        if not (folder/'fixture.json').exists(): fixtures.generate_case(vtr_root/SOURCE_ARCH,folder,kind)
        case=dict(name=name,arch=str(folder/'arch.xml'),blif=str(folder/'tiny.blif'),width=320,layers=2,
                  options=['--device','probe'])
        if name=='asymmetric':
            constraint=folder/'fixed.xml'
            if not constraint.exists():
                atom=next(line.split()[-1] for line in (folder/'tiny.blif').read_text().splitlines()
                          if line.startswith('.names ') and len(line.split())>=3)
                make_fixed_constraints(constraint,folder/'tiny.blif',atom)
            case['options']+=['--read_vpr_constraints',str(constraint)]
            case['constraints']=str(constraint)
        cases.append(case)
    write(out/'qualification_inputs.json',cases)
    return cases


def structure(binary, case, out, timeout):
    folder=out/'structures'/case['name']
    prefix=folder/'structure'
    folder.mkdir(parents=True,exist_ok=True)
    command=vpr_command(binary,case['arch'],case['blif'],case['width'],2701)+case['options']
    row=run_trial(folder/'export',command,inputs=[binary,case['arch'],case['blif']]+([case['constraints']] if case.get('constraints') else []),
        env=dict(TC_GP_STRUCTURE=prefix,TC_GP_STRUCTURE_ONLY='1'),marker='TAILCOMPILE_V7_STRUCTURE_ONLY_COMPLETE',
        timeout=timeout,metadata=dict(stage='layer_qualification_structure_only',case=case['name']))
    if row['status']!='complete': raise RuntimeError(f'structure export failed: {row["log"]}')
    return prefix


def write_prior(prefix,path):
    snapshot=read_snapshot(prefix,structure=True)
    w,h,layers=snapshot['size']
    rows=[]
    for index,block in enumerate(snapshot['blocks']):
        if block['fixed']: continue
        x=1.+(index*37%max(1,w-2))
        y=1.+(index*13%max(1,h-2))
        z=index%layers
        for atom in block['atoms']: rows.append(f'{atom}\t{x}\t{y}\t{z}\t0\n')
    path.write_text(''.join(rows))
    return snapshot


def run(binary,vtr_root,out,*,timeout=180,only=None,asymmetric_init=None,prepare_only=False):
    binary, out=Path(binary).resolve(),Path(out).resolve()
    validate_output_roots((out,),dependency_roots=(binary,Path(vtr_root)))
    out.mkdir(parents=True,exist_ok=True)
    cases=prepare_inputs(vtr_root,out)
    freeze(out/'qualification_protocol.json',dict(schema='tailcompile-layer-qualification-007',
        binary_sha256=sha(binary),matrix='3 fixtures x I/S x 2 seeds = 12 P&R maximum',
        seeds=[2701,2702],timeout=timeout,
        purpose='engineering correctness; no learned performance or generalization claim',
        source_sha256={name:sha(Path(__file__).with_name(name)) for name in
                       ('install_layer_hook.py','layer_prior_hook.inc','layer_route_hook.inc','layer_qualification.py')}))
    for case in cases:
        prefix=structure(binary,case,out,timeout)
        case['structure']=str(prefix)
        prior=out/'inputs'/case['name']/'initial.ap.tsv'
        if not prior.exists(): write_prior(prefix,prior)
        case['prior']=str(prior)
    write(out/'qualification_inputs.json',cases)
    if prepare_only: return dict(prepared=True,cases=cases)
    rows=[]
    for case in cases:
        if only and case['name'] not in only: continue
        prior=Path(asymmetric_init).resolve() if asymmetric_init and case['name']=='asymmetric' else Path(case['prior'])
        if asymmetric_init and case['name']=='asymmetric':
            case['learned_initializer_sha256']=sha(prior)
        for seed in (2701,2702):
            pair={}
            for mode in ('I','S'):
                folder=out/'deployments'/case['name']/f's{seed}'/mode
                folder.mkdir(parents=True,exist_ok=True)
                teacher=folder/'post_gp'
                command=vpr_command(binary,case['arch'],case['blif'],case['width'],seed)+case['options']
                env=dict(TC_GP_INIT=prior,TC_GP_LAYER_PRIOR=str(int(mode=='S')),
                         TC_GP_EXPORT=teacher,TC_GP_TRACE=folder/'injected',TC_GP_Z_USAGE=folder/'z_usage')
                inputs=[binary,case['arch'],case['blif'],prior]+([case['constraints']] if case.get('constraints') else [])
                row=run_trial(folder/'vpr',command,inputs=inputs,env=env,timeout=timeout,prune=False,
                              metadata=dict(stage='layer_prior_qualification',case=case['name'],mode=mode,seed=seed))
                text=Path(row['log']).read_text(errors='replace')
                schedule=[dict(iteration=int(i),factor=float(f),beta=float(b)) for i,f,b in re.findall(
                    r'TAILCOMPILE_V7_LAYER_PRIOR iteration=(\d+) factor=([^ ]+) beta=([^ ]+)',text)]
                expected=(not schedule if mode=='I' or case['layers']==1 else
                          len(schedule)>=3 and [s['factor'] for s in schedule[:3]]==[1.,.5,0.] and
                          'TAILCOMPILE_V7_LAYER_PRIOR_RELEASED native_exit=1' in text)
                fixed_source=read_snapshot(case['structure'],structure=True)
                placed=read_snapshot(teacher)
                fixed_by_atom={a:b for b in fixed_source['blocks'] if b['fixed'] for a in b['atoms']}
                fixed_ok=all(not any(a in fixed_by_atom for a in b['atoms']) or
                             all(all(abs(v-fixed_by_atom[a]['loc'][axis])<1e-7
                                     for axis,v in enumerate(b['loc'][:3]) if fixed_by_atom[a]['loc'][axis]>=0)
                                 for a in b['atoms'] if a in fixed_by_atom) for b in placed['blocks'])
                check=dict(case=case['name'],mode=mode,seed=seed,status=row['status'],metrics=row['metrics'],
                    binary_sha256=sha(binary),initial_sha256=sha(prior),
                    init_consumed='TAILCOMPILE_V7_GP_INIT_CONSUMED' in text,
                    native_continued='TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' in text,
                    prior_schedule=schedule,schedule_passed=bool(expected),fixed_passed=fixed_ok,
                    vertical_usage_present=Path(str(folder/'z_usage')+'.first.tsv').exists(),
                    post_gp_sha256=sha(str(teacher)+'.blocks.tsv'),log=row['log'])
                check['passed']=row['status']=='success' and check['init_consumed'] and check['native_continued'] and expected and fixed_ok and check['vertical_usage_present']
                pair[mode]=read_snapshot(teacher)
                rows.append(check)
                write(folder/'qualification_check.json',check)
                if row['status'] in ('success','unroutable','complete'): prune_derived(row['workdir'],protected=inputs)
            if case['layers']==1:
                equal=pair['I']==pair['S']
                for check in rows[-2:]:
                    check['single_layer_zero_prior_identical']=equal
                    check['passed'] &= equal
    # Preserve checks from earlier --only calls; never hide unexecuted conditions.
    indexed={(r['case'],r['seed'],r['mode']):r for r in rows}
    for path in (out/'deployments').glob('*/s*/*/qualification_check.json'):
        r=read(path); indexed.setdefault((r['case'],r['seed'],r['mode']),r)
    all_rows=[indexed[k] for k in sorted(indexed)]
    report=dict(schema='tailcompile-layer-qualification-007',binary=str(binary),binary_sha256=sha(binary),
        passed=len(all_rows)==12 and all(r['passed'] for r in all_rows),complete=len(all_rows)==12,
        pnr_count=len(all_rows),maximum_pnr_count=12,structure_only_exports=3,
        research_performance_evaluated=False,cases=cases,trials=all_rows)
    write(out/'qualification_report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--binary',type=Path,required=True)
    p.add_argument('--vtr-root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--timeout',type=int,default=180)
    p.add_argument('--only',nargs='+',choices=['single','balanced','asymmetric'])
    p.add_argument('--asymmetric-init',type=Path)
    p.add_argument('--prepare-only',action='store_true')
    a=p.parse_args()
    result=run(a.binary,a.vtr_root,a.out,timeout=a.timeout,only=a.only,
               asymmetric_init=a.asymmetric_init,prepare_only=a.prepare_only)
    print(json.dumps({k:v for k,v in result.items() if k not in ('trials','cases')}))
    if not a.prepare_only and not a.only and not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
