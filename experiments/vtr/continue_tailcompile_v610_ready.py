"""Continue qualified v6.10 designs while explicitly retaining timeout exclusions.

This pre-evaluation migration preserves the complete inventory and every trial.
Only recorded preparation timeouts may be deferred; no timeout becomes an
unroutable label or a successful benchmark. Final budgets and weights stay frozen.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import time

import retry_tailcompile_v610_preparation as retry
from retry_tailcompile_v610_preparation import before_final_preparation, cohort_lock

REPO=Path(__file__).resolve().parents[2]
SCHEMA='tailcompile-v610-preparation-admission-1'
JOURNAL_SCHEMA='tailcompile-v610-preparation-admission-repair-1'
RELEASED=dict(retry.RELEASED,**{
    'experiments/vtr/tailcompile_v610/run.py':
        '8818c709e019643910fac1a5ae7a0f5759f3b6b8a8019d636a3b90f43c46fa6a',
})
ADDED={retry.ADDED,'experiments/vtr/tailcompile_v610/preparation_admission.py'}
read,write,sha,fingerprint=retry.read,retry.write,retry.sha,retry.fingerprint


def validate_root(path):
    out=Path(path).resolve()
    relative=out.relative_to(REPO/'experiments/vtr/tailcompile_v610/runs')
    if len(relative.parts)!=1 or not out.is_dir():
        raise ValueError('Select one existing v610 cohort')
    return out


def verify_transition(old,new):
    if (old.get('schema')!='tailcompile-v610-frozen-sweep-1' or
            old.get('training_permitted') is not False or not old.get('preparation_policy')):
        raise ValueError('Expected the released frozen recovery preparation protocol')
    if {k:v for k,v in old.items() if k!='code'}!={k:v for k,v in new.items() if k!='code'}:
        raise ValueError('Admission cannot change scientific settings, budgets or checkpoint')
    current=retry.current_implementation()
    previous=old['code']
    if new['code']!=current or set(current)-set(previous)!=ADDED or set(previous)-set(current):
        raise ValueError('Unexpected frozen implementation additions/removals')
    for name,digest in previous.items():
        if name in RELEASED:
            if digest!=RELEASED[name]:
                raise ValueError(f'Unrecognized released preparation implementation: {name}')
        elif current[name]!=digest:
            raise ValueError(f'Unrelated frozen implementation changed: {name}')
    for name,digest in current.items():
        path=(REPO/name).resolve();path.relative_to(REPO)
        if not path.is_file() or sha(path)!=digest:
            raise ValueError(f'Current admission implementation changed: {name}')
    if sha(old['checkpoint'])!=old['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    retry.verify_source(old)


def _deferred(out,protocol,report,inventory,names):
    if len(set(names))!=len(names) or not names:
        raise ValueError('Name each deferred benchmark exactly once')
    if any(not re.fullmatch(r'[A-Za-z0-9_.-]+',name) for name in names):
        raise ValueError('Unsafe deferred benchmark name')
    known={row['design'] for row in inventory['designs']}
    if len(known)!=len(inventory['designs']) or not set(names)<=known:
        raise ValueError('Deferred benchmark is missing from the frozen inventory')
    failures=report.get('failures',[])
    if len(failures)!=len(names) or {row.get('design') for row in failures}!=set(names):
        raise ValueError('Explicit deferrals must match all current preparation failures exactly')
    result=[]
    for failure in failures:
        name=failure['design']
        if failure.get('status')!='timeout' or not failure.get('directory'):
            raise ValueError(f'Only a recorded preparation timeout can be deferred: {name}')
        folder=Path(failure['directory']).resolve()
        relative=folder.relative_to(out)
        path=folder/'trial.json'
        trial=read(path)
        identity=trial.get('identity',{})
        stage='calibration' if identity.get('stage')=='route' else identity.get('stage')
        if stage=='synthesis':
            valid_path=relative.parts==('inputs',name,'synthesis')
        elif stage=='calibration':
            valid_path=(len(relative.parts)==6 and relative.parts[:2]==('screen',name)
                and re.fullmatch(r'g\d+x\d+',relative.parts[2]) is not None
                and relative.parts[3]=='calibration' and re.fullmatch(r'w\d+',relative.parts[4]) is not None
                and relative.parts[5] in {f's{s}' for s in protocol['calibration_seeds']})
        else:
            valid_path=False
        if (not valid_path or trial.get('status')!='timeout' or trial.get('returncode')!=124 or
                identity.get('design')!=name or not isinstance(trial.get('command'),list) or
                not trial['command'] or trial['command']!=identity.get('command') or
                any(identity.get(k)!=v for k,v in protocol['identity'].items())):
            raise ValueError(f'Preparation timeout stage/backend evidence differs: {folder}')
        timeout=trial.get('timeout')
        if (isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or not math.isfinite(timeout) or
                timeout!=protocol['preparation_policy'][stage+'_timeout']):
            raise ValueError('Recorded preparation timeout differs from the frozen budget')
        if stage=='calibration' and (identity.get('teacher') or identity.get('initializer_sha256') is not None):
            raise ValueError('Deferred calibration must be uninitialized Default')
        result.append(dict(design=name,stage=stage,status='timeout',directory=str(folder),
                           trial_sha256=sha(path),timeout=timeout))
    return sorted(result,key=lambda row:row['design'])


def _ready_coverage(out,inventory,deferred):
    prepared=read(out/'prepared_inputs.json')['designs']
    partial=read(out/'contexts.partial.json')['contexts'] if (out/'contexts.partial.json').exists() else []
    ready={row['design'] for row in prepared}
    cached={row['design'] for row in partial}
    known={row['design'] for row in inventory['designs']}
    pending={row['design'] for row in deferred}
    if (len(ready)!=len(prepared) or ready&cached or (ready|cached)&pending or
            ready|cached|pending!=known):
        raise ValueError('Ready records, reusable graph grids and deferred designs do not partition the inventory')
    return dict(inventory_design_count=len(known),admitted_design_count=len(ready|cached),
                reused_design_count=len(cached),qualified_design_count=len(ready),
                deferred_design_count=len(pending),deferred_designs=sorted(pending))


def make_plan(out,names):
    before_final_preparation(out)
    if (out/'preparation_retry.json').exists():
        raise ValueError('A preparation budget retry is already installed; do not mix retry and admission migrations')
    old=read(out/'protocol.json')
    new=dict(old,code=retry.current_implementation())
    verify_transition(old,new)
    report=read(out/'preparation_report.json')
    if report.get('complete') or report.get('schema')!='tailcompile-v610-recovery-preparation-1':
        raise ValueError('Expected an incomplete recovery qualification report')
    inventory=read(out/'inventory.json')
    deferred=_deferred(out,old,report,inventory,names)
    coverage=_ready_coverage(out,inventory,deferred)
    for key in ('inventory_design_count','reused_design_count'):
        if key in report and report[key]!=coverage[key]:
            raise ValueError(f'Preparation report coverage differs from saved evidence: {key}')
    if report.get('prepared_designs',coverage['qualified_design_count'])!=coverage['qualified_design_count']:
        raise ValueError('Preparation report qualified coverage differs from saved evidence')
    admission=dict(schema=SCHEMA,out=str(out),base_protocol_fingerprint=fingerprint({k:v for k,v in old.items() if k!='code'}),
        deferred=deferred,policy='defer_recorded_preparation_timeouts',created_unix=time.time(),
        reason='Continue every qualified benchmark now; retain explicit timeout coverage gaps without changing labels, final budgets or weights')
    protected=['inventory.json','prepared_inputs.json','preparation_report.json']
    protected += [name for name in ('contexts.partial.json','boundary_plan.json','recovery_manifest.json') if (out/name).is_file()]
    protected += [str(path.relative_to(out)) for pattern in ('screen/*/qualified_input.json','screen/*/adopted_boundary.json')
                  for path in out.glob(pattern)]
    return dict(schema=JOURNAL_SCHEMA,out=str(out),complete=False,created_unix=time.time(),
        old_protocol=old,new_protocol=new,admission_plan=admission,
        audit=dict(report_before=report,inventory_sha256=sha(out/'inventory.json'),coverage=coverage,
                   protected_sha256={name:sha(out/name) for name in sorted(set(protected))}))


def validate_plan(out,plan):
    if plan.get('schema')!=JOURNAL_SCHEMA or plan.get('out')!=str(out):
        raise ValueError('Admission repair journal belongs to another cohort')
    if (out/'preparation_retry.json').exists():
        raise ValueError('Do not mix a preparation budget retry with an admission migration')
    old,new=plan['old_protocol'],plan['new_protocol']
    verify_transition(old,new)
    admission=plan['admission_plan']
    if (admission.get('schema')!=SCHEMA or admission.get('out')!=str(out) or
            admission.get('policy')!='defer_recorded_preparation_timeouts' or not admission.get('reason') or
            admission.get('base_protocol_fingerprint')!=fingerprint({k:v for k,v in old.items() if k!='code'}) or
            not math.isfinite(float(admission.get('created_unix',float('nan'))))):
        raise ValueError('Admission plan changed the base scientific protocol/policy')
    if sha(out/'inventory.json')!=plan['audit']['inventory_sha256']:
        raise ValueError('Full frozen benchmark inventory changed')
    names=[row['design'] for row in admission['deferred']]
    expected=_deferred(out,old,plan['audit']['report_before'],read(out/'inventory.json'),names)
    if expected!=admission['deferred']:
        raise ValueError('Original preparation timeout evidence changed')
    if read(out/'protocol.json') not in (old,new):
        raise ValueError('Protocol changed outside this admission repair')
    path=out/'preparation_admission.json'
    if path.exists() and read(path)!=admission:
        raise ValueError('Preparation admission plan changed outside this repair')


def apply_plan(out,plan):
    validate_plan(out,plan)
    if plan.get('complete'):
        if read(out/'protocol.json')!=plan['new_protocol'] or read(out/'preparation_admission.json')!=plan['admission_plan']:
            raise ValueError('Previously completed admission repair changed')
        return plan
    before_final_preparation(out)
    for name,digest in plan['audit']['protected_sha256'].items():
        path=(out/name).resolve();path.relative_to(out)
        if sha(path)!=digest:
            raise ValueError(f'Qualified evidence changed during admission repair: {name}')
    journal=out/'_repairs/preparation_admission/repair.json'
    if journal.exists() and read(journal)!=plan:
        raise ValueError('Preparation admission journal changed')
    if not journal.exists():
        write(journal,plan)
    write(out/'preparation_admission.json',plan['admission_plan'])
    write(out/'protocol.json',plan['new_protocol'])
    done=dict(plan,complete=True,completed_unix=time.time(),new_protocol_sha256=sha(out/'protocol.json'),
              admission_plan_sha256=sha(out/'preparation_admission.json'))
    write(journal,done)
    return done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--defer',nargs='+',required=True,metavar='DESIGN')
    parser.add_argument('--apply',action='store_true',help='Install the audited admission plan; default previews without writing')
    args=parser.parse_args()
    out=validate_root(args.out)
    with cohort_lock(out):
        journal=out/'_repairs/preparation_admission/repair.json'
        plan=read(journal) if journal.exists() else make_plan(out,args.defer)
        if sorted(args.defer)!=[row['design'] for row in plan['admission_plan']['deferred']]:
            raise ValueError('Requested deferrals differ from the existing admission journal')
        validate_plan(out,plan)
        already=bool(plan.get('complete'))
        if args.apply:
            plan=apply_plan(out,plan)
        print(json.dumps(dict(status='already_applied' if already else ('applied' if args.apply else 'preview'),
            coverage=plan['audit']['coverage'],deferred=plan['admission_plan']['deferred'],
            final_route_timeout=plan['new_protocol']['timeout'],
            checkpoint_sha256=plan['new_protocol']['checkpoint_sha256'],journal=str(journal)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
