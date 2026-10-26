"""Audit a preparation-only timeout retry without replacing successful inputs.

The original device/width selection, checkpoint and final evaluation budget stay
frozen. Only the two released preparation modules and their new retry helper
may change. The complete old/new records are durable before either write.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import sys
import time

REPO = Path(__file__).resolve().parents[2]
V610 = REPO/'experiments/vtr/tailcompile_v610'
sys.path.insert(0, str(V610))
from bridge import fingerprint, read, sha, write
from repair_tailcompile_v610_inventory import before_final_preparation

SCHEMA = 'tailcompile-v610-preparation-retry-1'
JOURNAL_SCHEMA = 'tailcompile-v610-preparation-timeout-repair-1'
RELEASED = {
    'experiments/vtr/tailcompile_v610/recovery.py':
        '38f5d3f3ab0e4ef3914f2b07c1fa3ecc47b98081c0e7ff4bd6b4f5c4b06e376b',
    'experiments/vtr/tailcompile_v610/recovery_inputs.py':
        'ed06e86f22e6756d418a38a4ec93b78f1eba7facd0a4b9714fe392aefec534b6',
}
ADDED = 'experiments/vtr/tailcompile_v610/preparation_retry.py'
MAX_TIMEOUT = 7*24*3600


def validate_root(out):
    out=Path(out).resolve()
    relative=out.relative_to(REPO/'experiments/vtr/tailcompile_v610/runs')
    if len(relative.parts)!=1 or not out.is_dir():
        raise ValueError('Select one existing v610 cohort')
    return out


@contextmanager
def cohort_lock(out):
    if os.name!='posix':
        raise RuntimeError('Run this preparation repair on the Linux server')
    import fcntl
    # Preview must not create or update a lock file in the experiment.
    with (out/'.lock').open('rb') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('The cohort is still running; stop it before repair') from error
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def current_implementation():
    spec=importlib.util.spec_from_file_location('_v610_retry_run',V610/'run.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.implementation()


def verify_source(protocol):
    from recovery_source import fingerprint_source
    expected=protocol.get('recovery_source')
    if not expected or fingerprint_source(expected['source'])!=expected:
        raise ValueError('Read-only recovery source changed or is missing')


def verify_transition(old,new):
    if old.get('schema')!='tailcompile-v610-frozen-sweep-1' or old.get('training_permitted') is not False:
        raise ValueError('Expected a frozen, no-training v610 protocol')
    if not old.get('preparation_policy'):
        raise ValueError('Expected the released recovery preparation policy')
    if {k:v for k,v in old.items() if k!='code'}!={k:v for k,v in new.items() if k!='code'}:
        raise ValueError('Repair cannot change scientific protocol fields or final timeout')
    previous,current=old['code'],current_implementation()
    if new['code']!=current:
        raise ValueError('Repair new code map differs from the current implementation')
    if set(current)-set(previous)!={ADDED} or set(previous)-set(current):
        raise ValueError('Unexpected added or removed frozen implementation')
    for name,digest in previous.items():
        path=(REPO/name).resolve();path.relative_to(REPO)
        if name in RELEASED:
            if digest!=RELEASED[name]:
                raise ValueError(f'Unrecognized released preparation implementation: {name}')
        elif current[name]!=digest:
            raise ValueError(f'Unrelated frozen implementation changed: {name}')
    for name,digest in current.items():
        path=(REPO/name).resolve();path.relative_to(REPO)
        if not path.is_file() or sha(path)!=digest:
            raise ValueError(f'Current implementation changed while preparing repair: {name}')
    if sha(old['checkpoint'])!=old['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    verify_source(old)


def parse_override(value):
    name,separator,raw=value.partition('=')
    if not separator or not re.fullmatch(r'[A-Za-z0-9_.-]+',name):
        raise ValueError('Expected benchmark=seconds')
    try:
        budget=float(raw)
    except ValueError as error:
        raise ValueError('Retry timeout must be a number') from error
    if not math.isfinite(budget) or not 0<budget<=MAX_TIMEOUT or int(budget)!=budget:
        raise ValueError('Retry timeout must use positive integer seconds and at most seven days')
    return name,budget


def requests(synthesis,calibration):
    result={}
    for stage,values in [('synthesis',synthesis),('calibration',calibration)]:
        for value in values:
            name,budget=parse_override(value)
            if (name,stage) in result:
                raise ValueError(f'Duplicate timeout override: {name}/{stage}')
            result[name,stage]=budget
    if not result:
        raise ValueError('Specify at least one --synthesis or --calibration timeout')
    return result


def validated_overrides(out,protocol,report,inventory,wanted):
    names={r['design'] for r in inventory['designs']}
    if len(names)!=len(inventory['designs']):
        raise ValueError('Duplicate inventory design')
    overrides=[]
    used=set()
    for failure in report.get('failures',[]):
        name=failure.get('design')
        candidates=[stage for design,stage in wanted if design==name]
        if not candidates:
            continue
        if failure.get('status')!='timeout' or not failure.get('directory'):
            raise ValueError(f'Requested design does not have a reported timeout: {name}')
        folder=Path(failure['directory']).resolve()
        relative=folder.relative_to(out)
        trial=read(folder/'trial.json')
        identity=trial.get('identity',{})
        actual=identity.get('stage')
        stage='calibration' if actual=='route' else actual
        key=name,stage
        if key not in wanted:
            raise ValueError(f'Timeout stage does not match requested override: {name}/{actual}')
        if key in used or name not in names:
            raise ValueError('Duplicate failure or design missing from frozen inventory')
        if stage=='synthesis':
            valid_path=relative.parts==('inputs',name,'synthesis')
        else:
            valid_path=(len(relative.parts)==6 and relative.parts[:2]==('screen',name)
                and re.fullmatch(r'g[0-9]+x[0-9]+',relative.parts[2]) is not None
                and relative.parts[3]=='calibration'
                and re.fullmatch(r'w[0-9]+',relative.parts[4]) is not None
                and relative.parts[5] in {f's{s}' for s in protocol['calibration_seeds']})
        if not valid_path or stage not in {'synthesis','calibration'}:
            raise ValueError(f'Timeout is not an original preparation attempt: {folder}')
        if (trial.get('status')!='timeout' or trial.get('returncode')!=124 or
                identity.get('design')!=name or not isinstance(trial.get('command'),list) or
                not trial['command'] or trial['command']!=identity.get('command') or
                any(identity.get(k)!=v for k,v in protocol['identity'].items())):
            raise ValueError(f'Timeout evidence/identity does not match the frozen backend: {folder}')
        previous=float(trial['timeout'])
        expected=protocol['preparation_policy'][stage+'_timeout']
        if previous!=expected or not math.isfinite(previous) or wanted[key]<=previous:
            raise ValueError('Retry budget must strictly exceed the original preparation timeout')
        if stage=='calibration' and (identity.get('teacher') or identity.get('initializer_sha256') is not None):
            raise ValueError('Calibration retry must be uninitialized Default')
        overrides.append(dict(design=name,stage=stage,previous_timeout=previous,new_timeout=wanted[key],
            failed_directory=str(folder),failed_trial_sha256=sha(folder/'trial.json')))
        used.add(key)
    if used!=set(wanted):
        raise ValueError('Every override must match exactly one reported preparation timeout')
    return sorted(overrides,key=lambda r:(r['design'],r['stage']))


def make_plan(out,wanted):
    before_final_preparation(out)
    old=read(out/'protocol.json')
    new=dict(old,code=current_implementation())
    verify_transition(old,new)
    report=read(out/'preparation_report.json')
    if report.get('complete') or report.get('schema')!='tailcompile-v610-recovery-preparation-1':
        raise ValueError('Expected an incomplete recovery qualification report')
    inventory=read(out/'inventory.json')
    overrides=validated_overrides(out,old,report,inventory,wanted)
    retry=dict(schema=SCHEMA,out=str(out),base_protocol_fingerprint=fingerprint({k:v for k,v in old.items() if k!='code'}),
        overrides=overrides,created_unix=time.time(),
        reason='Retry reported preparation timeouts with explicit larger budgets; preserve all completed qualifications and formal evaluation settings')
    return dict(schema=JOURNAL_SCHEMA,complete=False,out=str(out),created_unix=time.time(),
        old_protocol=old,new_protocol=new,retry_plan=retry,
        audit=dict(report_before=report,report_before_sha256=sha(out/'preparation_report.json'),
                   inventory_sha256=sha(out/'inventory.json'),old_protocol_sha256=sha(out/'protocol.json')))


def validate_plan(out,plan):
    if plan.get('schema')!=JOURNAL_SCHEMA or plan.get('out')!=str(out):
        raise ValueError('Repair journal does not belong to this cohort')
    old,new=plan['old_protocol'],plan['new_protocol']
    verify_transition(old,new)
    retry=plan['retry_plan']
    if (retry.get('schema')!=SCHEMA or retry.get('out')!=str(out) or
            retry.get('base_protocol_fingerprint')!=fingerprint({k:v for k,v in old.items() if k!='code'}) or
            not retry.get('reason') or not math.isfinite(float(retry.get('created_unix',float('nan'))))):
        raise ValueError('Invalid preparation retry plan')
    if sha(out/'inventory.json')!=plan['audit']['inventory_sha256']:
        raise ValueError('Frozen inventory changed')
    wanted={(r['design'],r['stage']):r['new_timeout'] for r in retry['overrides']}
    if len(wanted)!=len(retry['overrides']):
        raise ValueError('Duplicate preparation retry plan override')
    for budget in wanted.values():
        if (isinstance(budget,bool) or not isinstance(budget,(int,float)) or
                not math.isfinite(budget) or not 0<budget<=MAX_TIMEOUT or int(budget)!=budget):
            raise ValueError('Retry timeout must use positive integer seconds and at most seven days')
    expected=validated_overrides(out,old,plan['audit']['report_before'],read(out/'inventory.json'),wanted)
    if expected!=retry['overrides']:
        raise ValueError('Preparation timeout evidence changed after planning')
    if read(out/'protocol.json') not in (old,new):
        raise ValueError('Protocol changed outside this repair')
    path=out/'preparation_retry.json'
    if path.exists() and read(path)!=retry:
        raise ValueError('Preparation retry plan changed outside this repair')


def apply_plan(out,plan):
    validate_plan(out,plan)
    if plan.get('complete'):
        if read(out/'protocol.json')!=plan['new_protocol'] or read(out/'preparation_retry.json')!=plan['retry_plan']:
            raise ValueError('Previously applied preparation repair changed')
        return plan
    before_final_preparation(out)
    if sha(out/'preparation_report.json')!=plan['audit']['report_before_sha256']:
        raise ValueError('Preparation report changed while repair was pending')
    journal=out/'_repairs/preparation_timeouts/repair.json'
    if journal.exists() and read(journal)!=plan:
        raise ValueError('Preparation repair journal changed')
    if not journal.exists():
        write(journal,plan)
    # Runtime accepts the retry plan only after the complete durable journal.
    write(out/'preparation_retry.json',plan['retry_plan'])
    write(out/'protocol.json',plan['new_protocol'])
    done=dict(plan,complete=True,completed_unix=time.time(),new_protocol_sha256=sha(out/'protocol.json'),
              retry_plan_sha256=sha(out/'preparation_retry.json'))
    write(journal,done)
    return done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--synthesis',action='append',default=[],metavar='DESIGN=SECONDS')
    parser.add_argument('--calibration',action='append',default=[],metavar='DESIGN=SECONDS')
    parser.add_argument('--apply',action='store_true',help='Apply the audited retry plan; default is a read-only preview')
    args=parser.parse_args()
    out=validate_root(args.out)
    wanted=requests(args.synthesis,args.calibration)
    with cohort_lock(out):
        journal=out/'_repairs/preparation_timeouts/repair.json'
        plan=read(journal) if journal.exists() else make_plan(out,wanted)
        actual={(r['design'],r['stage']):r['new_timeout'] for r in plan['retry_plan']['overrides']}
        if actual!=wanted:
            raise ValueError('Requested budgets differ from the existing repair journal')
        validate_plan(out,plan)
        already=bool(plan.get('complete'))
        if args.apply:
            plan=apply_plan(out,plan)
        print(json.dumps(dict(status='already_applied' if already else ('applied' if args.apply else 'preview'),
            overrides=plan['retry_plan']['overrides'],final_route_timeout=plan['new_protocol']['timeout'],
            checkpoint_sha256=plan['new_protocol']['checkpoint_sha256'],journal=str(journal)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
