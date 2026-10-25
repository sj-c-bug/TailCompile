"""Repair only the two macro-only Koios entries before v6.10 graph preparation.

Preserves checkpoint, backend, calibration settings and every trial. The old and
new frozen records are journaled before replacement, so interrupted repairs can
be resumed without relaxing scientific input checks.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[2]
CODE_KEY = 'experiments/vtr/tailcompile_v610/inventory.py'
LEGACY_SHA256 = '359966c30e3fe4ce5f2102efe069bb1cc7d46215c95ad80bcdd6aedc9b123051'
HELPER_SOURCES = {'koios/hard_mem_include.v', 'koios/complex_dsp_include.v'}
sys.path.insert(0, str(REPO/'experiments/vtr/tailcompile_v610'))
from bridge import read, write, sha
from inventory import discover


def validate_root(out):
    out = Path(out).resolve()
    relative = out.relative_to(REPO/'experiments/vtr/tailcompile_v610/runs')
    if len(relative.parts) != 1 or not out.is_dir():
        raise ValueError('Select one existing v610 cohort, not a broad directory')
    return out


@contextmanager
def cohort_lock(out):
    if os.name != 'posix':
        raise RuntimeError('Apply this repair on the Linux server')
    import fcntl
    with (out/'.lock').open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('The cohort is still running; stop its pipeline before repair') from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def before_final_preparation(out):
    for name in ('prepared_signature.json','contexts.json','evaluation_plan.json'):
        if (out/name).exists():
            raise ValueError(f'{name} already exists; this pre-qualification repair is no longer allowed')
    for name in ('graphs','device_cache','generation','evaluation','timing'):
        if (out/name).exists() and any(p.is_file() for p in (out/name).rglob('*')):
            raise ValueError(f'{name} already contains generated payloads; refuse late migration')
    for marker in out.rglob('started.json'):
        if {'_attempts','_interrupted','_repairs'} & set(marker.parts):
            continue
        if not (marker.parent/'trial.json').exists():
            raise ValueError(f'Interrupted/active trial remains: {marker.parent}; recover before repair')


def verify_implementation(protocol, *, legacy=False):
    code = protocol['code']
    if legacy and code.get(CODE_KEY) != LEGACY_SHA256:
        raise ValueError('Expected the released v610 inventory implementation; no arbitrary code-hash rewrite allowed')
    for name,digest in code.items():
        path = (REPO/name).resolve()
        path.relative_to(REPO)
        if legacy and name == CODE_KEY:
            continue
        if sha(path) != digest:
            raise ValueError(f'Unrelated frozen implementation changed: {name}')
    if sha(protocol['checkpoint']) != protocol['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')


def validate_inventory_transition(old, new):
    old_rows = {r['design']:r for r in old['designs']}
    new_rows = {r['design']:r for r in new['designs']}
    if len(old_rows)!=len(old['designs']) or len(new_rows)!=len(new['designs']):
        raise ValueError('Duplicate inventory design')
    removed = [old_rows[k] for k in sorted(old_rows.keys()-new_rows.keys())]
    if (not removed or new_rows.keys()-old_rows.keys() or
            any(r['source'] not in HELPER_SOURCES for r in removed)):
        raise ValueError('Inventory change is not exclusively removal of the two macro-only include files')
    for name,row in new_rows.items():
        if old_rows[name] != row:
            raise ValueError(f'Runnable benchmark metadata/input changed: {name}')
    removed_sources = {r['source'] for r in removed}
    added_exclusions = [r for r in new['excluded'] if r.get('source') in removed_sources]
    if [r for r in new['excluded'] if r.get('source') not in removed_sources] != old['excluded']:
        raise ValueError('Unrelated exclusions changed')
    if {r['source'] for r in added_exclusions} != removed_sources:
        raise ValueError('Removed helper evidence is missing')
    for row in removed:
        evidence = next(r for r in added_exclusions if r['source'] == row['source'])
        if evidence['sha256'] != row['source_sha256']:
            raise ValueError(f'Include source changed: {row["source"]}')
    for key in set(old)|set(new):
        if key not in {'designs','excluded','design_count','family_count','koios_design_count'} and old.get(key) != new.get(key):
            raise ValueError(f'Unrelated inventory metadata changed: {key}')
    if new['design_count']!=len(new_rows) or old['design_count']-new['design_count']!=len(removed):
        raise ValueError('Invalid inventory design count')
    return removed


def make_plan(out, vtr):
    before_final_preparation(out)
    protocol = read(out/'protocol.json')
    if protocol.get('schema') != 'tailcompile-v610-frozen-sweep-1' or protocol.get('training_permitted') is not False:
        raise ValueError('Expected the frozen, no-training v610 protocol')
    verify_implementation(protocol, legacy=True)
    old = read(out/'inventory.json')
    new = discover(vtr, protocol['source'], protocol.get('checkpoint_metadata'),
                   protocol.get('designs'), scope=protocol['scope'])
    removed=validate_inventory_transition(old,new)
    report = read(out/'preparation_report.json')
    removed_names = {r['design'] for r in removed}
    for name in removed_names:
        failures = [f for f in report['failures'] if f['design']==name]
        if not failures or any(f['stage']!='input' for f in failures):
            raise ValueError(f'Expected only failed input qualification for helper {name}')
    updated = dict(protocol, code=dict(protocol['code'], **{CODE_KEY:sha(REPO/CODE_KEY)}))
    return dict(schema='tailcompile-v610-helper-inventory-repair-1', complete=False,
        created_unix=time.time(), out=str(out), removed_helpers=removed,
        old_protocol=protocol,new_protocol=updated,old_inventory=old,new_inventory=new,
        preparation_report_before=report,
        report_before_sha256=sha(out/'preparation_report.json'),
        policy='Only auxiliary-file discovery corrected before final preparation; all trials and scientific settings preserved')


def apply_plan(out, plan):
    if plan.get('schema') != 'tailcompile-v610-helper-inventory-repair-1' or plan['out'] != str(out):
        raise ValueError('Repair journal does not belong to this cohort')
    before_final_preparation(out)
    old_protocol=plan['old_protocol']
    expected_protocol=dict(old_protocol,code=dict(old_protocol['code'],**{CODE_KEY:sha(REPO/CODE_KEY)}))
    if old_protocol['code'].get(CODE_KEY)!=LEGACY_SHA256 or plan['new_protocol']!=expected_protocol:
        raise ValueError('Repair journal changed scientific protocol fields')
    if validate_inventory_transition(plan['old_inventory'],plan['new_inventory'])!=plan['removed_helpers']:
        raise ValueError('Repair journal changed removed helper evidence')
    verify_implementation(plan['new_protocol'])
    for filename,oldkey,newkey in [('protocol.json','old_protocol','new_protocol'),
                                   ('inventory.json','old_inventory','new_inventory')]:
        current = read(out/filename)
        if current != plan[oldkey] and current != plan[newkey]:
            raise ValueError(f'{filename} changed outside this repair; refuse to overwrite')
    # The durable journal holds exact old/new JSON before either replacement.
    journal = out/'_repairs/helper_inventory/repair.json'
    if not journal.exists():
        write(journal,plan)
    elif read(journal) != plan:
        raise ValueError('Repair journal changed')
    write(out/'inventory.json',plan['new_inventory'])
    write(out/'protocol.json',plan['new_protocol'])
    done = dict(plan,complete=True,completed_unix=time.time(),
                new_protocol_sha256=sha(out/'protocol.json'),new_inventory_sha256=sha(out/'inventory.json'))
    write(journal,done)
    return done


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--vtr-root',type=Path,default=os.environ.get('VTR_ROOT'))
    parser.add_argument('--apply',action='store_true',help='Apply the narrowly audited correction; default previews only')
    args=parser.parse_args()
    out=validate_root(args.out)
    with cohort_lock(out):
        journal=out/'_repairs/helper_inventory/repair.json'
        if journal.exists():
            plan=read(journal)
            if plan.get('complete'):
                verify_implementation(plan['new_protocol'])
                if read(out/'protocol.json') != plan['new_protocol'] or read(out/'inventory.json') != plan['new_inventory']:
                    raise ValueError('Previously repaired records have changed')
                print(json.dumps(dict(status='already_applied',journal=str(journal)),ensure_ascii=False))
                return
        else:
            if not args.vtr_root:
                parser.error('Set VTR_ROOT or --vtr-root')
            plan=make_plan(out,args.vtr_root)
        if args.apply:
            plan=apply_plan(out,plan)
        print(json.dumps(dict(status='applied' if args.apply else 'preview',
            removed=[r['design'] for r in plan['removed_helpers']],
            before=plan['old_inventory']['design_count'],after=plan['new_inventory']['design_count'],
            remaining_failures=[f for f in read(out/'preparation_report.json')['failures']
                                if f['design'] not in {r['design'] for r in plan['removed_helpers']}],
            checkpoint_sha256=plan['new_protocol']['checkpoint_sha256'],
            audit_journal=str(journal)),ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
