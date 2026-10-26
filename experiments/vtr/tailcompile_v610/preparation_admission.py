"""Audited deferral of recorded preparation timeouts, retaining unknown coverage."""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import re

from bridge import REPO, read, sha, write
from inventory import SAFE_NAME
from preparation_retry import base_fingerprint, _stamp

SCHEMA = 'tailcompile-v610-preparation-admission-1'
PLAN_NAME = 'preparation_admission.json'
JOURNAL_NAME = '_repairs/preparation_admission/repair.json'
POLICY = 'defer_recorded_preparation_timeouts'
RUNTIME_NAME = 'preparation_deferred.json'


def validate_deferred(out, item):
    design, stage = item.get('design'), item.get('stage')
    if not isinstance(design,str) or not SAFE_NAME.fullmatch(design):
        raise ValueError('Unsafe deferred benchmark name')
    if stage not in {'synthesis','calibration'} or item.get('status') != 'timeout':
        raise ValueError('Only recorded synthesis/calibration timeouts may be deferred')
    timeout = item.get('timeout')
    if (isinstance(timeout,bool) or not isinstance(timeout,(int,float)) or
            not math.isfinite(timeout) or timeout <= 0):
        raise ValueError('Deferred timeout must retain its original positive time budget')
    folder = Path(item['directory']).resolve()
    relative = folder.relative_to(out)
    if stage == 'synthesis':
        if relative != Path('inputs')/design/'synthesis':
            raise ValueError('Deferred synthesis directory differs from the recorded benchmark')
    elif (len(relative.parts) != 6 or relative.parts[:2] != ('screen',design) or
            not re.fullmatch(r'g\d+x\d+',relative.parts[2]) or
            relative.parts[3] != 'calibration' or
            not re.fullmatch(r'w\d+',relative.parts[4]) or
            not re.fullmatch(r's\d+',relative.parts[5])):
        raise ValueError('Deferred calibration directory is not a preparation grid trial')
    trial_path = folder/'trial.json'
    if sha(trial_path) != item['trial_sha256']:
        raise ValueError('Deferred timeout evidence changed')
    trial = read(trial_path)
    if (trial.get('status') != 'timeout' or trial.get('returncode') != 124 or
            trial.get('timeout') != timeout):
        raise ValueError('Admission may only defer an exact timeout, never an unroutable or tool failure')
    identity = trial.get('identity',{})
    if (identity.get('design') != design or
            identity.get('stage') != ('synthesis' if stage == 'synthesis' else 'route')):
        raise ValueError('Deferred evidence belongs to another stage/benchmark')


@lru_cache(maxsize=16)
def _load(out_string, stamps):
    out = Path(out_string)
    plan, journal, protocol = (read(out/PLAN_NAME),read(out/JOURNAL_NAME),read(out/'protocol.json'))
    if (plan.get('schema') != SCHEMA or plan.get('out') != str(out) or
            plan.get('policy') != POLICY or not journal.get('complete') or
            journal.get('admission_plan') != plan or journal.get('new_protocol') != protocol):
        raise ValueError('Preparation admission lacks a matching completed audit journal')
    if base_fingerprint(protocol) != plan.get('base_protocol_fingerprint'):
        raise ValueError('Preparation admission changed the base scientific protocol')
    if not isinstance(plan.get('reason'),str) or not plan['reason'].strip():
        raise ValueError('Preparation admission requires a recorded reason')
    for name,digest in protocol['code'].items():
        path = (REPO/name).resolve()
        path.relative_to(REPO)
        if sha(path) != digest:
            raise ValueError(f'Frozen admission implementation changed: {name}')
    deferred = plan.get('deferred')
    if not isinstance(deferred,list) or not deferred:
        raise ValueError('Admission plan requires explicit deferred timeout records')
    names = set()
    for item in deferred:
        validate_deferred(out,item)
        if item['design'] in names:
            raise ValueError('Duplicate deferred benchmark')
        names.add(item['design'])
    return plan


def load_admission_plan(a_or_out):
    out = Path(getattr(a_or_out,'out',a_or_out)).resolve()
    plan_path = out/PLAN_NAME
    if not plan_path.exists():
        if (out/JOURNAL_NAME).exists():
            raise ValueError('Preparation admission journal exists but its plan is missing')
        return None
    if (out/'preparation_retry.json').exists():
        raise ValueError('Preparation admission and extended-budget retries require separate execution plans')
    paths = (plan_path,out/JOURNAL_NAME,out/'protocol.json')
    plan = _load(str(out),tuple(_stamp(p) for p in paths))
    for item in plan['deferred']:
        if sha(Path(item['directory'])/'trial.json') != item['trial_sha256']:
            raise ValueError('Deferred timeout evidence changed')
    return plan


def deferred_designs(a_or_out):
    return load_deferred(a_or_out)


def timeout_evidence(a_or_out, design, directory):
    """Derive deferral evidence from an actual terminal trial, never an exception label."""
    out = Path(getattr(a_or_out,'out',a_or_out)).resolve()
    folder = Path(directory).resolve()
    relative = folder.relative_to(out)
    path = folder/'trial.json'
    trial = read(path)
    if trial.get('status') != 'timeout' or trial.get('returncode') != 124:
        raise ValueError('Dynamic admission requires an actual terminal timeout trial')
    budget = trial.get('timeout')
    if (isinstance(budget,bool) or not isinstance(budget,(int,float)) or
            not math.isfinite(budget) or budget <= 0):
        raise ValueError('Dynamic timeout trial has no valid time budget')
    identity = trial.get('identity',{})
    original_stage = identity.get('stage')
    stage = original_stage
    if original_stage == 'route':
        stage = 'qualification' if identity.get('role') == 'default_placement_qualification' else 'calibration'
    if original_stage == 'rr_graph':
        if (len(relative.parts) != 4 or relative.parts[0] != 'device_cache' or
                not re.fullmatch(r'\d+x\d+',relative.parts[1]) or
                not re.fullmatch(r'w\d+',relative.parts[2]) or relative.parts[3] != 'build'):
            raise ValueError('Dynamic RRG timeout is not a shared device-cache build trial')
        if identity.get('grid') != '/'.join(relative.parts[1:3]):
            raise ValueError('Shared RRG timeout belongs to a different device/width')
    elif (stage not in {'synthesis','structure','qualification','calibration'} or
            identity.get('design') != design or len(relative.parts) < 3 or
            relative.parts[:2] not in {('inputs',design),('screen',design)}):
        raise ValueError('Dynamic timeout is not preparation evidence for this benchmark')
    return dict(design=design,stage=stage,status='timeout',directory=str(folder),
        trial_sha256=sha(path),timeout=budget)


def load_deferred(a_or_out):
    plan = load_admission_plan(a_or_out)
    if plan is None:
        return {}
    out = Path(getattr(a_or_out,'out',a_or_out)).resolve()
    rows = {item['design']:item for item in plan['deferred']}
    path = out/RUNTIME_NAME
    if path.is_file():
        saved = read(path)
        if (saved.get('schema') != 'tailcompile-v610-runtime-deferrals-1' or
                saved.get('admission_sha256') != sha(out/PLAN_NAME)):
            raise ValueError('Dynamic deferrals changed their admission policy')
        for item in saved.get('deferred',[]):
            expected = timeout_evidence(out,item['design'],item['directory'])
            if item != expected or item['design'] in rows:
                raise ValueError('Dynamic deferral evidence changed or duplicated a benchmark')
            rows[item['design']] = item
    return rows


def save_deferred(a_or_out, admission, deferred):
    out = Path(getattr(a_or_out,'out',a_or_out)).resolve()
    names = {item['design'] for item in admission['deferred']}
    rows = [deferred[name] for name in sorted(deferred) if name not in names]
    write(out/RUNTIME_NAME,dict(schema='tailcompile-v610-runtime-deferrals-1',
        admission_sha256=sha(out/PLAN_NAME),deferred=rows,
        policy=POLICY,coverage='All deferred benchmarks retain unknown routability'))
