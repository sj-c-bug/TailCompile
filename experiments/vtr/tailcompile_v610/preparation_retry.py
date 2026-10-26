"""Audited, stage-specific preparation retries without changing formal budgets.

The original trial, base preparation policy and successful certificates remain
immutable. Only an explicit completed migration journal authorizes a retry.
"""
from __future__ import annotations

from functools import lru_cache
import math
from pathlib import Path
import re

from bridge import REPO, fingerprint, read, sha
from inventory import SAFE_NAME

SCHEMA = 'tailcompile-v610-preparation-retry-1'
PLAN_NAME = 'preparation_retry.json'
JOURNAL_NAME = '_repairs/preparation_timeouts/repair.json'


def _stamp(path):
    stat = path.stat()
    return stat.st_mtime_ns, stat.st_ctime_ns, stat.st_size


def base_fingerprint(protocol):
    return fingerprint({key:value for key,value in protocol.items() if key != 'code'})


def _validate_override(out, item):
    design, stage = item.get('design'), item.get('stage')
    if not isinstance(design,str) or not SAFE_NAME.fullmatch(design):
        raise ValueError('Unsafe preparation retry benchmark')
    if stage not in {'synthesis','calibration'}:
        raise ValueError('Only synthesis/calibration preparation retries are allowed')
    previous, current = item.get('previous_timeout'), item.get('new_timeout')
    if (isinstance(previous,bool) or isinstance(current,bool) or
            not isinstance(previous,(int,float)) or not isinstance(current,(int,float)) or
            not math.isfinite(previous) or not math.isfinite(current) or
            previous <= 0 or current <= previous or int(current) != current):
        raise ValueError('A preparation retry must strictly increase a finite integer budget')
    folder = Path(item['failed_directory']).resolve()
    relative = folder.relative_to(out)
    if stage == 'synthesis':
        expected = Path('inputs')/design/'synthesis'
        if relative != expected:
            raise ValueError('Synthesis retry directory does not match the benchmark')
    elif (len(relative.parts) != 6 or relative.parts[:2] != ('screen',design) or
            not re.fullmatch(r'g\d+x\d+',relative.parts[2]) or
            relative.parts[3] != 'calibration' or
            not re.fullmatch(r'w\d+',relative.parts[4]) or
            not re.fullmatch(r's\d+',relative.parts[5])):
        raise ValueError('Calibration retry directory does not match a frozen grid trial')
    trial_path = folder/'trial.json'
    if sha(trial_path) != item['failed_trial_sha256']:
        raise ValueError('Original preparation timeout evidence changed')
    trial = read(trial_path)
    if trial.get('status') != 'timeout' or trial.get('returncode') != 124 or trial.get('timeout') != previous:
        raise ValueError('Only an exact recorded preparation timeout can authorize a retry')
    identity = trial.get('identity',{})
    if identity.get('design') != design or identity.get('stage') != ('synthesis' if stage == 'synthesis' else 'route'):
        raise ValueError('Preparation timeout evidence belongs to another stage/benchmark')
    return item


@lru_cache(maxsize=16)
def _load(out_string, stamps):
    out = Path(out_string)
    plan = read(out/PLAN_NAME)
    protocol = read(out/'protocol.json')
    journal = read(out/JOURNAL_NAME)
    if (plan.get('schema') != SCHEMA or plan.get('out') != str(out) or
            not journal.get('complete') or journal.get('retry_plan') != plan or
            journal.get('new_protocol') != protocol):
        raise ValueError('Preparation retry plan lacks a matching completed migration journal')
    if base_fingerprint(protocol) != plan.get('base_protocol_fingerprint'):
        raise ValueError('Preparation retry changed the base scientific protocol')
    if not isinstance(plan.get('reason'),str) or not plan['reason'].strip():
        raise ValueError('Preparation retry requires a recorded reason')
    for name,digest in protocol['code'].items():
        path = (REPO/name).resolve()
        path.relative_to(REPO)
        if sha(path) != digest:
            raise ValueError(f'Frozen retry implementation changed: {name}')
    items = plan.get('overrides')
    if not isinstance(items,list) or not items:
        raise ValueError('Preparation retry plan needs explicit stage overrides')
    seen = set()
    for item in items:
        _validate_override(out,item)
        key = (item['design'],item['stage'])
        if key in seen:
            raise ValueError('Duplicate preparation retry stage override')
        seen.add(key)
    return plan


def load_retry_plan(a_or_out):
    out = Path(getattr(a_or_out,'out',a_or_out)).resolve()
    path = out/PLAN_NAME
    if not path.exists():
        if (out/JOURNAL_NAME).exists():
            raise ValueError('Preparation retry journal exists but its plan is missing')
        return None
    paths = (path,out/JOURNAL_NAME,out/'protocol.json')
    plan = _load(str(out),tuple(_stamp(p) for p in paths))
    # Timeouts remain the immutable anchors even when the parsed plan is cached.
    for item in plan['overrides']:
        if sha(Path(item['failed_directory'])/'trial.json') != item['failed_trial_sha256']:
            raise ValueError('Original preparation timeout evidence changed')
    return plan


def override_for(a, design, stage):
    plan = load_retry_plan(a)
    if plan is None:
        return None
    item = next((x for x in plan['overrides'] if (x['design'],x['stage']) == (design,stage)),None)
    if item is None:
        return None
    baseline = a.synthesis_timeout if stage == 'synthesis' else a.calibration_timeout
    if baseline != item['previous_timeout']:
        raise ValueError('Base preparation budget changed outside the audited retry')
    return dict(item,plan_sha256=sha(Path(a.out)/PLAN_NAME))


def audit(item):
    if item is None:
        return None
    return {key:item[key] for key in ('design','stage','previous_timeout','new_timeout',
        'failed_directory','failed_trial_sha256','plan_sha256')}


def suffix(item):
    return f'_retry_{int(item["new_timeout"])}' if item else ''


def calibration_destination(root, width, seed, item):
    """Keep original known outcomes; all retried/new work has its own budget path."""
    original = Path(root)/'calibration'/f'w{width}'/f's{seed}'
    if item is None:
        return original, False
    failed = Path(item['failed_directory']).resolve()
    if failed.parents[2] != Path(root).resolve():
        raise ValueError('Calibration retry must retain the original qualified device grid')
    trial_path = original/'trial.json'
    if trial_path.is_file():
        trial = read(trial_path)
        if trial.get('status') in {'success','unroutable'}:
            return original, False
        if trial.get('status') != 'timeout':
            raise ValueError('Preparation retry cannot replace a non-timeout tool failure')
    elif (original/'started.json').exists():
        raise ValueError('Interrupted/active original calibration trial requires explicit recovery')
    return Path(root)/('calibration'+suffix(item))/f'w{width}'/f's{seed}', True
