"""Upgrade a pre-VPR v6.9 cohort for the audited fixed_xy sentinel fix."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from bridge import HERE, REPO, V68, emit, read, sha, write
from generate import PRE_FIXED_XY_FIX


def repair(out):
    out=Path(out).resolve()
    out.relative_to(REPO/'experiments/vtr/tailcompile_v69/runs')
    path=out/'protocol.json'
    protocol=read(path)
    if protocol.get('schema')!='tailcompile-v69-runtime-1' or not protocol.get('code'):
        raise ValueError('Expected a frozen v6.9 protocol')
    changes={}
    for name,old in protocol['code'].items():
        source=HERE/name if (HERE/name).exists() else V68/name
        current=sha(source)
        if old==current:
            continue
        if PRE_FIXED_XY_FIX.get(name)!=old:
            raise ValueError(f'Unrelated implementation change: {name}; refusing to rewrite its fingerprint')
        changes[name]=dict(before=old,after=current)
    if not changes:
        emit('fixed_xy_repair',changed=False,out=str(out),reason='current implementation already recorded')
        return dict(changed=False)

    # Reusing cached candidates is safe because old successful inference had
    # already accepted every coordinate. Runtime measurements would still
    # have different validation costs, so never mix pre-fix VPR/timing trials.
    forbidden={'trial.json','started.json','wave.json','feedback_dataset.json','evaluation_plan.json'}
    for base,dirs,names in os.walk(out):
        if forbidden.intersection(names):
            raise ValueError(f'VPR/training stage evidence already exists at {base}; use a new cohort')
        if Path(base).name=='runtime' and 'models' in Path(base).parts and names:
            raise ValueError('Runtime training already exists; use a new cohort')
    before_sha=sha(path)
    backup=out/'repairs/fixed_xy/protocol.json'
    if backup.exists():
        original=read(backup)
        if original!=protocol:
            raise ValueError('Existing repair backup differs; inspect it before retrying')
    else:
        write(backup,protocol)
    updated=dict(protocol,code=dict(protocol['code']))
    for name,change in changes.items():
        updated['code'][name]=change['after']
    write(path,updated)
    audit=dict(schema='tailcompile-v69-fixed-xy-validation-repair-1',changed=True,
        before_protocol_sha256=before_sha,after_protocol_sha256=sha(path),code_changes=changes,
        backup=str(backup),
        scope='Range-check only nonmovable fixed_xy rows; retain movable sentinel values',
        cached_candidates='Original generation manifests and artifact hashes remain unchanged',
        pre_fix_vpr_trials=0)
    write(out/'fixed_xy_repair.json',audit)
    emit('fixed_xy_repair',**audit)
    return audit


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    repair(parser.parse_args().out)
