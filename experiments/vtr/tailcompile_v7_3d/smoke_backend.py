"""Run a small real native 3D GP/APPack replay and changed-layer qualification."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import time
import xml.etree.ElementTree as ET


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vpr',type=Path,required=True)
    p.add_argument('--arch',type=Path,required=True)
    p.add_argument('--blif',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--width',type=int,default=320)
    p.add_argument('--seed',type=int,default=701)
    p.add_argument('--timeout',type=int,default=180)
    a=p.parse_args()
    out=a.out.resolve()
    if out.exists():
        p.error('output must be new')
    out.mkdir(parents=True)
    fixed=ET.parse(a.arch).getroot().find('layout/fixed_layout')
    device=fixed.get('name') if fixed is not None else 'auto'
    base=[str(a.vpr.resolve()),str(a.arch.resolve()),str(a.blif.resolve()),
          '--analytical_place','--route','--analysis','--ap_full_legalizer','appack',
          '--ap_detailed_placer','annealer','--route_chan_width',str(a.width),
          '--device',device,'--num_workers','1','--seed',str(a.seed),
          '--write_flat_place','post_sa.fplace','--write_legalized_flat_place','post_appack.fplace',
          '--flat_place_verbosity','1']
    records=[]
    def run(name,extra):
        directory=out/name
        directory.mkdir()
        env={k:v for k,v in os.environ.items() if not k.startswith('TC_GP_')}
        env.update({k:str(v) for k,v in extra.items()})
        start=time.monotonic()
        result=subprocess.run(base,cwd=directory,env=env,capture_output=True,text=True,timeout=a.timeout)
        text=result.stdout+result.stderr
        (directory/'vpr.out').write_text(text)
        row=dict(name=name,returncode=result.returncode,elapsed_seconds=time.monotonic()-start,
                 command=base,env=extra,markers=[line for line in text.splitlines() if 'TAILCOMPILE_V7_' in line])
        records.append(row)
        (out/'progress.json').write_text(json.dumps(records,indent=2,default=str)+'\n')
        if result.returncode or 'VPR succeeded' not in text:
            raise RuntimeError(f'VPR qualification failed: {directory}/vpr.out')
        return text
    run('default',{'TC_GP_EXPORT':out/'default/teacher'})
    rows=[line.split('\t') for line in (out/'default/teacher.blocks.tsv').read_text().splitlines()]
    layer_count=int((out/'default/teacher.meta.tsv').read_text().split()[2])
    for name,flip in [('replay',False),('changed_layer',True)]:
        prior=out/(name+'.ap.tsv')
        lines=[]
        for row in rows:
            if row[3]=='1':
                continue
            z=max(0,min(layer_count-1,math.floor(float(row[6])+0.5)))
            if flip:
                z=layer_count-1-z
            lines.append(' '.join([row[1],row[4],row[5],str(z),row[7]])+'\n')
        prior.write_text(''.join(lines))
        text=run(name,{'TC_GP_INIT':prior,'TC_GP_TRACE':out/name/'injected','TC_GP_EXPORT':out/name/'teacher'})
        if 'TAILCOMPILE_V7_GP_INIT_CONSUMED' not in text or 'TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' not in text or 'skipping the Global Placement' in text:
            raise AssertionError('native initialization consumption/continuation absent')
    report=dict(schema='tailcompile-v7-backend-smoke-1',status='passed',
                binary_sha256=hashlib.sha256(a.vpr.read_bytes()).hexdigest(),
                layer_count=layer_count,ap_blocks=len({r[0] for r in rows}),trials=records,
                scope='Native GP/APPack interface qualification only; no GNN/performance claim')
    (out/'smoke_report.json').write_text(json.dumps(report,indent=2,default=str)+'\n')
    print(json.dumps(dict(status='passed',report=str(out/'smoke_report.json'))))


if __name__=='__main__':
    main()
