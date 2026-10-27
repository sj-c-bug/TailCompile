"""Retire only completed prototype RRG/pack-net payloads after dependency audit."""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from io_utils import read,sha,write

ALLOWED_ROOT=Path('/home/phoenix/vtr_work/tailcompile_3d')
NAMES={'v7_hz_probe_20261024','v7_hz_probe_20261024_r2','v7_hz_probe_20261024_final'}


def plan(root):
    root=Path(root).resolve()
    if root.parent!=ALLOWED_ROOT.resolve() or root.name not in NAMES:
        raise ValueError('only the three explicitly reviewed prototype directories are eligible')
    audit=read(root/'architecture_audit.json')
    if not audit.get('four_cell_matches'):
        raise ValueError('actual four-cell audit must be durable before retirement')
    targets=[]
    for condition in ('H0Z0','H0Z1','H1Z0','H1Z1'):
        for width in (64,128):
            folder=root/condition/f'w{width}'
            manifest=read(folder/'manifest.json')
            if manifest.get('returncode')!=0 or sha(folder/'device_graph.json')!=manifest['device_graph_sha256']:
                raise ValueError('incomplete or modified graph evidence')
            for target in [folder/'rr_graph.xml',*folder.glob('*.net')]:
                if not target.exists():
                    continue
                if target.is_symlink() or target.resolve().parent!=folder.resolve():
                    raise ValueError('payload escaped exact completed work directory')
                digest=sha(target)
                if target.name=='rr_graph.xml' and digest!=manifest['rr_graph_sha256']:
                    raise ValueError('raw RRG differs from persisted audit provenance')
                targets.append(dict(path=str(target),bytes=target.stat().st_size,sha256=digest,
                    dependency_reason='graph/audit/hash durable; downstream local smoke reads device_graph JSON, arch and BLIF only',
                    recovery='rerun recorded pack-only VPR command with original arch/BLIF/backend'))
    return dict(root=str(root),created_unix=time.time(),targets=targets,
                total_bytes=sum(t['bytes'] for t in targets),recursive_delete=False,
                retained=['architecture_audit.json','device_graph.json','arch.xml','tiny.blif','manifest.json','vpr.out'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',nargs='+',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--apply',action='store_true')
    args=p.parse_args()
    plans=[plan(root) for root in args.roots]
    write(args.out,dict(plans=plans,applied=False,total_bytes=sum(p['total_bytes'] for p in plans)))
    if args.apply:
        removed=[]
        for entry in plans:
            for item in entry['targets']:
                target=Path(item['path'])
                if target.stat().st_size!=item['bytes'] or sha(target)!=item['sha256']:
                    raise ValueError('payload changed after cleanup planning')
                target.unlink()
                removed.append(item)
        write(args.out,dict(plans=plans,applied=True,removed=removed,
                            total_bytes=sum(p['bytes'] for p in removed)))
    print(dict(applied=args.apply,total_bytes=sum(p['total_bytes'] for p in plans)))


if __name__=='__main__':
    main()
