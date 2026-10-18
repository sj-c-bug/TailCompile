"""Real pinned-VPR GP/replay smoke; no synthesis or training required."""
import argparse
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from campaign import environment, run
from core import snapshot, positions, flat, candidates, write

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--blif',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--width',type=int,default=100)
    p.add_argument('--seed',type=int,default=201)
    a=p.parse_args()
    a.out=a.out.resolve()
    inp=a.out/'inputs/sha/sha.pre-vpr.blif'; inp.parent.mkdir(parents=True,exist_ok=True)
    if not inp.exists(): shutil.copy2(a.blif,inp)
    opts=SimpleNamespace(out=a.out,vpr_threads=1,timeout=600)
    environment(opts)
    original=run(opts,'sha',a.width,a.seed,a.out/'default')
    graph=snapshot(a.out/'default/gp')
    flat(a.out/'replay.fplace',graph,positions(graph))
    replay=run(opts,'sha',a.width,a.seed,a.out/'replay',a.out/'replay.fplace',original['gp_sha256'])
    def coordinates(path):
        return {line.split()[0]:line.split()[1:5] for line in path.read_text().splitlines()
                if line and not line.startswith('#') and len(line.split())>=5}
    identical=coordinates(a.out/'default/post_route.fplace')==coordinates(a.out/'replay/post_route.fplace')
    start=time.monotonic()
    pool=candidates(graph,a.width,replicas=1)
    generation=time.monotonic()-start
    changed=next(((name,xy,meta) for name,xy,meta in pool if meta['moved']),None)
    result=None
    if changed:
        name,xy,meta=changed
        flat(a.out/'candidate.fplace',graph,xy)
        result=run(opts,'sha',a.width,a.seed,a.out/'candidate',a.out/'candidate.fplace',original['gp_sha256'])
    write(a.out/'smoke_report.json',dict(default=original,replay=replay,
        exact_post_route_coordinate_replay=identical,candidate=result,
        candidate_generation_seconds=generation,ap_blocks=len(graph['blocks']),
        moved=changed[2]['moved'] if changed else 0,action=changed[0] if changed else None))
    if not identical or original['status']!=replay['status']:
        raise RuntimeError('unchanged replay altered flow; do not launch production')

if __name__=='__main__': main()
