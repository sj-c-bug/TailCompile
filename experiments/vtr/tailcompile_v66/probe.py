"""Local diagnostic oracle on existing smoke anchors; not independent validation."""
import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from campaign import environment,run,parallel
from core import read,write,sha,snapshot,candidates,flat,retention

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--roots',type=Path,nargs='+',required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args();tasks=[]
    for root in a.roots:
        root=root.resolve();opts=SimpleNamespace(out=root,vpr_threads=1,timeout=600)
        environment(opts)
        baseline=read(root/'default/trial.json');width=baseline['signature']['width'];seed=baseline['signature']['seed']
        graph=snapshot(root/'default/gp');start=time.monotonic()
        pool=candidates(graph,width)
        folder=root/'probe';folder.mkdir(exist_ok=True)
        for name,xy,meta in pool:
            fplace=folder/(name+'.fplace');flat(fplace,graph,xy)
            tasks.append((opts,root,width,seed,name,fplace,baseline,graph,xy,meta))
        write(folder/'generation.json',dict(seconds=time.monotonic()-start,distinct=len({sha(folder/(n+'.fplace')) for n,_,_ in pool})))
    def one(t):
        opts,root,width,seed,name,fplace,baseline,graph,xy,meta=t
        folder=root/'probe'/name
        result=run(opts,'sha',width,seed,folder,fplace,baseline['gp_sha256'])
        return dict(seed=seed,width=width,action=name,status=result['status'],baseline_status=baseline['status'],
                    wall_seconds=result['wall_seconds'],cpd=result['cpd'],wirelength=result['wirelength'],moved=meta['moved'],
                    retention=retention(root/'default/post_route.fplace',folder/'post_route.fplace',graph,xy),
                    signature=result['signature'])
    rows=parallel(one,tasks,a.jobs)
    summary=[]
    for seed in sorted({r['seed'] for r in rows}):
        group=[r for r in rows if r['seed']==seed]
        summary.append(dict(seed=seed,default=group[0]['baseline_status'],
            successful_actions=[r['action'] for r in group if r['status']=='success'],
            oracle_success=any(r['status']=='success' for r in group)))
    write(a.out,dict(diagnostic_only=True,design='sha',summary=summary,rows=rows,
                    warning='Previously developed design, width 90 only. Oracle looks at all outcomes; not a deployable result or Wmin sweep.'))

if __name__=='__main__':main()
