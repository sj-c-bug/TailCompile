"""CPU-only subprocess: proposal and feature-cache construction, no torch."""
import argparse
import time
from pathlib import Path
import numpy as np
from core import read,write,sha,snapshot,candidates,positions,flat,features

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,required=True)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--replicas',type=int,default=2)
    a=p.parse_args();folder=a.folder
    start=time.monotonic();graph=snapshot(folder/'default/gp')
    write(folder/'graph.json',graph)
    pool=candidates(graph,a.width,a.replicas);actions=[]
    for name,xy,meta in pool:
        flat(folder/(name+'.fplace'),graph,xy)
        np.save(folder/(name+'.npy'),xy)
        np.savez_compressed(folder/(name+'.features.npz'),**features(graph,xy,a.width))
        actions.append(dict(name=name,features=meta,sha256=sha(folder/(name+'.fplace'))))
    np.savez_compressed(folder/'default.features.npz',**features(graph,positions(graph),a.width))
    write(folder/'pool.json',dict(gp_sha256=read(folder/'default/trial.json')['gp_sha256'],
        generation_seconds=time.monotonic()-start,unique_fplaces=len({r['sha256'] for r in actions}),actions=actions))

if __name__=='__main__':main()
