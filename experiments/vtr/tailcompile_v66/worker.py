"""Synchronous CPU post-GP deployment worker. Never reads routing labels."""
import argparse
import time
from pathlib import Path
import torch
from core import snapshot, positions, candidates, flat, features, write
from learn import Selector, choose

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--width',type=int,required=True)
    p.add_argument('--replicas',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(1);start=time.monotonic()
    graph=snapshot(a.prefix)
    pool=[('default',positions(graph),{})]+candidates(graph,a.width,a.replicas)
    model=Selector()
    model.load_state_dict(torch.load(a.model,map_location='cpu',weights_only=False)['model']);model.eval()
    with torch.no_grad():
        output=torch.stack([model({k:torch.as_tensor(v) for k,v in features(graph,xy,a.width).items()}) for _,xy,_ in pool])
    picked=choose(output)
    flat(a.output,graph,pool[picked][1])
    write(a.output.with_suffix('.selection.json'),dict(action=pool[picked][0],
        worker_seconds=time.monotonic()-start,scores=output.tolist(),reads_routing_labels=False))

if __name__=='__main__':main()
