"""GPU graph selector and CPU inference, separated from all VPR work.

Learns AP-graph candidate ranking, success and relative QoR/runtime. The graph
is the actual VPR molecule/RAM grouping, not the old guessed packed grouping.
RUDY is an auxiliary feature; selection is supervised by VPR, not minimized
RUDY. This experiment does NOT claim to improve the previous RRG encoder.
"""
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from core import read,write,sha,features,positions
from campaign import contexts, frozen, code_identity

class Selector(nn.Module):
    def __init__(self, hidden=48):
        super().__init__()
        self.embed=nn.Linear(41,hidden)
        self.layers=nn.ModuleList([nn.Linear(2*hidden,hidden) for _ in range(2)])
        self.head=nn.Sequential(nn.Linear(3*hidden+5,hidden),nn.ReLU(),nn.Linear(hidden,5))

    def forward(self,g):
        x=torch.relu(self.embed(g['x']))
        for layer in self.layers:
            agg=torch.zeros_like(x)
            den=x.new_zeros((len(x),1))
            agg.index_add_(0,g['dst'],x[g['src']])
            den.index_add_(0,g['dst'],x.new_ones((len(g['dst']),1)))
            x=torch.relu(layer(torch.cat((x,agg/den.clamp_min(1)),dim=1)))
        moved=g['x'][:,8:9]
        pooled=torch.cat((x.mean(0),x.max(0).values,(x*moved).sum(0)/moved.sum().clamp_min(1),g['g']))
        return self.head(pooled)

def graph(folder,name,width,device):
    cache=folder/(name+'.features.npz')
    if not cache.exists():
        g=read(folder/'graph.json')
        xy=positions(g) if name=='default' else np.load(folder/(name+'.npy'))
        values=features(g,xy,width)
        np.savez_compressed(cache,**values)
    with np.load(cache) as f:
        return {k:torch.as_tensor(f[k],device=device) for k in f.files}

def utility(row,base):
    if row['status']!='success': return 0.
    penalty=0.
    for key,weight in [('wirelength',.15),('cpd',.25),('wall_seconds',.10)]:
        if row.get(key) and base.get(key) and base['status']=='success':
            penalty+=weight*np.clip(math.log(row[key]/base[key]),-1,1)
    return 2.-float(penalty)

def load_groups(a,split):
    result=[]
    for d,w,s,folder in contexts(a,{split}):
        pool=read(folder/'pool.json')
        names=['default']+[r['name'] for r in pool['actions']]
        rows=[read(folder/n/'trial.json') for n in names]
        if not all(r['complete'] for r in rows): raise ValueError('non-routing failures in labels')
        result.append((folder,w,names,rows))
    return result

def scores(model,group,device):
    folder,w,names,_=group
    return torch.stack([model(graph(folder,n,w,device)) for n in names])

def evaluate(model,groups,device):
    model.eval()
    correct,total=0,0
    with torch.no_grad():
        for group in groups:
            output=scores(model,group,device)[:,4].cpu().tolist()
            rows=group[3]
            truth=[utility(r,rows[0]) for r in rows]
            for i in range(len(rows)):
                for j in range(i):
                    if abs(truth[i]-truth[j])<.01: continue
                    total+=1
                    correct+= .5 if abs(output[i]-output[j])<1e-6 else float((output[i]-output[j])*(truth[i]-truth[j])>0)
    return dict(pair_count=total,ranking_accuracy=correct/total if total else None)

def train(a):
    if read(a.out/'protocol.json')['code']!=code_identity(): raise ValueError('code changed after CPU label freeze')
    if not (a.out/'labels_complete.json').exists(): raise ValueError('finish CPU labels first')
    train_groups=load_groups(a,'train'); validation=load_groups(a,'validation')
    if not train_groups or not validation: raise ValueError('need disjoint training and validation families')
    device=torch.device(a.device)
    torch.manual_seed(6601)
    model=Selector().to(device); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    directory=a.out/'model';directory.mkdir(exist_ok=True)
    import hashlib
    label_digest=hashlib.sha256()
    for group in train_groups+validation:
        for name in group[2]: label_digest.update(sha(group[0]/name/'trial.json').encode())
    signature=dict(cohort_sha256=sha(a.out/'cohort.json'),protocol_sha256=sha(a.out/'protocol.json'),
        labels_sha256=label_digest.hexdigest(),
        plan_sha256=sha(a.out/'plan.json'),epochs=a.epochs,seed=6601,code_sha256=sha(__file__),
        training_groups=len(train_groups),validation_groups=len(validation))
    frozen(directory/'signature.json',signature)
    start_epoch=0; best=-1.;history=[]
    if (directory/'last.pt').exists():
        old=torch.load(directory/'last.pt',map_location=device,weights_only=False)
        model.load_state_dict(old['model']);optimizer.load_state_dict(old['optimizer'])
        start_epoch=old['epoch'];best=old['best'];history=old['history']
    for epoch in range(start_epoch,a.epochs):
        model.train(); losses=[]
        order=np.random.default_rng(6601+epoch).permutation(len(train_groups))
        for index in order:
            group=train_groups[index];rows=group[3]
            optimizer.zero_grad(set_to_none=True)
            outputs=scores(model,group,device)
            success=outputs.new_tensor([r['status']=='success' for r in rows])
            loss=nn.functional.binary_cross_entropy_with_logits(outputs[:,0],success)
            truth=[utility(r,rows[0]) for r in rows]
            pairs=[(i,j,1 if truth[i]>truth[j] else -1) for i in range(len(rows)) for j in range(i) if abs(truth[i]-truth[j])>=.01]
            if pairs:
                loss+=torch.stack([nn.functional.softplus(-sign*(outputs[i,4]-outputs[j,4])) for i,j,sign in pairs]).mean()
            # Predict log ratios only where both runs succeeded; failure runtimes are not speedups.
            for i,r in enumerate(rows):
                if r['status']!=rows[0]['status'] or r['status']!='success': continue
                for head,key in enumerate(['wirelength','cpd','wall_seconds'],1):
                    if r.get(key) and rows[0].get(key):
                        target=outputs.new_tensor(math.log(r[key]/rows[0][key]))
                        loss+=.1*nn.functional.smooth_l1_loss(outputs[i,head],target)/len(rows)
            loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
            losses.append(float(loss.detach()))
        metrics=evaluate(model,validation,device)
        item=dict(epoch=epoch+1,loss=float(np.mean(losses)),validation=metrics)
        history.append(item);print(json.dumps(dict(stage='epoch',**item)),flush=True)
        value=metrics['ranking_accuracy']
        if value is not None and value>best:
            best=value
            torch.save(dict(model=model.state_dict(),signature=signature,epoch=epoch+1),directory/'ranker.tmp')
            (directory/'ranker.tmp').replace(directory/'ranker.pt')
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),epoch=epoch+1,best=best,history=history),directory/'last.tmp')
        (directory/'last.tmp').replace(directory/'last.pt')
    if not (directory/'ranker.pt').exists(): raise ValueError('no informative validation pairs; no model approved')
    write(directory/'training_report.json',dict(complete=True,signature=signature,best_validation_accuracy=best,
        history=history,test_labels_used=False,warning='Validation families may guide development; test families remain sealed.'))

def select(a):
    if read(a.out/'protocol.json')['code']!=code_identity(): raise ValueError('code changed after freeze')
    if not (a.out/'model/training_report.json').exists(): raise ValueError('training incomplete')
    if (a.out/'selection.json').exists():
        if read(a.out/'selection.json')['model_sha256']!=sha(a.out/'model/ranker.pt'): raise ValueError('frozen model changed')
        return
    # No test candidate labels may exist before selection is frozen.
    for _,_,_,folder in contexts(a,{'test'}):
        if any((folder/r['name']/'trial.json').exists() for r in read(folder/'pool.json')['actions']):
            raise ValueError('test labels already exist without selection freeze; new cohort required')
    device=torch.device(a.device); model=Selector().to(device)
    checkpoint=torch.load(a.out/'model/ranker.pt',map_location=device,weights_only=False)
    model.load_state_dict(checkpoint['model']);model.eval()
    rows=[]; start=time.monotonic()
    with torch.no_grad():
        for d,w,s,folder in contexts(a,{'test'}):
            names=['default']+[r['name'] for r in read(folder/'pool.json')['actions']]
            output=scores(model,(folder,w,names,[]),device).cpu()
            chosen=choose(output)
            rows.append(dict(context=str(folder.relative_to(a.out)),design=d['design'],width=w,seed=s,
                             action=names[chosen],scores={n:output[i].tolist() for i,n in enumerate(names)}))
    write(a.out/'selection.json',dict(model_sha256=sha(a.out/'model/ranker.pt'),rows=rows,
        inference_seconds_per_context=(time.monotonic()-start)/max(1,len(rows)),
        gates=dict(rank_margin=.05,route_drop=.02,wire_ratio=1.03,cpd_ratio=1.03),test_labels_used=False))

def choose(output):
    base=output[0]
    eligible=[i for i in range(1,len(output)) if output[i,4]>base[4]+.05
              and torch.sigmoid(output[i,0])>=torch.sigmoid(base[0])-.02
              and output[i,1]<=math.log(1.03) and output[i,2]<=math.log(1.03)]
    return max(eligible,key=lambda i:float(output[i,4])) if eligible else 0

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['train','select'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--epochs',type=int,default=80)
    p.add_argument('--threads',type=int,default=1)
    a=p.parse_args();a.out=a.out.resolve();torch.set_num_threads(a.threads)
    globals()[a.phase](a)

if __name__=='__main__':main()
