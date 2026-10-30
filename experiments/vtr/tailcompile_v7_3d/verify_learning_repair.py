"""Bounded local evidence; synthetic AP workloads, no VPR performance claim."""
import argparse
import cProfile
import io
import os
from pathlib import Path
import platform
import pstats
import statistics
import time
import torch
import data
from fast_graph import make_graph, semantic_digest
from io_utils import sha, write
from repository_guard import validate_output_roots
from test_learning_repair import graph_fixture, trial
from reward_v2 import outcome_reward, pairs
from train import outcome_reward as clipped


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    validate_output_roots((args.out,))
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    rows=[]
    for nodes in (20000,125796):
        logic,device=graph_fixture(n=nodes,regions=128)
        digests=set()
        for rep in range(3):
            for engine in (('legacy','fast') if rep%2==0 else ('fast','legacy')):
                started=time.perf_counter()
                graph=data.make_graph(logic,device) if engine=='legacy' else make_graph(logic,device)
                seconds=time.perf_counter()-started
                digests.add(semantic_digest(graph));del graph
                rows.append(dict(nodes=nodes,regions=128,engine=engine,repetition=rep,seconds=seconds))
        if len(digests)!=1:
            raise AssertionError('semantic graph inequality')
        print(f'checked nodes={nodes} exact_equal=True',flush=True)
    profiler=cProfile.Profile()
    logic,device=graph_fixture(n=20000,regions=128)
    profiler.runcall(data.make_graph,logic,device)
    stream=io.StringIO()
    pstats.Stats(profiler,stream=stream).strip_dirs().sort_stats('cumulative').print_stats(15)
    base=[trial()]
    old=[clipped([trial(seconds=t)],base) for t in (150,200,400)]
    new=[outcome_reward([trial(seconds=t)],base) for t in (150,200,400)]
    summary=[]
    for n in (20000,125796):
        med={e:statistics.median(r['seconds'] for r in rows if r['nodes']==n and r['engine']==e)
             for e in ('legacy','fast')}
        summary.append(dict(nodes=n,**med,speedup=med['legacy']/med['fast']))
    here=Path(__file__).resolve().parent
    write(args.out/'local_performance_report.json',dict(schema='repair-local-evidence-1',
        host=platform.platform(),torch=torch.__version__,threads=1,affinity=sorted(os.sched_getaffinity(0)),
        scope='synthetic graph kernel only; not native MCML, parsing/subprocess/VPR excluded',
        exact_graph_equivalence=True,rows=rows,medians=summary,legacy_profile=stream.getvalue(),
        synthetic_reward=dict(clipped=old,guarded=new,guarded_pairs=pairs(new)),
        code={n:sha(here/n) for n in ('data.py','fast_graph.py','reward_v2.py','train_repair.py')}))
    print(summary,flush=True)


if __name__=='__main__':
    main()
