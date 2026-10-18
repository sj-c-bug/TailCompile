"""Frozen, label-free AP-block local refinement shared by routes C and D."""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent/'tailcompile_v66'))
from core import snapshot, candidates, flat, write


def choose(pool):
    """Same proxy for both origins; replay wins ties. Never inspect VPR labels."""
    base = pool[0][2]
    def score(item):
        f = item[2]
        return f['tail']/max(base['tail'], 1e-9) + f['hpwl']/max(base['hpwl'], 1e-9)
    return min(range(len(pool)), key=lambda i: (score(pool[i]), i))


def run(prefix, output, width):
    start = time.perf_counter()
    graph = snapshot(prefix)
    loaded = time.perf_counter()
    pool = candidates(graph, width, replicas=2)
    selected = choose(pool)
    searched = time.perf_counter()
    flat(output, graph, pool[selected][1])
    ended = time.perf_counter()
    write(Path(output).with_suffix('.selection.json'), dict(
        action=pool[selected][0], moved=pool[selected][2]['moved'],
        policy='min normalized RUDY-tail + normalized HPWL; replay wins ties',
        reads_routing_labels=False, candidate_count=len(pool),
        snapshot_seconds=loaded-start, search_seconds=searched-loaded,
        lowering_seconds=ended-searched, worker_seconds=ended-start,
        candidates=[dict(name=name, **features) for name, _, features in pool]))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prefix', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--width', type=int, required=True)
    a = p.parse_args()
    run(a.prefix, a.output, a.width)
