"""Read durable qualification evidence; summarize real vertical RR-node usage."""
from __future__ import annotations
import argparse
from pathlib import Path
from io_utils import read, write, sha


def node_usage(path):
    path=Path(path)
    if not path.is_file(): return dict(present=False)
    nodes={}
    for line in path.read_text().splitlines():
        if line.startswith('#') or not line.strip(): continue
        node,capacity,occupancy=map(int,line.split())
        if node in nodes: raise ValueError('duplicate RR node in usage dump')
        nodes[node]=(capacity,occupancy)
    positive=[(occupancy/capacity,capacity) for capacity,occupancy in nodes.values() if capacity>0]
    total=sum(cap for _,cap in positive)
    remaining=.05*total
    tail_mass=0.
    for value,weight in sorted(positive,reverse=True):
        mass=min(remaining,weight); tail_mass+=value*mass; remaining-=mass
        if remaining<=1e-12: break
    return dict(present=True,sha256=sha(path),independent_rr_nodes=len(nodes),capacity=total,
        occupied_nodes=sum(occupancy>0 for _,occupancy in nodes.values()),
        occupancy=sum(occupancy for _,occupancy in nodes.values()),
        overuse=sum(max(0,occupancy-capacity) for capacity,occupancy in nodes.values()),
        capacity_weighted_cvar95=tail_mass/(.05*total) if total else None,
        level='independent CHANZ RR nodes; not the model regional pool partition')


def audit(out,backend):
    out,backend=Path(out),Path(backend)
    qualified=read(out/'qualification_report.json')
    identity=read(backend/'backend_manifest.json')
    rows=[]
    for row in qualified['trials']:
        folder=out/'deployments'/row['case']/f's{row["seed"]}'/row['mode']
        rows.append(dict(case=row['case'],mode=row['mode'],seed=row['seed'],
            passed=row['passed'],status=row['status'],wall_seconds=row['metrics']['wall_seconds'],
            cpd_ns=row['metrics'].get('cpd_ns'),wirelength=row['metrics'].get('wirelength'),
            first=node_usage(folder/'z_usage.first.tsv'),last_iteration=node_usage(folder/'z_usage.last.tsv')))
    cleanup=[read(path) for path in (out/'deployments').glob('*/s*/*/vpr/attempts/*/payload_pruned.json')]
    report=dict(schema='tailcompile-layer-backend-audit-007',
        passed=qualified['passed'],pnr_count=qualified['pnr_count'],
        qualification_report_sha256=sha(out/'qualification_report.json'),
        binary_sha256=identity['binary_sha256'],
        binary_matches_qualification=identity['binary_sha256']==qualified['binary_sha256'],
        original_backend_unchanged=identity['original_unchanged'],
        cleanup=dict(completed_attempts=len(cleanup),removed_bytes=sum(r['removed_bytes'] for r in cleanup)),
        interpretation='Engineering qualification only. Times are reported, not controlled performance evidence. Last iteration precedes any later restoration of best routing.',
        trials=rows)
    write(out/'backend_qualification_audit.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--backend',type=Path,required=True)
    a=p.parse_args(); r=audit(a.out,a.backend)
    print({k:r[k] for k in ('passed','pnr_count','binary_matches_qualification','original_backend_unchanged','cleanup')})


if __name__=='__main__': main()
