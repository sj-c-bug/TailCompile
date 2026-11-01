"""Compact per-net route resource accounting before disposable route cleanup.

Tile-coordinate lengths are not micrometres. Missing Z coordinates are reported
as unknown, never silently interpreted as a zero-hop vertical connection.
"""
from collections import Counter
from pathlib import Path
import re


def summarize_route(path):
    path=Path(path)
    net=None; resources={}; duplicates=0; without_net=0
    with path.open(encoding='utf-8',errors='replace') as stream:
        for line in stream:
            header=re.match(r'^Net\s+(\d+)\b',line)
            if header: net=int(header[1])
            match=re.search(r'Node:\s*(\d+)\s+(CHANX|CHANY|CHANZ)\b',line)
            if not match: continue
            if net is None: without_net+=1; continue
            key=(net,int(match[1]))
            if key in resources: duplicates+=1; continue
            points=re.findall(r'\(\s*(-?\d+)\s*,\s*(-?\d+)(?:\s*,\s*(-?\d+))?\s*\)',line[match.end():])
            points=[(int(x),int(y),int(z) if z else None) for x,y,z in points]
            resources[key]=(match[2],points)
    xy_span=0; xy_unknown=0; z_hops=0; z_unknown=0
    for kind,points in resources.values():
        if kind in ('CHANX','CHANY'):
            if not points: xy_unknown+=1; continue
            a,b=points[0],points[-1]
            xy_span+=abs(a[0]-b[0])+abs(a[1]-b[1])+1
        elif len(points)>=2 and points[0][2] is not None and points[-1][2] is not None:
            z_hops+=abs(points[0][2]-points[-1][2])
        else: z_unknown+=1
    kinds=Counter(kind for kind,_ in resources.values())
    return dict(schema='qor-route-usage-009-1',source=str(path),net_rr_node_counts=dict(kinds),
        unique_chanz_nodes=len({node for (net,node),(kind,_) in resources.items() if kind=='CHANZ'}),
        per_net_unique_chanz_nodes=kinds['CHANZ'],repeated_branch_rows_ignored=duplicates,
        xy_tile_span_sum=xy_span if not xy_unknown else None,xy_coordinate_missing=xy_unknown,
        z_hop_sum=z_hops if not z_unknown else None,z_coordinate_missing=z_unknown,
        rows_without_net_header=without_net,physical_wirelength_micrometres=None,
        configured_vertical_capacity_not_inferred=True,capacity_gt_one_slot_occupancy_not_inferred=True)
