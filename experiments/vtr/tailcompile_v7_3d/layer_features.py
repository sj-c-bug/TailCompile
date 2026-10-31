"""007-only directional logic and vertical-access enrichment of frozen graphs."""
from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent


def augment_graph(graph, logic, device, rr_summary, blif, prefix=None):
    """Return enriched graph without changing frozen XY feature widths.

    BLIF depth is an explicitly named unit-delay proxy, never VPR slack. AP
    exported pin directions take precedence; unknown drivers are audited.
    Ordinary high-fanout nets are retained even when old GNN omitted messages.
    """
    import torch
    sys.path.insert(0,str(HERE.parent))
    from tailrisk_core import parse_blif
    from tailcompile_v2 import contract_edges
    from vertical_risk import SCHEMA
    if rr_summary.get('schema')!=SCHEMA or rr_summary['grid']!=list(logic['grid']):
        raise ValueError('007 vertical access summary/grid mismatch')
    atom_to_ap={atom:c['id'] for c in logic['clusters'] for atom in c['atoms']}
    proxy_by_pins=defaultdict(list)
    proxy_error=None
    audit={}
    try:
        primitives,parse_audit=parse_blif(Path(blif))
        edges,edge_audit=contract_edges(primitives,atom_to_ap)
        audit=dict(parser=parse_audit,edges=edge_audit)
        for edge in edges:
            proxy_by_pins[tuple(edge.pins)].append(edge)
    except ValueError as exc:
        # Unsupported combinational cycles/ports are visible missing proxy;
        # structural AP nets still enter the demand estimator unchanged.
        proxy_error=str(exc)
    pin_directions=[]
    if prefix is not None and Path(str(prefix)+'.pins.tsv').is_file():
        raw_to_id={int(c['ap_block_id']):int(c['id']) for c in logic['clusters']}
        by_net=defaultdict(lambda:dict(drivers=set(),sinks=set()))
        for line in Path(str(prefix)+'.pins.tsv').read_text().splitlines():
            net,raw,driver=line.split('\t')
            if int(raw) not in raw_to_id:
                raise ValueError('AP direction snapshot contains unknown block')
            by_net[net]['drivers' if int(driver) else 'sinks'].add(raw_to_id[int(raw)])
        for line in Path(str(prefix)+'.nets.tsv').read_text().splitlines():
            parts=line.split('\t')
            pins=sorted({raw_to_id[int(p)] for p in parts[1:]})
            if len(pins)>1:
                pin_directions.append((pins,by_net[parts[0]]))
        if [row[0] for row in pin_directions]!=[sorted(pins) for pins in logic['nets']]:
            raise ValueError('AP directional net order differs from structural nets')
    risk_nets=[]
    unknown=high=0
    critical_lookup={}
    for ni,pins in enumerate(logic['nets']):
        pins=sorted(set(pins))
        matches=proxy_by_pins.get(tuple(pins),[])
        selected=max(matches,key=lambda e:e.criticality) if matches else None
        if pin_directions:
            directions=pin_directions[ni][1]
            drivers,sinks=sorted(directions['drivers']),sorted(directions['sinks'])
        else:
            drivers=list(selected.drivers) if selected else []
            sinks=list(selected.sinks) if selected else []
        critical=float(selected.criticality) if selected else 0.
        unknown+=selected is None
        high+=len(pins)>128
        risk_nets.append(dict(id=ni,pins=pins,drivers=drivers,sinks=sinks,
            weight=1.+.5*critical,criticality=critical,proxy_known=selected is not None,
            ignored=False,ignore_reason=None))
        # Match the old bounded message expansion, without a high-fanout clique.
        if len(pins)<=16:
            pairs=((a,b) for a in pins for b in pins if a!=b)
        elif len(pins)<=128:
            pairs=(pair for a,b in zip(pins,pins[1:]+pins[:1]) for pair in ((a,b),(b,a)))
        else:
            pairs=()
        for key in pairs:
            critical_lookup[key]=max(critical_lookup.get(key,0.),critical)
    value=dict(graph)
    value['risk_nets']=risk_nets
    value['vertical_risk_device']=rr_summary
    value['logic_criticality']=torch.tensor([critical_lookup.get((a,b),0.) for a,b in
        zip(graph['logic_src'].tolist(),graph['logic_dst'].tolist())],dtype=torch.float32)
    value['criticality_semantics']='pre-placement unit-delay topological-depth proxy, NOT measured VPR timing'
    value['layer_feature_audit']=dict(schema='tailcompile-v7-layer-features-1',
        logic_net_count=len(risk_nets),ordinary_high_fanout_nets_retained=high,
        unknown_depth_proxy_nets=unknown,depth_proxy_error=proxy_error,
        direction_source='native AP pins snapshot' if pin_directions else 'mapped BLIF contraction; fallback explicitly audited',
        unknown_driver_nets=sum(len(n['drivers'])!=1 for n in risk_nets),
        ignored_globals='VPR net_is_ignored already omitted from AP snapshot; mapped clock list in depth audit',
        movable_teacher_coordinates_used=False,cluster_width_unchanged=True,region_width_unchanged=True,
        depth_audit=audit)
    return value
