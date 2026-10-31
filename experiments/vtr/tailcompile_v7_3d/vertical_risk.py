"""Candidate-dependent vertical demand with independently counted RR resources.

Device preparation contracts *actual directed RR edges* to coarse regions. The
result is an access upper bound: region contraction can join disconnected wires
and does not certify a particular packed pin/site. It is never a router.
"""
from __future__ import annotations

from array import array
from collections import defaultdict, deque, OrderedDict
import gzip
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

SCHEMA = 'tailcompile-v7-vertical-risk-1'
TOP_K = 4
TEMPERATURE = .1


def _hash(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def _closure(seeds, adjacency):
    seen, queue = set(seeds), deque(sorted(seeds))
    while queue:
        for other in adjacency.get(queue.popleft(), ()):
            if other not in seen:
                seen.add(other)
                queue.append(other)
    return sorted(seen)


def prepare_device(rr_graph, device, out=None, *, max_seconds=300):
    """Stream full pool member IDs and directed coarse access, once per device.

    The node table stores one footprint integer per RR ID, not Python RR edge
    objects. All original directed edges are streamed and discarded. Serial Z
    connections are explicitly unsupported in this two-layer estimator.
    """
    started = time.perf_counter()
    rr_graph = Path(rr_graph)
    signature = dict(schema=SCHEMA, implementation=_hash(__file__),
                     rr_graph_sha256=_hash(rr_graph),
                     device_sha256=hashlib.sha256(json.dumps(device, sort_keys=True,
                                                            allow_nan=False).encode()).hexdigest())
    if out is not None and Path(out).exists():
        value = json.loads(Path(out).read_text(encoding='utf-8'))
        if value.get('signature') != signature:
            raise ValueError('vertical access cache signature changed')
        return value
    width, height, layers = (int(device[k]) for k in ('grid_width', 'grid_height', 'layers'))
    if layers not in (1, 2):
        raise ValueError('vertical demand estimator is qualified for one/two layers only')
    tile_size = int(device['tile_size'])
    nx, ny = math.ceil(width/tile_size), math.ceil(height/tile_size)
    per_layer = nx*ny
    def region(x, y, z):
        if not (0 <= x < width and 0 <= y < height and 0 <= z < layers):
            raise ValueError('RR location lies outside device graph')
        return z*per_layer + (y//tile_size)*nx + x//tile_size
    footprints, footprint_index = [], {}
    node_footprint = array('i')
    vertical, segments = {}, {}
    adjacency, reverse = defaultdict(set), defaultdict(set)
    pair_cache = OrderedDict()
    node_count = edge_count = serial_z_edges = 0
    opener = gzip.open if str(rr_graph).endswith('.gz') else open
    with opener(rr_graph, 'rb') as stream:
        stack = []
        for event, item in ET.iterparse(stream, events=('start', 'end')):
            if event == 'start':
                stack.append(item)
                continue
            parent = stack[-2] if len(stack) > 1 else None
            tag = parent.tag if parent is not None else None
            if item.tag == 'segment' and tag == 'segments':
                segments[item.get('id')] = item.get('name', item.get('id'))
            elif item.tag == 'node' and tag == 'rr_nodes':
                node_count += 1
                rid = int(item.get('id'))
                if rid < len(node_footprint) or rid > 20000000:
                    raise ValueError('requires increasing RR IDs within 20M table budget')
                node_footprint.extend([-1]*(rid-len(node_footprint)+1))
                node_capacity=float(item.get('capacity','1'))
                if not math.isfinite(node_capacity) or node_capacity<0:
                    raise ValueError('RR capacity must be finite and nonnegative')
                loc = item.find('loc')
                if loc is not None:
                    coords = tuple(int(loc.get(k, '0')) for k in ('xlow','xhigh','ylow','yhigh'))
                    lo = int(loc.get('layer_low', loc.get('layer', '0')))
                    hi = int(loc.get('layer_high', loc.get('layer', '0')))
                    key = coords + (lo,hi)
                    if key not in footprint_index:
                        x0,x1,y0,y1 = coords
                        touched = tuple(sorted({region(x,y,z) for z in range(lo,hi+1)
                            for y in range(y0//tile_size*tile_size, y1+1, tile_size)
                            for x in range(x0//tile_size*tile_size, x1+1, tile_size)}))
                        footprint_index[key] = len(footprints)
                        footprints.append(touched)
                    node_footprint[rid] = footprint_index[key]
                    if node_capacity==0 and item.get('type')!='CHANZ':
                        node_footprint[rid]=-1
                    if item.get('type') == 'CHANZ':
                        cap = node_capacity
                        if lo != 0 or hi != 1 or coords[0] != coords[1] or coords[2] != coords[3]:
                            raise ValueError('requires a single-XY CHANZ crossing the two-layer interface')
                        segment = item.find('segment')
                        sid = segment.get('segment_id') if segment is not None else 'unknown'
                        vertical[rid] = dict(rr_node_id=rid, capacity=cap,
                            direction=item.get('direction', 'UNKNOWN'), segment=segments.get(sid,sid),
                            xy=[coords[0]/width,coords[2]/height],
                            regions=list(footprints[node_footprint[rid]]), incoming=set(), outgoing=set())
            elif item.tag == 'edge' and tag == 'rr_edges':
                edge_count += 1
                a,b = int(item.get('src_node')),int(item.get('sink_node'))
                if a != b:
                    if max(a,b) >= len(node_footprint):
                        raise ValueError('RR edge has undefined node')
                    if a in vertical and b in vertical:
                        serial_z_edges += 1
                    elif b in vertical:
                        if node_footprint[a] >= 0:
                            vertical[b]['incoming'].update(footprints[node_footprint[a]])
                    elif a in vertical:
                        if node_footprint[b] >= 0:
                            vertical[a]['outgoing'].update(footprints[node_footprint[b]])
                    elif min(node_footprint[a], node_footprint[b]) >= 0:
                        pair = (node_footprint[a],node_footprint[b])
                        if pair not in pair_cache:
                            for u in footprints[pair[0]]:
                                for v in footprints[pair[1]]:
                                    if u//per_layer == v//per_layer and u != v:
                                        adjacency[u].add(v)
                                        reverse[v].add(u)
                            pair_cache[pair] = True
                            if len(pair_cache) > 100000:
                                pair_cache.popitem(last=False)
            if (node_count + edge_count) % 100000 == 0 and time.perf_counter()-started > max_seconds:
                raise TimeoutError('directed vertical access preparation exceeded budget')
            if parent is not None and tag in ('rr_nodes','rr_edges','grid','segments','channels','block_types','switches'):
                parent.remove(item)
            stack.pop()
    if serial_z_edges:
        raise ValueError(f'{serial_z_edges} serial CHANZ edges require a separate capacity adapter')
    groups = {}
    closure_cache = {}
    def cached_closure(seeds, backwards=False):
        key = (backwards,tuple(sorted(seeds)))
        if key not in closure_cache:
            closure_cache[key] = _closure(seeds,reverse if backwards else adjacency)
        return closure_cache[key]
    excluded = []
    for row in vertical.values():
        orientations = []
        for source_layer, sink_layer in ((0,1),(1,0)):
            incoming = {r for r in row['incoming'] if r//per_layer == source_layer}
            outgoing = {r for r in row['outgoing'] if r//per_layer == sink_layer}
            if incoming and outgoing:
                orientations.append(dict(source_layer=source_layer,sink_layer=sink_layer,
                    source_regions=cached_closure(incoming,True), sink_regions=cached_closure(outgoing)))
        if row['capacity'] <= 0 or not orientations:
            excluded.append(row['rr_node_id'])
            continue
        # Identical access summaries are safely pooled without losing member IDs.
        key = (tuple(row['regions']), row['direction'], row['segment'],
               tuple((o['source_layer'],tuple(o['source_regions']),tuple(o['sink_regions'])) for o in orientations))
        pool = groups.setdefault(key,dict(id=len(groups),capacity=0., direction=row['direction'],
            segment=row['segment'],xy=[0.,0.],regions=row['regions'],orientations=orientations,members=[]))
        pool['capacity'] += row['capacity']
        for axis in range(2):
            pool['xy'][axis]+=row['capacity']*row['xy'][axis]
        pool['members'].append(dict(rr_node_id=row['rr_node_id'],capacity=row['capacity'],xy=row['xy']))
    for pool in groups.values():
        pool['xy']=[v/pool['capacity'] for v in pool['xy']]
        pool['xy_semantics']='independent-capacity weighted centroid within frozen leaf footprint'
    value = dict(schema=SCHEMA, signature=signature,grid=[width,height,layers],tile_size=tile_size,
        pools=list(groups.values()),region_bounds=device['region_bounds'],region_layer=device['region_layer'],
        top_k=TOP_K,temperature=TEMPERATURE,access_semantics='directed RR region-contraction upper bound; no site/pin certificate',
        demand_semantics='one shared net/pool crossing per driver-layer, target-region branches combined by noisy-OR',
        audit=dict(rr_node_count=node_count,rr_edge_count=edge_count,vertical_candidate_nodes=len(vertical),
            included_member_count=sum(len(p['members']) for p in groups.values()),excluded_member_ids=excluded,
            connected_capacity=sum(p['capacity'] for p in groups.values()),serial_z_edges=serial_z_edges,
            region_arc_count=sum(map(len,adjacency.values())),full_rr_edges_retained=0),
        extraction_seconds=time.perf_counter()-started)
    expected = device.get('audit',{}).get('unique_connected_z_capacity')
    if expected is not None and not math.isclose(value['audit']['connected_capacity'],float(expected),abs_tol=1e-6):
        raise ValueError('directed terminal-access capacity disagrees with existing RRG audit')
    if out is not None:
        from repository_guard import validate_output_roots
        from io_utils import write
        validate_output_roots((Path(out),))
        write(out,value)
    return value


def weighted_cvar(values, weights, alpha=.95):
    """Exact upper-tail weighted CVaR, including fractional boundary weight."""
    import torch
    if not 0 <= alpha < 1:
        raise ValueError('CVaR alpha must lie in [0,1)')
    if values.numel() == 0:
        return values.sum()
    if weights.shape != values.shape or (weights <= 0).any():
        raise ValueError('positive weights matching values required')
    order = torch.argsort(values,descending=True,stable=True)
    sorted_weight = weights[order]
    budget = sorted_weight.sum()*(1-alpha)
    previous = sorted_weight.cumsum(0)-sorted_weight
    used = torch.minimum(sorted_weight,(budget-previous).clamp_min(0))
    return (values[order]*used).sum()/budget


def _region_ids(xy, summary):
    import torch
    width,height,_ = summary['grid']
    size = summary['tile_size']
    nx,ny = math.ceil(width/size),math.ceil(height/size)
    # CPU integer hashing avoids serializing N*L nested Python lists every step.
    xy=xy.detach().cpu()
    x=torch.floor(xy[...,0]*width).long().clamp(0,width-1)//size
    y=torch.floor(xy[...,1]*height).long().clamp(0,height-1)//size
    layer=torch.arange(xy.shape[1])[None,:]
    return (layer*nx*ny+y*nx+x).contiguous()


def _branch_plan(data, conditional_xy):
    """Small discrete plan, cached only for the frozen conditional XY values."""
    summary = data['vertical_risk_device']
    started=time.perf_counter()
    ids_tensor = _region_ids(conditional_xy,summary)
    fingerprint = hashlib.sha256(ids_tensor.numpy().tobytes()).hexdigest()
    caches=data.setdefault('_vertical_branch_caches',OrderedDict())
    if fingerprint in caches:
        caches.move_to_end(fingerprint)
        data['_vertical_branch_cache']=caches[fingerprint]
        return caches[fingerprint]
    ids=ids_tensor.tolist()
    layers = summary['grid'][2]
    pair_candidates={}
    source_access=defaultdict(list)
    for pool in summary['pools']:
        for orientation in pool['orientations']:
            sl,tl=orientation['source_layer'],orientation['sink_layer']
            targets=frozenset(orientation['sink_regions'])
            for sr in orientation['source_regions']:
                source_access[sl,tl,sr].append((pool,targets))
    def candidates(source_region,sink_region,source_layer,target_layer):
        key=(source_region,sink_region,source_layer,target_layer)
        if key not in pair_candidates:
            sb,tb=summary['region_bounds'][source_region],summary['region_bounds'][sink_region]
            sx,sy=(sb[0]+sb[1])/2,(sb[2]+sb[3])/2
            tx,ty=(tb[0]+tb[1])/2,(tb[2]+tb[3])/2
            eligible=[]
            for pool,targets in source_access[source_layer,target_layer,source_region]:
                if sink_region in targets:
                    x,y=pool['xy']
                    eligible.append((abs(x-sx)+abs(y-sy)+abs(x-tx)+abs(y-ty),pool['id']))
            eligible=sorted(eligible)[:TOP_K]
            if eligible:
                base=eligible[0][0]
                weights=[math.exp(-(c-base)/TEMPERATURE) for c,p in eligible]
                total=sum(weights)
                pair_candidates[key]=[(pid,w/total) for (cost,pid),w in zip(eligible,weights)]
            else:
                pair_candidates[key]=[]
        return pair_candidates[key]
    branch_sinks,branch_id=[],[]
    branches=[]
    groups={}
    group_pool,group_driver,group_layer=[],[],[]
    term_branch,term_group,term_q=[],[],[]
    active=ambiguous=highfanout=0
    for ni,net in enumerate(data['risk_nets']):
        if ni%10000==0 and time.perf_counter()-started>float(data.get('risk_prepare_max_seconds',30)):
            raise TimeoutError('sparse vertical branch preparation exceeded fixed 30s/default budget')
        if net.get('ignored') or len(net['pins'])<2:
            continue
        active+=1
        highfanout+=len(net['pins'])>128
        drivers=net.get('drivers',[])
        driver=drivers[0] if drivers else net['pins'][0]
        ambiguous+=len(drivers)!=1
        sinks=sorted(set(net.get('sinks',[]) or net['pins'])-{driver})
        for sl in range(layers):
            for tl in range(layers):
                if sl==tl:
                    continue
                by_region=defaultdict(list)
                for sink in sinks:
                    by_region[ids[sink][tl]].append(sink)
                for target_region,members in sorted(by_region.items()):
                    bi=len(branches)
                    choices=candidates(ids[driver][sl],target_region,sl,tl)
                    branches.append((driver,sl,tl,bool(choices)))
                    branch_sinks.extend(members)
                    branch_id.extend([bi]*len(members))
                    for pid,q in choices:
                        groupkey=(ni,sl,pid)
                        if groupkey not in groups:
                            groups[groupkey]=len(groups)
                            group_pool.append(pid); group_driver.append(driver); group_layer.append(sl)
                        term_branch.append(bi); term_group.append(groups[groupkey]); term_q.append(q)
    value=dict(fingerprint=fingerprint,branches=branches,branch_sinks=branch_sinks,branch_id=branch_id,
        group_pool=group_pool,group_driver=group_driver,group_layer=group_layer,term_branch=term_branch,
        term_group=term_group,term_q=term_q,active=active,ambiguous=ambiguous,highfanout=highfanout)
    # Transient caches must not be written into the graph artifact by callers.
    caches[fingerprint]=value
    if len(caches)>2:
        caches.popitem(last=False)
    data['_vertical_branch_cache']=value
    return value


def risk_loss(data, layer_prob, conditional_xy):
    """Differentiable layer-probability surrogate; all topology is pre-route.

    Returns CVaR and diagnostics. Pop ``unassigned_mass_tensor`` before JSON
    serialization; it is a distinct zero-capacity constraint penalty, never a
    fake epsilon-normalized utilization. No gradient through topology/decoder.
    """
    import torch
    if 'vertical_risk_device' not in data or 'risk_nets' not in data:
        raise ValueError('007 risk requires versioned vertical access and complete risk nets')
    summary=data['vertical_risk_device']
    n,layers=layer_prob.shape
    if conditional_xy.shape != (n,layers,2) or layers!=summary['grid'][2]:
        raise ValueError('risk requires N x L probabilities and N x L x 2 conditional XY')
    if not torch.isfinite(layer_prob).all() or (layer_prob<0).any() or not torch.allclose(
            layer_prob.sum(1),torch.ones(n,device=layer_prob.device),atol=1e-5):
        raise ValueError('invalid normalized layer probabilities')
    if not torch.isfinite(conditional_xy).all() or (conditional_xy<0).any() or (conditional_xy>1).any():
        raise ValueError('invalid normalized conditional XY')
    plan=_branch_plan(data,conditional_xy)
    device=layer_prob.device
    tensor_key=(str(device),str(layer_prob.dtype))
    buffers=plan.setdefault('_tensor_buffers',{})
    if tensor_key not in buffers:
        ints=('branches','branch_id','branch_sinks','group_pool','group_driver','group_layer','term_branch','term_group')
        buffers[tensor_key]={name:torch.tensor(plan[name],device=device,dtype=torch.long) for name in ints}
        buffers[tensor_key]['term_q']=layer_prob.new_tensor(plan['term_q'])
    tensors=buffers[tensor_key]
    def idx(name): return tensors[name]
    zero=layer_prob.sum()*0
    cap=layer_prob.new_tensor([p['capacity'] for p in summary['pools']])
    demand=torch.zeros_like(cap)
    unassigned=zero
    if plan['branches']:
        branches=idx('branches')
        bid=idx('branch_id')
        sink_prob=layer_prob[idx('branch_sinks'),branches[bid,2]]
        # Shared target-region branches: probability that at least one sink
        # needs this layer. No fanout multiplication of a shared vertical wire.
        log_not=torch.zeros(len(branches),device=device,dtype=layer_prob.dtype)
        log_not.index_add_(0,bid,torch.log1p(-sink_prob.clamp(max=1-1e-7)))
        branch_prob=-torch.expm1(log_not)
        missing=~branches[:,3].bool()
        if missing.any():
            unassigned=(layer_prob[branches[missing,0],branches[missing,1]]*branch_prob[missing]).sum()
        if plan['group_pool']:
            term=branch_prob[idx('term_branch')]*tensors['term_q']
            group_log=torch.zeros(len(plan['group_pool']),device=device,dtype=layer_prob.dtype)
            group_log.index_add_(0,idx('term_group'),torch.log1p(-term.clamp(max=1-1e-7)))
            usage=layer_prob[idx('group_driver'),idx('group_layer')]*(-torch.expm1(group_log))
            demand.index_add_(0,idx('group_pool'),usage)
    rho=demand/cap if cap.numel() else cap
    cvar=weighted_cvar(rho,cap) if cap.numel() else zero
    def quantile(q):
        if not len(rho): return 0.
        order=torch.argsort(rho)
        cum=cap[order].cumsum(0)/cap.sum()
        at=int(torch.searchsorted(cum,layer_prob.new_tensor(q)).clamp(max=len(rho)-1))
        return float(rho[order[at]].detach())
    details=dict(schema=SCHEMA,cvar95=float(cvar.detach()),p95=quantile(.95),p99=quantile(.99),
        maximum=float(rho.max().detach()) if len(rho) else 0.,
        expected_vertical_usage=float(demand.sum().detach()),pool_demand=demand.detach().cpu().tolist(),
        pool_pressure=rho.detach().cpu().tolist(),pool_capacity=cap.detach().cpu().tolist(),
        unassigned_expected_crossings=float(unassigned.detach()),
        unassigned_fraction=float(unassigned.detach())/max(1,plan['active']),
        active_net_count=plan['active'],ordinary_high_fanout_nets_included=plan['highfanout'],
        ambiguous_driver_nets=plan['ambiguous'],pool_count=len(cap),branch_count=len(plan['branches']),
        ignored_net_count=sum(bool(net.get('ignored')) for net in data['risk_nets']),
        unassigned_mass_tensor=unassigned,
        access_semantics=summary['access_semantics'],
        demand_semantics=summary['demand_semantics'])
    return cvar,details


def hard_risk(data, layer_assignment, xy):
    """Evaluate the final hard initial layer/XY; returns JSON-only diagnostics."""
    import torch
    layer_assignment=torch.as_tensor(layer_assignment,device=xy.device,dtype=torch.long)
    layers=data['vertical_risk_device']['grid'][2]
    probability=torch.nn.functional.one_hot(layer_assignment,layers).to(xy)
    conditional=xy[:,None,:].expand(-1,layers,-1)
    with torch.no_grad():
        _,detail=risk_loss(data,probability,conditional)
    detail.pop('unassigned_mass_tensor')
    return detail


def route_pool_summary(summary, tsv):
    """Join hook RR occupancy to all independent pool members, never RR edges.

    Missing members make the summary incomplete. Pool-average overuse and the
    sum of *member* overuse are both recorded, as aggregation can hide a hot
    individual track. Excluded/disconnected CHANZ nodes are audited separately.
    """
    import re
    observed={}
    iteration=None
    for line in Path(tsv).read_text().splitlines():
        if not line.strip():
            continue
        if line.startswith('#'):
            match=re.search(r'iteration=(\d+)',line)
            if match: iteration=int(match.group(1))
            continue
        raw,cap,occ=line.split('\t')
        rid=int(raw);capacity=float(cap);occupancy=float(occ)
        if rid in observed or not all(math.isfinite(v) and v>=0 for v in (capacity,occupancy)):
            raise ValueError('duplicate or invalid routed vertical RR occupancy')
        observed[rid]=(capacity,occupancy)
    mapped=set()
    pools=[]
    missing=[]
    for pool in summary['pools']:
        complete=True;total=member_overuse=0.;overused=0
        for member in pool['members']:
            rid=member['rr_node_id']
            if rid in mapped:
                raise ValueError('vertical RR member belongs to multiple pools')
            mapped.add(rid)
            if rid not in observed:
                missing.append(rid);complete=False
                continue
            cap,occ=observed[rid]
            if not math.isclose(cap,float(member['capacity']),abs_tol=1e-6):
                raise ValueError('route occupancy capacity differs from frozen pool member')
            total+=occ
            member_overuse+=max(0.,occ-cap)
            overused+=occ>cap
        pools.append(dict(pool_id=pool['id'],capacity=pool['capacity'],complete=complete,
            occupancy=total if complete else None,
            utilization=total/pool['capacity'] if complete else None,
            aggregated_overuse=max(0.,total-pool['capacity']) if complete else None,
            member_overuse=member_overuse if complete else None,
            overused_member_count=overused if complete else None))
    available=[p for p in pools if p['complete']]
    total_capacity=sum(p['capacity'] for p in available)
    def cvar():
        if not total_capacity: return None
        budget=.05*total_capacity
        left=budget;value=0.
        for pool in sorted(available,key=lambda p:p['utilization'],reverse=True):
            taken=min(left,pool['capacity']);value+=taken*pool['utilization'];left-=taken
            if left<=1e-12: break
        return value/budget
    excluded=set(summary.get('audit',{}).get('excluded_member_ids',[]))
    return dict(schema='tailcompile-v7-routed-vertical-pools-1',iteration=iteration,
        complete=not missing and not (set(observed)-mapped-set(summary.get('audit',{}).get('excluded_member_ids',[]))),
        pools=pools,pool_count=len(pools),complete_pool_count=len(available),
        matched_member_count=len(mapped)-len(missing),missing_member_ids=missing,
        observed_rr_count=len(observed),excluded_observed_rr_ids=sorted((set(observed)-mapped)&excluded),
        unmatched_observed_rr_ids=sorted(set(observed)-mapped-excluded),
        utilization_cvar95=cvar(),
        member_overuse=sum(p['member_overuse'] for p in available),
        interpretation='static routing occupancy, not runtime traffic; tail conditional on complete pools')


def benchmark(node_counts=(10000,100000), repeats=3):
    """Sparse synthetic scaling check, not FPGA performance evidence."""
    import torch
    torch.set_num_threads(1)
    results=[]
    region_bounds=[[x/16,(x+1)/16,y/16,(y+1)/16] for y in range(16) for x in range(16)]*2
    pools=[]
    for i in range(32):
        pools.append(dict(id=i,capacity=32.,xy=[((i*7)%16+.5)/16,((i*11)%16+.5)/16],
            orientations=[dict(source_layer=sl,sink_layer=1-sl,
                source_regions=list(range(sl*256,(sl+1)*256)),
                sink_regions=list(range((1-sl)*256,(2-sl)*256))) for sl in (0,1)]))
    summary=dict(schema=SCHEMA,grid=[64,64,2],tile_size=4,pools=pools,
        region_bounds=region_bounds,region_layer=[0]*256+[1]*256,
        access_semantics='synthetic all-region directed upper bound',demand_semantics='shared net/pool noisy-OR')
    for n in node_counts:
        generator=torch.Generator().manual_seed(2801)
        data=dict(vertical_risk_device=summary,risk_nets=[dict(id=i,pins=[i,(i+17)%n],
            drivers=[i],sinks=[(i+17)%n],ignored=False) for i in range(n)])
        xy=torch.rand((n,2,2),generator=generator)
        p=torch.rand((n,2),generator=generator);p=p/p.sum(1,keepdim=True)
        samples=[]
        for repeat in range(repeats):
            logits=p.detach().clone().requires_grad_(True)
            started=time.perf_counter()
            loss,detail=risk_loss(data,logits,xy)
            loss.backward()
            samples.append(time.perf_counter()-started)
        plan=data['_vertical_branch_cache']
        results.append(dict(node_count=n,net_count=n,pool_count=len(pools),
            branch_count=len(plan['branches']),sparse_term_count=len(plan['term_branch']),
            first_seconds=samples[0],cached_seconds=samples[1:],finite_gradient=bool(torch.isfinite(logits.grad).all()),
            prepare_budget_seconds=30,iterations=repeats))
    return dict(schema='tailcompile-v7-vertical-risk-scaling-1',torch_version=torch.__version__,
        threads=1,source_sha256=_hash(__file__),results=results,
        scope='synthetic estimator complexity only; no VPR/QoR claim')


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if not args.benchmark: parser.error('use --benchmark')
    from repository_guard import validate_output_roots
    from io_utils import write
    validate_output_roots((args.out,))
    write(args.out,benchmark())
