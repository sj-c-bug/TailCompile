"""Variable-size hierarchy and candidate-conditioned logic/device matching.

This module consumes pre-route features only. Shared routing pools are encoded
once and broadcast to their incident regions; capacities are never interpreted
as independent parallel copies at each crossed boundary.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import math
from collections import defaultdict
import torch
from torch import nn
from tailcompile_v3 import Tower, GraphLayer, graph_inputs
from tailcompile_v6 import spatial_proxy

SCHEMA = 'tailcompile-v61-hierarchy-matching-1'


def pool_mean(x, membership, count):
    total = x.new_zeros((count, x.shape[-1]))
    total.index_add_(0, membership, x)
    sizes = torch.bincount(membership, minlength=count).to(x).clamp_min(1)
    return total / sizes[:, None]


def hierarchy(device):
    """Geometry-defined parents; works for rectangular, non-power-of-two grids."""
    coords = [(int(r['column']), int(r['row'])) for r in device['regions']]
    levels = []
    while len(coords) > 1:
        parents = [(x // 2, y // 2) for x, y in coords]
        unique = sorted(set(parents), key=lambda p: (p[1], p[0]))
        if len(unique) == len(coords):
            break
        ids = {xy: i for i, xy in enumerate(unique)}
        levels.append(torch.tensor([ids[xy] for xy in parents], dtype=torch.long))
        coords = unique
    return levels


def static_graph(logic, device, blif, supply):
    data = graph_inputs(logic, device, blif, int(supply['channel_width']))
    data['parents'] = hierarchy(device)
    data['resource'] = torch.tensor([{'clb': 0, 'memory': 1, 'dsp_top': 2}.get(c['resource'], 3)
                                     for c in logic['clusters']], dtype=torch.long)
    data['capacity'] = torch.tensor([[r['tile_counts'].get(t, 0)
                                      for t in ('clb', 'memory', 'dsp_top', 'io')]
                                     for r in device['regions']], dtype=torch.float32)
    # Directed connectivity from actual wire footprints, not symmetric switch counts.
    arcs = defaultdict(float)
    pool_features, pool_ids, pool_regions = [], [], []
    for i, p in enumerate(supply['pools']):
        name = str(p['segment']).upper()
        pool_features.append([math.log1p(p['capacity']), float(p['kind'] == 'CHANX'),
                              float(p['kind'] == 'CHANY'), float(p['direction'] == 'INC_DIR'),
                              float(p['direction'] == 'DEC_DIR'), float(p['direction'] == 'BI_DIR'),
                              float('L4' in name), float('L16' in name)])
        for a, b in p['arcs']:
            arcs[int(a), int(b)] += p['capacity']
        for region in sorted({int(r) for arc in p['arcs'] for r in arc}):
            pool_ids.append(i)
            pool_regions.append(region)
    data.update(region_src=torch.tensor([a for a, b in arcs], dtype=torch.long),
                region_dst=torch.tensor([b for a, b in arcs], dtype=torch.long),
                region_weight=torch.tensor(list(arcs.values()), dtype=torch.float32),
                pool_x=torch.tensor(pool_features, dtype=torch.float32).reshape(-1, 8),
                pool_ids=torch.tensor(pool_ids, dtype=torch.long),
                pool_regions=torch.tensor(pool_regions, dtype=torch.long))
    return data


def candidate_graph(base, logic, device, evaluator, assignment):
    spatial, global_x = spatial_proxy(logic, device, evaluator, assignment)
    state = evaluator.state(assignment)
    result = dict(base)
    result.update(assignment=torch.tensor(assignment, dtype=torch.long),
                  spatial_region_x=spatial, spatial_global_x=global_x,
                  pool_rho=torch.tensor([d / c for d, c in zip(state['loads'], evaluator.capacities)],
                                        dtype=torch.float32))
    return result


def to_device(data, device):
    return {k: v.to(device) if torch.is_tensor(v) else
            [item.to(device) for item in v] if k == 'parents' else v
            for k, v in data.items()}


class HierarchicalMatcher(nn.Module):
    def __init__(self, cluster_width=8, region_width=13, hidden=48):
        super().__init__()
        self.logic = Tower(cluster_width, hidden)
        self.region = nn.Linear(region_width, hidden)
        self.pool = nn.Sequential(nn.Linear(9, hidden), nn.ReLU(), nn.Linear(hidden, hidden))
        self.match = nn.Sequential(nn.Linear(2 * hidden + 7, hidden), nn.ReLU())
        self.fuse = nn.Sequential(nn.Linear(3 * hidden + 11, hidden), nn.ReLU())
        self.local = GraphLayer(hidden)
        self.up = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU())
        self.down = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.LayerNorm(hidden))
        self.readout = nn.Sequential(nn.Linear(4 * hidden + 9, hidden), nn.ReLU())
        self.route = nn.Linear(hidden, 1)
        self.qor = nn.Linear(hidden, 4)
        self.rank = nn.Linear(hidden, 1)
        self.hotspot = nn.Linear(hidden, 1)

    def forward(self, data):
        logic = self.logic(data['cluster_x'], data['logic_src'], data['logic_dst'], data['logic_weight'])
        region = torch.relu(self.region(data['region_x']))
        assignment, count = data['assignment'], len(region)
        spatial = torch.log1p(data['spatial_region_x'].clamp_min(0))
        # Each logical cluster explicitly interacts with its assigned resource neighborhood.
        match = self.match(torch.cat((logic, region[assignment], spatial[assignment]), 1))
        assigned = pool_mean(match, assignment, count)
        use = region.new_zeros((count, 4))
        use.index_add_(0, assignment, torch.nn.functional.one_hot(data['resource'], 4).to(region))
        utilization = use / data['capacity'].clamp_min(1)
        pool = self.pool(torch.cat((data['pool_x'], torch.log1p(data['pool_rho'])[:, None]), 1))
        pool_context = pool_mean(pool[data['pool_ids']], data['pool_regions'], count)
        x = self.fuse(torch.cat((region, assigned, pool_context, spatial, utilization), 1))
        x = self.local(x, data['region_src'], data['region_dst'], data['region_weight'])
        levels = [x]
        for parent in data['parents']:
            levels.append(self.up(pool_mean(levels[-1], parent, int(parent.max()) + 1)))
        x = levels[-1]
        for i in reversed(range(len(data['parents']))):
            x = self.down(torch.cat((levels[i], x[data['parents'][i]]), 1))
        summary = torch.cat((x.mean(0), x.amax(0), match.mean(0), match.amax(0),
                             data['spatial_global_x']))
        context = self.readout(summary)
        return dict(route_logit=self.route(context).squeeze(), qor=self.qor(context),
                    score=self.rank(context).squeeze(), hotspot=self.hotspot(x).squeeze(-1),
                    embedding=context)


def preference(left, right, epsilon=1e-4):
    """Success fraction first; on ties compare log-ratio QoR with timing protection.

    No failed-run QoR imputation. Threshold is fixed before training.
    """
    delta = left['route'] - right['route']
    if abs(delta) > epsilon:
        return 1 if delta > 0 else -1
    if any(v is None for v in left['qor'] + right['qor']):
        return 0
    # Two congestion metrics, wirelength, and CPD, all dimensionless log ratios.
    delta = sum(w * math.log(max(b, 1e-9) / max(a, 1e-9))
                for w, a, b in zip((0.25, 0.15, 0.20, 0.40), left['qor'], right['qor']))
    return (1 if delta > 0 else -1) if abs(delta) > epsilon else 0


def swap_proposal(assignment, logic, evaluator, rng, reference, max_moved_fraction=0.10):
    movable = [c['id'] for c in logic['clusters'] if c['resource'] != 'io']
    if len(movable) < 2:
        return None
    a = rng.choice(movable)
    partners = [b for b in movable if b != a and assignment[b] != assignment[a]
                and logic['clusters'][b]['resource'] == logic['clusters'][a]['resource']]
    if not partners:
        return None
    b = rng.choice(partners)
    proposal = list(assignment)
    proposal[a], proposal[b] = proposal[b], proposal[a]
    if sum(x != y for x, y in zip(proposal, reference)) > math.floor(len(movable) * max_moved_fraction):
        return None
    return proposal if evaluator.legal(proposal, fixed=reference) else None
