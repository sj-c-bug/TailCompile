"""Trainable, variable-resolution hierarchical GP initializer (v6.8).

This is a *generator*, not the v6.1 candidate scorer.  Its logic nodes are AP
blocks/molecules from the pre-GP snapshot.  Device leaves are fixed-size tile
windows; their count is not restricted to 16.  The up/down hierarchy and
shared-pool incidence follow tailcompile_v61, without candidate placement input.

Data contract (all tensors on the model device):
  cluster_x[N,F], logic_src/dst[E], logic_weight[E];
  region_x[R,G], region_src/dst[A], region_weight[A];
  parents: list of child->parent long tensors, fine to coarse;
  pool_x[P,8], pool_ids[I], pool_regions[I] (one encoded shared wire pool);
  region_bounds[R,4]: (x_low,x_high,y_low,y_high), GLOBAL normalized [0,1];
  demand[N,T], capacity[R,T]: nonnegative, identical explicit proxy units;
  movable[N] optional bool; fixed_region[N] optional (-1 means not allocated);
  fixed_xy[N,2] optional global normalized coordinates;
  channel_width optional scalar; logic_criticality[E] optional nonnegative.

Teacher xy and output xy are GLOBAL normalized coordinates, not region-local.
Coordinates are conditioned on the selected device leaf.  No physical site is
snapped or claimed packing-legal: VPR GP/legalization retains that responsibility.
Regional capacities are upper bounds, never proportional equality targets.
They are proxy demands for AP molecules, NOT a proof of CLB packing feasibility.

forward is a convenient dense-logit API for small tests.  loss/predict use
chunked matching, and predict retains only top-k region candidates on CPU.  No
AP-node x physical-site matrix is constructed anywhere.
"""
from __future__ import annotations

import math
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

SCHEMA = 'tailcompile-v68-hierarchical-initializer-1'


def to_device(data, device):
    """Move a graph including arbitrary-depth hierarchy to CPU or CUDA."""
    return {key: value.to(device) if torch.is_tensor(value) else
            [item.to(device) for item in value] if key == 'parents' else value
            for key, value in data.items()}


def pool_mean(x, membership, count):
    total = x.new_zeros((count, x.shape[-1]))
    if membership.numel():
        total.index_add_(0, membership, x)
    sizes = torch.bincount(membership, minlength=count).to(x).clamp_min(1)
    return total / sizes[:, None]


def coordinate_interior(bounds):
    """Inset float32 leaves to respect native lower-index boundary ownership."""
    low = bounds[:, [0, 2]].float().clone()
    high = bounds[:, [1, 3]].float().clone()
    original_low, original_high = low.clone(), high.clone()
    for _ in range(4):
        low = torch.nextafter(low, original_high)
        high = torch.nextafter(high, original_low)
    low = torch.where(original_low == 0, original_low, low)
    if not (low < high).all():
        raise ValueError('region bounds have no representable float32 interior')
    return low, high


class GraphLayer(nn.Module):
    """Directed weighted messages; no CPU temporary allocation on CUDA."""
    def __init__(self, hidden):
        super().__init__()
        self.self_linear = nn.Linear(hidden, hidden)
        self.neighbor = nn.Linear(hidden, hidden, bias=False)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x, src, dst, weight):
        total, scale = torch.zeros_like(x), x.new_zeros((len(x), 1))
        if src.numel():
            weight = weight.to(x).clamp_min(0)
            weight = weight / weight.mean().clamp_min(1e-8)
            total.index_add_(0, dst, x[src] * weight[:, None])
            scale.index_add_(0, dst, weight[:, None])
        return F.gelu(self.norm(self.self_linear(x) +
                               self.neighbor(total / scale.clamp_min(1))))


class CapacityError(ValueError):
    """The deterministic upper-bound decoder did not find a feasible allocation.

    Weighted multidimensional bin packing is combinatorial.  Failure of this
    bounded greedy/one-displacement decoder does not prove global infeasibility.
    The caller must report it, increase candidate coverage, or skip the trial;
    never silently relax capacity or reinterpret it as a routed failure.
    """


def capacity_decode(scores, demand, capacity, movable=None, fixed_region=None,
                    candidate_regions=None):
    """Deterministic weighted upper-bound allocation, with one-item repair.

    scores is [N,R], or [N,K] with candidate_regions[N,K].  Input tensors may
    live on CUDA; the discrete decoder runs on CPU.  Nonmovable nodes retain
    fixed_region (default -1); their known regional demand is reserved first.
    Returns (assignment long[N], usage float64[R,T]) on CPU.
    """
    scores = torch.as_tensor(scores).detach().cpu().double()
    demand = torch.as_tensor(demand).detach().cpu().double()
    capacity = torch.as_tensor(capacity).detach().cpu().double()
    if scores.ndim != 2 or demand.ndim != 2 or capacity.ndim != 2:
        raise ValueError('scores, demand and capacity must be matrices')
    n, regions = len(demand), len(capacity)
    if len(scores) != n or demand.shape[1] != capacity.shape[1] or not regions:
        raise ValueError('incompatible node/resource/region dimensions')
    if not torch.isfinite(demand).all() or not torch.isfinite(capacity).all():
        raise ValueError('demand/capacity must be finite')
    if (demand < 0).any() or (capacity < 0).any():
        raise ValueError('demand/capacity must be nonnegative')
    movable = (torch.ones(n, dtype=torch.bool) if movable is None else
               torch.as_tensor(movable).detach().cpu().bool())
    assignment = (torch.full((n,), -1, dtype=torch.long) if fixed_region is None else
                  torch.as_tensor(fixed_region).detach().cpu().long().clone())
    if movable.shape != (n,) or assignment.shape != (n,):
        raise ValueError('movable/fixed_region must have one entry per node')
    if ((assignment < -1) | (assignment >= regions)).any():
        raise ValueError('fixed_region outside device')
    if candidate_regions is None:
        if scores.shape[1] != regions:
            raise ValueError('dense scores must have one column per region')
        candidate_regions = torch.arange(regions).expand(n, -1)
    else:
        candidate_regions = torch.as_tensor(candidate_regions).detach().cpu().long()
        if candidate_regions.shape != scores.shape:
            raise ValueError('sparse candidate region/score shapes differ')
        if ((candidate_regions < 0) | (candidate_regions >= regions)).any():
            raise ValueError('candidate region outside device')
    remaining = capacity.clone()
    for node in torch.where(~movable & (assignment >= 0))[0].tolist():
        remaining[assignment[node]] -= demand[node]
    tolerance = 1e-7 * capacity.abs().clamp_min(1)
    if (remaining < -tolerance).any():
        raise CapacityError('fixed regional demand already exceeds proxy capacity')
    assignment[movable] = -1
    choices, gaps = {}, {}
    # Stable tie breaking makes CPU/CUDA inference reproducible after scoring.
    for node in torch.where(movable)[0].tolist():
        seen, ranked = set(), []
        for index in sorted(range(scores.shape[1]),
                            key=lambda j: (-float(scores[node, j]),
                                           int(candidate_regions[node, j]))):
            region = int(candidate_regions[node, index])
            if region in seen or not math.isfinite(float(scores[node, index])):
                continue
            seen.add(region)
            if torch.all(demand[node] <= capacity[region] + tolerance[region]):
                ranked.append((region, float(scores[node, index])))
        if not ranked:
            raise CapacityError(f'node {node}: no individually feasible candidate region')
        choices[node] = [region for region, score in ranked]
        gaps[node] = ranked[0][1] - ranked[1][1] if len(ranked) > 1 else math.inf
    total = capacity.sum(0).clamp_min(1e-8)
    order = sorted(choices, key=lambda node: (len(choices[node]),
                   -float((demand[node] / total).max()), -gaps[node], node))
    residents = defaultdict(list)

    def fits(node, region):
        return bool(torch.all(demand[node] <= remaining[region] + tolerance[region]))

    for node in order:
        destination = next((region for region in choices[node] if fits(node, region)), None)
        if destination is None:
            # Move one previously allocated, movable resident out of the desired
            # bin when that suffices.  No global N x site cost/flow graph is built.
            for region in choices[node]:
                repaired = False
                for other in sorted(residents[region]):
                    if not torch.all(demand[node] <= remaining[region] + demand[other] +
                                     tolerance[region]):
                        continue
                    alternative = next((r for r in choices[other]
                                        if r != region and fits(other, r)), None)
                    if alternative is None:
                        continue
                    remaining[region] += demand[other]
                    remaining[alternative] -= demand[other]
                    residents[region].remove(other)
                    residents[alternative].append(other)
                    assignment[other] = alternative
                    destination, repaired = region, True
                    break
                if repaired:
                    break
        if destination is None:
            raise CapacityError(f'node {node}: candidate allocation exhausted; '
                                'global feasibility is not established')
        assignment[node] = destination
        remaining[destination] -= demand[node]
        residents[destination].append(node)
    usage = capacity - remaining
    if (usage > capacity + tolerance).any():
        raise AssertionError('internal decoder capacity violation')
    return assignment, usage


class HierarchicalInitializer(nn.Module):
    """Logic GNN + shared-pool device GNN + geometry-defined up/down hierarchy."""
    def __init__(self, cluster_width, region_width, hidden=64, chunk_size=512,
                 pool_width=8, demand_width=5):
        super().__init__()
        if chunk_size < 1:
            raise ValueError('chunk_size must be positive')
        self.config = dict(cluster_width=cluster_width, region_width=region_width,
                           hidden=hidden, chunk_size=chunk_size, pool_width=pool_width,
                           demand_width=demand_width)
        self.chunk_size, self.hidden = chunk_size, hidden
        self.logic_input = nn.Linear(cluster_width, hidden)
        self.demand_input = nn.Linear(demand_width + 1, hidden)
        self.logic_layers = nn.ModuleList([GraphLayer(hidden) for _ in range(3)])
        self.region_input = nn.Linear(region_width, hidden)
        self.capacity_input = nn.Linear(demand_width, hidden)
        self.geometry = nn.Linear(4, hidden)
        self.width_logic = nn.Linear(1, hidden)
        self.width_region = nn.Linear(1, hidden)
        self.pool = nn.Sequential(nn.Linear(pool_width, hidden), nn.GELU(),
                                  nn.Linear(hidden, hidden))
        self.pool_fuse = nn.Linear(2 * hidden, hidden)
        self.region_layers = nn.ModuleList([GraphLayer(hidden) for _ in range(2)])
        self.up = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU(), nn.LayerNorm(hidden))
        self.down = nn.Sequential(nn.Linear(2 * hidden, hidden), nn.GELU(),
                                  nn.LayerNorm(hidden))
        self.query = nn.Linear(hidden, hidden, bias=False)
        self.key = nn.Linear(hidden, hidden, bias=False)
        self.coordinate_head = nn.Sequential(nn.Linear(3 * hidden + 5, hidden),
                                             nn.GELU(), nn.Linear(hidden, 2))

    def _width(self, data, reference):
        value = torch.as_tensor(data.get('channel_width', 100.),
                                device=reference.device, dtype=reference.dtype)
        return torch.log1p(value.clamp_min(0)).reshape(1, 1) / math.log(301.)

    def encode_device(self, data):
        bounds = data['region_bounds']
        x = F.gelu(self.region_input(data['region_x']) + self.geometry(bounds) +
                   self.capacity_input(torch.log1p(data['capacity'].clamp_min(0))) +
                   self.width_region(self._width(data, bounds)))
        pools = self.pool(data['pool_x'])
        context = pool_mean(pools[data['pool_ids']], data['pool_regions'], len(x))
        x = F.gelu(self.pool_fuse(torch.cat((x, context), -1)))
        for layer in self.region_layers:
            x = layer(x, data['region_src'], data['region_dst'], data['region_weight'])
        levels = [x]
        for parent in data.get('parents', []):
            if parent.numel() != len(levels[-1]):
                raise ValueError('hierarchy membership size does not match child level')
            levels.append(self.up(pool_mean(levels[-1], parent, int(parent.max()) + 1)))
        x = levels[-1]
        for index in reversed(range(len(levels) - 1)):
            x = self.down(torch.cat((levels[index], x[data['parents'][index]]), -1))
        return x

    def encode_logic(self, data):
        movable = data.get('movable', torch.ones(len(data['cluster_x']),
                          device=data['cluster_x'].device, dtype=torch.bool))
        demand = torch.cat((torch.log1p(data['demand'].clamp_min(0)),
                            (~movable.bool()).to(data['cluster_x'])[:, None]), -1)
        x = F.gelu(self.logic_input(data['cluster_x']) +
                   self.demand_input(demand) +
                   self.width_logic(self._width(data, data['cluster_x'])))
        weight = data['logic_weight']
        if 'logic_criticality' in data:
            weight = weight * (1 + data['logic_criticality'].clamp_min(0))
        for layer in self.logic_layers:
            x = layer(x, data['logic_src'], data['logic_dst'], weight)
        return x

    def encode(self, data):
        return self.encode_logic(data), self.encode_device(data)

    def matching_logits(self, logic, regions, demand=None, capacity=None):
        scores = self.query(logic) @ self.key(regions).T / math.sqrt(self.hidden)
        if demand is not None and capacity is not None:
            # Scalar resources loop bounds memory at chunk_size x R, not x R x T.
            feasible = torch.ones_like(scores, dtype=torch.bool)
            for resource in range(demand.shape[-1]):
                feasible &= (demand[:, resource, None] <=
                             capacity[None, :, resource] + 1e-6)
            scores = scores.masked_fill(~feasible, -torch.inf)
        return scores

    def coordinates_from_encoded(self, data, logic, regions, assignment):
        assignment = assignment.to(device=logic.device, dtype=torch.long)
        valid = assignment >= 0
        selected = assignment.clamp_min(0)
        bounds = data['region_bounds'][selected]
        local = torch.sigmoid(self.coordinate_head(torch.cat((
            logic, regions[selected], logic * regions[selected], bounds,
            self._width(data, logic).expand(len(logic), 1)), -1)))
        low, high = bounds[:, [0, 2]], bounds[:, [1, 3]]
        xy = low + (high - low) * local
        # Leaves own half-open spatial intervals.  Float32 sigmoid saturation or
        # sampling must not place a node on the next leaf/device's upper edge.
        safe_low, safe_high = coordinate_interior(bounds)
        xy = torch.maximum(safe_low.to(xy), torch.minimum(safe_high.to(xy), xy))
        fixed = data.get('fixed_xy', torch.zeros_like(xy))
        if 'movable' in data:
            valid = valid & data['movable'].bool()
        return torch.where(valid[:, None], xy, fixed)

    def forward(self, data, teacher_region=None):
        logic, regions = self.encode(data)
        logits = torch.cat([self.matching_logits(logic[start:start + self.chunk_size],
                           regions, data['demand'][start:start + self.chunk_size],
                           data['capacity']) for start in range(0, len(logic), self.chunk_size)])
        assignment = logits.argmax(-1) if teacher_region is None else teacher_region
        xy = self.coordinates_from_encoded(data, logic, regions, assignment)
        return dict(logits=logits, xy=xy, assignment=assignment,
                    logic_embedding=logic, region_embedding=regions)

    def _supervised_chunk(self, logic, regions, demand, target, capacity):
        if (demand > capacity[target] + 1e-6).any():
            raise ValueError('teacher AP demand exceeds individual region proxy capacity; '
                             'use resource-aware derived supervision, not raw GP leaf labels')
        logits = self.matching_logits(logic, regions, demand, capacity)
        if torch.isnan(logits).any() or torch.isposinf(logits).any():
            raise FloatingPointError('NaN/Inf matching scores, distinct from masked capacity exclusions')
        if not torch.isfinite(logits.gather(1, target[:, None])).all():
            raise FloatingPointError('nonfinite teacher matching scores after capacity validation')
        return (F.cross_entropy(logits, target, reduction='sum'),
                logits.softmax(-1).T @ demand)

    def loss(self, data, teacher_region, teacher_xy, mask=None, *,
             coordinate_weight=5., capacity_weight=.1, geometry_weight=.1,
             return_details=False, geometry_teacher_xy=None):
        """Supervise actual deployed region/coordinate generator without dense NxR.

        Region CE and region-conditioned coordinate regression use teacher nodes.
        Expected regional overflow is penalized; unused capacity is not penalized.
        Optional edge-length protection uses raw teacher geometry and criticality,
        not routed-test data.  Timing criticality defaults to neutral when absent.
        Resource-projected xy supervises coordinates only; geometry_teacher_xy
        preserves the original VPR teacher for edge-length protection.
        """
        logic, regions = self.encode(data)
        teacher_region = teacher_region.to(logic.device).long()
        teacher_xy = teacher_xy.to(logic)
        geometry_teacher_xy = (teacher_xy if geometry_teacher_xy is None else
                               geometry_teacher_xy.to(logic))
        if (geometry_teacher_xy.shape != teacher_xy.shape or
                not torch.isfinite(geometry_teacher_xy).all()):
            raise ValueError('geometry teacher coordinates must be finite and match teacher xy')
        mask = (data.get('movable', torch.ones(len(logic), device=logic.device,
                                             dtype=torch.bool)) if mask is None else mask)
        mask = mask.to(logic.device).bool() & (teacher_region >= 0)
        if not mask.any():
            raise ValueError('no supervised movable AP blocks')
        if not torch.isfinite(teacher_xy[mask]).all():
            raise ValueError('teacher coordinates must be finite global normalized xy')
        selected = teacher_region[mask]
        if (selected >= len(regions)).any():
            raise ValueError('teacher region outside device')
        xy = self.coordinates_from_encoded(data, logic, regions, teacher_region)
        ce = logic.new_zeros(())
        expected = logic.new_zeros(data['capacity'].shape)
        fixed_region = data.get('fixed_region')
        if fixed_region is not None:
            fixed = ~data.get('movable', mask).bool() & (fixed_region >= 0)
            if fixed.any():
                expected.index_add_(0, fixed_region[fixed], data['demand'][fixed])
        for start in range(0, len(logic), self.chunk_size):
            end = start + self.chunk_size
            active = mask[start:end]
            if not active.any():
                continue
            demand = data['demand'][start:end][active]
            target = teacher_region[start:end][active]
            # Recompute logits during backward.  Merely looping over chunks
            # otherwise retains every chunk's NxR softmax activation and does
            # not actually bound training memory as device area grows.
            if torch.is_grad_enabled():
                chunk_ce, chunk_usage = checkpoint(self._supervised_chunk,
                    logic[start:end][active], regions, demand, target,
                    data['capacity'], use_reentrant=False)
            else:
                chunk_ce, chunk_usage = self._supervised_chunk(
                    logic[start:end][active], regions, demand, target, data['capacity'])
            ce = ce + chunk_ce
            expected = expected + chunk_usage
        ce = ce / mask.sum()
        bounds = data['region_bounds'][selected]
        # Local regression is comparable for large and small boundary windows.
        span = (bounds[:, [1, 3]] - bounds[:, [0, 2]]).clamp_min(1e-6)
        coordinate = F.smooth_l1_loss(xy[mask] / span, teacher_xy[mask] / span)
        overflow = F.relu(expected - data['capacity']) / data['capacity'].clamp_min(1)
        capacity_loss = overflow.square().mean()
        geometry = logic.new_zeros(())
        src, dst = data['logic_src'], data['logic_dst']
        if src.numel() and geometry_weight:
            edges = mask[src] & mask[dst]
            if edges.any():
                predicted = (xy[src[edges]] - xy[dst[edges]]).abs().sum(-1)
                target = (geometry_teacher_xy[src[edges]] - geometry_teacher_xy[dst[edges]]).abs().sum(-1)
                weight = data['logic_weight'][edges].clamp_min(0)
                if 'logic_criticality' in data:
                    weight = weight * (1 + data['logic_criticality'][edges].clamp_min(0))
                geometry = (F.relu(predicted - target) * weight).sum() / weight.sum().clamp_min(1)
        total = ce + coordinate_weight * coordinate + capacity_weight * capacity_loss + geometry_weight * geometry
        if return_details:
            return total, dict(region_ce=float(ce.detach()), coordinate=float(coordinate.detach()),
                               capacity_overflow=float(capacity_loss.detach()),
                               geometry=float(geometry.detach()), supervised=int(mask.sum()))
        return total

    supervision_loss = loss

    @torch.no_grad()
    def predict(self, data, *, topk=16, device_embedding=None, logic_embedding=None,
                score_noise_std=0., coordinate_noise_std=0., seed=0):
        """Sparse candidate matching + deterministic capacity-bound GP coordinates.

        Candidate coverage doubles on allocation failure.  Reported upper-bound
        usage is always verified.  Even the final full-region fallback is N x R,
        never N x physical sites; callers can configure graph leaf size/top-k.

        Nonzero noise is intended for TRAIN-only intervention label collection.
        Deployment defaults are deterministic, one inference, no search.  Chunk
        RNG is reset on coverage retries, so expanding k does not resample scores.
        """
        if (not math.isfinite(score_noise_std) or score_noise_std < 0 or
                not math.isfinite(coordinate_noise_std) or coordinate_noise_std < 0):
            raise ValueError('sampling noise standard deviations must be finite and nonnegative')
        logic = self.encode_logic(data) if logic_embedding is None else logic_embedding
        regions = self.encode_device(data) if device_embedding is None else device_embedding
        k = min(max(int(topk), 1), len(regions))
        while True:
            values, indices = [], []
            generator = torch.Generator(device='cpu').manual_seed(int(seed))
            for start in range(0, len(logic), self.chunk_size):
                logits = self.matching_logits(logic[start:start + self.chunk_size], regions,
                         data['demand'][start:start + self.chunk_size], data['capacity'])
                if score_noise_std:
                    logits = logits + score_noise_std * torch.randn(
                        logits.shape, generator=generator).to(logits)
                # Stable argsort handles equal scores consistently across devices.
                order = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :k]
                values.append(logits.gather(1, order).cpu())
                indices.append(order.cpu())
            try:
                assignment, usage = capacity_decode(torch.cat(values), data['demand'],
                    data['capacity'], data.get('movable'), data.get('fixed_region'),
                    torch.cat(indices))
                break
            except CapacityError:
                if k == len(regions):
                    raise
                k = min(k * 2, len(regions))
        xy = self.coordinates_from_encoded(data, logic, regions, assignment)
        if coordinate_noise_std:
            generator = torch.Generator(device='cpu').manual_seed(int(seed) ^ 0x5DEECE66D)
            selected = assignment.to(xy.device).clamp_min(0)
            bounds = data['region_bounds'][selected]
            low, high = bounds[:, [0, 2]], bounds[:, [1, 3]]
            safe_low, safe_high = coordinate_interior(bounds)
            noise = torch.randn(xy.shape, generator=generator).to(xy)
            perturbed = torch.maximum(safe_low.to(xy), torch.minimum(safe_high.to(xy),
                xy + coordinate_noise_std * (high - low) * noise))
            active = assignment.to(xy.device) >= 0
            if 'movable' in data:
                active &= data['movable'].bool()
            xy = torch.where(active[:, None], perturbed, xy)
        return dict(assignment=assignment, xy=xy, usage=usage,
                    candidate_region_count=k, regional_capacity_is_proxy=True)
