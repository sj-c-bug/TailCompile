"""007: explicit layer policy and a separately frozen conditional leaf/XY GNN.

P(layer) is used identically by supervision, feedback likelihood and decoding.
The nondifferentiable capacity decoder is a bounded proxy, not a packing proof.
The old model/entry points are deliberately not modified.
"""
from __future__ import annotations

import math
import time
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .model import (Hierarchical3DInitializer, validate, compatibility_mask,
                        allocate_layers, dispersed_xy, CapacityError)
except ImportError:
    from model import (Hierarchical3DInitializer, validate, compatibility_mask,
                       allocate_layers, dispersed_xy, CapacityError)

SCHEMA = 'tailcompile-v7-explicit-layer-generator-1'


class ExplicitLayerInitializer(nn.Module):
    def __init__(self, cluster_width, region_width, hidden=64, chunk_size=256,
                 pool_width=8, demand_width=5, vertical_tail_scale=1.):
        super().__init__()
        base_config = dict(cluster_width=cluster_width, region_width=region_width,
                           hidden=hidden, chunk_size=chunk_size, pool_width=pool_width,
                           demand_width=demand_width)
        if not math.isfinite(vertical_tail_scale) or vertical_tail_scale <= 0:
            raise ValueError('positive fixed train-only vertical_tail_scale required')
        self.vertical_tail_scale = float(vertical_tail_scale)
        self.config = dict(base_config, vertical_tail_scale=self.vertical_tail_scale)
        self.shared_xy = Hierarchical3DInitializer(**base_config)
        # A separate encoder is essential: updating it cannot change shared XY.
        self.layer_network = Hierarchical3DInitializer(**base_config)
        self.layer_query = nn.Linear(hidden, hidden, bias=False)
        self.layer_key = nn.Linear(hidden, hidden, bias=False)
        self.resource_layer = nn.Linear(5, hidden)
        self.chunk_size = chunk_size
        self.shared_xy_frozen = False

    def freeze_shared_xy(self):
        self.shared_xy_frozen = True
        self.shared_xy.requires_grad_(False)
        self.shared_xy.eval()
        return self

    def train(self, mode=True):
        super().train(mode)
        if self.shared_xy_frozen:
            self.shared_xy.eval()
        return self

    def _allowed(self, data, meta):
        n, layers = len(data['demand']), len(meta[1])
        allowed = torch.zeros((n, layers), dtype=torch.bool, device=data['demand'].device)
        for start in range(0, n, self.chunk_size):
            ids = torch.arange(start, min(n, start + self.chunk_size), device=allowed.device)
            mask = compatibility_mask(data, ids, meta)
            for z in range(layers):
                allowed[ids, z] = mask[:, meta[0] == z].any(1)
        return allowed

    def layer_distribution(self, data):
        meta = validate(data)
        logic, regions = self.layer_network.encode(data)
        pooled = regions.new_zeros((len(meta[1]), regions.shape[1]))
        pooled.index_add_(0, meta[0], regions)
        counts = torch.bincount(meta[0], minlength=len(pooled)).to(pooled).clamp_min(1)
        pooled = pooled / counts[:, None] + self.resource_layer(torch.log1p(meta[1]))
        logits = self.layer_query(logic) @ self.layer_key(pooled).T / math.sqrt(logic.shape[1])
        allowed = self._allowed(data, meta)
        logits = logits.masked_fill(~allowed, -torch.inf)
        return logits, logits.softmax(1), allowed, meta

    def conditional_positions(self, data, *, encoded=None):
        """N×L counterfactual XY, individually compatible; no global layer fill."""
        meta = validate(data)
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.shared_xy_frozen):
            logic, regions = self.shared_xy.encode(data) if encoded is None else encoded
            result = logic.new_zeros((len(logic), len(meta[1]), 2))
            allowed = self._allowed(data, meta)
            for z in range(len(meta[1])):
                assignment = torch.empty(len(logic), dtype=torch.long, device=logic.device)
                for start in range(0, len(logic), self.chunk_size):
                    ids = torch.arange(start, min(len(logic), start+self.chunk_size), device=logic.device)
                    mask = compatibility_mask(data, ids, meta) & (meta[0][None, :] == z)
                    values = self.shared_xy.matching_logits(logic[ids], regions).masked_fill(~mask, -torch.inf)
                    assignment[ids] = values.argmax(1)
                # Invalid layer alternatives are explicitly masked by P(layer).
                xy = self.shared_xy.coordinates_from_encoded(data, logic, regions, assignment)
                result[:, z] = torch.where(allowed[:, z, None], xy, torch.zeros_like(xy))
        return result

    def supervision_loss(self, data, teacher, *, tail_weight=0., shared=False):
        logits, prob, _, meta = self.layer_distribution(data)
        movable = data.get('movable', torch.ones(len(prob), device=prob.device, dtype=torch.bool)).bool()
        target = torch.as_tensor(teacher['teacher_region'], device=prob.device, dtype=torch.long)
        if target.shape != (len(prob),) or (target < 0).any() or (target >= len(meta[0])).any():
            raise ValueError('teacher leaf coverage/range mismatch')
        target_layer = meta[0][target]
        if not movable.any() or not torch.isfinite(logits.gather(1, target_layer[:, None])).all():
            raise ValueError('no movable teacher or incompatible teacher layer')
        ce = F.cross_entropy(logits[movable], target_layer[movable])
        expected = prob.T @ data['demand']
        capacity = ((expected-meta[1]).clamp_min(0) / meta[1].clamp_min(1)).square().mean()
        src, dst = data['logic_src'], data['logic_dst']
        weight = data['logic_weight'] * (1+data.get('logic_criticality', torch.zeros_like(data['logic_weight'])).clamp_min(0))
        cut = ((1-(prob[src]*prob[dst]).sum(1))*weight).sum()/weight.sum().clamp_min(1)
        conditional = self.conditional_positions(data)
        expected_xy = (prob[:, :, None]*conditional.detach()).sum(1)
        wl = ((expected_xy[src]-expected_xy[dst]).abs().sum(1)*weight).sum()/weight.sum().clamp_min(1)
        loss = ce + .1*capacity + .1*cut + .1*wl
        detail = dict(layer_ce=float(ce.detach()), expected_capacity_overflow=float(capacity.detach()),
                      timing_proxy_cut=float(cut.detach()), xy_wire_proxy=float(wl.detach()))
        if shared:
            if self.shared_xy_frozen:
                raise ValueError('cannot pretrain frozen shared XY')
            # Explicit conditional leaf CE, never a competing marginal layer score.
            logic, regions = self.shared_xy.encode(data)
            conditional_ce = logic.new_zeros(())
            count = 0
            for start in range(0, len(logic), self.chunk_size):
                ids = torch.arange(start, min(len(logic), start+self.chunk_size), device=prob.device)
                ids = ids[movable[ids]]
                if not len(ids):
                    continue
                mask = compatibility_mask(data, ids, meta) & (meta[0][None, :] == target_layer[ids, None])
                scores = self.shared_xy.matching_logits(logic[ids], regions).masked_fill(~mask, -torch.inf)
                if not torch.isfinite(scores.gather(1, target[ids, None])).all():
                    raise ValueError('teacher leaf incompatible')
                conditional_ce = conditional_ce + F.cross_entropy(scores, target[ids], reduction='sum')
                count += len(ids)
            predicted_xy = self.shared_xy.deployment_coordinates(data, logic, regions, target)
            teacher_xy = torch.as_tensor(teacher['teacher_xy'], device=prob.device, dtype=prob.dtype)
            bounds = data['region_bounds'][target]
            span = (bounds[:, [1, 3]]-bounds[:, [0, 2]]).clamp_min(1e-6)
            xy_loss = F.smooth_l1_loss(predicted_xy[movable]/span[movable], teacher_xy[movable]/span[movable])
            loss = loss + conditional_ce/count + 5*xy_loss
            detail.update(conditional_leaf_ce=float((conditional_ce/count).detach()), coordinate=float(xy_loss.detach()))
        if tail_weight or 'vertical_risk_device' in data:
            try:
                from .vertical_risk import risk_loss
            except ImportError:
                from vertical_risk import risk_loss
            tail, diagnostics = risk_loss(data, prob, conditional.detach())
            unassigned = diagnostics.pop('unassigned_mass_tensor', tail.new_zeros(()))
            unreachable = unassigned/max(1, diagnostics.get('active_net_count', 1))
            loss = loss + unreachable + float(tail_weight)*tail/self.vertical_tail_scale
            detail['expected_unassigned_constraint'] = float(unreachable.detach())
            detail.update(vertical_tail=float(tail.detach()), vertical_tail_scale=self.vertical_tail_scale,
                          vertical_risk=diagnostics)
        return loss, detail

    def action_log_probability(self, data, action, *, probability=None):
        prob = self.layer_distribution(data)[1] if probability is None else probability
        if 'layer' in action:
            target = torch.as_tensor(action['layer'], device=prob.device, dtype=torch.long)
            if 'assignment' in action:
                leaves = torch.as_tensor(action['assignment'], device=prob.device, dtype=torch.long)
                if leaves.shape != target.shape or (leaves < 0).any() or (leaves >= len(data['region_layer'])).any():
                    raise ValueError('action leaf dimensions/range mismatch')
                if not torch.equal(data['region_layer'][leaves], target):
                    raise ValueError('action layer and conditional leaf disagree')
        else:
            target = data['region_layer'][torch.as_tensor(action['assignment'], device=prob.device, dtype=torch.long)]
        if target.shape != (len(prob),) or (target < 0).any() or (target >= prob.shape[1]).any():
            raise ValueError('action layer dimensions/range mismatch')
        movable = data.get('movable', torch.ones(len(prob), device=prob.device, dtype=torch.bool)).bool()
        selected = prob.gather(1, target[:, None]).squeeze(1)
        if not movable.any() or (selected[movable] <= 0).any():
            raise ValueError('action incompatible with layer policy')
        return selected[movable].log().mean()


def deterministic_layers(data, allowed, *, passes=2):
    """Capacity greedy + at most two incremental weighted cut sweeps."""
    if not 0 <= passes <= 2:
        raise ValueError('deterministic partition permits at most two sweeps')
    meta = validate(data)
    assignment, usage = allocate_layers(torch.zeros_like(allowed, dtype=data['demand'].dtype),
        data['demand'], meta[1], allowed, balanced=True)
    adjacency = [[] for _ in assignment]
    critical = data.get('logic_criticality', torch.zeros_like(data['logic_weight']))
    for source, sink, weight in zip(data['logic_src'].tolist(), data['logic_dst'].tolist(),
                                   (data['logic_weight']*(1+critical.clamp_min(0))).tolist()):
        if source != sink:
            adjacency[source].append((sink, weight)); adjacency[sink].append((source, weight))
    # Native scalar bookkeeping avoids millions of tiny Torch dispatches on
    # MCML; the only Torch/CPU transfer is before the bounded sweeps.
    capacity = meta[1].cpu().double().tolist()
    demand, possible = data['demand'].cpu().double().tolist(), allowed.cpu().tolist()
    assigned, used = assignment.tolist(), usage.tolist()
    for _ in range(passes):
        moved = False
        for node in range(len(assigned)):
            old = assigned[node]
            old_cost = sum(w for other, w in adjacency[node] if assigned[other] != old)
            best, gain = old, 0.
            for z in range(len(capacity)):
                if z == old or not possible[node][z] or any(used[z][t]+demand[node][t] > capacity[z][t]+1e-6 for t in range(5)):
                    continue
                improvement = old_cost-sum(w for other, w in adjacency[node] if assigned[other] != z)
                if improvement > gain+1e-12:
                    best, gain = z, improvement
            if best != old:
                for resource in range(5):
                    used[old][resource] -= demand[node][resource]
                    used[best][resource] += demand[node][resource]
                assigned[node] = best; moved = True
        if not moved:
            break
    return torch.tensor(assigned, dtype=torch.long), torch.tensor(used, dtype=usage.dtype)


@torch.no_grad()
def generate(model, data, *, method='L', seed=0, score_noise_std=0., density_passes=2):
    """Frozen conditional decoding; method D changes only the layer partition."""
    if method not in ('D', 'L', 'T') or not math.isfinite(score_noise_std) or score_noise_std < 0:
        raise ValueError('invalid layer generator method/noise')
    if isinstance(density_passes, bool) or not isinstance(density_passes, int) or not 0 <= density_passes <= 4:
        raise ValueError('density_passes must be bounded [0,4]')
    if not model.shared_xy_frozen:
        raise ValueError('deployment requires frozen shared XY')
    started = time.perf_counter()
    model.eval()
    if method == 'D':
        meta = validate(data)
        allowed = model._allowed(data, meta)
        chosen, layer_usage = deterministic_layers(data, allowed)
        prob = F.one_hot(chosen.to(data['demand'].device), len(meta[1])).to(data['demand'])
    else:
        scores, prob, allowed, meta = model.layer_distribution(data)
        if score_noise_std:
            generator = torch.Generator(device='cpu').manual_seed(seed)
            scores = scores + score_noise_std*torch.randn(scores.shape, generator=generator).to(scores)
        chosen, layer_usage = allocate_layers(scores.detach(), data['demand'], meta[1], allowed)
    chosen = chosen.to(data['demand'].device)
    logic, regions = model.shared_xy.encode(data)
    encoded = time.perf_counter()
    assignment = torch.empty(len(logic), dtype=torch.long, device=logic.device)
    prices = torch.zeros_like(data['capacity'])
    for iteration in range(density_passes+1):
        expected = torch.zeros_like(prices)
        for start in range(0, len(logic), model.chunk_size):
            ids = torch.arange(start, min(len(logic), start+model.chunk_size), device=logic.device)
            mask = compatibility_mask(data, ids, meta) & (meta[0][None, :] == chosen[ids, None])
            values = model.shared_xy.matching_logits(logic[ids], regions) - data['demand'][ids] @ (prices/data['capacity'].clamp_min(1)).T
            values = values.masked_fill(~mask, -torch.inf)
            if iteration == density_passes:
                assignment[ids] = values.argmax(1)
            else:
                expected.add_(values.softmax(1).T @ data['demand'][ids])
        if iteration < density_passes:
            prices = (prices+(expected-data['capacity'])/data['capacity'].clamp_min(1)).clamp(0, 100)
    xy = model.shared_xy.deployment_coordinates(data, logic, regions, assignment)
    if not torch.isfinite(xy).all() or (xy < 0).any() or (xy >= 1).any():
        raise ValueError('decoded coordinates outside device')
    usage = torch.zeros_like(data['capacity']); usage.index_add_(0, assignment, data['demand'])
    overflow = (usage-data['capacity']).clamp_min(0)
    result = dict(assignment=assignment.cpu(), layer=chosen.cpu(), xy=xy.cpu(), usage=usage.cpu(),
        layer_usage=layer_usage.cpu(), initializer='explicit_layer_'+method,
        decoder_kind='explicit-layer-conditional-xy-1',
        capacity_diagnostics=dict(hard_layer_capacity_enforced=True, hard_leaf_capacity_enforced=False,
            packing_legality_claimed=False, overflow_by_resource=overflow.sum(0).cpu().tolist(),
            overflow_region_count=int((overflow > 1e-6).any(1).sum()), used_region_count=int(assignment.unique().numel())),
        phase_seconds=dict(encoding=encoded-started, decoding=time.perf_counter()-encoded))
    if 'vertical_risk_device' in data:
        try:
            from .vertical_risk import risk_loss, hard_risk
        except ImportError:
            from vertical_risk import risk_loss, hard_risk
        _, soft = risk_loss(data, prob, model.conditional_positions(data, encoded=(logic, regions)))
        soft.pop('unassigned_mass_tensor', None)
        result.update(soft_vertical_risk=soft, hard_vertical_risk=hard_risk(data, chosen, xy))
    result['phase_seconds']['total'] = time.perf_counter()-started
    return result
