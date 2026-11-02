"""010: resource-feasible topology/context initializer, without TailRisk.

The discrete allocator is bounded and not differentiated. Geometry uses actual
deployed XY in the forward pass, direct coordinate gradients and explicitly
biased straight-through region-centre gradients. These are timing/length
proxies, never substituted for final VPR QoR.
"""
from __future__ import annotations

import math
import time
import torch
from torch import nn
from torch.nn import functional as F

try:
    from .qor_model import QorInitializer, validate_collected_action as validate_old_action
    from .aligned_model import validate_action, action_digest
    from .aligned_teacher import constraints_digest, deployed_xy_bounds, verify_teacher
    from .model import validate, compatibility_mask
    from .structured_decoder import allocate_regions, validate_leaf_capacity
except ImportError:
    from qor_model import QorInitializer, validate_collected_action as validate_old_action
    from aligned_model import validate_action, action_digest
    from aligned_teacher import constraints_digest, deployed_xy_bounds, verify_teacher
    from model import validate, compatibility_mask
    from structured_decoder import allocate_regions, validate_leaf_capacity

SCHEMA = 'tailcompile-v7-structured-generator-010-1'
DECODER_SCHEMA = 'structured-hard-region-capacity-010-1'
ACTION_SCHEMA = 'tailcompile-v7-structured-action-010-1'
TEACHER_SCHEMA = 'tailcompile-v7-structured-teacher-010-1'
OBJECTIVE = 'structured-qor-010-1'


def topology_context(data):
    """Fourteen bounded features, O(N+E), no route/teacher information."""
    demand = data['demand']
    n = len(demand)
    src, dst = data['logic_src'], data['logic_dst']
    ones = demand.new_ones(len(src))
    indeg, outdeg = demand.new_zeros(n), demand.new_zeros(n)
    indeg.index_add_(0, dst, ones); outdeg.index_add_(0, src, ones)
    critical = data.get('logic_criticality', torch.zeros_like(data['logic_weight'])).clamp(0, 1)
    inc, outc = demand.new_zeros(n), demand.new_zeros(n)
    inc.index_add_(0, dst, critical); outc.index_add_(0, src, critical)
    mass = torch.log1p(demand)
    mass = mass / mass.amax(0).clamp_min(1)
    neighbours = demand.new_zeros((n, 5))
    neighbours.index_add_(0, src, mass[dst]); neighbours.index_add_(0, dst, mass[src])
    neighbours = neighbours / (indeg+outdeg).clamp_min(1)[:, None]
    degree = torch.stack((indeg, outdeg), 1).log1p()
    degree = degree / degree.amax(0).clamp_min(1)
    return torch.cat((degree, (inc/indeg.clamp_min(1))[:, None],
                      (outc/outdeg.clamp_min(1))[:, None], mass, neighbours), 1)


def _residual(hidden):
    result = nn.Sequential(nn.Linear(14, hidden), nn.GELU(), nn.Linear(hidden, hidden))
    nn.init.zeros_(result[-1].weight); nn.init.zeros_(result[-1].bias)
    return result


def leaf_noise(ids, regions, seed, dtype):
    """Chunk/order invariant integer-hashed uniform jitter, unit variance."""
    columns = torch.arange(regions, device=ids.device, dtype=torch.long)
    value = torch.remainder(ids[:, None]*1103515245+columns[None]*12345+(int(seed) % 2147483647)*2654435761, 2147483647)
    value = torch.bitwise_xor(value, value >> 16)
    value = torch.remainder(value*2246822519, 2147483647)
    return (value.to(dtype)/2147483647*2-1)*math.sqrt(3)


def balanced_layer_loss(logits, labels, demand, movable):
    """Equal observed teacher-layer/resource groups, not a forced 50/50 split."""
    per_node = F.cross_entropy(logits[movable], labels[movable], reduction='none')
    layer_terms, resource_terms = [], []
    for layer in range(logits.shape[1]):
        selected = labels[movable] == layer
        if selected.any():
            layer_terms.append(per_node[selected].mean())
        for resource in range(5):
            mass = demand[movable, resource]*selected
            if float(mass.sum()) > 0:
                resource_terms.append((per_node*mass).sum()/mass.sum())
    if not layer_terms:
        raise ValueError('010 supervision needs movable teacher atoms')
    resource = torch.stack(resource_terms).mean() if resource_terms else per_node.mean()
    return .5*torch.stack(layer_terms).mean()+.5*resource


def validate_structured_action(data, action, *, chunk_size=256, collected=False):
    if collected:
        if action.get('schema') == ACTION_SCHEMA:
            if action.get('decoder_kind') != DECODER_SCHEMA:
                raise ValueError('010 collected decoder mismatch')
            if action.get('constraints_sha256') != constraints_digest(data):
                raise ValueError('010 collected constraints mismatch')
            if not isinstance(action.get('collection_checkpoint_sha256'), str) or len(action['collection_checkpoint_sha256']) != 64:
                raise ValueError('010 collection checkpoint identity missing')
            if action.get('deployed_action_sha256') != action_digest(action):
                raise ValueError('010 collected action digest mismatch')
        else:
            validate_old_action(data, action, chunk_size=chunk_size)
    validate_action(data, action, chunk_size=chunk_size)
    return validate_leaf_capacity(data, torch.as_tensor(action['assignment'], device=data['demand'].device), chunk_size=chunk_size)


def validate_collected_action(data, action, *, chunk_size=256):
    return validate_structured_action(data, action, chunk_size=chunk_size, collected=True)


@torch.no_grad()
def project_teacher(data, teacher, *, chunk_size=256):
    """Project supervision only; no old P&R reward is attached to projection."""
    meta = validate(data)
    if teacher.get('schema') == TEACHER_SCHEMA:
        action = dict(assignment=teacher['teacher_region'], layer=teacher['teacher_layer'], xy=teacher['teacher_xy'])
        validate_structured_action(data, action, chunk_size=chunk_size)
        if teacher.get('constraints_sha256') != constraints_digest(data) or teacher.get('target_sha256') != action_digest(action):
            raise ValueError('010 teacher constraints/digest mismatch')
        return teacher
    verify_teacher(data, teacher, chunk_size=chunk_size)
    leaves = torch.as_tensor(teacher['teacher_region'], device=data['demand'].device, dtype=torch.long)
    layers = torch.as_tensor(teacher.get('teacher_layer', meta[0][leaves]), device=leaves.device, dtype=torch.long)
    target = torch.as_tensor(teacher['teacher_xy'], device=leaves.device, dtype=data['demand'].dtype)
    if leaves.shape != (len(data['demand']),) or target.shape != (len(leaves), 2):
        raise ValueError('teacher shape mismatch')
    layer_scores = F.one_hot(layers, len(meta[1])).to(data['demand']) * 12
    centres = (data['region_bounds'][:, [0, 2]]+data['region_bounds'][:, [1, 3]])/2
    def scores(ids):
        return -((target[ids, None]-centres[None]).square().sum(-1))*4
    action = allocate_regions(data, layer_scores, scores, chunk_size=chunk_size)
    low, high = deployed_xy_bounds(data, action['assignment'], meta)
    xy = torch.maximum(low.to(target), torch.minimum(target, high.to(target)))
    xy = torch.where(meta[4] >= 0, meta[4].to(xy), xy)
    action['xy'] = xy
    validate_structured_action(data, action, chunk_size=chunk_size)
    return dict(schema=TEACHER_SCHEMA, teacher_region=action['assignment'].cpu().tolist(),
                teacher_layer=action['layer'].cpu().tolist(), teacher_xy=xy.cpu().tolist(),
                constraints_sha256=constraints_digest(data), target_sha256=action_digest(action),
                projection=dict(source_action_sha256=action_digest(dict(assignment=leaves, layer=layers, xy=target)),
                                moved_regions=int((leaves != action['assignment']).sum()),
                                moved_layers=int((layers != action['layer']).sum()),
                                hard_leaf_capacity=True, packing_legality_claimed=False,
                                old_qor_labels_reused_for_projection=False))


class StructuredInitializer(QorInitializer):
    def __init__(self, *args, arm='X', allocator_restarts=2, **kwargs):
        if arm != 'X':
            raise ValueError('010 explicitly migrates the trainable 009 X actor')
        super().__init__(*args, arm='X', **kwargs)
        self.allocator_restarts = allocator_restarts
        self.config.update(allocator_restarts=allocator_restarts)
        hidden = self.layer_query.in_features
        self.layer_context = _residual(hidden)
        self.xy_context = _residual(hidden)

    def encode_shared(self, data):
        logic, regions = self.shared_xy.encode(data)
        return logic+self.xy_context(topology_context(data)), regions

    def layer_distribution(self, data):
        meta = validate(data)
        logic, regions = self.layer_network.encode(data)
        logic = logic+self.layer_context(topology_context(data))
        pooled = regions.new_zeros((len(meta[1]), regions.shape[1]))
        pooled.index_add_(0, meta[0], regions)
        counts = torch.bincount(meta[0], minlength=len(pooled)).to(pooled).clamp_min(1)
        pooled = pooled/counts[:, None]+self.resource_layer(torch.log1p(meta[1]))
        scores = self.layer_query(logic) @ self.layer_key(pooled).T/math.sqrt(logic.shape[1])
        allowed = self._allowed(data, meta)
        scores = scores.masked_fill(~allowed, -torch.inf)
        return scores, scores.softmax(1), allowed, meta

    def conditional_components(self, data, action, *, encoded=None, metadata=None):
        return super().conditional_components(data, action,
            encoded=self.encode_shared(data) if encoded is None else encoded, metadata=metadata)

    @torch.no_grad()
    def decode_action(self, data, *, policy=None, encoded=None, method=None, seed=0,
                      score_noise_std=0., training=False):
        if method not in (None, 'D', 'N') or not math.isfinite(score_noise_std) or score_noise_std < 0:
            raise ValueError('010 method/noise invalid')
        started = time.perf_counter()
        policy = self.layer_distribution(data) if policy is None else policy
        layer_scores = policy[0].detach()
        if score_noise_std:
            rng = torch.Generator(device='cpu').manual_seed(seed)
            layer_scores = layer_scores+score_noise_std*torch.randn(layer_scores.shape, generator=rng).to(layer_scores)
        logic, regions = self.encode_shared(data) if encoded is None else encoded
        def scores(ids):
            values = self.shared_xy.matching_logits(logic[ids], regions)
            if score_noise_std:
                values = values+score_noise_std*leaf_noise(ids, len(regions), seed, values.dtype)
            return values
        action = allocate_regions(data, layer_scores, scores, chunk_size=self.chunk_size,
                                  max_restarts=self.allocator_restarts)
        action['xy'] = self.shared_xy.deployment_coordinates(data, logic, regions, action['assignment'])
        validate_structured_action(data, action, chunk_size=self.chunk_size)
        action.update(initializer='structured_'+(method or 'N'), decoder_kind=DECODER_SCHEMA,
                      risk_diagnostics_computed=False, tail_weight=0.,
                      score_noise_semantics='Gaussian layer + stateless unit-variance uniform leaf jitter; no jitter at zero',
                      phase_seconds=dict(total=time.perf_counter()-started))
        action['deployed_action_sha256'] = action_digest(action)
        return action

    def differentiable_objectives(self, data, *, policy=None, encoded=None, action=None):
        policy = self.layer_distribution(data) if policy is None else policy
        layer_scores, prob, allowed, meta = policy
        logic, regions = self.encode_shared(data) if encoded is None else encoded
        action = self.decode_action(data, policy=policy, encoded=(logic, regions), training=True) if action is None else action
        leaves = action['assignment'].to(logic.device)
        prediction = self.shared_xy.deployment_coordinates(data, logic, regions, leaves)
        centres = (data['region_bounds'][:, [0, 2]]+data['region_bounds'][:, [1, 3]])/2
        expected_usage = torch.zeros_like(data['capacity'])
        soft_centres = []
        loglayer = F.log_softmax(layer_scores, 1)
        for start in range(0, len(logic), self.chunk_size):
            ids = torch.arange(start, min(start+self.chunk_size, len(logic)), device=logic.device)
            mask = compatibility_mask(data, ids, meta)
            raw = self.shared_xy.matching_logits(logic[ids], regions)
            norms = []
            for z in range(len(meta[1])):
                local = (meta[0] == z)[None] & mask
                norms.append(torch.logsumexp(raw.masked_fill(~local, -1e9), 1))
            norms = torch.stack(norms, 1)
            joint = (raw-norms[:, meta[0]]+loglayer[ids][:, meta[0]]).masked_fill(~mask, -torch.inf)
            probability = joint.softmax(1)
            expected_usage = expected_usage+probability.T @ data['demand'][ids]
            soft_centres.append(probability @ centres)
        raw_soft = torch.cat(soft_centres)
        # A fixed physical axis cannot gain a fictitious movement gradient by
        # selecting a leaf with a different centre at a shared boundary.
        fixed_xy = meta[4].to(raw_soft)
        soft = torch.where(fixed_xy >= 0, fixed_xy, raw_soft)
        # Forward is the current deployed coordinate head; discrete choices get
        # a biased centre surrogate, while the coordinate gradient is direct.
        xy = prediction+(soft-soft.detach())
        src, dst = data['logic_src'], data['logic_dst']
        weights = data['logic_weight'].clamp_min(0)
        length = (xy[src]-xy[dst]).abs().sum(1)
        wire = (length*weights).sum()/weights.sum().clamp_min(1)
        criticality = data.get('logic_criticality', torch.zeros_like(weights)).clamp_min(0)
        timing = (length*weights*(1+criticality)).sum()/(weights*(1+criticality)).sum().clamp_min(1)
        rel = (expected_usage-data['capacity']).clamp_min(0)/data['capacity'].clamp_min(1)
        density = rel.square().mean()
        return dict(wire=wire, timing=timing, density=density, action=action,
                    probability=prob, expected_usage=expected_usage, xy=prediction,
                    soft_region_xy=raw_soft)

    def supervision_loss(self, data, teacher, *, tail_weight=0., shared=False):
        if tail_weight != 0:
            raise ValueError('010 has no TailRisk term')
        teacher = project_teacher(data, teacher, chunk_size=self.chunk_size)
        policy = self.layer_distribution(data)
        logits, prob, _, meta = policy
        encoded = self.encode_shared(data)
        target_layers = torch.as_tensor(teacher['teacher_layer'], device=prob.device)
        target_leaf = torch.as_tensor(teacher['teacher_region'], device=prob.device)
        movable = data.get('movable', torch.ones(len(prob), device=prob.device, dtype=torch.bool)).bool()
        ce = balanced_layer_loss(logits, target_layers, data['demand'], movable)
        geo = self.differentiable_objectives(data, policy=policy, encoded=encoded)
        hard = F.one_hot(geo['action']['layer'].to(prob.device), len(meta[1])).to(prob)
        st = hard+(prob-prob.detach())
        total_mass = data['demand'].sum(0).clamp_min(1)
        actual_mass = st.T @ data['demand']/total_mass
        teacher_mass = F.one_hot(target_layers, len(meta[1])).to(prob).T @ data['demand']/total_mass
        mass_loss = (actual_mass-teacher_mass).square().mean()
        target_action = dict(assignment=target_leaf, layer=target_layers,
                             xy=torch.as_tensor(teacher['teacher_xy'], device=prob.device, dtype=prob.dtype))
        detail = self.conditional_components(data, target_action, encoded=encoded, metadata=meta)
        loss = ce+mass_loss+.2*(-detail['leaf_log_probability'])+.5*detail['teacher_xy_smooth_l1']+.1*geo['density']+.2*geo['wire']+.2*geo['timing']
        return loss, dict(layer_ce=float(ce.detach()), teacher_mass_loss=float(mass_loss.detach()),
            teacher_layer_mass=teacher_mass.detach().cpu().tolist(), deployed_layer_mass=actual_mass.detach().cpu().tolist(),
            conditional_leaf_ce=float(-detail['leaf_log_probability'].detach()), coordinate=float(detail['teacher_xy_smooth_l1'].detach()),
            xy_wire_proxy=float(geo['wire'].detach()), criticality_xy_proxy=float(geo['timing'].detach()),
            soft_density_overflow=float(geo['density'].detach()), tail_weight=0., tail_evaluated=False,
            hard_leaf_capacity_valid=bool(geo['action']['capacity_diagnostics'].get('valid', True)),
            teacher_projection=teacher.get('projection'), deployed_action_sha256=geo['action']['deployed_action_sha256'],
            proxy_boundary='normalized XY geometry/logic-depth criticality, not actual VPR timing',
            gradient_boundary='direct XY head; biased straight-through region/layer proxy; no allocator/VPR gradient')


@torch.no_grad()
def generate(model, data, *, seed=0, method=None, score_noise_std=0., diagnostics=False):
    if diagnostics:
        raise ValueError('010 does not compute TailRisk diagnostics')
    model.eval()
    return model.decode_action(data, seed=seed, method=method, score_noise_std=score_noise_std)
