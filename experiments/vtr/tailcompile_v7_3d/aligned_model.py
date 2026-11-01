"""008: one deployment decoder, feasible labels and hard-forward risk training.

The forward risk is evaluated on the very same capacity-decoded layer and
finite-density XY used by deployment. Its probability gradient is explicitly a
biased straight-through surrogate; neither the allocator nor VPR is
differentiated. Historical 007 modules remain read-only dependencies.
"""
from __future__ import annotations

import math
import time
import hashlib
import torch
from torch.nn import functional as F

try:
    from .layer_model import ExplicitLayerInitializer, deterministic_layers
    from .model import validate, compatibility_mask, allocate_layers, CapacityError
except ImportError:
    from layer_model import ExplicitLayerInitializer, deterministic_layers
    from model import validate, compatibility_mask, allocate_layers, CapacityError

SCHEMA = 'tailcompile-v7-aligned-layer-generator-008-1'
DECODER_SCHEMA = 'aligned-hard-layer-density-008-1'
RISK_GRADIENT = ('hard decoded layer/XY forward; straight-through layer-probability gradient; '
                 'no allocator/XY/VPR gradient; noisy-OR clamped active sinks can have zero gradient')


def action_digest(action):
    result = hashlib.sha256()
    for key, dtype in (('assignment', torch.long), ('layer', torch.long), ('xy', torch.float32)):
        tensor = torch.as_tensor(action[key], dtype=dtype).detach().cpu().contiguous()
        result.update(key.encode()); result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def validate_action(data, action, *, chunk_size=256):
    """Check a complete action, not a teacher stamp or a physical packing claim."""
    try:
        from .aligned_teacher import TOLERANCE, deployed_xy_bounds
    except ImportError:
        from aligned_teacher import TOLERANCE, deployed_xy_bounds
    meta = validate(data)
    device = data['demand'].device
    n = len(data['demand'])
    def indices(name):
        raw = torch.as_tensor(action[name], device=device)
        if raw.dtype == torch.bool or raw.is_complex() or not torch.isfinite(raw).all():
            raise ValueError('aligned action indices must be finite integers')
        if raw.is_floating_point() and (raw != raw.floor()).any():
            raise ValueError('aligned action indices must be integers without truncation')
        return raw.long()
    leaves, layers = indices('assignment'), indices('layer')
    xy = torch.as_tensor(action['xy'], device=device, dtype=data['demand'].dtype)
    if leaves.shape != (n,) or layers.shape != (n,) or xy.shape != (n, 2):
        raise ValueError('aligned action must cover all APBlocks with leaf/layer/XY')
    if (leaves < 0).any() or (leaves >= len(data['capacity'])).any():
        raise ValueError('aligned action leaf outside device')
    if not torch.equal(layers, meta[0][leaves]):
        raise ValueError('aligned action leaf/layer disagreement')
    if not torch.isfinite(xy).all() or (xy < 0).any() or (xy >= 1).any():
        raise ValueError('aligned action XY outside normalized device')
    for start in range(0, n, chunk_size):
        ids = torch.arange(start, min(start+chunk_size, n), device=device)
        mask = compatibility_mask(data, ids, meta)
        if not mask.gather(1, leaves[ids, None]).all():
            raise ValueError('aligned action violates resource/fixed compatibility')
    bounds = data['region_bounds'][leaves]
    if ((xy[:, 0] < bounds[:, 0]-TOLERANCE) | (xy[:, 0] > bounds[:, 1]+TOLERANCE) |
        (xy[:, 1] < bounds[:, 2]-TOLERANCE) | (xy[:, 1] > bounds[:, 3]+TOLERANCE)).any():
        raise ValueError('aligned action XY outside selected leaf')
    known = meta[4] >= 0
    if not torch.allclose(xy[known], meta[4][known], atol=TOLERANCE, rtol=0):
        raise ValueError('aligned action moved a fixed XY axis')
    low, high = deployed_xy_bounds(data, leaves, meta)
    if ((xy < low.to(xy)-TOLERANCE) | (xy > high.to(xy)+TOLERANCE)).any():
        raise ValueError('aligned action XY outside the deployed blend-reachable range')
    usage = data['demand'].new_zeros(meta[1].shape)
    usage.index_add_(0, layers, data['demand'])
    overflow = (usage-meta[1]).clamp_min(0)
    if (overflow > TOLERANCE).any():
        raise ValueError('aligned action exceeds hard layer-capacity proxy')
    return dict(layer_usage=usage.detach().cpu().tolist(),
                layer_capacity=meta[1].detach().cpu().tolist(),
                layer_overflow_by_resource=overflow.sum(0).detach().cpu().tolist(),
                valid=True, packing_legality_claimed=False)


class AlignedLayerInitializer(ExplicitLayerInitializer):
    def __init__(self, *args, density_passes=2, resource_ce_fraction=.5, **kwargs):
        super().__init__(*args, **kwargs)
        if isinstance(density_passes, bool) or not isinstance(density_passes, int) or not 0 <= density_passes <= 4:
            raise ValueError('density_passes must be a bounded integer in [0,4]')
        if not math.isfinite(resource_ce_fraction) or not 0 <= resource_ce_fraction <= 1:
            raise ValueError('resource_ce_fraction must lie in [0,1]')
        self.density_passes = density_passes
        self.resource_ce_fraction = float(resource_ce_fraction)
        self.config.update(density_passes=density_passes, resource_ce_fraction=self.resource_ce_fraction)

    @torch.no_grad()
    def decode_action(self, data, *, policy=None, method='L', seed=0, score_noise_std=0., training=False):
        """One hard decoder for training and serving; no risk diagnostics here."""
        if method not in ('D', 'L', 'T') or not math.isfinite(score_noise_std) or score_noise_std < 0:
            raise ValueError('invalid aligned method/noise')
        if not self.shared_xy_frozen and not training:
            raise ValueError('deployment requires the complete frozen XY module')
        started = time.perf_counter()
        if method == 'D':
            meta = validate(data)
            allowed = self._allowed(data, meta)
            chosen, layer_usage = deterministic_layers(data, allowed)
        else:
            scores, probability, allowed, meta = policy if policy is not None else self.layer_distribution(data)
            scores = scores.detach()
            if score_noise_std:
                rng = torch.Generator(device='cpu').manual_seed(seed)
                scores = scores + score_noise_std*torch.randn(scores.shape, generator=rng).to(scores)
            chosen, layer_usage = allocate_layers(scores, data['demand'], meta[1], allowed)
        chosen = chosen.to(data['demand'].device)
        was_training = self.shared_xy.training
        self.shared_xy.eval()
        try:
            logic, regions = self.shared_xy.encode(data)
            encoded = time.perf_counter()
            assignment = torch.empty(len(logic), dtype=torch.long, device=logic.device)
            prices = torch.zeros_like(data['capacity'])
            for iteration in range(self.density_passes+1):
                expected = torch.zeros_like(prices)
                for start in range(0, len(logic), self.chunk_size):
                    ids = torch.arange(start, min(start+self.chunk_size, len(logic)), device=logic.device)
                    mask = compatibility_mask(data, ids, meta) & (meta[0][None, :] == chosen[ids, None])
                    scores = self.shared_xy.matching_logits(logic[ids], regions)
                    scores = scores-data['demand'][ids] @ (prices/data['capacity'].clamp_min(1)).T
                    scores = scores.masked_fill(~mask, -torch.inf)
                    if iteration == self.density_passes:
                        assignment[ids] = scores.argmax(1)
                    else:
                        expected.add_(scores.softmax(1).T @ data['demand'][ids])
                if iteration < self.density_passes:
                    prices = (prices+(expected-data['capacity'])/data['capacity'].clamp_min(1)).clamp(0, 100)
            xy = self.shared_xy.deployment_coordinates(data, logic, regions, assignment)
        finally:
            self.shared_xy.train(was_training and not self.shared_xy_frozen)
        action = dict(assignment=assignment.cpu(), layer=chosen.cpu(), xy=xy.cpu())
        layer_audit = validate_action(data, action, chunk_size=self.chunk_size)
        usage = torch.zeros_like(data['capacity']); usage.index_add_(0, assignment, data['demand'])
        overflow = (usage-data['capacity']).clamp_min(0)
        action.update(usage=usage.cpu(), layer_usage=layer_usage.cpu(),
            initializer='aligned_layer_'+method, decoder_kind=DECODER_SCHEMA,
            capacity_diagnostics=dict(**layer_audit, hard_layer_capacity_enforced=True,
                hard_leaf_capacity_enforced=False,
                overflow_by_resource=overflow.sum(0).cpu().tolist(),
                overflow_region_count=int((overflow > 1e-6).any(1).sum()),
                used_region_count=int(assignment.unique().numel())),
            phase_seconds=dict(encoding=encoded-started, decoding=time.perf_counter()-encoded,
                               total=time.perf_counter()-started))
        action['deployed_action_sha256'] = action_digest(action)
        return action

    def supervision_loss(self, data, teacher, *, tail_weight=0., shared=False):
        try:
            from .aligned_teacher import verify_teacher
        except ImportError:
            from aligned_teacher import verify_teacher
        teacher_audit = verify_teacher(data, teacher, chunk_size=self.chunk_size)
        policy = self.layer_distribution(data)
        logits, prob, _, meta = policy
        movable = data.get('movable', torch.ones(len(prob), dtype=torch.bool, device=prob.device)).bool()
        if not movable.any():
            raise ValueError('aligned training needs at least one movable APBlock')
        labels = torch.as_tensor(teacher['teacher_region'], dtype=torch.long, device=prob.device)
        target_layer = torch.as_tensor(teacher['teacher_layer'], dtype=torch.long, device=prob.device)
        per_node = F.cross_entropy(logits[movable], target_layer[movable], reduction='none')
        demand = data['demand'][movable]
        resource_ce, resource_details = [], []
        for resource in range(5):
            mass = demand[:, resource]
            if float(mass.sum()) > 0:
                value = (per_node*mass).sum()/mass.sum()
                resource_ce.append(value); resource_details.append(float(value.detach()))
            else:
                resource_details.append(None)
        ce_plain = per_node.mean()
        ce_resource = torch.stack(resource_ce).mean() if resource_ce else ce_plain
        ce = (1-self.resource_ce_fraction)*ce_plain+self.resource_ce_fraction*ce_resource
        expected = prob.T @ data['demand']
        relative = (expected-meta[1]).clamp_min(0)/meta[1].clamp_min(1.)
        per_resource_capacity = relative.amax(0).square()
        capacity = per_resource_capacity.sum()
        action = self.decode_action(data, policy=policy, training=shared)
        hard_layers = action['layer'].to(prob.device)
        hard_xy = action['xy'].to(prob)
        hard_prob = F.one_hot(hard_layers, prob.shape[1]).to(prob)
        # Parentheses make the forward correction exactly zero, rather than
        # adding/subtracting a soft probability after rounding the hard value.
        st_prob = hard_prob+(prob-prob.detach())
        source, sink = data['logic_src'], data['logic_dst']
        weight = data['logic_weight']*(1+data.get('logic_criticality', torch.zeros_like(data['logic_weight'])).clamp_min(0))
        cut = ((1-(st_prob[source]*st_prob[sink]).sum(1))*weight).sum()/weight.sum().clamp_min(1.)
        # These positions provide only the layer-score surrogate. XY is
        # supervised separately below, so building its autograd graph here
        # would consume memory and then discard all of those gradients.
        with torch.no_grad():
            conditional = self.conditional_positions(data)
        soft_xy = (prob[:, :, None]*conditional).sum(1)
        st_xy = hard_xy+(soft_xy-soft_xy.detach())
        wire = ((st_xy[source]-st_xy[sink]).abs().sum(1)*weight).sum()/weight.sum().clamp_min(1.)
        loss = ce+.1*capacity+.1*cut+.1*wire
        detail = dict(aligned_teacher_valid=teacher_audit['valid'], layer_ce=float(ce.detach()),
            layer_ce_unweighted=float(ce_plain.detach()), resource_ce=resource_details,
            resource_ce_fraction=self.resource_ce_fraction,
            expected_capacity_overflow=float(capacity.detach()),
            expected_capacity_per_resource=per_resource_capacity.detach().cpu().tolist(),
            decoded_timing_proxy_cut=float(cut.detach()), decoded_xy_wire_proxy=float(wire.detach()),
            capacity_decoder_schema=DECODER_SCHEMA, density_passes=self.density_passes,
            deployed_action_sha256=action['deployed_action_sha256'],
            decoded_layer_counts=torch.bincount(hard_layers, minlength=prob.shape[1]).detach().cpu().tolist(),
            risk_gradient_semantics=RISK_GRADIENT,
            cut_xy_gradient='hard forward; straight-through layer/conditional-XY surrogate')
        if shared:
            if self.shared_xy_frozen:
                raise ValueError('cannot pretrain frozen shared XY')
            logic, regions = self.shared_xy.encode(data)
            conditional_ce = logic.new_zeros(())
            count = 0
            for start in range(0, len(logic), self.chunk_size):
                ids = torch.arange(start, min(start+self.chunk_size, len(logic)), device=prob.device)
                ids = ids[movable[ids]]
                if not len(ids): continue
                mask = compatibility_mask(data, ids, meta) & (meta[0][None, :] == target_layer[ids, None])
                scores = self.shared_xy.matching_logits(logic[ids], regions).masked_fill(~mask, -torch.inf)
                conditional_ce = conditional_ce+F.cross_entropy(scores, labels[ids], reduction='sum')
                count += len(ids)
            predicted = self.shared_xy.deployment_coordinates(data, logic, regions, labels)
            target_xy = torch.as_tensor(teacher['teacher_xy'], device=prob.device, dtype=prob.dtype)
            bounds = data['region_bounds'][labels]
            span = (bounds[:, [1, 3]]-bounds[:, [0, 2]]).clamp_min(1e-6)
            xy_loss = F.smooth_l1_loss(predicted[movable]/span[movable], target_xy[movable]/span[movable])
            loss = loss+conditional_ce/count+5*xy_loss
            detail.update(conditional_leaf_ce=float((conditional_ce/count).detach()), coordinate=float(xy_loss.detach()))
        if tail_weight or 'vertical_risk_device' in data:
            try:
                from .vertical_risk import risk_loss
            except ImportError:
                from vertical_risk import risk_loss
            tail, diagnostics = risk_loss(data, st_prob, hard_xy[:, None, :].expand(-1, prob.shape[1], -1))
            unassigned = diagnostics.pop('unassigned_mass_tensor', tail.new_zeros(()))
            unavailable = unassigned/max(1, diagnostics.get('active_net_count', 1))
            loss = loss+unavailable+float(tail_weight)*tail/self.vertical_tail_scale
            detail.update(vertical_tail=float(tail.detach()), vertical_tail_scale=self.vertical_tail_scale,
                decoded_unassigned_constraint=float(unavailable.detach()), vertical_risk=diagnostics,
                risk_forward='actual capacity-decoded layer and finite-density XY',
                risk_solver_differentiated=False)
        return loss, detail

    def action_log_probability(self, data, action, *, probability=None):
        validate_action(data, action, chunk_size=self.chunk_size)
        return super().action_log_probability(data, action, probability=probability)


@torch.no_grad()
def generate(model, data, *, method='L', seed=0, score_noise_std=0., diagnostics=False, density_passes=None):
    if density_passes is not None and density_passes != model.density_passes:
        raise ValueError('training/deployment density_passes must use the checkpoint contract')
    model.eval()
    action = model.decode_action(data, method=method, seed=seed, score_noise_std=score_noise_std)
    if diagnostics:
        try:
            from .vertical_risk import hard_risk
        except ImportError:
            from vertical_risk import hard_risk
        action['hard_vertical_risk'] = hard_risk(data, action['layer'], action['xy'].to(data['demand']))
    action['risk_diagnostics_computed'] = bool(diagnostics)
    return action
