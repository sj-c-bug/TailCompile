"""009 QoR actor: common hard deployment, optional trainable conditional XY.

Historical 008 actors remain immutable. The full action score below is an
off-policy surrogate, not the probability of the constrained hard allocator.
"""
from __future__ import annotations

import math
import torch
from torch.nn import functional as F

try:
    from .aligned_model import AlignedLayerInitializer, validate_action, action_digest, RISK_GRADIENT
    from .aligned_teacher import verify_teacher, constraints_digest
    from .model import validate, compatibility_mask
except ImportError:
    from aligned_model import AlignedLayerInitializer, validate_action, action_digest, RISK_GRADIENT
    from aligned_teacher import verify_teacher, constraints_digest
    from model import validate, compatibility_mask

SCHEMA = 'tailcompile-v7-qor-generator-009-1'
DECODER_SCHEMA = 'qor-hard-layer-density-009-1'
ACTION_SCHEMA = 'tailcompile-v7-qor-action-009-1'
ARMS = ('R', 'Q', 'C', 'X', 'T')
SIGMA = .25


def _identity(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_collected_action(data, action, *, allowed_xy_sha256=None, chunk_size=256):
    """Verify collection provenance; old actions keep their original schema."""
    schema, decoder = action.get('schema'), action.get('decoder_kind')
    if schema == 'tailcompile-v7-aligned-action-008-1':
        if decoder != 'aligned-hard-layer-density-008-1':
            raise ValueError('legacy action decoder identity mismatch')
    elif schema == ACTION_SCHEMA:
        if decoder != DECODER_SCHEMA or action.get('collection_arm') not in ARMS:
            raise ValueError('009 collection decoder/arm identity mismatch')
        if not _identity(action.get('collection_checkpoint_sha256')):
            raise ValueError('009 action requires its collection checkpoint identity')
    else:
        raise ValueError('only explicitly collected 008/009 actions are supported')
    collected_xy = action.get('shared_xy_sha256')
    if not _identity(collected_xy):
        raise ValueError('collection XY identity missing')
    if allowed_xy_sha256 is not None and collected_xy != allowed_xy_sha256:
        raise ValueError('action outside frozen XY actor support')
    if action.get('constraints_sha256') != constraints_digest(data):
        raise ValueError('action graph constraints identity mismatch')
    return validate_action(data, action, chunk_size=chunk_size)


class QorInitializer(AlignedLayerInitializer):
    def __init__(self, *args, arm='Q', **kwargs):
        if arm not in ARMS:
            raise ValueError('unknown QoR arm')
        super().__init__(*args, **kwargs)
        self.arm = arm
        self.config.update(arm=arm)
        if arm in ('R', 'Q', 'C'):
            self.freeze_shared_xy()
        else:
            self.shared_xy_frozen = False
            self.shared_xy.requires_grad_(True)

    @torch.no_grad()
    def decode_action(self, data, *, policy=None, method=None, seed=0,
                      score_noise_std=0., training=False):
        if method is not None and method != self.arm:
            raise ValueError('requested method must match checkpoint arm')
        # The bounded 008 allocator is reused verbatim. `training=True` only
        # bypasses its legacy frozen-XY assertion; this entire method is no-grad.
        action = super().decode_action(data, policy=policy, method='L', seed=seed,
            score_noise_std=score_noise_std, training=True)
        action.update(initializer='qor_'+self.arm, decoder_kind=DECODER_SCHEMA,
                      actor_arm=self.arm)
        action['deployed_action_sha256'] = action_digest(action)
        return action

    def conditional_components(self, data, action, *, encoded=None, metadata=None):
        """Differentiable teacher-forced leaf/XY on a fixed complete action.

        Coordinates are the same finite-density blend as lowering. Leaf
        log-probability is a pre-price conditional surrogate; neither discrete
        leaf argmax nor density prices are differentiated.
        """
        meta = validate(data) if metadata is None else metadata
        logic, regions = self.shared_xy.encode(data) if encoded is None else encoded
        leaves = torch.as_tensor(action['assignment'], device=logic.device, dtype=torch.long)
        layers = torch.as_tensor(action['layer'], device=logic.device, dtype=torch.long)
        target_xy = torch.as_tensor(action['xy'], device=logic.device, dtype=logic.dtype)
        movable = data.get('movable', torch.ones(len(logic), dtype=torch.bool, device=logic.device)).bool()
        if not movable.any(): raise ValueError('QoR action has no movable APBlock')
        total = logic.new_zeros(())
        for start in range(0, len(logic), self.chunk_size):
            ids = torch.arange(start, min(start+self.chunk_size, len(logic)), device=logic.device)
            ids = ids[movable[ids]]
            if not len(ids): continue
            mask = compatibility_mask(data, ids, meta) & (meta[0][None, :] == layers[ids, None])
            logits = self.shared_xy.matching_logits(logic[ids], regions).masked_fill(~mask, -torch.inf)
            chosen = logits.gather(1, leaves[ids, None])
            if not torch.isfinite(chosen).all(): raise ValueError('candidate conditional leaf incompatible')
            total = total+F.log_softmax(logits, dim=1).gather(1, leaves[ids,None]).sum()
        prediction = self.shared_xy.deployment_coordinates(data, logic, regions, leaves)
        bounds = data['region_bounds'][leaves]
        span = (bounds[:,[1,3]]-bounds[:,[0,2]]).clamp_min(1e-6)
        fixed = data.get('fixed_xy', torch.full_like(prediction,-1))
        active = movable[:,None] & (fixed < 0)
        squared = ((prediction-target_xy)/span).square()
        mse = squared[active].mean() if active.any() else prediction.sum()*0
        smooth = F.smooth_l1_loss(prediction[active]/span[active],target_xy[active]/span[active]) if active.any() else mse
        return dict(leaf_log_probability=total/movable.sum(), xy_squared_error=mse,
                    teacher_xy_smooth_l1=smooth, prediction=prediction)

    def action_score(self, data, action, *, probability=None, log_probability=None,
                     encoded=None, validated=False, constant=None):
        if not validated: validate_action(data, action, chunk_size=self.chunk_size)
        # Do not log an already rounded softmax probability: valid historical
        # actions can be many orders of magnitude below the current layer mode.
        # The optional `probability` argument is accepted for old callers, but
        # logits are recomputed if no stable log_probability was supplied.
        log_prob = F.log_softmax(self.layer_distribution(data)[0],dim=1) if log_probability is None else log_probability
        layers = torch.as_tensor(action['layer'], device=log_prob.device, dtype=torch.long)
        movable = data.get('movable',torch.ones(len(log_prob),dtype=torch.bool,device=log_prob.device)).bool()
        selected = log_prob.gather(1,layers[:,None]).squeeze(1)
        if not movable.any() or not torch.isfinite(selected[movable]).all():
            raise ValueError('action incompatible with current layer policy')
        layer = selected[movable].mean()
        if constant is None:
            parts = self.conditional_components(data,action,encoded=encoded)
            constant = parts['leaf_log_probability']-parts['xy_squared_error']/(2*SIGMA**2)
        return layer+constant

    def supervision_loss(self, data, teacher, *, tail_weight=None, shared=False):
        if shared: raise ValueError('009 starts from a signed 008 checkpoint, not fresh shared pretraining')
        audit = verify_teacher(data,teacher,chunk_size=self.chunk_size)
        policy = self.layer_distribution(data)
        logits, prob, _, meta = policy
        movable = data.get('movable',torch.ones(len(prob),dtype=torch.bool,device=prob.device)).bool()
        labels = torch.as_tensor(teacher['teacher_region'],dtype=torch.long,device=prob.device)
        target_layers = torch.as_tensor(teacher['teacher_layer'],dtype=torch.long,device=prob.device)
        per_node = F.cross_entropy(logits[movable],target_layers[movable],reduction='none')
        mass = data['demand'][movable]
        terms = [(per_node*mass[:,r]).sum()/mass[:,r].sum() for r in range(5) if float(mass[:,r].sum()) > 0]
        plain = per_node.mean()
        ce = (1-self.resource_ce_fraction)*plain+self.resource_ce_fraction*torch.stack(terms).mean()
        relative = (prob.T @ data['demand']-meta[1]).clamp_min(0)/meta[1].clamp_min(1)
        capacity = relative.amax(0).square().sum()
        action = self.decode_action(data,policy=policy)
        layers = action['layer'].to(prob.device); xy = action['xy'].to(prob)
        hard = F.one_hot(layers,prob.shape[1]).to(prob)
        st_prob = hard+(prob-prob.detach())
        src,dst = data['logic_src'],data['logic_dst']
        weights = data['logic_weight']*(1+data.get('logic_criticality',torch.zeros_like(data['logic_weight'])).clamp_min(0))
        cut = ((1-(st_prob[src]*st_prob[dst]).sum(1))*weights).sum()/weights.sum().clamp_min(1)
        # Keep the existing XY geometry proxy identical across arms: its
        # surrogate updates layers only, not an undeclared coordinate objective.
        with torch.no_grad(): conditional = self.conditional_positions(data)
        soft_xy = (prob[:,:,None]*conditional).sum(1)
        st_xy = xy+(soft_xy-soft_xy.detach())
        wire = ((st_xy[src]-st_xy[dst]).abs().sum(1)*weights).sum()/weights.sum().clamp_min(1)
        cut_weight = .1 if self.arm in ('R','Q') else 0.
        teacher_action = dict(assignment=labels,layer=target_layers,
            xy=torch.as_tensor(teacher['teacher_xy'],device=prob.device,dtype=prob.dtype))
        conditional_detail = self.conditional_components(data,teacher_action,metadata=meta)
        coordinate_loss = conditional_detail['teacher_xy_smooth_l1']
        loss = ce+.1*capacity+cut_weight*cut+.1*wire-conditional_detail['leaf_log_probability']+5*coordinate_loss
        detail = dict(aligned_teacher_valid=audit['valid'],layer_ce=float(ce.detach()),
            layer_ce_unweighted=float(plain.detach()),expected_capacity_overflow=float(capacity.detach()),
            decoded_timing_proxy_cut=float(cut.detach()),cut_weight=cut_weight,
            decoded_xy_wire_proxy=float(wire.detach()),conditional_leaf_ce=float(-conditional_detail['leaf_log_probability'].detach()),
            coordinate=float(coordinate_loss.detach()),xy_trainable=not self.shared_xy_frozen,
            deployed_action_sha256=action['deployed_action_sha256'],capacity_decoder_schema=DECODER_SCHEMA,
            density_passes=self.density_passes,risk_gradient_semantics=RISK_GRADIENT,
            decoded_layer_counts=torch.bincount(layers,minlength=prob.shape[1]).cpu().tolist())
        actual_tail_weight = .1 if self.arm == 'T' else 0.
        if tail_weight is not None and tail_weight != actual_tail_weight:
            raise ValueError('Tail weight is fixed by the arm contract')
        if 'vertical_risk_device' in data:
            try: from .vertical_risk import risk_loss
            except ImportError: from vertical_risk import risk_loss
            tail,diag = risk_loss(data,st_prob,xy[:,None,:].expand(-1,prob.shape[1],-1))
            unavailable = diag.pop('unassigned_mass_tensor',tail.new_zeros(()))/max(1,diag.get('active_net_count',1))
            loss = loss+unavailable+actual_tail_weight*tail/self.vertical_tail_scale
            detail.update(vertical_tail=float(tail.detach()),vertical_tail_scale=self.vertical_tail_scale,
                decoded_unassigned_constraint=float(unavailable.detach()),vertical_risk=diag,
                risk_forward='actual capacity-decoded layer and finite-density XY',risk_solver_differentiated=False)
        elif actual_tail_weight:
            raise ValueError('Tail arm requires real vertical-risk device input')
        return loss,detail


@torch.no_grad()
def generate(model,data,*,method=None,seed=0,score_noise_std=0.,diagnostics=False):
    model.eval()
    action = model.decode_action(data,method=method,seed=seed,score_noise_std=score_noise_std)
    if diagnostics:
        try: from .vertical_risk import hard_risk
        except ImportError: from vertical_risk import hard_risk
        action['hard_vertical_risk'] = hard_risk(data,action['layer'],action['xy'].to(data['demand']))
    action['risk_diagnostics_computed'] = bool(diagnostics)
    return action
