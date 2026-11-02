"""011: frozen D GNN plus learned selection of bounded, feasible residuals.

The small ranking heads choose an actual member of a finite candidate bank.
They do not differentiate through the capacity allocator or VPR. The backbone
and candidate construction run on CPU to keep collected/deployed action bytes
identical when the ranking heads are trained on CUDA.
"""
from __future__ import annotations

import copy
import math
import torch
from torch import nn

try:
    from .structured_model import generate as generate_d
    from .aligned_model import action_digest
    from .residual_actions import make_candidates, action_features, ACTION_FEATURE_DIM, validate_residual_action
except ImportError:
    from structured_model import generate as generate_d
    from aligned_model import action_digest
    from residual_actions import make_candidates, action_features, ACTION_FEATURE_DIM, validate_residual_action

SCHEMA = 'tailcompile-v7-residual-selector-011-1'
OBJECTIVE = 'measured-vpr-qor-residual-ranking-011-1'
ARMS = ('L', 'Y', 'J')


class ResidualSelector(nn.Module):
    def __init__(self, backbone, *, hidden=32, strengths=(.15, .35), max_moves=256,
                 fraction=.02, seed=11301, source_checkpoint_sha256=None):
        super().__init__()
        if not isinstance(hidden, int) or isinstance(hidden, bool) or not 4 <= hidden <= 256:
            raise ValueError('011 ranking hidden width must be bounded in [4,256]')
        self.backbone = backbone.cpu().eval()
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.source_checkpoint_sha256 = source_checkpoint_sha256
        self.config = dict(hidden=hidden, strengths=list(strengths), max_moves=max_moves,
                           fraction=fraction, seed=seed)
        width = 4*backbone.layer_query.in_features+ACTION_FEATURE_DIM
        self.heads = nn.ModuleDict({arm: nn.Sequential(nn.Linear(width, hidden), nn.GELU(), nn.Linear(hidden, 1))
                                   for arm in ARMS})
        # Identity wins every initial tie. Training must earn a change to D.
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def ranking_to(self, device):
        """Move only trainable heads; immutable candidate computation stays CPU."""
        self.heads.to(device)
        return self

    @torch.no_grad()
    def candidate_bank(self, data, arm, *, base_action=None):
        if arm not in ARMS:
            raise ValueError('011 arm must be L, Y or J')
        if data['demand'].device.type != 'cpu':
            raise ValueError('011 deterministic candidate graphs must remain on CPU')
        base = (generate_d(self.backbone, data, method='D', score_noise_std=0., diagnostics=False)
                if base_action is None else base_action)
        candidates = make_candidates(data, base, arm, **{k: self.config[k] for k in
                                      ('strengths', 'max_moves', 'fraction', 'seed')})
        if not candidates or candidates[0]['candidate_id'] != 'identity':
            raise ValueError('011 candidate bank must start with unchanged D identity')
        if action_digest(candidates[0]) != action_digest(base):
            raise ValueError('011 candidate identity changed D')
        for action in candidates:
            action['source_checkpoint_sha256'] = self.source_checkpoint_sha256
        return base, candidates

    def generate_candidates(self, data, arm):
        return self.candidate_bank(data, arm)[1]

    @torch.no_grad()
    def context(self, data):
        if data['demand'].device.type != 'cpu':
            raise ValueError('011 context graphs must remain on CPU')
        self.backbone.eval()
        logic, region = self.backbone.encode_shared(data)
        pooled = []
        for embedding in (logic, region):
            if len(embedding) == 0:
                raise ValueError('011 empty logical/device graph')
            pooled.extend((embedding.mean(0), embedding.std(0, unbiased=False)))
        result = torch.cat(pooled).tanh().detach()
        if not torch.isfinite(result).all():
            raise FloatingPointError('011 frozen GNN context is nonfinite')
        return result

    @torch.no_grad()
    def candidate_features(self, data, candidates, base_action, *, context=None):
        context = self.context(data) if context is None else context.detach().cpu()
        features = []
        for candidate in candidates:
            validate_residual_action(data, candidate, base_action=base_action,
                                     arm=candidate.get('residual_arm'))
            local = action_features(data, base_action, candidate).detach().cpu().float()
            if local.shape != (ACTION_FEATURE_DIM,) or not torch.isfinite(local).all():
                raise ValueError('011 invalid candidate descriptor')
            features.append(torch.cat((context, local)))
        return torch.stack(features)

    def score_features(self, features, arm):
        if arm not in ARMS:
            raise ValueError('011 unknown ranking arm')
        device = next(self.heads.parameters()).device
        scores = self.heads[arm](features.to(device)).flatten()
        # Index zero is identity. This offset cancels for every measured pair.
        return scores-scores[0]

    def score_candidates(self, data, candidates, base_action, *, arm=None):
        arm = arm or candidates[0].get('residual_arm')
        return self.score_features(self.candidate_features(data, candidates, base_action), arm)

    @torch.no_grad()
    def select_action(self, data, arm, *, enabled=True):
        self.eval()
        base, candidates = self.candidate_bank(data, arm)
        return self.select_from_bank(data, arm, base, candidates, enabled=enabled)

    @torch.no_grad()
    def select_from_bank(self, data, arm, base, candidates, *, enabled=True):
        """Select and return a candidate without repeating frozen D generation."""
        self.eval()
        if not candidates or candidates[0]['candidate_id'] != 'identity' or action_digest(candidates[0]) != action_digest(base):
            raise ValueError('011 precomputed bank lacks unchanged identity')
        scores = self.score_candidates(data, candidates, base, arm=arm)
        if not torch.isfinite(scores).all():
            raise FloatingPointError('011 nonfinite learned candidate score')
        index = int(scores.argmax()) if enabled else 0
        result = copy.deepcopy(candidates[index])
        result.update(initializer='residual_'+arm,
                      selection=dict(candidate_id=result['candidate_id'], enabled=bool(enabled),
                                     scores=scores.detach().cpu().tolist(),
                                     candidate_ids=[c['candidate_id'] for c in candidates],
                                     policy='finite feasible residual bank; identity first on ties',
                                     teacher_or_qor_used_at_inference=False),
                      source_checkpoint_sha256=self.source_checkpoint_sha256,
                      tail_weight=0., risk_diagnostics_computed=False)
        if result['deployed_action_sha256'] != action_digest(result):
            raise ValueError('011 selected action digest mismatch')
        return result


@torch.no_grad()
def generate(model, data, *, method='J', arm=None, enabled=True, **kwargs):
    if kwargs.get('diagnostics') or kwargs.get('score_noise_std', 0.):
        raise ValueError('011 has no TailRisk diagnostics or stochastic deployment')
    return model.select_action(data, arm or method, enabled=enabled)
