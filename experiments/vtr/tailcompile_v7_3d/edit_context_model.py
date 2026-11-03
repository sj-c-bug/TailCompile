"""012 controlled candidate readout, with frozen GNN and unchanged 011 actions.

G and E have identical dimensions/parameters/initial weights. G masks the local
feature block to zero; E observes it. This model scores a finite candidate bank,
not arbitrary new coordinates. Measured QoR is never an inference input.
"""
from __future__ import annotations

import copy
import torch
from torch import nn

try:
    from .residual_model import ARMS
    from .residual_actions import action_features, ACTION_FEATURE_DIM, validate_residual_action
    from .aligned_model import action_digest
    from .edit_context_features import local_features, edit_feature_dim, FEATURE_SCHEMA
except ImportError:
    from residual_model import ARMS
    from residual_actions import action_features, ACTION_FEATURE_DIM, validate_residual_action
    from aligned_model import action_digest
    from edit_context_features import local_features, edit_feature_dim, FEATURE_SCHEMA

SCHEMA = 'tailcompile-v7-edit-context-head-012-1'
OBJECTIVE = 'same-measured-011-actions-local-readout-012-1'


@torch.no_grad()
def feature_blocks(backbone, data, base_action, candidates, *, encoded=None):
    """One frozen CPU encoding, no targets or post-route fields consumed."""
    if data['demand'].device.type != 'cpu':
        raise ValueError('012 frozen features/candidate generation must stay on CPU')
    backbone.eval()
    logic, regions = backbone.encode_shared(data) if encoded is None else encoded
    context = torch.cat([v for embedding in (logic, regions) for v in
                         (embedding.mean(0), embedding.std(0, unbiased=False))]).tanh()
    global_rows = []
    for action in candidates:
        validate_residual_action(data, action, base_action=base_action,
                                 arm=action.get('residual_arm'))
        global_rows.append(torch.cat((context, action_features(data, base_action, action).float())))
    global_rows = torch.stack(global_rows).detach().cpu().float()
    local_rows = local_features(data, base_action, candidates, logic, regions).detach().cpu().float()
    if not torch.isfinite(global_rows).all() or not torch.isfinite(local_rows).all():
        raise ValueError('012 feature blocks must be finite')
    return global_rows, local_rows


class EditContextModel(nn.Module):
    def __init__(self, global_dim, local_dim, *, mode='G', hidden=32,
                 source_checkpoint_sha256=None, backbone_sha256=None,
                 cache_sha256=None, feature_schema=FEATURE_SCHEMA):
        super().__init__()
        if mode not in ('G', 'E') or not all(isinstance(v, int) and not isinstance(v, bool) and v>0
                                            for v in (global_dim, local_dim, hidden)):
            raise ValueError('012 valid G/E mode and positive integer dimensions required')
        if hidden > 256 or feature_schema != FEATURE_SCHEMA:
            raise ValueError('012 ranking dimensions/schema out of bounds')
        self.config = dict(global_dim=global_dim, local_dim=local_dim, mode=mode, hidden=hidden,
            source_checkpoint_sha256=source_checkpoint_sha256, backbone_sha256=backbone_sha256,
            cache_sha256=cache_sha256, feature_schema=feature_schema)
        self.mode, self.global_dim, self.local_dim = mode, global_dim, local_dim
        self.heads = nn.ModuleDict({arm: nn.Sequential(nn.Linear(global_dim+local_dim, hidden),
            nn.GELU(), nn.Linear(hidden, 1)) for arm in ARMS})
        for head in self.heads.values():
            nn.init.zeros_(head[-1].weight); nn.init.zeros_(head[-1].bias)

    def score_features(self, global_features, local_features, arm):
        if arm not in ARMS:
            raise ValueError('012 arm must be L/Y/J')
        if global_features.ndim!=2 or local_features.ndim!=2 or \
            global_features.shape[1]!=self.global_dim or local_features.shape[1]!=self.local_dim or \
            len(global_features)!=len(local_features) or len(global_features)==0:
            raise ValueError('012 feature dimensions/coverage mismatch')
        if not torch.isfinite(global_features).all() or not torch.isfinite(local_features).all():
            raise ValueError('012 nonfinite scoring input')
        device=next(self.parameters()).device
        local = local_features if self.mode=='E' else torch.zeros_like(local_features)
        values=torch.cat((global_features, local),1).to(device=device,dtype=torch.float32)
        scores=self.heads[arm](values).flatten()
        return scores-scores[0]

    @torch.no_grad()
    def select_action(self, data, arm, source_checkpoint, *, enabled=True):
        """Deploy one unchanged 011 candidate without consulting cached labels."""
        try:
            from .train_residual import from_source
            from .train import state_sha256
        except ImportError:
            from train_residual import from_source
            from train import state_sha256
        source=from_source(source_checkpoint, expected_sha256=self.config['source_checkpoint_sha256'])
        if state_sha256(source.backbone)!=self.config['backbone_sha256']:
            raise ValueError('012 frozen source backbone differs from training features')
        base,candidates=source.candidate_bank(data,arm)
        global_features,local_values=feature_blocks(source.backbone,data,base,candidates)
        scores=self.score_features(global_features,local_values,arm)
        if not torch.isfinite(scores).all():
            raise ValueError('012 nonfinite selection scores')
        index=int(scores.argmax()) if enabled else 0
        chosen=copy.deepcopy(candidates[index])
        chosen.update(initializer='edit_context_'+self.mode+'_'+arm,
            selection=dict(candidate_id=chosen['candidate_id'],candidate_ids=[a['candidate_id'] for a in candidates],
                scores=scores.cpu().tolist(),enabled=bool(enabled),mode=self.mode,
                teacher_or_qor_used_at_inference=False,action_space='unchanged finite 011 candidate bank'),
            local_feature_schema=FEATURE_SCHEMA)
        assert chosen['deployed_action_sha256']==action_digest(chosen)
        return chosen
