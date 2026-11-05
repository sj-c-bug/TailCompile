"""013 finite-action scorer with train-fold feature normalization."""
from __future__ import annotations

import torch

try:
    from .edit_context_model import EditContextModel
    from .edit_context_features import FEATURE_SCHEMA
    from .residual_model import ARMS
    from .train_edit_context import _digest
except ImportError:
    from edit_context_model import EditContextModel
    from edit_context_features import FEATURE_SCHEMA
    from residual_model import ARMS
    from train_edit_context import _digest

SCHEMA = 'tailcompile-v7-generalization-head-013-1'
OBJECTIVE = 'same-measured-011-actions-generalization-013-1'
STD_EPS = 1e-8


class GeneralizationModel(EditContextModel):
    """Keep the raw012 readout exact and normalize only when configured."""

    def __init__(self, global_dim, local_dim, *, mode='E', hidden=32,
                 normalization='raw', normalizer_train_ids=(),
                 source_checkpoint_sha256=None, backbone_sha256=None,
                 cache_sha256=None, feature_schema=FEATURE_SCHEMA):
        if normalization not in ('raw', 'train_std'):
            raise ValueError('013 normalization must be raw or train_std')
        super().__init__(global_dim, local_dim, mode=mode, hidden=hidden,
                         source_checkpoint_sha256=source_checkpoint_sha256,
                         backbone_sha256=backbone_sha256, cache_sha256=cache_sha256,
                         feature_schema=feature_schema)
        self.normalization = normalization
        self.config.update(normalization=normalization,
                           normalizer_train_ids=sorted(set(normalizer_train_ids)))
        if normalization == 'train_std':
            width = global_dim + local_dim
            for arm in ARMS:
                self.register_buffer('mean_'+arm, torch.zeros(width))
                self.register_buffer('scale_'+arm, torch.ones(width))
                self.register_buffer('active_'+arm, torch.zeros(width, dtype=torch.bool))
            self.register_buffer('normalizer_ready', torch.tensor(False))

    @torch.no_grad()
    def fit_normalizer(self, cache, train_ids):
        """Estimate every column independently using only train-fold rows."""
        ids = set(train_ids)
        known = {bank['condition_id'] for bank in cache['banks']}
        if not ids or not ids <= known:
            raise ValueError('013 normalizer requires known nonempty train IDs')
        if cache['global_dim'] != self.global_dim or cache['local_dim'] != self.local_dim:
            raise ValueError('013 normalizer/cache feature dimensions differ')
        if self.normalization == 'train_std' and bool(self.normalizer_ready):
            raise ValueError('013 normalizer is already fitted')
        self.config['normalizer_train_ids'] = sorted(ids)
        columns = {}
        for arm in ARMS:
            selected = [b for b in cache['banks'] if b['condition_id'] in ids and b['arm'] == arm]
            if not selected:
                raise ValueError('013 normalizer has no train rows for arm '+arm)
            if self.normalization == 'raw':
                columns[arm] = dict(rows=sum(len(b['candidate_ids']) for b in selected),
                                    active_columns=None, constant_columns=None)
                continue
            rows = []
            for bank in selected:
                rows.append(torch.cat((bank['global_features'], bank['local_features']), dim=1).detach().cpu().double())
            values = torch.cat(rows, dim=0)
            if not torch.isfinite(values).all():
                raise ValueError('013 normalizer features must be finite')
            mean = values.mean(dim=0)
            std = values.std(dim=0, unbiased=False)
            active = std > STD_EPS
            scale = torch.where(active, std, torch.ones_like(std))
            getattr(self, 'mean_'+arm).copy_(mean.float())
            getattr(self, 'scale_'+arm).copy_(scale.float())
            getattr(self, 'active_'+arm).copy_(active)
            columns[arm] = dict(rows=len(values), active_columns=int(active.sum()),
                                constant_columns=int((~active).sum()))
        if self.normalization == 'train_std':
            self.normalizer_ready.fill_(True)
        return self.normalization_metadata(columns=columns)

    def normalization_metadata(self, *, columns=None):
        state = {name: value for name, value in self.state_dict().items()
                 if not name.startswith('heads.')}
        result = dict(kind=self.normalization, train_ids=self.config['normalizer_train_ids'],
                      source='train-fold feature rows only; per arm and column',
                      uses_labels=False, fit_dtype='float64', buffer_dtype='float32',
                      population_std=True, constant_std_threshold=STD_EPS,
                      train_constant_policy='zero for all inference values')
        result['statistics_sha256'] = _digest(dict(result, state=state))
        if columns is not None:
            result['columns'] = columns
        return result

    def normalized_features(self, global_features, local_features, arm):
        """Apply stored train statistics without adapting to the input batch."""
        if arm not in ARMS:
            raise ValueError('013 arm must be L/Y/J')
        if global_features.ndim != 2 or local_features.ndim != 2 or \
                global_features.shape[1] != self.global_dim or local_features.shape[1] != self.local_dim or \
                len(global_features) != len(local_features) or len(global_features) == 0:
            raise ValueError('013 feature dimensions/coverage mismatch')
        if not torch.isfinite(global_features).all() or not torch.isfinite(local_features).all():
            raise ValueError('013 nonfinite scoring input')
        values = torch.cat((global_features, local_features), dim=1).to(
            device=next(self.parameters()).device, dtype=torch.float32)
        if self.normalization == 'raw':
            return torch.cat((values[:, :self.global_dim], torch.zeros_like(values[:, self.global_dim:])), dim=1) \
                if self.mode == 'G' else values
        if not bool(self.normalizer_ready):
            raise ValueError('013 normalizer has not been fitted')
        mean = getattr(self, 'mean_'+arm)
        scale = getattr(self, 'scale_'+arm)
        active = getattr(self, 'active_'+arm)
        values = torch.where(active, (values-mean)/scale, torch.zeros_like(values))
        if self.mode == 'G':
            values = torch.cat((values[:, :self.global_dim], torch.zeros_like(values[:, self.global_dim:])), dim=1)
        if not torch.isfinite(values).all():
            raise FloatingPointError('013 nonfinite normalized features')
        return values

    def score_features(self, global_features, local_features, arm):
        # Preserve the original operation order for the raw32, L2=0 control.
        if self.normalization == 'raw':
            return super().score_features(global_features, local_features, arm)
        values = self.normalized_features(global_features, local_features, arm)
        scores = self.heads[arm](values).flatten()
        return scores-scores[0]

    @torch.no_grad()
    def select_action(self, data, arm, source_checkpoint, *, enabled=True):
        chosen = super().select_action(data, arm, source_checkpoint, enabled=enabled)
        chosen['initializer'] = 'generalization_'+self.mode+'_'+arm
        chosen['selection'].update(normalization=self.normalization,
                                   normalizer_train_ids=self.config['normalizer_train_ids'])
        return chosen
