"""Frozen 013 controls for capacity, L2 and train-fold normalization."""
from copy import deepcopy
import itertools

EPOCHS = 80
SNAPSHOTS = (5, 20, 80)
TRAIN_SEEDS = (12301, 12302)
LEARNING_RATE = .001


def configuration_id(mode, hidden, normalization, weight_decay):
    penalty = '0' if weight_decay == 0 else 'p01'
    return f'{mode}_h{hidden}_{normalization}_l2{penalty}'


CONFIGS = []
for hidden, normalization, weight_decay in itertools.product((32, 8), ('raw', 'train_std'), (0., .01)):
    CONFIGS.append(dict(id=configuration_id('E', hidden, normalization, weight_decay),
                        mode='E', hidden=hidden, normalization=normalization,
                        weight_decay=weight_decay))
for hidden, normalization, weight_decay in ((32, 'raw', 0.), (8, 'train_std', .01)):
    CONFIGS.append(dict(id=configuration_id('G', hidden, normalization, weight_decay),
                        mode='G', hidden=hidden, normalization=normalization,
                        weight_decay=weight_decay))

PRIMARY_CONFIG = configuration_id('E', 8, 'train_std', .01)


def config_by_id(name):
    matches = [c for c in CONFIGS if c['id'] == name]
    if len(matches) != 1:
        raise ValueError('013 unknown frozen configuration: '+str(name))
    return deepcopy(matches[0])


def plan():
    return dict(schema='generalization-campaign-013-1', configurations=deepcopy(CONFIGS),
                factors=dict(hidden=[32, 8], weight_decay=[0., .01], normalization=['raw', 'train_std']),
                factorial_mode='E', matched_global_controls=['original', 'all_three_changes'],
                optimizer='Adam', learning_rate=LEARNING_RATE, epochs=EPOCHS,
                snapshots=list(SNAPSHOTS), training_seeds=list(TRAIN_SEEDS),
                split_groups=dict(full_fit_diagnostic=1, scorer_leave_family_out=2,
                                  scorer_leave_architecture_out=4), offline_fits=140,
                primary_configuration=PRIMARY_CONFIG, primary_training_seed=TRAIN_SEEDS[0],
                primary_selected_before_results=True, holdout_used_for_model_selection=False,
                normalization_source='train-fold feature rows only, per arm and column',
                train_constant_columns='center then zero, including held-out values',
                normalization_uses_targets=False, candidate_bank='unchanged measured011',
                source_feature_cache='read-only signed012', source_backbone_frozen=True,
                new_pnr=0, Wmin_measured=False, independent_family_test=False,
                scorer_label_holdout_only=True, frozen_backbone_family_exposure=True,
                future_pnr_requires_frozen_configuration_and_new_seeds=True,
                metric_rule='common successful historical seed means, then family/architecture equal geometric ratios',
                runtime_role='offline cost only', threads_per_worker=1)
