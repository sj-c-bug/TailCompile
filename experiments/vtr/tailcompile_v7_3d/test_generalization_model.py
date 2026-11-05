"""013 regressions for fold isolation, raw012 parity and deployable transforms."""
from __future__ import annotations

import copy
from pathlib import Path
import tempfile
import unittest

import torch

from aligned_model import action_digest
from edit_context_features import FEATURE_SCHEMA
from edit_context_model import EditContextModel, feature_blocks
from generalization_model import GeneralizationModel, STD_EPS
from generalization_spec import config_by_id
from test_residual_model import source_case
from train import state_sha256
from train_edit_context import CACHE_SCHEMA, cache_digest, fit_cache as fit012
from train_generalization import fit_cache, load_model, evaluate_cache
from train_residual import from_source

HERE = Path(__file__).resolve().parent


def make_cache():
    banks = []
    for family in ('alpha', 'beta'):
        for architecture in ('H0Z0', 'H0Z1', 'H1Z0', 'H1Z1'):
            for arm, factor in (('L', 1.), ('Y', 2.), ('J', 3.)):
                trials = [{seed: dict(seed=seed, status='success', wirelength=wl,
                                    cpd_ns=10.-index*.2) for seed in (11401, 11402)}
                          for index, wl in enumerate((100., 90., 110.))]
                banks.append(dict(condition_id=f'{family}/{architecture}/w64',
                    family=family, architecture=architecture, arm=arm,
                    global_features=torch.tensor([[1., 0., 0., .2], [0., 1., 0., .2], [0., 0., 1., .2]])*factor,
                    local_features=torch.tensor([[0., 0., 0.], [.4, .5, .2], [-.3, .1, .2]])*factor,
                    candidate_ids=['identity', arm.lower()+'_s1', arm.lower()+'_s2'],
                    digests=[str(i)*64 for i in range(3)], trials=trials,
                    generated_candidate_count=3, measured_candidate_count=3))
    cache = dict(schema=CACHE_SCHEMA, feature_schema=FEATURE_SCHEMA, banks=banks,
                 global_dim=4, local_dim=3, source_checkpoint_sha256='a'*64,
                 backbone_sha256='b'*64, source_inputs={}, feedback_sha256='c'*64,
                 synthetic_fixture=True, new_pnr=0)
    cache['cache_sha256'] = cache_digest(cache)
    return cache


def train_ids(cache):
    return sorted({bank['condition_id'] for bank in cache['banks'] if bank['family'] == 'alpha'})


def model(cache, *, mode='E', hidden=8, normalization='train_std'):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(12301)
        value = GeneralizationModel(cache['global_dim'], cache['local_dim'], mode=mode,
                                    hidden=hidden, normalization=normalization,
                                    cache_sha256=cache['cache_sha256'],
                                    source_checkpoint_sha256=cache['source_checkpoint_sha256'],
                                    backbone_sha256=cache['backbone_sha256'])
    value.fit_normalizer(cache, train_ids(cache))
    return value


class GeneralizationModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def folder(self):
        return tempfile.TemporaryDirectory(dir=HERE/'runs', prefix='development_013_model_test_')

    def test_raw_state_initialization_scores_and_small_parameter_budget(self):
        cache = make_cache()
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12301)
            previous = EditContextModel(4, 3, mode='E')
        raw = model(cache, hidden=32, normalization='raw')
        self.assertEqual(state_sha256(previous), state_sha256(raw))
        self.assertEqual(raw.state_dict().keys(), previous.state_dict().keys())
        for value in (previous, raw):
            with torch.no_grad():
                value.heads['J'][2].weight.fill_(.3)
        bank = cache['banks'][-1]
        self.assertTrue(torch.equal(previous.score_features(bank['global_features'], bank['local_features'], 'J'),
                                    raw.score_features(bank['global_features'], bank['local_features'], 'J')))
        small = model(cache, hidden=8, normalization='raw')
        self.assertLess(sum(p.numel() for p in small.parameters()), sum(p.numel() for p in raw.parameters())/3)

    def test_train_statistics_are_per_arm_float64_and_constant_columns_stay_zero(self):
        cache = make_cache()
        value = model(cache)
        for arm in ('L', 'Y', 'J'):
            banks = [b for b in cache['banks'] if b['arm'] == arm and b['family'] == 'alpha']
            rows = torch.cat([torch.cat((b['global_features'], b['local_features']), 1).double() for b in banks])
            self.assertTrue(torch.equal(getattr(value, 'mean_'+arm), rows.mean(0).float()))
            active = rows.std(0, unbiased=False) > STD_EPS
            self.assertTrue(torch.equal(getattr(value, 'active_'+arm), active))
            global_features = banks[0]['global_features'].clone()
            global_features[:, 3] = 1e20
            transformed = value.normalized_features(global_features, banks[0]['local_features'], arm)
            self.assertTrue(torch.equal(transformed[:, 3], torch.zeros(3)))
            self.assertTrue(torch.isfinite(transformed).all())
        self.assertEqual(value.normalization_metadata()['train_ids'], train_ids(cache))
        with self.assertRaisesRegex(ValueError, 'already fitted'):
            value.fit_normalizer(cache, train_ids(cache))

    def test_G_E_use_identical_statistics_and_G_masks_local_after_normalization(self):
        cache = make_cache()
        g, e = model(cache, mode='G'), model(cache, mode='E')
        self.assertEqual(state_sha256(g), state_sha256(e))
        self.assertEqual(g.normalization_metadata(), e.normalization_metadata())
        bank = cache['banks'][0]
        x = g.normalized_features(bank['global_features'], bank['local_features'], 'L')
        self.assertFalse(torch.count_nonzero(x[:, 4:]))
        changed = bank['local_features']*1e8
        self.assertTrue(torch.equal(g.normalized_features(bank['global_features'], changed, 'L'), x))
        self.assertFalse(torch.equal(e.normalized_features(bank['global_features'], changed, 'L'),
                                     e.normalized_features(bank['global_features'], bank['local_features'], 'L')))

    def test_heldout_features_and_labels_cannot_change_training_or_normalizer(self):
        cache = make_cache()
        changed = copy.deepcopy(cache)
        for bank in changed['banks']:
            if bank['family'] == 'beta':
                bank['global_features'] *= -1e8
                bank['local_features'] *= 1e8
                for trials in bank['trials']:
                    for row in trials.values():
                        row['wirelength'] *= 1000.
                        row['cpd_ns'] *= 2000.
        changed['cache_sha256'] = cache_digest(changed)
        with self.folder() as folder:
            cfg = config_by_id('E_h8_train_std_l2p01')
            a = fit_cache(cache, train_ids(cache), Path(folder)/'base', config=cfg, epochs=3, snapshots=(3,))
            b = fit_cache(changed, train_ids(cache), Path(folder)/'heldout_changed', config=cfg, epochs=3, snapshots=(3,))
            self.assertEqual(a['initial_state_sha256'], b['initial_state_sha256'])
            self.assertEqual(a['final_state_sha256'], b['final_state_sha256'])
            self.assertEqual(a['normalization'], b['normalization'])
            self.assertEqual(a['calibration'], b['calibration'])
            self.assertEqual(a['history'], b['history'])
            self.assertFalse(a['validation_labels_used_for_optimization'])

    def test_raw_zero_penalty_training_exactly_reproduces_012(self):
        cache = make_cache()
        ids = train_ids(cache)
        with self.folder() as folder:
            for mode in ('G', 'E'):
                previous = fit012(cache, ids, Path(folder)/(mode+'_012'), mode=mode,
                                  epochs=3, snapshots=(3,), seed=12301)
                current = fit_cache(cache, ids, Path(folder)/(mode+'_013'),
                                    config=config_by_id(mode+'_h32_raw_l20'),
                                    epochs=3, snapshots=(3,), seed=12301)
                for key in ('initial_state_sha256', 'final_state_sha256', 'parameter_count',
                            'calibration', 'history', 'training_evaluation'):
                    self.assertEqual(previous[key], current[key], key)

    def test_checkpoint_roundtrip_preserves_scores_statistics_and_fold(self):
        cache = make_cache()
        cfg = config_by_id('E_h8_train_std_l2p01')
        with self.folder() as folder:
            report = fit_cache(cache, train_ids(cache), folder, config=cfg, epochs=3, snapshots=(1, 3))
            restored = load_model(report['checkpoint'])
            self.assertEqual(restored.training_metadata['factor_config'], cfg)
            self.assertEqual(restored.training_metadata['normalization']['statistics_sha256'],
                             report['normalization']['statistics_sha256'])
            self.assertEqual(evaluate_cache(restored, cache, train_ids(cache)), report['training_evaluation'])
            self.assertEqual(report['optimizer']['weight_decay'], .01)
            payload = torch.load(report['checkpoint'], weights_only=False)
            payload['config']['normalizer_train_ids'] = ['beta/H0Z0/w64']
            broken = Path(folder)/'tampered.pt'
            torch.save(payload, broken)
            with self.assertRaisesRegex(ValueError, 'metadata/state hash'):
                load_model(broken)

    def test_L2_changes_updates_with_same_initial_state_and_normalizer(self):
        cache = make_cache()
        with self.folder() as folder:
            reports = []
            for suffix in ('0', 'p01'):
                reports.append(fit_cache(cache, train_ids(cache), Path(folder)/suffix,
                    config=config_by_id('E_h8_train_std_l2'+suffix), epochs=2, snapshots=(2,)))
            self.assertEqual(reports[0]['initial_state_sha256'], reports[1]['initial_state_sha256'])
            self.assertEqual(reports[0]['normalization'], reports[1]['normalization'])
            self.assertNotEqual(reports[0]['final_state_sha256'], reports[1]['final_state_sha256'])
            a, b = [load_model(report['checkpoint']) for report in reports]
            self.assertFalse(torch.equal(a.heads['L'][0].weight, b.heads['L'][0].weight))

    def test_unfitted_normalizer_and_nonfinite_inputs_are_rejected(self):
        value = GeneralizationModel(4, 3, normalization='train_std')
        x, y = torch.zeros(3, 4), torch.zeros(3, 3)
        with self.assertRaisesRegex(ValueError, 'not been fitted'):
            value.score_features(x, y, 'L')
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            value.score_features(x+float('nan'), y, 'L')

    def test_normalized_deployment_still_selects_original_011_actions(self):
        with self.folder() as folder:
            data, source_path = source_case(Path(folder))
            source = from_source(source_path)
            base, candidates = source.candidate_bank(data, 'J')
            global_features, local_features = feature_blocks(source.backbone, data, base, candidates)
            value = GeneralizationModel(global_features.shape[1], local_features.shape[1],
                       mode='E', hidden=8, normalization='train_std',
                       source_checkpoint_sha256=source.source_checkpoint_sha256,
                       backbone_sha256=state_sha256(source.backbone))
            # The deployment smoke uses pre-route features only, with one bank per arm.
            cache = dict(global_dim=global_features.shape[1], local_dim=local_features.shape[1],
                         banks=[dict(condition_id='train', arm=arm, global_features=global_features,
                                     local_features=local_features, candidate_ids=[a['candidate_id'] for a in candidates])
                                for arm in ('L', 'Y', 'J')])
            value.fit_normalizer(cache, ['train'])
            chosen = value.select_action(data, 'J', source_path)
            self.assertIn(action_digest(chosen), [action_digest(a) for a in candidates])
            self.assertFalse(chosen['selection']['teacher_or_qor_used_at_inference'])
            self.assertEqual(chosen['selection']['normalization'], 'train_std')
            self.assertEqual(action_digest(value.select_action(data, 'J', source_path, enabled=False)), action_digest(base))


if __name__ == '__main__':
    unittest.main()
