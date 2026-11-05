"""013 controller regressions for frozen factors, folds and safe resume."""
from copy import deepcopy
import contextlib
import io
import itertools
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import generalization_campaign as campaign
from generalization_spec import CONFIGS, PRIMARY_CONFIG, TRAIN_SEEDS
from io_utils import read, sha, write


def panel():
    return dict(banks=[dict(condition_id=f'{f}/{a}/w64', family=f, architecture=a)
                      for f in ('alpha', 'beta') for a in ('H0Z0', 'H0Z1', 'H1Z0', 'H1Z1')],
                cache_sha256='a'*64, source_checkpoint_sha256='b'*64)


def fit_fixture(split, config, seed):
    ids = sorted(split['train_ids'])
    rows = [dict(condition_id=cid, family=cid.split('/')[0], arm=arm,
                 selected_index=0, selected_candidate_id='identity', selected_digest='d'*64,
                 selected_WL=100., selected_CPD=10., D_WL=100., D_CPD=10.,
                 selected_success_count=2, D_success_count=2, selected_is_offline_best=True,
                 q_regret=0.) for cid, arm in itertools.product(split['evaluation_ids'], ('L','Y','J'))]
    evaluation = dict(rows=rows, ranking={a: dict(correct=3*len(split['evaluation_ids']),
                      pairs=3*len(split['evaluation_ids']), ties=0) for a in ('L','Y','J')})
    training = dict(mode=config['mode'], seed=seed, epochs=80, device='cpu', factor_config=deepcopy(config),
                    cache_sha256='a'*64, source_checkpoint_sha256='b'*64, train_ids=ids,
                    calibration=dict(train_ids=ids), normalization=dict(kind=config['normalization'], train_ids=ids),
                    test_checkpoint_epoch_fixed=80,
                    optimizer=dict(name='Adam', learning_rate=.001, weight_decay=config['weight_decay'],
                                   penalty='coupled L2 on head parameters only'), checkpoint_sha256={})
    return dict(split=split, config=config, seed=seed, training=training, evaluation=evaluation,
                seed_evaluation={str(s): deepcopy(evaluation) for s in (11401,11402)}, worker_threads=1)


class GeneralizationCampaignTests(unittest.TestCase):
    def folder(self):
        return tempfile.TemporaryDirectory(dir=campaign.HERE/'runs', prefix='development_013_controller_test_')

    def test_complete_factorial_and_matched_endpoints_are_frozen(self):
        self.assertEqual(len(CONFIGS), 10)
        self.assertEqual(len({c['id'] for c in CONFIGS}), 10)
        local = [c for c in CONFIGS if c['mode'] == 'E']
        self.assertEqual({(c['hidden'], c['normalization'], c['weight_decay']) for c in local},
                         set(itertools.product((32,8), ('raw','train_std'), (0.,.01))))
        self.assertEqual(PRIMARY_CONFIG, 'E_h8_train_std_l2p01')
        self.assertEqual(campaign.execution_options('cpu')['jobs'], 16)
        self.assertEqual(campaign.execution_options('cuda')['jobs'], 1)
        with self.assertRaises(ValueError):
            campaign.execution_options('cuda', 2)

    def test_factor_contrasts_keep_population_and_do_not_select_winner(self):
        fits = [fit_fixture(s,c,seed) for s,c,seed in itertools.product(campaign.splits(panel()['banks']),CONFIGS,TRAIN_SEEDS)]
        report = campaign.summarize_fits(fits)
        self.assertEqual(report['offline_fits'], 140)
        self.assertEqual(len(report['summaries']), 60)
        self.assertEqual(len(report['factorial_and_local_contrasts']), 84)
        self.assertEqual(report['primary_configuration'], PRIMARY_CONFIG)
        self.assertFalse(report['heldout_used_for_model_selection'])
        for arms in report['factorial_and_local_contrasts'].values():
            for value in arms.values():
                self.assertEqual(value['metrics']['WL']['ratio'], 1.)
                self.assertEqual(value['conditions'], 8)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            campaign.summarize_fits(fits+[fits[0]])

    def test_changed_actual_regularization_or_normalizer_fold_is_rejected(self):
        cache = panel()
        split = campaign.splits(cache['banks'])[1]
        config = CONFIGS[0]
        fit = fit_fixture(split, config, TRAIN_SEEDS[0])
        campaign.verify_fit(fit, split, config, TRAIN_SEEDS[0], cache, 'cpu')
        for key in ('factor_config', 'normalization', 'optimizer'):
            bad = deepcopy(fit)
            if key == 'factor_config':
                bad['training'][key]['hidden'] = 123
            elif key == 'normalization':
                bad['training'][key]['train_ids'] = bad['training'][key]['train_ids']+split['evaluation_ids']
            else:
                bad['training'][key]['weight_decay'] = .5
            with self.assertRaisesRegex(ValueError, 'factors|normalization'):
                campaign.verify_fit(bad, split, config, TRAIN_SEEDS[0], cache, 'cpu')

    def test_resume_keeps_completed_fits_when_available_memory_changes(self):
        with self.folder() as tmp:
            out = Path(tmp)
            cache = panel()
            write(out/'prepared_report.json', dict(cache=str(out/'cache.pt'), cache_file_sha256='c'*64))
            for split, config, seed in itertools.product(campaign.splits(cache['banks']), CONFIGS, TRAIN_SEEDS):
                folder = out/'models'/split['id']/config['id']/f's{seed}'
                manifest = folder/'worker/manifest.json'
                write(manifest, dict(status='complete', returncode=0))
                write(folder/'fit_report.json', dict(fit_fixture(split,config,seed), worker_manifest=str(manifest)))
            runner = campaign.Campaign(SimpleNamespace(out=out,device='cpu',jobs=2,fit_timeout=120))
            runner.cache = lambda: cache
            budgets = [dict(effective=2,physical_cores=[0,1],available_mib=10000),
                       dict(effective=1,physical_cores=[0],available_mib=2500)]
            with patch.object(campaign, 'resource_budget', side_effect=budgets), \
                    patch.object(campaign, 'run_trial', side_effect=AssertionError('completed fit rerun')), \
                    contextlib.redirect_stdout(io.StringIO()):
                runner.train()
                original = sha(out/'trained_report.json')
                runner.train()
            self.assertEqual(original, sha(out/'trained_report.json'))
            self.assertEqual(read(out/'train_progress_report.json')['completed'], 140)
            self.assertEqual(len(list((out/'execution/train/parallel_runs').glob('attempt_*'))), 2)

    def test_changed_worker_cache_fails_before_creating_output(self):
        with self.folder() as tmp:
            path = Path(tmp)/'cache.pt'
            path.write_bytes(b'changed')
            out = Path(tmp)/'new-fit'
            with self.assertRaisesRegex(ValueError, 'read-only012 namespace'):
                campaign.fit_worker(dict(cache=str(path)), out)
            self.assertFalse(out.exists())

    def test_absent_original_baselines_are_unmeasured(self):
        with self.folder() as tmp:
            split = campaign.splits(panel()['banks'])[0]
            result = campaign.baseline_reproduction([fit_fixture(split, CONFIGS[0], TRAIN_SEEDS[0])], Path(tmp))
            self.assertEqual(result['checked_same_device'], 0)
            self.assertIsNone(result['exact_reproduction_all_available_same_device'])

    @unittest.skipUnless(os.name == 'posix', 'Linux isolated training workers')
    def test_real_workers_consume_frozen_fold_and_save_complete_evidence(self):
        from test_generalization_model import make_cache
        from train_edit_context import save_cache
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs', prefix='development_012_generalization_test_') as origin, self.folder() as tmp:
            origin, out = Path(origin), Path(tmp)
            cache = make_cache()
            path = origin/'feature_cache.pt'
            save_cache(cache, path)
            split = campaign.splits(cache['banks'])[1]
            config = next(c for c in CONFIGS if c['id'] == PRIMARY_CONFIG)
            requests = []
            for seed in TRAIN_SEEDS:
                folder = out/f's{seed}'
                request = dict(cache=str(path), cache_file_sha256=sha(path), cache_sha256=cache['cache_sha256'],
                    implementation=campaign.implementation(), split=split, config=config, seed=seed,
                    device='cpu', fit_timeout=120)
                write(folder/'request.json', request)
                requests.append(folder)
            def work(folder):
                result = campaign.run_trial(folder/'worker', [sys.executable, campaign.HERE/'generalization_campaign.py',
                    'fit-worker', '--out', folder, '--request', folder/'request.json'],
                    timeout=120, marker='TAILCOMPILE_GENERALIZATION_FIT_COMPLETE', prune=False)
                self.assertEqual(result['status'], 'complete', result)
                fit = read(folder/'fit_report.json')
                campaign.verify_fit(fit, split, config, fit['seed'], cache, 'cpu')
                return fit
            fits = campaign.parallel_fit_map(requests, jobs=2, run_task=work)
            self.assertEqual(len({f['worker_pid'] for f in fits}), 2)
            for fit in fits:
                self.assertEqual(fit['training']['normalization']['train_ids'], sorted(split['train_ids']))
                self.assertEqual(len(fit['evaluation']['rows']), 12)
                self.assertEqual(fit['worker_threads'], 1)


if __name__ == '__main__':
    unittest.main()
