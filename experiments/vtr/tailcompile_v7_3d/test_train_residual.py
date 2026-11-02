"""011 uses only exact deployed candidate QoR and trains only the small heads."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from aligned_model import action_digest
from train import state_sha256
from train_residual import fit, from_source, load_model, pair_delta, _trial_map
from test_residual_model import source_case


class ResidualTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def folder(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_residual_training_')

    def entries(self, data, source):
        model = from_source(source); entries = []; base = None
        for arm in ('L', 'Y', 'J'):
            base, candidates = model.candidate_bank(data, arm, base_action=base)
            for index, action in enumerate(candidates):
                if index == 0 and arm != 'L': continue
                entries.append(dict(condition_id='fixture/H0Z0/w64', split='train',
                    arm='D' if index == 0 else arm, candidate_id=action['candidate_id'], action=action,
                    trials=[dict(seed=s, status='success', wirelength=100.-index*10., cpd_ns=10.-index)
                            for s in (11401, 11402)]))
        return entries

    def test_real_preference_training_changes_only_heads_and_roundtrips(self):
        with self.folder() as tmp:
            folder=Path(tmp); data, source=source_case(folder)
            records=[dict(condition_id='fixture/H0Z0/w64', split='train', graph=data)]
            initial=from_source(source)
            report=fit(records, source, folder/'train', self.entries(data,source), epochs=2, snapshots=(1,2))
            self.assertGreater(report['feedback_pairs'], 0)
            self.assertFalse(report['backbone_trained'])
            self.assertTrue(report['backbone_unchanged'])
            self.assertEqual(report['backbone_sha256'], state_sha256(initial.backbone))
            self.assertNotEqual(report['initial_state_sha256'], report['final_state_sha256'])
            self.assertFalse(report['teacher_labels_used'])
            self.assertFalse(report['geometry_proxy_used_as_target'])
            self.assertFalse(report['tail_evaluated'])
            loaded=load_model(report['checkpoint'])
            self.assertEqual(state_sha256(loaded),report['final_state_sha256'])
            self.assertTrue(all(p.grad is None for p in loaded.backbone.parameters()))
            self.assertEqual(len(report['checkpoints']), 2)
            self.assertGreater(report['training_rank']['Y']['pairs'],0)
            self.assertTrue(report['candidate_pool']['offline_best_is_not_deployable_performance'])
            self.assertFalse(report['candidate_pool']['inference_uses_measured_qor'])
            self.assertGreater(report['candidate_pool']['conditions_with_qor_improvement'],0)
            self.assertEqual(sum(report['candidate_pool']['effective_pairs_per_arm'].values()),report['feedback_pairs'])

    def test_labels_average_seeds_before_log_and_failure_has_priority(self):
        def trial(w, c=1., status='success'): return dict(status=status, wirelength=w, cpd_ns=c)
        a={1:trial(1.,1.),2:trial(9.,9.)}; b={1:trial(4.,4.),2:trial(4.,4.)}
        delta,kind=pair_delta(a,b)
        self.assertLess(delta,0.)  # Ratio of arithmetic means is 5/4, not geometric 3/4.
        self.assertEqual(kind,'qor_log_seed_mean_ratio')
        b[2]=trial(None,None,status='unroutable')
        self.assertEqual(pair_delta(a,b),(1.,'success_count'))
        self.assertIsNone(pair_delta({1:trial(1.)},{2:trial(1.)}))

    def test_changed_action_or_unsupported_measured_bytes_are_rejected(self):
        with self.folder() as tmp:
            folder=Path(tmp); data,source=source_case(folder)
            records=[dict(condition_id='fixture/H0Z0/w64',split='train',graph=data)]
            entries=self.entries(data,source)
            entry=next(e for e in entries if e['arm']=='Y')
            entry['action']['xy'][0,0] += .001
            with self.assertRaisesRegex(ValueError,'action digest mismatch'):
                fit(records,source,folder/'bad',entries,epochs=1,snapshots=(1,))
            entry['action']['deployed_action_sha256']=action_digest(entry['action'])
            with self.assertRaisesRegex(ValueError,'outside regenerated'):
                fit(records,source,folder/'bad2',entries,epochs=1,snapshots=(1,))

    def test_no_signal_or_validation_data_stops_instead_of_claiming_training(self):
        with self.folder() as tmp:
            folder=Path(tmp); data,source=source_case(folder)
            records=[dict(condition_id='fixture/H0Z0/w64',split='train',graph=data)]
            entries=self.entries(data,source)
            entries=[e for e in entries if e['arm']=='D']
            with self.assertRaisesRegex(ValueError,'no effective'):
                fit(records,source,folder/'empty',entries,epochs=1,snapshots=(1,))
            self.assertFalse((folder/'empty/epoch_0001.pt').exists())
            self.assertTrue((folder/'empty/zero_signal_report.json').exists())
            entries[0]['split']='validation'
            with self.assertRaisesRegex(ValueError,'train-only'):
                fit(records,source,folder/'leak',entries,epochs=1,snapshots=(1,))
            with self.assertRaisesRegex(ValueError,'bounded'):
                fit(records,source,folder/'long',entries,epochs=6,snapshots=(6,))

    def test_incomplete_trials_or_infrastructure_failure_are_not_feedback(self):
        with self.folder() as tmp:
            folder=Path(tmp); data,source=source_case(folder)
            records=[dict(condition_id='fixture/H0Z0/w64',split='train',graph=data)]
            entries=self.entries(data,source)
            entry=next(e for e in entries if e['arm']=='Y')
            entry['trials'].pop()
            with self.assertRaisesRegex(ValueError,'same complete measured seed set'):
                fit(records,source,folder/'incomplete',entries,epochs=1,snapshots=(1,))
            entries=self.entries(data,source)
            entries[0]['trials'][0]['status']='infrastructure_error'
            with self.assertRaisesRegex(ValueError,'infrastructure/unknown'):
                fit(records,source,folder/'infrastructure',entries,epochs=1,snapshots=(1,))

    def test_only_confirmed_resource_infeasibility_can_be_a_packing_failure_label(self):
        entry=dict(trials=[dict(seed=11401,status='packing_or_placement_failure',wirelength=None,cpd_ns=None)])
        with self.assertRaisesRegex(ValueError,'unconfirmed packing'):
            _trial_map(entry)
        entry['trials'][0]['confirmed_candidate_infeasibility']=True
        self.assertEqual(_trial_map(entry)[11401]['status'],'packing_or_placement_failure')
        for status in ('timeout','resource_budget_exceeded'):
            entry['trials'][0]['status']=status
            with self.assertRaisesRegex(ValueError,'infrastructure/unknown'):
                _trial_map(entry)


if __name__=='__main__': unittest.main()
