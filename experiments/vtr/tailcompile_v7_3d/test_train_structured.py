"""010 training uses true feasible actions, never relabelled legacy rewards."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import torch

from aligned_teacher import constraints_digest
from aligned_model import action_digest
from qor_model import QorInitializer, generate as generate009
from train_qor import save_model as save009
from structured_model import generate, ACTION_SCHEMA, DECODER_SCHEMA
from train_structured import from009, save_model, load_model, fit
from train import state_sha256
from test_train_aligned_layers import record


class StructuredTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def case(self, folder):
        torch.manual_seed(10301)
        source = QorInitializer(3,5,hidden=16,chunk_size=2,arm='X')
        with torch.no_grad(): source.layer_query.weight.zero_()
        path = folder/'009.pt'; save009(source, path, epoch=10)
        return record(), path, source

    def collect(self, action, model, data, checkpoint):
        return dict(schema=ACTION_SCHEMA, decoder_kind=DECODER_SCHEMA,
            assignment=action['assignment'].tolist(), layer=action['layer'].tolist(), xy=action['xy'].tolist(),
            deployed_action_sha256=action['deployed_action_sha256'],
            constraints_sha256=constraints_digest(data), shared_xy_sha256=state_sha256(model.shared_xy),
            collection_checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest())

    def test_genuine_new_pairs_train_and_resume_with_ranking_metric(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_structured_training_') as tmp:
            folder = Path(tmp); rec, source, _ = self.case(folder); model = from009(source)
            control = folder/'D.pt'; save_model(model, control)
            actions = [generate(model, rec['graph'], score_noise_std=8., seed=s) for s in (11,12)]
            self.assertNotEqual(actions[0]['deployed_action_sha256'], actions[1]['deployed_action_sha256'])
            feedback = [dict(condition_id=rec['condition_id'], candidate_id=str(i), split='train',
                action=self.collect(a, model, rec['graph'], control),
                trials=[dict(seed=10401, status='success', wirelength=100.+10*i, cpd_ns=10.+i)]) for i,a in enumerate(actions)]
            report = fit([rec], source, folder/'train', feedback, epochs=1, snapshots=(1,))
            self.assertEqual(report['feedback_pairs'], 1)
            self.assertTrue(report['actual_qor_feedback_trained'])
            self.assertEqual(report['training_rank']['pairs'], 1)
            self.assertIsNotNone(report['training_rank']['accuracy'])
            self.assertEqual(report['initial_state_sha256'], state_sha256(model))
            self.assertNotEqual(report['initial_shared_xy_sha256'], report['shared_xy_sha256'])
            self.assertFalse(report['validation_gradients_used'])
            self.assertFalse(report['tail_evaluated'])
            self.assertIn('010 current eligible train feedback pool', report['scales']['source'])
            self.assertEqual(report['scales']['eligible_action_count'], 2)
            self.assertTrue(report['teacher_projection'][0]['hard_leaf_capacity'])
            loaded = load_model(report['checkpoint'])
            self.assertEqual(state_sha256(loaded), report['final_state_sha256'])
            resumed = fit([rec], report['checkpoint'], folder/'resume', feedback, epochs=1, snapshots=(1,))
            self.assertEqual(resumed['initial_state_sha256'], report['final_state_sha256'])

    def test_legacy_overloaded_action_is_filtered_without_projection_of_reward(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_structured_legacy_') as tmp:
            folder = Path(tmp); rec, source, old_model = self.case(folder)
            old = generate009(old_model, rec['graph'])
            self.assertGreater(old['capacity_diagnostics']['overflow_region_count'], 0)
            action = dict(schema='tailcompile-v7-aligned-action-008-1', decoder_kind='aligned-hard-layer-density-008-1',
                assignment=old['assignment'].tolist(), layer=old['layer'].tolist(), xy=old['xy'].tolist(),
                shared_xy_sha256=state_sha256(old_model.shared_xy), constraints_sha256=constraints_digest(rec['graph']))
            # Legacy payload has no embedded deployed_action_sha256. Its
            # recorded generation hash is verified by recomputing the action.
            entry = dict(condition_id=rec['condition_id'], candidate_id='old', action=action,
                generation=dict(deployed_action_sha256=action_digest(action)),
                trials=[dict(seed=1,status='success',wirelength=100.,cpd_ns=10.)])
            report = fit([rec], source, folder/'train', [entry], epochs=1, snapshots=(1,))
            self.assertEqual(report['feedback_pairs'], 0)
            self.assertFalse(report['actual_qor_feedback_trained'])
            self.assertEqual(report['feedback_eligibility'][0]['reason'], 'old_action_not_010_leaf_feasible')
            self.assertFalse(report['feedback_eligibility'][0]['old_qor_attached_to_projection'])

    def test_train_split_and_finite_budget_are_enforced(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_structured_split_') as tmp:
            folder=Path(tmp); rec, source, _ = self.case(folder)
            rec['split']='validation'
            with self.assertRaisesRegex(ValueError, 'train-only'): fit([rec], source, folder/'val', epochs=1, snapshots=(1,))
            rec['split']='train'
            with self.assertRaisesRegex(ValueError, 'bounded'): fit([rec], source, folder/'too_long', epochs=21, snapshots=(21,))
            bad=[dict(condition_id=rec['condition_id'],split='validation',action={})]
            with self.assertRaisesRegex(ValueError, 'outside train'): fit([rec],source,folder/'bad',bad,epochs=1,snapshots=(1,))


if __name__ == '__main__': unittest.main()
