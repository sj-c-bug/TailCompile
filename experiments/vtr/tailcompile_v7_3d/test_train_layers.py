"""007 paired training tests, including all-negative/no-information feedback."""
import copy
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from layer_model import generate
from model import capacity_only
from test_model import fixture
from train import state_sha256
from train_layers import (feedback_preferences, train_shared, prepare_arms,
                          train_feedback, load_model)


def record():
    data = fixture()
    baseline = capacity_only(data)
    return dict(condition_id='fixture/H0Z0/w64', family='fixture', split='train', graph=data,
        teacher=dict(teacher_region=baseline['assignment'].tolist(), teacher_xy=baseline['xy'].tolist()))


def trial(time, *, status='success', seed=2721, cpd=20.):
    result = dict(seed=seed, status=status, cpd_ns=cpd, wirelength=2000.)
    if status == 'success': result['vpr_seconds'] = time
    return result


class LayerTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_runtime_preferences_keep_high_cpd_and_distinct_negative_results(self):
        action = dict(layer=[0])
        entries = [dict(condition_id='c', action=action, candidate_id='a', trials=[trial(12.)]),
                   dict(condition_id='c', action=action, candidate_id='b', trials=[trial(15.)])]
        pairs, audit = feedback_preferences(entries)
        self.assertEqual(len(pairs['c']), 1)
        self.assertAlmostEqual(pairs['c'][0]['weight'], __import__('math').log(15/12))
        self.assertTrue(all(a['eligible'] for a in audit))
        self.assertEqual(pairs['c'][0]['kind'], 'raw_log_backend_runtime')

    def test_failure_not_fabricated_as_fast_runtime(self):
        entries = [dict(condition_id='c', action={'layer':[0]}, trials=[trial(None, status='timeout')]),
                   dict(condition_id='c', action={'layer':[1]}, trials=[trial(30.)])]
        pairs, _ = feedback_preferences(entries)
        self.assertEqual(pairs['c'][0]['better']['layer'], [1])
        self.assertEqual(pairs['c'][0]['kind'], 'success_count')
        entries[1]['trials'] = [trial(None, status='unroutable')]
        self.assertEqual(feedback_preferences(entries)[0]['c'], [])

    def test_paired_training_freezes_complete_xy_and_updates_layer_parameters(self):
        # Boundary mock isolates trainer behavior; actual risk is tested separately.
        def risk(data, prob, xy):
            return prob[:, 0].mean(), {'active_net_count': 1}
        fake = types.ModuleType('vertical_risk'); fake.risk_loss = risk
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_layer_') as directory:
            root = Path(directory); records = [record()]
            shared = train_shared(records, root/'shared', epochs=1, hidden=16)
            initial_xy_hash = state_sha256(load_model(shared['checkpoint']).shared_xy)
            with patch.dict('sys.modules', {'vertical_risk': fake}):
                prepared = prepare_arms(shared, records, root/'prepare', epochs=1)
            self.assertEqual(prepared['arms']['L']['initial_state_sha256'], prepared['arms']['T']['initial_state_sha256'])
            self.assertEqual(prepared['arms']['T']['shared_xy_sha256'], initial_xy_hash)
            model = load_model(prepared['arms']['L']['checkpoint'])
            a = generate(model, records[0]['graph'])
            other = dict(layer=(1-a['layer']).tolist(), assignment=((a['assignment']+2)%4).tolist(), xy=a['xy'].tolist())
            feedback = [dict(condition_id=records[0]['condition_id'], action=dict(layer=a['layer'].tolist()), trials=[trial(10.)]),
                        dict(condition_id=records[0]['condition_id'], action=other, trials=[trial(15.)])]
            with patch.dict('sys.modules', {'vertical_risk': fake}):
                result = train_feedback(records, prepared, root/'feedback', feedback, epochs=2)
            self.assertEqual(result['feedback_pairs'], 1)
            self.assertEqual(len(result['arms']['L']['checkpoints']), 2)
            self.assertNotEqual(result['arms']['L']['initial_state_sha256'], result['arms']['L']['final_state_sha256'])
            self.assertEqual(result['arms']['L']['shared_xy_sha256'], initial_xy_hash)
            self.assertEqual(result['arms']['T']['shared_xy_sha256'], initial_xy_hash)

    def test_no_preferences_skips_feedback_without_claiming_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_layer_') as directory:
            root = Path(directory); records = [record()]
            shared = train_shared(records, root/'shared', epochs=1, hidden=16)
            arm = dict(checkpoint=shared['checkpoint'], shared_xy_sha256=shared['shared_xy_sha256'])
            prepared = dict(arms={'L':arm, 'T':arm}, shared_xy_sha256=arm['shared_xy_sha256'])
            result = train_feedback(records, prepared, root/'feedback', [], epochs=2)
            self.assertFalse(result['arms']['L']['actual_generator_trained'])
            self.assertEqual(result['arms']['L']['checkpoint'], shared['checkpoint'])

    def test_validation_records_cannot_enter_gradients(self):
        r = record(); r['split'] = 'validation'
        with self.assertRaises(ValueError): train_shared([r], 'unused')

    def test_checkpoint_rejects_changed_layer_weights_even_with_unchanged_xy(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_layer_') as directory:
            root = Path(directory)
            shared = train_shared([record()], root/'shared', epochs=1, hidden=16)
            payload = torch.load(shared['checkpoint'], weights_only=False)
            payload['state_dict']['layer_query.weight'][0, 0] += 1.
            torch.save(payload, root/'changed.pt')
            with self.assertRaisesRegex(ValueError, 'generator checkpoint hash'):
                load_model(root/'changed.pt')

    def test_train_teacher_normalization_is_shared_and_persisted(self):
        from test_vertical_risk import summary
        from vertical_risk import hard_risk
        r = record()
        r['graph']['vertical_risk_device'] = summary(capacities=(2.,))
        r['graph']['risk_nets'] = [dict(id=0, pins=[0,1], drivers=[0], sinks=[1], ignored=False)]
        layer = r['graph']['region_layer'][torch.tensor(r['teacher']['teacher_region'])]
        expected = hard_risk(r['graph'], layer, torch.tensor(r['teacher']['teacher_xy']))['cvar95']
        self.assertGreater(expected, 0.)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_layer_') as directory:
            root = Path(directory)
            shared = train_shared([r], root/'shared', epochs=1, hidden=16)
            self.assertAlmostEqual(shared['tail_normalization']['scale'], expected)
            self.assertFalse(shared['tail_normalization']['validation_or_test_used'])
            prepared = prepare_arms(shared, [r], root/'prepared', epochs=1)
            for arm in ('L', 'T'):
                loaded = load_model(prepared['arms'][arm]['checkpoint'])
                self.assertAlmostEqual(loaded.vertical_tail_scale, expected)


if __name__ == '__main__':
    unittest.main()
