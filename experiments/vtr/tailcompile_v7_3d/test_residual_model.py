"""011 learns finite feasible choices while D weights remain immutable."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from aligned_model import action_digest
from residual_actions import ACTION_FEATURE_DIM, FEATURE_NAMES, validate_residual_action
from residual_model import ResidualSelector
from structured_model import StructuredInitializer
from train import state_sha256
from train_structured import save_model as save_d
from train_residual import from_source, load_model, save_model
from test_model import fixture


def source_case(folder):
    data = fixture()
    torch.manual_seed(11301)
    backbone = StructuredInitializer(3, 5, hidden=16, chunk_size=2)
    with torch.no_grad(): backbone.layer_query.weight.zero_()
    path = folder/'D.pt'; save_d(backbone, path)
    return data, path


class ResidualModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def folder(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs', prefix='test_residual_model_')

    def test_initial_heads_select_exact_identity_and_never_change_backbone(self):
        with self.folder() as tmp:
            graph, source = source_case(Path(tmp)); model = from_source(source)
            frozen = state_sha256(model.backbone)
            for arm in ('L', 'Y', 'J'):
                base, bank = model.candidate_bank(graph, arm)
                selected = model.select_from_bank(graph, arm, base, bank)
                self.assertEqual(selected['candidate_id'], 'identity')
                self.assertEqual(action_digest(selected), action_digest(base))
                validate_residual_action(graph, selected, base_action=base, arm=arm)
            model.train()
            self.assertFalse(model.backbone.training)
            self.assertTrue(all(not p.requires_grad for p in model.backbone.parameters()))
            self.assertEqual(state_sha256(model.backbone), frozen)

    def test_learned_scores_choose_an_actual_bounded_candidate_and_gate_can_disable(self):
        with self.folder() as tmp:
            graph, source = source_case(Path(tmp)); model = from_source(source)
            base, bank = model.candidate_bank(graph, 'Y')
            self.assertGreater(len(bank), 1)
            head = model.heads['Y']
            with torch.no_grad():
                head[0].weight.zero_(); head[0].bias.zero_()
                head[0].weight[0, -ACTION_FEATURE_DIM+FEATURE_NAMES.index('strength')] = 1.
                head[-1].weight.zero_(); head[-1].weight[0, 0] = 1.
            selected = model.select_from_bank(graph, 'Y', base, bank)
            self.assertNotEqual(selected['candidate_id'], 'identity')
            self.assertIn(action_digest(selected), [action_digest(a) for a in bank])
            self.assertEqual(model.select_action(graph, 'Y', enabled=False)['candidate_id'], 'identity')
            scores = model.score_candidates(graph, bank, base, arm='Y')
            scores.sum().backward()
            self.assertTrue(any(p.grad is not None for p in model.heads['Y'].parameters()))
            self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))

    def test_checkpoint_roundtrip_preserves_bank_scores_and_rejects_tampering(self):
        with self.folder() as tmp:
            folder = Path(tmp); graph, source = source_case(folder); model = from_source(source)
            path = folder/'selector.pt'; save_model(model, path, epoch=2)
            loaded = load_model(path)
            self.assertEqual(state_sha256(model), state_sha256(loaded))
            self.assertEqual(action_digest(model.select_action(graph, 'J')), action_digest(loaded.select_action(graph, 'J')))
            self.assertEqual(loaded.backbone.layer_query.weight.device.type, 'cpu')
            payload = torch.load(path, weights_only=False)
            payload['state_dict']['heads.Y.2.weight'][0, 0] += 1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, 'hash mismatch'): load_model(path)
            with self.assertRaisesRegex(ValueError, 'identity changed'): from_source(source, expected_sha256='0'*64)


if __name__ == '__main__': unittest.main()
