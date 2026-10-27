"""Training-state evidence and bounded input-cache tests."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from model import Hierarchical3DInitializer, capacity_only
from test_model import fixture
from train import TrainingInputCache, state_sha256, train


class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_cancelled_training_stops_before_first_optimizer_step(self):
        graph = fixture(n=3)
        action = capacity_only(graph)
        target = dict(teacher_region=action['assignment'].tolist(), teacher_xy=action['xy'].tolist())
        records = [dict(graph=graph,teacher=target,split='train',condition_id='cancel_case',family='fixture')]
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)/'model'
            with patch('train.raise_if_cancelled',side_effect=[None,None,KeyboardInterrupt()]):
                with self.assertRaises(KeyboardInterrupt):
                    train(records,output,epochs=20,hidden=16)
            self.assertFalse((output/'training_report.json').exists())
            self.assertFalse(list(output.glob('epoch_*.pt')))

    def test_many_teacher_records_share_one_graph_and_weight_hashes(self):
        graph = fixture(n=3)
        action = capacity_only(graph)
        target = dict(teacher_region=action['assignment'].tolist(), teacher_xy=action['xy'].tolist())
        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            graph_path, target_path = folder/'shared.pt', folder/'projected.json'
            torch.save(graph, graph_path)
            target_path.write_text(json.dumps(target), encoding='utf-8')
            records = [dict(graph=str(graph_path), teacher=str(target_path), split='train',
                            condition_id=f'condition_{i}', family=f'family_{i//2}') for i in range(8)]
            report = train(records, folder/'model', epochs=2, hidden=16)
            cache = report['training_input_cache']
            self.assertEqual(cache['graph_disk_loads'], 1)
            self.assertEqual(cache['max_cpu_graphs'], 1)
            self.assertEqual(cache['max_device_graphs'], 0)
            self.assertEqual(cache['teacher_projection_calls'], 0)
            self.assertEqual(cache['label_loads'], 1)
            self.assertGreaterEqual(cache['cpu_graph_hits'], 16)
            self.assertTrue(report['weights_changed'])
            self.assertNotEqual(report['initial_state_sha256'], report['final_state_sha256'])
            snapshot = torch.load(report['checkpoints'][-1], map_location='cpu', weights_only=False)
            model = Hierarchical3DInitializer(**snapshot['config'])
            model.load_state_dict(snapshot['state_dict'])
            self.assertEqual(state_sha256(model), report['final_state_sha256'])

    def test_cpu_cache_eviction_and_same_path_reuse(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder)/f'graph{i}.pt' for i in range(3)]
            for i, path in enumerate(paths):
                torch.save(fixture(n=i+2), path)
            cache = TrainingInputCache(cpu_graph_entries=2)
            first = cache.graph(paths[0])
            self.assertIs(first, cache.graph(paths[0]))
            cache.graph(paths[1]); cache.graph(paths[2]); cache.graph(paths[0])
            self.assertEqual(len(cache.cpu_graphs), 2)
            self.assertEqual(cache.audit()['max_cpu_graphs'], 2)
            self.assertEqual(cache.audit()['graph_disk_loads'], 4)
            self.assertEqual(cache.audit()['cpu_graph_evictions'], 2)

    def test_device_lru_evicts_before_transfer_without_requiring_cuda(self):
        # Tests device-cache administration only; the transfer is mocked and
        # this is explicitly not hardware CUDA verification.
        cache = TrainingInputCache(device='cuda:0', cpu_graph_entries=1, device_graph_entries=2)
        payloads = [fixture(n=i+2) for i in range(3)]
        allocations = []
        def transfer(graph, device):
            if torch.device(device).type == 'cuda':
                self.assertLess(len(cache.device_graphs), 2)
                allocations.append(id(graph))
            return graph
        with patch('train.to_device', side_effect=transfer):
            first = cache.graph(payloads[0])
            self.assertIs(first, cache.graph(payloads[0]))
            cache.graph(payloads[1]); cache.graph(payloads[2])
        self.assertEqual(len(allocations), 3)
        self.assertEqual(len(cache.device_graphs), 2)
        self.assertEqual(cache.audit()['max_device_graphs'], 2)
        self.assertEqual(cache.audit()['device_graph_evictions'], 1)
        with self.assertRaises(ValueError):
            TrainingInputCache(device_graph_entries=3)

    def test_changed_graph_source_rejected_during_training(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/'graph.pt'
            torch.save(fixture(n=2), path)
            cache = TrainingInputCache()
            cache.graph(path)
            torch.save(fixture(n=5), path)
            with self.assertRaisesRegex(ValueError, 'changed during'):
                cache.graph(path)

    def test_missing_decode_action_retained_but_not_given_fake_gradient(self):
        graph = fixture(n=3); action = capacity_only(graph)
        target = dict(teacher_region=action['assignment'].tolist(), teacher_xy=action['xy'].tolist())
        record = dict(graph=graph, teacher=target, split='train', condition_id='c1', family='f')
        feedback = [dict(condition_id='c1', candidate_id='failed', action=None,
                         trials=[dict(seed=521, status='decode_failure')],
                         default_trials=[dict(seed=521, status='success', total_seconds=1.,
                                              cpd_ns=1., wirelength=1.)])]
        with tempfile.TemporaryDirectory() as folder:
            report = train([record], folder, hidden=16, epochs=1, feedback=feedback)
            self.assertEqual(report['eligible_feedback_actions'], 0)
            self.assertEqual(report['effective_nonzero_feedback_actions'], 0)
            audit = report['feedback_audit'][0]
            self.assertEqual(audit['reason'], 'no_complete_deployment_action')
            self.assertEqual(audit['observed_outcome_statuses'], ['decode_failure'])
            self.assertFalse(audit['outcome_reward_before_action_mask']['failure_runtime_imputed'])


if __name__ == '__main__':
    unittest.main()
