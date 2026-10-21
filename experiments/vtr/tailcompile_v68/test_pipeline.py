"""Synthetic integration tests; no VTR installation or benchmark labels needed.

These check plumbing/guards, not evidence of FPGA QoR improvement.  Temporary
graphs have distinct family identities and no teacher coordinates in features.
"""
import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

from argparse import Namespace
import copy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import torch

from common import execute, read, sha, write
import experiment
from infer import Generator
from model import SCHEMA
from test_model import graph
import train as training


class PipelineTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tailcompile-v68-pipeline-')
        self.root = Path(self.temporary.name)
        self.stdout = io.StringIO()
        self.redirect = redirect_stdout(self.stdout)
        self.redirect.__enter__()
        torch.set_num_threads(1)

    def tearDown(self):
        self.redirect.__exit__(None, None, None)
        self.temporary.cleanup()

    def fixture(self):
        examples = []
        context = None
        for design, family, split in [('tiny_train', 'family_A', 'train'),
                                      ('tiny_val', 'family_B', 'validation')]:
            folder = self.root / design
            folder.mkdir(parents=True)
            data = graph(2, 2)
            if split == 'validation':
                data['cluster_x'] = data['cluster_x'] + .123
            data['movable'][0] = False
            data['fixed_xy'] = torch.zeros(12, 2)
            data['fixed_xy'][0] = torch.tensor([.05, .1])
            data['fixed_region'] = torch.full((12,), -1)
            data['fixed_region'][0] = 0
            graph_path = folder / 'graph.pt'
            torch.save(data, graph_path)
            target = torch.arange(12) % 4
            bounds = data['region_bounds'][target]
            xy = (bounds[:, [0, 2]] + bounds[:, [1, 3]]) / 2
            xy[0] = data['fixed_xy'][0]
            target = dict(teacher_region=target.tolist(), teacher_xy=xy.tolist(),
                          movable=data['movable'].tolist())
            examples.append(dict(design=design, family=family, split=split, width=100,
                                 graph=str(graph_path), graph_sha256=sha(graph_path),
                                 target=target, weight=1.))
            clusters = [dict(id=i, ap_block_id=i + 100, resource='clb',
                             atoms=[f'{design}_atom_{i}', f'{design}_atom_{i}_b'])
                        for i in range(12)]
            write(folder / 'logic.json', dict(clusters=clusters))
            write(folder / 'device.json', dict(grid_width=10, grid_height=8,
                                              regions=[dict(id=i) for i in range(4)]))
            write(folder / 'supply.json', dict(channel_width=100))
            if split == 'train':
                context = dict(design=design, family=family, split=split, width=100,
                    graph=str(graph_path), logic=str(folder / 'logic.json'),
                    device=str(folder / 'device.json'), supply=str(folder / 'supply.json'))
        dataset = dict(schema='tailcompile-v68-dataset-1', examples=examples)
        path = self.root / 'dataset.json'
        write(path, dataset)
        return dataset, context, Namespace(dataset=path, out=self.root / 'pretrain',
            initialize=None, feedback=None, device='cpu', epochs=2, hidden=16,
            chunk_size=4, tile_size=4, threads=1, graph_cache=2, lr=.005, seed=7,
            allow_no_validation=False)

    def test_train_load_generate_native_float_coordinates_and_reuse(self):
        dataset, context, args = self.fixture()
        training.train(args)
        report = read(args.out / 'training_report.json')
        self.assertTrue(report['complete'])
        self.assertTrue(report['weights_changed'])
        self.assertEqual(report['epochs_completed'], 2)
        self.assertEqual(report['training_families'], ['family_A'])
        self.assertEqual(report['validation_families'], ['family_B'])
        self.assertEqual(report['final_test_examples'], 0)
        saved = torch.load(args.out / 'generator.pt', weights_only=True)
        self.assertEqual(saved['schema'], SCHEMA)
        self.assertEqual(saved['role'], 'native_gp_initializer_generator')
        checkpoint_sha = sha(args.out / 'generator.pt')
        training.train(args)
        self.assertEqual(sha(args.out / 'generator.pt'), checkpoint_sha)

        generator = Generator(args.out / 'generator.pt')
        folder = self.root / 'generated'
        generated = generator.generate(context, folder)
        target = read(generated['target'])
        rows = [line.split() for line in Path(generated['ap_init']).read_text().splitlines()
                if line and not line.startswith('#')]
        flat = {r[0]: r for r in [line.split() for line in Path(generated['fplace']).read_text().splitlines()
                                if line and not line.startswith('#')]}
        self.assertEqual(len(rows), 22)  # 11 movable AP blocks, two atoms each.
        self.assertFalse(any(row[0].endswith('_atom_0') for row in rows))
        by_name = {row[0]: row for row in rows}
        for index in range(1, 12):
            atom = f'tiny_train_atom_{index}'
            row = by_name[atom]
            x, y = float(row[1]), float(row[2])
            self.assertAlmostEqual(x, target['teacher_xy'][index][0] * 10, places=10)
            self.assertAlmostEqual(y, target['teacher_xy'][index][1] * 8, places=10)
            self.assertGreaterEqual(x, 0)
            self.assertLess(x, 10)
            self.assertGreaterEqual(y, 0)
            self.assertLess(y, 8)
            self.assertEqual(int(flat[atom][1]), int(x))
            self.assertEqual(int(flat[atom][2]), int(y))
            self.assertEqual(row[1:3], by_name[atom + '_b'][1:3])
        repeated = generator.generate(context, folder)
        self.assertEqual(generated, repeated)
        self.assertFalse(generated['source_placement_used'])
        self.assertTrue(generated['native_gp_required'])
        self.assertGreater(generated['warm_inference_lowering_seconds'], 0)
        self.assertNotIn('teacher_xy', torch.load(context['graph'], weights_only=True))

    def test_training_resumes_saved_epoch_and_optimizer(self):
        _, _, args = self.fixture()
        original = training.emit

        def interrupt(stage, **values):
            if stage == 'epoch' and values['epoch'] == 2:
                raise RuntimeError('simulated GPU instance interruption')
            return original(stage, **values)

        with patch.object(training, 'emit', side_effect=interrupt):
            with self.assertRaisesRegex(RuntimeError, 'simulated GPU'):
                training.train(args)
        latest = torch.load(args.out / 'latest.pt', weights_only=True)
        self.assertEqual(latest['epoch'], 1)
        self.assertTrue(latest['optimizer']['state'])
        training.train(args)
        report = read(args.out / 'training_report.json')
        self.assertEqual([e['epoch'] for e in report['history']], [1, 2])
        self.assertTrue(report['complete'])
        self.assertIn('training_resumed', self.stdout.getvalue())

    def test_resource_conflict_is_projected_once_per_condition_and_audited(self):
        dataset, context, args = self.fixture()
        original_targets = copy.deepcopy([r['target'] for r in dataset['examples']])
        graph_hashes = {}
        for row in dataset['examples']:
            data = torch.load(row['graph'], weights_only=True)
            data['capacity'][:, 3] = 0
            data['capacity'][2, 3] = 1
            # Movable DSP node originally taught to occupy leaf 1, which has no DSP.
            data['demand'][1] = torch.tensor([0., 0., 0., 1., 0.])
            torch.save(data, row['graph'])
            row['graph_sha256'] = sha(row['graph'])
            graph_hashes[row['graph']] = row['graph_sha256']
        write(args.dataset, dataset)
        dataset_hash = sha(args.dataset)
        with patch.object(training, 'project_teacher', wraps=training.project_teacher) as project:
            training.train(args)
        self.assertEqual(project.call_count, 2)  # Not once per epoch/loss call.
        audit = read(args.out / 'supervision_audit.json')
        self.assertTrue(audit['complete'])
        self.assertEqual(audit['audited_conditions'], 2)
        self.assertEqual(audit['summary']['conflicted_nodes'], 2)
        self.assertEqual(audit['summary']['reassigned_nodes'], 2)
        self.assertTrue(all(c['distance_units'] == 'physical_tiles' for c in audit['conditions']))
        self.assertTrue(all(c['raw_target_sha256'] != c['projected_target_sha256']
                            for c in audit['conditions']))
        report = read(args.out / 'training_report.json')
        saved = torch.load(args.out / 'generator.pt', weights_only=True)
        self.assertEqual(report['supervision_schema'], audit['schema'])
        self.assertEqual(saved['supervision_audit_sha256'], audit['conditions_sha256'])
        self.assertEqual(sha(args.dataset), dataset_hash)
        self.assertEqual([r['target'] for r in read(args.dataset)['examples']], original_targets)
        for path, expected in graph_hashes.items():
            self.assertEqual(sha(path), expected)
        audit['conditions'][0]['reassigned_nodes'] += 1
        write(args.out / 'supervision_audit.json', audit)
        with self.assertRaisesRegex(ValueError, 'audit changed'):
            training.train(args)

    def test_old_training_signature_is_not_silently_reused_after_projection_fix(self):
        _, _, args = self.fixture()
        legacy = dict(schema=SCHEMA, code={'model.py': 'old-hard-masked-supervision'})
        write(args.out / 'training_signature.json', legacy)
        with self.assertRaisesRegex(ValueError, 'Frozen inputs changed'):
            training.train(args)
        self.assertEqual(read(args.out / 'training_signature.json'), legacy)
        self.assertFalse((args.out / 'latest.pt').exists())

    def test_dataset_and_evaluation_reject_final_test_family_leakage(self):
        dataset, context, args = self.fixture()
        leaked = copy.deepcopy(dataset)
        leaked['examples'][0]['split'] = 'final_test'
        with self.assertRaisesRegex(ValueError, 'Final-test'):
            training.validate_dataset(leaked)
        leaked = copy.deepcopy(dataset)
        leaked['examples'][1]['family'] = leaked['examples'][0]['family']
        with self.assertRaisesRegex(ValueError, 'Family leakage'):
            training.validate_dataset(leaked)
        training.train(args)
        evaluation = self.root / 'eval'
        write(evaluation / 'protocol.json', dict(identity={}, evaluation_seeds=[341]))
        write(evaluation / 'contexts.json', dict(contexts=[dict(context, split='final_test')]))
        evaluate_args = Namespace(out=evaluation, identity={},
            checkpoint=args.out / 'generator.pt', device='cpu', jobs=1)
        with self.assertRaisesRegex(ValueError, 'Final-test family'):
            experiment.evaluate(evaluate_args)
        self.assertFalse((evaluation / 'evaluation' / 'runs').exists())

    def test_feedback_quality_guards_and_higher_width_rescue_reference(self):
        good = dict(status='success', cpd=10., wirelength=100.)
        fail = dict(status='unroutable', cpd=None, wirelength=None)
        timeout = dict(status='timeout', cpd=None, wirelength=None)
        quality = lambda baseline, candidate, reference=None: experiment.feedback_quality(
            baseline, candidate, timing_guard=.03, wire_guard=.03, reference=reference)
        self.assertFalse(quality([good], [timeout])['acceptable'])
        self.assertFalse(quality([good], [fail])['acceptable'])
        self.assertFalse(quality([good], [dict(good, cpd=10.4)])['acceptable'])
        self.assertTrue(quality([good], [dict(good, cpd=9.8)])['acceptable'])
        self.assertFalse(quality([fail], [good])['acceptable'])
        rescue = quality([fail], [good], reference=good)
        self.assertTrue(rescue['acceptable'])
        self.assertIn('higher_width', rescue['qor_comparison'])
        # Having one joint success must not hide unguarded CPD of a new rescue.
        self.assertFalse(quality([good, fail], [good, dict(good, cpd=12.)],
                                 reference=good)['acceptable'])

    def test_feedback_target_hash_and_split_are_verified(self):
        dataset, _, _ = self.fixture()
        target = self.root / 'feedback_target.json'
        write(target, dataset['examples'][0]['target'])
        row = dict(design='tiny_train', family='family_A', split='train', width=100,
                   target=str(target), target_sha256=sha(target))
        feedback = dict(selected=[row])
        rows = training.rows_for_training(dataset, feedback)
        self.assertEqual(rows[0]['target_kind'], 'real_vpr_selected_initializer')
        self.assertEqual(rows[0]['weight'], 2.)
        bad = copy.deepcopy(feedback)
        bad['selected'][0]['split'] = 'final_test'
        with self.assertRaisesRegex(ValueError, 'train-family'):
            training.rows_for_training(dataset, bad)
        write(target, dict(dataset['examples'][0]['target'], modified=True))
        with self.assertRaisesRegex(ValueError, 'target changed'):
            training.rows_for_training(dataset, feedback)

    def test_real_feedback_finetunes_same_deployed_generator(self):
        dataset, context, args = self.fixture()
        training.train(args)
        pretrained = args.out / 'generator.pt'
        generated = Generator(pretrained).generate(context, self.root / 'train_candidate',
            candidate=1, seed=58, score_noise_std=.25, coordinate_noise_std=.05)
        feedback_path = self.root / 'feedback.json'
        selected = dict(design='tiny_train', family='family_A', split='train', width=100,
            target=generated['target'], target_sha256=sha(generated['target']), candidate=1)
        feedback = dict(selected=[selected], generator_checkpoint_sha256=sha(pretrained))
        write(feedback_path, feedback)
        fine = Namespace(**vars(args))
        fine.out = self.root / 'finetune'
        fine.feedback = feedback_path
        fine.initialize = pretrained
        training.train(fine)
        report = read(fine.out / 'training_report.json')
        self.assertTrue(report['complete'])
        self.assertEqual(report['real_feedback_targets'], 1)
        self.assertTrue(report['weights_changed'])
        self.assertEqual(report['initial_weights_sha256'],
                         training.weights_sha(torch.load(pretrained, weights_only=True)['model_state']))
        feedback['generator_checkpoint_sha256'] = '0' * 64
        write(feedback_path, feedback)
        bad = Namespace(**vars(fine))
        bad.out = self.root / 'bad_finetune'
        with self.assertRaisesRegex(ValueError, 'checkpoint'):
            training.train(bad)

    def test_timeouts_are_censored_not_false_width_failures(self):
        observed = experiment.width_observation([80, 84, 88], [
            dict(width=80, status='unroutable'), dict(width=84, status='timeout'),
            dict(width=88, status='success')])
        self.assertEqual(observed['minimum_success_in_grid'], 88)
        self.assertEqual(observed['greatest_known_failure_below'], 80)
        self.assertEqual(observed['unknown_widths'], [84])
        self.assertFalse(observed['exact_wmin_proven'])
        self.assertFalse(observed['left_censored'])

    def test_process_success_reused_and_timeout_retained_without_retry(self):
        folder = self.root / 'process'
        command = [sys.executable, '-c', 'print("VPR succeeded")']
        first = execute(command, folder, dict(test='success'), timeout=5)
        second = execute(command, folder, dict(test='success'), timeout=5,
                         retry_failed=True, retry_timeouts=True)
        self.assertEqual(first['status'], 'success')
        self.assertEqual(first, second)
        slow = [sys.executable, '-c', 'import time;time.sleep(10)']
        first = execute(slow, self.root / 'timeout', dict(test='timeout'), timeout=.05)
        second = execute(slow, self.root / 'timeout', dict(test='timeout'), timeout=.05)
        self.assertEqual(first['status'], 'timeout')
        self.assertEqual(first['returncode'], 124)
        self.assertEqual(first, second)


if __name__ == '__main__':
    unittest.main()
