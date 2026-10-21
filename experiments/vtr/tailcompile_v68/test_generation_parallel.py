"""Real spawn-process candidate generation checks; no VPR or benchmark needed.

These tests establish scheduler/reuse equivalence, not FPGA performance gains.
The tiny checkpoint is trained locally; all features and teachers are synthetic.
"""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import torch

import generation
from common import read, sha, write
from infer import Generator
from model import HierarchicalInitializer, SCHEMA
from test_model import graph


class ParallelGenerationTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='v68-generation-parallel-')
        self.root = Path(self.temporary.name)
        self.output = io.StringIO()
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        torch.set_num_threads(1)
        self.contexts = []
        for offset, design in enumerate(('tiny_alpha', 'tiny_beta')):
            folder = self.root / 'inputs' / design
            folder.mkdir(parents=True)
            data = graph(2, 2)
            data['cluster_x'] = data['cluster_x'] + offset * .123
            data['movable'][0] = False
            data['fixed_xy'] = torch.zeros(12, 2)
            data['fixed_xy'][0] = torch.tensor([.05, .1])
            data['fixed_region'] = torch.full((12,), -1)
            data['fixed_region'][0] = 0
            torch.save(data, folder / 'graph.pt')
            write(folder / 'logic.json', dict(clusters=[
                dict(id=i, ap_block_id=i + 100, resource='clb',
                     atoms=[f'{design}_atom_{i}', f'{design}_atom_{i}_b'])
                for i in range(12)]))
            write(folder / 'device.json', dict(grid_width=10, grid_height=8,
                                               regions=[dict(id=i) for i in range(4)]))
            write(folder / 'supply.json', dict(channel_width=100))
            self.contexts.append(dict(design=design, family=f'family_{offset}', split='train',
                width=100, graph=str(folder / 'graph.pt'), logic=str(folder / 'logic.json'),
                device=str(folder / 'device.json'), supply=str(folder / 'supply.json')))
        data = torch.load(self.contexts[0]['graph'], weights_only=True)
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=.01)
        target = torch.arange(12) % 4
        bounds = data['region_bounds'][target]
        xy = (bounds[:, [0, 2]] + bounds[:, [1, 3]]) / 2
        for _ in range(4):
            optimizer.zero_grad()
            model.loss(data, target, xy).backward()
            optimizer.step()
        self.checkpoint = self.root / 'generator.pt'
        torch.save(dict(schema=SCHEMA, model_config=model.config,
            model_state=model.state_dict(), role='native_gp_initializer_generator',
            training_families=['family_0', 'family_1'], validation_families=[], tile_size=4),
            self.checkpoint)

    def tearDown(self):
        self.redirect.__exit__(None, None, None)
        self.temporary.cleanup()

    def candidate_folder(self, root, context, candidate=0, mode='feedback'):
        folder = Path(root) / context['design'] / f'w{context["width"]}'
        return folder / f'c{candidate}' if mode == 'feedback' else folder

    def serial_candidate(self, generator, root, context, candidate, mode='feedback'):
        return generator.generate(context, self.candidate_folder(root, context, candidate, mode),
            candidate=candidate, seed=680000 + candidate if mode == 'feedback' else 0,
            score_noise_std=.25 if mode == 'feedback' and candidate else 0.,
            coordinate_noise_std=.05 if mode == 'feedback' and candidate else 0.)

    def snapshots(self, root):
        return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mtime_ns)
                for p in Path(root).rglob('*') if p.is_file()
                and p.name in ('generation.json', 'init.ap.tsv', 'init.fplace', 'target.json')}

    def assert_complete_progress(self, path, total, reused, new):
        progress = read(path)
        self.assertEqual(progress['schema'], 'tailcompile-v68-generation-progress-1')
        self.assertTrue(progress['complete'])
        self.assertEqual(progress['total_candidates'], total)
        self.assertEqual(progress['completed_candidates'], total)
        self.assertEqual(progress['reused_candidates'], reused)
        self.assertEqual(progress['new_candidates'], new)
        self.assertEqual(progress['pending_candidates'], 0)
        self.assertEqual(progress['failed_contexts'], 0)
        return progress

    def test_actual_two_spawn_workers_match_serial_candidate_bytes(self):
        baseline = self.root / 'serial'
        generator = Generator(self.checkpoint)
        expected = {}
        for context in self.contexts:
            for candidate in range(2):
                expected[context['design'], context['width'], candidate] = self.serial_candidate(
                    generator, baseline, context, candidate)
        parallel_root, progress_path = self.root / 'parallel', self.root / 'parallel_progress.json'
        generated = generation.generate_contexts(self.contexts, self.checkpoint, parallel_root,
            jobs=2, candidates=2, progress_path=progress_path)
        self.assertEqual(set(generated), set(expected))
        for key, old in expected.items():
            new = generated[key]
            for field in ('ap_init', 'fplace', 'target'):
                self.assertEqual(Path(new[field]).read_bytes(), Path(old[field]).read_bytes(),
                                 (key, field))
            self.assertEqual(new['signature'], old['signature'])
        progress = self.assert_complete_progress(progress_path, 4, 0, 4)
        self.assertEqual(progress['jobs'], 2)

    def test_partial_resume_preserves_completed_candidate_and_then_reuses_everything(self):
        root, progress_path = self.root / 'resume', self.root / 'resume_progress.json'
        self.serial_candidate(Generator(self.checkpoint), root, self.contexts[0], 0)
        before = self.snapshots(root)
        result = generation.generate_contexts(self.contexts, self.checkpoint, root,
            jobs=2, candidates=2, progress_path=progress_path)
        self.assertEqual(len(result), 4)
        after = self.snapshots(root)
        for path, original in before.items():
            self.assertEqual(after[path], original, path)
        self.assert_complete_progress(progress_path, 4, 1, 3)
        before = self.snapshots(root)
        # An all-cached resume should not create a process pool or load a model.
        with patch.object(generation.cf, 'ProcessPoolExecutor', side_effect=AssertionError(
                'completed candidates must not start workers')):
            reused = generation.generate_contexts(self.contexts, self.checkpoint, root,
                jobs=2, candidates=2, progress_path=progress_path)
        self.assertEqual(len(reused), 4)
        self.assertEqual(self.snapshots(root), before)
        self.assert_complete_progress(progress_path, 4, 4, 0)

    def test_changed_checkpoint_is_rejected_without_overwriting_completed_candidate(self):
        root = self.root / 'changed_checkpoint'
        self.serial_candidate(Generator(self.checkpoint), root, self.contexts[0], 0)
        before = self.snapshots(root)
        saved = torch.load(self.checkpoint, weights_only=True)
        first = next(iter(saved['model_state']))
        saved['model_state'][first] = saved['model_state'][first] + .01
        torch.save(saved, self.checkpoint)
        with self.assertRaises((ValueError, RuntimeError)):
            generation.generate_contexts(self.contexts[:1], self.checkpoint, root,
                jobs=2, candidates=1)
        self.assertEqual(self.snapshots(root), before)

    def test_changed_graph_input_is_rejected_without_overwriting_completed_candidate(self):
        root = self.root / 'changed_input'
        self.serial_candidate(Generator(self.checkpoint), root, self.contexts[0], 0)
        before = self.snapshots(root)
        write(self.contexts[0]['supply'], dict(channel_width=100, changed=True))
        with self.assertRaises((ValueError, RuntimeError)):
            generation.generate_contexts(self.contexts[:1], self.checkpoint, root,
                jobs=2, candidates=1)
        self.assertEqual(self.snapshots(root), before)

    def test_corrupted_completed_output_is_not_silently_regenerated(self):
        root = self.root / 'corrupt_output'
        result = self.serial_candidate(Generator(self.checkpoint), root, self.contexts[0], 0)
        Path(result['ap_init']).write_text('corrupted completed native prior\n')
        before = self.snapshots(root)
        with self.assertRaises((ValueError, RuntimeError)):
            generation.generate_contexts(self.contexts[:1], self.checkpoint, root,
                jobs=2, candidates=1)
        self.assertEqual(self.snapshots(root), before)

    def test_recorded_seed_and_noise_cannot_change_when_reusing_candidate(self):
        for name, value in [('seed', 999), ('score_noise_std', .75), ('coordinate_noise_std', .5)]:
            root = self.root / f'changed_{name}'
            self.serial_candidate(Generator(self.checkpoint), root, self.contexts[0], 0)
            path = self.candidate_folder(root, self.contexts[0]) / 'generation.json'
            modified = read(path)
            modified['signature'][name] = value
            write(path, modified)
            before = self.snapshots(root)
            with self.subTest(field=name), self.assertRaises((ValueError, RuntimeError)):
                generation.generate_contexts(self.contexts[:1], self.checkpoint, root,
                    jobs=2, candidates=1)
            self.assertEqual(self.snapshots(root), before)

    def test_evaluation_generation_matches_serial_zero_noise_policy(self):
        root, baseline = self.root / 'evaluation', self.root / 'evaluation_serial'
        generator = Generator(self.checkpoint)
        expected = {context['design']: self.serial_candidate(generator, baseline, context, 0,
                                                              mode='evaluation')
                    for context in self.contexts}
        result = generation.generate_contexts(self.contexts, self.checkpoint, root,
            jobs=2, candidates=1, mode='evaluation')
        for context in self.contexts:
            actual = result[context['design'], context['width'], 0]
            for field in ('ap_init', 'fplace', 'target'):
                self.assertEqual(Path(actual[field]).read_bytes(),
                                 Path(expected[context['design']][field]).read_bytes())

    def test_parallel_cuda_and_nonpositive_job_budgets_are_rejected_before_launch(self):
        with self.assertRaises((ValueError, RuntimeError)):
            generation.generate_contexts(self.contexts, self.checkpoint, self.root / 'cuda',
                jobs=2, candidates=2, device='cuda')
        with self.assertRaises((ValueError, RuntimeError)):
            generation.generate_contexts(self.contexts, self.checkpoint, self.root / 'zero',
                jobs=0, candidates=2)

    def test_duplicate_context_destinations_are_rejected_before_any_worker(self):
        contexts = [self.contexts[0], copy.deepcopy(self.contexts[0])]
        with self.assertRaises((ValueError, RuntimeError)):
            generation.generate_contexts(contexts, self.checkpoint, self.root / 'duplicate',
                jobs=2, candidates=2)

    def test_one_context_candidates_can_use_two_distinct_worker_processes(self):
        # Give each worker meaningful decoder work; otherwise tiny tasks can
        # all finish while the second spawned process is still importing torch.
        n = 20000
        context = dict(self.contexts[0], design='single_large_context')
        folder = self.root / 'large_inputs'
        folder.mkdir()
        data = graph(2, 2)
        data.update(cluster_x=torch.randn(n, 8), logic_src=torch.arange(n - 1),
            logic_dst=torch.arange(1, n), logic_weight=torch.ones(n - 1),
            logic_criticality=torch.linspace(0, 1, n - 1),
            demand=torch.nn.functional.one_hot(torch.arange(n) % 2, 5).float() * .25,
            capacity=torch.full((4, 5), 10000.), movable=torch.ones(n, dtype=torch.bool))
        torch.save(data, folder / 'graph.pt')
        write(folder / 'logic.json', dict(clusters=[
            dict(id=i, ap_block_id=i + 100, resource='clb', atoms=[f'large_atom_{i}'])
            for i in range(n)]))
        for name in ('graph', 'logic'):
            context[name] = str(folder / ('graph.pt' if name == 'graph' else 'logic.json'))
        saved = torch.load(self.checkpoint, weights_only=True)
        saved['model_config']['chunk_size'] = 512
        checkpoint = folder / 'generator.pt'
        torch.save(saved, checkpoint)
        progress_path = self.root / 'single_context_progress.json'
        result = generation.generate_contexts([context], checkpoint,
            self.root / 'single_context', jobs=2, candidates=8, progress_path=progress_path)
        self.assertEqual(len(result), 8)
        progress = self.assert_complete_progress(progress_path, 8, 0, 8)
        self.assertEqual(progress['task_granularity'], 'candidate')
        self.assertEqual(len(progress['worker_pids']), 2)
        self.assertEqual(progress['workers'], 2)


if __name__ == '__main__':
    unittest.main()
