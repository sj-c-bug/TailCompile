"""Frozen checkpoint, whole-molecule lowering and preloaded timing contracts."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch

import generate as generation
from bridge import read, write
from tailcompile_v68.model import HierarchicalInitializer, SCHEMA


def toy_graph():
    torch.manual_seed(610)
    n, r = 8, 4
    return dict(cluster_x=torch.randn(n, 8), logic_src=torch.arange(n - 1),
        logic_dst=torch.arange(1, n), logic_weight=torch.ones(n - 1),
        region_x=torch.randn(r, 13), region_src=torch.tensor([0, 1, 2]),
        region_dst=torch.tensor([1, 2, 3]), region_weight=torch.ones(3),
        parents=[torch.zeros(r, dtype=torch.long)], pool_x=torch.randn(2, 8),
        pool_ids=torch.tensor([0, 0, 1, 1]), pool_regions=torch.arange(r),
        demand=torch.nn.functional.one_hot(torch.arange(n) % 2, 5).float() * .25,
        capacity=torch.ones(r, 5),
        region_bounds=torch.tensor([[0., .5, 0., .5], [.5, 1., 0., .5],
                                    [0., .5, .5, 1.], [.5, 1., .5, 1.]]),
        movable=torch.tensor([False] + [True] * (n - 1)),
        fixed_region=torch.tensor([0] + [-1] * (n - 1)),
        fixed_xy=torch.tensor([[.125, .25]] + [[-1., -1.]] * (n - 1)),
        channel_width=100.)


class GenerationTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        generation.clear_model_cache()
        self.tmp = tempfile.TemporaryDirectory(prefix='.test-generation-',
                                               dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(generation.clear_model_cache)
        self.root = Path(self.tmp.name)
        data = toy_graph()
        torch.save(data, self.root / 'graph.pt')
        write(self.root / 'logic.json', dict(clusters=[dict(atoms=['fixed'])] +
              [dict(atoms=[f'm{i}_a', f'm{i}_b']) for i in range(1, len(data['movable']))]))
        write(self.root / 'device.json', dict(grid_width=24, grid_height=20,
                                            regions=[dict(id=i) for i in range(4)]))
        write(self.root / 'supply.json', dict(schema='test-only'))
        model = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        self.checkpoint = self.root / 'model.pt'
        torch.save(dict(schema=SCHEMA, model_config=model.config,
                        model_state=model.state_dict(), training_families=['seen'],
                        validation_families=['validation']), self.checkpoint)
        self.context = dict(design='toy', width=100, family='seen', split='train',
                            **{k: str(self.root / (k + ('.pt' if k == 'graph' else '.json')))
                               for k in ('graph', 'logic', 'device', 'supply')})
        self.options = dict(decoder='balanced', soft_density_passes=2,
                            repair_passes=2, repair_topk=4)

    def run_generation(self, name='out', **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return generation.generate(self.context, self.checkpoint, self.root / name,
                                       self.options, **kwargs)

    def test_frozen_seen_family_allowed_and_model_does_not_change(self):
        before = self.checkpoint.read_bytes()
        first = self.run_generation()
        self.assertEqual(first['checkpoint_exposure']['category'], 'training_family')
        self.assertFalse(first['model_cache_hit'])
        state = {k: v.clone() for k, v in generation._MODEL_CACHE['model'].state_dict().items()}
        second = self.run_generation('second')
        self.assertTrue(second['model_cache_hit'])
        self.assertFalse(second['graph_embedding_cache_hit'])
        self.assertEqual(first['ap_init_sha256'], second['ap_init_sha256'])
        self.assertEqual(self.checkpoint.read_bytes(), before)
        for key, value in state.items():
            self.assertTrue(torch.equal(value, generation._MODEL_CACHE['model'].state_dict()[key]))

    def test_lowering_preserves_molecules_and_fixed_coordinates(self):
        saved = self.run_generation()
        target = read(saved['target'])
        self.assertEqual(target['teacher_xy'][0], [.125, .25])
        self.assertEqual(target['teacher_region'][0], 0)
        rows = {}
        for line in Path(saved['ap_init']).read_text().splitlines():
            if line.startswith('#'):
                continue
            atom, x, y, layer, subtile = line.split()
            rows[atom] = (float(x), float(y), int(layer), int(subtile))
        self.assertNotIn('fixed', rows)
        self.assertEqual(len(rows), 14)
        for i in range(1, 8):
            self.assertEqual(rows[f'm{i}_a'], rows[f'm{i}_b'])
            x, y, layer, subtile = rows[f'm{i}_a']
            self.assertTrue(0 <= x < 24 and 0 <= y < 20)
            self.assertEqual((layer, subtile), (0, 0))
        self.assertFalse(saved['hard_regional_capacity_enforced'])
        self.assertGreaterEqual(saved['compute_generation_seconds'], saved['inference_seconds'])

    def test_reuse_checks_artifacts_and_timed_generation_is_fresh(self):
        first = self.run_generation()
        with patch.object(generation.BalancedHierarchicalInitializer, 'predict',
                          side_effect=AssertionError('cached generation should not infer')):
            self.assertEqual(self.run_generation(), first)
        with self.assertRaisesRegex(ValueError, 'fresh measured directory'):
            self.run_generation(reuse=False)
        timed = self.run_generation('timed', reuse=False)
        self.assertEqual(first['ap_init_sha256'], timed['ap_init_sha256'])
        Path(first['ap_init']).write_text('# corrupted\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Changed generation artifact'):
            self.run_generation()

    def test_loads_and_gate_excluded_but_both_gnns_decode_and_io_charged(self):
        # Deterministic clock, independent of host load: loading costs 222,
        # barrier costs 50, encoders/decode/lowering cost exactly 17 units.
        clock = [0.]
        events = []
        old_load, old_read, old_lower = generation.torch.load, generation.read, generation._lower
        cls = generation.BalancedHierarchicalInitializer
        old_device, old_logic, old_predict = cls.encode_device, cls.encode_logic, cls.predict

        def load(*args, **kwargs):
            result = old_load(*args, **kwargs)
            clock[0] += 100.; events.append('load')
            return result

        def read_metadata(*args, **kwargs):
            result = old_read(*args, **kwargs)
            clock[0] += 11.; events.append('metadata')
            return result

        def device(model, data):
            self.assertEqual(events[-1], 'gate')
            result = old_device(model, data); clock[0] += 2.
            return result

        def logic(model, data):
            result = old_logic(model, data); clock[0] += 3.
            return result

        def predict(model, data, **kwargs):
            result = old_predict(model, data, **kwargs); clock[0] += 5.
            return result

        def lower(*args):
            result = old_lower(*args); clock[0] += 7.
            return result

        def gate():
            self.assertEqual(events.count('load'), 2)
            self.assertEqual(events.count('metadata'), 2)
            events.append('gate'); clock[0] += 50.

        with patch.object(generation.time, 'perf_counter', lambda: clock[0]), \
             patch.object(generation.torch, 'load', side_effect=load), \
             patch.object(generation, 'read', side_effect=read_metadata), \
             patch.object(cls, 'encode_device', device), patch.object(cls, 'encode_logic', logic), \
             patch.object(cls, 'predict', predict), patch.object(generation, '_lower', side_effect=lower):
            result = self.run_generation(before_inference=gate)
        self.assertEqual(result['excluded_setup_seconds'], 222.)
        self.assertEqual(result['gate_wait_seconds'], 50.)
        self.assertEqual(result['inference_seconds'], 5.)
        self.assertEqual(result['compute_generation_seconds'], 17.)

    def test_partition_and_frozen_options_rejected_before_publication(self):
        logic = read(self.context['logic'])
        logic['clusters'][1]['atoms'][0] = 'fixed'
        write(self.context['logic'], logic)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.run_generation()
        self.assertFalse((self.root / 'out/generation.json').exists())
        with self.assertRaisesRegex(ValueError, 'candidate 0'):
            self.run_generation('bad-candidate', candidate=1)


if __name__ == '__main__':
    unittest.main()
