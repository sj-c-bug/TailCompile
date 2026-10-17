import os
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import copy
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from tailcompile_v61 import (SCHEMA, HierarchicalMatcher, hierarchy, preference, swap_proposal)
from prepare_tailcompile_v61 import tile_device
from train_tailcompile_v61 import group_loss, run
from train_tailcompile_v6 import ranks, evaluate
from run_tailcompile_v61 import reusable_generation_signature


def graph(columns=4, rows=4):
    n = columns * rows
    device = dict(regions=[dict(column=x, row=y) for y in range(rows) for x in range(columns)])
    return dict(cluster_x=torch.randn(n, 8), logic_src=torch.arange(n), logic_dst=torch.roll(torch.arange(n), 1),
        logic_weight=torch.ones(n), region_x=torch.randn(n, 13), region_src=torch.arange(n),
        region_dst=torch.roll(torch.arange(n), 1), region_weight=torch.ones(n),
        capacity=torch.ones(n, 4) * 10, resource=torch.arange(n) % 3, assignment=torch.arange(n),
        parents=hierarchy(device), pool_x=torch.ones(n, 8), pool_rho=torch.rand(n),
        pool_ids=torch.arange(n), pool_regions=torch.arange(n),
        spatial_region_x=torch.rand(n, 7), spatial_global_x=torch.rand(9))


class TestV61(unittest.TestCase):
    def test_generation_signature_only_migrates_known_legacy_hash(self):
        expected = dict(bundle_sha256='b', budget=64, implementation='new')
        legacy = dict(expected, implementation=
            '88ef9b376311b61b5f72d4b8b4c03d0e45da08a2c217a0929852688889ddccdb')
        self.assertTrue(reusable_generation_signature(legacy, expected))
        self.assertFalse(reusable_generation_signature(dict(legacy, budget=63), expected))
        self.assertFalse(reusable_generation_signature(dict(legacy, implementation='unknown'), expected))

    def test_hierarchy_and_matching_influence_final_score(self):
        torch.manual_seed(0)
        model = HierarchicalMatcher(hidden=12)
        data = graph(3, 5)
        data['spatial_region_x'].requires_grad_()
        a = model(data)['score']
        a.backward()
        self.assertGreater(float(data['spatial_region_x'].grad.abs().sum()), 0)
        changed = dict(data, assignment=torch.roll(data['assignment'], 1))
        self.assertGreater(abs(float((model(changed)['score'] - a).detach())), 1e-7)
        self.assertEqual(model(data)['hotspot'].shape, (15,))

    def test_preference_success_and_qor(self):
        good = dict(route=1., qor=[.5, .5, 100., 5.])
        bad = dict(route=.5, qor=[.1, .1, 10., 1.])
        self.assertEqual(preference(good, bad), 1)
        self.assertEqual(preference(good, dict(good)), 0)
        self.assertEqual(preference(good, dict(route=1., qor=[.5, .5, 100., 10.])), 1)
        self.assertEqual(ranks([1., 1. + 1e-7, 2.]), [.5, .5, 2.])

    def test_fixed_tiles_preserve_unique_anchors(self):
        old = dict(grid_width=12, grid_height=8, regions=[dict(tile_coordinates={
            'clb': [[1, 1], [4, 2], [10, 6]], 'memory': [[7, 4]]})])
        d = tile_device(old, 4)
        self.assertEqual(d['region_grid']['count'], 6)
        self.assertEqual(sum(r['tile_counts'].get('clb', 0) for r in d['regions']), 3)
        self.assertEqual(sum(r['tile_counts'].get('memory', 0) for r in d['regions']), 1)

    def test_pair_training_and_checkpoint_report(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        examples = []
        for design in ('a', 'b', 'c'):
            for method, route in (('gnn_base', .25), ('gnn_tail', .75)):
                examples.append(dict(design=design, width=80, method=method, route=route,
                    qor=[.5, .5, 100., 5.], hotspot=[.3] * 4, graph=graph(2, 2)))
        model = HierarchicalMatcher(hidden=12)
        loss = group_loss(model, examples[:2], [(0., 1.)] * 4, torch.device('cpu'))
        loss.backward()
        self.assertGreater(float(model.rank.weight.grad.abs().sum()), 0.)
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            torch.save(dict(schema=SCHEMA, examples=examples, contexts={}), root / 'bundle.pt')
            args = SimpleNamespace(bundle=root / 'bundle.pt', out=root / 'models',
                epochs=1, hidden=12, seed=2027, device='cpu')
            run(args)
            self.assertTrue((root / 'models/report.json').is_file())
            import json
            before = json.loads((root / 'models/holdout_a.json').read_text())['checkpoint_sha256']
            run(args)
            after = json.loads((root / 'models/holdout_a.json').read_text())['checkpoint_sha256']
            self.assertEqual(before, after)

    def test_swaps_preserve_resource_and_io(self):
        from collections import Counter
        logic = dict(clusters=[dict(id=i, resource='io' if i == 4 else 'clb') for i in range(5)])
        evaluator = SimpleNamespace(legal=lambda proposal, fixed: proposal[4] == fixed[4])
        start = [0, 0, 1, 1, 0]
        proposal = swap_proposal(start, logic, evaluator, random.Random(1), start, 1.)
        self.assertEqual(Counter(proposal[:4]), Counter(start[:4]))
        self.assertEqual(proposal[4], start[4])


if __name__ == '__main__':
    unittest.main()
