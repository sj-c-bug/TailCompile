"""Real-checkpoint generation/lowering/provenance integration tests, no VPR needed."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch

import generate as generation
from bridge import read, write
from fast_model import CHECKPOINT_SCHEMA, HierarchicalInitializer
from test_fast_model import graph


class GenerationTest(unittest.TestCase):
    def setUp(self):
        # Keep test artifacts under the checkout, not a platform scratch/home
        # path; Windows sandbox temporary directories can deny torch's writer.
        self.temporary = tempfile.TemporaryDirectory(prefix='.test-generation-',
            dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.options = dict(soft_density_passes=2, density_price_weight=1., density_step=1.)

    def inputs(self, *, overflow=False, family='unseen'):
        data = graph(1, 1) if overflow else graph(2, 2)
        data['movable'][0] = False
        data['fixed_xy'] = torch.full((len(data['cluster_x']), 2), -1.)
        data['fixed_xy'][0] = torch.tensor([.125, .25])
        data['fixed_region'] = torch.full((len(data['cluster_x']),), -1)
        data['fixed_region'][0] = 0
        if overflow:
            data['capacity'][:] = .3  # Each .25 molecule fits, aggregate does not.
        graph_path = self.root / 'graph.pt'
        torch.save(data, graph_path)
        clusters = [dict(atoms=['fixed_atom'])]
        for i in range(1, len(data['cluster_x'])):
            clusters.append(dict(atoms=[f'm{i}_a', f'm{i}_b']))
        logic_path, device_path, supply_path = (self.root / f'{name}.json'
                                                for name in ('logic', 'device', 'supply'))
        write(logic_path, dict(clusters=clusters))
        write(device_path, dict(grid_width=24, grid_height=20,
                               regions=[dict(id=i) for i in range(len(data['capacity']))]))
        write(supply_path, dict(schema='toy-test-supply'))
        original = HierarchicalInitializer(8, 13, hidden=16, chunk_size=3)
        checkpoint = self.root / 'generator.pt'
        torch.save(dict(schema=CHECKPOINT_SCHEMA, model_config=original.config,
                        model_state=original.state_dict(), training_families=['training'],
                        validation_families=['validation']), checkpoint)
        context = dict(design='toy', width=100, split='final_test', family=family,
                       graph=str(graph_path), logic=str(logic_path),
                       device=str(device_path), supply=str(supply_path))
        return context, checkpoint, data

    def generate(self, context, checkpoint, folder='output', **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return generation.generate(context, checkpoint, self.root / folder,
                                       self.options, **kwargs)

    def test_old_checkpoint_generates_json_safe_complete_prior_and_costs(self):
        context, checkpoint, _ = self.inputs()
        result = self.generate(context, checkpoint)
        saved = read(self.root / 'output/generation.json')
        self.assertEqual(saved, result)
        json.dumps(result, allow_nan=False)  # Regression for Tensor JSON failures.
        self.assertTrue(result['native_gp_required'])
        self.assertFalse(result['hard_regional_capacity_enforced'])
        self.assertEqual(result['leaf_count'], 4)
        self.assertGreater(result['hierarchy_level_count'], 0)
        self.assertGreater(result['timing']['logic_forward'], 0.)
        self.assertGreater(result['timing']['matching_decode_coordinate'], 0.)
        self.assertGreater(result['phase_seconds']['matching'], 0.)
        self.assertAlmostEqual(result['generation_seconds_excluding_rrg_encoding'],
                               result['measured_generation_seconds'] - result['timing']['rrg_encoding'])
        for key in ('ap_init', 'fplace', 'target'):
            self.assertTrue(Path(result[key]).is_file())
            self.assertEqual(generation.sha(result[key]), result[key + '_sha256'])

    def test_lowerer_keeps_molecules_together_and_omits_fixed_atoms(self):
        context, checkpoint, data = self.inputs()
        result = self.generate(context, checkpoint, candidate=1)
        target = read(result['target'])
        self.assertEqual(target['teacher_region'][0], 0)
        self.assertEqual(target['teacher_xy'][0], data['fixed_xy'][0].tolist())
        coordinates = {}
        for line in Path(result['ap_init']).read_text().splitlines():
            if line.startswith('#'):
                continue
            atom, x, y, layer, subtile = line.split()
            self.assertNotIn(atom, coordinates)
            coordinates[atom] = (float(x), float(y), int(layer), int(subtile))
        self.assertNotIn('fixed_atom', coordinates)
        self.assertEqual(len(coordinates), 22)
        for i in range(1, 12):
            self.assertEqual(coordinates[f'm{i}_a'], coordinates[f'm{i}_b'])
            x, y, layer, subtile = coordinates[f'm{i}_a']
            self.assertTrue(0 <= x < 24)
            self.assertTrue(0 <= y < 20)
            self.assertEqual((layer, subtile), (0, 0))

    def test_real_deployment_sentinels_generate_valid_prior_and_preserve_graph(self):
        context, checkpoint, data = self.inputs()
        before = Path(context['graph']).read_bytes()
        self.assertTrue(torch.equal(data['fixed_xy'][1:], torch.full((11, 2), -1.)))
        result = self.generate(context, checkpoint, candidate=1)
        target = read(result['target'])
        self.assertEqual(target['teacher_xy'][0], data['fixed_xy'][0].tolist())
        for xy in target['teacher_xy'][1:]:
            self.assertTrue(all(0 <= coordinate <= 1 for coordinate in xy))
        self.assertEqual(Path(context['graph']).read_bytes(), before)

    def test_overflow_is_saved_instead_of_disguised_as_legal_capacity(self):
        context, checkpoint, _ = self.inputs(overflow=True)
        result = self.generate(context, checkpoint)
        quality = result['capacity_diagnostics']
        self.assertEqual(read(self.root / 'output/generation.json')['capacity_diagnostics'], quality)
        self.assertGreater(quality['total_capacity_overflow'], 0.)
        self.assertGreater(quality['overflow_region_count'], 0)
        self.assertFalse(quality['hard_regional_capacity_enforced'])
        self.assertFalse(quality['packing_legality_claimed'])
        self.assertTrue(quality['movable_individual_resource_compatibility_enforced'])

    def test_frozen_identity_reuses_without_forward_and_rejects_artifact_tampering(self):
        context, checkpoint, _ = self.inputs()
        first = self.generate(context, checkpoint)
        with patch.object(generation.FastHierarchicalInitializer, 'predict',
                          side_effect=AssertionError('reuse must not recompute')):
            self.assertEqual(self.generate(context, checkpoint), first)
        Path(first['ap_init']).write_text('# modified artifact\n', encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'Changed generation artifact'):
            self.generate(context, checkpoint)

    def test_changed_options_checkpoint_or_graph_are_not_reused(self):
        context, checkpoint, data = self.inputs()
        self.generate(context, checkpoint)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            with contextlib.redirect_stdout(io.StringIO()):
                generation.generate(context, checkpoint, self.root / 'output',
                                    dict(self.options, soft_density_passes=0))
        data['cluster_x'][0, 0] += 1.
        torch.save(data, context['graph'])
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.generate(context, checkpoint)
        # A distinct folder can intentionally start with the new graph.
        self.generate(context, checkpoint, folder='new_graph')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        saved['model_state']['query.weight'].add_(.01)
        torch.save(saved, checkpoint)
        with self.assertRaisesRegex(ValueError, 'identity changed'):
            self.generate(context, checkpoint, folder='new_graph')

    def test_timed_generation_refuses_old_directory_but_reproduces_prior(self):
        context, checkpoint, _ = self.inputs()
        first = self.generate(context, checkpoint)
        with self.assertRaisesRegex(ValueError, 'fresh measured directory'):
            self.generate(context, checkpoint, reuse=False)
        timed = self.generate(context, checkpoint, folder='timed', reuse=False)
        self.assertEqual(first['ap_init_sha256'], timed['ap_init_sha256'])
        self.assertEqual(first['fplace_sha256'], timed['fplace_sha256'])
        self.assertEqual(first['target_sha256'], timed['target_sha256'])

    def test_final_test_leakage_and_duplicate_movable_atoms_rejected(self):
        context, checkpoint, _ = self.inputs(family='training')
        with self.assertRaisesRegex(ValueError, 'leaked'):
            self.generate(context, checkpoint)
        context['family'] = 'unseen'
        logical = read(context['logic'])
        logical['clusters'][2]['atoms'][0] = logical['clusters'][1]['atoms'][0]
        write(context['logic'], logical)
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.generate(context, checkpoint)

    def test_fixed_and_movable_atoms_must_be_globally_disjoint(self):
        context, checkpoint, _ = self.inputs()
        logical = read(context['logic'])
        logical['clusters'][1]['atoms'][0] = 'fixed_atom'
        write(context['logic'], logical)
        with self.assertRaisesRegex(ValueError, 'Duplicate AP atom'):
            self.generate(context, checkpoint)
        # Validate the entire partition before publishing any native prior.
        self.assertFalse((self.root / 'output/init.ap.tsv').exists())
        self.assertFalse((self.root / 'output/generation.json').exists())

    def test_invalid_atom_names_and_empty_molecules_rejected_including_fixed(self):
        bad_names = ('', None, 42, [], 'white space', 'tab\tname', 'line\nname',
                     '#would_be_a_comment', 'nul\x00name')
        for cluster_index in (0, 1):
            for case_index, atom in enumerate(bad_names):
                with self.subTest(cluster=cluster_index, atom=repr(atom)):
                    context, checkpoint, _ = self.inputs()
                    logical = read(context['logic'])
                    logical['clusters'][cluster_index]['atoms'][0] = atom
                    write(context['logic'], logical)
                    with self.assertRaisesRegex(ValueError, 'Invalid AP atom name'):
                        self.generate(context, checkpoint, folder=f'bad_name_{cluster_index}_{case_index}')
            for case_index, atoms in enumerate(([], None, 'not-an-atoms-list')):
                with self.subTest(cluster=cluster_index, atoms=atoms):
                    context, checkpoint, _ = self.inputs()
                    logical = read(context['logic'])
                    logical['clusters'][cluster_index]['atoms'] = atoms
                    write(context['logic'], logical)
                    with self.assertRaisesRegex(ValueError, 'nonempty atoms list'):
                        self.generate(context, checkpoint, folder=f'bad_molecule_{cluster_index}_{case_index}')


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
