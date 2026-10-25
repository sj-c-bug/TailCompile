import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import inventory


class InventoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.vtr = self.root/'vtr'
        self.bench = self.vtr/'vtr_flow/benchmarks/verilog'
        self.koios = self.bench/'koios'
        self.koios.mkdir(parents=True)
        self.base = self.root/'baseline'
        self.base.mkdir()
        self.rows = [dict(design='sha',family='sha',split='train',source='sha.v'),
            dict(design='tpu_like.small.os',family='tpu',split='train',source='koios/tpu_like.small.os.v'),
            dict(design='spree',family='spree',split='final_test',source='spree.v')]
        (self.base/'cohort.json').write_text(json.dumps(dict(designs=self.rows)))
        for relative in ('sha.v','spree.v','koios/tpu_like.small.os.v',
                         'koios/tpu_like.large.ws.v','koios/clstm_like.small.v',
                         'koios/dla_like.large.v','koios/hard_block_include.v'):
            (self.bench/relative).write_text('module test; endmodule\n')
        (self.koios/'hard_mem_include.v').write_text('`define hard_mem\n')
        (self.koios/'complex_dsp_include.v').write_text('`define complex_dsp\n')
        self.patch = patch.object(inventory,'V68',self.base)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_union_excludes_shared_include_and_groups_parameter_variants(self):
        value = inventory.discover(self.vtr, checkpoint_metadata={'training_families':['tpu']})
        self.assertEqual(value['design_count'],6)
        self.assertEqual(value['koios_design_count'],4)
        self.assertEqual(value['family_count'],4)
        rows = {r['design']:r for r in value['designs']}
        self.assertEqual(rows['tpu_like.large.ws']['family'],'tpu')
        self.assertEqual(rows['tpu_like.large.ws']['checkpoint_exposure'],'checkpoint_training_family')
        self.assertEqual(rows['dla_like.large']['family'],rows['clstm_like.small']['family'])
        self.assertEqual(rows['spree']['checkpoint_exposure'],'previously_observed_holdout_family')
        self.assertTrue(all(r['source_sha256'] and r['split']=='sweep' for r in rows.values()))
        helpers = {row['source']:row for row in value['excluded']}
        self.assertEqual(set(helpers), {'koios/hard_block_include.v',
            'koios/hard_mem_include.v','koios/complex_dsp_include.v'})
        self.assertEqual(helpers['koios/hard_block_include.v']['reason'],'shared_hard_block_include')
        self.assertEqual(helpers['koios/hard_mem_include.v']['reason'],'architecture_memory_configuration_include')
        self.assertEqual(helpers['koios/complex_dsp_include.v']['reason'],'architecture_dsp_configuration_include')
        for source, row in helpers.items():
            self.assertEqual(row['sha256'],inventory.sha(self.bench/source))
            self.assertNotIn(Path(source).stem,rows)

    def test_configuration_includes_cannot_be_requested_as_designs(self):
        for name in ('hard_mem_include','complex_dsp_include','hard_block_include'):
            with self.assertRaisesRegex(ValueError,'Unknown requested designs'):
                inventory.discover(self.vtr,designs=[name])

    def test_test_named_real_module_is_not_filtered_by_name(self):
        # Koios test.v is executable regression RTL, unlike macro-only includes.
        # Inventory does not infer standalone RTL eligibility from a name.
        (self.koios/'test.v').write_text('module top(input a, output b); assign b=a; endmodule\n')
        value=inventory.discover(self.vtr)
        self.assertEqual(value['design_count'],7)
        self.assertEqual(value['koios_design_count'],5)
        self.assertIn('test',{row['design'] for row in value['designs']})

    def test_scopes_and_explicit_subset_are_auditable(self):
        self.assertEqual(inventory.discover(self.vtr,scope='koios')['design_count'],4)
        self.assertEqual(inventory.discover(self.vtr,scope='existing')['design_count'],3)
        self.assertEqual(inventory.discover(self.vtr,designs=['spree'])['design_count'],1)
        with self.assertRaisesRegex(ValueError,'outside scope'):
            inventory.discover(self.vtr,designs=['spree'],scope='koios')
        with self.assertRaisesRegex(ValueError,'Unknown requested'):
            inventory.discover(self.vtr,designs=['missing'])

    def test_missing_existing_input_is_not_silently_dropped(self):
        (self.bench/'sha.v').unlink()
        rows = inventory.discover(self.vtr)['designs']
        row = next(r for r in rows if r['design']=='sha')
        self.assertEqual(row['input_status'],'missing_input')
        self.assertIsNone(row['source_sha256'])
        self.assertEqual(len(row['missing_inputs']),1)

    def test_source_escape_and_collision_are_rejected(self):
        self.rows[0]['source']='../../outside.v'
        (self.base/'cohort.json').write_text(json.dumps(dict(designs=self.rows)))
        with self.assertRaises(ValueError):
            inventory.discover(self.vtr)
        self.rows[0]['source']='sha.v'
        (self.base/'cohort.json').write_text(json.dumps(dict(designs=self.rows)))
        (self.koios/'sha.v').write_text('module sha; endmodule')
        with self.assertRaisesRegex(ValueError,'two sources'):
            inventory.discover(self.vtr)


if __name__ == '__main__':
    unittest.main()
