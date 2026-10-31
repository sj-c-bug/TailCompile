from pathlib import Path
import tempfile
import unittest
import json

from layer_observations import collect_layers


class LayerObservationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.folder=Path(self.temp.name)
        self.initial=self.folder/'init.ap.tsv'
        self.initial.write_text('# atom x y z subtile\na\t1\t1\t0\t0\nb\t2\t2\t1\t0\nc\t3\t3\t0\t0\n')
        self.prefix=self.folder/'post_gp'

    def gp(self, rows):
        Path(str(self.prefix)+'.blocks.tsv').write_text(''.join(
            f'{i}\t{atom}\t0\t{fixed}\t1\t1\t{z}\t0\t.names\t{i}\n'
            for i,(atom,z,fixed) in enumerate(rows)))

    def test_continuous_gp_rounds_half_up_and_tracks_atom_changes(self):
        self.gp([('a',.5,0),('b',.49,1),('c',.2,0)])
        result=collect_layers(self.initial,self.prefix)
        gp=result['stages']['post_gp']
        self.assertEqual(gp['layer_counts'],{'0':2,'1':1})
        self.assertEqual(gp['changed_atoms'],2)
        self.assertAlmostEqual(gp['changed_fraction'],2/3)
        self.assertEqual(gp['native_fixed_layer_changes'],1)
        self.assertEqual(gp['fixed_flag_coverage'],1.)
        self.assertEqual(result['initial']['native_fixed_atom_count'],1)
        self.assertIsNone(result['stages']['final']['changed_fraction'])
        self.assertEqual(result['stages']['post_appack_legalization']['status'],'missing')
        json.dumps(result,allow_nan=False)

    def test_optional_legalized_and_final_stages_have_transition_boundaries(self):
        self.gp([('a',1.,0),('b',1.,1),('c',0.,0)])
        legal=self.folder/'post_legalizer.fplace'
        final=self.folder/'post_route.fplace'
        legal.write_text('a 1 1 1 0\nb 2 2 1 0\nc 3 3 1 0\n')
        final.write_text('a 1 1 0 0\nb 2 2 1 0\nc 3 3 1 0\n')
        result=collect_layers(self.initial,self.prefix,final,legal)
        legal_report=result['stages']['post_appack_legalization']
        final_report=result['stages']['final']
        self.assertEqual(legal_report['previous_available_stage'],'post_gp')
        self.assertAlmostEqual(legal_report['changed_fraction_from_previous'],1/3)
        self.assertEqual(final_report['previous_available_stage'],'post_appack_legalization')
        self.assertAlmostEqual(final_report['changed_fraction_from_previous'],1/3)
        self.assertAlmostEqual(final_report['changed_fraction'],1/3)
        self.assertEqual(final_report['native_fixed_layer_changes'],0)
        self.assertNotEqual(legal_report['source_sha256'],final_report['source_sha256'])

    def test_unmatched_and_missing_atoms_do_not_become_unchanged(self):
        self.gp([('a',1.,0),('extra',0.,1)])
        result=collect_layers(self.initial,self.prefix)
        gp=result['stages']['post_gp']
        self.assertEqual(gp['status'],'partial_coverage')
        self.assertEqual(gp['matched_initial_atoms'],1)
        self.assertEqual(gp['missing_initial_atoms'],2)
        self.assertEqual(gp['unmatched_atom_rows'],1)
        self.assertEqual(gp['unmatched_native_fixed_rows'],1)
        self.assertEqual(gp['changed_fraction'],1.)
        self.assertAlmostEqual(gp['matched_fraction_of_initial'],1/3)

    def test_absent_gp_does_not_invent_fixed_flags_or_previous_stage(self):
        final=self.folder/'final.fplace'
        final.write_text('a 1 1 0 0\nb 2 2 1 0\nc 3 3 0 0\n')
        result=collect_layers(self.initial,self.prefix,final)
        row=result['stages']['final']
        self.assertEqual(row['changed_fraction'],0.)
        self.assertEqual(row['fixed_flag_coverage'],0.)
        self.assertEqual(row['previous_available_stage'],'initial')
        self.assertIsNone(row['native_fixed_changed_fraction'])

    def test_missing_reference_produces_missing_comparison(self):
        self.gp([('a',0.,0)])
        result=collect_layers(None,self.prefix)
        self.assertEqual(result['initial']['status'],'missing_or_empty')
        self.assertIsNone(result['stages']['post_gp']['changed_fraction'])
        self.assertIsNone(result['stages']['post_gp']['matched_fraction_of_initial'])

    def test_duplicate_initial_or_stage_atom_rejected(self):
        self.gp([('a',0.,0),('a',1.,0)])
        with self.assertRaisesRegex(ValueError,'duplicate matched atom'):
            collect_layers(self.initial,self.prefix)
        self.initial.write_text('a 1 1 0 0\na 1 1 1 0\n')
        with self.assertRaisesRegex(ValueError,'duplicate initial atom'):
            collect_layers(self.initial,self.prefix)

    def test_flat_fractional_layer_is_rejected(self):
        final=self.folder/'wrong.fplace'
        final.write_text('a 1 1 0.5 0\n')
        with self.assertRaisesRegex(ValueError,'integer layer'):
            collect_layers(self.initial,None,final)

    def test_native_verbose_flat_comments_preserve_hash_inside_atom(self):
        self.initial.write_text('a#inside 1 1 0 0\nb 2 2 1 0\n')
        final=self.folder/'verbose.fplace'
        final.write_text('# native atom placement\na#inside 1 1 1 0 #42 .names clb[0]/fle[1]/lut\n'
                         'b 2 2 1 0 #43 .latch\n')
        result=collect_layers(self.initial,None,final)
        stage=result['stages']['final']
        self.assertEqual(stage['status'],'observed')
        self.assertEqual(stage['matched_initial_atoms'],2)
        self.assertEqual(stage['changed_fraction'],.5)

    def test_extra_flat_fields_must_start_with_native_comment_marker(self):
        final=self.folder/'invalid.fplace'
        final.write_text('a 1 1 0 0 malformed_tail\n')
        with self.assertRaisesRegex(ValueError,'expected atom x y layer subtile'):
            collect_layers(self.initial,None,final)


if __name__=='__main__':
    unittest.main()
