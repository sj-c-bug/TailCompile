import json
from pathlib import Path
import tempfile
import unittest

from residual_observations import collect_residual_observations, _sha


class ResidualObservationTests(unittest.TestCase):
    def test_edited_subset_retention_and_partial_coverage(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp); source = p/'edits.json'
            source.write_text(json.dumps(dict(grid=[10, 20], atoms=[
                dict(name='edited', base_layer=0, target_layer=1, base_xy=[.2, .3], target_xy=[.3, .4]),
                dict(name='missing', base_layer=1, target_layer=0, base_xy=[.1, .2], target_xy=[.2, .3])])), encoding='utf-8')
            (p/'post_gp.blocks.tsv').write_text('0\tedited\t0\t0\t3\t8\t0.6\t0\tmodel\t0\n1\tunchanged\t0\t0\t1\t1\t0\t0\tmodel\t1\n', encoding='utf-8')
            (p/'post_route.fplace').write_text('edited 2 6 0 0 # annotation\n', encoding='utf-8')
            gen = dict(residual_edits=str(source), residual_edits_sha256=_sha(source))
            out = collect_residual_observations(gen, p, p)
            gp = out['stages']['post_gp']
            self.assertEqual(gp['status'], 'partial_coverage')
            self.assertEqual(gp['matched_atoms'], 1)
            self.assertEqual(gp['layer_target_retained_fraction'], 1)
            self.assertAlmostEqual(gp['mean_normalized_xy_l1_to_target'], 0)
            self.assertEqual(out['stages']['final']['layer_reverted_to_base_fraction'], 1)
            self.assertEqual(out['stages']['post_appack_legalization']['status'], 'missing')
            source.write_text('{}', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'changed'):
                collect_residual_observations(gen, p, p)

    def test_no_edits_and_no_manifest_are_explicit(self):
        self.assertEqual(collect_residual_observations(None, '.', '.')['status'], 'not_applicable')


if __name__ == '__main__':
    unittest.main()
