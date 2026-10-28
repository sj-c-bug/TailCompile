"""Evidence, frozen reuse, failure denominators and cold comparison contracts."""
import copy
from pathlib import Path
import shutil
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import qualified_learning as q
from io_utils import read, sha, write

HERE = Path(__file__).resolve().parent
SOURCE = HERE/'runs/development_004_capacity'


class QualifiedLearningTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(prefix='qualified-test-', dir=HERE/'runs')

    def panel(self):
        return q.qualified_panel(SOURCE)[2]

    def rows(self):
        rows = []
        for c in self.panel():
            for seed in q.SEEDS:
                for method in q.METHODS:
                    rows.append(dict(condition_id=c['id'], seed=seed, method=method,
                        status='success', total_seconds=90. if method == 'gnn' else 100.,
                        cpd_ns=1., wirelength=100., **{k: c[k] for k in
                            ('design', 'family', 'split', 'architecture', 'width_role')}))
        return rows

    def copy_evidence(self, target):
        for name in ('campaign.json', 'protocol.json', 'prepared.json', 'conditions_manifest.json',
                     'conditions.json', 'boundaries.json'):
            target.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(SOURCE/name, target/name)
        for pattern in ('calibration/*/trials.json', 'architectures/*/actual_audit.json'):
            for path in SOURCE.glob(pattern):
                out = target/path.relative_to(SOURCE)
                out.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, out)

    def test_real_synced_panel_and_counts(self):
        _, records, conditions = q.qualified_panel(SOURCE)
        self.assertEqual(len(records), 5)
        self.assertEqual(sum(r['split'] == 'train' for r in conditions), 24)
        self.assertEqual(sum(r['split'] == 'validation' for r in conditions), 16)
        self.assertEqual(q.queue_plan()['total_new_pnr_upper'], 1048)
        self.assertEqual(q.queue_plan()['comparison_runs'], 600)
        self.assertEqual({r['family'] for r in records if r['split'] == 'validation'}, {'blob_merge', 'diffeq'})

    def test_tampered_condition_catalog_rejected(self):
        with self.temporary() as tmp:
            root = Path(tmp)
            self.copy_evidence(root)
            value = read(root/'conditions.json')
            value['records'][0]['split'] = 'validation'
            write(root/'conditions.json', value)
            with self.assertRaisesRegex(ValueError, 'input changed'):
                q.qualified_panel(root)

    def test_reforged_hash_does_not_hide_family_leakage(self):
        with self.temporary() as tmp:
            root = Path(tmp)
            self.copy_evidence(root)
            value = read(root/'conditions.json')
            value['records'][0]['split'] = 'validation'
            write(root/'conditions.json', value)
            manifest = read(root/'conditions_manifest.json')
            manifest['conditions_sha256'] = sha(root/'conditions.json')
            write(root/'conditions_manifest.json', manifest)
            with self.assertRaisesRegex(ValueError, 'condition disagrees'):
                q.qualified_panel(root)

    def test_baseline_calibration_change_rejected(self):
        with self.temporary() as tmp:
            root = Path(tmp)
            self.copy_evidence(root)
            path = root/'calibration/sha/trials.json'
            value = read(path)
            value[0]['status'] = 'infrastructure_failure'
            write(path, value)
            with self.assertRaisesRegex(ValueError, 'normal Default'):
                q.qualified_panel(root)

    def test_reference_metrics_and_three_contrasts_separated(self):
        result = q.comparison_summary(self.rows(), self.panel())
        self.assertTrue(result['complete'])
        self.assertEqual(len(result['split_comparisons']), 6)
        for group in result['split_comparisons']:
            self.assertEqual(group['provisional_runtime_target'], group['method'] == 'gnn')
        self.assertFalse(result['full_acceptance_passed'])
        self.assertFalse(result['formal_480_completed'])
        self.assertFalse(result['generalization_claim'])

    def test_stress_failures_are_na_not_success_or_incomplete(self):
        rows = self.rows()
        for row in rows:
            if row['width_role'] == 'stress':
                row.update(status='unroutable', total_seconds=1., cpd_ns=None, wirelength=None)
        result = q.comparison_summary(rows, self.panel())
        self.assertTrue(result['complete'])
        stress = [s for s in result['strata'] if s['width_role'] == 'stress']
        self.assertTrue(all(s['runtime_ratio'] is None and s['jointly_successful'] == 0 for s in stress))
        self.assertTrue(all(s['provisional_runtime_target'] for s in result['split_comparisons'] if s['method'] == 'gnn'))

    def test_infrastructure_failure_never_awards_provisional_target(self):
        rows = self.rows()
        rows[0]['status'] = 'infrastructure_failure'
        result = q.comparison_summary(rows, self.panel())
        self.assertEqual(result['unknown_failure_count'], 1)
        self.assertFalse(any(s['provisional_runtime_target'] for s in result['split_comparisons']))

    def test_failure_regression_not_masked_by_fast_successes(self):
        rows = self.rows()
        failed = next(r for r in rows if r['method'] == 'gnn' and r['width_role'] == 'stress')
        failed.update(status='decode_failure', total_seconds=.001)
        result = q.comparison_summary(rows, self.panel())
        group = next(s for s in result['split_comparisons'] if s['split'] == failed['split'] and
                     s['method'] == 'gnn' and s['baseline'] == 'default')
        self.assertFalse(group['success_protection'])
        self.assertFalse(group['provisional_runtime_target'])
        self.assertEqual(result['unknown_failure_count'], 0)

    def test_missing_and_duplicate_rows(self):
        rows = self.rows()
        self.assertFalse(q.comparison_summary(rows[:-1], self.panel())['complete'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            q.comparison_summary(rows+[rows[0]], self.panel())

    def test_nonfinite_success_metrics_do_not_pass_quality(self):
        rows = self.rows()
        next(r for r in rows if r['method'] == 'gnn' and r['width_role'] == 'reference')['cpd_ns'] = float('inf')
        result = q.comparison_summary(rows, self.panel())
        self.assertFalse(next(r for r in result['split_comparisons'] if r['method'] == 'gnn' and
                              r['baseline'] == 'default' and r['split'] == 'train')['provisional_runtime_target'])

    def test_old_output_and_same_source_rejected(self):
        with self.temporary() as tmp:
            repo = Path(tmp)/'repo'
            runs = repo/'experiments/vtr/tailcompile_v7_3d/runs'
            a, b = runs/'a', runs/'b'
            write(b/'protocol.json', {})
            with self.assertRaisesRegex(ValueError, 'non-learning'):
                q.validate_paths(a, b, repo)
            with self.assertRaisesRegex(ValueError, 'separate'):
                q.validate_paths(a, a, repo)

    def test_borrowed_rr_cache_is_read_only_and_hash_checked(self):
        with self.temporary() as tmp:
            root = Path(tmp)
            arch = root/'arch.xml'
            arch.write_text('arch')
            key = q.identity(dict(architecture=sha(arch), binary='b', width=32))[:20]
            folder = root/key
            folder.mkdir()
            (folder/'rr_graph.xml').write_text('rr')
            write(folder/'device_graph.json', {})
            write(folder/'manifest.json', dict(rr_graph_sha256=sha(folder/'rr_graph.xml'),
                                              graph_sha256=sha(folder/'device_graph.json')))
            c = q.QualifiedCampaign.__new__(q.QualifiedCampaign)
            c.binary_hash = 'b'
            c.check_binary = lambda: None
            c.architecture = lambda *args: arch
            c._lock = lambda key: threading.Lock()
            c.verified_rr = set()
            c.catalog = dict(devices={key: str(folder)}, files={str((folder/'manifest.json').resolve()): sha(folder/'manifest.json')})
            before = {str(p): (sha(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()}
            c.rr({}, 'H0Z0', 32)
            self.assertEqual(before, {str(p): (sha(p), p.stat().st_mtime_ns) for p in root.rglob('*') if p.is_file()})
            (folder/'device_graph.json').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed during'):
                c.rr({}, 'H0Z0', 32)
            with self.assertRaisesRegex(ValueError, 'unregistered RRG'):
                c.rr({}, 'H0Z0', 64)


if __name__ == '__main__':
    unittest.main()
