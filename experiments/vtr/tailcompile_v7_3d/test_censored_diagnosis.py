"""Diagnostic evidence, relocation and source immutability contracts."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import diagnose_censored_design as diagnostic

HERE = Path(__file__).resolve().parent


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


class CensoredDiagnosisTests(unittest.TestCase):
    def setUp(self):
        (HERE/'runs').mkdir(exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix='censored-diagnostic-', dir=HERE/'runs')
        self.addCleanup(self.temp.cleanup)
        self.repo = Path(self.temp.name)/'repo'
        self.runs = self.repo/diagnostic.RUNS_REL
        self.old = self.runs/'original'
        self.source = self.runs/'recovered'
        self.output = self.runs/'learning'/'or1200_diagnostic_report.json'
        self.rows = []
        for width in (32, 64, 96):
            for arch in diagnostic.ARCHITECTURES:
                for seed in diagnostic.CALIBRATION_SEEDS:
                    status = 'success' if width >= 64 else 'unroutable'
                    if width == 96 and arch == 'H0Z1' and seed == 502:
                        status = 'unroutable'
                    manifest = self.old/'calibration/or1200'/arch/f'w{width}'/f's{seed}'/'vpr/manifest.json'
                    log = manifest.parent/'attempts/attempt_0001/vpr.out'
                    remote = lambda p: '/server/TailCompile/' + p.relative_to(self.repo).as_posix()
                    write(manifest, dict(status=status, returncode=0 if status == 'success' else 2, log=remote(log)))
                    self.rows.append(dict(design='or1200', architecture=arch, width=width, seed=seed,
                                          method='default', status=status, vpr_manifest=remote(manifest)))
        self.trials = self.source/'calibration/or1200/trials.json'
        write(self.trials, self.rows)
        write(self.source/'calibration/or1200/boundary.json', dict(status='right_censored',
              W_ref=None, W_stress=None, record=dict(grid=[20, 20])))
        write(self.source/'architectures/20x20/actual_audit.json', dict(four_cell_matches=True,
              width_coupling={a: dict(z_capacity_invariant=True) for a in diagnostic.ARCHITECTURES}))

    def diagnose(self):
        return diagnostic.diagnose(self.source, repo=self.repo)

    def test_nonmonotonic_cell_and_missing_logs_do_not_establish_cause(self):
        report = self.diagnose()
        self.assertEqual(report['statuses'], dict(success=15, unroutable=9))
        self.assertEqual(report['log_coverage']['available'], 0)
        self.assertEqual(report['nonmonotonic_events'], [dict(architecture='H0Z1', seed=502,
            lower_success_width=64, higher_failure_width=96)])
        self.assertFalse(report['conclusion']['cause_established'])
        self.assertFalse(report['conclusion']['learning_blocked'])
        self.assertTrue(report['rrg']['z_capacity_invariant']['H0Z0'])
        self.assertTrue(all(r['manifest_available'] for r in report['trials']))

    def test_dependency_failure_is_not_routing_negative(self):
        row = self.rows[0]
        row['status'] = 'other_failure'
        manifest = diagnostic._recorded_path(row['vpr_manifest'], self.runs.resolve())
        log = manifest.parent/'attempts/attempt_0001/vpr.out'
        log.parent.mkdir(parents=True)
        log.write_text('vpr: error while loading shared libraries: libtcl8.6.so: cannot open shared object file\n')
        write(manifest, dict(status='other_failure', returncode=127, log=str(log)))
        write(self.trials, self.rows)
        report = self.diagnose()
        self.assertEqual(report['categories']['infrastructure_or_interruption'], 1)
        found = next(r for r in report['trials'] if r['returncode'] == 127)
        self.assertIn('missing_dependency', found['log']['observations'][0]['tags'])
        self.assertFalse(found['log']['cause_established'])

    def test_source_and_reused_cohorts_remain_immutable(self):
        before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in self.runs.rglob('*') if p.is_file()}
        report = self.diagnose()
        diagnostic.write_report(report, self.output, repo=self.repo)
        for p, old in before.items():
            self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), old)
        for output in (self.source/'report.json', self.old/'report.json', self.repo/'outside.json'):
            with self.assertRaises(ValueError):
                diagnostic.write_report(report, output, repo=self.repo)

    def test_duplicate_and_missing_matrix_rows_are_visible(self):
        write(self.trials, self.rows + [self.rows[0]])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            self.diagnose()
        write(self.trials, self.rows[1:])
        report = self.diagnose()
        self.assertEqual(report['status'], 'needs_evidence_review')
        self.assertFalse(report['by_width'][0]['complete_expected_panel'])

    def test_manifest_status_mismatch_is_not_hidden(self):
        rows = copy.deepcopy(self.rows)
        rows[0]['status'] = 'success'
        write(self.trials, rows)
        report = self.diagnose()
        self.assertEqual(report['status'], 'needs_evidence_review')
        self.assertEqual(report['issues'][0]['kind'], 'status_mismatch')

    def test_paths_cannot_escape_through_parent_segments(self):
        with self.assertRaises(ValueError):
            diagnostic._recorded_path('/server/' + diagnostic.RUNS_REL.as_posix() + '/../evil', self.runs.resolve())
        with self.assertRaises(ValueError):
            diagnostic.diagnose(self.source, design='../other', repo=self.repo)

    def test_unrelated_output_is_never_overwritten(self):
        write(self.output, dict(important='model selection'))
        with self.assertRaisesRegex(ValueError, 'unrelated'):
            diagnostic.write_report(self.diagnose(), self.output, repo=self.repo)
        self.assertEqual(json.loads(self.output.read_text()), dict(important='model selection'))

    def test_bounded_log_tail_keeps_explicit_failure_evidence(self):
        log = self.source/'large.log'
        log.write_text('x' * (diagnostic.LOG_TAIL_BYTES + 10) + '\nCircuit is unroutable\n')
        evidence = diagnostic._log_evidence(log)
        self.assertTrue(evidence['tail_truncated'])
        self.assertEqual(evidence['tail_bytes'], diagnostic.LOG_TAIL_BYTES)
        self.assertEqual(evidence['observations'][-1]['tags'], ['routing_failure'])


if __name__ == '__main__':
    unittest.main()
