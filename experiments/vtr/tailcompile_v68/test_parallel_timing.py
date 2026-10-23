"""Controlled runtime measurement tests; synthetic records, no real VPR runs."""
from __future__ import annotations

from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from common import read, write
import parallel_timing as p


class ParallelTimingTests(unittest.TestCase):
    directory_lock = threading.Lock()

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='v68-parallel-timing-',
                                                     dir=Path(__file__).parent)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        cleanup = patch.object(p, 'after_metrics')
        cleanup.start()
        self.addCleanup(cleanup.stop)

    def fixture(self):
        plan = dict(designs=[dict(design=design, width=width)
                            for design, widths in [('zeta', [100, 110]),
                                                   ('alpha', [90, 100]),
                                                   ('unusable', [80])]
                            for width in widths],
                    seeds=[341, 342, 343, 344, 345],
                    methods=['default', 'gnn_warmstart'], checkpoint_sha256='fixture')
        write(self.root/'evaluation_plan.json', plan)
        write(self.root/'evaluation_report.json', dict(all_route_outcomes_known=True))
        for condition in plan['designs']:
            for seed in plan['seeds']:
                for method in plan['methods']:
                    status = 'success'
                    if condition['design'] == 'unusable':
                        status = 'unroutable'
                    if condition['design'] == 'alpha' and condition['width'] == 90:
                        if method == 'gnn_warmstart' and seed == 342:
                            status = 'unroutable'
                    if condition['design'] == 'zeta' and condition['width'] == 100:
                        if method == 'default' and seed == 341:
                            status = 'timeout'
                    # Timing intentionally uses only the first three frozen seeds.
                    if condition['design'] == 'alpha' and seed == 344:
                        status = 'unroutable'
                    write(self.root/'evaluation/runs'/condition['design']/
                          f'w{condition["width"]}'/f's{seed}'/method/'metrics.json',
                          dict(status=status, wall_seconds=10))
        return plan

    def test_conditions_choose_lowest_joint_success_and_only_first_three_seeds(self):
        self.fixture()
        selected = p.timing_conditions(self.root)
        self.assertEqual(selected, [
            dict(design='alpha', width=100, seeds=[341, 342, 343]),
            dict(design='zeta', width=110, seeds=[341, 342, 343])])

    def test_conditions_require_classified_scan_before_timing(self):
        self.fixture()
        write(self.root/'evaluation_report.json', dict(all_route_outcomes_known=False))
        with self.assertRaises(ValueError):
            p.timing_conditions(self.root)

    def test_missing_scan_metric_is_not_filled_from_serial_data(self):
        self.fixture()
        missing = self.root/'evaluation/runs/alpha/w100/s341/default/metrics.json'
        missing.unlink()
        write(self.root/'timing/serial_001/alpha/w100/s341/default/metrics.json',
              dict(status='success', wall_seconds=1))
        with self.assertRaises((ValueError, FileNotFoundError)):
            p.timing_conditions(self.root)

    def test_serial_evidence_cannot_change_parallel_condition_selection(self):
        self.fixture()
        original = read(self.root/'evaluation_plan.json')
        serial = self.root/'timing/serial_001/alpha/w90/s342/gnn_warmstart/metrics.json'
        write(serial, dict(status='success', wall_seconds=.001))
        write(self.root/'timing_report.json', dict(complete=True, rows=['serial fixture']))
        selected = p.timing_conditions(self.root)
        self.assertEqual(selected[0]['width'], 100)
        self.assertEqual(read(serial)['wall_seconds'], .001)
        self.assertEqual(read(self.root/'timing_report.json')['rows'], ['serial fixture'])
        self.assertEqual(read(self.root/'evaluation_plan.json'), original)

    def test_schedule_is_deterministic_paired_and_counterbalanced(self):
        conditions = [dict(design='alpha', width=100, seeds=[341, 342, 343]),
                      dict(design='zeta', width=110, seeds=[341, 342, 343])]
        cores = [dict(cpu=2), dict(cpu=4), dict(cpu=6), dict(cpu=8)]
        waves = p.schedule(conditions, cores)
        self.assertEqual(waves, p.schedule(conditions, cores))
        self.assertEqual([len(w['lanes']) for w in waves], [4, 2])
        lanes = [lane for wave in waves for lane in wave['lanes']]
        self.assertEqual(len({(r['design'], r['width'], r['seed']) for r in lanes}), 6)
        self.assertEqual({r['cpu'] for r in lanes}, {2, 4, 6, 8})
        self.assertEqual(sum(r['order'][0] == 'default' for r in lanes), 3)
        for wave in waves:
            self.assertEqual(len({r['cpu'] for r in wave['lanes']}), len(wave['lanes']))
            for lane in wave['lanes']:
                self.assertEqual(set(lane['order']), {'default', 'gnn_warmstart'})
                self.assertEqual(len(lane['order']), 2)

    @staticmethod
    def topology_rows():
        return [dict(cpu=cpu, package=0, core=core, numa=0)
                for cpu, core in [(0, 0), (1, 1), (2, 2), (3, 3),
                                  (8, 0), (9, 1), (10, 2), (11, 3)]]

    def test_hardware_caps_requested_jobs_by_physical_cores_quota_and_pairs(self):
        with patch.object(p, 'topology', return_value=self.topology_rows()), \
             patch.object(p, 'quota_info', return_value=3.5):
            measured = p.hardware(32, condition_count=12)
            self.assertEqual(measured['concurrency'], 3)
            self.assertEqual(len(measured['selected_cores']), 3)
            self.assertEqual(len({(r['package'], r['core'])
                                  for r in measured['selected_cores']}), 3)
            self.assertEqual(p.hardware(32, condition_count=2)['concurrency'], 2)
        with patch.object(p, 'topology', return_value=self.topology_rows()), \
             patch.object(p, 'quota_info', return_value=None):
            self.assertEqual(p.hardware(32, condition_count=12)['concurrency'], 4)

    def test_manual_hardware_rejects_smt_and_unavailable_cpu_and_caps_quota(self):
        with patch.object(p, 'topology', return_value=self.topology_rows()), \
             patch.object(p, 'quota_info', return_value=4):
            with self.assertRaises(ValueError):
                p.hardware(2, cpu_list=[0, 8], condition_count=12)
            with self.assertRaises(ValueError):
                p.hardware(2, cpu_list=[0, 99], condition_count=12)
            accepted = p.hardware(2, cpu_list=[1, 2], condition_count=12)
            self.assertEqual([r['cpu'] for r in accepted['selected_cores']], [1, 2])
        with patch.object(p, 'topology', return_value=self.topology_rows()), \
             patch.object(p, 'quota_info', return_value=1.5):
            capped = p.hardware(2, cpu_list=[0, 1], condition_count=12)
            self.assertEqual(capped['concurrency'], 1)
            self.assertEqual([r['cpu'] for r in capped['selected_cores']], [0])

    def execution_fixture(self):
        args = SimpleNamespace(out=self.root, run_name='test_c2', retry_timeouts=False,
                               retry_failed=False, confirm_remote_stopped=False)
        folder = p.run_root(args)
        conditions = [dict(design='alpha', width=100, seeds=[341, 342, 343])]
        waves = p.schedule(conditions, [dict(cpu=2), dict(cpu=4)])
        protocol = dict(schema='fixture', hardware=dict(concurrency=2))
        write(folder/'protocol.json', protocol)
        write(folder/'schedule.json', dict(waves=waves))
        generation = dict(ap_init='synthetic-init.tsv', warm_inference_lowering_seconds=1,
                          cold_from_prepared_graph_seconds=2, timing=dict(logic_forward=1))
        write(self.root/'evaluation/generation/alpha/w100/generation.json', generation)
        audit = self.root/'audit.json'
        write(audit, dict(timing=dict(logic_preparation=3, device_preprocessing=1000,
                                    device_provenance=1000)))
        context = dict(design='alpha', width=100, audit=str(audit))
        write(self.root/'contexts.json', dict(contexts=[context]))
        write(self.root/'inputs/alpha/structure_run/trial.json', dict(wall_seconds=4))
        files = dict(generation=str(self.root/'evaluation/generation/alpha/w100/generation.json'),
                     audit=str(audit),
                     structure_trial=str(self.root/'inputs/alpha/structure_run/trial.json'))
        protocol.update(contexts_file=str(self.root/'contexts.json'),
                        contexts_sha256=p.sha(self.root/'contexts.json'),
                        input_files={'alpha/w100': files},
                        input_sha256={'alpha/w100': {k: p.sha(v) for k, v in files.items()}})
        write(folder/'protocol.json', protocol)
        write(self.root/'timing_report.json', dict(serial_marker='must remain unchanged'))
        return args, folder, protocol, waves, {('alpha', 100): (context, generation)}

    @staticmethod
    def synthetic_trial(args, context, lane, directory, generation, barrier,
                        protocol, method):
        # Windows sandbox policy can reject simultaneous parent-directory creation.
        # This only serializes synthetic fixture setup, before the timing barrier.
        with ParallelTimingTests.directory_lock:
            directory.mkdir(parents=True, exist_ok=True)
        barrier.wait(timeout=10)
        row = dict(method=method, status='success', started_unix=time.time(),
                   wall_seconds=10 if method == 'default' else 5,
                   wirelength=100, cpd=10, cpu=lane['cpu'])
        write(directory/'trial.json', row)
        write(directory/'metrics.json', row)
        write(directory/'affinity.json', dict(allowed_cpus=[lane['cpu']],
                                             thread_env=dict(OMP_NUM_THREADS='1')))
        return row

    def test_run_resume_preserves_complete_waves_and_report_separates_tail_load(self):
        args, folder, protocol, waves, cases = self.execution_fixture()
        setup = (folder, protocol, waves, cases)
        with patch.object(p, 'prepare', return_value=setup), \
             patch.object(p, 'snapshot', return_value={}), \
             patch.object(p, 'run_trial', side_effect=self.synthetic_trial) as trial:
            report = p.run(args)
            self.assertEqual(trial.call_count, 6)
        self.assertTrue(report['complete'])
        self.assertFalse(report['serial_results_included'])
        self.assertEqual({c['effective_lanes'] for c in report['comparisons']}, {1, 2})
        self.assertTrue(all(c['geomean_speedup']['vpr_only'] == 2
                            for c in report['comparisons']))
        self.assertTrue(all(abs(row['speedup']['cached_graph_warm'] - 10/6) < 1e-12
                            for row in report['rows']))
        self.assertTrue(all(abs(row['speedup']['new_design_prepared_device'] - 10/14) < 1e-12
                            for row in report['rows']))
        self.assertEqual(read(self.root/'timing_report.json')['serial_marker'],
                         'must remain unchanged')
        old = [p.wave_folder(folder, wave).joinpath('wave.json').read_bytes()
               for wave in waves]
        with patch.object(p, 'prepare', return_value=setup), \
             patch.object(p, 'run_trial', side_effect=AssertionError('must reuse wave')):
            p.run(args)
        self.assertEqual(old, [p.wave_folder(folder, wave).joinpath('wave.json').read_bytes()
                               for wave in waves])

    def test_partial_wave_is_archived_and_rerun_as_whole_pair(self):
        args, folder, protocol, waves, cases = self.execution_fixture()
        partial = p.wave_folder(folder, waves[0])
        old = partial/'half0/lane00/trial.json'
        write(old, dict(status='success', wall_seconds=9999))
        write(partial/'wave.json', dict(complete=False))
        with patch.object(p, 'prepare', return_value=(folder, protocol, waves, cases)), \
             patch.object(p, 'snapshot', return_value={}), \
             patch.object(p, 'run_trial', side_effect=self.synthetic_trial) as trial:
            p.run(args)
            self.assertEqual(trial.call_count, 6)
        archived = list((folder/'_attempts').rglob('trial.json'))
        self.assertEqual(len(archived), 1)
        self.assertEqual(read(archived[0])['wall_seconds'], 9999)
        self.assertNotEqual(read(old)['wall_seconds'], 9999)

    def test_completed_wave_rejects_changed_evidence(self):
        args, folder, protocol, waves, cases = self.execution_fixture()
        with patch.object(p, 'prepare', return_value=(folder, protocol, waves, cases)), \
             patch.object(p, 'snapshot', return_value={}), \
             patch.object(p, 'run_trial', side_effect=self.synthetic_trial):
            p.run(args)
        changed = p.wave_folder(folder, waves[0])/'half0/lane00/trial.json'
        write(changed, dict(status='success', wall_seconds=9999))
        with self.assertRaises(ValueError):
            p.completed_wave(folder, waves[0])

    def test_completed_wave_rejects_embedded_result_tampering(self):
        args, folder, protocol, waves, cases = self.execution_fixture()
        with patch.object(p, 'prepare', return_value=(folder, protocol, waves, cases)), \
             patch.object(p, 'snapshot', return_value={}), \
             patch.object(p, 'run_trial', side_effect=self.synthetic_trial):
            p.run(args)
        path = p.wave_folder(folder, waves[0])/'wave.json'
        changed = read(path)
        changed['results']['0:0']['wall_seconds'] = .00001
        write(path, changed)
        with self.assertRaisesRegex(ValueError, 'Embedded wave results'):
            p.completed_wave(folder, waves[0])

    def test_report_rejects_changed_generation_costs(self):
        args, folder, protocol, waves, cases = self.execution_fixture()
        p.validate_inputs(protocol)
        path = self.root/'evaluation/generation/alpha/w100/generation.json'
        changed = read(path)
        changed['warm_inference_lowering_seconds'] = 1000
        write(path, changed)
        with self.assertRaisesRegex(ValueError, 'input/cost evidence changed'):
            p.validate_inputs(protocol)
        with self.assertRaises(ValueError):
            p.report(args)

    def test_report_rejects_changed_contexts(self):
        args, folder, protocol, waves, cases = self.execution_fixture()
        p.validate_inputs(protocol)
        write(self.root/'contexts.json', dict(contexts=[]))
        with self.assertRaisesRegex(ValueError, 'contexts changed'):
            p.validate_inputs(protocol)
        with self.assertRaises(ValueError):
            p.report(args)

    def test_prepare_rejects_initializer_from_different_checkpoint(self):
        args, folder, protocol, waves, cases = self.execution_fixture()
        args.timing_jobs, args.cpu_list = 2, None
        context, generation = cases['alpha', 100]
        generation['signature'] = dict(checkpoint_sha256='other-checkpoint')
        write(self.root/'evaluation/generation/alpha/w100/generation.json', generation)
        write(self.root/'evaluation_plan.json', dict(checkpoint_sha256='frozen-checkpoint'))
        conditions = [dict(design='alpha', width=100, seeds=[341, 342, 343])]
        with patch.object(p, 'timing_conditions', return_value=conditions), \
             patch.object(p, 'hardware', return_value={}), \
             patch.object(p.e, 'load_contexts', return_value=[context]):
            with self.assertRaisesRegex(ValueError, 'checkpoint differs'):
                p.prepare(args)

    def test_prepare_rejects_changed_initializer_payload(self):
        args, folder, protocol, waves, cases = self.execution_fixture()
        args.timing_jobs, args.cpu_list = 2, None
        context, generation = cases['alpha', 100]
        initial = self.root/'init.tsv'
        initial.write_text('before\n', encoding='utf-8')
        generation.update(signature=dict(checkpoint_sha256='frozen-checkpoint'),
                          ap_init=str(initial), ap_init_sha256=p.sha(initial))
        initial.write_text('after\n', encoding='utf-8')
        write(self.root/'evaluation/generation/alpha/w100/generation.json', generation)
        write(self.root/'evaluation_plan.json', dict(checkpoint_sha256='frozen-checkpoint'))
        conditions = [dict(design='alpha', width=100, seeds=[341, 342, 343])]
        with patch.object(p, 'timing_conditions', return_value=conditions), \
             patch.object(p, 'hardware', return_value={}), \
             patch.object(p.e, 'load_contexts', return_value=[context]):
            with self.assertRaisesRegex(ValueError, 'initializer changed'):
                p.prepare(args)

    def test_timeout_is_censored_not_success_and_complete_wave_can_resume(self):
        args, folder, protocol, waves, cases = self.execution_fixture()
        def timed_out(*values):
            row = self.synthetic_trial(*values)
            if row['method'] == 'gnn_warmstart':
                row['status'] = 'timeout'
                directory = values[3]
                write(directory/'trial.json', row)
                write(directory/'metrics.json', row)
            return row
        setup = (folder, protocol, waves, cases)
        with patch.object(p, 'prepare', return_value=setup), \
             patch.object(p, 'snapshot', return_value={}), \
             patch.object(p, 'run_trial', side_effect=timed_out):
            report = p.run(args)
        self.assertTrue(report['complete'])
        self.assertFalse(report['all_route_outcomes_known'])
        self.assertEqual(report['status_counts'], dict(success=3, timeout=3))
        self.assertEqual(report['comparisons'], [])
        self.assertTrue(all(not row['comparable'] for row in report['rows']))
        with patch.object(p, 'prepare', return_value=setup), \
             patch.object(p, 'run_trial', side_effect=AssertionError('timeout reused by default')):
            p.run(args)

    def test_partial_wave_on_other_host_requires_explicit_confirmation(self):
        folder = self.root/'timing_parallel/test_c2/waves/w0000'
        write(folder/'half0/lane00/started.json',
              dict(host='definitely-not-this-host.invalid', parent_pid=123, child_pid=456))
        with self.assertRaises(RuntimeError):
            p.assert_workers_stopped(folder)
        p.assert_workers_stopped(folder, confirm_remote_stopped=True)


if __name__ == '__main__':
    unittest.main()
