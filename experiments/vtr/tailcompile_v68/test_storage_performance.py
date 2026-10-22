"""Deterministic complexity checks using disposable synthetic trial payloads.

These tests count normalization, collection traversal, and directory work rather
than timing filesystem operations. They must stay portable without Git or /proc.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Set
from contextlib import contextmanager, redirect_stderr
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

import storage


class CountingIterable:
    def __init__(self, values):
        self.values = tuple(values)
        self.iterations = 0

    def __iter__(self):
        self.iterations += 1
        return iter(self.values)


class MembershipOnlySet(Set):
    """A global dependency index supports lookup, never copying or scanning."""

    def __init__(self, values):
        self.values = frozenset(values)
        self.lookups = 0
        self.iterations = 0

    def __contains__(self, value):
        self.lookups += 1
        return value in self.values

    def __len__(self):
        return len(self.values)

    def __iter__(self):
        self.iterations += 1
        raise AssertionError('Global dependencies must only use membership lookups')


class StoragePerformanceTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tailcompile-v68-storage-perf-')
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name).resolve()/'repo'
        self.cohort = self.repo/'experiments/vtr/tailcompile_v68/runs/fixture'
        self.cohort.mkdir(parents=True)

    def trial(self, index, *, payload_count=1):
        folder = self.cohort/'feedback/runs'/f'design{index}/w100/c0/s321'
        folder.mkdir(parents=True)
        identity = dict(stage='route', design=f'design{index}')
        storage.save(folder/'trial.json', dict(status='success', identity=identity))
        storage.save(folder/'metrics.json', dict(status='success', identity=identity))
        payloads = []
        for number in range(payload_count):
            path = folder/f'payload{number}.net'
            path.write_text(f'synthetic disposable payload {index}/{number}\n', encoding='utf-8')
            payloads.append(path)
        return folder, payloads

    def global_tables(self, size=200):
        # These paths deliberately never coincide with payloads or evidence, so
        # resolving a payload itself cannot inflate the global-table counters.
        base = self.repo/'global_path_probes'
        active = [base/'active'/f'worker{index}' for index in range(size)]
        tracked = [base/'tracked'/f'input{index}' for index in range(size)]
        return active, tracked

    @contextmanager
    def count_resolves(self, paths):
        counted = frozenset(paths)
        counts = Counter()
        original = Path.resolve

        def resolve(path, *args, **kwargs):
            if path in counted:
                counts[path] += 1
            return original(path, *args, **kwargs)

        with patch.object(Path, 'resolve', resolve):
            yield counts

    def plan(self, **kwargs):
        return storage.plan_cleanup(self.cohort, repo=self.repo,
            active=kwargs.pop('active', set()), tracked=kwargs.pop('tracked', set()), **kwargs)

    def apply(self, plan, **kwargs):
        return storage.apply_cleanup(plan, repo=self.repo,
            active=kwargs.pop('active', set()), tracked=kwargs.pop('tracked', set()), **kwargs)

    def assert_resolved_once(self, paths, counts):
        self.assertEqual(counts, Counter({path: 1 for path in paths}))

    def test_prepare_paths_normalizes_once_and_reuses_prepared_identity(self):
        active, _ = self.global_tables()
        values = CountingIterable(active)
        with self.count_resolves(active) as counts:
            prepared = storage.prepare_paths(values)
            self.assertIsInstance(prepared, storage.PreparedPaths)
            self.assertIs(storage.prepare_paths(prepared), prepared)
        self.assertEqual(values.iterations, 1)
        self.assert_resolved_once(active, counts)
        self.assertEqual(prepared, frozenset(active))

    def test_prepare_paths_already_resolved_does_not_resolve_again(self):
        active, _ = self.global_tables()
        values = CountingIterable(active)
        with self.count_resolves(active) as counts:
            prepared = storage.prepare_paths(values, already_resolved=True)
        self.assertIsInstance(prepared, storage.PreparedPaths)
        self.assertEqual(values.iterations, 1)
        self.assertFalse(counts)
        self.assertEqual(prepared, frozenset(active))

    def test_plan_normalizes_global_tables_once_across_five_directories(self):
        for index in range(5):
            self.trial(index)
        active_paths, tracked_paths = self.global_tables()
        active, tracked = CountingIterable(active_paths), CountingIterable(tracked_paths)
        with self.count_resolves(active_paths + tracked_paths) as counts:
            plan = self.plan(active=active, tracked=tracked)
        self.assertEqual(plan['candidate_files'], 5)
        self.assertEqual(active.iterations, 1)
        self.assertEqual(tracked.iterations, 1)
        self.assert_resolved_once(active_paths + tracked_paths, counts)

    def test_apply_normalizes_global_tables_once_across_multiple_candidates(self):
        for index in range(5):
            self.trial(index, payload_count=3)
        plan = self.plan()
        active_paths, tracked_paths = self.global_tables()
        active, tracked = CountingIterable(active_paths), CountingIterable(tracked_paths)
        with self.count_resolves(active_paths + tracked_paths) as counts:
            result = self.apply(plan, active=active, tracked=tracked)
        self.assertEqual(result['deleted_files'], 15)
        self.assertEqual(active.iterations, 1)
        self.assertEqual(tracked.iterations, 1)
        self.assert_resolved_once(active_paths + tracked_paths, counts)

    def test_inspect_reuses_prepared_tables_without_iteration_or_resolution(self):
        class NoIterationPreparedPaths(storage.PreparedPaths):
            def __iter__(self):
                raise AssertionError('Prepared global paths must be reused without iteration')

        folder, payloads = self.trial(0, payload_count=3)
        active_paths, tracked_paths = self.global_tables()
        active = NoIterationPreparedPaths(active_paths + [payloads[0]])
        tracked = NoIterationPreparedPaths(tracked_paths + [payloads[1]])
        with self.count_resolves(active_paths + tracked_paths) as counts:
            self.assertIs(storage.prepare_paths(active), active)
            self.assertIs(storage.prepare_paths(tracked), tracked)
            inspection = storage.inspect_trial(folder, repo=self.repo,
                active=active, tracked=tracked)
        self.assertFalse(counts)
        self.assertEqual([entry['path'] for entry in inspection['entries']], [str(payloads[2])])
        self.assertEqual(inspection['skipped'], {'active': 1, 'git_tracked': 1})

    def test_inspect_only_looks_up_global_dependencies(self):
        folder, payloads = self.trial(0, payload_count=2)
        active_paths, _ = self.global_tables()
        dependencies = MembershipOnlySet(active_paths + [payloads[0]])
        inspection = storage.inspect_trial(folder, repo=self.repo,
            active=storage.PreparedPaths(), tracked=storage.PreparedPaths(),
            dependencies=dependencies)
        self.assertEqual(dependencies.iterations, 0)
        self.assertGreaterEqual(dependencies.lookups, len(payloads))
        self.assertEqual([entry['path'] for entry in inspection['entries']], [str(payloads[1])])
        self.assertEqual(inspection['skipped'], {'input_dependency': 1})

    def test_inspect_materializes_non_set_dependencies_once(self):
        folder, payloads = self.trial(0, payload_count=3)
        active_paths, _ = self.global_tables()
        # Reverse order would lose the second dependency if a single-use
        # generator were searched directly for each successive payload.
        dependencies = CountingIterable([payloads[1], payloads[0]] + active_paths)
        inspection = storage.inspect_trial(folder, repo=self.repo,
            active=storage.PreparedPaths(), tracked=storage.PreparedPaths(),
            dependencies=dependencies)
        self.assertEqual(dependencies.iterations, 1)
        self.assertEqual([entry['path'] for entry in inspection['entries']], [str(payloads[2])])
        self.assertEqual(inspection['skipped'], {'input_dependency': 2})

    def test_plan_and_apply_do_not_traverse_global_dependency_index(self):
        payloads = [self.trial(index)[1][0] for index in range(5)]
        active_paths, _ = self.global_tables()
        dependencies = MembershipOnlySet(active_paths + [payloads[0]])
        with patch.object(storage, 'referenced_paths', return_value=dependencies):
            plan = self.plan()
            result = self.apply(plan)
        self.assertEqual(dependencies.iterations, 0)
        self.assertGreaterEqual(dependencies.lookups, 9)
        self.assertEqual(plan['candidate_files'], 4)
        self.assertEqual(result['deleted_files'], 4)
        self.assertTrue(payloads[0].exists())

    def test_apply_scans_each_directory_once_with_many_candidates(self):
        folder, payloads = self.trial(0, payload_count=12)
        plan = self.plan()
        iterations = Counter()
        original = Path.iterdir

        def iterdir(path):
            if path == folder:
                iterations[path] += 1
            return original(path)

        with patch.object(Path, 'iterdir', iterdir):
            result = self.apply(plan)
        self.assertEqual(result['deleted_files'], len(payloads))
        self.assertEqual(iterations[folder], 1)

    def test_apply_rechecks_next_payload_stat_after_prior_unlink(self):
        self.trial(0, payload_count=3)
        plan = self.plan()
        plan['entries'].sort(key=lambda entry: entry['path'])
        first, changed, last = [Path(entry['path']) for entry in plan['entries']]
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == first:
                changed.write_text('changed after previous unlink\n'*20, encoding='utf-8')
            return result

        with patch.object(Path, 'unlink', unlink):
            result = self.apply(plan)
        self.assertEqual(result['deleted_files'], 2)
        self.assertFalse(first.exists())
        self.assertTrue(changed.exists())
        self.assertFalse(last.exists())
        self.assertEqual([row['path'] for row in result['skipped']], [str(changed)])

    def test_apply_rechecks_evidence_after_each_prior_unlink(self):
        for index, evidence_name in enumerate(('trial.json', 'metrics.json')):
            with self.subTest(evidence=evidence_name):
                folder, _ = self.trial(index, payload_count=3)
                plan = self.plan()
                # Each subcase reviews only its own directory, leaving the
                # previous subcase's deliberately changed evidence intact.
                plan['entries'] = sorted(
                    (entry for entry in plan['entries'] if entry['folder'] == str(folder)),
                    key=lambda entry: entry['path'])
                first, second, third = [Path(entry['path']) for entry in plan['entries']]
                original = Path.unlink

                def unlink(path, *args, **kwargs):
                    result = original(path, *args, **kwargs)
                    if path == first:
                        evidence = storage.load(folder/evidence_name)
                        evidence['changed_during_apply'] = True
                        storage.save(folder/evidence_name, evidence)
                    return result

                with patch.object(Path, 'unlink', unlink):
                    result = self.apply(plan)
                self.assertEqual(result['deleted_files'], 1)
                self.assertFalse(first.exists())
                self.assertTrue(second.exists())
                self.assertTrue(third.exists())
                self.assertEqual(len(result['skipped']), 2)

    def test_apply_refreshes_system_active_paths_before_later_unlink(self):
        self.trial(0, payload_count=3)
        plan = self.plan()
        plan['entries'].sort(key=lambda entry: entry['path'])
        first, newly_active, last = [Path(entry['path']) for entry in plan['entries']]
        clock = [0.0]
        original = Path.unlink

        def unlink(path, *args, **kwargs):
            result = original(path, *args, **kwargs)
            if path == first:
                clock[0] = 6.0  # Cross the process-snapshot refresh boundary.
            return result

        with patch.object(storage, 'active_paths', side_effect=[set(), {newly_active}]) as active, \
                patch.object(storage.time, 'monotonic', side_effect=lambda: clock[0]), \
                patch.object(Path, 'unlink', unlink):
            result = self.apply(plan, active=None)
        self.assertEqual(active.call_count, 2)
        self.assertEqual(result['deleted_files'], 2)
        self.assertFalse(first.exists())
        self.assertTrue(newly_active.exists())
        self.assertFalse(last.exists())

    def test_progress_exposes_stages_and_each_directory_or_candidate_boundary(self):
        for index in range(5):
            self.trial(index)
        events = []

        def progress(stage, **fields):
            events.append((stage, fields))

        plan = self.plan(progress=progress)
        self.assertIn('inspect_folders', {stage for stage, _ in events})
        completed_folders = {fields['completed_folders'] for _, fields in events
                             if 'completed_folders' in fields}
        self.assertTrue(set(range(1, 6)).issubset(completed_folders))
        self.assertTrue(all(fields.get('total_folders', 5) == 5 for _, fields in events))
        events.clear()
        result = self.apply(plan, progress=progress)
        self.assertEqual(result['deleted_files'], 5)
        self.assertIn('validate_entries', {stage for stage, _ in events})
        processed_entries = {fields['processed_entries'] for _, fields in events
                             if 'processed_entries' in fields}
        self.assertTrue(set(range(1, 6)).issubset(processed_entries))
        self.assertTrue(all(fields.get('total_entries', 5) == 5 for _, fields in events))

    def test_reporter_heartbeats_without_more_main_thread_notifications(self):
        heartbeat = threading.Event()

        class HeartbeatBuffer(io.StringIO):
            slow_messages = 0

            def write(self, text):
                written = super().write(text)
                if text.startswith('{') and json.loads(text).get('phase') == 'slow_setup':
                    self.slow_messages += 1
                    if self.slow_messages >= 2:
                        heartbeat.set()
                return written

        output = HeartbeatBuffer()
        with redirect_stderr(output), storage.ProgressReporter('plan', seconds=.01) as progress:
            progress('slow_setup', completed_folders=0)
            # Wait for the observed heartbeat, not a runtime-performance bound.
            # There are no further progress() calls while the phase is blocked.
            self.assertTrue(heartbeat.wait(1), 'No asynchronous heartbeat for the blocked phase')
        rows = [json.loads(line) for line in output.getvalue().splitlines()]
        slow = [row for row in rows if row['phase'] == 'slow_setup']
        self.assertGreaterEqual(len(slow), 2)
        self.assertTrue(all(row['event'] == 'storage_progress' for row in rows))
        self.assertTrue(all(row['completed_folders'] == 0 for row in slow))
        self.assertEqual(rows[-1]['phase'], 'complete')

    def test_cli_inventory_keeps_stdout_json_and_progress_on_stderr(self):
        _, payloads = self.trial(0)
        result = subprocess.run([sys.executable, str(Path(storage.__file__).resolve()),
            'inventory', '--repo', str(self.repo), '--out', str(self.cohort),
            '--progress-seconds', '.01'], capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        summary = json.loads(result.stdout)
        self.assertEqual(summary['schema'], 'tailcompile-v68-storage-inventory-1')
        self.assertEqual(summary['file_types']['packed_net']['files'], len(payloads))
        rows = [json.loads(line) for line in result.stderr.splitlines()]
        self.assertTrue(rows)
        self.assertTrue(all(row['event'] == 'storage_progress' for row in rows))
        self.assertTrue(all(row['operation'] == 'inventory' for row in rows))
        self.assertEqual(rows[-1]['phase'], 'complete')


if __name__ == '__main__':
    unittest.main()
