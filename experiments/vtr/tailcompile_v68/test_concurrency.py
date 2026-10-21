"""Scheduling-only regression tests; no VPR jobs or GPU work are launched."""
from __future__ import annotations

import ast
import concurrent.futures as cf
import copy
from contextlib import redirect_stderr, redirect_stdout
import inspect
import io
import os
from pathlib import Path
import pickle
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import experiment
import scheduling
from common import sha, write

HERE = Path(__file__).resolve().parent
NATIVE_BASH = shutil.which('bash') if os.name != 'nt' else None


class RecordingExecutor:
    """Run submitted tiny tasks immediately, retain real Futures for as_completed."""
    instances = []

    def __init__(self, max_workers):
        self.max_workers = max_workers
        self.calls = []
        self.__class__.instances.append(self)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def submit(self, function, task):
        self.calls.append((function, task))
        future = cf.Future()
        try:
            future.set_result(function(task))
        except Exception as error:
            future.set_exception(error)
        return future


class ConcurrencyTest(unittest.TestCase):
    def parse(self, *arguments):
        return experiment.parse_args(['status', '--out', 'unused-status-test-output',
                                      *arguments])

    def test_python_defaults_are_sixteen_for_all_cpu_stages(self):
        args = self.parse()
        self.assertEqual((args.jobs, args.prepare_jobs, args.rr_jobs), (16, 16, 16))

    def test_general_override_is_inherited_by_both_specialized_budgets(self):
        args = self.parse('--jobs', '32')
        self.assertEqual((args.jobs, args.prepare_jobs, args.rr_jobs), (32, 32, 32))

    def test_specific_override_only_changes_requested_stage(self):
        args = self.parse('--jobs', '32', '--rr-jobs', '4')
        self.assertEqual((args.jobs, args.prepare_jobs, args.rr_jobs), (32, 32, 4))
        args = self.parse('--jobs', '32', '--prepare-jobs', '6')
        self.assertEqual((args.jobs, args.prepare_jobs, args.rr_jobs), (32, 6, 32))
        args = self.parse('--prepare-jobs', '3', '--rr-jobs', '2')
        self.assertEqual((args.jobs, args.prepare_jobs, args.rr_jobs), (16, 3, 2))

    def test_nonpositive_overrides_are_rejected(self):
        for flag in ('--jobs', '--prepare-jobs', '--rr-jobs'):
            for value in ('0', '-1'):
                with self.subTest(flag=flag, value=value):
                    with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                        self.parse(flag, value)
                    self.assertEqual(error.exception.code, 2)

    def test_bash_launcher_declares_the_same_default_and_inheritance(self):
        source = (HERE / 'run_pipeline.sh').read_text(encoding='utf-8')
        self.assertIn('${TAILCOMPILE_V68_JOBS:-16}', source)
        self.assertIn('${TAILCOMPILE_V68_PREPARE_JOBS:-$jobs}', source)
        self.assertIn('${TAILCOMPILE_V68_RR_JOBS:-$jobs}', source)
        self.assertIn('--jobs "$jobs" --prepare-jobs "$prepare_jobs"', source)
        self.assertIn('--rr-jobs "$rr_jobs"', source)

    @unittest.skipUnless(NATIVE_BASH, 'requires native Linux/macOS Bash; Windows CLI tests still run')
    def test_actual_bash_launch_arguments_default_override_and_specific_override(self):
        # A shell function captures python arguments.  status avoids flock and
        # does not invoke a real Python experiment, VPR process or GPU task.
        script = '''
python3() { printf '%s\\n' "$@" > "$TC_TEST_CAPTURE"; }
source "$1" status
'''
        with tempfile.TemporaryDirectory(prefix='v68-shell-concurrency-') as tmp:
            root = Path(tmp)
            for override, expected in [({}, (16, 16, 16)),
                    ({'TAILCOMPILE_V68_JOBS': '32'}, (32, 32, 32)),
                    ({'TAILCOMPILE_V68_JOBS': '32', 'TAILCOMPILE_V68_RR_JOBS': '4'}, (32, 32, 4))]:
                env = {key: value for key, value in os.environ.items()
                       if not key.startswith('TAILCOMPILE_V68_')}
                capture = root / 'arguments.txt'
                env.update(TAILCOMPILE_REPO=str(root),
                           TAILCOMPILE_V68_OUT=str(root / 'output'), TC_TEST_CAPTURE=str(capture),
                           **override)
                result = subprocess.run([NATIVE_BASH, '-c', script, 'test-shell',
                                         str(HERE / 'run_pipeline.sh')], env=env,
                                        capture_output=True, text=True, timeout=20)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                values = capture.read_text().splitlines()
                actual = tuple(int(values[values.index(flag) + 1]) for flag in
                               ('--jobs', '--prepare-jobs', '--rr-jobs'))
                self.assertEqual(actual, expected)

    def test_graph_builder_uses_independent_processes_and_preserves_sorted_results(self):
        tasks = [{'design': 'zeta', 'width': 104}, {'design': 'alpha', 'width': 108},
                 {'design': 'alpha', 'width': 100}]
        for jobs in (16, 32):
            RecordingExecutor.instances.clear()
            with patch.object(experiment.cf, 'ProcessPoolExecutor', RecordingExecutor), \
                    patch.object(experiment, 'graph_worker', side_effect=lambda task: task):
                result = experiment.build_graphs(tasks, jobs)
            executor = RecordingExecutor.instances[-1]
            self.assertEqual(executor.max_workers, jobs)
            self.assertEqual(len(executor.calls), 3)
            self.assertEqual([(r['design'], r['width']) for r in result],
                             [('alpha', 100), ('alpha', 108), ('zeta', 104)])
        # Spawn-based platforms must be able to resolve the worker by module.
        self.assertNotIn('<locals>', experiment.graph_worker.__qualname__)
        self.assertIs(pickle.loads(pickle.dumps(experiment.graph_worker)), experiment.graph_worker)

    def test_one_graph_failure_does_not_cancel_other_submitted_graph_results(self):
        attempted = []

        def worker(task):
            attempted.append(task)
            if task == 1:
                raise ValueError('synthetic failed graph')
            return dict(design='synthetic', width=task)

        with patch.object(experiment.cf, 'ProcessPoolExecutor', RecordingExecutor), \
                patch.object(experiment, 'graph_worker', side_effect=worker), \
                redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(RuntimeError, '1 graph builds failed; valid caches retained'):
                experiment.build_graphs([0, 1, 2], 16)
        self.assertEqual(attempted, [0, 1, 2])

    def test_prepare_routes_feature_graphs_to_prepare_budget_not_serial_loop(self):
        tree = ast.parse(inspect.getsource(experiment.prepare))
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == 'build_graphs']
        self.assertEqual(len(calls), 1)
        budget = calls[0].args[1]
        self.assertIsInstance(budget, ast.Attribute)
        self.assertEqual(budget.attr, 'prepare_jobs')
        self.assertIsInstance(budget.value, ast.Name)
        self.assertEqual(budget.value.id, 'a')

    def test_scheduler_compatibility_does_not_relax_scientific_inputs(self):
        previous = dict(identity={'binary_sha256': 'same-vpr'}, tile_size=4,
            calibration_seeds=[301, 302], width_step=4, timing_guard=.03,
            source={'experiment.py': 'old-experiment', 'data.py': 'same-data',
                    'common.py': 'same-common'})
        expected = copy.deepcopy(previous)
        expected['source'].update({'experiment.py': 'new-experiment',
                                   'scheduling.py': sha(Path(scheduling.__file__))})
        original = copy.deepcopy(previous)
        with tempfile.TemporaryDirectory(prefix='v68-scheduling-manifest-') as tmp:
            write(Path(tmp) / 'scheduling_compatibility.json', dict(
                legacy_experiment_sha256='old-experiment',
                updated_experiment_sha256='new-experiment',
                scheduling_module_sha256=sha(Path(scheduling.__file__))))
            with patch.object(scheduling, 'HERE', Path(tmp)):
                normalized, allowed = scheduling.compatible_protocol(previous, expected)
                self.assertTrue(allowed)
                self.assertEqual(normalized, previous)
                self.assertEqual(previous, original)
                for key, changed in [('tile_size', 8), ('calibration_seeds', [901, 902]),
                                     ('width_step', 8), ('timing_guard', .1),
                                     ('identity', {'binary_sha256': 'different-vpr'})]:
                    modified = copy.deepcopy(expected)
                    modified[key] = changed
                    self.assertFalse(scheduling.compatible_protocol(previous, modified)[1], key)
                modified = copy.deepcopy(expected)
                modified['source']['data.py'] = 'different-features'
                self.assertFalse(scheduling.compatible_protocol(previous, modified)[1])
                modified = copy.deepcopy(expected)
                modified['source']['experiment.py'] = 'unknown-new-experiment'
                self.assertFalse(scheduling.compatible_protocol(previous, modified)[1])


if __name__ == '__main__':
    unittest.main()
