"""Timing billing, frozen initializer verification and synchronized start tests."""
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

import timed_worker as worker
from bridge import read, write


class TimedWorkerTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='.test-timed-worker-',
                                               dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spec = dict(cpu=3, method='fast', vpr_command=['test-vpr'], expected_ap_sha256='frozen')
        self.affinity = dict(cpu=3, allowed_cpus=[3], thread_env={'OMP_NUM_THREADS': '1'})

    def test_only_compute_and_vpr_are_billed_and_environment_is_clean(self):
        clock = [0.]
        generated = dict(ap_init_sha256='frozen', ap_init='prior.tsv', compute_generation_seconds=3.,
                         excluded_setup_seconds=100., gate_wait_seconds=50., timing={'rrg_encoding': 1.})

        def generate(spec, folder):
            clock[0] += 153.
            return generated

        def vpr(command, env, timeout, folder):
            self.assertEqual(env['TC_GP_INIT'], 'prior.tsv')
            self.assertNotIn('TC_GP_WORKER', env)
            self.assertEqual(timeout, 7197.)
            clock[0] += 20.
            return types.SimpleNamespace(returncode=0, timed_out=False, launched=True)

        with patch.object(worker, '_pin', return_value=self.affinity), \
             patch.object(worker, '_generate', side_effect=generate), \
             patch.object(worker, '_run_vpr', side_effect=vpr), \
             patch.object(worker.time, 'perf_counter', lambda: clock[0]), \
             patch.dict(os.environ, {'TC_GP_WORKER': 'stale-worker'}):
            self.assertEqual(worker.run_spec(self.spec, self.root), 0)
        cost = read(self.root / 'cost.json')
        self.assertEqual(cost['total_seconds'], 23.)
        self.assertEqual(cost['vpr_seconds'], 20.)
        self.assertEqual(cost['worker_wall_seconds'], 173.)
        self.assertEqual(cost['excluded_setup_seconds'], 100.)
        self.assertEqual(cost['gate_wait_seconds'], 50.)

    def test_default_bills_vpr_and_waits_for_the_same_start_gate(self):
        clock = [0.]

        def gate(spec, folder):
            clock[0] += 100.
            return 100.

        def vpr(command, env, timeout, folder):
            self.assertNotIn('TC_GP_INIT', env)
            clock[0] += 7.
            return types.SimpleNamespace(returncode=2, timed_out=False, launched=True)

        with patch.object(worker, '_pin', return_value=self.affinity), \
             patch.object(worker, '_wait_gate', side_effect=gate), \
             patch.object(worker, '_generate', side_effect=AssertionError('Default must not load GNN')), \
             patch.object(worker, '_run_vpr', side_effect=vpr), \
             patch.object(worker.time, 'perf_counter', lambda: clock[0]), \
             patch.dict(os.environ, {'TC_GP_INIT': 'stale-prior'}):
            self.assertEqual(worker.run_spec(dict(self.spec, method='default'), self.root), 2)
        cost = read(self.root / 'cost.json')
        self.assertEqual(cost['total_seconds'], 7.)
        self.assertEqual(cost['compute_generation_seconds'], 0.)
        self.assertEqual(cost['gate_wait_seconds'], 100.)

    def test_changed_inference_cannot_run_vpr(self):
        with patch.object(worker, '_pin', return_value=self.affinity), \
             patch.object(worker, '_generate', return_value={'ap_init_sha256': 'changed'}), \
             patch.object(worker, '_run_vpr', side_effect=AssertionError('VPR must not run')):
            with self.assertRaisesRegex(ValueError, 'differs from the frozen'):
                worker.run_spec(self.spec, self.root)

    def test_start_gate_writes_ready_and_honors_abort(self):
        gate = self.root / 'go.json'
        spec = dict(self.spec, start_gate=str(gate))
        write(gate, {'go': True})
        self.assertGreaterEqual(worker._wait_gate(spec, self.root), 0.)
        self.assertTrue(read(self.root / 'ready.json')['ready'])
        write(gate, {'abort': 'incomplete lane'})
        with self.assertRaisesRegex(RuntimeError, 'incomplete lane'):
            worker._wait_gate(spec, self.root)

    def test_measured_timeout_subtracts_compute_not_setup_and_saves_cost(self):
        clock = [0.]
        generated = dict(ap_init_sha256='frozen', ap_init='prior.tsv', compute_generation_seconds=3.,
                         excluded_setup_seconds=100., gate_wait_seconds=50., timing={'rrg_encoding': 1.})

        def generate(spec, folder):
            clock[0] += 153.
            return generated

        def vpr(command, env, timeout, folder):
            self.assertEqual(timeout, 7.)
            clock[0] += timeout
            return types.SimpleNamespace(returncode=124, timed_out=True, launched=True)

        with patch.object(worker, '_pin', return_value=self.affinity), \
             patch.object(worker, '_generate', side_effect=generate), \
             patch.object(worker, '_run_vpr', side_effect=vpr), \
             patch.object(worker.time, 'perf_counter', lambda: clock[0]):
            self.assertEqual(worker.run_spec(dict(self.spec, measured_timeout=10.), self.root), 124)
        cost = read(self.root / 'cost.json')
        self.assertTrue(cost['timed_out'])
        self.assertEqual(cost['total_seconds'], 10.)
        self.assertEqual(cost['excluded_setup_seconds'], 100.)
        self.assertEqual(cost['vpr_remaining_budget_seconds'], 7.)

    def test_budget_exhausted_by_compute_does_not_launch_vpr(self):
        generated = dict(ap_init_sha256='frozen', ap_init='prior.tsv', compute_generation_seconds=11.,
                         excluded_setup_seconds=100., gate_wait_seconds=50., timing={'rrg_encoding': 1.})
        with patch.object(worker, '_pin', return_value=self.affinity), \
             patch.object(worker, '_generate', return_value=generated), \
             patch.object(worker.subprocess, 'Popen', side_effect=AssertionError('No VPR budget left')):
            self.assertEqual(worker.run_spec(dict(self.spec, measured_timeout=10.), self.root), 124)
        self.assertFalse(read(self.root / 'cost.json')['vpr_launched'])

    def test_real_vpr_deadline_preserves_diagnostics_and_cost(self):
        diagnostic = self.root / 'vpr.out'
        code = ('from pathlib import Path; import sys,time; '
                'Path(sys.argv[1]).write_text("diagnostic before timeout\\n"); time.sleep(10)')
        spec = dict(self.spec, method='default', measured_timeout=1.,
                    vpr_command=[sys.executable, '-c', code, str(diagnostic)])
        with patch.object(worker, '_pin', return_value=self.affinity):
            self.assertEqual(worker.run_spec(spec, self.root), 124)
        self.assertIn('before timeout', diagnostic.read_text())
        cost = read(self.root / 'cost.json')
        self.assertTrue(cost['timed_out'])
        self.assertTrue(cost['vpr_launched'])
        self.assertLess(cost['vpr_seconds'], 8.)

    def test_timeout_terminates_entire_posix_child_group_and_restores_handlers(self):
        process = Mock(pid=12345)
        process.wait.side_effect = [subprocess.TimeoutExpired(['vpr'], .1), -15, -15]
        process.poll.return_value = -15
        previous_term, previous_int = signal.getsignal(signal.SIGTERM), signal.getsignal(signal.SIGINT)
        with patch.object(worker.os, 'name', 'posix'), \
             patch.object(worker.signal, 'SIGKILL', 9, create=True), \
             patch.object(worker.os, 'killpg', create=True) as killpg, \
             patch.object(worker.subprocess, 'Popen', return_value=process) as popen, \
             patch.object(worker, 'write'):
            result = worker._run_vpr(['vpr'], {}, .1, self.root)
        self.assertEqual(result.returncode, 124)
        self.assertTrue(result.timed_out)
        self.assertTrue(popen.call_args.kwargs['start_new_session'])
        self.assertEqual([args.args for args in killpg.call_args_list],
                         [(12345, signal.SIGTERM), (12345, getattr(signal, 'SIGKILL', 9))])
        self.assertEqual(signal.getsignal(signal.SIGTERM), previous_term)
        self.assertEqual(signal.getsignal(signal.SIGINT), previous_int)


if __name__ == '__main__':
    unittest.main()
