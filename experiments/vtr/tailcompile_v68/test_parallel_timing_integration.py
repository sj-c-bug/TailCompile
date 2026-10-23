"""Real Linux pinned subprocesses and synchronized/resumable waves; no VPR."""
import os
from pathlib import Path
import subprocess
import socket
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import parallel_timing as p
import stop_timing
from common import read, sha, write


@unittest.skipUnless(hasattr(os, 'sched_setaffinity'), 'Linux CPU affinity required')
class PinnedTimingIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.a = SimpleNamespace(out=self.root, run_name='paired_test', timing_jobs=2,
            cpu_list=None, timeout=15, heartbeat=5, identity={'test': 'identity'},
            retry_failed=False, retry_timeouts=False, confirm_remote_stopped=False)
        self.env = patch.dict(os.environ, TAILCOMPILE_V68_PRUNE_COMPLETED='0',
                              TAILCOMPILE_V68_MIN_FREE_GIB='0')
        self.env.start()
        self.addCleanup(self.env.stop)
        if p.hardware(2, condition_count=3)['concurrency'] < 2:
            self.skipTest('Two CPU equivalents/physical cores needed')
        write(self.root/'protocol.json', dict(identity=self.a.identity))
        inputs = self.root/'inputs/tiny'
        inputs.mkdir(parents=True)
        (inputs/'mapped.blif').write_text('test mapped input')
        write(inputs/'audit.json', dict(timing=dict(logic_preprocessing=.2,
                                                   device_preprocessing=100)))
        write(inputs/'structure_run/trial.json', dict(wall_seconds=.1))
        context = dict(design='tiny', width=100, family='tiny', split='final_test',
            mapped_blif=str(inputs/'mapped.blif'), device_width=10, audit=str(inputs/'audit.json'))
        write(self.root/'contexts.json', dict(contexts=[context]))
        generation = self.root/'evaluation/generation/tiny/w100'
        generation.mkdir(parents=True)
        (generation/'init.ap.tsv').write_text('test AP prior')
        write(generation/'generation.json', dict(ap_init=str(generation/'init.ap.tsv'),
            ap_init_sha256=sha(generation/'init.ap.tsv'),
            signature=dict(checkpoint_sha256='test-checkpoint'),
            warm_inference_lowering_seconds=.02, cold_from_prepared_graph_seconds=.05,
            timing=dict(logic_forward=.01, fplace_lowering=.01)))
        write(self.root/'evaluation_plan.json', dict(designs=[dict(design='tiny', width=100)],
            seeds=[341,342,343], methods=p.METHODS, checkpoint_sha256='test-checkpoint'))
        write(self.root/'evaluation_report.json', dict(all_route_outcomes_known=True))
        for seed in (341,342,343):
            for method in p.METHODS:
                write(self.root/'evaluation/runs/tiny/w100'/f's{seed}'/method/'metrics.json',
                      dict(status='success'))
        code = ('import os,time; time.sleep(.08); '
                'print("TAILCOMPILE_V68_GP_INIT_CONSUMED"); '
                'print("TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1"); '
                'print("Final critical path delay (least slack): 10"); '
                'print("Total wirelength: 100"); print("VPR succeeded")')
        self.command = patch.object(p.e, 'vpr_command', return_value=[sys.executable, '-u', '-c', code])
        self.command.start()
        self.addCleanup(self.command.stop)

    def test_real_affinity_halfwave_barrier_and_completed_wave_resume(self):
        write(self.root/'timing_report.json', dict(serial='must stay unchanged'))
        result = p.run(self.a)
        self.assertTrue(result['complete'])
        self.assertTrue(result['all_route_outcomes_known'])
        self.assertFalse(result['serial_results_included'])
        self.assertEqual(result['waves_complete'], 2)
        self.assertEqual(result['status_counts'], {'success': 6})
        self.assertEqual({r['effective_lanes'] for r in result['comparisons']}, {1,2})
        root = p.run_root(self.a)
        waves = read(root/'schedule.json')['waves']
        for wave in waves:
            data = p.completed_wave(root, wave)
            first, second = [], []
            for lane in wave['lanes']:
                rows = [data['results'][f'{half}:{lane["lane"]}'] for half in (0,1)]
                self.assertEqual([r['method'] for r in rows], lane['order'])
                self.assertTrue(all(r['cpu_affinity']['allowed_cpus'] == [lane['cpu']] for r in rows))
                self.assertTrue(all(set(r['cpu_affinity']['thread_env'].values()) == {'1'} for r in rows))
                first.append(rows[0]['completed_unix'])
                second.append(rows[1]['started_unix'])
            self.assertLessEqual(max(first), min(second))
        before = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                  for path in (root/'waves').rglob('*.json')}
        with patch.object(p, 'run_trial', side_effect=AssertionError('Must reuse complete waves')):
            p.run(self.a)
        after = {str(path): (path.read_bytes(), path.stat().st_mtime_ns)
                 for path in (root/'waves').rglob('*.json')}
        self.assertEqual(before, after)
        self.assertEqual(read(self.root/'timing_report.json'), dict(serial='must stay unchanged'))

    def test_partial_resume_keeps_first_wave_and_archives_only_partial_wave(self):
        p.run(self.a)
        root = p.run_root(self.a)
        waves = read(root/'schedule.json')['waves']
        first = p.wave_folder(root, waves[0])/'wave.json'
        original = (first.read_bytes(), first.stat().st_mtime_ns)
        partial = p.wave_folder(root, waves[1])/'wave.json'
        data = read(partial)
        data['complete'] = False
        write(partial, data)
        p.run(self.a)
        self.assertEqual((first.read_bytes(), first.stat().st_mtime_ns), original)
        self.assertEqual(len(list((root/'_attempts').glob('*/w0001/wave.json'))), 1)
        self.assertTrue(read(root/'parallel_report.json')['complete'])

    def test_safe_stop_refuses_unrelated_process_and_preserves_completed_files(self):
        folder = self.root/'timing/serial_001/tiny/w100/s341/default'
        folder.mkdir(parents=True)
        script = 'import time; time.sleep(30)'
        target = subprocess.Popen([sys.executable, '-c', script], cwd=folder, start_new_session=True)
        unrelated = subprocess.Popen([sys.executable, '-c', script], cwd=self.root, start_new_session=True)
        same_folder = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(31)'],
                                       cwd=folder, start_new_session=True)
        self.addCleanup(lambda: target.poll() is None and target.kill())
        self.addCleanup(lambda: unrelated.poll() is None and unrelated.kill())
        self.addCleanup(lambda: same_folder.poll() is None and same_folder.kill())
        write(folder/'started.json', dict(host=socket.gethostname(), parent_pid=os.getpid(),
            child_pid=target.pid, command=[sys.executable, '-c', script]))
        evidence = self.root/'timing/serial_001/old_success/trial.json'
        write(evidence, dict(status='success'))
        time.sleep(.05)
        runners, workers = stop_timing.targets(self.root, 'serial', 'serial_001', 'paired_001')
        self.assertIn(target.pid, [r['pid'] for r in workers])
        self.assertNotIn(unrelated.pid, [r['pid'] for r in workers])
        self.assertNotIn(same_folder.pid, [r['pid'] for r in workers])
        stop_timing.stop(self.root, apply=True)
        target.wait(timeout=5)
        self.assertIsNone(unrelated.poll())
        self.assertIsNone(same_folder.poll())
        self.assertEqual(read(evidence), dict(status='success'))
        unrelated.terminate()
        unrelated.wait(timeout=5)
        same_folder.terminate()
        same_folder.wait(timeout=5)

    def test_stopper_matches_exact_cohort_phase_and_default_run_name(self):
        record = dict(cwd=str(Path(__file__).resolve().parent), command=[
            sys.executable, '-u', str(p.HERE/'experiment.py'), 'timing', '--out', str(self.root)])
        self.assertTrue(stop_timing.coordinator(record, self.root, 'serial', 'serial_001', 'paired_001'))
        self.assertFalse(stop_timing.coordinator(record, self.root, 'serial', 'another_run', 'paired_001'))
        self.assertFalse(stop_timing.coordinator(record, self.root/'another_cohort', 'serial', 'serial_001', 'paired_001'))
        self.assertFalse(stop_timing.coordinator(record, self.root, 'parallel', 'serial_001', 'paired_001'))
        record['command'][3] = 'evaluate'
        self.assertFalse(stop_timing.coordinator(record, self.root, 'serial', 'serial_001', 'paired_001'))


if __name__ == '__main__':
    unittest.main()
