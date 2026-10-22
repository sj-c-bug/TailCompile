"""Storage safety checks; only synthetic temporary payloads may be removed.

The direct API accepts explicit active/tracked snapshots for portable tests.
Linux CLI/lock/process inspection is separately exercised when available.
"""
from __future__ import annotations

import copy
import io
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import storage
from common import read, write


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='tailcompile-v68-storage-')
        self.addCleanup(self.temporary.cleanup)
        self.repo = Path(self.temporary.name).resolve()/'repo'
        self.repo.mkdir()
        self.cohort = self.repo/'experiments/vtr/tailcompile_v68/runs/fixture'
        self.cohort.mkdir(parents=True)
        self.active, self.tracked = set(), set()
        self.output = io.StringIO()
        self.redirect = redirect_stdout(self.output)
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def file(self, relative, content='synthetic disposable payload\n', *, root=None):
        path = (root or self.cohort)/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def trial(self, name='sha/w100/c0/s321', status='success', *, metrics=True):
        folder = self.cohort/'feedback/runs'/name
        folder.mkdir(parents=True, exist_ok=True)
        mapped = self.file('inputs/sha/sha.pre-vpr.blif', 'canonical BLIF\n')
        trial = dict(status=status, returncode=0 if status=='success' else 2,
            host=socket.gethostname(), parent_pid=999999999, completed_unix=1.,
            identity=dict(stage='route', command=['/missing/vpr', '/missing/arch.xml', str(mapped)],
                          hook_env={}, design='sha', width=100, seed=321))
        write(folder/'trial.json', trial)
        write(folder/'started.json', dict(host=socket.gethostname(), child_pid=999999999))
        if metrics:
            write(folder/'metrics.json', dict(status=status, cpd=1.25, wirelength=1000,
                phases=dict(global_placement_seconds=2.5), identity=trial['identity']))
        self.file('vpr.out', 'VPR succeeded\n', root=folder)
        return folder

    def plan(self):
        return storage.plan_cleanup(self.cohort, repo=self.repo,
            min_age_hours=0, active=self.active, tracked=self.tracked)

    def apply(self, plan):
        return storage.apply_cleanup(plan, repo=self.repo,
            active=self.active, tracked=self.tracked)

    def test_actual_net_postrouting_spellings_and_route_types(self):
        for name in ('sha.net', 'sha.net.post_routing', 'sha.net_postrouting',
                     'sha.net_post_routing', 'sha.post_routing.net', 'sha.route', 'sha.route.gz',
                     'critical_path.rpt', 'debug.echo'):
            with self.subTest(name=name):
                self.assertIsNotNone(storage.raw_kind(name))
        for name in ('sha.blif', 'logic.json', 'graph.pt', 'generator.pt',
                     'target.json', 'teacher.tsv', 'gp.blocks.csv', 'init.fplace', 'vpr.out'):
            with self.subTest(name=name):
                self.assertIsNone(storage.raw_kind(name))

    def test_inventory_counts_file_types_without_deleting_protected_inputs(self):
        payloads = [self.file('inputs/sha/sha.net', 'N'*17),
                    self.file('feedback/runs/sha/w100/c0/s321/sha.net.post_routing', 'P'*29),
                    self.file('models/pretrain/generator.pt', 'M'*31),
                    self.file('feedback/generation/sha/w100/c0/target.json', '{}\n')]
        expected = {p: p.read_bytes() for p in payloads}
        report = self.repo/'inventory_report.json'
        result = storage.inventory(self.cohort, repo=self.repo, report=report, top=3)
        self.assertEqual(result['logical_bytes'], sum(len(v) for v in expected.values()))
        self.assertEqual(result['file_types']['packed_net']['files'], 1)
        self.assertEqual(result['file_types']['packed_net']['logical_bytes'], 17)
        self.assertEqual(result['file_types']['postrouting_net']['logical_bytes'], 29)
        self.assertEqual(len(result['largest_files']), 3)
        self.assertEqual(result['largest_files'][0]['logical_bytes'], 31)
        self.assertEqual(read(report)['file_types'], result['file_types'])
        self.assertGreaterEqual(result['allocated_bytes'], 0)
        for path, content in expected.items():
            self.assertEqual(path.read_bytes(), content)

    def test_only_completed_route_payloads_pruned_and_metrics_stay_reusable(self):
        folder = self.trial()
        remove = [self.file(name, root=folder) for name in
                  ('sha.net', 'sha.net.post_routing', 'sha.net_postrouting',
                   'sha.route', 'critical_path.rpt', 'debug.echo')]
        keep = [self.file(name, root=folder) for name in
                ('input.net', 'sha.locked-input.net', 'input.place', 'init.fplace',
                 'gp.blocks.csv', 'stats.csv', 'device.xml', 'target.json')]
        records = {p: p.read_bytes() for p in [folder/'trial.json', folder/'metrics.json',
                                               folder/'started.json', folder/'vpr.out']}
        plan = self.plan()
        self.assertEqual(plan['candidate_files'], len(remove))
        self.assertTrue(all(p.exists() for p in remove))
        result = self.apply(plan)
        self.assertEqual(result['deleted_files'], len(remove))
        self.assertTrue(all(not p.exists() for p in remove))
        self.assertTrue(all(p.exists() for p in keep))
        self.assertTrue(folder.is_dir())
        for path, content in records.items():
            self.assertEqual(path.read_bytes(), content, str(path))
        self.assertEqual(self.plan()['candidate_files'], 0)

    def test_no_metrics_or_unfinished_trials_are_not_pruned(self):
        no_metrics = self.trial('sha/w100/c0/s321', metrics=False)
        running = self.trial('sha/w100/c0/s322', status='running')
        unrecorded = self.cohort/'feedback/runs/sha/w100/c0/s323'
        unrecorded.mkdir(parents=True)
        files = [self.file('sha.net', root=p) for p in (no_metrics, running, unrecorded)]
        plan = self.plan()
        self.assertEqual(plan['candidate_files'], 0)
        self.apply(plan)
        self.assertTrue(all(p.exists() for p in files))

    def test_mismatched_metrics_and_trial_status_are_preserved(self):
        folder = self.trial()
        payload = self.file('sha.net', root=folder)
        row = read(folder/'metrics.json')
        row['status'] = 'unroutable'
        write(folder/'metrics.json', row)
        plan = self.plan()
        self.assertEqual(plan['candidate_files'], 0)
        self.apply(plan)
        self.assertTrue(payload.exists())

    def test_inputs_generation_teachers_and_models_are_not_blanket_deleted(self):
        # These files matter to subsequent finetune/evaluate, not just run replay.
        names = ('inputs/sha/sha.net', 'inputs/sha/structure.net',
                 'feedback/generation/sha/w100/c0/init.fplace',
                 'feedback/generation/sha/w100/c0/target.json',
                 'feedback/generation/sha/w100/c0/generation.json',
                 'models/pretrain/generator.pt', 'models/finetune/generator.pt',
                 'graphs/sha/w100/graph.pt', 'device_cache/14x14/w100/rr_graph.xml.gz',
                 'teachers/sha/gp.blocks.csv')
        keep = [self.file(name) for name in names]
        folder = self.trial()
        raw = self.file('sha.net', root=folder)
        plan = self.plan()
        self.apply(plan)
        self.assertFalse(raw.exists())
        self.assertTrue(all(p.exists() for p in keep))

    def test_command_and_hook_input_paths_are_preserved(self):
        folder = self.trial()
        command_net = self.file('command_input.net', root=folder)
        hook_net = self.file('hook_input.net', root=folder)
        row = read(folder/'trial.json')
        row['identity']['command'] += ['--net_file', str(command_net)]
        row['identity']['hook_env']['TC_GP_INIT'] = str(hook_net)
        write(folder/'trial.json', row)
        # Some trial versions also contain a top-level command.
        row['command'] = row['identity']['command']
        write(folder/'trial.json', row)
        metrics = read(folder/'metrics.json')
        metrics['identity'] = row['identity']
        write(folder/'metrics.json', metrics)
        unused = self.file('sha.net', root=folder)
        self.apply(self.plan())
        self.assertTrue(command_net.exists())
        self.assertTrue(hook_net.exists())
        self.assertFalse(unused.exists())

    def test_another_trial_referencing_generated_net_protects_dependency(self):
        producer = self.trial()
        dependency = self.file('sha.net', root=producer)
        consumer = self.trial('sha/w100/c1/s321')
        row = read(consumer/'trial.json')
        row['identity']['command'] += ['--net_file', str(dependency)]
        row['command'] = row['identity']['command']
        write(consumer/'trial.json', row)
        metrics = read(consumer/'metrics.json')
        metrics['identity'] = row['identity']
        write(consumer/'metrics.json', metrics)
        disposable = self.file('sha.route', root=producer)
        self.apply(self.plan())
        self.assertTrue(dependency.exists())
        self.assertFalse(disposable.exists())

    def test_git_tracked_and_active_files_are_preserved(self):
        folder = self.trial()
        tracked = self.file('tracked.net', root=folder)
        opened = self.file('opened.net', root=folder)
        disposable = self.file('sha.net', root=folder)
        self.tracked.add(tracked.resolve())
        self.active.add(opened.resolve())
        plan = self.plan()
        self.assertEqual(plan['candidate_files'], 1)
        self.apply(plan)
        self.assertTrue(tracked.exists())
        self.assertTrue(opened.exists())
        self.assertFalse(disposable.exists())

    def test_symlink_and_hardlink_are_preserved(self):
        folder = self.trial()
        source = self.file('protected/sha.net')
        hard = folder/'hard.net'
        try:
            hard.hardlink_to(source)
        except OSError as error:
            self.skipTest(str(error))
        symlink = folder/'linked.net'
        made_symlink = True
        try:
            symlink.symlink_to(source)
        except OSError:
            made_symlink = False  # Windows may disallow creating symbolic links.
        self.apply(self.plan())
        self.assertTrue(hard.exists())
        self.assertTrue(source.exists())
        if made_symlink:
            self.assertTrue(symlink.is_symlink())

    def test_changed_payload_stat_since_plan_is_preserved(self):
        folder = self.trial()
        changed = self.file('sha.net', root=folder)
        valid = self.file('sha.route', root=folder)
        plan = self.plan()
        changed.write_text('a larger modified payload after planning\n')
        self.apply(plan)
        self.assertTrue(changed.exists())
        self.assertFalse(valid.exists())

    def test_changed_metrics_or_trial_evidence_blocks_deletion(self):
        for marker in ('metrics.json', 'trial.json'):
            with self.subTest(marker=marker):
                folder = self.trial(f'{marker}/w100/c0/s321')
                payload = self.file('sha.net', root=folder)
                plan = self.plan()
                row = read(folder/marker)
                row['changed_after_plan'] = True
                write(folder/marker, row)
                self.apply(plan)
                self.assertTrue(payload.exists())

    def test_newly_open_file_after_plan_is_preserved(self):
        folder = self.trial()
        payload = self.file('sha.net', root=folder)
        plan = self.plan()
        self.active.add(payload.resolve())
        self.apply(plan)
        self.assertTrue(payload.exists())

    def test_tampered_plan_cannot_escape_cohort(self):
        folder = self.trial()
        self.file('sha.net', root=folder)
        outside = self.file('valuable.net', root=self.repo)
        plan = self.plan()
        self.assertTrue(plan['entries'])
        entry = copy.deepcopy(plan['entries'][0])
        entry['path'] = str(outside)
        entry['stat'] = storage.snapshot(outside)
        plan['entries'] = [entry]
        try:
            self.apply(plan)
        except (ValueError, RuntimeError):
            pass
        self.assertTrue(outside.exists())

    def test_post_metrics_auto_prune_preserves_completed_evidence(self):
        folder = self.trial()
        raw = self.file('sha.net', root=folder)
        forbidden = self.file('input.net', root=folder)
        before = {p: p.read_bytes() for p in (folder/'metrics.json', folder/'trial.json', folder/'vpr.out')}
        with patch.object(storage, 'active_paths', return_value=set()), \
                patch.object(storage, 'tracked_files', return_value=set()):
            result = storage.prune_completed_trial(folder, repo=self.repo)
        self.assertEqual(result['deleted_files'], 1)
        self.assertFalse(raw.exists())
        self.assertTrue(forbidden.exists())
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)

    @unittest.skipUnless(sys.platform.startswith('linux'), 'requires real Linux /proc and Git')
    def test_real_linux_open_file_and_git_tracking_are_protected(self):
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        folder = self.trial()
        tracked = self.file('tracked.net', root=folder)
        opened = self.file('opened.net', root=folder)
        disposable = self.file('sha.net', root=folder)
        subprocess.run(['git', '-C', str(self.repo), 'add', str(tracked)], check=True)
        child = subprocess.Popen([sys.executable, '-u', '-c',
            "import sys,time; stream=open(sys.argv[1]); print('ready',flush=True); time.sleep(30)",
            str(opened)], cwd=folder, stdout=subprocess.PIPE, text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(), 'ready')
            # Live cwd conservatively protects the entire trial directory.
            plan = storage.plan_cleanup(self.cohort, repo=self.repo, min_age_hours=0)
            self.assertEqual(plan['candidate_files'], 0)
            self.assertTrue(tracked.exists())
            self.assertTrue(opened.exists())
        finally:
            child.terminate()
            child.wait(timeout=5)
            child.stdout.close()
        plan = storage.plan_cleanup(self.cohort, repo=self.repo, min_age_hours=0, active=set())
        storage.apply_cleanup(plan, repo=self.repo, active=set())
        self.assertTrue(tracked.exists())
        self.assertFalse(opened.exists())
        self.assertFalse(disposable.exists())

    @unittest.skipUnless(sys.platform.startswith('linux'), 'requires Linux flock')
    def test_cli_plan_refuses_active_cohort_lock(self):
        import fcntl
        subprocess.run(['git', 'init', '-q', str(self.repo)], check=True)
        folder = self.trial()
        payload = self.file('sha.net', root=folder)
        plan = Path(self.temporary.name)/'external_plan.json'
        with (self.cohort/'.lock').open('w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = subprocess.run([sys.executable, str(Path(storage.__file__).resolve()),
                'plan', '--repo', str(self.repo), '--out', str(self.cohort), '--plan', str(plan)],
                capture_output=True, text=True, timeout=10)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(plan.exists())
        self.assertTrue(payload.exists())


if __name__ == '__main__':
    unittest.main()
