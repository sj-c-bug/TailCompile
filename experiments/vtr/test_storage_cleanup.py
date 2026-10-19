"""Safety regression tests; every deleted payload is synthetic temporary data."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import storage_cleanup as cleanup


class CleanupTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='cleanup-test-')
        self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name).resolve()/'repo'
        self.repo.mkdir()
        self.scratch=Path(self.tmp.name).resolve()/'scratch'
        self.scratch.mkdir()
        self.planfile=self.repo/'cleanup.json'
        self.args=argparse.Namespace(repo=self.repo,scratch=self.scratch,
            plan=self.planfile,min_age_hours=0)
        self.tracked=set();self.busy_dirs=set();self.busy_files=set()
        self.track_patch=patch.object(cleanup,'tracked',side_effect=lambda _:self.tracked)
        self.active_patch=patch.object(cleanup,'active',side_effect=lambda *_:(self.busy_dirs,self.busy_files,0))
        self.track_patch.start();self.active_patch.start()
        self.addCleanup(self.track_patch.stop);self.addCleanup(self.active_patch.stop)

    def file(self,relative,content='synthetic'):
        p=self.repo/relative;p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(content,encoding='utf-8');return p

    def run_folder(self,relative,status='success'):
        self.file(relative+'/manifest.json',json.dumps({'vpr_status':status}))
        return self.repo/relative

    def invoke(self,phase):
        with contextlib.redirect_stdout(io.StringIO()):getattr(cleanup,phase)(self.args)

    def test_only_old_completed_raw_payloads_removed(self):
        base='experiments/vtr/tailcompile_v3/sha/4x4/runs/test'
        folder=self.run_folder(base)
        remove=[self.file(base+'/'+n) for n in ('sha.route','sha.rr_graph.xml.gz','detail.rpt','debug.echo')]
        keep=[self.file(base+'/'+n) for n in ('sha.net','mapped.blif','input.place','x.fplace','vpr.out','stats.csv','model.pt','device_ir.json','architecture.xml')]
        keep.append(self.file('experiments/vtr/tailcompile_v3/unfinished/sha.route'))
        self.run_folder('experiments/vtr/tailcompile_v3/running','running')
        keep.append(self.file('experiments/vtr/tailcompile_v3/running/sha.route'))
        for relative in ('tailcompile_v5/runs/holdout_001/inputs/sha/full',
                'tailcompile_v3/pools/saved','tailcompile_v67/runs/current','tailcompile_v66/runs/current',
                'tailcompile_v3_training/model','runs/pilot_v1/sha'):
            self.run_folder('experiments/vtr/'+relative)
            keep.append(self.file('experiments/vtr/'+relative+'/sha.route'))
        self.invoke('plan')
        self.assertEqual(cleanup.load(self.planfile)['candidate_files'],4)
        self.assertTrue(all(p.exists() for p in remove)) # dry-run changes no payload
        self.invoke('apply')
        self.assertTrue(all(not p.exists() for p in remove))
        self.assertTrue(all(p.exists() for p in keep))
        self.assertTrue(folder.is_dir())
        self.assertEqual(len(cleanup.load(folder/'payload_pruned.json')['cleanups']),1)
        self.invoke('apply') # repeat cannot overwrite previous receipt/audit trail
        self.assertEqual(len(list(self.repo.glob('*.receipt.json'))),2)

    def test_tracked_active_changed_and_recent_files_are_preserved(self):
        base='experiments/vtr/tailcompile_v5/runs/old'
        self.run_folder(base)
        tracked=self.file(base+'/tracked.route');self.tracked.add(tracked)
        active=self.file(base+'/active.route');self.busy_files.add(active)
        changed=self.file(base+'/changed.route')
        newly_active=self.file(base+'/newly_active.route')
        self.invoke('plan')
        self.assertEqual(cleanup.load(self.planfile)['candidate_files'],2)
        changed.write_text('changed since plan',encoding='utf-8')
        self.busy_files.add(newly_active)
        self.invoke('apply')
        self.assertTrue(all(p.exists() for p in (tracked,active,changed,newly_active)))
        self.planfile.unlink();self.args.min_age_hours=24
        self.invoke('plan')
        self.assertEqual(cleanup.load(self.planfile)['candidate_files'],0)

    def test_scratch_requires_finished_archived_counterpart(self):
        self.run_folder('experiments/vtr/tailcompile_v3/sha/4x4/runs/old')
        good=self.scratch/'tailcompile_v3/sha/4x4/old/sha.route'
        bad=self.scratch/'tailcompile_v3/sha/4x4/no_archive/sha.route'
        for p in (good,bad):p.parent.mkdir(parents=True,exist_ok=True);p.write_text('synthetic')
        self.invoke('plan');self.invoke('apply')
        self.assertFalse(good.exists());self.assertTrue(bad.exists())

    def test_changed_evidence_blocks_deletion(self):
        base='experiments/vtr/tailcompile_v63/runs/old'
        folder=self.run_folder(base);payload=self.file(base+'/old.route')
        self.invoke('plan')
        (folder/'manifest.json').write_text(json.dumps({'vpr_status':'running'}))
        self.invoke('apply');self.assertTrue(payload.exists())

    def test_symlinks_and_hardlinks_are_preserved(self):
        base='experiments/vtr/tailcompile_v64/runs/old'
        folder=self.run_folder(base)
        original=self.file('valuable.route')
        link=folder/'linked.route';hard=folder/'hard.route'
        try:
            link.symlink_to(original)
            hard.hardlink_to(original)
        except OSError as error:self.skipTest(str(error))
        self.invoke('plan');self.invoke('apply')
        self.assertTrue(link.is_symlink());self.assertTrue(hard.exists());self.assertTrue(original.exists())

    def test_controller_and_cwd_scope_not_confused_by_versions(self):
        self.assertEqual(cleanup.controller_versions(['python3','tailcompile_v67/four_routes.py']),set())
        self.assertEqual(cleanup.controller_versions(['python3','train_tailcompile_v61.py']),{'tailcompile_v61'})
        self.assertEqual(cleanup.controller_versions(['python3','prepare_tailcompile_v5_inputs.py']),set())
        self.assertEqual(cleanup.controller_versions(['python3','run_tailcompile_v0.py','--variant','v3']),{'tailcompile_v0','tailcompile_v3'})
        old=self.repo/'experiments/vtr/tailcompile_v3'
        dirs={self.repo,self.repo.parent,old/'sha/runs/test'}
        self.assertEqual(cleanup.scoped_directories(dirs,self.repo,self.scratch),{old/'sha/runs/test'})

    @unittest.skipUnless(sys.platform.startswith('linux'),'requires real Linux /proc')
    def test_real_git_and_proc_protect_tracked_and_live_runs(self):
        self.track_patch.stop();self.active_patch.stop()
        subprocess.run(['git','init','-q',str(self.repo)],check=True)
        base='experiments/vtr/tailcompile_v63/runs'
        self.run_folder(base+'/complete');self.run_folder(base+'/live')
        tracked=self.file(base+'/complete/tracked.route')
        disposable=self.file(base+'/complete/disposable.route')
        live=self.file(base+'/live/live.route')
        subprocess.run(['git','-C',str(self.repo),'add',str(tracked)],check=True)
        child=subprocess.Popen([sys.executable,'-u','-c',
            "import time; print('ready',flush=True); time.sleep(30)"],
            cwd=live.parent,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'ready')
            self.invoke('plan');self.invoke('apply')
            self.assertTrue(tracked.exists());self.assertTrue(live.exists())
            self.assertFalse(disposable.exists())
        finally:
            child.terminate();child.wait(timeout=5);child.stdout.close()


if __name__=='__main__':unittest.main()
