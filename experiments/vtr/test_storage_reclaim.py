"""Old-version reclamation safety tests; every payload is synthetic temporary data."""
from __future__ import annotations

import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest

import storage_reclaim as reclaim


class StorageReclaimTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tailcompile-old-reclaim-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.repo = self.root/'repo'
        self.scratch = self.root/'scratch'
        self.repo.mkdir()
        self.scratch.mkdir()
        subprocess.run(['git','init','-q',str(self.repo)],check=True)
        self.active = (set(), set(), 0)
        self.tracked = set()
        self.references = set()
        self.redirect = redirect_stdout(io.StringIO())
        self.redirect.__enter__()
        self.addCleanup(self.redirect.__exit__, None, None, None)

    def file(self, relative, content='synthetic disposable payload\n', *, root=None):
        path = (root or self.repo)/relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')
        return path

    def trial(self, relative='experiments/vtr/tailcompile_v3/sha/4x4/runs/closed',
              status='success', kind='manifest'):
        folder = self.repo/relative
        folder.mkdir(parents=True, exist_ok=True)
        data = dict(vpr_status=status, status=status, host=socket.gethostname(),
                    command=['/missing/vpr','/missing/arch.xml','/missing/mapped.blif'])
        if kind=='trial':
            data['identity'] = dict(stage='route', command=data['command'])
        self.file('trial.json' if kind=='trial' else 'manifest.json', json.dumps(data), root=folder)
        self.file('vpr.out','VPR succeeded\n',root=folder)
        return folder

    def plan(self, **kwargs):
        options = dict(include_netlists=True, deduplicate_netlists=False,
                       min_age_hours=0, active=self.active, tracked=self.tracked,
                       references=self.references)
        options.update(kwargs)
        return reclaim.plan_reclaim(self.repo,self.scratch,**options)

    def apply(self, plan, **kwargs):
        options = dict(repo=self.repo,scratch=self.scratch,active=self.active,
                       tracked=self.tracked,references=self.references)
        options.update(kwargs)
        return reclaim.apply_reclaim(plan,**options)

    def paths(self, plan, action='delete'):
        return {Path(row['path']) for row in plan['entries'] if row['action']==action}

    def test_completed_old_versions_including_v66_v67_are_in_scope(self):
        remove = []
        for version in ('v3','v5','v61','v63','v66','v67'):
            folder = self.trial(f'experiments/vtr/tailcompile_{version}/runs/closed',
                                kind='trial' if version in ('v66','v67') else 'manifest')
            remove.extend(self.file(name,root=folder) for name in
                          ('sha.net','sha.net.post_routing','sha.route','detail.rpt'))
        plan = self.plan()
        self.assertEqual(self.paths(plan),set(remove))
        self.assertTrue(all(p.exists() for p in remove))
        self.apply(plan)
        self.assertTrue(all(not p.exists() for p in remove))

    def test_current_v68_inputs_models_ir_candidates_and_evidence_are_preserved(self):
        keep = []
        for relative in ('tailcompile_v68/runs/current/feedback/runs/sha/w100/c0/s321',
                'tailcompile_v5/runs/old/inputs/sha/full','tailcompile_v3_training/bundle/designs/sha',
                'tailcompile_v63/runs/old/models','tailcompile_v63/runs/old/ir',
                'tailcompile_v63/runs/old/generation/sha'):
            folder = self.trial('experiments/vtr/'+relative)
            keep.extend(self.file(n,root=folder) for n in ('source.net','source.net.post_routing','source.route'))
        folder = self.trial()
        keep.extend(self.file(n,root=folder) for n in ('input.net','sha.locked-input.net',
            'init.fplace','gp.blocks.tsv','gp.nets.tsv','target.json','stats.csv','generator.pt'))
        records = {p:p.read_bytes() for p in (folder/'manifest.json',folder/'vpr.out')}
        self.apply(self.plan())
        self.assertTrue(all(p.exists() for p in keep))
        for path,content in records.items():
            self.assertEqual(path.read_bytes(),content)

    def test_net_deletion_requires_explicit_opt_in(self):
        folder = self.trial()
        net = self.file('sha.net',root=folder)
        post = self.file('sha.net.post_routing',root=folder)
        route = self.file('sha.route',root=folder)
        self.apply(self.plan(include_netlists=False))
        self.assertTrue(net.exists())
        self.assertTrue(post.exists())
        self.assertFalse(route.exists())

    def test_unreferenced_legacy_archive_shadow_inputs_can_be_deleted_with_exact_evidence(self):
        folder=self.trial()
        scratch=self.scratch/'tailcompile_v3/sha/4x4/closed'
        scratch.mkdir(parents=True)
        scratch_input=self.file('sha.locked-input.net','canonical frozen net\n',root=scratch)
        canonical=self.file('experiments/vtr/runs/pilot_v1/sha/full/sha.net','canonical frozen net\n')
        shadow=[self.file(name,'canonical frozen net\n',root=folder)
                for name in ('sha.locked-input.net','sha.input-packed.net')]
        manifest=json.loads((folder/'manifest.json').read_text())
        manifest.update(scratch=str(scratch),archive=str(folder),
            input_packed_net=str(canonical),input_packed_net_sha256=reclaim.digest(canonical),
            command=['/missing/vpr','--net_file',str(scratch_input)])
        self.file('manifest.json',json.dumps(manifest),root=folder)
        plan=self.plan(references=None)
        self.assertTrue(set(shadow).issubset(self.paths(plan)))
        self.apply(plan,references=None)
        self.assertTrue(all(not p.exists() for p in shadow))
        self.assertEqual(scratch_input.read_text(),'canonical frozen net\n')
        self.assertEqual(canonical.read_text(),'canonical frozen net\n')

    def test_archive_shadow_input_referenced_by_current_case_remains_protected(self):
        folder=self.trial()
        scratch=self.scratch/'tailcompile_v3/sha/4x4/closed'
        scratch.mkdir(parents=True)
        shadow=self.file('sha.locked-input.net','frozen net\n',root=folder)
        manifest=json.loads((folder/'manifest.json').read_text())
        manifest.update(scratch=str(scratch),archive=str(folder),
            input_packed_net='/source/original.net',input_packed_net_sha256='frozen-sha')
        self.file('manifest.json',json.dumps(manifest),root=folder)
        self.file('experiments/vtr/tailcompile_v68/runs/current/contexts.json',
                  json.dumps({'contexts':[{'packed_net':str(shadow)}]}))
        self.apply(self.plan(references=None),references=None)
        self.assertTrue(shadow.exists())

    def test_referenced_net_is_kept_even_when_it_is_a_completed_output(self):
        first = self.trial()
        net = self.file('sha.net',root=first)
        post = self.file('sha.net.post_routing',root=first)
        self.references.add(net)
        self.apply(self.plan())
        self.assertTrue(net.exists())
        self.assertFalse(post.exists())

    def test_global_json_command_reference_from_incomplete_trial_is_kept(self):
        first = self.trial()
        net = self.file('sha.net',root=first)
        consumer = self.repo/'experiments/vtr/tailcompile_v68/runs/current/evaluation/runs/sha/w100/s341/default'
        consumer.mkdir(parents=True)
        self.file('started.json',json.dumps(dict(identity={'command':['vpr','--net_file',str(net)]})),root=consumer)
        plan = self.plan(references=None)
        self.assertNotIn(net,self.paths(plan))
        self.apply(plan,references=None)
        self.assertTrue(net.exists())

    def test_missing_or_running_evidence_is_not_a_license_to_delete(self):
        running = self.trial('experiments/vtr/tailcompile_v5/runs/live',status='running')
        no_record = self.repo/'experiments/vtr/tailcompile_v5/runs/no_record'
        keep = [self.file('sha.net',root=folder) for folder in (running,no_record)]
        self.apply(self.plan())
        self.assertTrue(all(p.exists() for p in keep))

    def test_new_cross_version_reference_after_plan_prevents_deletion(self):
        folder=self.trial()
        net=self.file('sha.net',root=folder)
        plan=self.plan(references=None)
        self.assertIn(net,self.paths(plan))
        self.file('experiments/vtr/tailcompile_v68/runs/current/contexts.json',
                  json.dumps({'contexts':[{'packed_net':str(net)}]}))
        self.apply(plan,references=None)
        self.assertTrue(net.exists())

    def test_active_tracked_and_recent_files_are_excluded(self):
        folder = self.trial()
        tracked = self.file('tracked.net',root=folder)
        active = self.file('open.net',root=folder)
        recent = self.file('recent.net',root=folder)
        self.tracked.add(tracked)
        self.active[1].add(active)
        self.apply(self.plan(min_age_hours=24))
        self.assertTrue(all(p.exists() for p in (tracked,active,recent)))
        self.apply(self.plan())
        self.assertTrue(tracked.exists())
        self.assertTrue(active.exists())
        self.assertFalse(recent.exists())

    def test_payload_or_evidence_changed_after_plan_is_preserved(self):
        folder = self.trial()
        payload = self.file('sha.net',root=folder)
        plan = self.plan()
        payload.write_text('changed bytes since reviewed plan')
        self.apply(plan)
        self.assertTrue(payload.exists())
        plan = self.plan()
        self.file('manifest.json',json.dumps(dict(vpr_status='running')),root=folder)
        self.apply(plan)
        self.assertTrue(payload.exists())

    def test_plan_path_escape_is_rejected_without_deleting_outside_scope(self):
        folder = self.trial()
        payload = self.file('sha.net',root=folder)
        outside = self.file('valuable.net')
        plan = self.plan()
        bad = copy.deepcopy(plan)
        row = next(e for e in bad['entries'] if e['path']==str(payload))
        row['path']=str(outside)
        try:
            self.apply(bad)
        except (ValueError,RuntimeError):
            pass
        self.assertTrue(outside.exists())

    def test_borrowed_completed_evidence_from_another_trial_is_rejected(self):
        folder=self.trial()
        payload=self.file('sha.net',root=folder)
        other=self.trial('experiments/vtr/tailcompile_v3/sha/4x4/runs/other')
        plan=self.plan()
        row=next(e for e in plan['entries'] if e['path']==str(payload))
        row['evidence']=str(other/'manifest.json')
        row['evidence_sha256']=reclaim.digest(other/'manifest.json')
        try:
            self.apply(plan)
        except (ValueError,RuntimeError):
            pass
        self.assertTrue(payload.exists())

    def test_scratch_copy_requires_a_closed_archive_counterpart(self):
        self.trial()
        good=self.file('tailcompile_v3/sha/4x4/closed/sha.net',root=self.scratch)
        bad=self.file('tailcompile_v3/sha/4x4/no_archive/sha.net',root=self.scratch)
        self.apply(self.plan())
        self.assertFalse(good.exists())
        self.assertTrue(bad.exists())

    def test_inventory_reports_types_and_recent_versions_without_mutation(self):
        items = [self.file('experiments/vtr/tailcompile_v67/runs/a/sha.net','N'*17),
                 self.file('experiments/vtr/tailcompile_v5/inputs/sha.net.post_routing','P'*29),
                 self.file('tailcompile_v3/sha/net.net','S'*31,root=self.scratch)]
        before = {p:p.read_bytes() for p in items}
        report=self.root/'inventory.json'
        result=reclaim.inventory(self.repo,self.scratch,report=report)
        self.assertTrue(report.exists())
        text=json.dumps(result)
        self.assertIn('tailcompile_v67',text)
        self.assertIn('postrouting',text)
        for p,content in before.items():
            self.assertEqual(p.read_bytes(),content)

    @unittest.skipUnless(sys.platform.startswith('linux'),'quoted Unicode POSIX pathname requires Linux')
    def test_streamed_unicode_net_reference_and_escaped_quote_across_chunk_boundary(self):
        source=self.file('experiments/vtr/tailcompile_v5/inputs/路径"quoted".net','canonical net\n')
        other=self.file('experiments/vtr/tailcompile_v5/inputs/second.net','another net\n')
        literal=json.dumps(str(source),ensure_ascii=False)
        # Split the escape itself: final char in the first 1 MiB is backslash,
        # and the escaped quote starts the next read, inside an unfinished token.
        opening=1024*1024-(literal.index('\\"')+1)
        prefix='{"numbers":['+'0,'*500000+'0],"source":'
        prefix+=' '*(opening-len(prefix))
        record=self.file('stream_fixture.json',prefix+literal+',"second":'+json.dumps(str(other))+'}')
        paths=reclaim.streamed_paths(record)
        self.assertIn(source,paths)
        self.assertIn(other,paths)

    def test_malformed_or_unusually_long_ir_provenance_fails_closed(self):
        folder=self.trial()
        net=self.file('sha.net',root=folder)
        broken=self.file('experiments/vtr/tailcompile_v68/runs/current/logic_ir.json',
                         '{"packed_net":"unfinished')
        refs=reclaim.collect_references(self.repo,self.scratch)
        self.assertIn(str(broken),refs.unreadable)
        self.apply(self.plan(references=None),references=None)
        self.assertTrue(net.exists())
        with self.assertRaises(ValueError):
            reclaim.streamed_paths(broken)
        long_record=self.file('long_fixture.json','{"value":"'+'x'*(1024*1024+100)+'"}')
        with self.assertRaises(ValueError):
            reclaim.streamed_paths(long_record)

    @unittest.skipUnless(sys.platform.startswith('linux'),'same-filesystem hardlink semantics require Linux')
    def test_identical_referenced_input_copies_deduplicate_without_removing_paths(self):
        folders=[self.trial(f'experiments/vtr/tailcompile_v67/runs/dedup/{name}') for name in ('a','b')]
        copies=[self.file('input.net','same frozen netlist\n',root=f) for f in folders]
        for p in copies:
            p.chmod(0o444)
        self.references.update(copies)
        before={p:p.read_bytes() for p in copies}
        plan=self.plan(deduplicate_netlists=True)
        self.assertEqual(len(self.paths(plan,'hardlink')),1)
        self.apply(plan,ack_immutable_archives=True)
        self.assertEqual(copies[0].stat().st_ino,copies[1].stat().st_ino)
        for p,content in before.items():
            self.assertEqual(p.read_bytes(),content)
            self.assertTrue(p.exists())

    @unittest.skipUnless(sys.platform.startswith('linux'),'same-filesystem hardlink semantics require Linux')
    def test_different_net_contents_are_never_deduplicated(self):
        folders=[self.trial(f'experiments/vtr/tailcompile_v67/runs/dedup/{name}') for name in ('a','b')]
        copies=[self.file('input.net',text,root=f) for f,text in zip(folders,('first frozen net\n','other frozen net\n'))]
        for p in copies:
            p.chmod(0o444)
        self.assertEqual(len(self.paths(self.plan(deduplicate_netlists=True),'hardlink')),0)
        self.assertNotEqual(copies[0].stat().st_ino,copies[1].stat().st_ino)

    @unittest.skipUnless(sys.platform.startswith('linux'),'same-filesystem hardlink semantics require Linux')
    def test_three_duplicate_inputs_share_one_inode_and_remain_readonly(self):
        folders=[self.trial(f'experiments/vtr/tailcompile_v63/runs/dedup/{name}') for name in ('a','b','c')]
        copies=[self.file('input.net','one frozen netlist\n',root=f) for f in folders]
        for p in copies:
            p.chmod(0o444)
        plan=self.plan(deduplicate_netlists=True)
        self.assertEqual(len(self.paths(plan,'hardlink')),2)
        self.apply(plan,ack_immutable_archives=True)
        self.assertEqual(len({p.stat().st_ino for p in copies}),1)
        for p in copies:
            self.assertEqual(p.read_text(),'one frozen netlist\n')
            self.assertEqual(p.stat().st_mode & 0o222,0)

    @unittest.skipUnless(sys.platform.startswith('linux'),'canonical-input independence requires Linux')
    def test_archived_dedup_keeps_identical_canonical_input_on_independent_inode(self):
        canonical=self.file('experiments/vtr/tailcompile_v5/inputs/full/source.net','same canonical bytes\n')
        before=(canonical.stat().st_ino,canonical.stat().st_mode,canonical.read_bytes())
        folders=[self.trial(f'experiments/vtr/tailcompile_v63/runs/dedup/{n}') for n in ('a','b','c')]
        copies=[self.file('input.net','same canonical bytes\n',root=f) for f in folders]
        plan=self.plan(deduplicate_netlists=True)
        self.assertEqual(len(self.paths(plan,'hardlink')),2)
        self.apply(plan,ack_immutable_archives=True)
        self.assertEqual(len({p.stat().st_ino for p in copies}),1)
        self.assertNotEqual(canonical.stat().st_ino,copies[0].stat().st_ino)
        self.assertEqual((canonical.stat().st_ino,canonical.stat().st_mode,canonical.read_bytes()),before)

    @unittest.skipUnless(sys.platform.startswith('linux'),'same-filesystem hardlink semantics require Linux')
    def test_changed_dedup_canonical_is_not_linked_into_other_inputs(self):
        folders=[self.trial(f'experiments/vtr/tailcompile_v63/runs/dedup/{name}') for name in ('a','b')]
        copies=[self.file('input.net','one frozen netlist\n',root=f) for f in folders]
        for p in copies:
            p.chmod(0o444)
        plan=self.plan(deduplicate_netlists=True)
        row=next(e for e in plan['entries'] if e['action']=='hardlink')
        master=Path(row['canonical'])
        master.chmod(0o644)
        master.write_text('changed canonical input\n')
        self.apply(plan,ack_immutable_archives=True)
        self.assertNotEqual(copies[0].stat().st_ino,copies[1].stat().st_ino)
        target=Path(row['path'])
        self.assertEqual(target.read_text(),'one frozen netlist\n')

    @unittest.skipUnless(sys.platform.startswith('linux'),'hardlink archive acknowledgement requires Linux')
    def test_dedup_apply_requires_explicit_immutable_archive_acknowledgement(self):
        folders=[self.trial(f'experiments/vtr/tailcompile_v63/runs/dedup/{name}') for name in ('a','b')]
        copies=[self.file('input.net','frozen archived netlist\n',root=f) for f in folders]
        for p in copies:
            p.chmod(0o444)
        plan=self.plan(deduplicate_netlists=True)
        self.assertEqual(len(self.paths(plan,'hardlink')),1)
        with self.assertRaises((ValueError,RuntimeError)):
            self.apply(plan)
        self.assertNotEqual(copies[0].stat().st_ino,copies[1].stat().st_ino)

    @unittest.skipUnless(sys.platform.startswith('linux'),'hardlink current-dependency protection requires Linux')
    def test_new_current_reference_to_dedup_master_or_target_blocks_linking(self):
        for referenced_key in ('canonical','path'):
            with self.subTest(referenced_key=referenced_key):
                folders=[self.trial(f'experiments/vtr/tailcompile_v63/runs/{referenced_key}/{name}')
                         for name in ('a','b')]
                copies=[self.file('input.net',f'{referenced_key} frozen net\n',root=f) for f in folders]
                for p in copies:
                    p.chmod(0o444)
                plan=self.plan(deduplicate_netlists=True,references=None)
                row=next(e for e in plan['entries'] if e['action']=='hardlink' and e['path']==str(copies[1]))
                self.file('experiments/vtr/tailcompile_v68/runs/current/contexts.json',
                    json.dumps({'contexts':[{'packed_net':row[referenced_key]}]}))
                self.apply(plan,references=None,ack_immutable_archives=True)
                self.assertNotEqual(copies[0].stat().st_ino,copies[1].stat().st_ino)

    @unittest.skipUnless(sys.platform.startswith('linux'),'symlink/hardlink safety requires Linux')
    def test_existing_symlinks_and_hardlinks_are_not_directly_unlinked(self):
        folder=self.trial()
        precious=self.file('precious.net')
        symbolic=folder/'symbolic.net'
        hard=folder/'hard.net'
        symbolic.symlink_to(precious)
        hard.hardlink_to(precious)
        self.apply(self.plan())
        self.assertTrue(symbolic.is_symlink())
        self.assertTrue(hard.exists())
        self.assertEqual(precious.read_bytes(),hard.read_bytes())

    @unittest.skipUnless(sys.platform.startswith('linux'),'real Git/proc safety requires Linux')
    def test_real_git_and_open_worker_are_protected(self):
        subprocess.run(['git','init','-q',str(self.repo)],check=True)
        folder=self.trial()
        tracked=self.file('tracked.net',root=folder)
        live=self.file('live.net',root=folder)
        disposable=self.file('disposable.net',root=folder)
        subprocess.run(['git','-C',str(self.repo),'add',str(tracked)],check=True)
        child=subprocess.Popen([sys.executable,'-u','-c',
            'import sys,time; f=open(sys.argv[1]); print("ready",flush=True); time.sleep(30)',str(live)],
            cwd=self.repo,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'ready')
            plan=self.plan(active=None,tracked=None)
            self.apply(plan,active=None,tracked=None)
            self.assertTrue(tracked.exists())
            self.assertTrue(live.exists())
            self.assertFalse(disposable.exists())
        finally:
            child.terminate()
            child.wait(timeout=5)
            child.stdout.close()

    @unittest.skipUnless(sys.platform.startswith('linux'),'real controller/proc safety requires Linux')
    def test_live_controller_output_directory_blocks_later_trial_deletion(self):
        subprocess.run(['git','init','-q',str(self.repo)],check=True)
        folder=self.trial('experiments/vtr/tailcompile_v63/runs/closed')
        payload=self.file('sha.net',root=folder)
        version=self.repo/'experiments/vtr/tailcompile_v63'
        child=subprocess.Popen([sys.executable,'-u','-c',
            'import time; print("ready",flush=True); time.sleep(30)','--out',str(version)],
            cwd=self.repo,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'ready')
            self.apply(self.plan(active=None,tracked=None),active=None,tracked=None)
            self.assertTrue(payload.exists())
        finally:
            child.terminate()
            child.wait(timeout=5)
            child.stdout.close()


if __name__=='__main__':
    unittest.main()
