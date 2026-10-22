"""Synthetic read-only account inventory checks, without scanning real accounts."""
from __future__ import annotations

import csv
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import disk_audit


class DiskAuditTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='account-disk-audit-')
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name).resolve()
        self.root = self.base/'account'
        self.root.mkdir()

    def file(self, name, content=b'payload'):
        path = self.root/name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def run_audit(self, roots=None, out=None, **kwargs):
        out = out or self.base/'report'
        with redirect_stdout(io.StringIO()):
            result = disk_audit.audit(roots or [self.root], out,
                heartbeat_seconds=.01, scan_deleted=False, **kwargs)
        return out, result

    def rows(self, path):
        with Path(path).open(encoding='utf-8', newline='') as stream:
            return list(csv.DictReader(stream))

    def test_type_classification_uses_actual_vpr_suffixes(self):
        for name in ('sha.net.post_routing', 'sha.net_postrouting', 'sha.net_post_routing'):
            self.assertIn('post', disk_audit.classify_type(name), name)
        for name in ('sha.net', 'sha.route', 'mapped.blif', 'generator.pt', 'data.json'):
            self.assertTrue(disk_audit.classify_type(name))
        self.assertNotEqual(disk_audit.classify_type('sha.net'),
                            disk_audit.classify_type('sha.net.post_routing'))

    def test_overlapping_roots_collapse_but_similar_prefix_siblings_do_not(self):
        child = self.root/'workspace'/'TailCompile'
        child.mkdir(parents=True)
        sibling = self.base/'account2'
        sibling.mkdir()
        result = disk_audit.normalize_roots([child, self.root, child, sibling])
        self.assertEqual(set(result), {self.root, sibling})

    def test_entire_account_and_hidden_directories_reported_without_mutation(self):
        sources = [self.file('workspace/TailCompile/experiments/sha.net', b'x'*2000),
                   self.file('.cache/models/weights.pt', b'y'*1000),
                   self.file('miniconda3/pkgs/archive.tar.gz', b'z'*500)]
        before = {p:(p.read_bytes(), p.stat().st_ino, p.stat().st_mtime_ns) for p in sources}
        out, summary = self.run_audit()
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['files'], 3)
        self.assertEqual(summary['logical_bytes'], 3500)
        for name in ('report.log','summary.json','directories.csv','file_types.csv',
                     'largest_files.csv','errors.log','progress.json'):
            self.assertTrue((out/name).is_file(), name)
        paths = {row['path'] for row in self.rows(out/'directories.csv')}
        self.assertIn(str(self.root/'.cache'), paths)
        self.assertIn(str(self.root/'miniconda3'), paths)
        self.assertIn(str(self.root), paths)
        for path, expected in before.items():
            self.assertEqual((path.read_bytes(),path.stat().st_ino,path.stat().st_mtime_ns), expected)
        self.assertEqual({p for p in self.root.rglob('*') if p.is_file()}, set(sources))

    def test_report_directory_inside_account_is_excluded_from_scan(self):
        self.file('data.net',b'0123456789')
        out, summary = self.run_audit(out=self.root/'audit-output')
        self.assertTrue(summary['complete'])
        self.assertEqual(summary['files'],1)
        self.assertEqual(summary['logical_bytes'],10)
        self.assertTrue((out/'report.log').is_file())

    @unittest.skipUnless(sys.platform.startswith('linux'),'native Linux hardlinks and symlinks')
    def test_hardlink_allocations_deduplicated_and_interior_symlink_not_followed(self):
        first = self.file('a/input.net',b'x'*8192)
        second = self.root/'b'/'input.net'
        second.parent.mkdir()
        second.hardlink_to(first)
        outside = self.base/'outside'
        outside.mkdir()
        (outside/'not_in_scope.net').write_bytes(b'not followed')
        (self.root/'shortcut').symlink_to(outside, target_is_directory=True)
        out, summary = self.run_audit()
        self.assertEqual(summary['files'],1)
        self.assertEqual(summary['logical_bytes'],8192)
        self.assertEqual(summary['duplicate_hardlinks_skipped'],1)
        kinds = self.rows(out/'file_types.csv')
        nets = [row for row in kinds if 'net' in row['type']]
        self.assertEqual(sum(int(row['allocated_bytes']) for row in nets),first.stat().st_blocks*512)
        self.assertFalse(any('not_in_scope' in row['path'] for row in self.rows(out/'largest_files.csv')))

    @unittest.skipUnless(sys.platform.startswith('linux'),'Linux /proc open deleted files')
    def test_deleted_open_files_scoped_to_the_account(self):
        path = self.file('held.net',b'x'*8192)
        command = [sys.executable,'-u','-c',
            'import os,sys,time; f=open(sys.argv[1],"rb"); os.unlink(sys.argv[1]); '
            'print("ready",flush=True); time.sleep(30)',str(path)]
        child = subprocess.Popen(command,stdout=subprocess.PIPE,text=True)
        try:
            self.assertEqual(child.stdout.readline().strip(),'ready')
            out = self.base/'deleted-report'
            with redirect_stdout(io.StringIO()):
                summary = disk_audit.audit([self.root],out,scan_deleted=True)
            self.assertTrue(summary['complete'])
            rows = self.rows(out/'deleted_open.csv')
            self.assertTrue(any(str(child.pid)==row.get('pid') for row in rows))
            self.assertTrue(all(str(self.root) in row['path'] for row in rows))
        finally:
            child.terminate()
            child.wait(timeout=5)
            child.stdout.close()


if __name__=='__main__':
    unittest.main()
