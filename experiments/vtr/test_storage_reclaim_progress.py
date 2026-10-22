"""Progress and repeated-work regressions, using synthetic temporary archives."""
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import storage_reclaim as reclaim


class StorageReclaimProgressTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='tailcompile-reclaim-progress-')
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()/'repo'
        self.folder = self.repo/'experiments/vtr/tailcompile_v67/runs/closed'
        self.folder.mkdir(parents=True)
        self.evidence = self.folder/'manifest.json'
        self.evidence.write_text(json.dumps({'vpr_status':'success'}),encoding='utf-8')
        for name in ('one.route','two.rpt'):
            (self.folder/name).write_text('disposable synthetic payload',encoding='utf-8')

    def test_progress_is_rate_limited_and_phase_changes_are_immediate(self):
        stream = io.StringIO()
        now = [0.0]
        progress = reclaim.Progress('plan',30,stream=stream,clock=lambda:now[0])
        progress.phase('scan_payloads',files_seen=0)
        now[0] = 29
        progress.update(files_seen=20)
        self.assertEqual(len(stream.getvalue().splitlines()),1)
        now[0] = 30
        progress.update(files_seen=30)
        progress.phase('hash_netlist_duplicates',files_to_hash=2)
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertEqual([row['phase'] for row in rows],
                         ['scan_payloads','scan_payloads','hash_netlist_duplicates'])
        self.assertEqual(rows[1]['files_seen'],30)
        self.assertEqual(rows[1]['elapsed_seconds'],30)
        self.assertEqual(rows[2]['phase_elapsed_seconds'],0)
        self.assertTrue(all(row['event']=='storage_progress' for row in rows))

    def test_full_sha256_reports_bytes_before_a_single_file_finishes(self):
        source = self.folder/'large.input.net'
        payload = b'synthetic bytes\0'*(3*1024*1024//16)
        source.write_bytes(payload)
        now = [0.0]

        def clock():
            now[0] += 15
            return now[0]

        stream = io.StringIO()
        progress = reclaim.Progress('plan',30,stream=stream,clock=clock)
        progress.phase('hash_netlist_duplicates')
        value = reclaim._digest(source,progress,total_bytes=len(payload))
        self.assertEqual(value,hashlib.sha256(payload).hexdigest())
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        intermediate = [row for row in rows if 0 < row.get('file_bytes_read',0) < len(payload)]
        self.assertTrue(intermediate)
        self.assertEqual(intermediate[0]['current_path'],str(source))
        self.assertEqual(intermediate[0]['file_size_bytes'],len(payload))
        self.assertEqual(progress.hashed_bytes,len(payload))
        self.assertEqual(progress.hashed_files,1)

    def test_plan_hashes_shared_evidence_once_and_apply_rechecks_each_entry(self):
        with patch.object(reclaim,'digest',wraps=reclaim.digest) as digest:
            plan = reclaim.plan_reclaim(self.repo,min_age_hours=0,active=(set(),set(),0),
                                        tracked=set(),references=set())
            evidence_calls = [call for call in digest.call_args_list if Path(call.args[0])==self.evidence]
            self.assertEqual(len(plan['entries']),2)
            self.assertEqual(len(evidence_calls),1)
            digest.reset_mock()
            result = reclaim.apply_reclaim(plan,active=(set(),set(),0),tracked=set(),references=set())
            evidence_calls = [call for call in digest.call_args_list if Path(call.args[0])==self.evidence]
            self.assertEqual(len(evidence_calls),2)
            self.assertEqual(result['deleted_files'],2)

    def test_inventory_cli_keeps_stdout_a_single_json_document(self):
        out,err = io.StringIO(),io.StringIO()
        with patch.object(sys,'argv',['storage_reclaim.py','inventory','--repo',str(self.repo),
                                      '--progress-seconds','0.01']), redirect_stdout(out), redirect_stderr(err):
            reclaim.main()
        result = json.loads(out.getvalue())
        self.assertEqual(result['schema'],'tailcompile-legacy-storage-inventory-1')
        rows = [json.loads(line) for line in err.getvalue().splitlines()]
        self.assertEqual(rows[0]['phase'],'starting')
        self.assertEqual(rows[-1]['phase'],'complete')
        self.assertTrue(all(row['operation']=='inventory' for row in rows))

    def test_plan_and_apply_report_reference_scan_and_execution_phases(self):
        stream = io.StringIO()
        progress = reclaim.Progress('plan',stream=stream)
        plan = reclaim.plan_reclaim(self.repo,min_age_hours=0,active=(set(),set(),0),
                                    tracked=set(),deduplicate_netlists=True,progress=progress)
        phases = {json.loads(line)['phase'] for line in stream.getvalue().splitlines()}
        self.assertTrue({'collect_references','scan_payloads','hash_netlist_duplicates',
                         'summarize_plan'}.issubset(phases))
        stream = io.StringIO()
        result = reclaim.apply_reclaim(plan,active=(set(),set(),0),tracked=set(),
                                       progress=reclaim.Progress('apply',stream=stream))
        rows = [json.loads(line) for line in stream.getvalue().splitlines()]
        self.assertIn('collect_references',{row['phase'] for row in rows})
        self.assertEqual(rows[-1]['phase'],'apply_entries')
        self.assertEqual(rows[-1]['status'],'complete')
        self.assertEqual(rows[-1]['entries_checked'],2)
        self.assertEqual(result['deleted_files'],2)

    def test_invalid_progress_interval_is_rejected(self):
        for seconds in (0,-1,float('inf'),float('nan')):
            with self.subTest(seconds=seconds), self.assertRaises(ValueError):
                reclaim.Progress('plan',seconds,stream=io.StringIO())


if __name__=='__main__':
    unittest.main()
