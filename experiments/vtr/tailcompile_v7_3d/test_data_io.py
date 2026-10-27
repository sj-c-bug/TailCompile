"""Check leakage rejection, exact resume and protected completed-payload cleanup."""
from pathlib import Path
import tempfile
import unittest
from data import read_snapshot
from io_utils import prune_derived, run_trial, write, read


class DataAndStorageTests(unittest.TestCase):
    def test_archive_refuses_self_and_nested_paths_before_writing(self):
        from archive_local import archive
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            source.mkdir()
            write(source/'evidence.json', {'complete':True})
            for out in (source, source/'recursive', source.parent):
                with self.assertRaisesRegex(ValueError,'disjoint'):
                    archive(source,out)
            self.assertFalse((source/'recursive').exists())
            self.assertEqual(read(source/'evidence.json'),{'complete':True})
            result = archive(source,Path(tmp)/'safe_archive')
            self.assertEqual(result['archived_file_count'],1)

    def test_3d_structure_rejects_teacher_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp)/'ap'
            Path(str(prefix)+'.meta.tsv').write_text('12\t12\t2\n')
            Path(str(prefix)+'.nets.tsv').write_text('')
            row = Path(str(prefix)+'.blocks.tsv')
            row.write_text('0\ta\t0\t0\t-1\t-1\t-1\t-1\t.names\t0\n')
            self.assertEqual(read_snapshot(prefix,structure=True)['size'],[12,12,2])
            row.write_text('0\ta\t0\t0\t2\t3\t1\t0\t.names\t0\n')
            with self.assertRaisesRegex(ValueError,'leaked'):
                read_snapshot(prefix,structure=True)

    def test_cleanup_requires_metrics_and_preserves_unique_net_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            preserved = folder/'input.net'
            preserved.write_text('unique input')
            output = folder/'derived.net.post_routing'
            output.write_text('disposable')
            with self.assertRaises(ValueError):
                prune_derived(folder,protected=[preserved])
            write(folder/'started.json',{})
            write(folder/'metrics.json',{'status':'success'})
            teacher = folder/'teacher.blocks.tsv'
            teacher.write_text('unique teacher')
            prune_derived(folder,protected=[preserved])
            self.assertTrue(preserved.exists())
            self.assertTrue(teacher.exists())
            self.assertFalse(output.exists())
            self.assertEqual(read(folder/'payload_pruned.json')['removed_bytes'],10)

    def test_exact_resume_survives_payload_cleanup(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'trial'
            command = [sys.executable,'-c',"print('VPR succeeded'); print('Circuit successfully routed')"]
            first = run_trial(folder,command,timeout=10)
            second = run_trial(folder,command,timeout=10)
            self.assertEqual(first['workdir'],second['workdir'])
            self.assertEqual(len(list((folder/'attempts').iterdir())),1)
            with self.assertRaisesRegex(ValueError,'signature mismatch'):
                run_trial(folder,command,timeout=11)

    def test_unattributed_kill_is_infrastructure_and_retry_is_bounded(self):
        import sys
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/'killed'
            command = [sys.executable,'-c','import sys; sys.exit(137)']
            first = run_trial(folder,command,timeout=10)
            self.assertEqual(first['status'],'infrastructure_failure')
            run_trial(folder,command,timeout=10,retry_infrastructure=True)
            with self.assertRaisesRegex(RuntimeError,'retry limit'):
                run_trial(folder,command,timeout=10,retry_infrastructure=True)
            self.assertEqual(len(list((folder/'attempts').iterdir())),2)


if __name__=='__main__':
    unittest.main()
