"""Synthetic storage-pressure and post-metric lifecycle checks; no real VPR."""
from __future__ import annotations

from argparse import Namespace
from contextlib import redirect_stdout
import errno
import io
import os
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import experiment
import lifecycle
from common import read, write


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='v68-lifecycle-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()

    def test_free_space_guard_is_explicitly_not_a_quota_check(self):
        with patch.dict(os.environ, {'TAILCOMPILE_V68_MIN_FREE_GIB':'10'}), \
                patch.object(lifecycle.shutil, 'disk_usage', return_value=Namespace(free=2*1024**3)):
            with self.assertRaisesRegex(lifecycle.StoragePressureError, 'does not measure user/project quota'):
                lifecycle.check_free_space(self.root/'not_created')
        with patch.dict(os.environ, {'TAILCOMPILE_V68_MIN_FREE_GIB':'0'}), \
                patch.object(lifecycle.shutil, 'disk_usage', return_value=Namespace(free=1)):
            self.assertEqual(lifecycle.check_free_space(self.root), 1)

    def test_disk_log_classification_never_calls_infrastructure_unroutable(self):
        for text in ('Disk quota exceeded', 'write: No space left on device', 'EDQUOT'):
            self.assertTrue(lifecycle.log_storage_failure(text, 1))
        self.assertFalse(lifecycle.log_storage_failure('Circuit is unroutable with a channel width factor of 80',2))
        self.assertFalse(lifecycle.log_storage_failure('VPR succeeded',0))
        self.assertTrue(lifecycle.storage_error(OSError(errno.ENOSPC, 'full')))
        self.assertTrue(lifecycle.storage_error(OSError(122, 'quota')))
        self.assertFalse(lifecycle.storage_error(OSError(errno.EACCES, 'permission')))

    def test_quota_failure_stops_undispatched_tasks_and_keeps_completed_work(self):
        attempted = []

        def one(i):
            attempted.append(i)
            if i == 1:
                raise OSError(122, 'Disk quota exceeded')
            write(self.root/f'{i}.json', {'done':i})
            return i

        with redirect_stdout(io.StringIO()), self.assertRaises(lifecycle.StoragePressureError):
            lifecycle.parallel(one, range(50), jobs=1)
        self.assertEqual(attempted, [0,1])
        self.assertEqual(read(self.root/'0.json'), {'done':0})

    def test_active_sibling_drains_after_storage_failure(self):
        barrier = threading.Barrier(2)
        finished = threading.Event()
        attempted = []

        def one(i):
            attempted.append(i)
            barrier.wait(timeout=5)
            if i == 0:
                raise lifecycle.StoragePressureError('synthetic quota')
            time.sleep(.1)
            write(self.root/'sibling.json', {'complete':True})
            finished.set()
            return i

        with redirect_stdout(io.StringIO()), self.assertRaises(lifecycle.StoragePressureError):
            lifecycle.parallel(one, range(50), jobs=2)
        self.assertEqual(set(attempted), {0,1})
        self.assertTrue(finished.is_set())
        self.assertTrue(read(self.root/'sibling.json')['complete'])

    def test_nonstorage_failures_still_collect_other_tasks(self):
        attempted = []

        def one(i):
            attempted.append(i)
            if i == 2:
                raise ValueError('bad fixture')
            return i

        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, '1 tasks failed'):
            lifecycle.parallel(one, range(6), jobs=2)
        self.assertEqual(set(attempted), set(range(6)))

    def test_policy_configuration_and_opt_out(self):
        for key,value in [('TAILCOMPILE_V68_PRUNE_COMPLETED','yes'),
                          ('TAILCOMPILE_V68_MIN_FREE_GIB','nan'),
                          ('TAILCOMPILE_V68_MIN_FREE_GIB','-1')]:
            with patch.dict(os.environ, {key:value}), self.assertRaises(ValueError):
                lifecycle.record_policy(self.root)
        with patch.dict(os.environ, {'TAILCOMPILE_V68_PRUNE_COMPLETED':'0'}):
            self.assertIsNone(lifecycle.after_metrics(self.root))
            lifecycle.record_policy(self.root)
        self.assertFalse(read(self.root/'storage_policy.json')['prune_completed_vpr_payloads'])
        self.assertFalse(read(self.root/'storage_policy.json')['quota_visible'])

    def test_route_persists_metrics_before_pruning_without_requiring_raw_net(self):
        folder = self.root/'trial'
        folder.mkdir()
        (folder/'vpr.out').write_text('VPR succeeded\nFinal critical path delay (least slack): 5.0\nTotal wirelength: 100\n')
        blif = self.root/'mapped.blif'
        blif.write_text('synthetic mapped input')
        record = dict(mapped_blif=str(blif), design='x', family='xf', split='train', device_width=14)
        original = dict(status='success', returncode=0, wall_seconds=2.0)

        def prune(path, **kwargs):
            metrics = read(Path(path)/'metrics.json')
            self.assertEqual(metrics['cpd'], 5.0)
            self.assertEqual(metrics['wall_seconds'], 2.0)

        with patch.object(experiment, 'call', return_value=original), \
                patch.object(experiment, 'vpr_command', return_value=['vpr']), \
                patch.object(experiment, 'after_metrics', side_effect=prune) as cleanup:
            result = experiment.route(Namespace(), record, 80, 321, folder)
        cleanup.assert_called_once()
        self.assertEqual(result['status'], 'success')
        self.assertEqual(read(folder/'metrics.json'), result)


if __name__=='__main__':
    unittest.main()
