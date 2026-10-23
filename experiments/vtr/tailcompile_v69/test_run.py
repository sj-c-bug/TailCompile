"""CPU-only orchestration tests: resumability and evidence, no VPR or Torch."""
from argparse import Namespace
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import run
from bridge import read, write


class RunIntegrationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.checkpoint = self.root/'generator.pt'
        self.checkpoint.write_bytes(b'provenance-only test fixture')
        self.context = dict(design='design_a', family='family_a', split='train', width=100,
                            mapped_blif=str(self.root/'input.blif'), device_width=30)
        self.a = Namespace(out=self.root, timing_jobs=2, cpu_list=None, timeout=100,
                           heartbeat=30, retry_failed=False, retry_timeouts=False,
                           confirm_remote_stopped=False, identity={'binary_sha256':'test'})
        self.protocol = dict(options={}, runtime_metric='full worker excluding reusable RRG')
        self.generated = {('design_a',100,0):dict(ap_init_sha256='test-initializer')}
        self.hw = dict(host='test',boot_id='fixed',concurrency=2,
                       selected_cores=[dict(cpu=0),dict(cpu=1)])

    def fake_trial(self, a, p, context, generation, checkpoint, lane, method,
                   directory, barrier, protocol):
        row = dict(status='success', design=context['design'], width=context['width'],
                   method=method, seed=lane['seed'], candidate=lane['candidate'],
                   cpu=lane['cpu'], vpr_seconds=1., generation_seconds=.1)
        for name,value in [('trial.json',dict(status='success')),
                           ('metrics.json',row), ('affinity.json',dict(cpu=lane['cpu'])),
                           ('spec.json',dict(method=method)), ('cost.json',dict(vpr_seconds=1.))]:
            write(directory/name,value)
        return row

    def pairs(self,root):
        return run.timed_pairs(self.a,self.protocol,[self.context],self.checkpoint,
                               self.generated,root,[351,352])

    def common_patches(self):
        return (patch.object(run.pt,'hardware',return_value=self.hw),
                patch.object(run.pt,'snapshot',return_value={'test_load':0}),
                patch.object(run,'prune_trial'), patch.object(run,'emit'),
                patch.object(run.pt,'emit'))

    def test_completed_wave_is_reused_and_evidence_tampering_is_rejected(self):
        root=self.root/'timing'
        patches=self.common_patches()
        with patches[0],patches[1],patches[2],patches[3],patches[4], \
                patch.object(run,'timed_trial',side_effect=self.fake_trial) as trial:
            rows=self.pairs(root)
            self.assertEqual(len(rows),4)
            self.assertEqual(trial.call_count,4)
            resumed=self.pairs(root)
            self.assertEqual(resumed,rows)
            self.assertEqual(trial.call_count,4)
            path=root/'waves/w0000/half0/lane00/metrics.json'
            saved=read(path);saved['vpr_seconds']=999
            write(path,saved)
            with self.assertRaisesRegex(ValueError,'evidence changed'):
                self.pairs(root)
            self.assertEqual(trial.call_count,4)

    def test_interrupted_wave_is_archived_and_entire_pair_load_repeated(self):
        root=self.root/'timing'
        def failing(*args):
            directory=args[7]
            if directory.parent.name=='half1' and directory.name=='lane00':
                raise RuntimeError('simulated interruption')
            return self.fake_trial(*args)
        patches=self.common_patches()
        with patches[0],patches[1],patches[2],patches[3],patches[4]:
            with patch.object(run,'timed_trial',side_effect=failing):
                with self.assertRaisesRegex(RuntimeError,'Incomplete wave'):
                    self.pairs(root)
            self.assertFalse((root/'waves/w0000/wave.json').exists())
            first_half=read(root/'waves/w0000/half0/lane00/metrics.json')
            with patch.object(run,'timed_trial',side_effect=self.fake_trial) as retry:
                rows=self.pairs(root)
                self.assertEqual(retry.call_count,4)  # Not only the missing half.
            self.assertEqual(len(rows),4)
            archived=list((root/'_attempts').glob('*/w0000/half0/lane00/metrics.json'))
            self.assertEqual(len(archived),1)
            self.assertEqual(read(archived[0]),first_half)

    def test_cannot_archive_partial_wave_while_workers_might_be_alive(self):
        root=self.root/'timing'
        (root/'waves/w0000').mkdir(parents=True)
        patches=self.common_patches()
        with patches[0],patches[1],patches[2],patches[3],patches[4], \
                patch.object(run.pt,'assert_workers_stopped',side_effect=RuntimeError('Worker may still be alive')), \
                patch.object(run,'timed_trial') as trial:
            with self.assertRaisesRegex(RuntimeError,'still be alive'):
                self.pairs(root)
            trial.assert_not_called()
            self.assertTrue((root/'waves/w0000').exists())

    def test_phase_evidence_is_saved_and_native_gp_is_required(self):
        directory=self.root/'trial';directory.mkdir()
        log='''TAILCOMPILE_V68_GP_INIT_CONSUMED
TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1
## AP Global Placer took 4.2 seconds (max_rss 20 MiB)
# Routing took 8.0 seconds (max_rss 20 MiB)
Successfully routed after 12 routing iterations.
Final critical path delay (least slack): 5.2 ns
Total wirelength: 12345
VPR succeeded
'''
        (directory/'vpr.out').write_text(log,encoding='utf-8')
        result=dict(status='success',returncode=0,wall_seconds=13.)
        row=run.parse_row(result,directory,self.context,351,'fast')
        self.assertEqual(row['cpd'],5.2)
        self.assertEqual(row['phase_metrics']['seconds']['global_placement_seconds'],4.2)
        self.assertEqual(read(directory/'metrics.json'),row)
        (directory/'vpr.out').write_text('VPR succeeded\n',encoding='utf-8')
        with self.assertRaisesRegex(RuntimeError,'Native GP'):
            run.parse_row(result,directory,self.context,351,'fast')

    def test_disk_quota_error_retains_metrics_before_raising(self):
        directory=self.root/'trial';directory.mkdir()
        (directory/'vpr.out').write_text('Disk quota exceeded\n',encoding='utf-8')
        with self.assertRaises(run.StoragePressureError):
            run.parse_row(dict(status='other_failure',returncode=1),directory,self.context,351,'default')
        self.assertEqual(read(directory/'metrics.json')['status'],'other_failure')

    def test_default_and_fast_have_identical_vpr_only_clock_scope(self):
        lane=dict(cpu=0,seed=351,candidate=0)
        def execute(command,directory,*args,**kwargs):
            method=read(directory/'spec.json')['method']
            (directory/'vpr.out').write_text('''TAILCOMPILE_V68_GP_INIT_CONSUMED
TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1
VPR succeeded
''',encoding='utf-8')
            write(directory/'affinity.json',dict(allowed_cpus=[0],
                thread_env=dict(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')))
            write(directory/'cost.json',dict(vpr_seconds=100. if method=='default' else 50.,
                rrg_encoding_seconds=0. if method=='default' else 2.))
            return dict(status='success',returncode=0,wall_seconds=120. if method=='default' else 90.)
        with patch.object(run.e,'vpr_command',return_value=['fake-vpr']), \
                patch.object(run,'execute',side_effect=execute),patch.object(run,'check_free_space'):
            default=run.timed_trial(self.a,self.protocol,self.context,self.generated['design_a',100,0],
                self.checkpoint,lane,'default',self.root/'default',threading.Barrier(1),{})
            fast=run.timed_trial(self.a,self.protocol,self.context,self.generated['design_a',100,0],
                self.checkpoint,lane,'fast',self.root/'fast',threading.Barrier(1),{})
        self.assertEqual(default['vpr_seconds'],100.)
        self.assertEqual(fast['vpr_seconds'],50.)
        self.assertEqual(default['vpr_seconds']+default['generation_seconds'],120.)
        self.assertEqual(fast['vpr_seconds']+fast['generation_seconds'],88.)


if __name__=='__main__':
    unittest.main()
