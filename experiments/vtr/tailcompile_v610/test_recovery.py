from pathlib import Path
import concurrent.futures as cf
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import recovery
from bridge import read,write,sha
from prepare import PreparationFailure


class RecoveryPreparationTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.a=SimpleNamespace(out=self.root,identity={'architecture_sha256':'arch','vtr_commit':'commit'},
            binary='vpr',arch='arch.xml',calibration_seeds=[401,402],seeds=[411,412,413,414,415],
            width_start=48,width_max=128,width_step=4,half_band=3,timeout=7200,
            synthesis_timeout=21600,calibration_timeout=14400,device_width_steps=[0,4,8,16,32],
            heartbeat=60,retry_failed=False,retry_timeouts=False)
        self.record=dict(design='bwave',family='bwave',split='sweep',source='koios/bwave.v',
            source_sha256='source',mapped_blif='mapped.blif',mapped_blif_sha256='blif',
            device_width=90,device_height=90,structure_prefix='structure')
        self.source=dict(contexts=[],inputs={'bwave':self.record},boundaries={},observations={},failures=[])

    def capacity_error(self):
        error=PreparationFailure('other_failure','no legal sites',self.root/'attempt')
        error.device_capacity_failure=True
        return error

    def boundary(self,record):
        return dict(record,widths=[80,84,88],reference_width=84,calibration=[],calibration_status='reference_found',reference_is_success=True)

    def test_placement_gate_runs_both_default_seeds_without_routing(self):
        calls=[]
        def execute(command,folder,identity,**options):
            folder.mkdir(parents=True,exist_ok=True)
            (folder/'vpr.out').write_text('VPR succeeded')
            write(folder/'trial.json',dict(status='success'))
            calls.append((command,identity,options))
            return dict(status='success',returncode=0)
        with patch.object(recovery,'execute',side_effect=execute),patch.object(recovery,'prune_trial'),patch.object(recovery,'check_free_space'):
            recovery.qualify_placement(self.a,self.record,self.root/'screen/bwave/g90x90')
        self.assertEqual([c[1]['seed'] for c in calls],[401,402])
        for cmd,identity,options in calls:
            self.assertIn('--analytical_place',cmd)
            self.assertNotIn('--route',cmd)
            self.assertNotIn('--analysis',cmd)
            self.assertEqual(cmd[cmd.index('--device_width')+1],'90')
            self.assertEqual(options['timeout'],14400)
            self.assertEqual(identity['role'],'default_placement_qualification')
        self.assertEqual(self.a.timeout,7200)

    def test_resource_failure_uses_first_passing_grid_and_regenerates_structure(self):
        widths=[]
        def resized(a,record,width):
            widths.append(width)
            return dict(record,device_width=width,device_height=width,structure_prefix=f'new_w{width}')
        with patch.object(recovery,'prepare_input',return_value=self.record), \
             patch.object(recovery,'resource_failure_before',return_value={'trial':'old-failed-trial'}), \
             patch.object(recovery,'resize_input',side_effect=resized), \
             patch.object(recovery,'qualify_placement',side_effect=[self.capacity_error(),None]), \
             patch.object(recovery,'calibrate_record',side_effect=lambda a,r,s,p:self.boundary(r)):
            result=recovery.prepare_design(self.a,self.record,self.source)
        self.assertEqual(widths,[94,98])
        self.assertEqual(result['device_width'],98)
        self.assertEqual(result['structure_prefix'],'new_w98')
        saved=read(self.root/'screen/bwave/qualified_input.json')
        self.assertEqual(saved['initial_grid'],[90,90])
        self.assertEqual(saved['record']['device_width'],98)

    def test_timeout_never_triggers_larger_device(self):
        with patch.object(recovery,'prepare_input',return_value=self.record), \
             patch.object(recovery,'resource_failure_before',return_value=None), \
             patch.object(recovery,'resize_input') as resized, \
             patch.object(recovery,'qualify_placement',side_effect=PreparationFailure('timeout','deadline')):
            with self.assertRaises(PreparationFailure) as caught:
                recovery.prepare_design(self.a,self.record,self.source)
        self.assertEqual(caught.exception.status,'timeout')
        resized.assert_not_called()

    def test_structure_capacity_failure_continues_to_next_grid(self):
        second=dict(self.record,device_width=98,device_height=98,structure_prefix='w98')
        with patch.object(recovery,'prepare_input',return_value=self.record), \
             patch.object(recovery,'resource_failure_before',return_value={'old':'resource_error'}), \
             patch.object(recovery,'resize_input',side_effect=[self.capacity_error(),second]) as resize, \
             patch.object(recovery,'qualify_placement') as gate, \
             patch.object(recovery,'calibrate_record',side_effect=lambda a,r,s,p:self.boundary(r)):
            value=recovery.prepare_design(self.a,self.record,self.source)
        self.assertEqual(value['device_width'],98)
        self.assertEqual([c.args[2] for c in resize.call_args_list],[94,98])
        gate.assert_called_once()

    def test_device_growth_is_bounded(self):
        with patch.object(recovery,'prepare_input',return_value=self.record), \
             patch.object(recovery,'resource_failure_before',return_value=None), \
             patch.object(recovery,'resize_input',side_effect=lambda a,r,w:dict(r,device_width=w,device_height=w)) as resized, \
             patch.object(recovery,'qualify_placement',side_effect=self.capacity_error()):
            with self.assertRaises(PreparationFailure) as caught:
                recovery.prepare_design(self.a,self.record,self.source)
        self.assertEqual(caught.exception.status,'device_capacity_failure')
        self.assertEqual([c.args[2] for c in resized.call_args_list],[94,98,106,122])

    def test_complete_boundary_reuse_avoids_vpr(self):
        self.source['boundaries']['bwave']=self.boundary(self.record)
        with patch.object(recovery,'prepare_input',return_value=self.record), \
             patch.object(recovery,'qualify_placement') as gate,patch.object(recovery,'calibrate_record') as calibrate:
            result=recovery.prepare_design(self.a,self.record,self.source)
        gate.assert_not_called();calibrate.assert_not_called()
        self.assertEqual(result['widths'],[80,84,88])

    def test_same_grid_partial_observations_reused_and_new_grid_not_mixed(self):
        row=dict(status='unroutable',seed=401,width=48)
        self.source['observations']['bwave',48,401]=row
        self.assertEqual(recovery.old_observation(self.a,self.source,self.record,48,401),row)
        bigger=dict(self.record,device_width=94,device_height=94)
        self.assertIsNone(recovery.old_observation(self.a,self.source,bigger,48,401))
        self.source['observations']['bwave',48,401]=dict(row,status='timeout')
        self.assertIsNone(recovery.old_observation(self.a,self.source,self.record,48,401))

    def test_calibration_timeout_budget_separate_and_complete_resume_free(self):
        self.source['observations']['bwave',48,401]=dict(status='unroutable',width=48,seed=401)
        calls=[]
        def route(a,record,width,seed,folder):
            calls.append((a.timeout,width,seed))
            return dict(status='success' if width>=(80 if seed==401 else 84) else 'unroutable')
        root=self.root/'screen/bwave/g90x90'
        with patch.object(recovery.e,'route',side_effect=route),patch.object(recovery,'prune_trial'):
            result=recovery.calibrate_record(self.a,self.record,self.source,root)
            count=len(calls)
            self.assertEqual(recovery.calibrate_record(self.a,self.record,self.source,root),result)
            self.assertEqual(len(calls),count)
        self.assertEqual(result['reference_width'],84)
        self.assertNotIn((14400,48,401),calls)
        self.assertTrue(all(c[0]==14400 for c in calls))
        self.assertEqual(self.a.timeout,7200)

    def test_whole_cohort_resume_keeps_cached_grid_and_new_grid_separate(self):
        import inventory
        cached=dict(design='cached',family='cached',width=80,widths=[80],device_width=20,device_height=20)
        record=dict(self.record,design='new_design')
        inv=dict(designs=[dict(design='cached',family='cached'),record])
        self.a.vtr=self.root/'vtr';self.a.source=self.root/'source'
        self.a.checkpoint_metadata={};self.a.designs=None;self.a.scope='all'
        self.a.recovery_source=self.root/'old';self.a.prepare_jobs=2;self.a.rr_jobs=2
        source=dict(self.source,contexts=[cached])
        new_boundary=dict(record,device_width=98,device_height=98,widths=[84],reference_width=84)
        calls=[]
        def build(task):
            args,records,width=task
            calls.append((args.timeout,records[0]['design'],records[0]['device_width'],width))
            return [dict(records[0],width=width)]
        with patch.object(inventory,'discover',return_value=inv),patch.object(recovery,'load_recovery',return_value=source), \
             patch.object(recovery,'prepare_design',return_value=new_boundary) as prepare_design, \
             patch.object(recovery,'graph_group_worker',side_effect=build),patch.object(recovery,'check_free_space'), \
             patch.object(cf,'ProcessPoolExecutor',side_effect=lambda max_workers,mp_context:cf.ThreadPoolExecutor(max_workers=max_workers)):
            result=recovery.prepare(self.a)
        self.assertTrue(result['report']['complete'])
        self.assertEqual(result['report']['reused_context_count'],1)
        self.assertEqual(prepare_design.call_count,1)
        self.assertEqual(calls,[(14400,'new_design',98,84)])
        self.assertEqual([(c['design'],c['device_width']) for c in result['contexts']],[('cached',20),('new_design',98)])
        self.assertEqual(self.a.timeout,7200)


if __name__=='__main__':
    unittest.main()
