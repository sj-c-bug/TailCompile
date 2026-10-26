from pathlib import Path
import concurrent.futures as cf
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge import read,sha,write
import preparation_admission as pa
import recovery
from prepare import PreparationFailure


class PreparationAdmissionTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.out=self.root/'cohort'
        self.a=SimpleNamespace(out=self.out,timeout=7200,synthesis_timeout=21600,
            calibration_timeout=14400,calibration_seeds=[401,402],width_start=48,
            width_max=128,width_step=4,half_band=3,identity={'vtr_commit':'frozen'},
            retry_failed=False,retry_timeouts=False,heartbeat=60,device_width_steps=[0,4,8,16,32],
            vtr=self.root/'vtr',source=self.root/'source',checkpoint_metadata={},designs=None,
            scope='all',recovery_source=self.root/'old',prepare_jobs=2,rr_jobs=2)
        self.protocol=dict(schema='test',code={},timeout=7200,
            preparation_policy=dict(synthesis_timeout=21600,calibration_timeout=14400))

    def timeout(self,design='slow',stage='synthesis'):
        folder=(self.out/'inputs'/design/'synthesis' if stage=='synthesis' else
            self.out/'screen'/design/'g20x20/calibration/w48/s401')
        write(folder/'trial.json',dict(status='timeout',returncode=124,timeout=21600 if stage=='synthesis' else 14400,
            identity=dict(design=design,stage='synthesis' if stage=='synthesis' else 'route')))
        return folder

    def authorize(self):
        folder=self.timeout()
        item=pa.timeout_evidence(self.out,'slow',folder)
        plan=dict(schema=pa.SCHEMA,out=str(self.out),policy=pa.POLICY,reason='Finish ready benchmarks first',
            base_protocol_fingerprint=pa.base_fingerprint(self.protocol),deferred=[item],created_unix=1)
        write(self.out/'protocol.json',self.protocol)
        write(self.out/pa.PLAN_NAME,plan)
        write(self.out/pa.JOURNAL_NAME,dict(complete=True,admission_plan=plan,new_protocol=self.protocol))
        return plan

    def mocks(self,inventory,source,prepare,build):
        import inventory as inv
        from contextlib import ExitStack
        stack=ExitStack()
        stack.enter_context(patch.object(inv,'discover',return_value=inventory))
        stack.enter_context(patch.object(recovery,'load_recovery',return_value=source))
        stack.enter_context(patch.object(recovery,'prepare_design',side_effect=prepare))
        stack.enter_context(patch.object(recovery,'graph_group_worker',side_effect=build))
        stack.enter_context(patch.object(recovery,'check_free_space'))
        stack.enter_context(patch.object(cf,'ProcessPoolExecutor',side_effect=lambda max_workers,mp_context:
            cf.ThreadPoolExecutor(max_workers=max_workers)))
        return stack

    def test_explicit_timeouts_are_unknown_and_do_not_block_ready_graphs(self):
        self.authorize()
        cached=dict(design='cached',family='cached',width=80,widths=[80],device_width=20,device_height=20)
        good=dict(design='good',family='good',widths=[84],device_width=24,device_height=24)
        inventory=dict(designs=[dict(design=name,family=name) for name in ('cached','good','slow')])
        source=dict(contexts=[cached])
        calls=[]
        def prepare(a,record,recovery_source):
            calls.append(record['design'])
            self.assertEqual(record['design'],'good')
            return good
        with self.mocks(inventory,source,prepare,lambda task:[dict(good,width=task[2])]):
            result=recovery.prepare(self.a)
        report=result['report']
        self.assertTrue(report['complete']);self.assertFalse(report['coverage_complete'])
        self.assertEqual((report['requested_design_count'],report['eligible_design_count'],report['deferred_design_count']),
                         (3,2,1))
        self.assertEqual(calls,['good'])
        self.assertEqual([row['design'] for row in result['contexts']],['cached','good'])
        slow=next(row for row in report['benchmark_status'] if row['design']=='slow')
        self.assertEqual(slow['status'],'deferred');self.assertEqual(slow['routability'],'unknown')
        self.assertEqual(read(self.out/'inventory.json'),inventory)
        self.assertEqual(report['excluded'],[]);self.assertEqual(report['failures'],[])

    def test_new_qualification_timeout_is_deferred_and_restart_does_not_retry_it(self):
        self.authorize()
        folder=self.timeout('new_slow')
        cached=dict(design='cached',family='cached',width=80,widths=[80],device_width=20,device_height=20)
        good=dict(design='good',family='good',widths=[84],device_width=24,device_height=24)
        inventory=dict(designs=[dict(design=name,family=name) for name in ('cached','good','slow','new_slow')])
        calls=[]
        def prepare(a,record,recovery_source):
            calls.append(record['design'])
            if record['design']=='new_slow':
                raise PreparationFailure('timeout','deadline',folder)
            return good
        with self.mocks(inventory,dict(contexts=[cached]),prepare,lambda task:[dict(good,width=task[2])]):
            first=recovery.prepare(self.a)
            second=recovery.prepare(self.a)
        self.assertEqual(calls.count('new_slow'),1)
        self.assertTrue(first['report']['complete']);self.assertTrue(second['report']['complete'])
        self.assertEqual(second['report']['deferred_design_count'],2)
        self.assertEqual(set(pa.load_deferred(self.a)),{'slow','new_slow'})
        self.assertEqual(self.a.timeout,7200)

    def test_rrg_timeout_defers_affected_design_and_keeps_other_devices_running(self):
        self.authorize()
        cached=dict(design='cached',family='cached',width=80,widths=[80],device_width=20,device_height=20)
        records={name:dict(design=name,family=name,widths=[84],device_width=size,device_height=size)
                 for name,size in [('good',24),('graph_slow',28)]}
        records['graph_slow']['widths']=[80,84]
        inventory=dict(designs=[dict(design=name,family=name) for name in ('cached','good','graph_slow','slow')])
        def build(task):
            a,rows,width=task
            record=rows[0]
            if record['design']=='graph_slow' and width==84:
                folder=self.out/'device_cache/28x28/w84/build'
                write(folder/'trial.json',dict(status='timeout',returncode=124,timeout=14400,
                    identity=dict(stage='rr_graph',grid='28x28/w84')))
                raise RuntimeError(f'Pack-only RRG export failed: {folder}/vpr.out')
            return [dict(record,width=width)]
        with self.mocks(inventory,dict(contexts=[cached]),lambda a,r,s:records[r['design']],build):
            result=recovery.prepare(self.a)
        self.assertTrue(result['report']['complete'])
        self.assertEqual(result['report']['deferred_design_count'],2)
        self.assertEqual([row['design'] for row in result['contexts']],['cached','good'])
        self.assertEqual(pa.load_deferred(self.a)['graph_slow']['stage'],'rr_graph')

    def test_non_timeout_tool_failure_remains_fatal(self):
        self.authorize()
        inventory=dict(designs=[dict(design=name,family=name,source=name+'.v') for name in ('bad','slow')])
        def prepare(a,record,source):
            raise PreparationFailure('other_failure','library is missing',self.out/'inputs/bad/synthesis')
        with self.mocks(inventory,dict(contexts=[]),prepare,lambda task:[]):
            with self.assertRaisesRegex(RuntimeError,'1 failures'):
                recovery.prepare(self.a)
        report=read(self.out/'preparation_report.json')
        self.assertEqual(report['failures'][0]['status'],'other_failure')
        self.assertNotIn('bad',pa.load_deferred(self.a))

    def test_admission_refuses_tampered_evidence_and_protocol_changes(self):
        plan=self.authorize()
        self.assertEqual(len(pa.load_deferred(self.a)),1)
        trial=Path(plan['deferred'][0]['directory'])/'trial.json'
        data=read(trial);data['returncode']=1;write(trial,data)
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            pa.load_admission_plan(self.a)

    def test_restart_reuses_validated_graph_subset_without_changing_shared_consumer_proof(self):
        import prepare
        self.authorize()
        record=dict(design='good',device_width=20,device_height=20)
        cached=dict(record,width=80,widths=[80])
        write(self.out/'graphs/good/w80/sweep_graph.json',{'cached':'verified below'})
        proof=self.out/'device_cache/20x20/w80/consumers_complete.json'
        write(proof,dict(consumers=[dict(design='good'),dict(design='now_deferred')]))
        before=proof.read_bytes()
        with patch.object(prepare,'_cached_graph',return_value=cached) as validate,\
                patch.object(recovery,'graph_group_worker') as rebuild:
            result=recovery.admitted_graph_worker((self.a,[record],80))
        validate.assert_called_once_with(record,80,self.a)
        rebuild.assert_not_called()
        self.assertEqual(result,[cached])
        self.assertEqual(proof.read_bytes(),before)
        with patch.object(prepare,'_cached_graph',side_effect=ValueError('Corrupted graph/IR')):
            with self.assertRaisesRegex(ValueError,'Corrupted'):
                recovery.admitted_graph_worker((self.a,[record],80))


if __name__=='__main__':
    unittest.main()
