import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import retry_tailcompile_v610_preparation as repair


class PreparationTimeoutRepairTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.object(repair,'REPO',self.repo).start()
        self.out=self.repo/'experiments/vtr/tailcompile_v610/runs/cohort'
        self.out.mkdir(parents=True)
        (self.out/'.lock').touch()
        current={}
        for name in [*repair.RELEASED,repair.ADDED,'other.py']:
            path=self.repo/name;path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('current '+name,encoding='utf-8')
            current[name]=repair.sha(path)
        self.current=current
        patch.object(repair,'current_implementation',return_value=current).start()
        self.verify_source=patch.object(repair,'verify_source').start()
        checkpoint=self.repo/'frozen.pt';checkpoint.write_bytes(b'frozen weights')
        self.protocol=dict(schema='tailcompile-v610-frozen-sweep-1',training_permitted=False,
            checkpoint=str(checkpoint),checkpoint_sha256=repair.sha(checkpoint),timeout=7200,
            identity={'vtr_commit':'fixed','binary_sha256':'binary','architecture_sha256':'arch'},
            code=dict(current,**repair.RELEASED),calibration_seeds=[401,402],
            preparation_policy=dict(synthesis_timeout=21600,calibration_timeout=14400,final_route_timeout=7200),
            recovery_source={'source':'old cohort','files':{'metadata':'digest'}})
        self.protocol['code'].pop(repair.ADDED)
        self.inventory=dict(designs=[{'design':name} for name in ('dla_like.large','bwave_like.float.large','successful')])
        self.wanted={('dla_like.large','synthesis'):86400.,('bwave_like.float.large','calibration'):28800.}
        self.failures=[]
        self.trials=[]
        for name,stage in self.wanted:
            if stage=='synthesis':
                folder=self.out/'inputs'/name/'synthesis'
            else:
                folder=self.out/'screen'/name/'g198x198/calibration/w164/s401'
            budget=self.protocol['preparation_policy'][stage+'_timeout']
            command=['python','flow.py'] if stage=='synthesis' else ['vpr','architecture.xml','mapped.blif']
            identity=dict(self.protocol['identity'],design=name,stage='route' if stage=='calibration' else stage,
                command=command,teacher=False,initializer_sha256=None)
            trial=dict(status='timeout',returncode=124,timeout=budget,identity=identity,command=command)
            repair.write(folder/'trial.json',trial)
            (folder/'vpr.out').write_text('timeout evidence')
            self.trials.append(folder/'trial.json')
            self.failures.append(dict(design=name,stage='recovery_qualification',status='timeout',directory=str(folder)))
        self.report=dict(schema='tailcompile-v610-recovery-preparation-1',complete=False,failures=self.failures,
            reused_design_count=27,prepared_designs=15)
        repair.write(self.out/'protocol.json',self.protocol)
        repair.write(self.out/'inventory.json',self.inventory)
        repair.write(self.out/'preparation_report.json',self.report)
        repair.write(self.out/'screen/successful/qualified_input.json',{'success':'must stay untouched'})

    def test_preview_does_not_write_any_files(self):
        before={str(p.relative_to(self.out)):repair.sha(p) for p in self.out.rglob('*') if p.is_file()}
        plan=repair.make_plan(self.out,self.wanted)
        self.assertEqual(len(plan['retry_plan']['overrides']),2)
        self.assertEqual(before,{str(p.relative_to(self.out)):repair.sha(p) for p in self.out.rglob('*') if p.is_file()})

    def test_apply_changes_only_code_and_adds_audited_retry_plan(self):
        success=self.out/'screen/successful/qualified_input.json'
        protected=[*self.trials,success,self.out/'inventory.json',self.out/'preparation_report.json']
        digests={str(p):repair.sha(p) for p in protected}
        done=repair.apply_plan(self.out,repair.make_plan(self.out,self.wanted))
        self.assertTrue(done['complete'])
        actual=repair.read(self.out/'protocol.json')
        self.assertEqual({k:v for k,v in actual.items() if k!='code'},
                         {k:v for k,v in self.protocol.items() if k!='code'})
        self.assertEqual(actual['code'],self.current)
        self.assertEqual(digests,{str(p):repair.sha(p) for p in protected})
        self.assertEqual(repair.read(self.out/'preparation_retry.json'),done['retry_plan'])
        self.assertEqual(repair.apply_plan(self.out,done),done)
        import preparation_retry as runtime
        with patch.object(runtime,'REPO',self.repo):
            self.assertEqual(runtime.load_retry_plan(self.out),done['retry_plan'])

    def test_interrupted_writes_resume_in_either_order(self):
        plan=repair.make_plan(self.out,self.wanted)
        repair.write(self.out/'_repairs/preparation_timeouts/repair.json',plan)
        repair.write(self.out/'preparation_retry.json',plan['retry_plan'])
        self.assertTrue(repair.apply_plan(self.out,plan)['complete'])
        # A protocol-first interruption is also recoverable.
        repair.write(self.out/'protocol.json',plan['new_protocol'])
        (self.out/'preparation_retry.json').unlink()
        repair.write(self.out/'_repairs/preparation_timeouts/repair.json',plan)
        self.assertTrue(repair.apply_plan(self.out,plan)['complete'])

    def test_unrelated_code_checkpoint_and_source_are_rejected(self):
        self.current['other.py']='changed'
        with self.assertRaisesRegex(ValueError,'Unrelated frozen implementation'):
            repair.make_plan(self.out,self.wanted)
        self.current['other.py']=self.protocol['code']['other.py']
        Path(self.protocol['checkpoint']).write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):
            repair.make_plan(self.out,self.wanted)
        Path(self.protocol['checkpoint']).write_bytes(b'frozen weights')
        self.verify_source.side_effect=ValueError('source changed')
        with self.assertRaisesRegex(ValueError,'source changed'):
            repair.make_plan(self.out,self.wanted)

    def test_unknown_old_release_or_new_module_rejected(self):
        key=next(iter(repair.RELEASED))
        self.protocol['code'][key]='unrecognized'
        repair.write(self.out/'protocol.json',self.protocol)
        with self.assertRaisesRegex(ValueError,'Unrecognized released'):
            repair.make_plan(self.out,self.wanted)
        self.protocol['code'][key]=repair.RELEASED[key]
        repair.write(self.out/'protocol.json',self.protocol)
        self.current['extra.py']='unexpected'
        with self.assertRaisesRegex(ValueError,'Unexpected added'):
            repair.make_plan(self.out,self.wanted)

    def test_final_timeout_and_retry_trial_tampering_rejected(self):
        plan=repair.make_plan(self.out,self.wanted)
        tampered=copy.deepcopy(plan);tampered['new_protocol']['timeout']=86400
        with self.assertRaisesRegex(ValueError,'scientific protocol'):
            repair.apply_plan(self.out,tampered)
        trial=repair.read(self.trials[0]);trial['timeout']=22000
        repair.write(self.trials[0],trial)
        with self.assertRaisesRegex(ValueError,'original preparation timeout'):
            repair.apply_plan(self.out,plan)

    def test_changed_report_and_inventory_rejected(self):
        plan=repair.make_plan(self.out,self.wanted)
        repair.write(self.out/'preparation_report.json',dict(self.report,complete=True))
        with self.assertRaisesRegex(ValueError,'report changed'):
            repair.apply_plan(self.out,plan)
        repair.write(self.out/'preparation_report.json',self.report)
        repair.write(self.out/'inventory.json',dict(designs=[]))
        with self.assertRaisesRegex(ValueError,'inventory changed'):
            repair.apply_plan(self.out,plan)

    def test_late_repair_or_live_attempt_rejected(self):
        repair.write(self.out/'prepared_signature.json',{})
        with self.assertRaisesRegex(ValueError,'no longer allowed'):
            repair.make_plan(self.out,self.wanted)
        (self.out/'prepared_signature.json').unlink()
        repair.write(self.out/'inputs/running/synthesis/started.json',{})
        with self.assertRaisesRegex(ValueError,'Interrupted/active'):
            repair.make_plan(self.out,self.wanted)

    def test_missing_or_wrong_stage_failure_rejected(self):
        with self.assertRaisesRegex(ValueError,'exactly one'):
            repair.make_plan(self.out,{('successful','synthesis'):86400})
        with self.assertRaisesRegex(ValueError,'stage does not match'):
            repair.make_plan(self.out,{('dla_like.large','calibration'):28800})
        report=copy.deepcopy(self.report);report['failures'][0]['status']='preparation_failure'
        repair.write(self.out/'preparation_report.json',report)
        with self.assertRaisesRegex(ValueError,'reported timeout'):
            repair.make_plan(self.out,self.wanted)

    def test_equal_or_unbounded_retry_budget_rejected(self):
        with self.assertRaisesRegex(ValueError,'strictly exceed'):
            repair.make_plan(self.out,{('dla_like.large','synthesis'):21600})
        for value in ['name=nan','name=inf','name=0','name=604801','name=30000.5','../escape=100','name']:
            with self.subTest(value=value),self.assertRaises(ValueError):
                repair.parse_override(value)
        with self.assertRaisesRegex(ValueError,'Duplicate'):
            repair.requests(['name=30000','name=40000'],[])

    def test_corrupt_plan_budget_or_evidence_rejected(self):
        plan=repair.make_plan(self.out,self.wanted)
        plan['retry_plan']['overrides'][0]['failed_trial_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            repair.apply_plan(self.out,plan)
        plan=repair.make_plan(self.out,self.wanted)
        plan['retry_plan']['overrides'][0]['new_timeout']=repair.MAX_TIMEOUT+1
        with self.assertRaisesRegex(ValueError,'seven days'):
            repair.apply_plan(self.out,plan)

    def test_root_scope_rejected(self):
        self.assertEqual(repair.validate_root(self.out),self.out)
        with self.assertRaises(ValueError):
            repair.validate_root(self.out.parent)

    @unittest.skipUnless(os.name=='posix','Linux flock required')
    def test_running_cohort_lock_is_rejected(self):
        with repair.cohort_lock(self.out):
            with self.assertRaisesRegex(RuntimeError,'still running'):
                with repair.cohort_lock(self.out):
                    pass


if __name__=='__main__':
    unittest.main()
