import copy
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import continue_tailcompile_v610_ready as admission
import test_retry_tailcompile_v610_preparation as retry_tests


class PreparationAdmissionRepairTest(unittest.TestCase):
    def setUp(self):
        retry_tests.PreparationTimeoutRepairTest.setUp(self)
        patch.object(admission,'REPO',self.repo).start()
        run='experiments/vtr/tailcompile_v610/run.py'
        path=self.repo/run;path.write_text('new coverage reporting')
        self.current[run]=admission.sha(path)
        self.protocol['code'][run]=admission.RELEASED[run]
        new='experiments/vtr/tailcompile_v610/preparation_admission.py'
        path=self.repo/new;path.write_text('audited preparation admission')
        self.current[new]=admission.sha(path)
        self.names=['dla_like.large','bwave_like.float.large']
        self.report.update(inventory_design_count=3,reused_design_count=0,prepared_designs=1)
        admission.write(self.out/'protocol.json',self.protocol)
        admission.write(self.out/'preparation_report.json',self.report)
        admission.write(self.out/'prepared_inputs.json',{'designs':[{'design':'successful'}]})
        admission.write(self.out/'contexts.partial.json',{'contexts':[]})

    def test_preview_keeps_all_evidence_and_full_inventory(self):
        before={str(p.relative_to(self.out)):admission.sha(p) for p in self.out.rglob('*') if p.is_file()}
        plan=admission.make_plan(self.out,self.names)
        self.assertEqual(plan['audit']['coverage']['admitted_design_count'],1)
        self.assertEqual(plan['audit']['coverage']['inventory_design_count'],3)
        self.assertEqual(plan['audit']['coverage']['deferred_design_count'],2)
        self.assertEqual(before,{str(p.relative_to(self.out)):admission.sha(p) for p in self.out.rglob('*') if p.is_file()})
        self.assertEqual(plan['admission_plan']['policy'],'defer_recorded_preparation_timeouts')

    def test_apply_keeps_successes_and_timeouts_and_loads_runtime(self):
        protected=[*self.trials,self.out/'screen/successful/qualified_input.json',
                   self.out/'inventory.json',self.out/'prepared_inputs.json',self.out/'preparation_report.json']
        digests={str(p):admission.sha(p) for p in protected}
        done=admission.apply_plan(self.out,admission.make_plan(self.out,self.names))
        self.assertTrue(done['complete'])
        self.assertEqual(digests,{str(p):admission.sha(p) for p in protected})
        actual=admission.read(self.out/'protocol.json')
        self.assertEqual({k:v for k,v in actual.items() if k!='code'},
                         {k:v for k,v in self.protocol.items() if k!='code'})
        self.assertEqual(actual['timeout'],7200)
        self.assertEqual(admission.apply_plan(self.out,done),done)
        import preparation_admission as runtime
        with patch.object(runtime,'REPO',self.repo):
            self.assertEqual(runtime.load_admission_plan(self.out),done['admission_plan'])
        # Subsequent pipeline progress does not break completed repair idempotency.
        admission.write(self.out/'preparation_report.json',{'complete':True})
        admission.write(self.out/'contexts.json',{'contexts':[]})
        self.assertEqual(admission.apply_plan(self.out,done),done)

    def test_interrupted_migration_resumes_both_write_orders(self):
        plan=admission.make_plan(self.out,self.names)
        journal=self.out/'_repairs/preparation_admission/repair.json'
        admission.write(journal,plan)
        admission.write(self.out/'preparation_admission.json',plan['admission_plan'])
        self.assertTrue(admission.apply_plan(self.out,plan)['complete'])
        admission.write(journal,plan)
        (self.out/'preparation_admission.json').unlink()
        self.assertTrue(admission.apply_plan(self.out,plan)['complete'])

    def test_unknown_failure_or_unrequested_deletion_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'match all current'):
            admission.make_plan(self.out,[self.names[0]])
        with self.assertRaisesRegex(ValueError,'inventory'):
            admission.make_plan(self.out,[*self.names,'missing'])
        report=copy.deepcopy(self.report);report['failures'][0]['status']='killed'
        admission.write(self.out/'preparation_report.json',report)
        with self.assertRaisesRegex(ValueError,'recorded preparation timeout'):
            admission.make_plan(self.out,self.names)

    def test_partition_must_account_for_every_benchmark(self):
        admission.write(self.out/'prepared_inputs.json',{'designs':[]})
        with self.assertRaisesRegex(ValueError,'partition'):
            admission.make_plan(self.out,self.names)
        admission.write(self.out/'prepared_inputs.json',{'designs':[{'design':'successful'},{'design':self.names[0]}]})
        with self.assertRaisesRegex(ValueError,'partition'):
            admission.make_plan(self.out,self.names)

    def test_unrelated_code_or_frozen_weights_rejected(self):
        self.current['other.py']='changed'
        with self.assertRaisesRegex(ValueError,'Unrelated frozen implementation'):
            admission.make_plan(self.out,self.names)
        self.current['other.py']=self.protocol['code']['other.py']
        Path(self.protocol['checkpoint']).write_bytes(b'new weights')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):
            admission.make_plan(self.out,self.names)

    def test_changed_budget_or_timeout_proof_rejected(self):
        plan=admission.make_plan(self.out,self.names)
        tampered=copy.deepcopy(plan);tampered['new_protocol']['timeout']=14400
        with self.assertRaisesRegex(ValueError,'scientific settings'):
            admission.apply_plan(self.out,tampered)
        plan['admission_plan']['deferred'][0]['trial_sha256']='wrong'
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            admission.apply_plan(self.out,plan)

    def test_changed_success_certificate_prevents_pending_repair(self):
        plan=admission.make_plan(self.out,self.names)
        admission.write(self.out/'screen/successful/qualified_input.json',{'changed':True})
        with self.assertRaisesRegex(ValueError,'Qualified evidence changed'):
            admission.apply_plan(self.out,plan)

    def test_prepared_or_active_cohort_cannot_be_migrated(self):
        admission.write(self.out/'contexts.json',{})
        with self.assertRaisesRegex(ValueError,'no longer allowed'):
            admission.make_plan(self.out,self.names)
        (self.out/'contexts.json').unlink()
        admission.write(self.out/'inputs/active/synthesis/started.json',{})
        with self.assertRaisesRegex(ValueError,'Interrupted/active'):
            admission.make_plan(self.out,self.names)

    def test_budget_retry_plan_cannot_be_mixed_with_admission(self):
        admission.write(self.out/'preparation_retry.json',{})
        with self.assertRaisesRegex(ValueError,'budget retry'):
            admission.make_plan(self.out,self.names)

    def test_nonoriginal_route_or_wrong_budget_rejected(self):
        path=self.trials[1]
        trial=admission.read(path);trial['identity']['teacher']=True
        admission.write(path,trial)
        with self.assertRaisesRegex(ValueError,'uninitialized Default'):
            admission.make_plan(self.out,self.names)
        trial['identity']['teacher']=False;trial['timeout']=100
        admission.write(path,trial)
        with self.assertRaisesRegex(ValueError,'frozen budget'):
            admission.make_plan(self.out,self.names)

    def test_broad_root_rejected(self):
        self.assertEqual(admission.validate_root(self.out),self.out)
        with self.assertRaises(ValueError):
            admission.validate_root(self.out.parent)

    @unittest.skipUnless(os.name=='posix','Linux flock required')
    def test_running_pipeline_lock_rejected(self):
        with admission.cohort_lock(self.out):
            with self.assertRaisesRegex(RuntimeError,'still running'):
                with admission.cohort_lock(self.out):
                    pass


if __name__=='__main__':
    unittest.main()
