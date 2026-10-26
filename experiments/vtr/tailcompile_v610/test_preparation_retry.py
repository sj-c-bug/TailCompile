from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge import fingerprint, read, sha, write
import preparation_retry as pr
import recovery
import recovery_inputs as ri


class PreparationRetryTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name).resolve()
        self.out=self.root/'cohort'
        self.a=SimpleNamespace(out=self.out,timeout=7200,synthesis_timeout=21600,
            calibration_timeout=14400,calibration_seeds=[401,402],width_start=48,
            width_max=128,width_step=4,half_band=3,identity={'vtr_commit':'frozen'},
            retry_failed=False,retry_timeouts=False,heartbeat=60)
        self.protocol=dict(schema='test',code={},timeout=7200,
            preparation_policy=dict(synthesis_timeout=21600,calibration_timeout=14400))

    def authorize(self,stage='calibration',design='toy'):
        folder=(self.out/'inputs'/design/'synthesis' if stage=='synthesis' else
                self.out/'screen'/design/'g20x20'/'calibration'/'w48'/'s401')
        previous=21600 if stage=='synthesis' else 14400
        write(folder/'trial.json',dict(status='timeout',returncode=124,timeout=previous,
            identity=dict(design=design,stage='synthesis' if stage=='synthesis' else 'route')))
        (folder/'vpr.out').write_text('original timeout log\n')
        item=dict(design=design,stage=stage,previous_timeout=previous,new_timeout=previous*2,
            failed_directory=str(folder),failed_trial_sha256=sha(folder/'trial.json'))
        plan=dict(schema=pr.SCHEMA,out=str(self.out),reason='Observed preparation timeout',
            base_protocol_fingerprint=pr.base_fingerprint(self.protocol),overrides=[item],created_unix=1)
        write(self.out/'protocol.json',self.protocol)
        write(self.out/pr.PLAN_NAME,plan)
        write(self.out/pr.JOURNAL_NAME,dict(complete=True,retry_plan=plan,new_protocol=self.protocol))
        return plan,item

    def test_stage_override_leaves_final_and_other_design_budgets_unchanged(self):
        self.authorize()
        override=pr.override_for(self.a,'toy','calibration')
        self.assertEqual(override['new_timeout'],28800)
        self.assertIsNone(pr.override_for(self.a,'toy','synthesis'))
        self.assertIsNone(pr.override_for(self.a,'other','calibration'))
        self.assertEqual((self.a.timeout,self.a.synthesis_timeout,self.a.calibration_timeout),
                         (7200,21600,14400))

    def test_retry_requires_matching_completed_audit_and_immutable_timeout(self):
        plan,item=self.authorize()
        pr.load_retry_plan(self.a)
        path=Path(item['failed_directory'])/'trial.json'
        saved=read(path);saved['status']='success';write(path,saved)
        with self.assertRaisesRegex(ValueError,'evidence changed'):
            pr.load_retry_plan(self.a)
        write(path,dict(status='timeout',returncode=124,timeout=14400,
            identity=dict(design='toy',stage='route')))
        plan['reason']='edited without audit';write(self.out/pr.PLAN_NAME,plan)
        with self.assertRaisesRegex(ValueError,'migration journal'):
            pr.load_retry_plan(self.a)

    def test_missing_plan_with_journal_is_rejected(self):
        self.assertIsNone(pr.load_retry_plan(self.a))
        self.authorize()
        (self.out/pr.PLAN_NAME).unlink()
        with self.assertRaisesRegex(ValueError,'plan is missing'):
            pr.load_retry_plan(self.a)

    def test_formal_budget_cannot_change_even_with_matching_plan_journal(self):
        plan,_=self.authorize()
        changed=dict(self.protocol,timeout=14400)
        write(self.out/'protocol.json',changed)
        write(self.out/pr.JOURNAL_NAME,dict(complete=True,retry_plan=plan,new_protocol=changed))
        with self.assertRaisesRegex(ValueError,'base scientific protocol'):
            pr.load_retry_plan(self.a)

    def test_retry_cannot_target_non_timeout_or_different_stage_evidence(self):
        plan,item=self.authorize()
        trial_path=Path(item['failed_directory'])/'trial.json'
        trial=read(trial_path);trial['identity']['stage']='synthesis';write(trial_path,trial)
        item['failed_trial_sha256']=sha(trial_path)
        write(self.out/pr.PLAN_NAME,plan)
        write(self.out/pr.JOURNAL_NAME,dict(complete=True,retry_plan=plan,new_protocol=self.protocol))
        with self.assertRaisesRegex(ValueError,'another stage/benchmark'):
            pr.load_retry_plan(self.a)

    def test_known_outcomes_keep_original_paths_and_unknown_work_uses_new_budget_path(self):
        _,item=self.authorize()
        root=self.out/'screen/toy/g20x20'
        before=(root/'calibration/w48/s401/trial.json').read_bytes()
        folder,extended=pr.calibration_destination(root,48,401,item)
        self.assertTrue(extended)
        self.assertEqual(folder,root/'calibration_retry_28800/w48/s401')
        known=root/'calibration/w72/s401'
        write(known/'trial.json',dict(status='success'))
        self.assertEqual(pr.calibration_destination(root,72,401,item),(known,False))
        self.assertEqual(pr.calibration_destination(root,52,402,item),
                         (root/'calibration_retry_28800/w52/s402',True))
        self.assertEqual(before,(root/'calibration/w48/s401/trial.json').read_bytes())
        with self.assertRaisesRegex(ValueError,'original qualified device grid'):
            pr.calibration_destination(self.out/'screen/toy/g24x24',48,401,item)

    def test_synthesis_retry_preserves_original_flow_and_uses_fresh_attempt(self):
        _,item=self.authorize('synthesis')
        vtr=self.root/'vtr'
        source=vtr/'vtr_flow/benchmarks/verilog/toy.v'
        source.parent.mkdir(parents=True);source.write_text('module toy; endmodule\n')
        self.a.vtr=vtr;self.a.arch=vtr/'architecture.xml';self.a.binary=vtr/'vpr'
        record=dict(design='toy',source='toy.v',source_sha256=sha(source),include_sha256=None)
        oldflow=self.out/'inputs/toy/mapping';oldflow.mkdir()
        (oldflow/'abc.out').write_text('old partial synthesis\n')
        oldtrial=(Path(item['failed_directory'])/'trial.json').read_bytes()
        calls=[]
        def call(a,command,folder,signature,**kwargs):
            calls.append((a.timeout,command,folder,signature))
            flow=Path(command[command.index('-temp_dir')+1]);flow.mkdir(parents=True)
            (flow/'toy.abc.blif').write_text('.model toy\n.end\n')
            write(folder/'trial.json',dict(status='success',returncode=0))
            return dict(status='success',returncode=0)
        with patch.object(ri.e,'call',side_effect=call):
            result=ri._canonical_mapping(self.a,record)
            resumed=ri._canonical_mapping(self.a,record)
        self.assertEqual(result,resumed)
        self.assertEqual(len(calls),1)
        self.assertEqual(calls[0][0],43200)
        self.assertEqual(calls[0][2],self.out/'inputs/toy/synthesis_retry_43200')
        self.assertIn('preparation_retry',calls[0][3])
        self.assertEqual((oldflow/'abc.out').read_text(),'old partial synthesis\n')
        self.assertEqual(oldtrial,(Path(item['failed_directory'])/'trial.json').read_bytes())
        self.assertEqual(self.a.timeout,7200)

    def test_calibration_retry_reuses_known_results_and_audits_actual_budget(self):
        _,item=self.authorize()
        root=self.out/'screen/toy/g20x20'
        known=root/'calibration/w72/s401'
        write(known/'trial.json',dict(status='success',returncode=0,timeout=14400))
        oldtimeout=(Path(item['failed_directory'])/'trial.json').read_bytes()
        knownbytes=(known/'trial.json').read_bytes()
        record=dict(design='toy',mapped_blif_sha256='blif',source_sha256='source',
            device_width=20,device_height=20)
        source=dict(inputs={},observations={})
        calls=[]
        def route(a,record,width,seed,folder,extra=None):
            calls.append((a.timeout,width,seed,folder,extra))
            path=folder/'trial.json'
            if path.exists():
                return read(path)
            result=dict(status='success' if width>=52 else 'unroutable',returncode=0,timeout=a.timeout)
            write(path,result)
            return result
        with patch.object(recovery.e,'route',side_effect=route),patch.object(recovery,'prune_trial'):
            result=recovery.calibrate_record(self.a,record,source,root)
        self.assertEqual(result['reference_width'],52)
        original=[c for c in calls if c[3]==known]
        self.assertEqual(len(original),1)
        self.assertEqual(original[0][0],14400)
        retried=[c for c in calls if c[3]!=known]
        self.assertTrue(retried)
        self.assertTrue(all(c[0]==28800 and c[4]['preparation_retry'] for c in retried))
        self.assertEqual(result['preparation_retry']['authorization']['new_timeout'],28800)
        self.assertEqual(oldtimeout,(Path(item['failed_directory'])/'trial.json').read_bytes())
        self.assertEqual(knownbytes,(known/'trial.json').read_bytes())
        self.assertEqual(self.a.timeout,7200)


if __name__=='__main__':
    unittest.main()
