import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import repair_tailcompile_v610_inventory as repair


class InventoryRepairTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.repo=Path(self.tmp.name)
        self.addCleanup(patch.stopall)
        patch.object(repair,'REPO',self.repo).start()
        self.out=self.repo/'experiments/vtr/tailcompile_v610/runs/cohort'
        self.out.mkdir(parents=True)
        fixed=self.repo/repair.CODE_KEY;fixed.parent.mkdir(parents=True,exist_ok=True)
        fixed.write_text('fixed inventory')
        other=self.repo/'other.py';other.write_text('unchanged scientific code')
        cp=self.repo/'frozen.pt';cp.write_bytes(b'unchanged checkpoint')
        self.protocol=dict(schema='tailcompile-v610-frozen-sweep-1',training_permitted=False,
            checkpoint=str(cp),checkpoint_sha256=repair.sha(cp),source=str(self.repo/'source'),
            checkpoint_metadata={},designs=None,scope='all',timeout=7200,
            code={repair.CODE_KEY:repair.LEGACY_SHA256,'other.py':repair.sha(other)})
        real=dict(design='real',source='koios/real.v',source_sha256='abc')
        helpers=[dict(design=Path(p).stem,source=p,source_sha256='helper-sha') for p in sorted(repair.HELPER_SOURCES)]
        self.old=dict(designs=[real]+helpers,excluded=[],design_count=3,family_count=3,koios_design_count=3,scope='all')
        self.new=dict(self.old,designs=[real],excluded=[dict(source=r['source'],sha256=r['source_sha256'],reason='architecture_include') for r in helpers],
                      design_count=1,family_count=1,koios_design_count=1)
        self.report=dict(failures=[dict(design=h['design'],stage='input',status='preparation_failure') for h in helpers]+[
            dict(design='real',stage='calibration',status='timeout')])
        repair.write(self.out/'protocol.json',self.protocol)
        repair.write(self.out/'inventory.json',self.old)
        repair.write(self.out/'preparation_report.json',self.report)
        self.discover=patch.object(repair,'discover',return_value=self.new).start()

    def test_preview_is_read_only(self):
        before={name:repair.sha(self.out/name) for name in ('protocol.json','inventory.json','preparation_report.json')}
        plan=repair.make_plan(self.out,self.repo/'vtr')
        self.assertEqual(len(plan['removed_helpers']),2)
        self.assertEqual(before,{name:repair.sha(self.out/name) for name in before})
        self.assertFalse((self.out/'_repairs').exists())

    def test_apply_preserves_all_scientific_settings_and_trials(self):
        folder=self.out/'screen/real/w108/s401';folder.mkdir(parents=True)
        trial=folder/'trial.json';trial.write_text('{"status":"success"}')
        saved=repair.sha(trial)
        plan=repair.make_plan(self.out,self.repo/'vtr')
        done=repair.apply_plan(self.out,plan)
        protocol=repair.read(self.out/'protocol.json')
        self.assertTrue(done['complete'])
        self.assertEqual({k:v for k,v in protocol.items() if k!='code'},
                         {k:v for k,v in self.protocol.items() if k!='code'})
        self.assertEqual(repair.sha(trial),saved)
        self.assertEqual(repair.read(self.out/'preparation_report.json'),self.report)
        self.assertEqual(repair.read(self.out/'_repairs/helper_inventory/repair.json')['old_inventory'],self.old)

    def test_interrupted_two_file_repair_resumes(self):
        plan=repair.make_plan(self.out,self.repo/'vtr')
        repair.write(self.out/'_repairs/helper_inventory/repair.json',plan)
        repair.write(self.out/'inventory.json',plan['new_inventory'])
        self.assertTrue(repair.apply_plan(self.out,plan)['complete'])
        self.assertEqual(repair.read(self.out/'protocol.json'),plan['new_protocol'])

    def test_unrelated_code_or_checkpoint_change_rejected(self):
        (self.repo/'other.py').write_text('changed')
        with self.assertRaisesRegex(ValueError,'Unrelated frozen implementation'):
            repair.make_plan(self.out,self.repo/'vtr')
        (self.repo/'other.py').write_text('unchanged scientific code')
        Path(self.protocol['checkpoint']).write_bytes(b'changed weights')
        with self.assertRaisesRegex(ValueError,'checkpoint changed'):
            repair.make_plan(self.out,self.repo/'vtr')

    def test_real_design_metadata_change_rejected(self):
        bad=copy.deepcopy(self.new);bad['designs'][0]['source_sha256']='different'
        self.discover.return_value=bad
        with self.assertRaisesRegex(ValueError,'Runnable benchmark'):
            repair.make_plan(self.out,self.repo/'vtr')

    def test_postpreparation_and_active_evidence_rejected(self):
        repair.write(self.out/'prepared_signature.json',{})
        with self.assertRaisesRegex(ValueError,'no longer allowed'):
            repair.make_plan(self.out,self.repo/'vtr')
        (self.out/'prepared_signature.json').unlink()
        repair.write(self.out/'inputs/real/synthesis/started.json',{})
        with self.assertRaisesRegex(ValueError,'Interrupted/active'):
            repair.make_plan(self.out,self.repo/'vtr')

    def test_repair_journal_cannot_change_timeout(self):
        plan=repair.make_plan(self.out,self.repo/'vtr')
        plan['new_protocol']['timeout']=20000
        with self.assertRaisesRegex(ValueError,'scientific protocol'):
            repair.apply_plan(self.out,plan)

    def test_repair_journal_cannot_change_surviving_design(self):
        plan=repair.make_plan(self.out,self.repo/'vtr')
        plan['new_inventory']=copy.deepcopy(plan['new_inventory'])
        plan['new_inventory']['designs'][0]['source_sha256']='changed'
        with self.assertRaisesRegex(ValueError,'Runnable benchmark'):
            repair.apply_plan(self.out,plan)

    def test_broad_output_root_rejected(self):
        with self.assertRaises(ValueError):
            repair.validate_root(self.out.parent)

    @unittest.skipUnless(os.name=='posix','Requires Linux flock')
    def test_repair_rejects_a_running_cohort_lock(self):
        with repair.cohort_lock(self.out):
            with self.assertRaisesRegex(RuntimeError,'still running'):
                with repair.cohort_lock(self.out):
                    pass


if __name__=='__main__':
    unittest.main()
