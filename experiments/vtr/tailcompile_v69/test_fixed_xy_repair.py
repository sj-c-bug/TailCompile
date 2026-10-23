"""The sentinel fix resumes only its exact pre-VPR cohort and cached outputs."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import generate
import repair_fixed_xy as repair_module
from bridge import HERE, read, sha, write


class FixedXYRepairTest(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.repo=Path(self.temporary.name)
        self.out=self.repo/'experiments/vtr/tailcompile_v69/runs/runtime_001'
        self.protocol=dict(schema='tailcompile-v69-runtime-1',seeds=[371,372,373,374,375],
            options=dict(soft_density_passes=2),code=dict(generate.PRE_FIXED_XY_FIX,
                **{'run.py':sha(HERE/'run.py')}))
        write(self.out/'protocol.json',self.protocol)
        self.repo_patch=patch.object(repair_module,'REPO',self.repo)
        self.repo_patch.start();self.addCleanup(self.repo_patch.stop)

    def test_upgrade_backs_up_only_code_hashes_and_is_idempotent(self):
        report=repair_module.repair(self.out)
        after=read(self.out/'protocol.json')
        expected=copy.deepcopy(self.protocol)
        for name in generate.PRE_FIXED_XY_FIX:
            expected['code'][name]=sha(HERE/name)
        self.assertEqual(after,expected)
        self.assertEqual(read(self.out/'repairs/fixed_xy/protocol.json'),self.protocol)
        self.assertEqual(read(self.out/'fixed_xy_repair.json'),report)
        self.assertEqual(report['pre_fix_vpr_trials'],0)
        original_audit=sha(self.out/'fixed_xy_repair.json')
        self.assertFalse(repair_module.repair(self.out)['changed'])
        self.assertEqual(sha(self.out/'fixed_xy_repair.json'),original_audit)

    def test_unrelated_source_change_is_rejected_without_mutation(self):
        self.protocol['code']['run.py']='unreviewed-other-code'
        write(self.out/'protocol.json',self.protocol)
        before=sha(self.out/'protocol.json')
        with self.assertRaisesRegex(ValueError,'Unrelated implementation'):
            repair_module.repair(self.out)
        self.assertEqual(sha(self.out/'protocol.json'),before)
        self.assertFalse((self.out/'fixed_xy_repair.json').exists())

    def test_existing_vpr_or_training_evidence_blocks_mixed_runtime(self):
        for name in ('started.json','trial.json','wave.json','feedback_dataset.json','evaluation_plan.json'):
            with self.subTest(name=name):
                blocker=self.out/'feedback/runs/case'/name
                write(blocker,dict(status='success'))
                before=sha(self.out/'protocol.json')
                with self.assertRaisesRegex(ValueError,'evidence already exists'):
                    repair_module.repair(self.out)
                self.assertEqual(sha(self.out/'protocol.json'),before)
                blocker.unlink()

    def signatures(self):
        old=dict(schema='tailcompile-v69-fast-generation-1',checkpoint_sha256='frozen-model',
            options=dict(soft_density_passes=2),candidate=0,inputs=dict(graph='frozen-graph'),
            implementation=dict(generate.PRE_FIXED_XY_FIX,**{'model.py':'unchanged-model-source'}))
        new=copy.deepcopy(old)
        for name in generate.PRE_FIXED_XY_FIX:
            new['implementation'][name]=sha(HERE/name)
        return old,new

    def test_only_the_exact_validation_fix_is_signature_compatible(self):
        old,new=self.signatures()
        self.assertTrue(generate.validation_fix_compatible(old,new))
        for field,value in [('checkpoint_sha256','another-model'),('options',dict(soft_density_passes=0)),
                            ('candidate',1),('inputs',dict(graph='changed-graph'))]:
            changed=copy.deepcopy(new);changed[field]=value
            self.assertFalse(generate.validation_fix_compatible(old,changed))
        for name in ('model.py','fast_model.py','generate.py'):
            changed=copy.deepcopy(old);changed['implementation'][name]='unknown-implementation'
            self.assertFalse(generate.validation_fix_compatible(changed,new))

    def test_cached_original_provenance_and_artifact_checks_are_preserved(self):
        old,new=self.signatures()
        folder=self.out/'feedback/generation/toy/w100/c0'
        saved=dict(signature=old)
        for key in ('ap_init','fplace','target'):
            artifact=folder/(key+'.txt')
            write(artifact,dict(opaque='unchanged-artifact'))
            saved[key]=str(artifact);saved[key+'_sha256']=sha(artifact)
        write(folder/'generation.json',saved)
        before=sha(folder/'generation.json')
        self.assertEqual(generate.validate_generation(folder,new),saved)
        self.assertEqual(sha(folder/'generation.json'),before)
        write(saved['ap_init'],dict(opaque='tampered'))
        with self.assertRaisesRegex(ValueError,'Changed generation artifact'):
            generate.validate_generation(folder,new)


if __name__=='__main__':
    unittest.main()
