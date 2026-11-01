"""009 matched training, immutable source migration and checkpoint checks."""
from pathlib import Path
import tempfile
import unittest

import torch

from aligned_model import generate as generate008
from test_train_aligned_layers import record,action_json
from train_aligned_layers import train_shared,prepare_arms,load_model as load008
from train_qor import train_bootstrap,train_feedback,load_model
from train import state_sha256


class QorTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def setup_case(self,path):
        records=[record()]
        shared=train_shared(records,path/'shared',epochs=1,hidden=16)
        prepared=prepare_arms(shared,records,path/'prepared',epochs=1)
        checkpoint=prepared['arms']['L']['checkpoint']; m=load008(checkpoint)
        data=records[0]['graph']; identity=state_sha256(m.shared_xy)
        actions=[generate008(m,data,seed=s,score_noise_std=10.) for s in (11,12)]
        self.assertFalse(torch.equal(actions[0]['layer'],actions[1]['layer']))
        entries=[dict(candidate_id=str(i),condition_id=records[0]['condition_id'],split='train',
            action=action_json(a,identity,data),trials=[dict(seed=9101,status='success',vpr_seconds=t,wirelength=w,cpd_ns=c)])
            for i,(a,t,w,c) in enumerate(zip(actions,(10.,15.),(110.,90.),(110.,90.)))]
        return records,checkpoint,entries

    def test_all_arms_train_from_common_source_and_round_trip(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs',prefix='test_qor_') as tmp:
            path=Path(tmp);records,checkpoint,entries=self.setup_case(path)
            result=train_bootstrap(records,checkpoint,path/'bootstrap',entries,epochs=1)
            self.assertEqual(len({r['initial_state_sha256'] for r in result['arms'].values()}),1)
            # The two measured candidates have opposite runtime/QoR order. If
            # feedback were only reported, their identical base losses would
            # leave R and Q with the same final actor parameters.
            self.assertNotEqual(result['arms']['R']['final_state_sha256'],result['arms']['Q']['final_state_sha256'])
            for arm,r in result['arms'].items():
                self.assertEqual(r['epochs'],1);self.assertTrue(r['actual_generator_trained'])
                self.assertNotEqual(r['initial_state_sha256'],r['final_state_sha256'])
                self.assertEqual(r['feedback_pairs'],1)
                m=load_model(r['checkpoint']); self.assertEqual(m.arm,arm)
                self.assertEqual(m.shared_xy_frozen,arm in ('R','Q','C'))
                if arm in ('R','Q','C'):self.assertEqual(r['shared_xy_sha256'],result['frozen_xy_sha256'])
                else:self.assertNotEqual(r['shared_xy_sha256'],result['frozen_xy_sha256'])
            updated=train_feedback(records,result,path/'feedback',entries,epochs=2)
            for arm,r in updated['arms'].items():
                self.assertEqual(len(r['checkpoints']),2)
                self.assertEqual(r['scales'],result['scales'])
            with self.assertRaisesRegex(ValueError,'migrate 008'):load_model(checkpoint)
            saved=torch.load(result['arms']['T']['checkpoint'],weights_only=False)
            saved['state_dict']['layer_query.weight'][0,0]+=1
            torch.save(saved,path/'tampered.pt')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):load_model(path/'tampered.pt')

    def test_split_and_collection_provenance_cannot_be_bypassed(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs',prefix='test_qor_') as tmp:
            path=Path(tmp);records,checkpoint,entries=self.setup_case(path)
            records[0]['split']='validation'
            with self.assertRaisesRegex(ValueError,'train-only'):train_bootstrap(records,checkpoint,path/'bad',entries,epochs=1)
            records[0]['split']='train'
            entries[0]['generation']=dict(shared_xy_sha256='0'*64)
            with self.assertRaisesRegex(ValueError,'metadata/action'):train_bootstrap(records,checkpoint,path/'wrong',entries,epochs=1)

    def test_bootstrap_without_feedback_is_teacher_training_but_feedback_skip_is_explicit(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs',prefix='test_qor_') as tmp:
            path=Path(tmp);records,checkpoint,_=self.setup_case(path)
            result=train_bootstrap(records,checkpoint,path/'bootstrap',[],epochs=1)
            self.assertTrue(all(r['actual_generator_trained'] for r in result['arms'].values()))
            skipped=train_feedback(records,result,path/'no_pairs',[],epochs=2)
            self.assertTrue(all(not r['actual_generator_trained'] and r['epochs']==0 for r in skipped['arms'].values()))
            self.assertTrue(all(r['initial_state_sha256']==r['final_state_sha256'] for r in skipped['arms'].values()))


if __name__=='__main__':unittest.main()
