"""008 paired training accepts only feasible targets and new decoder actions."""
from pathlib import Path
import tempfile
import unittest

import torch
from aligned_model import DECODER_SCHEMA, generate
from aligned_teacher import project_teacher, TeacherAlignmentError, constraints_digest
from model import capacity_only
from test_aligned_model import graph
from train import state_sha256
from train_aligned_layers import (ACTION_SCHEMA, train_shared, prepare_arms, train_feedback, load_model)


def record():
    data=graph(); action=capacity_only(data)
    teacher,_=project_teacher(data,dict(teacher_region=action['assignment'].tolist(),teacher_xy=action['xy'].tolist()))
    return dict(condition_id='fixture/H0Z0/w64',family='fixture',split='train',graph=data,teacher=teacher)


def action_json(action,xy_hash,data):
    return dict(schema=ACTION_SCHEMA,decoder_kind=DECODER_SCHEMA,shared_xy_sha256=xy_hash,
        constraints_sha256=constraints_digest(data),
        assignment=action['assignment'].tolist(),layer=action['layer'].tolist(),xy=action['xy'].tolist())


class AlignedTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_raw_labels_and_validation_records_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs',prefix='test_aligned_') as tmp:
            r=record(); del r['teacher']['schema']
            with self.assertRaises(TeacherAlignmentError): train_shared([r],Path(tmp)/'bad',epochs=1,hidden=16)
            r=record(); r['split']='validation'
            with self.assertRaises(ValueError): train_shared([r],Path(tmp)/'validation',epochs=1)

    def test_paired_training_and_checkpoint_contract(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs',prefix='test_aligned_') as tmp:
            root=Path(tmp); records=[record()]
            shared=train_shared(records,root/'shared',epochs=1,hidden=16)
            prepared=prepare_arms(shared,records,root/'prepare',epochs=1)
            self.assertEqual(prepared['arms']['L']['initial_state_sha256'],prepared['arms']['T']['initial_state_sha256'])
            self.assertEqual(shared['shared_xy_sha256'],prepared['shared_xy_sha256'])
            for m in ('L','T'):
                self.assertNotEqual(prepared['arms'][m]['initial_state_sha256'],prepared['arms'][m]['final_state_sha256'])
                loaded=load_model(prepared['arms'][m]['checkpoint'])
                self.assertTrue(loaded.shared_xy_frozen)
                self.assertEqual(loaded.density_passes,2)
                self.assertEqual(state_sha256(loaded.shared_xy),prepared['shared_xy_sha256'])
            payload=torch.load(shared['checkpoint'],weights_only=False)
            payload['state_dict']['layer_query.weight'][0,0]+=1
            torch.save(payload,root/'changed.pt')
            with self.assertRaisesRegex(ValueError,'hash mismatch'): load_model(root/'changed.pt')

    def test_feedback_rejects_007_action_and_trains_fresh_feasible_actions(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs',prefix='test_aligned_') as tmp:
            root=Path(tmp); records=[record()]
            shared=train_shared(records,root/'shared',epochs=1,hidden=16)
            prepared=prepare_arms(shared,records,root/'prepare',epochs=1)
            old=[dict(condition_id=records[0]['condition_id'],action=dict(layer=[0]*6),trials=[])]
            with self.assertRaisesRegex(ValueError,'freshly evaluated 008'):
                train_feedback(records,prepared,root/'old',old,epochs=1)
            model=load_model(prepared['arms']['L']['checkpoint'])
            a=generate(model,records[0]['graph'],seed=11,score_noise_std=10.)
            b=generate(model,records[0]['graph'],seed=12,score_noise_std=10.)
            self.assertFalse(torch.equal(a['layer'],b['layer']))
            entries=[dict(condition_id=records[0]['condition_id'],action=action_json(v,prepared['shared_xy_sha256'],records[0]['graph']),
                trials=[dict(seed=2721,status='success',vpr_seconds=t,cpd_ns=100.)]) for v,t in ((a,10.),(b,15.))]
            result=train_feedback(records,prepared,root/'feedback',entries,epochs=2)
            self.assertEqual(result['feedback_pairs'],1)
            for m in ('L','T'):
                self.assertNotEqual(result['arms'][m]['initial_state_sha256'],result['arms'][m]['final_state_sha256'])
                self.assertEqual(len(result['arms'][m]['checkpoints']),2)
                self.assertEqual(result['arms'][m]['shared_xy_sha256'],prepared['shared_xy_sha256'])

    def test_no_pairs_does_not_claim_feedback_training(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs',prefix='test_aligned_') as tmp:
            root=Path(tmp); records=[record()]
            shared=train_shared(records,root/'shared',epochs=1,hidden=16)
            prepared=prepare_arms(shared,records,root/'prepare',epochs=1)
            report=train_feedback(records,prepared,root/'feedback',[],epochs=2)
            self.assertFalse(report['arms']['L']['actual_generator_trained'])
            self.assertEqual(report['arms']['L']['checkpoint'],prepared['arms']['L']['checkpoint'])


if __name__=='__main__': unittest.main()
