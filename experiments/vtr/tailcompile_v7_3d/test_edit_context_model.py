"""012 matched-readout controls and raw deployment of original 011 actions."""
from pathlib import Path
import tempfile
import unittest

import torch

from aligned_model import action_digest
from edit_context_model import EditContextModel, feature_blocks
from edit_context_features import edit_feature_dim
from test_residual_model import source_case
from train_residual import from_source
from train import state_sha256


class EditContextModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def model(self,mode='G'):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(12301)
            return EditContextModel(4,3,mode=mode)

    def test_same_parameters_and_initial_weights_mask_only_input(self):
        a,b=self.model('G'),self.model('E')
        self.assertEqual(state_sha256(a),state_sha256(b))
        self.assertEqual(sum(p.numel() for p in a.parameters()),sum(p.numel() for p in b.parameters()))
        glob=torch.tensor([[0.,1.,0.,0.],[0.,1.,0.,0.]])
        local=torch.tensor([[0.,0.,0.],[1.,2.,3.]])
        for model in (a,b):
            with torch.no_grad():
                head=model.heads['J'];head[0].weight.zero_();head[0].bias.zero_()
                head[0].weight[0,-1]=1.;head[2].weight.zero_();head[2].weight[0,0]=1.
        self.assertTrue(torch.equal(a.score_features(glob,local,'J'),torch.zeros(2)))
        self.assertGreater(float(b.score_features(glob,local,'J')[1]),0)
        with self.assertRaisesRegex(ValueError,'nonfinite'):
            b.score_features(glob,local*float('nan'),'J')

    def test_raw_inference_returns_original_bank_and_never_updates_backbone(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs',prefix='test_edit_model_') as tmp:
            data,path=source_case(Path(tmp));source=from_source(path)
            before=state_sha256(source.backbone)
            base,bank=source.candidate_bank(data,'Y')
            glob,local=feature_blocks(source.backbone,data,base,bank)
            model=EditContextModel(glob.shape[1],local.shape[1],mode='E',
                source_checkpoint_sha256=source.source_checkpoint_sha256,backbone_sha256=before)
            self.assertEqual(local.shape[1],edit_feature_dim(16))
            self.assertTrue(torch.equal(glob,source.candidate_features(data,bank,base)))
            self.assertEqual(action_digest(model.select_action(data,'Y',path)),action_digest(base))
            # Enable a known coordinate-strength decision using only old descriptor.
            with torch.no_grad():
                head=model.heads['Y'];head[0].weight.zero_();head[0].bias.zero_()
                head[0].weight[0,4*16+1]=1.;head[2].weight.zero_();head[2].weight[0,0]=1.
            chosen=model.select_action(data,'Y',path)
            self.assertIn(action_digest(chosen),[action_digest(a) for a in bank])
            self.assertNotEqual(chosen['candidate_id'],'identity')
            self.assertFalse(chosen['selection']['teacher_or_qor_used_at_inference'])
            self.assertEqual(action_digest(model.select_action(data,'Y',path,enabled=False)),action_digest(base))
            self.assertEqual(before,state_sha256(source.backbone))
            model.config['source_checkpoint_sha256']='0'*64
            with self.assertRaisesRegex(ValueError,'identity changed'):
                model.select_action(data,'Y',path)


if __name__=='__main__':unittest.main()
