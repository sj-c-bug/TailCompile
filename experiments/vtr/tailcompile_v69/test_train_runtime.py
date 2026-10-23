import json
from pathlib import Path
import tempfile
import unittest

from train_runtime import parser, training_inputs
from bridge import sha


class RuntimeTrainingTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        root = Path(self.temporary.name)
        self.source, self.out = root/'v68', root/'v69'
        self.source.mkdir(); (self.out/'feedback').mkdir(parents=True)
        self.checkpoint = self.source/'generator.pt'
        self.checkpoint.write_bytes(b'only provenance is inspected in this unit test')
        self.target = self.out/'target.json'
        self.save(self.target, {'teacher_region': [0], 'teacher_xy': [[.2,.3]], 'movable':[True]})
        self.dataset = dict(tile_size=4, examples=[
            dict(design='train_a', width=100, family='family_a', split='train'),
            dict(design='val_b', width=110, family='family_b', split='validation')])
        self.save(self.source/'dataset.json', self.dataset)
        self.protocol = dict(source=str(self.source), source_dataset_sha256=sha(self.source/'dataset.json'),
            initial_checkpoint=str(self.checkpoint), initial_checkpoint_sha256=sha(self.checkpoint),
            budgets=dict(width_regression=.03, wirelength_regression=.03, cpd_regression=.03,
                         min_total_time_reduction=.05),
            options=dict(soft_density_passes=2,density_price_weight=1.,density_step=1.))
        self.save(self.out/'protocol.json',self.protocol)
        self.feedback = dict(complete=True, all_route_outcomes_known=True,
            generator_checkpoint_sha256=sha(self.checkpoint), selected=[
                dict(design='train_a',width=100,family='family_a',split='train',
                     target=str(self.target),target_sha256=sha(self.target))])
        self.save(self.out/'feedback/feedback_dataset.json', self.feedback)
        self.args = parser().parse_args(['--source',str(self.source),'--out',str(self.out)])

    @staticmethod
    def save(path, data):
        path.write_text(json.dumps(data), encoding='utf-8')

    def test_legacy_arguments_and_runtime_protocol(self):
        args, signature = training_inputs(self.args)
        self.assertEqual(args.epochs,50)
        self.assertEqual(args.device,'cuda')
        self.assertEqual(args.out,self.out/'models/runtime')
        self.assertEqual(args.initialize,self.checkpoint)
        self.assertEqual(signature['selected_conditions'],1)
        self.assertEqual(signature['final_test_examples'],0)
        self.assertFalse(args.allow_no_validation)
        self.assertEqual(signature['budgets']['cpd_regression'],.03)

    def test_empty_feedback_does_not_launch_dummy_training(self):
        self.feedback['selected']=[]
        self.save(self.out/'feedback/feedback_dataset.json',self.feedback)
        with self.assertRaisesRegex(ValueError,'No runtime-beneficial'):
            training_inputs(self.args)

    def test_unknown_outcomes_and_wrong_checkpoint_rejected(self):
        self.feedback['all_route_outcomes_known']=False
        self.save(self.out/'feedback/feedback_dataset.json',self.feedback)
        with self.assertRaisesRegex(ValueError,'unresolved'):
            training_inputs(self.args)
        self.feedback['all_route_outcomes_known']=True
        self.feedback['generator_checkpoint_sha256']='wrong'
        self.save(self.out/'feedback/feedback_dataset.json',self.feedback)
        with self.assertRaisesRegex(ValueError,'another generator'):
            training_inputs(self.args)

    def test_validation_feedback_rejected(self):
        self.feedback['selected'][0].update(design='val_b',width=110,family='family_b',split='validation')
        self.save(self.out/'feedback/feedback_dataset.json',self.feedback)
        with self.assertRaisesRegex(ValueError,'train-family'):
            training_inputs(self.args)

    def test_test_leakage_is_rejected(self):
        self.dataset['examples'][1]['split']='final_test'
        self.save(self.source/'dataset.json',self.dataset)
        self.protocol['source_dataset_sha256']=sha(self.source/'dataset.json')
        self.save(self.out/'protocol.json',self.protocol)
        with self.assertRaisesRegex(ValueError,'Final-test'):
            training_inputs(self.args)


if __name__=='__main__':
    unittest.main()
