"""Real one-epoch CPU training through projection, loss, optimizer and saving."""
import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch

from bridge import REPO, read, sha, write
from fast_model import CHECKPOINT_SCHEMA, HierarchicalInitializer
from generate import generate
from test_fast_model import graph
from train_runtime import parser, train_runtime


class RuntimeTrainingSmokeTest(unittest.TestCase):
    def test_real_fast_target_trains_updates_saves_resumes_and_generates(self):
        # Keep all transient data in the explicitly requested checkout test root.
        parent=REPO/'test_tmp_v69'
        parent.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='training-smoke-',dir=parent) as temp:
            root=Path(temp);source=root/'v68';out=root/'v69'
            source.mkdir();out.mkdir()
            torch.set_num_threads(1)
            torch.manual_seed(42)
            original=HierarchicalInitializer(8,13,hidden=16,chunk_size=3)
            checkpoint=source/'generator.pt'
            torch.save(dict(schema=CHECKPOINT_SCHEMA,model_config=original.config,
                model_state=original.state_dict(),training_families=['family_train'],
                validation_families=['family_validation']),checkpoint)
            examples=[];contexts=[]
            for design,family,split in [('toy_train','family_train','train'),
                                         ('toy_validation','family_validation','validation')]:
                folder=source/'graphs'/design;folder.mkdir(parents=True)
                data=graph(2,2)
                data['movable'][0]=False
                data['fixed_xy']=torch.zeros((12,2))
                data['fixed_xy'][0]=torch.tensor([.125,.25])
                data['fixed_region']=torch.full((12,),-1)
                data['fixed_region'][0]=0
                if split=='validation':
                    data['cluster_x']=data['cluster_x']+.05
                graph_path=folder/'graph.pt';torch.save(data,graph_path)
                write(folder/'logic.json',dict(clusters=[dict(atoms=[f'{design}_atom{i}']) for i in range(12)]))
                write(folder/'device.json',dict(grid_width=24,grid_height=20,
                    regions=[dict(id=i) for i in range(4)]))
                write(folder/'supply.json',dict(schema='test-supply'))
                assignments=torch.arange(12)%4
                bounds=data['region_bounds'][assignments]
                xy=(bounds[:,[0,2]]+bounds[:,[1,3]])/2
                xy[0]=data['fixed_xy'][0]
                target=dict(teacher_region=assignments.tolist(),teacher_xy=xy.tolist(),
                            movable=data['movable'].tolist())
                examples.append(dict(design=design,family=family,split=split,width=100,
                    graph=str(graph_path),graph_sha256=sha(graph_path),target=target))
                contexts.append(dict(design=design,family=family,split=split,width=100,
                    graph=str(graph_path),logic=str(folder/'logic.json'),
                    device=str(folder/'device.json'),supply=str(folder/'supply.json')))
            dataset=source/'dataset.json'
            write(dataset,dict(tile_size=4,examples=examples))
            options=dict(soft_density_passes=2,density_price_weight=1.,density_step=1.)
            stdout=io.StringIO()
            with contextlib.redirect_stdout(stdout):
                candidate=generate(contexts[0],checkpoint,out/'feedback/generation/toy_train/w100/c0',options)
            write(out/'protocol.json',dict(source=str(source),source_dataset_sha256=sha(dataset),
                initial_checkpoint=str(checkpoint),initial_checkpoint_sha256=sha(checkpoint),
                budgets=dict(width_regression=.03,wirelength_regression=.03,cpd_regression=.03,
                             min_total_time_reduction=.05),options=options))
            write(out/'feedback/feedback_dataset.json',dict(complete=True,all_route_outcomes_known=True,
                generator_checkpoint_sha256=sha(checkpoint),selected=[
                    dict(design='toy_train',family='family_train',split='train',width=100,
                         target=candidate['target'],target_sha256=candidate['target_sha256'])]))
            args=parser().parse_args(['--source',str(source),'--out',str(out),'--device','cpu',
                                      '--epochs','1','--threads','1','--chunk-size','3','--graph-cache','1'])
            with contextlib.redirect_stdout(stdout):
                train_runtime(args)  # Actual legacy trainer; no optimizer/model mocking.
            model_dir=out/'models/runtime'
            report=read(model_dir/'training_report.json')
            runtime_report=read(model_dir/'runtime_training_report.json')
            audit=read(model_dir/'supervision_audit.json')
            self.assertTrue(report['complete'])
            self.assertTrue(report['weights_changed'])
            self.assertEqual(report['epochs_completed'],1)
            self.assertEqual(report['real_feedback_targets'],1)
            self.assertEqual(report['final_test_examples'],0)
            self.assertEqual(report['training_families'],['family_train'])
            self.assertEqual(report['validation_families'],['family_validation'])
            self.assertTrue(audit['complete'])
            self.assertEqual(audit['audited_conditions'],2)
            self.assertFalse(runtime_report['heldout_runtime_validated'])
            digest=sha(model_dir/'generator.pt')
            self.assertEqual(digest,report['checkpoint_sha256'])
            latest=torch.load(model_dir/'latest.pt',map_location='cpu',weights_only=True)
            self.assertEqual(latest['epoch'],1)
            self.assertIn('optimizer',latest)
            self.assertIn('torch_rng',latest)
            with contextlib.redirect_stdout(stdout):
                train_runtime(args)  # Completed exact-protocol resume must not retrain.
                regenerated=generate(contexts[0],model_dir/'generator.pt',root/'after_training',options)
            self.assertEqual(sha(model_dir/'generator.pt'),digest)
            self.assertIn('reuse_complete_training',stdout.getvalue())
            self.assertTrue(Path(regenerated['ap_init']).is_file())
            self.assertEqual(len(read(regenerated['target'])['teacher_region']),12)


if __name__=='__main__':
    unittest.main()
