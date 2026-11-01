"""009 source, split, budget and immutable teacher contracts."""
from pathlib import Path
import tempfile
import unittest

import torch
import qor_campaign as campaign
from aligned_teacher import project_teacher
from io_utils import read,write,sha
from test_model import fixture
from test_aligned_model import graph as risk_graph
from test_aligned_teacher import target
from train_aligned_layers import train_shared,prepare_arms


class CampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_009_namespace(self):
        for name in ('development_007_layer','development_008_aligned/child'):
            with self.assertRaises(ValueError): campaign.owned(campaign.HERE/'runs'/name)
        self.assertEqual(campaign.owned(campaign.HERE/'runs/development_009_test'),campaign.HERE/'runs/development_009_test')

    def test_source_panel_and_budget(self):
        records=campaign.panel(campaign.HERE/'runs/development_008_aligned')
        self.assertEqual(sum(r['split']=='train' for r in records),16)
        self.assertEqual(sum(r['split']=='train' and r['width_role']=='reference' for r in records),8)
        self.assertEqual(sum(r['split']=='validation' for r in records),8)
        plan=campaign.queue_plan()
        self.assertEqual(plan['feedback_runs_upper']+plan['validation_runs_upper']+plan['comparison_runs'],512)
        self.assertEqual(len(plan['comparison_seeds'])*8*len(plan['methods']),240)
        self.assertEqual(plan['backend_layer_prior'],0)
        self.assertFalse(plan['Wmin_measured'])

    def test_missing_source_is_read_only(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            p=Path(d); before=list(p.iterdir())
            result=campaign.check(SimpleNamespace(source=p))
            self.assertFalse(result['ready']); self.assertEqual(list(p.iterdir()),before)

    def test_feedback_cleanup_keeps_training_action(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            folder=Path(d);payload=folder/'generation/payload';payload.mkdir(parents=True)
            for name in ('graph.pt','logic.json'): (payload/name).write_text('duplicate')
            action=payload/'target.json';action.write_text('preserved')
            campaign.QorCampaign.prune_feedback_graph(folder)
            self.assertFalse((payload/'graph.pt').exists());self.assertFalse((payload/'logic.json').exists())
            self.assertEqual(action.read_text(),'preserved')
            self.assertEqual(len(read(folder/'payload_pruned.json')['files']),2)

    def test_prepare_reuses_signed_labels_without_writing_old_data(self):
        data=risk_graph(); label,audit=project_teacher(data,target(data,[0,0,0,1,1,1]))
        with tempfile.TemporaryDirectory(dir=campaign.HERE/'runs') as d:
            root=Path(d); source=root/'source'; out=root/'out'
            graph=source/'graph.pt'; graph.parent.mkdir(); torch.save(data,graph)
            teacher=source/'teacher.json';write(teacher,label)
            write(source/'aligned_teachers/tiny/projection_audit.json',audit)
            write(source/'feedback.json',[])
            records=[dict(condition_id='tiny',split='train',family='tiny',graph=str(graph),teacher=str(teacher))]
            shared=train_shared(records,source/'models/shared',epochs=1,hidden=8)
            prepared=prepare_arms(shared,records,source/'models/arms',epochs=1)
            runner=campaign.QorCampaign.__new__(campaign.QorCampaign)
            from types import SimpleNamespace
            runner.args=SimpleNamespace(source=source);runner.out=out
            runner.conditions=[dict(id='tiny',split='train',family='tiny')]
            runner.train=runner.conditions;runner.validation=[];runner.teacher_rows={'tiny':records[0]}
            runner.source_model_report=prepared['arms']['L'];runner.source_checkpoint=Path(prepared['arms']['L']['checkpoint'])
            before={p:sha(p) for p in (teacher,graph,runner.source_checkpoint)}
            runner.prepare();runner.prepare()
            self.assertEqual(read(out/'teacher_records.json')['records'][0]['teacher'],str(teacher))
            self.assertEqual(read(out/'prepared_report.json')['train_conditions'],1)
            self.assertEqual({p:sha(p) for p in before},before)
            tampered=dict(label,teacher_layer=[1]*6);write(teacher,tampered)
            with self.assertRaises(ValueError):runner.prepare()


if __name__=='__main__':unittest.main()
