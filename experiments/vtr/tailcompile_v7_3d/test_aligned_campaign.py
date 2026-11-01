"""008 stage contracts: old data stays immutable and bad labels cannot pass."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

import aligned_campaign as campaign
from aligned_teacher import SCHEMA, verify_teacher
from io_utils import read, write
from test_model import fixture
from test_aligned_teacher import target


class AlignedCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_old_output_namespace_is_rejected(self):
        with self.assertRaises(ValueError):
            campaign.owned(campaign.HERE/'runs/development_007_layer')
        with self.assertRaises(ValueError):
            campaign.owned(campaign.HERE/'runs/development_007_layer/child')
        self.assertEqual(campaign.owned(campaign.HERE/'runs/development_008_test'),
                         campaign.HERE/'runs/development_008_test')

    def test_registered_source_split_and_budget(self):
        rows = campaign.panel(campaign.HERE/'runs/development_007_layer')
        self.assertEqual(sum(r['split'] == 'train' for r in rows), 16)
        self.assertEqual(sum(r['split'] == 'validation' for r in rows), 8)
        plan = campaign.queue_plan()
        self.assertEqual(plan['total_runs_upper'], 464)
        self.assertEqual(plan['local_qualification_upper'], 0)
        self.assertFalse(plan['inference_risk_diagnostics'])

    def test_incomplete_source_rejected(self):
        with tempfile.TemporaryDirectory(dir=campaign.HERE) as temporary:
            root = Path(temporary)
            write(root/'layer_report.json', dict(complete=False))
            with self.assertRaisesRegex(ValueError, 'completed'):
                campaign.panel(root)

    def test_prepare_corrects_capacity_keeps_raw_and_resumes(self):
        data = fixture()
        data['demand'].zero_(); data['demand'][:4, 2] = 1; data['demand'][4:, 0] = 1
        raw = target(data, [0]*6)
        with tempfile.TemporaryDirectory(dir=campaign.HERE) as temporary:
            root = Path(temporary)
            graph, label = root/'raw_graph.pt', root/'raw_teacher.json'
            torch.save(data, graph); write(label, raw)
            before = {p: p.read_bytes() for p in (graph, label)}
            runner = campaign.AlignedCampaign.__new__(campaign.AlignedCampaign)
            runner.out = root/'output'
            runner.conditions = [dict(id='engineering/memory', family='engineering', split='train')]
            runner.teacher_rows = {'engineering/memory': dict(graph=str(graph), teacher=str(label))}
            runner.prepare()
            records = read(runner.out/'teacher_records.json')['records']
            result = read(records[0]['teacher'])
            audit = read(records[0]['projection_audit'])
            self.assertEqual(result['schema'], SCHEMA)
            self.assertEqual(audit['raw_capacity']['layer_overflow_by_resource'][2], 2.)
            self.assertEqual(verify_teacher(data, result)['layer_overflow_by_resource'], [0.]*5)
            files = {p: p.read_bytes() for p in (Path(records[0]['teacher']), Path(records[0]['projection_audit']))}
            runner.prepare()
            self.assertTrue(all(p.read_bytes() == b for p, b in files.items()))
            self.assertTrue(all(p.read_bytes() == b for p, b in before.items()))


if __name__ == '__main__':
    unittest.main()
