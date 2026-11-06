"""014 public entry contracts: log discovery, locked choices and native evidence."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prospective_campaign as campaign
from io_utils import read, sha, write
from test_prospective_evaluation import fixture


class ProspectiveEntryTests(unittest.TestCase):
    def test_seed_ledger_detects_nested_historical_assignments(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            relative = 'experiments/vtr/test/runs/previous/protocol.json'
            write(repo/relative,dict(seeds={'validation':[14102]}))
            with patch('prospective_campaign.subprocess.check_output',return_value=(relative+'\0').encode()):
                with self.assertRaisesRegex(ValueError,'reserved seeds'):
                    campaign.seed_ledger(repo)
                write(repo/relative,dict(timeout=14102,seeds={'validation':[400]}))
                self.assertIn(relative,campaign.seed_ledger(repo)['tracked_protocols'])

    def test_report_uses_frozen_choices_and_native_measurements(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            args = SimpleNamespace(out=out)
            rows, conditions, predictions = fixture()
            write(out/'qualified_panel.json',dict(conditions=conditions))
            write(out/'predictions/prediction_report.json',predictions)
            with patch('prospective_inputs.qualified_conditions',return_value=conditions), \
                    patch.object(campaign,'predict_panel',return_value=predictions):
                campaign.execute('predict',args,object(),{})
            for r in rows:
                folder = out/'comparison'/r['condition_id']/f's{r["seed"]}'/r['method']
                path = folder/'vpr/manifest.json'
                init = r['ap_init_sha256']
                signature = dict(command=['vpr','--seed',str(r['seed']),'--route_chan_width',str(r['width'])],
                                 env={'TC_GP_INIT':'init'} if init else {},inputs={'init':init} if init else {})
                write(path,dict(signature=signature,metrics={k:r[k] for k in ('wirelength','cpd_ns')}))
                r.update(vpr_manifest=str(path),vpr_manifest_sha256=sha(path))
                write(folder/'metrics.json',r)
            raw = dict(complete=True, predictions_sha256=sha(out/'predictions/prediction_report.json'),rows=rows)
            write(out/'comparison/trials_report.json',raw)
            campaign.execute('report',args)
            self.assertEqual(read(out/'prospective_report.json')['logical_native_trials'],96)
            rows[0]['wirelength'] += 1.
            write(out/'comparison/trials_report.json',raw)
            with self.assertRaisesRegex(ValueError,'per-trial evidence'):
                campaign.execute('report',args)
            write(out/'predictions/prediction_report.json',{})
            with self.assertRaisesRegex(ValueError,'prediction lock'):
                campaign.execute('report',args)

    def test_cannot_publish_a_new_selection_after_comparison_started(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _, conditions, _ = fixture()
            write(out/'qualified_panel.json',dict(conditions=conditions))
            (out/'comparison').mkdir()
            with patch('prospective_inputs.qualified_conditions',return_value=conditions), \
                    self.assertRaisesRegex(ValueError,'after primary'):
                campaign.execute('predict',SimpleNamespace(out=out),object(),{})

    @unittest.skipUnless(os.name == 'posix', 'Bash background launcher')
    def test_background_log_and_pid_have_v8_014_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)/'repo'
            mod = repo/'experiments/vtr/tailcompile_v7_3d'
            mod.mkdir(parents=True)
            (repo/'REPOSITORY_WORKFLOW.md').touch()
            script = mod/'run_prospective_campaign.sh'
            shutil.copyfile(campaign.HERE/script.name,script)
            (mod/'prospective_campaign.py').write_text('print("V8_014_GROUP_COMPLETE group=all")\n')
            env = {k:v for k,v in os.environ.items() if not k.startswith('TAILCOMPILE_')}
            env['TAILCOMPILE_3D_PYTHON'] = sys.executable
            proc = subprocess.run(['bash',script,'all','--background'],env=env,text=True,capture_output=True,check=True)
            values = dict(line.split('=',1) for line in proc.stdout.splitlines())
            log = Path(values['LOG'])
            self.assertEqual(log.parent,repo/'logs/v8_014')
            self.assertTrue(log.name.startswith('v8_014_all_'))
            self.assertEqual(Path(values['PIDFILE']).read_text().strip(),values['PID'])
            for _ in range(100):
                if 'V8_014_GROUP_COMPLETE' in log.read_text(): break
                time.sleep(.05)
            self.assertIn('V8_014_GROUP_COMPLETE',log.read_text())


if __name__ == '__main__':
    unittest.main()
