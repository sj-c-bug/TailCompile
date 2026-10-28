"""Bounded retry behavior and real stage subprocess lifetime contracts."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from io_utils import read, write
from resume_device_preparation import ResumeFailure, _terminate_owned, resume, run_calibration

HERE = Path(__file__).resolve().parent
TIMEOUT = 'Traceback (most recent call last):\nTimeoutError: complete device graph extraction exceeded its preparation budget\n'


class PreparationResumeTests(unittest.TestCase):
    def fixture(self, temporary):
        repo = Path(temporary).resolve()/'repo'
        module = repo/'experiments/vtr/tailcompile_v7_3d'
        module.mkdir(parents=True)
        entry = module/'prepare_server.sh'
        entry.write_text('# fixture\n')
        out = module/'runs/cohort'
        write(out/'campaign.json', dict(width_grid=[32,64], designs=[dict(grid=[84,84])]))
        write(out/'protocol.json', dict(frozen=True))
        return repo, out, entry

    def temporary(self):
        (HERE/'runs').mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='prep-resume-', dir=HERE/'runs')

    def test_only_known_budget_failure_with_new_cache_progress_retries(self):
        with self.temporary() as temporary:
            repo,out,entry = self.fixture(temporary)
            completions = iter([['already-repaired'], ['new-wide-cache']])
            calls=[]
            def complete(path, max_seconds):
                calls.append(('complete', path, max_seconds))
                return next(completions)
            outcomes = iter([dict(returncode=1, output_tail=TIMEOUT), dict(returncode=0, output_tail='done')])
            def stage(path, log, env):
                calls.append(('stage', path, env['TAILCOMPILE_V7_RRG_JOBS']))
                return next(outcomes)
            result = resume(out,entry,900,stage,complete,repo=repo)
            self.assertEqual([c[0] for c in calls], ['complete','stage','complete','stage'])
            self.assertEqual(result['completed_cache_keys'], ['already-repaired','new-wide-cache'])
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['attempts'][0]['returncode'],1)
            self.assertTrue(result['attempts'][0]['known_preparation_timeout'])
            self.assertEqual(read(out/'device_preparation_resume_report.json')['stage_attempt_count'],2)

    def test_non_preparation_error_does_not_retry_or_change_its_exit_code(self):
        with self.temporary() as temporary:
            repo,out,entry = self.fixture(temporary)
            calls=[]
            def complete(path, max_seconds):
                calls.append('complete')
                return []
            def stage(path, log, env):
                calls.append('stage')
                return dict(returncode=17, output_tail='RuntimeError: Default calibration has a non-routing failure')
            with self.assertRaises(ResumeFailure) as caught:
                resume(out,entry,900,stage,complete,repo=repo)
            self.assertEqual(caught.exception.returncode,17)
            self.assertEqual(calls,['complete','stage'])
            self.assertEqual(read(out/'device_preparation_resume_report.json')['status'],'stopped_non_preparation_failure')

    def test_repeated_cache_key_is_not_progress(self):
        with self.temporary() as temporary:
            repo,out,entry = self.fixture(temporary)
            calls=[]
            def complete(path,max_seconds):
                calls.append('complete')
                return ['same-key']
            def stage(path,log,env):
                calls.append('stage')
                return dict(returncode=1,output_tail=TIMEOUT)
            with self.assertRaisesRegex(ResumeFailure,'no new cache'):
                resume(out,entry,900,stage,complete,repo=repo)
            self.assertEqual(calls,['complete','stage','complete'])
            self.assertEqual(read(out/'device_preparation_resume_report.json')['status'],'stopped_no_preparation_progress')

    def test_supply_deadline_is_supported_but_an_arbitrary_timeout_is_not(self):
        for text, code, expected in [('TimeoutError: 3D supply preparation exceeded 300.0s\n',1,True),
                               ('TimeoutError: VPR wall-clock budget exceeded\n',1,False),
                               ('warning: complete device graph extraction exceeded its preparation budget',1,False),
                               (TIMEOUT,-9,False), (TIMEOUT,137,False)]:
            with self.subTest(text=text),self.temporary() as temporary:
                repo,out,entry = self.fixture(temporary)
                complete_calls=[]
                def complete(path,max_seconds):
                    complete_calls.append(1)
                    return [] if len(complete_calls)==1 else ['new-cache']
                outputs=iter([dict(returncode=code,output_tail=text),dict(returncode=0,output_tail='done')])
                stage=lambda *args:next(outputs)
                if expected:
                    self.assertEqual(resume(out,entry,900,stage,complete,repo=repo)['status'],'complete')
                    self.assertEqual(len(complete_calls),2)
                else:
                    with self.assertRaises(ResumeFailure):
                        resume(out,entry,900,stage,complete,repo=repo)
                    self.assertEqual(len(complete_calls),1)

    def test_bad_budget_or_output_scope_fails_before_controller_writes(self):
        with self.temporary() as temporary:
            repo,out,entry = self.fixture(temporary)
            for budget in (0,-1,float('inf'),float('nan')):
                with self.subTest(budget=budget),self.assertRaises(ValueError):
                    resume(out,entry,budget,repo=repo)
            self.assertFalse((out/'device_preparation_resume').exists())
            external=Path(temporary)/'outside'
            with self.assertRaisesRegex(ValueError,'canonical v7'):
                resume(external,entry,repo=repo)
            self.assertFalse(external.exists())

    def test_retry_count_is_bounded_by_the_frozen_panel(self):
        with self.temporary() as temporary:
            repo,out,entry = self.fixture(temporary)
            count=[0]
            def complete(path,max_seconds):
                count[0]+=1
                return [f'key-{count[0]}']
            def stage(path,log,env):
                return dict(returncode=1,output_tail=TIMEOUT)
            with self.assertRaisesRegex(ResumeFailure,'retry limit'):
                resume(out,entry,900,stage,complete,repo=repo)
            latest=read(out/'device_preparation_resume_report.json')
            self.assertEqual(latest['status'],'stopped_retry_limit')
            self.assertEqual(latest['stage_attempt_count'],9)

    @unittest.skipUnless(os.name=='posix' and shutil.which('bash'),'Linux stage process contract')
    def test_real_stage_output_is_teed_and_its_return_code_preserved(self):
        with self.temporary() as temporary:
            root=Path(temporary).resolve()
            entry=root/'prepare_server.sh'
            entry.write_text('printf "stdout:%s\\n" "$*"\nprintf "stderr-message\\n" >&2\nexit 17\n')
            log=root/'stage.log'
            result=run_calibration(entry,log,dict(os.environ))
            self.assertEqual(result['returncode'],17)
            self.assertIn('stdout:--stage calibrate',log.read_text())
            self.assertIn('stderr-message',result['output_tail'])

    @unittest.skipUnless(os.name=='posix','POSIX owned process groups')
    def test_cancellation_kills_only_the_owned_process_group(self):
        owned=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],start_new_session=True)
        unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(10)'],start_new_session=True)
        try:
            _terminate_owned(owned)
            self.assertIsNotNone(owned.poll())
            self.assertIsNone(unrelated.poll())
        finally:
            if owned.poll() is None:
                owned.kill();owned.wait(timeout=5)
            if unrelated.poll() is None:
                unrelated.terminate();unrelated.wait(timeout=5)


if __name__ == '__main__':
    unittest.main()
