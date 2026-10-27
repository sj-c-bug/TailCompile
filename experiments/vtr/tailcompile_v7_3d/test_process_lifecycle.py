"""Small real subprocess checks for TERM/INT, own-group cleanup and resume."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

HERE=Path(__file__).resolve().parent


def running(pid):
    try:
        # A terminated descendant can briefly await init's zombie reaping.
        return Path(f'/proc/{pid}/stat').read_text().split()[2] not in ('Z','X')
    except FileNotFoundError:
        return False


@unittest.skipUnless(os.name=='posix' and Path('/proc').is_dir(),'POSIX process-group contract')
class ProcessLifecycle(unittest.TestCase):
    def test_term_and_interrupt_retire_only_owned_groups_and_resume(self):
        unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            for signum in (signal.SIGTERM,signal.SIGINT):
                with self.subTest(signal=int(signum)),tempfile.TemporaryDirectory(prefix='v7-process-contract-') as tmp:
                    root=Path(tmp)
                    commands=[]
                    for index in range(2):
                        child=('import os,signal,time; from pathlib import Path; '
                               'signal.signal(signal.SIGTERM,signal.SIG_IGN); '
                               f'Path({str(root/f"child{index}.pid")!r}).write_text(str(os.getpid())); '
                               'time.sleep(1)')
                        commands.append([sys.executable,'-c',child])
                    controller=(
                        'import sys; from pathlib import Path; from concurrent.futures import ThreadPoolExecutor\n'
                        f'sys.path.insert(0,{str(HERE)!r})\n'
                        'from io_utils import run_trial,install_signal_handlers,raise_if_cancelled,write\n'
                        f'root=Path({str(root)!r}); commands={commands!r}\n'
                        'install_signal_handlers()\n'
                        'try:\n'
                        ' with ThreadPoolExecutor(max_workers=2) as pool:\n'
                        '  tasks=[pool.submit(run_trial,root/f"trial{i}",cmd,timeout=10) for i,cmd in enumerate(commands)]\n'
                        '  for task in tasks: task.result()\n'
                        'except KeyboardInterrupt:\n'
                        ' try:\n'
                        '  run_trial(root/"must_not_dispatch",commands[0],timeout=10)\n'
                        ' except KeyboardInterrupt:\n'
                        '  write(root/"controller_stopped.json",dict(cancelled=True))\n')
                    process=subprocess.Popen([sys.executable,'-c',controller],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
                    try:
                        deadline=time.monotonic()+5
                        while not all((root/f'child{i}.pid').is_file() for i in range(2)):
                            if process.poll() is not None:
                                out,err=process.communicate()
                                self.fail(f'controller exited early: {out}\n{err}')
                            if time.monotonic()>deadline:
                                self.fail('owned workers did not start in bounded time')
                            time.sleep(.01)
                        pids=[int((root/f'child{i}.pid').read_text()) for i in range(2)]
                        os.kill(process.pid,signum)
                        out,err=process.communicate(timeout=5)
                        self.assertEqual(process.returncode,0,f'{out}\n{err}')
                        self.assertTrue((root/'controller_stopped.json').is_file())
                        self.assertFalse((root/'must_not_dispatch').exists())
                        deadline=time.monotonic()+3
                        while any(running(pid) for pid in pids) and time.monotonic()<deadline:
                            time.sleep(.01)
                        self.assertTrue(all(not running(pid) for pid in pids),'owned group survived cancellation')
                        self.assertIsNone(unrelated.poll(),'another session was incorrectly terminated')
                        for index in range(2):
                            record=json.loads((root/f'trial{index}/manifest.json').read_text())
                            self.assertEqual(record['status'],'interrupted')
                            self.assertEqual(record['metrics']['interruption_signal'],int(signum))
                        resume=(f'import sys; from pathlib import Path; sys.path.insert(0,{str(HERE)!r}); '
                                'from io_utils import run_trial; '
                                f'r=run_trial(Path({str(root/"trial0")!r}),{commands[0]!r},timeout=10); '
                                'assert r["status"]=="complete"')
                        result=subprocess.run([sys.executable,'-c',resume],capture_output=True,text=True,timeout=5)
                        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
                        attempts=sorted((root/'trial0/attempts').iterdir())
                        self.assertEqual(len(attempts),2)
                        old=json.loads((attempts[0]/'metrics.json').read_text())
                        self.assertEqual(old['status'],'interrupted')
                    finally:
                        if process.poll() is None:
                            process.kill()
                            process.wait()
        finally:
            if unrelated.poll() is None:
                unrelated.terminate()
                unrelated.wait(timeout=5)


if __name__=='__main__':
    unittest.main()
