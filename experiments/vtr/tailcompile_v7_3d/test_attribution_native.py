"""015 actual child process, resume, consumer and storage boundary tests."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from campaign import resource_budget
from io_utils import read, write, sha
from attribution_campaign import arguments, Controller, verify_measurements
from attribution_native import run_native

HERE=Path(__file__).resolve().parent


class AttributionNativeTests(unittest.TestCase):
    def test_old_cohort_is_never_a_write_target(self):
        with self.assertRaises(SystemExit):
            arguments(['all','--out',str(HERE/'runs/development_014_prospective_joint')])

    def test_parallel_really_dispatches_independent_tasks(self):
        barrier=Barrier(3)
        controller=SimpleNamespace(budget={'effective':3},guard=lambda:None)
        def work(i):
            barrier.wait(timeout=5)
            return i*i
        self.assertEqual(sorted(Controller.parallel(controller,range(3),work)),[0,1,4])

    @unittest.skipUnless(os.name=='posix','Linux child process contract')
    def test_native_consumes_init_continues_and_reclaims_disposable_files(self):
        with tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='development_015_local_native_') as tmp:
            out=Path(tmp)
            binary=out/'fake-vpr'
            binary.write_text('''#!/usr/bin/env python3
import os
from pathlib import Path
if 'TC_GP_INIT' in os.environ:
    print('TAILCOMPILE_V7_GP_INIT_CONSUMED path='+os.environ['TC_GP_INIT'])
    print('TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1')
Path('design.route').write_text('Net 0 (a)\\nNode: 2 CHANZ (1,2,0) to (1,2,1)\\n')
Path('design.net').write_text('temporary')
Path('design.net.post_routing').write_text('temporary')
Path('post_route.fplace').write_text('a 1 2 0 0\\n')
print('Circuit successfully routed')
print('Total wirelength: 10')
print('Final critical path delay: 5 ns')
''')
            binary.chmod(0o755)
            files={k:out/name for k,name in dict(arch='arch.xml',blif='design.blif',rr_graph='rr.xml').items()}
            for file in files.values(): file.write_text(file.name)
            device=out/'device.json';write(device,{})
            manifest=out/'rr_manifest.json'
            write(manifest,dict(binary_sha256='older-producer',channel_width=128,
                architecture_sha256=sha(files['arch']),rr_graph_sha256=sha(files['rr_graph']),graph_sha256=sha(device)))
            fields={k:str(v) for k,v in files.items()}
            fields.update(device=str(device),device_manifest=str(manifest))
            c=dict(id='x/H0Z0/w128',family='x',design='x',architecture='H0Z0',width=128,fields=fields)
            controller=SimpleNamespace(out=out,binary=binary,binary_hash=sha(binary),
                binary_reference=binary,guard=lambda:None,args=SimpleNamespace(timeout=60))
            a=run_native(controller,c,15101,out/'comparison'/c['id']/'s15101/Default','Default')
            init=out/'predictions/J/init.ap.tsv';init.parent.mkdir(parents=True);init.write_text('a 1 2 0 0\n')
            edits=init.parent/'residual_edits.json';write(edits,dict(grid=[4,4],atoms=[]))
            candidate=dict(ap_init=str(init),ap_init_sha256=sha(init),residual_edits=str(edits),
                residual_edits_sha256=sha(edits),deployed_action_sha256='a'*64)
            folder=out/'comparison'/c['id']/'s15101/J'
            b=run_native(controller,c,15101,folder,'J',candidate)
            self.assertEqual(a['execution_signature'],b['execution_signature'])
            self.assertTrue(b['input_consumed'] and b['native_solver_continued'])
            self.assertEqual(b['unique_chanz_nodes'],1)
            work=Path(read(folder/'vpr/manifest.json')['workdir'])
            self.assertFalse((work/'design.net').exists())
            self.assertFalse((work/'design.net.post_routing').exists())
            self.assertFalse((work/'design.route').exists())
            self.assertTrue((work/'payload_pruned.json').is_file())
            self.assertEqual(b,run_native(controller,c,15101,folder,'J',candidate))
            verify_measurements(out,[a,b])
            mutated=dict(b,wirelength=99.)
            with self.assertRaisesRegex(ValueError,'summary differs'):
                verify_measurements(out,[mutated])
            init.write_text('different\n')
            with self.assertRaisesRegex(ValueError,'payload changed'):
                run_native(controller,c,15101,folder,'J',candidate)


if __name__=='__main__':
    unittest.main()
