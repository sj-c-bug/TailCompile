"""Preparation-only cache recovery must preserve the frozen experiment evidence."""
import copy
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import complete_device_cache as recovery
from campaign import campaign_lock, identity
from device_graph import build_device_graph
from io_utils import COMMIT, read, sha, write

HERE = Path(__file__).resolve().parent


class DeviceCacheCompletionTests(unittest.TestCase):
    def temporary(self):
        (HERE/'runs').mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='cache-completion-', dir=HERE/'runs')

    def fixture(self, root):
        repo = root/'repo'
        out = repo/'experiments/vtr/tailcompile_v7_3d/runs/cohort'
        out.mkdir(parents=True)
        binary = root/'readonly_backend/vpr'
        binary.parent.mkdir()
        binary.write_bytes(b'qualified fixed backend')
        binary_hash = sha(binary)
        reference = out/'binary_identity.json'
        write(reference, dict(binary=str(binary), sha256=binary_hash, commit=COMMIT,
                              backend_manifest_sha256='backend-source'))
        arch = out/'architectures/84x84/H0Z0/arch.xml'
        arch.parent.mkdir(parents=True)
        arch.write_text('<architecture/>')
        blif = out/'inputs/mcml/design.blif'
        blif.parent.mkdir(parents=True)
        blif.write_text('.model mcml\n.end\n')
        key = identity(dict(architecture=sha(arch),binary=binary_hash,width=80))[:20]
        folder = out/'device_cache'/key
        folder.mkdir(parents=True)
        rrg = folder/'rr_graph.xml'
        rrg.write_bytes(b'complete existing RRG output')
        command = [str(binary),str(arch),str(blif),'--pack','--device','pilot',
                   '--route_chan_width','80','--place_chan_width','80',
                   '--num_workers','1','--write_rr_graph',str(rrg)]
        signature = dict(command=command, inputs={str(p):sha(p) for p in (reference,arch,blif)},
                         env={},timeout=3600,marker=None,affinity=None,
                         metadata=dict(stage='reusable_rrg',width=80))
        write(folder/'build/manifest.json',dict(status='complete',returncode=0,signature=signature))
        write(out/'protocol.json',dict(untouched=True))
        write(out/'calibration/mcml/w64/result.json',dict(status='unroutable',not_a_parser_failure=True))
        context = dict(binary=binary,binary_sha256=binary_hash,binary_reference=reference,
                       tile_size=4,source_hashes={},protocol_sha256=sha(out/'protocol.json'))
        return repo,out,folder,context

    def factory(self, rr_graph, architecture, tile_size, max_seconds):
        self.assertEqual(tile_size,4)
        self.assertEqual(max_seconds,3600)
        return dict(schema='tailcompile-v7-layered-device-1',tile_size=tile_size,channel_width=80,
                    provenance=dict(rr_graph_sha256=sha(rr_graph),architecture_sha256=sha(architecture)),
                    extraction_seconds=301,region_x=[[1]],capacity=[[2]],pools=[],audit={})

    @unittest.skipUnless(os.name=='posix','Linux server stage lock')
    def test_completes_only_preparation_and_exact_repeat_does_not_rewrite_cache(self):
        with self.temporary() as tmp:
            repo,out,folder,context=self.fixture(Path(tmp).resolve())
            preserved={str(p):sha(p) for p in out.rglob('*') if p.is_file()}
            with patch.object(recovery,'_validate_protocol',return_value=context), \
                 patch.object(recovery,'build_device_graph',side_effect=self.factory) as build:
                self.assertEqual(recovery.complete_pending(out,3600,repo=repo),[folder.name])
                manifest=read(folder/'manifest.json')
                self.assertEqual(manifest['rr_graph_sha256'],sha(folder/'rr_graph.xml'))
                self.assertEqual(manifest['graph_sha256'],sha(folder/'device_graph.json'))
                self.assertEqual(manifest['channel_width'],80)
                report=read(folder/'extraction_completion_report.json')
                self.assertTrue(report['preparation_only'])
                self.assertTrue(report['no_vpr_execution'])
                self.assertEqual(report['status'],'complete')
                cache_before={str(p):sha(p) for p in folder.rglob('*') if p.is_file()}
                self.assertEqual(recovery.complete_pending(out,3600,repo=repo),[])
                self.assertEqual(build.call_count,1)
                self.assertEqual(cache_before,{p:sha(p) for p in cache_before})
            self.assertEqual(preserved,{p:sha(p) for p in preserved})

    @unittest.skipUnless(os.name=='posix','Linux server stage lock')
    def test_timeout_preserves_raw_rrg_and_build_without_publishing_a_cache(self):
        with self.temporary() as tmp:
            repo,out,folder,context=self.fixture(Path(tmp).resolve())
            before={str(p):sha(p) for p in folder.rglob('*') if p.is_file()}
            with patch.object(recovery,'_validate_protocol',return_value=context), \
                 patch.object(recovery,'build_device_graph',side_effect=TimeoutError('still exceeds budget')):
                with self.assertRaisesRegex(TimeoutError,'still exceeds'):
                    recovery.complete_pending(out,3600,repo=repo)
            self.assertFalse((folder/'manifest.json').exists())
            self.assertFalse((folder/'device_graph.json').exists())
            report=read(folder/'extraction_completion_report.json')
            self.assertEqual(report['status'],'failed')
            self.assertTrue(report['rr_graph_preserved'])
            self.assertEqual(before,{p:sha(p) for p in before})

    @unittest.skipUnless(os.name=='posix','Linux server stage lock')
    def test_changed_input_rrg_and_invalid_complete_cache_are_not_silently_accepted(self):
        for kind in ('input','rrg','manifest'):
            with self.subTest(kind=kind),self.temporary() as tmp:
                repo,out,folder,context=self.fixture(Path(tmp).resolve())
                if kind=='input':
                    (out/'inputs/mcml/design.blif').write_text('changed')
                    action=self.factory
                elif kind=='rrg':
                    def action(rr_graph,architecture,tile_size,max_seconds):
                        graph=self.factory(rr_graph,architecture,tile_size,max_seconds)
                        rr_graph.write_text('modified during extraction')
                        return graph
                else:
                    write(folder/'manifest.json',dict(architecture_sha256='wrong',
                          binary_sha256=context['binary_sha256'],channel_width=80))
                    action=self.factory
                with patch.object(recovery,'_validate_protocol',return_value=context), \
                     patch.object(recovery,'build_device_graph',side_effect=action):
                    with self.assertRaises(ValueError):
                        recovery.complete_pending(out,3600,repo=repo)
                if kind!='manifest':
                    self.assertFalse((folder/'manifest.json').exists())
                else:
                    self.assertEqual(read(folder/'manifest.json')['architecture_sha256'],'wrong')

    def test_bad_output_path_and_budget_rejected_before_runtime_mutation(self):
        with self.temporary() as tmp:
            root=Path(tmp).resolve()
            repo,out,folder,context=self.fixture(root)
            for budget in (0,-1,float('inf'),True,None):
                with self.subTest(budget=budget),self.assertRaises(ValueError):
                    recovery.complete_pending(out,budget,repo=repo)
            forbidden=root/'outside'
            with self.assertRaisesRegex(ValueError,'canonical'):
                recovery.complete_pending(forbidden,3600,repo=repo)
            self.assertFalse(forbidden.exists())

    def test_stage_lock_prevents_recovery_while_campaign_is_running(self):
        if not hasattr(recovery.signal,'SIGALRM'):
            self.skipTest('Linux server locking contract')
        with self.temporary() as tmp:
            repo,out,folder,context=self.fixture(Path(tmp).resolve())
            with patch.object(recovery,'_validate_protocol',return_value=context),campaign_lock(out,'calibrate'):
                with self.assertRaisesRegex(RuntimeError,'another campaign stage'):
                    recovery.complete_pending(out,3600,repo=repo)
            self.assertFalse((folder/'manifest.json').exists())

    def test_linux_deadline_stops_late_checked_phase(self):
        if not hasattr(recovery.signal,'setitimer'):
            self.skipTest('Linux wall clock deadline')
        with self.assertRaisesRegex(TimeoutError,'wall-clock'):
            with recovery._extraction_deadline(.03):
                time.sleep(.2)

    @unittest.skipUnless(os.name=='posix','Linux symlink contract')
    def test_stage_lock_symlink_cannot_write_outside_the_cohort(self):
        with self.temporary() as tmp:
            repo,out,folder,context=self.fixture(Path(tmp).resolve())
            external=Path(tmp)/'outside_lock'
            external.write_text('preserved')
            (out/'.campaign-stage.lock').symlink_to(external)
            with patch.object(recovery,'_validate_protocol',return_value=context):
                with self.assertRaisesRegex(ValueError,'lock escapes'):
                    recovery.complete_pending(out,3600,repo=repo)
            self.assertEqual(external.read_text(),'preserved')
            self.assertFalse((folder/'manifest.json').exists())

    def test_budget_change_keeps_real_tiny_graph_features_identical(self):
        from test_device_graph import fixture_module
        with self.temporary() as tmp:
            folder=Path(tmp)
            arch,root,nodes,edges=fixture_module.fixture(folder,width=10,height=6)
            tree=ET.parse(arch)
            cbl=ET.SubElement(tree.getroot(),'complexblocklist')
            for typ in ('clb','memory','dsp_top','io'):
                pb=ET.SubElement(cbl,'pb_type',name=typ)
                if typ=='clb':
                    ET.SubElement(pb,'pb_type',name='lut',num_pb='4',blif_model='.names')
                    ET.SubElement(pb,'pb_type',name='ff',num_pb='8',blif_model='.latch')
                else:
                    pb.set('blif_model','.input' if typ=='io' else '.subckt hard')
                sub=tree.getroot().find(f"./tiles/tile[@name='{typ}']/sub_tile")
                ET.SubElement(ET.SubElement(sub,'equivalent_sites'),'site',pb_type=typ)
            tree.write(arch)
            fixture_module.wire(nodes,0,'CHANX','INC_DIR',x1=9)
            fixture_module.wire(nodes,1,'CHANX','INC_DIR',x1=9,z0=1,z1=1)
            fixture_module.wire(nodes,2,'CHANZ','BI_DIR',x1=0,z1=1,capacity=3)
            fixture_module.connect(edges,0,2,repeats=4)
            fixture_module.connect(edges,2,1,repeats=4)
            rrg=folder/'rr.xml'
            ET.ElementTree(root).write(rrg)
            first=build_device_graph(rrg,arch,tile_size=4,max_seconds=5)
            second=build_device_graph(rrg,arch,tile_size=4,max_seconds=30)
            first.pop('extraction_seconds')
            second.pop('extraction_seconds')
            self.assertEqual(first,second)


if __name__=='__main__':
    unittest.main()
