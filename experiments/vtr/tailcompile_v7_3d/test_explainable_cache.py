"""016 cache reuse follows the complete old receipt chain, never a signature alone."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from explainable_campaign import prepare_vertical_summary
from io_utils import read, write, sha
import vertical_risk

HERE=Path(__file__).resolve().parent


class ExplainableCacheTests(unittest.TestCase):
    def setUp(self):
        self.target_tmp=tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='development_016_local_cache_')
        self.source_tmp=tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='development_015_local_cache_')
        self.addCleanup(self.target_tmp.cleanup)
        self.addCleanup(self.source_tmp.cleanup)
        self.root=Path(self.source_tmp.name)
        self.out=Path(self.target_tmp.name)
        self.cid='case/H0Z0/w128'
        self.target=self.out/'predictions'/self.cid/'vertical_summary.json'
        self.cache=self.root/'predictions'/self.cid/'vertical_summary.json'
        # This is deliberately not parseable XML. A verified cache hit must
        # only hash it, avoiding the multi-GB XML graph traversal entirely.
        self.rr=self.out/'rr.xml'; self.rr.write_text('hash-only RR input')
        self.device=dict(grid_width=4,grid_height=4,layers=2,tile_size=4)
        self.device_path=self.out/'device.json';write(self.device_path,self.device)
        self.summary=dict(signature=dict(schema=vertical_risk.SCHEMA,
            implementation=sha(Path(vertical_risk.__file__)),rr_graph_sha256=sha(self.rr),
            device_sha256=hashlib.sha256(json.dumps(self.device,sort_keys=True,allow_nan=False).encode()).hexdigest()),
            pools=[],marker='original frozen payload')
        write(self.cache,self.summary)
        self.request_path=self.cache.parent/'request.json'
        write(self.request_path,dict(device_signature='device-016',vertical_summary=str(self.cache),
            inputs={str(self.cache):sha(self.cache)}))
        self.report_path=self.root/'predictions/prediction_report.json'
        write(self.report_path,dict(conditions={self.cid:dict(device_signature='device-016',
            request_sha256=sha(self.request_path))}))
        write(self.root/'predictions/selection.json',dict(pre_measurement=True,predictions_sha256=sha(self.report_path)))
        self.condition=dict(id=self.cid,fields=dict(rr_graph=str(self.rr),device=str(self.device_path)))

    def prepare(self):
        return prepare_vertical_summary(self.condition,self.target,max_seconds=5,
            expected_device_signature='device-016',source_root=self.root)

    def test_verified_reuse_skips_xml_parse_and_resumes_without_touching_old_cache(self):
        old=self.cache.read_bytes()
        with patch('vertical_risk.ET.iterparse',side_effect=AssertionError('must not reparse RR XML')):
            receipt=self.prepare()
            self.assertEqual(receipt['mode'],'reused_015')
            self.assertEqual(self.target.read_bytes(),old)
            self.assertEqual(self.cache.read_bytes(),old)
            self.assertEqual(receipt['source_cache_sha256'],sha(self.cache))
            self.assertEqual(self.prepare(),receipt)

    def _fake_rebuild(self,rr_graph,device,out,*,max_seconds):
        self.assertFalse(Path(out).exists())
        write(out,dict(self.summary,marker='independently rebuilt'))
        return read(out)

    def test_signature_preserving_cache_tamper_falls_back_to_new_owned_rebuild(self):
        # Even an unchanged embedded signature cannot bypass the request hash.
        write(self.cache,dict(self.summary,marker='tampered but signature unchanged'))
        old=self.cache.read_bytes()
        with patch('vertical_risk.prepare_device',side_effect=self._fake_rebuild) as derive:
            receipt=self.prepare()
        self.assertEqual(derive.call_count,1)
        self.assertEqual(receipt['mode'],'rebuilt')
        self.assertIn('differs from frozen request',receipt['reuse_rejected_reason'])
        self.assertEqual(self.cache.read_bytes(),old)
        self.assertEqual(read(self.target)['marker'],'independently rebuilt')

    def test_request_tamper_is_rejected_before_copy(self):
        request=read(self.request_path);request['device_signature']='wrong'
        write(self.request_path,request)
        with patch('vertical_risk.prepare_device',side_effect=self._fake_rebuild):
            receipt=self.prepare()
        self.assertEqual(receipt['mode'],'rebuilt')
        self.assertIn('receipt mismatch',receipt['reuse_rejected_reason'])

    def test_global_report_tamper_falls_back(self):
        report=read(self.report_path);report['extra']='unlocked change'
        write(self.report_path,report)
        with patch('vertical_risk.prepare_device',side_effect=self._fake_rebuild):
            receipt=self.prepare()
        self.assertEqual(receipt['mode'],'rebuilt')
        self.assertIn('receipt mismatch',receipt['reuse_rejected_reason'])

    def test_resume_rejects_target_digest_tamper(self):
        self.prepare()
        write(self.target,dict(self.summary,marker='changed own cache'))
        with self.assertRaisesRegex(ValueError,'frozen payload changed'):
            self.prepare()

    def test_interrupted_unpublished_derived_cache_can_be_recovered(self):
        write(self.target,dict(marker='incomplete copy before receipt'))
        with patch('vertical_risk.ET.iterparse',side_effect=AssertionError('verified source available')):
            receipt=self.prepare()
        self.assertEqual(receipt['mode'],'reused_015')
        self.assertEqual(self.target.read_bytes(),self.cache.read_bytes())


if __name__=='__main__':
    unittest.main()
