"""Read-only cache reuse binds source, actions, algorithms and the D backbone."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from edit_context_cache_import import import_cache, HERE, FEATURE_FILES, SCHEDULING_FILES
from edit_context_features import FEATURE_SCHEMA, edit_feature_dim
from train_edit_context import CACHE_SCHEMA, cache_digest, _digest, save_cache
from train_structured import save_model
from structured_model import StructuredInitializer
from train import state_sha256
from io_utils import read, write, sha


class EditContextCacheImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def fixture(self):
        temp = tempfile.TemporaryDirectory(dir=HERE/'runs', prefix='development_012_import_test_')
        self.addCleanup(temp.cleanup)
        origin = Path(temp.name); source = origin/'source011'; source.mkdir()
        torch.manual_seed(9)
        model = StructuredInitializer(3,5,hidden=16,chunk_size=2)
        checkpoint = source/'D.pt'; save_model(model,checkpoint)
        graph = source/'graph.pt'; graph.write_bytes(b'signed fixture graph')
        actions = []; feedback = []
        cid='fixture/H0Z0/w64'
        for arm in ('D','L','Y','J'):
            action=source/(arm+'.json'); write(action,dict(arm=arm))
            actions.append(action);feedback.append(dict(condition_id=cid,arm=arm,action=str(action)))
        inputs = {str(p):sha(p) for p in (checkpoint,graph,*actions)}
        records = [dict(condition_id=cid,graph=str(graph),split='train')]
        info=dict(source=str(source),source_inputs=inputs,checkpoint=str(checkpoint),
                  checkpoint_sha256=sha(checkpoint),records=records,feedback=feedback)
        code={name:sha(HERE/name) for name in FEATURE_FILES | SCHEDULING_FILES | {'train_residual.py'}}
        old=copy.deepcopy(code)
        for name in SCHEDULING_FILES:old[name]='0'*64  # Only scheduler changes are allowed.
        write(origin/'edit_context_protocol.json',dict(source=str(source),source_inputs=inputs,
            base_checkpoint_sha256=sha(checkpoint),implementation=old))
        trial={11401:dict(seed=11401,status='success',wirelength=100.,cpd_ns=10.)}
        banks=[]
        for arm in ('L','Y','J'):
            banks.append(dict(condition_id=cid,arm=arm,candidate_ids=['identity',arm.lower()+'_s2'],
                digests=['a'*64,'b'*64],trials=[copy.deepcopy(trial),copy.deepcopy(trial)],
                global_features=torch.zeros(2,96),local_features=torch.cat((torch.zeros(1,237),torch.ones(1,237))),
                generated_candidate_count=2,measured_candidate_count=2))
        cache=dict(schema=CACHE_SCHEMA,feature_schema=FEATURE_SCHEMA,banks=banks,global_dim=96,local_dim=237,
            source_checkpoint_sha256=sha(checkpoint),backbone_sha256=state_sha256(model),
            source_inputs=inputs,feedback_sha256=_digest(feedback))
        cache['cache_sha256']=cache_digest(cache)
        path=origin/'feature_cache.pt';save_cache(cache,path)
        write(origin/'prepared_report.json',dict(complete=True,cache=str(path),cache_file_sha256=sha(path),
            cache_sha256=cache['cache_sha256']))
        return origin,info,code,path

    def snapshot(self,root):
        return {str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}

    def test_scheduler_changes_reuse_original_path_without_any_writes(self):
        origin,info,code,path=self.fixture();before=self.snapshot(origin)
        result=import_cache(info,origin,current_implementation=code)
        self.assertEqual(Path(result['cache']),path)
        self.assertEqual(result['cache_file_sha256'],sha(path))
        self.assertTrue(result['original_cache_read_only']);self.assertEqual(result['new_pnr'],0)
        self.assertEqual(result['banks'],3);self.assertEqual(result['conditions'],1)
        self.assertEqual(before,self.snapshot(origin))

    def test_algorithm_and_source_changes_fail_before_cache_loading(self):
        origin,info,code,path=self.fixture();before=self.snapshot(origin)
        for name in FEATURE_FILES | {'train_residual.py'}:
            changed=dict(code);changed[name]='f'*64
            with self.assertRaisesRegex(ValueError,'implementation changed'):
                import_cache(info,origin,current_implementation=changed)
        changed=copy.deepcopy(info);changed['source']=str(origin/'another_source')
        with self.assertRaisesRegex(ValueError,'different source'):
            import_cache(changed,origin,current_implementation=code)
        changed=copy.deepcopy(info);changed['source_inputs'][str(path)]='f'*64
        with self.assertRaisesRegex(ValueError,'input signatures differ'):
            import_cache(changed,origin,current_implementation=code)
        self.assertEqual(before,self.snapshot(origin))

    def test_payload_and_dependency_mutation_cannot_be_imported(self):
        origin,info,code,path=self.fixture()
        path.write_bytes(path.read_bytes()+b'changed')
        with self.assertRaisesRegex(ValueError,'file signature mismatch'):
            import_cache(info,origin,current_implementation=code)
        origin,info,code,path=self.fixture()
        action=Path(info['feedback'][0]['action']);action.write_bytes(b'changed original action')
        with self.assertRaisesRegex(ValueError,'graph/action/checkpoint signatures differ'):
            import_cache(info,origin,current_implementation=code)

    def test_feedback_or_backbone_forgery_is_rejected_even_with_resigned_cache(self):
        origin,info,code,path=self.fixture()
        changed=copy.deepcopy(info);changed['feedback'][0]['arm']='different'
        with self.assertRaisesRegex(ValueError,'measured feedback differs'):
            import_cache(changed,origin,current_implementation=code)
        cache=torch.load(path,weights_only=False);cache['backbone_sha256']='f'*64
        cache['cache_sha256']=cache_digest(cache);save_cache(cache,path)
        prepared=read(origin/'prepared_report.json');prepared.update(cache_file_sha256=sha(path),cache_sha256=cache['cache_sha256'])
        write(origin/'prepared_report.json',prepared)
        with self.assertRaisesRegex(ValueError,'backbone differs'):
            import_cache(info,origin,current_implementation=code)

    def test_noncanonical_origin_and_incomplete_preparation_rejected(self):
        origin,info,code,path=self.fixture()
        with self.assertRaisesRegex(ValueError,'canonical development_012'):
            import_cache(info,origin/'nested',current_implementation=code)
        (origin/'prepared_report.json').unlink()
        with self.assertRaisesRegex(ValueError,'completed signed preparation'):
            import_cache(info,origin,current_implementation=code)


if __name__=='__main__':unittest.main()
