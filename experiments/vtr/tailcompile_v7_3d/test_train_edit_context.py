"""012 source support, fold isolation, head-only learning and checkpoint pins."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from test_residual_model import source_case
import test_train_residual as old_training_tests
from train_edit_context import (prepare_cache,validate_cache,cache_digest,save_cache,load_cache,
    fit_cache,load_model,select_indices,evaluate_cache)
from train import state_sha256


class EditContextTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(1)

    def folder(self):
        return tempfile.TemporaryDirectory(dir=Path(__file__).parent/'runs',prefix='test_edit_train_')

    def prepare(self,path):
        data,source=source_case(path)
        helper=old_training_tests.ResidualTrainingTests()
        entries=helper.entries(data,source)
        records=[];all_entries=[]
        for family in ('alpha','beta'):
            cid=family+'/H0Z0/w64'
            records.append(dict(condition_id=cid,family=family,split='train',graph=data))
            for entry in entries:
                item=copy.deepcopy(entry);item['condition_id']=cid;all_entries.append(item)
        return prepare_cache(records,source,all_entries),source,records,all_entries

    def test_preparation_preserves_global_features_and_exact_measured_support(self):
        with self.folder() as tmp:
            cache,source,records,entries=self.prepare(Path(tmp))
            self.assertEqual(len(cache['banks']),6)
            self.assertEqual(cache['new_pnr'],0)
            self.assertFalse(cache['graph_or_teacher_labels_used_as_features'])
            for bank in cache['banks']:
                self.assertEqual(bank['candidate_ids'][0],'identity')
                self.assertFalse(torch.count_nonzero(bank['local_features'][0]))
                self.assertEqual(len(bank['candidate_ids']),len(bank['trials']))
                self.assertEqual(bank['digests'][0],bank['full_candidate_digests'][0])
            saved=save_cache(cache,Path(tmp)/'cache.pt')
            self.assertEqual(load_cache(saved)['cache_sha256'],cache['cache_sha256'])
            bad=copy.deepcopy(cache);bad['banks'][0]['local_features'][1,0]+=.1
            with self.assertRaisesRegex(ValueError,'hash mismatch'):validate_cache(bad)
            bad=copy.deepcopy(cache)
            first=next(iter(bad['banks'][0]['trials'][0].values()))
            first['status']='timeout';bad['cache_sha256']=cache_digest(bad)
            with self.assertRaisesRegex(ValueError,'infrastructure/unknown'):validate_cache(bad)
            changed=copy.deepcopy(entries)
            changed[1]['action']['xy'][0,0]+=.01
            with self.assertRaisesRegex(ValueError,'digest mismatch'):
                prepare_cache(records,source,changed)

    def test_matched_training_changes_head_roundtrips_and_measures_actual_selection(self):
        with self.folder() as tmp:
            path=Path(tmp);cache,source,_,_=self.prepare(path)
            ids=['alpha/H0Z0/w64']
            reports={m:fit_cache(cache,ids,path/m,mode=m,epochs=3,snapshots=(1,3)) for m in ('G','E')}
            self.assertEqual(reports['G']['initial_state_sha256'],reports['E']['initial_state_sha256'])
            self.assertEqual(reports['G']['parameter_count'],reports['E']['parameter_count'])
            self.assertEqual(reports['G']['train_pairs'],reports['E']['train_pairs'])
            for mode,r in reports.items():
                self.assertNotEqual(r['initial_state_sha256'],r['final_state_sha256'])
                self.assertFalse(r['backbone_trained'])
                self.assertFalse(r['validation_labels_used_for_optimization'])
                self.assertEqual(r['new_pnr'],0)
                model=load_model(r['checkpoint'])
                self.assertEqual(state_sha256(model),r['final_state_sha256'])
                with patch('train_edit_context._pairs',side_effect=AssertionError('labels read during selection')):
                    predictions=select_indices(model,cache,['beta/H0Z0/w64'])
                self.assertEqual(len(predictions),3)
                evaluated=evaluate_cache(model,cache,['beta/H0Z0/w64'])
                self.assertEqual(len(evaluated['rows']),3)
                self.assertFalse(evaluated['selection_used_labels'])
                for row,pred in zip(evaluated['rows'],predictions):
                    self.assertEqual(row['selected_digest'],pred['selected_digest'])
                    self.assertGreaterEqual(row['q_regret'],-1e-12)
                payload=torch.load(r['checkpoint'],weights_only=False)
                payload['state_dict']['heads.L.2.weight'][0,0]+=.01
                torch.save(payload,path/f'tampered_{mode}.pt')
                with self.assertRaisesRegex(ValueError,'hash mismatch'):load_model(path/f'tampered_{mode}.pt')
                for key,value in (('mode','E' if mode=='G' else 'G'),('source_checkpoint_sha256','0'*64),('cache_sha256','0'*64)):
                    payload=torch.load(r['checkpoint'],weights_only=False)
                    payload['config'][key]=value
                    torch.save(payload,path/'metadata_tampered.pt')
                    with self.assertRaisesRegex(ValueError,'metadata/state hash mismatch'):
                        load_model(path/'metadata_tampered.pt')

    def test_heldout_labels_never_fit_scale_or_change_updates(self):
        with self.folder() as tmp:
            path=Path(tmp);cache,_,_,_=self.prepare(path);changed=copy.deepcopy(cache)
            for bank in changed['banks']:
                if bank['family']=='beta':
                    bank['global_features'][1:]*=100
                    for trials in bank['trials'][1:]:
                        for trial in trials.values():trial['wirelength']*=.001;trial['cpd_ns']*=10000
            changed['cache_sha256']=cache_digest(changed)
            a=fit_cache(cache,['alpha/H0Z0/w64'],path/'original',mode='E',epochs=2,snapshots=(2,))
            b=fit_cache(changed,['alpha/H0Z0/w64'],path/'changed',mode='E',epochs=2,snapshots=(2,))
            self.assertEqual(a['calibration'],b['calibration'])
            self.assertEqual(a['initial_state_sha256'],b['initial_state_sha256'])
            self.assertEqual(a['final_state_sha256'],b['final_state_sha256'])
            self.assertEqual(a['train_pairs'],b['train_pairs'])
            self.assertFalse(a['calibration']['holdout_labels_used'])

    def test_partial_measured_bank_and_zero_signal_are_not_fabricated(self):
        with self.folder() as tmp:
            path=Path(tmp);cache,source,records,entries=self.prepare(path)
            partial=[e for e in entries if not e['candidate_id'].endswith('s1')]
            small=prepare_cache(records,source,partial)
            self.assertTrue(all(b['measured_candidate_count']<=b['generated_candidate_count'] for b in small['banks']))
            identity=prepare_cache(records,source,[e for e in entries if e['arm']=='D'])
            with self.assertRaisesRegex(ValueError,'no effective'):
                fit_cache(identity,['alpha/H0Z0/w64'],path/'empty',epochs=1,snapshots=(1,))
            self.assertFalse((path/'empty/epoch_0001.pt').exists())
            self.assertTrue((path/'empty/zero_signal_report.json').is_file())
            with self.assertRaisesRegex(ValueError,'bounded'):
                fit_cache(cache,['alpha/H0Z0/w64'],path/'long',epochs=81,snapshots=(81,))


if __name__=='__main__':unittest.main()
