"""014 candidate identity, prospective locking and native process contracts."""
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from io_utils import read, write, sha
import prospective_native as native

HERE = Path(__file__).resolve().parent
LOCAL = HERE/'runs/development_012_edit_context_local'


class ProspectiveNativeTests(unittest.TestCase):
    def temp(self, prefix):
        return tempfile.TemporaryDirectory(dir=HERE/'runs', prefix=prefix)

    def test_models_require_real_full_fit_and_hash(self):
        with self.temp('development_010_t_') as old, self.temp('development_012_t_') as a, self.temp('development_013_t_') as b:
            old, a, b = map(Path, (old,a,b))
            base = old/'D.pt'; base.write_bytes(b'D')
            write(a/'edit_context_protocol.json',dict(base_checkpoint=str(base),base_checkpoint_sha256=sha(base)))
            indices = {a:[],b:[]}
            for root, variant, key in ((a,'G','mode'),(a,'E','mode'),(b,'E_h8_train_std_l2p01','config_id')):
                folder = root/'models/full_fit'/variant/'s12301'
                folder.mkdir(parents=True)
                model = folder/'epoch_0080.pt'; model.write_bytes(b'head')
                report = folder/'fit_report.json'
                write(report,dict(split={'id':'full_fit'},seed=12301,training=dict(mode=variant[0],epochs=80,
                    factor_config={'id':variant},source_checkpoint_sha256=sha(base),checkpoint=str(model),
                    checkpoint_sha256={str(model):sha(model)},backbone_sha256='backbone',cache_sha256='cache',train_ids=['a'])))
                indices[root].append(dict(split_id='full_fit',seed=12301,**{key:variant},
                                          path=str(report),sha256=sha(report)))
            for root, index in indices.items():
                write(root/'trained_report.json',dict(complete=True,fits=index))
            self.assertEqual(set(native.model_sources(a,b)),{'D','G012','E012','E013'})
            model.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError,'signature'):
                native.model_sources(a,b)

    def test_missing_lock_after_comparison_never_restarts_predictions(self):
        with self.temp('development_014_native_test_') as out:
            out = Path(out)
            (out/'comparison').mkdir()
            with self.assertRaisesRegex(ValueError,'after primary'):
                native.predict_panel(SimpleNamespace(out=out),[{'id':str(i)} for i in range(8)],{})

    def test_duplicate_candidate_bytes_share_physical_trials_within_same_seed_only(self):
        from prospective_evaluation import evaluate
        with self.temp('development_014_native_test_') as tmp:
            out=Path(tmp)
            conditions=[]; catalog={}
            for design in ('a','b'):
                for arch in ('H0Z0','H0Z1','H1Z0','H1Z1'):
                    cid=f'{design}/{arch}/w128'
                    conditions.append(dict(id=cid, family=design,design=design,architecture=arch,width=128))
                    folder=out/'predictions'/cid
                    folder.mkdir(parents=True)
                    candidates={}
                    for name in native.METHODS[1:]:
                        file=folder/(name+'.tsv')
                        file.write_text('strong' if name=='j_s2' else 'D')
                        candidates[name]=dict(ap_init=str(file),ap_init_sha256=sha(file),deployed_action_sha256=sha(file))
                    choices={k:dict(method='D',candidate_ids=['D','j_s2'],scores=[0.,-1.],
                        scoring_seconds=.1,ap_init_sha256=candidates['D']['ap_init_sha256'],
                        deployed_action_sha256=candidates['D']['deployed_action_sha256']) for k in ('E012','G012','E013')}
                    pred=dict(models={},condition_id=cid,candidates=candidates,choices=choices,
                              device_signature='c'*64,graph_sha256='a'*64,generation_seconds=1.,
                              prediction_worker_internal_seconds=1.3,generation_seconds_excludes_scorer_times=True)
                    payload=folder/'payload/prediction_report.json';write(payload,pred)
                    write(folder/'request.json',dict(inputs={}))
                    catalog[cid]=dict(pred,prediction_payload_sha256=sha(payload),
                                      prediction_request_sha256=sha(folder/'request.json'))
            panel=dict(models={},conditions=catalog)
            write(out/'predictions/prediction_report.json',panel)
            campaign=SimpleNamespace(out=out,parallel=lambda items,fn:[fn(i) for i in items])
            physical=[]
            def native_fixture(camp,c,seed,folder,method,candidate):
                physical.append((c['id'],seed,method))
                return dict(condition_id=c['id'],design=c['design'],family=c['family'],architecture=c['architecture'],
                    width=128,seed=seed,method=method,status='success',device_signature='c'*64,
                    execution_signature=native.identity([c['id'],seed]),wirelength=100.,cpd_ns=1.,
                    ap_init_sha256=candidate['ap_init_sha256'] if candidate else None,
                    deployed_action_sha256=candidate['deployed_action_sha256'] if candidate else None,
                    input_consumed=bool(candidate),native_solver_continued=bool(candidate),vpr_manifest=str(folder/'manifest.json'),
                    vpr_seconds=1.,unique_chanz_nodes=1)
            with patch.object(native,'run_native',side_effect=native_fixture),patch.object(native,'_device',return_value=({},'c'*64)):
                rows=native.run_panel(campaign,conditions,panel)
            self.assertEqual(len(physical),72)
            self.assertEqual(len(rows),96)
            self.assertEqual(sum(bool(r.get('identical_input_reuse')) for r in rows),24)
            self.assertEqual(len({(r['condition_id'],r['seed'],r['method']) for r in rows}),96)
            evaluate(rows,conditions,panel)

    @unittest.skipUnless(os.name=='posix','native Linux process')
    def test_native_pins_inputs_requires_consumer_and_prunes_after_metrics(self):
        with self.temp('development_014_native_test_') as out:
            out = Path(out)
            binary, arch, blif, rr = (out/n for n in ('fake-vpr','arch','design.blif','rr.xml'))
            binary.write_text("""#!/usr/bin/env python3
import os
from pathlib import Path
if 'TC_GP_INIT' in os.environ:
    print('TAILCOMPILE_V7_GP_INIT_CONSUMED path='+os.environ['TC_GP_INIT'])
    print('TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1')
Path('design.route').write_text('Net 0 (a)\\nNode: 2 CHANZ (1,2,0) to (1,2,1)\\n')
Path('design.net').write_text('transient')
Path('post_route.fplace').write_text('a 1 2 0 0\\n')
print('Circuit successfully routed')
print('Total wirelength: 10')
print('Final critical path delay: 5 ns')
""")
            binary.chmod(0o755)
            for path in (arch,blif,rr):
                path.write_text(path.name)
            dev = out/'device.json'; write(dev,{})
            manifest = out/'rr_manifest.json'
            write(manifest,dict(binary_sha256=sha(binary),channel_width=128,architecture_sha256=sha(arch),
                rr_graph_sha256=sha(rr),graph_sha256=sha(dev)))
            campaign = SimpleNamespace(out=out, binary=binary,binary_hash=sha(binary),binary_reference=binary,
                guard=lambda:None,check_binary=lambda:None,args=SimpleNamespace(timeout=60),
                architecture=lambda *_:arch,rr=lambda *_:dict(rr_graph=str(rr),device=str(dev),manifest=str(manifest)))
            c=dict(id='x/H0Z1/w128',family='x',design='x',architecture='H0Z1',split='validation',width=128,
                   record={'blif':str(blif)})
            default=native.run_native(campaign,c,14101,out/'native/default','Default')
            init=out/'predictions/D/ap.tsv';init.parent.mkdir(parents=True);init.write_text('a 1 2 0 0\n')
            candidate=dict(ap_init=str(init),ap_init_sha256=sha(init),deployed_action_sha256='a'*64)
            result=native.run_native(campaign,c,14101,out/'native/d','D',candidate)
            self.assertEqual(default['execution_signature'],result['execution_signature'])
            self.assertTrue(result['input_consumed'])
            self.assertEqual(result['unique_chanz_nodes'],1)
            trial=read(result['vpr_manifest'])
            self.assertFalse((Path(trial['workdir'])/'design.net').exists())
            self.assertTrue((Path(trial['workdir'])/'payload_pruned.json').exists())
            self.assertEqual(result,native.run_native(campaign,c,14101,out/'native/d','D',candidate))
            init.write_text('different')
            with self.assertRaisesRegex(ValueError,'signature'):
                native.run_native(campaign,c,14102,out/'native/bad','D',candidate)
            self.assertFalse((out/'native/bad').exists())

    @unittest.skipUnless(os.name=='posix' and (LOCAL/'feature_cache.pt').is_file(),
                         'requires the local tiny 012 fixtures')
    def test_real_local_014_prediction_worker_matches_deployment_and_bytes(self):
        import torch
        from train import state_sha256
        from train_edit_context import load_cache,load_model as load012
        from generalization_spec import config_by_id, PRIMARY_CONFIG
        from train_generalization import fit_cache,load_model as load013
        from train_residual import from_source
        from aligned_model import action_digest
        from generation import lower_initialization
        from data import build_logic
        from fast_graph import make_graph
        from layer_features import augment_graph
        torch.set_num_threads(1)
        with self.temp('development_014_native_test_') as out:
            out=Path(out)
            req=read(HERE/'runs/development_007_layer_local/worker_contract/request.json')
            req.update(arch=req['architecture'],out=str(out),condition_id='local/asymmetric/w320',
                       device_signature='engineered-asymmetric')
            base=HERE/'runs/development_010_structured_local/models/decoder.pt'
            cache=load_cache(LOCAL/'feature_cache.pt')
            ids=sorted({b['condition_id'] for b in cache['banks']})
            fit=fit_cache(cache,ids,out/'local_013',config=config_by_id(PRIMARY_CONFIG),snapshots=(80,))
            models=dict(D=dict(path=str(base),sha256=sha(base)))
            for name,path in [('E013',fit['checkpoint'])]+[(m+'012',LOCAL/'fits/full_fit/s12301'/m/'epoch_0080.pt') for m in ('G','E')]:
                models[name]=dict(path=str(path),sha256=sha(path),backbone_sha256=cache['backbone_sha256'],
                                  cache_sha256=cache['cache_sha256'],train_ids=ids)
            req.update(models=models,inputs={b['path']:b['sha256'] for b in models.values()})
            pins = Path(req['prefix']+'.pins.tsv').resolve()
            if pins.exists():
                req['inputs'][str(pins)]=sha(pins)
            output=out/'predictions/local/payload';request=out/'request.json';write(request,req)
            result=subprocess.run([sys.executable,HERE/'prospective_native.py','worker',request,output],
                                  capture_output=True,text=True,timeout=120)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            pred=read(output/'prediction_report.json')
            logic=build_logic(req['prefix'],req['design'],req['blif'],req['arch'],family=req['family'],split=req['split'])
            device=read(req['device'])
            graph=augment_graph(make_graph(logic,device),logic,device,read(req['vertical_summary']),req['blif'],req['prefix'])
            for name,choice in pred['choices'].items():
                scorer=(load013 if name=='E013' else load012)(models[name]['path'])
                selected=scorer.select_action(graph,'J',base)
                lowered=lower_initialization(logic,device,selected,out/'independent'/name)
                self.assertEqual(action_digest(selected),choice['deployed_action_sha256'])
                self.assertEqual(lowered['ap_init_sha256'],choice['ap_init_sha256'])
            self.assertEqual(set(pred['candidates']),{'D','j_s1','j_s2'})
            self.assertEqual(state_sha256(from_source(base).backbone),cache['backbone_sha256'])


if __name__=='__main__':
    unittest.main()
