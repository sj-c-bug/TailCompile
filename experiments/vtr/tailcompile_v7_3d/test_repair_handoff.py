"""Run real new graph workers; exercise read-only handoff and 200-case scheduler."""
from collections import Counter
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import repair_learning as repair
import qualified_learning as qualified
from io_utils import read,write,sha
import test_qualified_handoff as old_handoff
from test_learning_repair import graph_fixture

HERE=Path(__file__).resolve().parent


class RepairHandoffTests(unittest.TestCase):
    def temporary(self):
        return tempfile.TemporaryDirectory(dir=HERE/'runs',prefix='repair-handoff-')

    def test_catalog_readonly_and_mutation_rejected(self):
        helper=old_handoff.QualifiedHandoffTests()
        with self.temporary() as tmp:
            f=helper.fixture(Path(tmp))
            prior=f.out;out=f.root/'repair006'
            with patch.object(qualified,'_validate_protocol',return_value=f.context):
                qualified.prepare_handoff(f.source,prior)
                records=[]
                for c in f.conditions:
                    folder=prior/'graphs'/c['id']
                    folder.mkdir(parents=True)
                    (folder/'graph.pt').write_bytes(b'structural graph '+c['id'].encode())
                    write(folder/'target.json',{'immutable':True})
                    records.append(dict(condition_id=c['id'],family=c['family'],split=c['split'],
                        teacher_seed=511,graph=str(folder/'graph.pt'),teacher=str(folder/'target.json')))
                write(prior/'teacher_records.json',dict(records=records))
                model=prior/'checkpoint.pt';model.write_bytes(b'pretrained checkpoint')
                write(prior/'models/pretrain/completed.json',dict(report=dict(checkpoints=[str(model)]),checkpoint_sha256={str(model):sha(model)}))
                write(prior/'models/pretrain/input_signature.json',dict(records_sha256=sha(prior/'teacher_records.json'),
                    record_payloads={r['condition_id']:{k:sha(r[k]) for k in ('graph','teacher')}
                                     for r in records if r['split']=='train'}))
                write(prior/'selection.json',dict(checkpoint=str(model),checkpoint_sha256=sha(model)))
                before=helper.snapshot(prior,f.source,f.borrowed,f.backend)
                catalog,protocol=repair.prepare_catalog(f.source,prior,out)
                self.assertEqual(len(protocol['teachers']),40)
                self.assertEqual(before,helper.snapshot(prior,f.source,f.borrowed,f.backend))
                self.assertEqual((catalog,protocol),repair.prepare_catalog(f.source,prior,out))
                Path(records[0]['teacher']).write_text('modified')
                with self.assertRaisesRegex(ValueError,'changed'):
                    repair.prepare_catalog(f.source,prior,out)

    @unittest.skipUnless(hasattr(os,'sched_setaffinity'),'Linux graph subprocess')
    def test_real_graph_worker_profile_and_resume_cost(self):
        with self.temporary() as tmp:
            root=Path(tmp);prefix=root/'structure'
            Path(str(prefix)+'.meta.tsv').write_text('16\t16\t2\n')
            Path(str(prefix)+'.blocks.tsv').write_text('\n'.join(
                f'{i}\ta{i}\t0\t0\t-1\t-1\t-1\t-1\t.names\t{i}' for i in range(10))+'\n')
            Path(str(prefix)+'.nets.tsv').write_text('n0\t0\t1\t2\n')
            arch=root/'arch.xml';arch.write_text('<architecture><complexblocklist><pb_type name="clb"><pb_type name="lut" blif_model=".names"/></pb_type></complexblocklist></architecture>')
            blif=root/'input.blif';blif.write_text('.model test\n.end\n')
            _,device=graph_fixture(n=10)
            write(root/'device.json',device)
            request=dict(prefix=str(prefix),design='tiny',family='tiny',split='train',blif=str(blif),
                         arch=str(arch),device=str(root/'device.json'),engine='fast')
            core=min(os.sched_getaffinity(0));folder=root/'build'
            first=repair.graph_process(request,folder,affinity=core,timeout=30)
            cost=read(folder/'context_cost.json')
            self.assertGreater(cost['measured_graph_build_seconds'],0)
            self.assertEqual(first['node_count'],10)
            self.assertEqual(repair.graph_process(request,folder,affinity=core,timeout=30),first)
            self.assertEqual(cost,read(folder/'context_cost.json'))
            logic=Path(first['logic']);original=logic.read_bytes()
            logic.write_bytes(original+b' ')
            with self.assertRaisesRegex(ValueError,'input changed'):
                repair.graph_process(request,folder,affinity=core,timeout=30)
            logic.write_bytes(original)
            (folder/'context_cost.json').unlink()
            with self.assertRaisesRegex(RuntimeError,'durable cost'):
                repair.graph_process(request,folder,affinity=core,timeout=30)

    @unittest.skipUnless(hasattr(os,'sched_setaffinity'),'Linux affinity scheduler')
    def test_cold_matrix_failures_and_completed_resume(self):
        with self.temporary() as tmp:
            f=old_handoff.QualifiedHandoffTests().fixture(Path(tmp))
            instance=repair.RepairCampaign.__new__(repair.RepairCampaign)
            instance.out=f.out;instance.conditions=lambda:f.conditions
            checkpoint=f.root/'checkpoint.pt';checkpoint.write_bytes(b'checkpoint')
            instance.checkpoints=lambda:{p:str(checkpoint) for p in ('frozen_gnn','clipped','guarded')}
            original=os.sched_getaffinity(0)
            instance.budget=dict(physical_cores=sorted(original)[:4])
            instance.args=SimpleNamespace(timeout=10)
            instance.binary=f.binary;instance.binary_reference=f.source/'binary_identity.json'
            instance.guard=lambda:None;instance.rr=lambda *args:{}
            instance.architecture=lambda r,a:f.source/'architectures/16x16'/a/'arch.xml'
            calls=Counter();lock=threading.Lock()
            def prep(folder,command,**kw):
                self.assertEqual(os.sched_getaffinity(0),{kw['affinity']})
                with lock:
                    calls['prep']+=1;fail=calls['prep']==1
                return dict(status='infrastructure_failure' if fail else 'complete')
            def execute(c,s,folder,*,method='default',checkpoint=None,affinity=None):
                self.assertEqual(os.sched_getaffinity(0),{affinity})
                if method!='default':
                    self.assertEqual(c['record']['structure_export_seconds'],0.)
                    self.assertEqual(c['_affinity'],affinity)
                    self.assertIn('cold_graph',c['graph_root'])
                with lock:calls['execute']+=1
                return dict(condition_id=c['id'],seed=s,method=method,status='success',cpd_ns=1.,wirelength=1.,
                    vpr_manifest='mock',**{k:c[k] for k in ('design','family','architecture','width','width_role')})
            instance.execute=execute
            with patch.object(repair,'run_trial',side_effect=prep),patch.object(repair,'emit'):
                rows=instance.compare()
            self.assertEqual(len(rows),200)
            self.assertEqual(calls,Counter(prep=160,execute=199))
            self.assertEqual(os.sched_getaffinity(0),original)
            self.assertEqual(len({(r['condition_id'],r['seed'],r['method']) for r in rows}),200)
            for method in repair.METHODS:
                self.assertEqual(Counter(r['order'].index(method) for r in rows if r['method']==method),Counter({i:8 for i in range(5)}))
            with patch.object(repair,'run_trial',side_effect=AssertionError('rerun')),patch.object(instance,'execute',side_effect=AssertionError('rerun')):
                self.assertEqual(instance.compare(),rows)
            with patch.object(repair.os,'uname',return_value=('changed',)):
                with self.assertRaisesRegex(ValueError,'frozen input changed'):
                    instance.compare()

    def test_cleanup_retains_replay_inputs_and_rejects_escape(self):
        with self.temporary() as tmp:
            owner=Path(tmp)
            root=owner/'comparison/wave_0000/attempt_0001'
            payload=root/'lane_000/cold_graph/payload'
            payload.mkdir(parents=True)
            for name in ('graph.pt','logic.json'):
                (payload/name).write_bytes(b'duplicate')
            action=root/'lane_000/execution/initializer/payload/target.json'
            write(action,dict(unique_action=True))
            teacher=owner/'graphs/teacher.json';write(teacher,dict(unique_teacher=True))
            repair.prune_temporary_graphs(root,owner)
            self.assertFalse((payload/'graph.pt').exists())
            self.assertFalse((payload/'logic.json').exists())
            self.assertTrue(action.exists());self.assertTrue(teacher.exists())
            self.assertEqual(read(root/'payload_pruned.json')['removed_bytes'],18)
            repair.prune_temporary_graphs(root,owner)
            with self.assertRaisesRegex(ValueError,'cleanup must'):
                repair.prune_temporary_graphs(owner/'graphs',owner)


if __name__=='__main__':
    unittest.main()
