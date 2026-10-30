"""Graph equivalence, constrained ranking and real short-training regressions."""
import copy
import itertools
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
import data
import fast_graph
import reward_v2 as reward
import train as old_train
from train_repair import train


def graph_fixture(n=300, regions=32):
    rng=random.Random(410)
    demands=([1,0,0,0,0],[0,1,0,0,0],[2,2,0,0,0],[0,0,1,0,0],[0,0,0,1,0],[0,0,0,0,1])
    kinds=('clb','clb','clb','memory','dsp_top','io')
    clusters=[]
    for i in range(n):
        k=i%len(demands)
        c=dict(id=i,ap_block_id=i,atoms=[f'a{i}'],resource=kinds[k],demand=list(demands[k]),fixed=False)
        if i==0:
            c.update(fixed=True,fixed_location=[1.,1.,1.,0])
        elif i==1:
            c.update(fixed=True,fixed_location=[-1.,1.,0.,0])
        clusters.append(c)
    nets=[rng.sample(range(n),min(n,fanout)) for fanout in (2,3,16,17,128,129)]
    nets += [[i,i+1] for i in range(n-1)]
    logic=dict(clusters=clusters,nets=nets,grid=[16,16,2],demand_semantics=data.DEMAND_SEMANTICS)
    cells=regions//2
    device=dict(grid_width=16,grid_height=16,layers=2,channel_width=80,
        capacity=[[10.,10.,2.,2.,2.] for i in range(regions)],
        region_layer=[0]*cells+[1]*cells,region_bounds=[[i/cells,(i+1)/cells,0.,1.] for i in range(cells)]*2,
        region_x=[[1.,2.,3.] for _ in range(regions)],pool_x=[],region_weight=[1.,1.],
        region_src=[0,1],region_dst=[1,0],pool_ids=[],pool_regions=[],parents=[[0]*cells+[1]*cells])
    return logic,device


def trial(seconds=100.,cpd=10.,wl=100.,status='success',seed=1):
    return dict(seed=seed,status=status,total_seconds=seconds,cpd_ns=cpd,wirelength=wl)


class RepairTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_exact_graph_schema_tensor_and_metadata(self):
        for regions in (4,32,128):
            logic,device=graph_fixture(regions=regions)
            a=data.make_graph(logic,device)
            profile={}
            b=fast_graph.make_graph(logic,device,profile=profile)
            self.assertEqual(fast_graph.semantic_digest(a),fast_graph.semantic_digest(b))
            self.assertEqual(a.keys(),b.keys())
            for key in a:
                if isinstance(a[key],torch.Tensor):
                    self.assertTrue(torch.equal(a[key],b[key]),key)
                    self.assertEqual(a[key].dtype,b[key].dtype)
            self.assertEqual(profile['unique_demands'],6)
            self.assertEqual(a['graph_audit']['skipped_global_nets'],1)

    def test_mask_partial_fixed_and_incompatible_match(self):
        logic,dev=graph_fixture()
        dev['capacity'][:16]=[[10,10,2,0,2]]*16
        graph=fast_graph.make_graph(logic,dev)
        self.assertEqual(graph['legal_layers'][4].tolist(),[False,True])
        self.assertEqual(graph['legal_layers'][0].tolist(),[False,True])
        self.assertEqual(graph['fixed_region'][1].item(),-1)
        dev['capacity'][16:]=[[10,10,2,0,2]]*16
        for function in (data.make_graph,fast_graph.make_graph):
            with self.assertRaisesRegex(ValueError,'no individually compatible layer'):
                function(logic,dev)

    def test_frozen_model_outputs_identical_initialization(self):
        from model import Hierarchical3DInitializer
        logic,device=graph_fixture(n=36,regions=32)
        old=data.make_graph(logic,device)
        new=fast_graph.make_graph(logic,device)
        torch.manual_seed(82)
        model=Hierarchical3DInitializer(old['cluster_x'].shape[1],old['region_x'].shape[1],hidden=8)
        a=model.predict(old,seed=7,score_noise_std=.2,coordinate_noise_std=.03)
        b=model.predict(new,seed=7,score_noise_std=.2,coordinate_noise_std=.03)
        for key in ('assignment','layer','xy'):
            self.assertTrue(torch.equal(a[key],b[key]),key)

    def test_float32_tolerance_at_capacity_edge(self):
        logic,dev=graph_fixture()
        logic['clusters'][2]['demand']=[10.0000005,0,0,0,0]
        self.assertEqual(fast_graph.semantic_digest(data.make_graph(logic,dev)),
                         fast_graph.semantic_digest(fast_graph.make_graph(logic,dev)))

    def test_saturated_negative_candidates_still_ordered(self):
        base=[trial()]
        candidates=[[trial(seconds=t)] for t in (150,200,400)]
        self.assertEqual([old_train.outcome_reward(c,base)['advantage'] for c in candidates],[-1.]*3)
        actions=[reward.outcome_reward(c,base) for c in candidates]
        self.assertEqual(reward.pairs(actions),[(0,1),(0,2),(1,2)])
        self.assertTrue(all(r['advantage']<0 for r in actions))
        self.assertEqual(len({r['raw_log_runtime_advantage'] for r in actions}),3)

    def test_guard_dominates_speed_and_violation_remains_negative(self):
        safe=reward.outcome_reward([trial(seconds=10000)],[trial()])
        unsafe=reward.outcome_reward([trial(seconds=1,cpd=10.4)],[trial()])
        worse=reward.outcome_reward([trial(seconds=.1,cpd=12)],[trial()])
        self.assertEqual(reward.preference(safe,unsafe),1)
        self.assertEqual(reward.preference(unsafe,worse),1)
        self.assertFalse(unsafe['quality_protection'])
        self.assertEqual(unsafe['advantage'],-1.)
        lost=reward.outcome_reward([trial(status='unroutable')],[trial()])
        self.assertEqual(reward.preference(worse,lost),1)
        self.assertFalse(lost['success_protection'])

    def test_missing_infra_duplicate_failed_seeds_and_runtime_deadband(self):
        self.assertFalse(reward.outcome_reward([trial(status='unroutable')],[trial(status='unroutable')])['eligible'])
        self.assertFalse(reward.outcome_reward([trial(status='infrastructure_failure')],[trial()])['eligible'])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            reward.outcome_reward([trial(),trial()],[trial()])
        missing=trial();missing['total_seconds']=None
        self.assertFalse(reward.outcome_reward([missing],[trial()])['eligible'])
        a=reward.outcome_reward([trial(seconds=150)],[trial()])
        b=reward.outcome_reward([trial(seconds=151)],[trial()])
        self.assertEqual(reward.preference(a,b),0)
        self.assertFalse(a['failure_runtime_imputed'])

    def test_short_training_both_arms_same_initial_weights_only_train(self):
        from test_model import fixture
        from model import Hierarchical3DInitializer,SCHEMA,capacity_only
        graph=fixture()
        config=dict(cluster_width=3,region_width=5,hidden=8,chunk_size=256,pool_width=8,demand_width=5)
        torch.manual_seed(10)
        model=Hierarchical3DInitializer(**config)
        target=capacity_only(graph)
        teacher=dict(teacher_xy=target['xy'].tolist(),teacher_region=target['assignment'].tolist(),teacher_layer=target['layer'].tolist())
        a=dict(assignment=target['assignment'].tolist(),xy=target['xy'].tolist())
        b=copy.deepcopy(a);b['xy']=[[min(.99,x+.05),y] for x,y in b['xy']]
        feedback=[dict(condition_id='train0',action=action,trials=[trial(seconds=t)],default_trials=[trial()],candidate_id=i)
                  for i,(action,t) in enumerate(((a,150),(b,300)))]
        records=[dict(graph=graph,teacher=teacher,split='train',family='f',condition_id='train0')]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);checkpoint=root/'source.pt'
            torch.save(dict(schema=SCHEMA,config=config,state_dict=model.state_dict()),checkpoint)
            reports=[]
            for policy in ('clipped','guarded'):
                report=train(records,root/policy,epochs=3,hidden=8,seed=601,checkpoint=checkpoint,feedback=feedback,policy=policy)
                self.assertTrue(report['weights_changed'])
                self.assertFalse(report['validation_gradients_used'])
                self.assertEqual([Path(p).name for p in report['checkpoints']],['epoch_0001.pt','epoch_0003.pt'])
                reports.append(report)
            self.assertEqual(reports[0]['initial_state_sha256'],reports[1]['initial_state_sha256'])
            self.assertEqual(reports[0]['history'][0]['feedback'][0]['preference_pairs'],0)
            self.assertGreater(reports[1]['history'][0]['feedback'][0]['preference_pairs'],0)
            self.assertNotEqual(reports[0]['final_state_sha256'],reports[1]['final_state_sha256'])
            with self.assertRaisesRegex(ValueError,'only explicit train'):
                train([dict(records[0],split='validation')],root/'invalid',epochs=1)
            invalid=copy.deepcopy(feedback)
            invalid[1]['default_trials'][0]['total_seconds']=99.
            with self.assertRaisesRegex(ValueError,'different Default controls'):
                train(records,root/'mixed',epochs=1,hidden=8,checkpoint=checkpoint,feedback=invalid)

    def test_report_keeps_all_failure_denominators(self):
        from repair_learning import METHODS,COMPARE_SEEDS,summarize
        conditions=[dict(id='d/a/w1',split='validation',design='d',width_role='reference')]
        rows=[dict(condition_id='d/a/w1',seed=s,method=m,status='success',total_seconds=10,cpd_ns=1,wirelength=1)
              for s,m in itertools.product(COMPARE_SEEDS,METHODS)]
        rows[-1]['status']='infrastructure_failure'
        summary=summarize(rows,conditions)
        guarded=[r for r in summary['contrasts'] if r['method']=='guarded' and r['split']=='validation']
        self.assertTrue(summary['complete'])
        self.assertTrue(all(not r['development_runtime_target'] for r in guarded))
        self.assertFalse(summary['full_acceptance_passed'])
        with self.assertRaisesRegex(ValueError,'duplicate'):
            summarize(rows+rows[:1],conditions)


if __name__=='__main__':
    unittest.main()
