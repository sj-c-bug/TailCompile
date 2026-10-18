import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from core import candidates,positions,features,flat,write,read,retention
from campaign import frozen,flat_coordinates,parallel

def small_graph():
    return dict(size=[40,40],blocks=[dict(atoms=[f'a{i}'],models=['2'],fixed=i==0,
        loc=[float(2+i%8*3),float(2+i//8*3),0.,0]) for i in range(32)],
        nets=[[i,(i+7)%32,(i+13)%32] for i in range(32)])

class ContractTests(unittest.TestCase):
    def test_candidate_atomicity_occupancy_fixed_budget(self):
        g=small_graph();original=positions(g)
        actions=candidates(g,100)
        self.assertEqual(len(actions),13)
        for name,xy,meta in actions:
            self.assertEqual(sorted(map(tuple,xy)),sorted(map(tuple,original)))
            np.testing.assert_array_equal(xy[0],original[0])
            self.assertLessEqual(meta['moved'],max(2,int(31*meta.get('budget',0))))

    def test_flat_roundtrip_double_precision(self):
        g=small_graph();xy=positions(g);xy[1,0]=1.2345678901234567
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'a.fplace';flat(path,g,xy)
            self.assertEqual(flat_coordinates(path)['a1'][0],xy[1,0])

    def test_graph_features_and_linear_net_edges(self):
        g=small_graph();f=features(g,positions(g),100)
        self.assertEqual(f['x'].shape,(32,41))
        self.assertEqual(len(f['src']),2*sum(len(n)-1 for n in g['nets']))
        self.assertTrue(np.isfinite(f['x']).all())

    def test_immutability(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)/'test.json';frozen(p,{'a':1});frozen(p,{'a':1})
            with self.assertRaises(ValueError): frozen(p,{'a':2})

    def test_worker_failures_are_not_hidden(self):
        def fail(i):
            if i==1: raise ValueError('expected')
            return i
        with self.assertRaises(RuntimeError):parallel(fail,[0,1,2],2)

    def test_family_splits(self):
        cohort=read(Path(__file__).with_name('cohort.json'))['designs']
        self.assertEqual(len({r['family'] for r in cohort}),20)
        previous={'sha','attention_layer','spmv','bnn','conv_layer','eltwise_layer','lenet','lstm','robot_rl','bgm','or1200','stereovision0','tpu_like.small.os'}
        self.assertTrue(all(r['split']=='train' for r in cohort if r['design'] in previous))

    def test_retention_uses_atom_identity(self):
        g=small_graph();base=positions(g);changed=base.copy();changed[[1,2]]=changed[[2,1]]
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            a,b=Path(tmp)/'a',Path(tmp)/'b';flat(a,g,base);flat(b,g,changed)
            result=retention(a,b,g,changed)
            self.assertEqual(result['directional_retention'],1)
            self.assertEqual(result['displacement_error'],0)

    def test_torch_forward_backward_and_fallback(self):
        try: import torch
        except ImportError: self.skipTest('torch unavailable')
        from learn import Selector,choose
        device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        g=small_graph(); data={k:torch.as_tensor(v,device=device) for k,v in features(g,positions(g),100).items()}
        model=Selector().to(device);output=model(data);output.sum().backward()
        self.assertEqual(output.device,device)
        self.assertEqual(choose(torch.zeros(3,5)),0)

    def test_training_selection_and_report_plumbing_synthetic_only(self):
        try: import torch
        except ImportError: self.skipTest('torch unavailable')
        from learn import train,select
        from campaign import contexts,report,code_identity
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp)
            a=SimpleNamespace(out=root,device='cpu',epochs=1)
            designs=[dict(design=s,family=s,split=s,widths=[90]) for s in ['train','validation','test']]
            write(root/'cohort.json',dict(designs=designs))
            write(root/'protocol.json',dict(code=code_identity()))
            write(root/'plan.json',dict(designs=designs))
            write(root/'labels_complete.json',dict(complete=True))
            for d,w,seed,folder in contexts(a):
                folder.mkdir(parents=True)
                g=small_graph();xy=positions(g);moved=xy.copy();moved[[1,2]]=moved[[2,1]]
                write(folder/'graph.json',g)
                write(folder/'pool.json',dict(actions=[dict(name='replay'),dict(name='move')],generation_seconds=.1))
                for name,coords in [('default',xy),('replay',xy),('move',moved)]:
                    np.save(folder/(name+'.npy'),coords)
                    if d['split']=='test' and name!='default':continue
                    write(folder/name/'trial.json',dict(complete=True,status='success',wirelength=100 if name!='move' else 90,
                        cpd=10,wall_seconds=10))
            train(a);select(a)
            self.assertTrue(read(root/'model/training_report.json')['complete'])
            selection=read(root/'selection.json')
            self.assertEqual(len(selection['rows']),5)
            for d,w,seed,folder in contexts(a,{'test'}):
                for name in ['replay','move','timing/default','timing/selected']:
                    write(folder/name/'trial.json',dict(complete=True,status='success',wirelength=100,cpd=10,wall_seconds=10))
                write(folder/'replay/retention.json',dict(exact_post_route_replay=True,same_status=True))
            report(a)
            self.assertTrue(read(root/'evaluation_report.json')['complete'])

    def test_synchronous_cpu_worker_no_label_access(self):
        try: import torch
        except ImportError: self.skipTest('torch unavailable')
        from learn import Selector
        from worker import main as worker_main
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);g=small_graph();prefix=root/'gp'
            Path(str(prefix)+'.blocks.tsv').write_text(''.join(
                f"{i}\t{b['atoms'][0]}\t2\t{int(b['fixed'])}\t{b['loc'][0]}\t{b['loc'][1]}\t0\t0\n"
                for i,b in enumerate(g['blocks'])))
            Path(str(prefix)+'.nets.tsv').write_text(''.join(str(i)+'\t'+'\t'.join(map(str,n))+'\n' for i,n in enumerate(g['nets'])))
            Path(str(prefix)+'.meta.tsv').write_text('40\t40\n')
            model=Selector()
            for parameter in model.parameters():parameter.data.zero_()
            torch.save(dict(model=model.state_dict()),root/'test.pt')
            with patch('sys.argv',['worker.py','--prefix',str(prefix),'--model',str(root/'test.pt'),
                                   '--output',str(root/'out.fplace'),'--width','90']):
                worker_main()
            self.assertEqual(read(root/'out.selection.json')['action'],'default')
            self.assertTrue((root/'out.fplace').is_file())

if __name__=='__main__':unittest.main()
