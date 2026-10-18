import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import four_routes as f
from refine import choose


class FourRoutesTests(unittest.TestCase):
    def test_backend_initialization_command_before_binary_exists(self):
        b=object.__new__(f.direct.DirectBackend)
        b.distro=None
        self.assertEqual(b.command(['git','--version']),['git','--version'])

    def test_commands_and_isolated_environment(self):
        backend=SimpleNamespace(binary='/vpr',arch='/arch')
        c=dict(blif='/input.blif',width=100,device_width=14,fplace='/gnn.fplace')
        with patch.dict(os.environ,{'TC_GP_REPLACE':'stale','TC_GP_WORKER':'stale'}):
            for method in f.METHODS:
                cmd=f.command(backend,c,251,method,Path('/trial'))
                env=f.environment(Path('/trial'),method,100)
                self.assertEqual('--read_flat_place' in cmd,method in ('B_gnn','D_gnn_refine'))
                self.assertEqual('TC_GP_WORKER' in env,method in f.METHODS[2:])
                self.assertEqual(cmd[cmd.index('--num_workers')+1],'1')
            self.assertEqual(os.environ['TC_GP_WORKER'],'stale')

    def test_cost_never_double_counts_worker(self):
        timing={k:1 for keys in f.SCOPES.values() for k in keys}
        for method in f.METHODS:
            row=dict(method=method,wall_seconds=100,local_refinement={'worker_seconds':90})
            extra=f.overhead(timing,'warm_inference_lowering') if method in ('B_gnn','D_gnn_refine') else 0
            self.assertEqual(f.total(row,timing,'warm_inference_lowering'),100+extra)

    def test_policy_replay_on_tie_and_no_labels(self):
        pool=[('replay',None,dict(tail=2,hpwl=10)),('swap',None,dict(tail=2,hpwl=10))]
        self.assertEqual(choose(pool),0)
        pool[1][2]['tail']=1
        self.assertEqual(choose(pool),1)

    def test_four_routes_report_requires_all_trials(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as temp:
            root=Path(temp);cases=[];seeds=[251,252,253]
            timing={k:1 for keys in f.SCOPES.values() for k in keys}
            for width in (80,90,100):
                g=root/f'g{width}.json';f.write(g,dict(timing=timing))
                c=dict(design='test',width=width,generation=str(g));cases.append(c)
                for phase in ('runs','timing'):
                    for seed in seeds:
                        for method in f.METHODS:
                            f.write(root/phase/'test'/f'w{width}'/f's{seed}'/method/'trial.json',
                                dict(width=width,method=method,status='success',wall_seconds=100,
                                     wirelength=10,cpd=1))
            f.write(root/'plan.json',dict(cases=cases,seeds=seeds))
            a=SimpleNamespace(out=root)
            f.report(a)
            r=f.read(root/'evaluation_report.json')
            self.assertEqual(len(r['width_results']),4)
            self.assertEqual(len(r['runtime_comparisons']),5)
            self.assertTrue(all(x['joint_success_pairs']==9 for x in r['runtime_comparisons']))
            (root/'runs/test/w80/s251/D_gnn_refine/trial.json').unlink()
            with self.assertRaises(FileNotFoundError): f.report(a)

    def test_tasks_are_four_per_condition(self):
        a=SimpleNamespace(out=Path('/tmp/test'))
        plan=dict(cases=[dict(design='test',width=100)],seeds=[251,252,253])
        tasks=list(f.tasks(a,plan,False))
        self.assertEqual(len(tasks),12)
        self.assertEqual(len({str(t[3]) for t in tasks}),12)


if __name__=='__main__':unittest.main()
