import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
import sys
import os
if os.name=='nt':os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
sys.path.insert(0,str(Path(__file__).resolve().parent))
from direct import SCOPES,overhead,freeze,report,write,read

class TimingTests(unittest.TestCase):
    def test_rrg_excluded_from_warm_but_not_cold(self):
        t={k:1. for keys in SCOPES.values() for k in keys}
        baseline=overhead(t,'warm_inference_lowering')
        t['rrg_encoding']=10000.
        self.assertEqual(overhead(t,'warm_inference_lowering'),baseline)
        self.assertGreater(overhead(t,'cold_from_prepared_ir'),10000.)
        self.assertIn('capacity_decode',SCOPES['warm_inference_lowering'])

    def test_frozen_protocol(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            p=Path(tmp)/'a.json';freeze(p,{'a':1})
            with self.assertRaises(ValueError):freeze(p,{'a':2})

    def test_report_success_pairs_only_and_censoring(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root=Path(tmp);cases=[]
            t={k:1. for keys in SCOPES.values() for k in keys}
            for width in [80,90,100]:
                generation=root/f'g{width}.json';write(generation,dict(timing=t))
                cases.append(dict(design='test',width=width,generation=str(generation)))
                for seed in [251,252,253]:
                    for method in ['default','gnn_only']:
                        status='success' if width >= (100 if method=='default' else 90) else 'unroutable'
                        row=dict(width=width,status=status,wall_seconds=20 if method=='default' else 10,wirelength=100,cpd=10)
                        for phase in ['runs','timing']:
                            write(root/phase/'test'/f'w{width}'/f's{seed}'/method/'trial.json',row)
            write(root/'plan.json',dict(cases=cases,seeds=[251,252,253]))
            report(SimpleNamespace(out=root))
            r=read(root/'evaluation_report.json')
            self.assertEqual([x['w80'] for x in r['width_results']],[100,90])
            self.assertEqual(r['runtime_comparisons'][0]['joint_success_pairs'],3)
            self.assertIsNone(r['paired_rows'][0]['full_cold_from_raw_rrg'])
            self.assertAlmostEqual(r['runtime_comparisons'][0]['speedup']['warm_inference_lowering'],20/(10+overhead(t,'warm_inference_lowering')))

    def test_split_tower_matches_original_forward(self):
        import torch
        from tailcompile_v3 import TwoTowerPlacementGNN
        torch.manual_seed(67)
        model=TwoTowerPlacementGNN(8,13,8).eval()
        d=dict(cluster_x=torch.randn(6,8),region_x=torch.randn(4,13),logic_src=torch.tensor([0,1]),
            logic_dst=torch.tensor([1,2]),logic_weight=torch.ones(2),region_src=torch.tensor([0,1]),
            region_dst=torch.tensor([1,2]),region_weight=torch.ones(2))
        with torch.no_grad():
            expected,xy=model(d)
            logic=model.logic_tower(d['cluster_x'],d['logic_src'],d['logic_dst'],d['logic_weight'])
            region=model.region_projection(model.rrg_tower(d['region_x'],d['region_src'],d['region_dst'],d['region_weight']))
            actual=model.logic_projection(logic)@region.T/(logic.shape[1]**.5)
            torch.testing.assert_close(actual,expected)
            torch.testing.assert_close(model.coordinate_head(logic),xy)

if __name__=='__main__':unittest.main()
