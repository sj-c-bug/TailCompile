import copy
import json
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

import torch

from vertical_risk import prepare_device,weighted_cvar,risk_loss,hard_risk,route_pool_summary,SCHEMA
from layer_features import augment_graph


def summary(capacities=(1.,1.),both=True):
    bounds=[[0.,.5,0.,1.],[.5,1.,0.,1.]]*2
    pools=[]
    for i,cap in enumerate(capacities):
        orientations=[dict(source_layer=0,sink_layer=1,source_regions=[0,1],sink_regions=[2,3])]
        if both:
            orientations.append(dict(source_layer=1,sink_layer=0,source_regions=[2,3],sink_regions=[0,1]))
        pools.append(dict(id=i,capacity=cap,xy=[.25+.5*i,.5],orientations=orientations))
    return dict(schema=SCHEMA,grid=[8,4,2],tile_size=4,pools=pools,
        region_bounds=bounds,region_layer=[0,0,1,1],access_semantics='fixture directed RR upper bound',
        demand_semantics='shared net/pool noisy-OR')


def data(pins=(0,1),capacities=(1.,1.),both=True):
    return dict(vertical_risk_device=summary(capacities,both),
                risk_nets=[dict(id=0,pins=list(pins),drivers=[0],sinks=list(pins[1:]),ignored=False)])


class RiskTests(unittest.TestCase):
    def test_weighted_cvar_uses_partial_tail_mass(self):
        values=torch.tensor([1.,4.],requires_grad=True)
        actual=weighted_cvar(values,torch.tensor([9.,1.]),.8)
        self.assertAlmostEqual(float(actual),2.5,places=5)
        actual.backward()
        torch.testing.assert_close(values.grad,torch.tensor([.5,.5]))

    def test_crossing_probability_and_gradient_are_hand_computable(self):
        graph=data(capacities=(2.,))
        p=torch.tensor([[.8,.2],[.2,.8]],requires_grad=True)
        xy=torch.tensor([[[.25,.5],[.25,.5]],[[.25,.5],[.25,.5]]])
        loss,detail=risk_loss(graph,p,xy)
        # 0->1: .8*.8; 1->0: .2*.2; capacity=2.
        self.assertAlmostEqual(float(loss),.34,places=5)
        self.assertAlmostEqual(detail['expected_vertical_usage'],.68,places=5)
        loss.backward()
        self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(float(p.grad.abs().sum()),0.)

    def test_multisink_shared_pool_not_sink_count(self):
        graph=data(pins=(0,1,2,3),capacities=(1.,))
        result=hard_risk(graph,torch.tensor([0,1,1,1]),torch.tensor([[.25,.5]]*4))
        self.assertAlmostEqual(result['expected_vertical_usage'],1.,places=5)
        self.assertAlmostEqual(result['cvar95'],1.,places=5)
        json.dumps(result,allow_nan=False)

    def test_two_target_regions_share_one_wire_per_net_pool(self):
        graph=data(pins=(0,1,2),capacities=(1.,))
        xy=torch.tensor([[.25,.5],[.25,.5],[.75,.5]])
        result=hard_risk(graph,torch.tensor([0,1,1]),xy)
        self.assertEqual(result['branch_count'],4)
        self.assertAlmostEqual(result['expected_vertical_usage'],1.,places=5)

    def test_highfanout_retained_and_clocks_explicitly_ignored(self):
        graph=data(pins=tuple(range(150)),capacities=(1.,))
        graph['risk_nets'].append(dict(id=1,pins=list(range(150)),ignored=True,ignore_reason='clock'))
        result=hard_risk(graph,torch.tensor([0]+[1]*149),torch.tensor([[.25,.5]]*150))
        self.assertEqual(result['ordinary_high_fanout_nets_included'],1)
        self.assertEqual(result['ignored_net_count'],1)
        self.assertEqual(result['active_net_count'],1)
        self.assertAlmostEqual(result['expected_vertical_usage'],1.,places=5)

    def test_zero_pool_is_unassigned_not_epsilon_pressure(self):
        graph=data(capacities=())
        loss,detail=risk_loss(graph,torch.tensor([[1.,0.],[0.,1.]]),torch.tensor([[[.25,.5]]*2]*2))
        self.assertEqual(float(loss),0.)
        self.assertEqual(detail['pool_pressure'],[])
        self.assertAlmostEqual(detail['unassigned_expected_crossings'],1.,places=5)

    def test_reverse_direction_is_not_invented(self):
        graph=data(capacities=(1.,),both=False)
        result=hard_risk(graph,torch.tensor([1,0]),torch.tensor([[.25,.5]]*2))
        self.assertEqual(result['expected_vertical_usage'],0.)
        self.assertAlmostEqual(result['unassigned_expected_crossings'],1.,places=5)

    def test_soft_and_hard_pressure_differ_and_cache_tracks_regions(self):
        graph=data(capacities=(1.,1.))
        p=torch.tensor([[.5,.5],[.5,.5]])
        xy=torch.tensor([[[.25,.5]]*2]*2)
        loss,detail=risk_loss(graph,p,xy)
        initial=graph['_vertical_branch_cache']['fingerprint']
        hard=hard_risk(graph,torch.tensor([0,1]),torch.tensor([[.75,.5]]*2))
        self.assertNotEqual(initial,graph['_vertical_branch_cache']['fingerprint'])
        self.assertGreater(hard['cvar95'],detail['cvar95'])


class DeviceTests(unittest.TestCase):
    def fixture(self,folder):
        root=ET.Element('rr_graph')
        segs=ET.SubElement(root,'segments')
        ET.SubElement(segs,'segment',id='0',name='vertical')
        nodes=ET.SubElement(root,'rr_nodes')
        def node(rid,kind,layer,capacity=1):
            item=ET.SubElement(nodes,'node',id=str(rid),type=kind,direction='INC_DIR',capacity=str(capacity))
            ET.SubElement(item,'loc',xlow='0',xhigh='0',ylow='0',yhigh='0',
                layer_low=str(layer),layer_high=str(1 if kind=='CHANZ' else layer))
            ET.SubElement(item,'segment',segment_id='0')
        node(0,'OPIN',0);node(1,'CHANZ',0,3);node(2,'IPIN',1);node(3,'CHANZ',0,4)
        edges=ET.SubElement(root,'rr_edges')
        for _ in range(20):
            ET.SubElement(edges,'edge',src_node='0',sink_node='1')
            ET.SubElement(edges,'edge',src_node='1',sink_node='2')
        path=folder/'rr.xml'
        ET.ElementTree(root).write(path)
        device=dict(grid_width=8,grid_height=4,layers=2,tile_size=4,
            region_bounds=[[0,.5,0,1],[.5,1,0,1]]*2,region_layer=[0,0,1,1],
            audit=dict(unique_connected_z_capacity=3))
        return path,device

    def test_full_member_capacity_and_actual_direction(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path,device=self.fixture(Path(tmp))
            result=prepare_device(path,device)
        self.assertEqual(result['audit']['connected_capacity'],3)
        self.assertEqual(result['audit']['excluded_member_ids'],[3])
        self.assertEqual(result['pools'][0]['members'],[dict(rr_node_id=1,capacity=3.,xy=[0.,0.])])
        self.assertEqual(result['pools'][0]['orientations'],[
            dict(source_layer=0,sink_layer=1,source_regions=[0],sink_regions=[2])])
        self.assertEqual(result['audit']['full_rr_edges_retained'],0)

    def test_route_member_join_keeps_hidden_track_overuse_and_missing(self):
        summary=dict(pools=[dict(id=0,capacity=2.,members=[dict(rr_node_id=1,capacity=1.),
            dict(rr_node_id=2,capacity=1.)])],audit=dict(excluded_member_ids=[3]))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path=Path(tmp)/'usage.tsv'
            path.write_text('# iteration=1 rr_id\tcapacity\toccupancy\n1\t1\t2\n2\t1\t0\n3\t4\t0\n')
            result=route_pool_summary(summary,path)
            self.assertTrue(result['complete'])
            self.assertEqual(result['pools'][0]['aggregated_overuse'],0.)
            self.assertEqual(result['member_overuse'],1.)
            self.assertEqual(result['excluded_observed_rr_ids'],[3])
            self.assertEqual(result['unmatched_observed_rr_ids'],[])
            path.write_text('1\t1\t1\n')
            result=route_pool_summary(summary,path)
            self.assertFalse(result['complete'])
            self.assertEqual(result['missing_member_ids'],[2])
            self.assertIsNone(result['utilization_cvar95'])

    def test_serial_z_is_rejected_instead_of_double_counted(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path,device=self.fixture(Path(tmp))
            tree=ET.parse(path)
            ET.SubElement(tree.getroot().find('rr_edges'),'edge',src_node='1',sink_node='3')
            tree.write(path)
            with self.assertRaisesRegex(ValueError,'serial CHANZ'):
                prepare_device(path,device)

    def test_coarse_pool_uses_capacity_centroid_not_first_member(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path,device=self.fixture(Path(tmp))
            tree=ET.parse(path)
            item=ET.SubElement(tree.getroot().find('rr_nodes'),'node',id='4',type='CHANZ',
                capacity='1',direction='INC_DIR')
            ET.SubElement(item,'loc',xlow='3',xhigh='3',ylow='0',yhigh='0',layer_low='0',layer_high='1')
            ET.SubElement(item,'segment',segment_id='0')
            edges=tree.getroot().find('rr_edges')
            ET.SubElement(edges,'edge',src_node='0',sink_node='4')
            ET.SubElement(edges,'edge',src_node='4',sink_node='2')
            tree.write(path);device['audit']['unique_connected_z_capacity']=4
            result=prepare_device(path,device)
        self.assertEqual(len(result['pools']),1)
        self.assertEqual(len(result['pools'][0]['members']),2)
        self.assertAlmostEqual(result['pools'][0]['xy'][0],3/8/4)

    def test_zero_capacity_horizontal_node_cannot_supply_access(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            path,device=self.fixture(Path(tmp))
            tree=ET.parse(path)
            tree.getroot().find('rr_nodes/node').set('capacity','0')
            tree.write(path)
            with self.assertRaisesRegex(ValueError,'capacity disagrees'):
                prepare_device(path,device)

    def test_feature_widths_teacher_separation_and_proxy(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            folder=Path(tmp)
            blif=folder/'tiny.blif'
            blif.write_text('.model top\n.inputs a\n.outputs z\n.names a n\n1 1\n.names n z\n1 1\n.end\n')
            logic=dict(grid=[8,4,2],clusters=[dict(id=0,ap_block_id=0,atoms=['n']),
                dict(id=1,ap_block_id=1,atoms=['z'])],nets=[[0,1]])
            graph=dict(cluster_x=torch.ones(2,4),region_x=torch.ones(4,5),
                logic_src=torch.tensor([0,1]),logic_dst=torch.tensor([1,0]))
            result=augment_graph(graph,logic,{},summary(),blif)
        self.assertEqual(result['cluster_x'].shape,(2,4))
        self.assertEqual(result['region_x'].shape,(4,5))
        self.assertEqual(result['risk_nets'][0]['drivers'],[0])
        self.assertEqual(result['risk_nets'][0]['sinks'],[1])
        self.assertTrue(result['risk_nets'][0]['proxy_known'])
        self.assertFalse(result['layer_feature_audit']['movable_teacher_coordinates_used'])
        self.assertNotIn('risk_nets',graph)


if __name__=='__main__':
    unittest.main()
