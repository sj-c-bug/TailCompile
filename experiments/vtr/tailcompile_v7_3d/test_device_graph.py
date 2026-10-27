import copy
import importlib.util
import math
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

from architectures import LEGACY, audit_family, vertical_locations
from device_graph import build_device_graph

sys.path.insert(0, str(LEGACY))
spec = importlib.util.spec_from_file_location('_legacy_supply_fixture', LEGACY/'test_device_supply_3d.py')
fixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixture_module)


class GraphTests(unittest.TestCase):
    def test_geometry_pair_has_equal_budget_but_different_concentration(self):
        spread = vertical_locations(16, 20, False)
        center = vertical_locations(16, 20, True)
        self.assertEqual(len(spread), len(center))
        self.assertNotEqual(set(spread), set(center))
        rms = lambda coords: sum((x-7.5)**2+(y-9.5)**2 for x, y in coords)/len(coords)
        self.assertGreater(rms(spread), rms(center))

    def test_partial_leaf_fixed_size_shared_wire_capacity_and_root_io(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            folder = Path(directory)
            arch, root, nodes, edges = fixture_module.fixture(folder, width=10, height=6)
            tree = ET.parse(arch)
            cbl = ET.SubElement(tree.getroot(), 'complexblocklist')
            for typ in ('clb', 'memory', 'dsp_top', 'io'):
                pb = ET.SubElement(cbl, 'pb_type', name=typ)
                if typ == 'clb':
                    ET.SubElement(pb, 'pb_type', name='lut', num_pb='4', blif_model='.names')
                    ET.SubElement(pb, 'pb_type', name='ff', num_pb='8', blif_model='.latch')
                else:
                    pb.set('blif_model', '.input' if typ == 'io' else '.subckt hard')
                sub = tree.getroot().find(f"./tiles/tile[@name='{typ}']/sub_tile")
                ET.SubElement(ET.SubElement(sub, 'equivalent_sites'), 'site', pb_type=typ)
            tree.write(arch)
            fixture_module.wire(nodes, 0, 'CHANX', 'INC_DIR', x1=9)
            fixture_module.wire(nodes, 1, 'CHANX', 'INC_DIR', x1=9, z0=1, z1=1)
            fixture_module.wire(nodes, 2, 'CHANZ', 'BI_DIR', x1=0, z1=1, capacity=3)
            fixture_module.connect(edges, 0, 2, repeats=40)
            fixture_module.connect(edges, 2, 1, repeats=40)
            path = folder/'rr.xml'
            ET.ElementTree(root).write(path)
            graph = build_device_graph(path, arch, tile_size=4)
            self.assertEqual(len(graph['regions']), 12)  # 3 x 2 x 2 leaves
            self.assertEqual(graph['regions'][2]['bounds'], [8, 10, 0, 4])
            self.assertEqual(graph['region_bounds'][2], [.8, 1., 0., 4/6])
            self.assertEqual(graph['audit']['unique_connected_z_capacity'], 3)
            vertical = next(p for p in graph['pools'] if p['kind'] == 'CHANZ')
            self.assertEqual(vertical['capacity'], 3)  # BI direction and 80 edges do not multiply it.
            self.assertEqual(vertical['arcs'], [[0, 6], [6, 0]])
            self.assertEqual(graph['layer_capacity'][0][4], 2)  # 1 IO root x capacity 2, not 4.
            self.assertEqual(graph['layer_capacity'][1][4], 2)
            self.assertEqual(len(graph['parents'][0]), 12)
            self.assertEqual(len(graph['parents'][1]), 4)
            self.assertTrue(all(len(x) == 8 for x in graph['pool_x']))
            self.assertTrue(all(len(x) == len(graph['region_feature_names']) for x in graph['region_x']))

    def test_actual_audit_rejects_missing_or_mismatched_budget_and_reports_width_coupling(self):
        base = dict(grid=[12, 12, 2], total_resource_capacity={'clb': 120},
                    z_capacity_by_class={'BI_DIR|z': 40}, z_switch_classes=['z_mux'],
                    z_actual_adjacent_switch_classes=['L4_driver', 'z_mux'])
        graphs = {f'H{h}Z{z}': [dict(channel_width=64, audit=copy.deepcopy(base)),
                                dict(channel_width=128, audit=copy.deepcopy(base))]
                  for h in (0, 1) for z in (0, 1)}
        for condition, values in graphs.items():
            for graph in values:
                graph['audit'].update(z_capacity_by_xy={'1,1' if condition.endswith('0') else '4,4': 40},
                    z_mean_squared_distance_to_center=10 if condition.endswith('0') else 3)
        value = audit_family(graphs)
        self.assertTrue(value['four_cell_matches'])
        self.assertTrue(value['width_coupling']['H0Z0']['z_capacity_invariant'])
        graphs['H1Z1'][0]['audit']['z_capacity_by_class']['BI_DIR|z'] += 1
        with self.assertRaisesRegex(ValueError, 'do not match'):
            audit_family(graphs)


if __name__ == '__main__':
    unittest.main()
