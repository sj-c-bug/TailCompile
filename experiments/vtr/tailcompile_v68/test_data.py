"""Fast data-contract regression tests; no server experiments or GPU required."""
import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from data import (load_config, validate_splits, inventory, read_ap_snapshot,
                  build_ap_logic, parse_teacher, stream_device_ir,
                  architecture_resource_units, teacher_record)


def snapshot(prefix, rows, nets='0\t9\t2\n', size='10\t10\n'):
    Path(str(prefix) + '.blocks.tsv').write_text('\n'.join(rows) + '\n')
    Path(str(prefix) + '.nets.tsv').write_text(nets)
    Path(str(prefix) + '.meta.tsv').write_text(size)


def row(block, atom, model='.names', fixed=0, x=-1, y=-1):
    return f'{block}\t{atom}\t2\t{fixed}\t{x}\t{y}\t0\t0\t{model}\t{block}'


def device():
    return dict(grid_width=10, grid_height=10, regions=[
        dict(id=0, x_low=1, x_high=4, y_low=1, y_high=8),
        dict(id=1, x_low=5, x_high=8, y_low=1, y_high=8)])


class DataTests(unittest.TestCase):
    def test_graph_contract_preserves_io_demand_and_fine_bounds(self):
        # The project already uses this Windows-only test-runtime workaround.
        os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
        try:
            import torch
        except ImportError:
            self.skipTest('torch unavailable')
        from data import prepare_graph
        from prepare_tailcompile_v61 import supply_features
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blif = root / 'toy.blif'
            blif.write_text('.model toy\n.inputs in\n.outputs out\n.names in a\n1 1\n.latch a b\n.names b out\n1 1\n.end\n')
            prefix = root / 'structure'
            snapshot(prefix, [row(9, 'a'), row(9, 'b', model='.latch'),
                row(2, 'in', model='.input'), row(3, 'out'), row(4, 'out:out', model='.output')])
            logic = build_ap_logic(prefix, 'toy', blif)
            regions = []
            for r in range(2):
                for c in range(2):
                    regions.append(dict(id=len(regions), row=r, column=c,
                        x_low=1+c*4, x_high=4+c*4, y_low=1+r*4, y_high=4+r*4,
                        center_x=2.5+c*4, center_y=2.5+r*4, channel_internal=0.,
                        escape_left=0., escape_right=0., escape_up=0., escape_down=0.,
                        segment_supply={}, tile_counts=dict(clb=1, io=1), tile_coordinates={}))
            d = dict(grid_width=10, grid_height=10, tile_size=4, regions=regions,
                resource_units=dict(clb=dict(lut=40, ff=40), io=dict(io=8)))
            supply = dict(channel_width=80, pools=[dict(segment='L4', capacity=80,
                kind='CHANX', direction='INC_DIR', arcs=[[0, 1]])])
            supply_features(d, supply)
            graph = prepare_graph(logic, d, supply, blif)
            self.assertEqual(graph['demand'].shape, (4, 5))
            self.assertEqual(graph['capacity'].shape, (4, 5))
            self.assertEqual(int(graph['movable'].sum()), 4)
            self.assertEqual(float(graph['demand'][:, 4].sum()), 2)
            self.assertEqual(graph['region_bounds'][0, 0], 0)
            self.assertEqual(graph['region_bounds'][3, 1], 1)
            self.assertEqual(graph['channel_width'], 80)
            self.assertTrue(torch.isfinite(graph['cluster_x']).all())
            self.assertEqual(graph['logic_criticality'].shape, graph['logic_src'].shape)

    def test_cohort_preserves_untouched_family_splits(self):
        config = load_config()
        result = validate_splits(config)
        self.assertEqual(result['families'], 27)
        self.assertEqual(result['split_designs'], dict(train=20, validation=3, final_test=4))
        self.assertEqual({r['design'] for r in config['designs'] if r['split'] == 'final_test'},
                         {'arm_core', 'spree', 'mkPktMerge', 'LU8PEEng'})

    def test_split_leakage_rejected(self):
        config = load_config()
        config['designs'].append(dict(design='LU32PEEng', family='LU', split='train', source='LU32PEEng.v'))
        with self.assertRaisesRegex(ValueError, 'leakage'):
            validate_splits(config)

    def test_development_and_duplicate_design_guards(self):
        config = load_config()
        config['designs'][0]['split'] = 'final_test'
        with self.assertRaisesRegex(ValueError, 'previous'):
            validate_splits(config)
        config = load_config()
        config['designs'].append(dict(config['designs'][0]))
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validate_splits(config)

    def test_inventory_does_not_claim_runnable_and_records_missing_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            report = inventory(load_config(), tmp)
            self.assertEqual(report['available_designs'], 0)
            self.assertTrue(all(r['runnable_status'] == 'not_yet_verified' for r in report['designs']))

    def test_ap_teacher_joins_by_atom_coverage_not_numeric_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'teacher'
            snapshot(prefix, [row(9, 'a', x=2, y=3), row(2, 'b', x=7, y=3)])
            logic = dict(clusters=[dict(id=0, atoms=['a'], fixed=False),
                                   dict(id=1, atoms=['b'], fixed=False)])
            result = parse_teacher(prefix, logic, device(), split='train')
            self.assertEqual(result['teacher_region'], [0, 1])
            self.assertEqual(result['movable'], [True, True])
            self.assertAlmostEqual(result['teacher_xy'][0][0], 2 / 10)

    def test_teacher_missing_atom_and_split_molecule_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'teacher'
            snapshot(prefix, [row(9, 'a', x=2, y=3), row(2, 'b', x=7, y=3)])
            with self.assertRaisesRegex(ValueError, 'coverage'):
                parse_teacher(prefix, dict(clusters=[dict(atoms=['a'], fixed=False)]), device())
            with self.assertRaisesRegex(ValueError, 'splits or merges'):
                parse_teacher(prefix, dict(clusters=[dict(atoms=['a', 'b'], fixed=False)]), device())

    def test_teacher_nonfinite_bounds_and_fixed_guards(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'teacher'
            snapshot(prefix, [row(9, 'a', x='nan', y=3), row(2, 'b', x=7, y=3)])
            with self.assertRaisesRegex(ValueError, 'nonfinite'):
                read_ap_snapshot(prefix)
            snapshot(prefix, [row(9, 'a', x=20, y=3), row(2, 'b', x=7, y=3)])
            logic = dict(clusters=[dict(atoms=['a'], fixed=False), dict(atoms=['b'], fixed=False)])
            with self.assertRaisesRegex(ValueError, 'outside'):
                parse_teacher(prefix, logic, device())
            snapshot(prefix, [row(9, 'a', fixed=1, x=2, y=3), row(2, 'b', x=7, y=3)])
            logic['clusters'][0].update(fixed=True, fixed_location=[1, 3, 0, 0])
            with self.assertRaisesRegex(ValueError, 'constrained'):
                parse_teacher(prefix, logic, device())

    def test_native_ap_last_half_tile_is_not_out_of_bounds(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'teacher'
            logic = dict(clusters=[dict(atoms=['a'], fixed=False), dict(atoms=['b'], fixed=False)])
            snapshot(prefix, [row(9, 'a', x=9.5, y=9.99), row(2, 'b', x=2, y=3)])
            result = parse_teacher(prefix, logic, device())
            self.assertEqual(result['teacher_xy'][0], [.95, .999])
            snapshot(prefix, [row(9, 'a', x=10, y=3), row(2, 'b', x=2, y=3)])
            with self.assertRaisesRegex(ValueError, 'outside'):
                parse_teacher(prefix, logic, device())

    def test_final_test_teachers_always_rejected(self):
        with self.assertRaisesRegex(ValueError, 'final-test'):
            parse_teacher('nonexistent', {}, {}, split='final_test')
        with self.assertRaisesRegex(ValueError, 'forbidden'):
            teacher_record(dict(split='final_test'), 'nonexistent', {})

    def test_structure_never_accepts_movable_teacher_coordinates(self):
        with tempfile.TemporaryDirectory() as tmp:
            prefix = Path(tmp) / 'structure'
            snapshot(prefix, [row(9, 'a', x=2, y=3), row(2, 'b')])
            with self.assertRaisesRegex(ValueError, 'leakage'):
                read_ap_snapshot(prefix, structure=True)

    def test_ap_logic_preserves_molecules_and_demand_units(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blif = root / 'toy.blif'
            blif.write_text('.model toy\n.inputs in\n.outputs out\n.names in a\n1 1\n.latch a b\n.names b out\n1 1\n.end\n')
            prefix = root / 'structure'
            snapshot(prefix, [row(9, 'a'), row(9, 'b', model='.latch'),
                              row(2, 'in', model='.input', fixed=1, x=0, y=2),
                              row(3, 'out'), row(4, 'out:out', model='.output')])
            logic = build_ap_logic(prefix, 'toy', blif)
            ap9 = next(c for c in logic['clusters'] if c['ap_block_id'] == 9)
            self.assertEqual(ap9['atoms'], ['a', 'b'])
            self.assertEqual(ap9['resource_demand']['lut'], 1)
            self.assertEqual(ap9['resource_demand']['ff'], 1)
            self.assertNotIn('fixed_location', ap9)
            self.assertFalse(logic['movable_coordinates_in_features'])

    def test_architecture_capacity_mode_max_and_io_subtiles(self):
        text = '''<architecture><tiles><tile name="io"><sub_tile capacity="8"><equivalent_sites><site pb_type="io"/></equivalent_sites></sub_tile></tile></tiles>
        <complexblocklist><pb_type name="clb"><mode name="a"><pb_type name="lut" num_pb="10" blif_model=".names"/></mode><mode name="b"><pb_type name="lut" num_pb="20" blif_model=".names"/><pb_type name="ff" num_pb="10" blif_model=".latch"/></mode></pb_type><pb_type name="io"><mode name="in"><pb_type name="inpad" blif_model=".input"/></mode><mode name="out"><pb_type name="outpad" blif_model=".output"/></mode></pb_type></complexblocklist></architecture>'''
        with tempfile.TemporaryDirectory() as tmp:
            arch = Path(tmp) / 'arch.xml'
            arch.write_text(text)
            units = architecture_resource_units(arch)
            self.assertEqual(units['clb']['lut'], 20)
            self.assertEqual(units['clb']['ff'], 10)
            self.assertEqual(units['io']['io'], 8)

    def test_streaming_grid_root_anchors_not_multiheight_offsets(self):
        text = '''<rr_graph><channels><channel chan_width_max="100"/></channels>
        <block_types><block_type id="0" name="EMPTY"/><block_type id="1" name="clb"/><block_type id="2" name="memory"/></block_types><grid>
        <grid_loc x="0" y="0" block_type_id="0"/><grid_loc x="9" y="9" block_type_id="0"/>
        <grid_loc x="2" y="2" block_type_id="1"/>
        <grid_loc x="5" y="2" block_type_id="2" height_offset="0"/>
        <grid_loc x="5" y="3" block_type_id="2" height_offset="1"/></grid>
        <rr_nodes><node id="1"><loc xlow="2" xhigh="2" ylow="2" yhigh="2"/></node></rr_nodes>
        <rr_edges><edge src_node="1" sink_node="1"/></rr_edges></rr_graph>'''
        with tempfile.TemporaryDirectory() as tmp:
            rr = Path(tmp) / 'rr.xml'
            rr.write_text(text)
            result = stream_device_ir(rr, 4)
            self.assertEqual(result['region_grid']['count'], 4)
            self.assertEqual(sum(r['tile_counts'].get('memory', 0) for r in result['regions']), 1)
            self.assertEqual(result['canonical_channel_width'], 100)

    def test_fixed_tile_fast_lookup_matches_nearest_for_ties_and_perimeter(self):
        from tailcompile_v4_risk import region_at
        regions = []
        for row_index, (yl, yh) in enumerate(((1, 4), (5, 7))):
            for col_index, (xl, xh) in enumerate(((1, 4), (5, 8), (9, 10))):
                regions.append(dict(id=len(regions), column=col_index, row=row_index,
                                    x_low=xl, x_high=xh, y_low=yl, y_high=yh))
        d = dict(grid_width=12, grid_height=9, tile_size=4,
                 region_grid=dict(columns=3, rows=2, count=6), regions=regions)
        def brute(x, y):
            return min(range(len(regions)), key=lambda i: (
                max(regions[i]['x_low'] - x, 0, x - regions[i]['x_high']) ** 2 +
                max(regions[i]['y_low'] - y, 0, y - regions[i]['y_high']) ** 2, i))
        for x in (0, 1, 3.99, 4, 4.49, 4.5, 4.51, 8.5, 10, 11):
            for y in (0, 1, 4, 4.49, 4.5, 4.51, 7, 8):
                self.assertEqual(region_at(d, x, y), brute(x, y), (x, y))


if __name__ == '__main__':
    unittest.main()
