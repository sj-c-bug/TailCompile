import copy
import gzip
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

import factorial_timing as timing


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture(folder, *, up=2e-10, down=4e-10, wire_r=0., wire_c=0., layers=(0, 1),
            dynamic_missing=False, z_nodes=True):
    architecture = ET.Element('architecture')
    tiles = ET.SubElement(architecture, 'tiles')
    tile = ET.SubElement(tiles, 'tile', name='clb')
    sub = ET.SubElement(tile, 'sub_tile', name='clb')
    fc = ET.SubElement(sub, 'fc', in_type='frac', in_val='0.15', out_type='frac', out_val='0.10')
    ET.SubElement(fc, 'fc_override', segment_name='z_segment', fc_type='abs', fc_val='0')
    device = ET.SubElement(architecture, 'device')
    ET.SubElement(device, 'connection_block', input_switch_name='ipin_cblock')
    switchlist = ET.SubElement(architecture, 'switchlist')
    for name, delay in (('L4_driver', '1e-10'), ('L16_driver', '2e-10'), ('ipin_cblock', '5e-11')):
        ET.SubElement(switchlist, 'switch', name=name, type='mux', Tdel=delay, R='0', Cin='0', Cout='0')
    for name in ('z_up', 'z_down'):
        switch = ET.SubElement(switchlist, 'switch', name=name, type='mux', R='0', Cin='0', Cout='0')
        ET.SubElement(switch, 'Tdel', num_inputs='8', delay='9e-8')  # deliberately not the resolved scalar
        ET.SubElement(switch, 'Tdel', num_inputs='16', delay='8e-8')
    segmentlist = ET.SubElement(architecture, 'segmentlist')
    for name, length, frequency, mux in (('L4', '4', '280', 'L4_driver'),
                                         ('L16', '16', '40', 'L16_driver'),
                                         ('z_segment', '1', '0', 'z_up')):
        segment = ET.SubElement(segmentlist, 'segment', name=name, length=length, freq=frequency, Rmetal='0', Cmetal='0')
        ET.SubElement(segment, 'mux', name=mux)
    sg = ET.SubElement(architecture, 'scatter_gather_list')
    links = ET.SubElement(ET.SubElement(sg, 'sg_pattern', type='bidir'), 'sg_link_list')
    ET.SubElement(links, 'sg_link', name='L_UP', z_offset='1', mux='z_up', seg_type='z_segment')
    ET.SubElement(links, 'sg_link', name='L_DOWN', z_offset='-1', mux='z_down', seg_type='z_segment')
    arch = folder / 'arch.xml'
    ET.ElementTree(architecture).write(arch)
    rr = ET.Element('rr_graph')
    switches = ET.SubElement(rr, 'switches')
    for sid, (name, delay) in enumerate((('L4_driver', 1e-10), ('L16_driver', 2e-10),
                                       ('ipin_cblock', 5e-11), ('z_up', up), ('z_down', down))):
        switch = ET.SubElement(switches, 'switch', id=str(sid), name=name, type='mux')
        values = {} if dynamic_missing and name == 'z_up' else {'Tdel': str(delay)}
        ET.SubElement(switch, 'timing', **values)  # actual VPR zero omission
    segments = ET.SubElement(rr, 'segments')
    for sid, name in enumerate(('L4', 'L16', 'z_segment')):
        ET.SubElement(segments, 'segment', id=str(sid), name=name)
    nodes = ET.SubElement(rr, 'rr_nodes')
    nid = 0
    for layer in layers:
        for kind in ('CHANX', 'CHANY'):
            for sid, length in enumerate((4, 16)):
                for _ in range(2):
                    node = ET.SubElement(nodes, 'node', id=str(nid), type=kind, direction='INC_DIR')
                    ET.SubElement(node, 'loc', layer_low=str(layer), layer_high=str(layer), xlow='0', ylow='0',
                                  xhigh=str(length-1 if kind == 'CHANX' else 0),
                                  yhigh=str(length-1 if kind == 'CHANY' else 0))
                    ET.SubElement(node, 'timing', R=str(wire_r), C=str(wire_c))
                    ET.SubElement(node, 'segment', segment_id=str(sid))
                    nid += 1
    if z_nodes:
        node = ET.SubElement(nodes, 'node', id=str(nid), type='CHANZ', direction='BI_DIR')
        ET.SubElement(node, 'loc', layer_low='0', layer_high='1', xlow='1', xhigh='1', ylow='1', yhigh='1')
        ET.SubElement(node, 'timing', R='0', C='0')
        ET.SubElement(node, 'segment', segment_id='2')
    # An invalid edge table detects accidental traversal past rr_nodes.
    edges = ET.SubElement(rr, 'rr_edges')
    ET.SubElement(edges, 'edge', src_node='invalid', sink_node='invalid', switch_id='invalid')
    rrg = folder / 'rr.xml'
    ET.ElementTree(rr).write(rrg)
    graph = dict(layers=2, channel_width=224,
        provenance=dict(vtr_commit=timing.PINNED_VTR_COMMIT, rr_graph_sha256=_sha(rrg), architecture_sha256=_sha(arch)),
        audit=dict(z_actual_adjacent_switch_classes=['L4_driver', 'z_up', 'z_down']))
    graph_path = folder / 'device_graph.json'
    graph_path.write_text(json.dumps(graph), encoding='utf-8')
    manifest = dict(rr_graph_sha256=_sha(rrg), architecture_sha256=_sha(arch), graph_sha256=_sha(graph_path), channel_width=224)
    manifest_path = folder / 'manifest.json'
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    return rrg, arch, graph_path, manifest_path


class TimingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)

    def build(self, **fixture_args):
        paths = fixture(self.folder, **fixture_args)
        return timing.build_timing_model(*paths[:3], device_manifest=paths[3]), paths

    def test_actual_resolved_scalar_units_and_direction(self):
        model, _ = self.build()
        # L4=.1ns / 4, L16=.2ns / 16, architecture frequency 280:40.
        expected = (280 * .1/4 + 40 * .2/16) / 320
        for value in model['horizontal_delay_per_tile_ns']:
            self.assertAlmostEqual(value, expected)
        self.assertAlmostEqual(model['vertical_delay_ns'][0][1], .2)
        self.assertAlmostEqual(model['vertical_delay_ns'][1][0], .4)
        self.assertAlmostEqual(model['terminal_delay_ns'], .05)
        self.assertEqual(model['sampling']['rr_edges_visited'], 0)
        self.assertTrue(model['provenance']['graph_file_pin_verified'])
        self.assertIs(timing.validate_timing_model(model), model)

    def test_actual_node_wire_rc_changes_coefficients_with_correct_units(self):
        model, _ = self.build(wire_r=100., wire_c=2e-12)
        # Distributed wire: .5 * 100ohm * 2pF = .1ns per segment.
        expected = (280 * .2/4 + 40 * .3/16) / 320
        self.assertAlmostEqual(model['horizontal_delay_per_tile_ns'][0], expected)

    def test_actual_layer_rc_difference_is_retained(self):
        paths = fixture(self.folder)
        tree = ET.parse(paths[0])
        for node in tree.getroot().findall('./rr_nodes/node'):
            if node.get('type') != 'CHANZ' and node.find('loc').get('layer_low') == '1':
                node.find('timing').set('R', '100')
                node.find('timing').set('C', '2e-12')
        tree.write(paths[0])
        graph = json.loads(paths[2].read_text())
        graph['provenance']['rr_graph_sha256'] = _sha(paths[0])
        model = timing.build_timing_model(paths[0], paths[1], graph)
        self.assertGreater(model['horizontal_delay_per_tile_ns'][1], model['horizontal_delay_per_tile_ns'][0])
        self.assertTrue(all(s['actual_stratum_sample_count'] > 0 for layer in model['layers'] for s in layer['strata']))

    def test_resolved_rr_switch_change_changes_model(self):
        first, _ = self.build(up=2e-10)
        second, _ = self.build(up=7e-10)
        self.assertAlmostEqual(second['vertical_delay_ns'][0][1] - first['vertical_delay_ns'][0][1], .5)
        self.assertEqual(first['vertical_delay_ns'][1][0], second['vertical_delay_ns'][1][0])
        self.assertNotEqual(first['provenance']['signature_sha256'], second['provenance']['signature_sha256'])

    def test_fanin_table_without_resolved_scalar_rejected(self):
        with self.assertRaisesRegex(ValueError, 'resolved scalar'):
            self.build(dynamic_missing=True)

    def test_cache_exact_reuse_and_tamper_rejection(self):
        paths = fixture(self.folder)
        out = self.folder / 'timing_model.json'
        first = timing.build_timing_model(*paths[:3], device_manifest=paths[3], out=out)
        second = timing.build_timing_model(*paths[:3], device_manifest=paths[3], out=out)
        self.assertEqual(first, second)
        tampered = copy.deepcopy(first)
        tampered['horizontal_delay_per_tile_ns'][0] *= 2
        out.write_text(json.dumps(tampered), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'content digest'):
            timing.build_timing_model(*paths[:3], device_manifest=paths[3], out=out)

    def test_cached_model_cannot_hide_missing_or_modified_input(self):
        paths = fixture(self.folder)
        out = self.folder / 'model.json'
        timing.build_timing_model(*paths[:3], device_manifest=paths[3], out=out)
        paths[0].write_bytes(paths[0].read_bytes() + b'\n')
        with self.assertRaisesRegex(ValueError, 'frozen device provenance mismatch'):
            timing.build_timing_model(*paths[:3], device_manifest=paths[3], out=out)
        paths[0].unlink()
        with self.assertRaisesRegex(FileNotFoundError, 'existing input artifact'):
            timing.build_timing_model(*paths[:3], device_manifest=paths[3], out=out)

    def test_manifest_graph_and_backend_pins(self):
        paths = fixture(self.folder)
        graph = json.loads(paths[2].read_text())
        graph['audit']['extra'] = 'changed'
        paths[2].write_text(json.dumps(graph), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'graph_sha256'):
            timing.build_timing_model(*paths[:3], device_manifest=paths[3])
        graph['provenance']['vtr_commit'] = 'different-backend'
        with self.assertRaisesRegex(ValueError, 'pinned 40f8'):
            timing.build_timing_model(paths[0], paths[1], graph)

    def test_missing_axis_or_layer_is_disclosed_mirror_not_fake_sample(self):
        model, _ = self.build(layers=(0,))
        for stratum in model['layers'][1]['strata']:
            self.assertEqual(stratum['actual_stratum_sample_count'], 0)
            self.assertEqual(stratum['mirrored_from'][0], 0)

    def test_zero_declared_z_metal_allows_unsampled_vertical_wire(self):
        model, _ = self.build(z_nodes=False)
        self.assertFalse(model['sampled_vertical_nodes'])
        self.assertIn('explicit zero', model['vertical_wire_rc_source'])
        self.assertGreater(model['vertical_delay_ns'][0][1], 0)

    def test_early_prefix_cap_exits_before_late_node_and_edge_tables(self):
        paths = fixture(self.folder, z_nodes=False)
        with patch.object(timing, 'EARLY_SAMPLE_NODES', 4), patch.object(timing, 'SAMPLES_PER_STRATUM', 2):
            model = timing.build_timing_model(*paths[:3])
        self.assertEqual(model['sampling']['nodes_visited'], 4)
        self.assertIn('explicit_shared_segment_mirroring', model['sampling']['stop_reason'])
        self.assertTrue(any(s['mirrored_from'] for layer in model['layers'] for s in layer['strata']))

    def test_cb_and_missing_node_rc_rejected(self):
        paths = fixture(self.folder)
        root = ET.parse(paths[1])
        root.getroot().find('.//fc_override').set('fc_val', '1')
        root.write(paths[1])
        graph = json.loads(paths[2].read_text())
        graph['provenance']['architecture_sha256'] = _sha(paths[1])
        with self.assertRaisesRegex(ValueError, 'CB/direct Z'):
            timing.build_timing_model(paths[0], paths[1], graph)
        paths = fixture(self.folder)
        root = ET.parse(paths[0])
        root.getroot().find('.//node/timing').attrib.pop('R')
        root.write(paths[0])
        graph = json.loads(paths[2].read_text())
        graph['provenance']['rr_graph_sha256'] = _sha(paths[0])
        with self.assertRaisesRegex(ValueError, 'lacks actual R/C'):
            timing.build_timing_model(paths[0], paths[1], graph)

    def test_node_byte_and_time_caps(self):
        paths = fixture(self.folder)
        with patch.object(timing, 'MAX_SAMPLE_NODES', 1):
            with self.assertRaisesRegex(ValueError, 'node cap'):
                timing.build_timing_model(*paths[:3])
        with patch.object(timing, 'MAX_SAMPLE_BYTES', 100):
            with self.assertRaisesRegex(ValueError, 'byte cap'):
                timing.build_timing_model(*paths[:3])
        with self.assertRaises(TimeoutError):
            timing.build_timing_model(*paths[:3], max_seconds=1e-12)
        for value in (0, -1, float('inf'), True):
            with self.assertRaises(ValueError):
                timing.build_timing_model(*paths[:3], max_seconds=value)

    def test_gzip_hashes_compressed_artifact_and_bounded_sample(self):
        paths = fixture(self.folder)
        zipped = self.folder / 'rr.xml.gz'
        with gzip.open(zipped, 'wb') as stream:
            stream.write(paths[0].read_bytes())
        graph = json.loads(paths[2].read_text())
        graph['provenance']['rr_graph_sha256'] = _sha(zipped)
        model = timing.build_timing_model(zipped, paths[1], graph)
        self.assertEqual(model['provenance']['rr_graph_sha256'], _sha(zipped))
        self.assertEqual(model['sampling']['rr_edges_visited'], 0)


if __name__ == '__main__':
    unittest.main()
