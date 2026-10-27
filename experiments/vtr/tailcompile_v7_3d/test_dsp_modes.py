"""Mode occupancy comes from mapped pin widths, never model-name heuristics."""
from pathlib import Path
import tempfile
import unittest
import xml.etree.ElementTree as ET

from data import build_logic, dsp_mode_catalog, dsp_blif_instances, DEMAND_SEMANTICS
from device_graph import resource_units


def fixture(folder):
    root = ET.Element('architecture')
    tile = ET.SubElement(ET.SubElement(root, 'tiles'), 'tile', name='dsp_top')
    sub = ET.SubElement(tile, 'sub_tile', name='dsp_top', capacity='1')
    ET.SubElement(ET.SubElement(sub, 'equivalent_sites'), 'site', pb_type='dsp_top')
    top = ET.SubElement(ET.SubElement(root, 'complexblocklist'), 'pb_type', name='dsp_top')
    wrapper = ET.SubElement(top, 'pb_type', name='dsp', num_pb='1')
    for name, count, widths in (('wide', 1, (27, 27, 54)), ('dual', 2, (18, 19, 37))):
        mode = ET.SubElement(wrapper, 'mode', name=name)
        bank = ET.SubElement(mode, 'pb_type', name=name, num_pb=str(count))
        leaf = ET.SubElement(bank, 'pb_type', name='multiply', num_pb='1', blif_model='.subckt multiply')
        for kind, port, width in zip(('input', 'input', 'output'), ('a', 'b', 'out'), widths):
            ET.SubElement(leaf, kind, name=port, num_pins=str(width))
    arch = folder/'arch.xml'
    ET.ElementTree(root).write(arch)
    return arch


def instance(widths, output='p', model='multiply'):
    terms = [f'{port}[{i}]={output}{i}' if port == 'out' else f'{port}[{i}]=same_signal'
             for port, width in zip(('a', 'b', 'out'), widths) for i in range(width)]
    return '.subckt '+model+' '+' '.join(terms)


def blif(folder, lines):
    path = folder/'mapped.blif'
    path.write_text('.model top\n'+ '\n'.join(lines)+'\n.end\n')
    return path


def snapshot(folder, rows):
    prefix = folder/'ap'
    Path(str(prefix)+'.meta.tsv').write_text('10\t10\t2\n')
    Path(str(prefix)+'.nets.tsv').write_text('')
    Path(str(prefix)+'.blocks.tsv').write_text(''.join(
        f'{block}\t{atom}\t3\t0\t-1\t-1\t0\t0\t{model}\t{n}\n'
        for n, (block, atom, model) in enumerate(rows)))
    return prefix


class DspModeTests(unittest.TestCase):
    def test_wide_single_and_dual_small_keep_two_units_per_physical_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            arch = fixture(folder)
            catalog = dsp_mode_catalog(arch)
            self.assertEqual(catalog['units_per_root'], 2)
            self.assertEqual(resource_units(arch)['dsp_top']['dsp_top'], 2)
            mapped = blif(folder, [instance((27, 27, 54), 'wide'),
                                   instance((18, 19, 37), 'small')])
            rows = dsp_blif_instances(mapped, catalog)
            self.assertEqual(rows['wide0']['demand'], 2)
            self.assertEqual(rows['small0']['demand'], 1)
            # All 27 positions are real width evidence, even repeated nets.
            self.assertEqual(rows['wide0']['port_widths']['a'], 27)
            graph = build_logic(snapshot(folder, [(0, 'wide0', 'multiply'),
                                      (1, 'small0', 'multiply')]), 'case', mapped, arch)
            self.assertEqual([row['demand'][3] for row in graph['clusters']], [2, 1])
            self.assertEqual(graph['demand_semantics'], DEMAND_SEMANTICS)

    def test_two_small_instances_in_one_molecule_sum_to_full_root(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            arch = fixture(folder)
            mapped = blif(folder, [instance((18, 19, 37), 'p'), instance((18, 19, 37), 'q')])
            graph = build_logic(snapshot(folder, [(0, 'p0', 'multiply'), (0, 'q0', 'multiply')]),
                                'case', mapped, arch)
            self.assertEqual(graph['clusters'][0]['demand'][3], 2)
            self.assertEqual(len(graph['clusters'][0]['dsp_mode_audit']), 2)

    def test_oversize_unknown_model_and_missing_atom_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            arch = fixture(folder)
            catalog = dsp_mode_catalog(arch)
            with self.assertRaisesRegex(ValueError, 'no compatible XML mode'):
                dsp_blif_instances(blif(folder, [instance((28, 27, 55))]), catalog)
            with self.assertRaisesRegex(ValueError, 'no exact mapped mode evidence'):
                build_logic(snapshot(folder, [(0, 'p0', 'dsp_unknown')]), 'case',
                            blif(folder, [instance((18, 19, 37), model='dsp_unknown')]), arch)
            with self.assertRaisesRegex(ValueError, 'no exact mapped mode evidence'):
                build_logic(snapshot(folder, [(0, 'missing', 'multiply')]), 'case',
                            blif(folder, [instance((18, 19, 37))]), arch)

    def test_sparse_pins_unknown_ports_and_ambiguous_alias_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            catalog = dsp_mode_catalog(fixture(folder))
            with self.assertRaisesRegex(ValueError, 'sparse'):
                dsp_blif_instances(blif(folder, [instance((18, 19, 37)).replace('a[1]=same_signal ', '')]), catalog)
            with self.assertRaisesRegex(ValueError, 'no compatible XML mode'):
                dsp_blif_instances(blif(folder, [instance((18, 19, 37))+' mystery=net']), catalog)
            with self.assertRaisesRegex(ValueError, 'ambiguous'):
                dsp_blif_instances(blif(folder, [instance((18, 19, 37)), instance((18, 19, 37))]), catalog)

    def test_explicit_cname_resolves_same_instance(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            catalog = dsp_mode_catalog(fixture(folder))
            rows = dsp_blif_instances(blif(folder, [instance((27, 27, 54)), '.cname explicit_atom']), catalog)
            self.assertIs(rows['explicit_atom'], rows['p0'])
            self.assertEqual(rows['explicit_atom']['demand'], 2)


if __name__ == '__main__':
    unittest.main()
