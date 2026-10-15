import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from tailcompile_v4_risk import (RiskEvaluator, build_supply, cvar,
                                 matched_scales_for, objective, search)
from run_tailcompile_v4 import classify_vpr_result, run_one


def fixture():
    device = {'grid_width': 6, 'grid_height': 3, 'regions': [
        {'id': i, 'center_x': x, 'center_y': 1, 'x_low': x, 'x_high': x + 1,
         'y_low': 1, 'y_high': 1, 'tile_coordinates': {'clb': [[x, 1], [x + 1, 1]]}}
        for i, x in enumerate((1, 3))]}
    logic = {'clusters': [{'id': i, 'resource': 'clb', 'atoms': [str(i)]} for i in range(4)]}
    supply = {'pools': [{'id': 0, 'capacity': 1, 'arcs': [[0, 1], [1, 0]]}]}
    edges = [SimpleNamespace(name='a', pins=(0, 1), drivers=(0,), sinks=(1,), criticality=1.0),
             SimpleNamespace(name='b', pins=(2, 3), drivers=(2,), sinks=(3,), criticality=0.5)]
    return logic, device, supply, edges


class RiskTest(unittest.TestCase):
    def test_fractional_cvar(self):
        self.assertAlmostEqual(cvar([1, 2, 3], .5), 8 / 3)
        self.assertEqual(cvar([1, 2, 3], 1), 2)

    def test_confirmatory_mean_and_tail_start_with_matched_magnitude(self):
        state = {'cvar': 4.0, 'mean': 1.0, 'wire': 10.0, 'timing': 5.0}
        scales = matched_scales_for(state)
        shared = 0.15 + 0.25
        self.assertAlmostEqual(objective(state, scales, 'tail') - shared, 1.0)
        self.assertAlmostEqual(objective(state, scales, 'mean') - shared, 1.0)
        self.assertAlmostEqual(objective(state, scales, 'none'), shared)

    def test_server_vtr_code_one_unroutable_is_not_other_failure(self):
        message = ('Routing failed for sink 1\n'
                   'Circuit is unroutable with a channel width factor of 20.\n'
                   'Error 1: in get_num_bends_and_length: net #0 has no routing.')
        self.assertEqual(classify_vpr_result(1, message), 'unroutable')
        self.assertEqual(classify_vpr_result(2, message), 'unroutable')
        self.assertEqual(classify_vpr_result(1, 'Error 1: unrelated failure'),
                         'other_failure')

    def test_unknown_full_legalizer_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unsupported AP full legalizer'):
            run_one(None, Path('unused-run'), Path('unused.blif'),
                    Path('unused.net'), 20, 10, 1,
                    full_legalizer='not-a-legalizer')

    def test_incremental_matches_full_and_capacity(self):
        args = fixture()
        e = RiskEvaluator(*args)
        initial = [0, 1, 0, 1]
        old = e.state(initial)
        new = [0, 0, 1, 1]
        incremental = e.state(new, old, [1, 2])
        self.assertEqual(incremental['loads'], e.state(new)['loads'])
        self.assertEqual(incremental['wire'], 0)
        self.assertFalse(e.legal([0, 0, 0, 1]))

    def test_search_legal_reproducible_and_improves(self):
        e = RiskEvaluator(*fixture())
        a, report = search(e, [0, 1, 0, 1], budget=32)
        b, again = search(e, [0, 1, 0, 1], budget=32)
        self.assertEqual(a, b)
        self.assertEqual(report['trace'], again['trace'])
        self.assertTrue(e.legal(a))
        self.assertLess(report['final']['cvar'], report['initial']['cvar'])
        for row in report['trace']:
            self.assertLess(row['objective_after'], row['objective_before'])
        self.assertEqual(report['evaluations'], 32)

    def test_directed_unreachable_and_allowed_regions(self):
        logic, device, supply, edges = fixture()
        supply['pools'][0]['arcs'] = [[0, 1]]
        e = RiskEvaluator(logic, device, supply, edges)
        self.assertIsNone(e.state([1, 0, 1, 0]))
        logic['clusters'][0]['allowed_regions'] = [0]
        self.assertFalse(e.legal([1, 0, 1, 0]))

    def test_shared_long_wire_once_per_multisink_net(self):
        logic, device, _, _ = fixture()
        device['regions'].append(dict(device['regions'][1], id=2, center_x=5,
                                      x_low=5, x_high=6, tile_coordinates={'clb': [[5, 1], [6, 1]]}))
        supply = {'pools': [{'id': 0, 'capacity': 1, 'arcs': [[0, 1], [1, 2]]}]}
        edge = SimpleNamespace(name='fanout', pins=(0, 1, 2), drivers=(0,), sinks=(1, 2), criticality=1)
        e = RiskEvaluator(logic, device, supply, [edge])
        self.assertEqual(e.state([0, 1, 2, 2])['loads'], [1.0])

    def test_raw_rrg_counts_nodes_not_switches(self):
        _, device, _, _ = fixture()
        xml = '''<rr_graph><channels><channel chan_width_max="10"/></channels>
        <segments><segment id="0" name="L4"/></segments>
        <grid><grid_loc x="5" y="2" layer="0"/></grid>
        <rr_nodes><node id="7" type="CHANX" direction="INC_DIR" capacity="1">
        <loc xlow="1" xhigh="4" ylow="1" yhigh="1" layer="0"/><segment segment_id="0"/>
        </node></rr_nodes><rr_edges><edge src_node="7" sink_node="8"/>
        <edge src_node="7" sink_node="9"/></rr_edges></rr_graph>'''
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'rr.xml'
            path.write_text(xml)
            result = build_supply(path, device)
        self.assertEqual(len(result['pools']), 1)
        self.assertEqual(result['pools'][0]['capacity'], 1)
        self.assertEqual(result['pools'][0]['arcs'], [[0, 1]])

    def test_explicit_macro_rejected(self):
        args = list(fixture())
        args[0] = copy.deepcopy(args[0])
        args[0]['macros'] = [{'members': [0, 1]}]
        with self.assertRaisesRegex(ValueError, 'macros'):
            RiskEvaluator(*args)

    def test_runner_resume_verifies_inputs_and_preserves_packed_net(self):
        from run_tailcompile_v4 import run_one
        backend = SimpleNamespace(binary_hash='binary', arch_hash='arch', binary='vpr', arch='arch.xml',
                                  distro=None, command=lambda command, directory: command)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blif, packed = root / 'a.blif', root / 'a.net'
            blif.write_text('blif')
            packed.write_text('original packed')
            def simulate(command, **kwargs):
                Path(command[command.index('--net_file') + 1]).write_text('repacked by AP')
                kwargs['stdout'].write('VPR succeeded\n')
                return SimpleNamespace(returncode=0)
            with patch('run_tailcompile_v4.subprocess.run', side_effect=simulate) as call:
                result = run_one(backend, root / 'run', blif, packed, 90, 14, 11)
                self.assertEqual(result['vpr_status'], 'success')
                run_one(backend, root / 'run', blif, packed, 90, 14, 11)
                self.assertEqual(call.call_count, 1)
                self.assertEqual(packed.read_text(), 'original packed')
                with self.assertRaisesRegex(ValueError, 'inputs changed'):
                    run_one(backend, root / 'run', blif, packed, 100, 14, 11)

    def test_failed_or_interrupted_run_is_not_skipped(self):
        from run_tailcompile_v4 import run_one
        backend = SimpleNamespace(binary_hash='binary', arch_hash='arch', binary='vpr', arch='arch.xml',
                                  distro=None, command=lambda command, directory: command)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blif, packed = root / 'a.blif', root / 'a.net'
            blif.write_text('blif')
            packed.write_text('packed')
            with patch('run_tailcompile_v4.subprocess.run', return_value=SimpleNamespace(returncode=127)):
                with self.assertRaises(RuntimeError):
                    run_one(backend, root / 'run', blif, packed, 90, 14, 11)
                with self.assertRaisesRegex(ValueError, 'previous run failed'):
                    run_one(backend, root / 'run', blif, packed, 90, 14, 11)
            (root / 'partial').mkdir()
            (root / 'partial/vpr.out').write_text('interrupted')
            with self.assertRaisesRegex(ValueError, 'interrupted'):
                run_one(backend, root / 'partial', blif, packed, 90, 14, 11)

    def test_wsl_enforces_timeout_inside_linux(self):
        from run_tailcompile_v4 import run_one
        backend = SimpleNamespace(binary_hash='binary', arch_hash='arch', binary='vpr', arch='arch.xml',
                                  distro='test', command=lambda command, directory: command)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            blif, packed = root / 'a.blif', root / 'a.net'
            blif.write_text('blif')
            packed.write_text('packed')
            def simulate(command, **kwargs):
                self.assertEqual(command[:4], ['timeout', '--signal=TERM', '--kill-after=15s', '3'])
                self.assertEqual(kwargs['timeout'], 33)
                kwargs['stdout'].write('VPR succeeded\n')
                return SimpleNamespace(returncode=0)
            with patch('run_tailcompile_v4.subprocess.run', side_effect=simulate):
                run_one(backend, root / 'run', blif, packed, 90, 14, 11, timeout=3)


if __name__ == '__main__':
    unittest.main()
