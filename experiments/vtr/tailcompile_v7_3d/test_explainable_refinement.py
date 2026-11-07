"""016 independent geometry, constraints and finite-work refinement tests."""
import copy
import itertools
import math
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import torch

import explainable_refinement as refinement
from aligned_model import action_digest
from generation import lower_initialization
from residual_actions import validate_residual_action
from test_model import fixture


HERE = Path(__file__).resolve().parent


def case(n=6):
    """Feasible same-layer start; original net incidences are explicit."""
    data = fixture(n)
    data['capacity'] = torch.tensor([[float(n), float(n), 2., 2., 2.]] * 4)
    data['density_spread_fraction'] = 0.
    data['grid'] = [16, 16, 2]
    data['teacher_axis_scale'] = [16, 16]
    data['nets'] = [[i, i + 1] for i in range(n - 1)]
    data['risk_nets'] = [dict(id=i, pins=net, drivers=[net[0]], sinks=net[1:])
                         for i, net in enumerate(data['nets'])]
    leaves = torch.arange(n) % 2
    xy = torch.stack((torch.where(leaves == 0, .15, .85),
                      torch.linspace(.2, .8, n)), 1)
    base = dict(assignment=leaves, layer=torch.zeros(n, dtype=torch.long), xy=xy)
    return data, base


def portals(points=((.15, .15),), reverse_only=False):
    orientations = [dict(source_layer=1 if reverse_only else 0,
                         sink_layer=0 if reverse_only else 1,
                         source_regions=[2, 3] if reverse_only else [0, 1],
                         sink_regions=[0, 1] if reverse_only else [2, 3])]
    return dict(grid=[16, 16, 2], pools=[dict(id=0, capacity=float(len(points)),
        xy=[.99, .99], orientations=orientations,
        members=[dict(rr_node_id=i, capacity=1., xy=list(p))
                 for i, p in enumerate(points)])])


def set_nets(data, nets):
    data['nets'] = nets
    data['risk_nets'] = [dict(id=i, pins=net, drivers=[net[0]], sinks=net[1:])
                         for i, net in enumerate(nets)]


def hpwl_oracle(data, action):
    """Independent exact single-layer formula, without helper edge expansion."""
    xy = action['xy'].double() * torch.tensor(data['grid'][:2]).double()
    total = 0.
    for net in data['nets']:
        ids = sorted(set(net))
        if len(ids) > 1:
            points = xy[ids]
            total += float((points.max(0).values - points.min(0).values).sum())
    return total


def assert_unchanged(test, current, original):
    if isinstance(original, torch.Tensor):
        test.assertTrue(torch.equal(current, original))
    elif isinstance(original, dict):
        test.assertEqual(set(current), set(original))
        for key in original:
            assert_unchanged(test, current[key], original[key])
    elif isinstance(original, (list, tuple)):
        test.assertEqual(len(current), len(original))
        for value, old in zip(current, original):
            assert_unchanged(test, value, old)
    else:
        test.assertEqual(current, original)


class ExplainableRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_same_layer_cost_is_net_hpwl_in_tile_units(self):
        data, base = case(3)
        base['assignment'] = torch.tensor([0, 1, 0])
        base['xy'] = torch.tensor([[.1, .3], [.9, .8], [.4, .9]])
        set_nets(data, [[0, 1, 2]])
        expected = hpwl_oracle(data, base)
        result = refinement.objective(data, base)
        self.assertAlmostEqual(result['value'], expected, places=6)
        # A symmetric message expansion can change without changing physical nets.
        data['logic_src'] = torch.tensor([0, 1, 0, 2, 1, 2])
        data['logic_dst'] = torch.tensor([1, 0, 2, 0, 2, 1])
        data['logic_weight'] = torch.tensor([99.] * 6)
        data['logic_criticality'] = torch.ones(6)
        self.assertAlmostEqual(refinement.objective(data, base)['value'], expected, places=6)

    def test_multiterminal_and_duplicate_pins_do_not_multiply_wirelength(self):
        data, base = case(150)
        set_nets(data, [list(range(150))])
        expected = hpwl_oracle(data, base)
        self.assertAlmostEqual(refinement.objective(data, base)['value'], expected, places=6)
        set_nets(data, [list(range(150)) + [0, 1, 149]])
        self.assertAlmostEqual(refinement.objective(data, base)['value'], expected, places=6)

    def test_cross_layer_cost_uses_actual_member_not_pool_centroid(self):
        data, base = case(2)
        base['assignment'] = torch.tensor([0, 2])
        base['layer'] = torch.tensor([0, 1])
        base['xy'] = torch.tensor([[.1, .1], [.2, .2]])
        summary = portals(((.15, .15), (.9, .9)))
        expected = 16 * (abs(float(base['xy'][0, 0]) - .15) +
                         abs(float(base['xy'][0, 1]) - .15) +
                         abs(float(base['xy'][1, 0]) - .15) +
                         abs(float(base['xy'][1, 1]) - .15))
        result = refinement.objective(data, base, summary)
        self.assertEqual(result['unsupported_net_ids'], [])
        self.assertAlmostEqual(result['value'], expected, places=6)
        shifted = copy.deepcopy(summary)
        shifted['pools'][0]['xy'] = [0., 0.]
        self.assertEqual(refinement.objective(data, base, shifted)['value'], result['value'])

    def test_unknown_cross_layer_access_is_explicit_and_incident_nodes_frozen(self):
        data, base = case(3)
        base['assignment'] = torch.tensor([0, 2, 0])
        base['layer'] = torch.tensor([0, 1, 0])
        base['xy'] = torch.tensor([[.1, .1], [.2, .2], [.3, .3]])
        set_nets(data, [[0, 1], [0, 2]])
        obj = refinement.objective(data, base, dict(grid=[16, 16, 2], pools=[]))
        self.assertEqual(obj['unsupported_net_ids'], [0])
        result = refinement.refine(data, base, dict(grid=[16, 16, 2], pools=[]))
        for key in ('assignment', 'layer', 'xy'):
            self.assertTrue(torch.equal(result[key][:2], base[key][:2]))

    def test_known_driver_cannot_use_a_reverse_only_vertical_channel(self):
        data, base = case(2)
        base['assignment'] = torch.tensor([0, 2])
        base['layer'] = torch.tensor([0, 1])
        base['xy'] = torch.tensor([[.1, .1], [.2, .2]])
        summary = portals(reverse_only=True)
        result = refinement.objective(data, base, summary)
        self.assertEqual(result['unsupported_net_ids'], [0])
        action = refinement.refine(data, base, summary)
        self.assertEqual(action_digest(action), action_digest(base))

    def test_unknown_driver_requires_access_in_both_directions(self):
        data, base = case(2)
        base['assignment'] = torch.tensor([0, 2])
        base['layer'] = torch.tensor([0, 1])
        base['xy'] = torch.tensor([[.1, .1], [.2, .2]])
        data['risk_nets'][0]['drivers'] = []
        summary = portals()
        self.assertEqual(refinement.objective(data, base, summary)['unsupported_net_ids'], [0])
        summary['pools'][0]['orientations'].extend(portals(reverse_only=True)['pools'][0]['orientations'])
        self.assertEqual(refinement.objective(data, base, summary)['unsupported_net_ids'], [])

    def test_empty_connectivity_and_fixed_design_are_identity(self):
        data, base = case()
        set_nets(data, [])
        result = refinement.refine(data, base)
        self.assertEqual(action_digest(result), action_digest(base))
        data, base = case()
        data['movable'].fill_(False)
        result = refinement.refine(data, base)
        self.assertEqual(action_digest(result), action_digest(base))

    def test_capacity_fixed_axes_regions_and_immovable_are_preserved(self):
        data, base = case(12)
        data['movable'][0] = False
        data['fixed_xy'][1, 0] = base['xy'][1, 0]
        data['fixed_layer'][2] = 0
        data['fixed_region'][3] = int(base['assignment'][3])
        data['demand'][4] = torch.tensor([0., 0., 0., 1., 0.])
        data['capacity'][:, 3] = 0.
        data['capacity'][0, 3] = 2.
        old = copy.deepcopy(base)
        result = refinement.refine(data, base, portals(), dict(max_seconds=10.))
        self.assertTrue(validate_residual_action(data, result)['valid'])
        self.assertTrue(torch.equal(result['xy'][0], base['xy'][0]))
        self.assertEqual(int(result['assignment'][0]), int(base['assignment'][0]))
        self.assertEqual(float(result['xy'][1, 0]), float(base['xy'][1, 0]))
        self.assertEqual(int(result['layer'][2]), 0)
        self.assertEqual(int(result['assignment'][3]), int(base['assignment'][3]))
        self.assertEqual(int(result['assignment'][4]), 0)
        for key in old:
            self.assertTrue(torch.equal(old[key], base[key]))

    def test_invalid_base_cannot_be_silently_repaired(self):
        data, base = case()
        data['capacity'][:, 0] = 1.
        with self.assertRaises(ValueError):
            refinement.refine(data, base)
        data, base = case()
        base['xy'][0, 0] = float('nan')
        with self.assertRaises(ValueError):
            refinement.refine(data, base)

    def test_final_deployed_action_cost_matches_independent_oracle(self):
        data, base = case(12)
        data['legal_layers'] = torch.tensor([[True, False]] * 12)
        expected_initial = hpwl_oracle(data, base)
        result = refinement.refine(data, base, config=dict(max_seconds=10.))
        diag = result['refinement_diagnostics']
        expected_final = hpwl_oracle(data, result)
        self.assertAlmostEqual(diag['initial_objective'], expected_initial, places=6)
        self.assertAlmostEqual(diag['final_objective'], expected_final, places=6)
        self.assertLess(expected_final, expected_initial)
        self.assertEqual(result['xy'].dtype, base['xy'].dtype)
        self.assertTrue(validate_residual_action(data, result)['valid'])

    def test_actions_are_deterministic_and_inputs_are_read_only(self):
        data, base = case(12)
        old = copy.deepcopy(data)
        first = refinement.refine(data, base, config=dict(max_seconds=10.))
        torch.manual_seed(777)
        torch.randn(500)
        second = refinement.refine(data, base, config=dict(max_seconds=10.))
        self.assertEqual(action_digest(first), action_digest(second))
        assert_unchanged(self, data, old)

    def test_each_accepted_move_replays_to_the_reported_current_state_gain(self):
        data, base = case(12)
        data['legal_layers'] = torch.tensor([[True, False]] * 12)
        result = refinement.refine(data, base, config=dict(max_seconds=10.))
        diag = result['refinement_diagnostics']
        history = diag['history']
        self.assertGreater(len(history), 1)
        self.assertEqual(len(history), len(diag['accepted_moves']) + 1)
        self.assertTrue(all(a > b for a, b in zip(history, history[1:])))
        state = copy.deepcopy(base)
        for move in diag['accepted_moves']:
            self.assertEqual(len(move['nodes']), 1)
            node = move['nodes'][0]
            before = hpwl_oracle(data, state)
            self.assertEqual(int(state['assignment'][node]), move['old']['assignment'])
            self.assertEqual(int(state['layer'][node]), move['old']['layer'])
            self.assertEqual(state['xy'][node].tolist(), move['old']['xy'])
            for key in ('assignment', 'layer', 'xy'):
                state[key][node] = torch.as_tensor(move['new'][key], dtype=state[key].dtype)
            after = hpwl_oracle(data, state)
            self.assertAlmostEqual(before, move['before'], places=6)
            self.assertAlmostEqual(after, move['after'], places=6)
            self.assertAlmostEqual(before - after, move['gain'], places=6)
            self.assertGreater(before - after, diag['config']['gain_absolute'])
        for key in ('assignment', 'layer', 'xy'):
            self.assertTrue(torch.equal(state[key], result[key]))

    def test_budget_caps_stop_with_feasible_incumbent(self):
        data, base = case(16)
        for key, stop in (('max_moves', 'move_budget'),
                          ('max_evaluations', 'evaluation_budget'),
                          ('max_passes', 'pass_budget')):
            for cap in (0, 1):
                with self.subTest(key=key, cap=cap):
                    result = refinement.refine(data, base, config={key: cap, 'max_seconds': 10.})
                    diag = result['refinement_diagnostics']
                    self.assertTrue(validate_residual_action(data, result)['valid'])
                    counter = {'max_moves': 'accepted_moves', 'max_evaluations': 'evaluations',
                               'max_passes': 'passes'}[key]
                    self.assertLessEqual(diag['counters'][counter], cap)
                    if cap == 0:
                        self.assertEqual(action_digest(result), action_digest(base))
                        self.assertEqual(diag['stop_reason'], stop)
                        self.assertFalse(diag['stationarity_certified'])

    def test_early_deadline_and_work_exhaustion_are_not_claimed_convergence(self):
        data, base = case(12)
        ticks = itertools.count()
        with patch.object(refinement.time, 'perf_counter', side_effect=lambda: float(next(ticks))):
            result = refinement.refine(data, base, config=dict(max_seconds=.1))
        diag = result['refinement_diagnostics']
        self.assertEqual(action_digest(result), action_digest(base))
        self.assertEqual(diag['stop_reason'], 'time_budget_setup')
        self.assertIsNone(diag['initial_objective'])
        self.assertIsNone(diag['final_objective'])
        self.assertFalse(diag['stationarity_certified'])
        result = refinement.refine(data, base, config=dict(max_work_units=0))
        self.assertEqual(result['refinement_diagnostics']['stop_reason'], 'work_budget_setup')
        self.assertEqual(action_digest(result), action_digest(base))

    def test_base_relative_trust_box_prevents_cumulative_coordinate_drift(self):
        data, base = case(20)
        fraction = .07
        result = refinement.refine(data, base, config=dict(max_passes=6,
            max_evaluations=3000, trust_fraction=fraction, max_seconds=10.))
        selected = data['region_bounds'][base['assignment']]
        radius = (selected[:, [1, 3]] - selected[:, [0, 2]]) * fraction
        self.assertTrue((torch.abs(result['xy'] - base['xy']) <= radius + 1e-7).all())

    def test_fixed_coordinate_on_leaf_boundary_cannot_acquire_wrong_leaf(self):
        data, base = case(2)
        base['assignment'] = torch.tensor([1, 0])
        base['xy'] = torch.tensor([[.5, .7], [.25, .3]])
        data['fixed_xy'][0, 0] = .5
        data['movable'][1] = False
        data['legal_layers'] = torch.tensor([[True, False]] * 2)
        result = refinement.refine(data, base, config=dict(max_moves=1, max_passes=1))
        self.assertEqual(int(result['assignment'][0]), 1)
        self.assertEqual(float(result['xy'][0, 0]), .5)
        bounds = data['region_bounds'][result['assignment']]
        self.assertTrue((result['xy'] >= bounds[:, [0, 2]]).all())
        self.assertTrue((result['xy'] < bounds[:, [1, 3]]).all())

    def test_objective_and_history_use_deployed_dtype_before_accepting_gain(self):
        data, base = case(3)
        base['xy'] = base['xy'].double()
        base['xy'][0, 0] += 1e-9
        base['xy'][-1, 1] -= 1e-9
        deployed = dict(base, xy=base['xy'].to(data['demand'].dtype))
        result = refinement.refine(data, base, config=dict(max_moves=0))
        self.assertEqual(result['xy'].dtype, data['demand'].dtype)
        self.assertTrue(torch.equal(result['xy'], deployed['xy']))
        self.assertAlmostEqual(result['refinement_diagnostics']['initial_objective'],
                               hpwl_oracle(data, deployed), places=10)
        self.assertAlmostEqual(result['refinement_diagnostics']['final_objective'],
                               hpwl_oracle(data, result), places=10)

    def test_refined_action_lowers_complete_native_atom_initialization(self):
        data, base = case(5)
        result = refinement.refine(data, base)
        logical = dict(clusters=[dict(id=i, atoms=[f'a{i}', f'b{i}']) for i in range(5)])
        device = dict(grid_width=16, grid_height=16, layers=2)
        with tempfile.TemporaryDirectory(dir=HERE/'runs', prefix='development_016_local_lower_') as tmp:
            lowered = lower_initialization(logical, device, result, Path(tmp), method='E')
            lines = [line for line in Path(lowered['ap_init']).read_text().splitlines()
                     if not line.startswith('#')]
            self.assertEqual(len(lines), 10)
            self.assertEqual(lowered['molecule_count'], 5)
            self.assertEqual(lowered['atom_count'], 10)
            self.assertTrue(lowered['native_gp_required'])
            self.assertFalse(lowered['packing_legality_claimed'])

    def test_large_high_fanout_case_has_finite_work_and_strict_move_cap(self):
        data, base = case(4000)
        set_nets(data, [list(range(4000))] + [[i, i + 1] for i in range(99)])
        started = time.perf_counter()
        result = refinement.refine(data, base, config=dict(max_evaluations=30,
            max_moves=2, max_passes=2, max_seconds=2., max_work_units=100000))
        elapsed = time.perf_counter() - started
        diag = result['refinement_diagnostics']
        self.assertLessEqual(diag['counters']['evaluations'], 30)
        self.assertLessEqual(diag['counters']['accepted_moves'], 2)
        self.assertLess(elapsed, 5.)
        self.assertTrue(validate_residual_action(data, result)['valid'])

    def test_invalid_work_budgets_are_rejected(self):
        data, base = case()
        for config in (dict(max_evaluations=True), dict(max_evaluations=-1),
                       dict(max_moves=True), dict(max_moves=-1),
                       dict(max_passes=True), dict(max_passes=-1),
                       dict(max_seconds=float('nan')), dict(max_seconds=float('inf')),
                       dict(max_seconds=-1.), dict(unknown_knob=1)):
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    refinement.refine(data, base, config=config)


if __name__ == '__main__':
    unittest.main()
