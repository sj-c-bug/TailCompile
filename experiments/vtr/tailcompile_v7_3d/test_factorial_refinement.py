"""017 independent portal, timing, atomic exchange and scale regressions."""
import copy
import math
import time
import unittest

import numpy as np
import torch

import explainable_refinement as frozen016
import factorial_refinement as f
from aligned_model import action_digest
from test_explainable_refinement import case, portals, set_nets


def timing():
    return dict(horizontal_delay_per_tile_ns=[.1, 2.],
                vertical_delay_ns=[[0., .3], [.7, 0.]], terminal_delay_ns=.02)


def scale_probe(n=125796, net_count=130836, mixed=False, symmetric=False):
    """Opt-in MCML-sized reproducible setup probe, excluded from unit tests.

    64 duplicate directed pool profiles / an irregular 1681-member patch;
    all nets cross-layer by default, or half cross / half same-layer.
    Returns completed baseline measurements and all four bounded-arm audits.
    """
    torch.set_num_threads(1)
    data, base = case(n)
    data['grid'] = [84, 84, 2]
    ids = np.arange(n)
    xy = np.stack((.02+.96*((ids*17)%997)/997,
                   .02+.96*((ids*31)%991)/991), 1).astype(np.float32)
    z = (ids >= n//2).astype(np.int64)
    leaves = (xy[:, 0] >= .5).astype(np.int64)+2*z
    base = dict(xy=torch.from_numpy(xy), assignment=torch.from_numpy(leaves), layer=torch.from_numpy(z))
    pairs = []
    for i in range(net_count):
        a, b = i%(n//2), n//2+(i*17+19)%(n//2)
        if mixed and i%2:
            a = (a+n//2) if i%4 == 1 else a
            b = n//2+(i*17+19)%(n//2) if i%4 == 1 else (i*17+19)%(n//2)
        pairs.append([a, b])
    set_nets(data, pairs)
    positions = sorted(((x, y) for x in range(84) for y in range(84)),
        key=lambda p: ((p[0]-41.5)**2+(p[1]-41.5)**2, p))[:1681]
    summary = portals(tuple(((x+.5)/84, (y+.5)/84) for x, y in positions))
    summary['grid'] = [84, 84, 2]
    summary['pools'][0]['orientations'].extend(portals(reverse_only=True)['pools'][0]['orientations'])
    summary['pools'] *= 64
    start = time.perf_counter()
    model = timing()
    if symmetric:
        model['horizontal_delay_per_tile_ns'] = [.1, .1]
    p = f.prepare_problem(data, base, summary, model,
        config=dict(max_setup_work_units=20000000 if symmetric else 200000000))
    report = dict(n=n, nets=net_count, mixed=mixed, irregular_portals=len(positions),
        raw_pool_count=64, setup_seconds=time.perf_counter()-start,
        setup_work_units=p.setup_work, normalizers=p.normalizers, symmetric=symmetric,
        baseline_complete=all(c is not None for v in p.baseline.values() for c in v),
        merged_access_profiles=len(p.profiles), coordinate_sets=len(p.coordinate_indices), arms=[])
    for mode in ('wl', 'joint'):
        for swaps in (False, True):
            action = f.refine(data, base, problem=p, objective_mode=mode, allow_swaps=swaps,
                config=dict(max_seconds=.25))
            d = action['refinement_diagnostics']
            report['arms'].append(dict(objective=mode, swaps=swaps, setup_complete=d['setup_complete'],
                initial_objective=d['initial_objective'], final_objective=d['final_objective'],
                stop_reason=d['stop_reason'], seconds=d['elapsed_search_seconds'], counters=d['counters']))
    return report


class FactorialRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_wl_matches_frozen016_actual_irregular_portals(self):
        data, base = case(7)
        base['assignment'] = torch.tensor([0, 2, 1, 3, 0, 2, 0])
        base['layer'] = torch.tensor([0, 1, 0, 1, 0, 1, 0])
        base['xy'] = torch.tensor([[.1, .1], [.2, .4], [.9, .3], [.8, .7], [.3, .9], [.4, .3], [.2, .8]])
        set_nets(data, [[0, 1], [0, 2, 3], [1, 4, 5, 6]])
        summary = portals(((.1, .3), (.2, .5), (.2, .1), (.7, .7), (.9, .2)))
        summary['pools'][0]['orientations'].extend(portals(reverse_only=True)['pools'][0]['orientations'])
        old = frozen016.objective(data, base, summary)
        new = f.objective(data, base, summary, timing())
        self.assertEqual(new['unsupported_net_ids'], [])
        self.assertAlmostEqual(new['wirelength'], old['value'], places=10)
        for a, b in zip(new['net_costs'], old['net_costs']):
            self.assertAlmostEqual(a[0], b, places=10)

    def test_column_index_two_pin_exact_against_exhaustive(self):
        rng = np.random.default_rng(43)
        points = np.unique(rng.integers(0, 30, (400, 2)), axis=0).astype(float)
        index = f._PortalIndex(points)
        a, b = rng.uniform(0, 30, (2, 121, 2))
        wa, wb = rng.uniform(.1, 3., (2, 121))
        chosen = index.two(a, b, wa, wb)
        actual = wa*np.abs(a-chosen).sum(1)+wb*np.abs(b-chosen).sum(1)
        expected = (wa[:, None]*np.abs(a[:, None, :]-points).sum(2)+wb[:, None]*np.abs(b[:, None, :]-points).sum(2)).min(1)
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_column_index_multiterminal_boxes_exact(self):
        rng = np.random.default_rng(11)
        points = np.unique(rng.integers(0, 20, (200, 2)), axis=0).astype(float)
        for points in (points, np.array([(x, y) for x in range(7) for y in range(4)])):
            index = f._PortalIndex(points)
            boxes = np.sort(rng.uniform(0, 20, (71, 2, 2, 2)), axis=2)
            lo, hi = boxes[:, :, 0, :], boxes[:, :, 1, :]
            chosen = index.boxes(lo, hi)
            actual = np.maximum(lo-chosen[:, None, :], 0).sum((1, 2))+np.maximum(chosen[:, None, :]-hi, 0).sum((1, 2))
            expected = (np.maximum(lo[:, None, :, :]-points[None, :, None, :], 0).sum((2, 3)) + np.maximum(points[None, :, None, :]-hi[:, None, :, :], 0).sum((2, 3))).min(1)
            np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)

    def test_joint_route_is_device_delay_not_reweighted_hpwl(self):
        data, base = case(2)
        base['assignment'] = torch.tensor([0, 3]); base['layer'] = torch.tensor([0, 1])
        base['xy'] = torch.tensor([[.1, .1], [.9, .9]])
        summary = portals(((.1, .1), (.9, .9)))
        p = f.prepare_problem(data, base, summary, timing())
        self.assertAlmostEqual(p.baseline['wl'][0][0], p.baseline['joint'][0][0], places=5)
        self.assertGreater(p.baseline['wl'][0][1], p.baseline['joint'][0][1])
        # Identical HPWL can have different max driver-sink Manhattan delay.
        data, a = case(3); set_nets(data, [[0, 1, 2]])
        a['assignment'] = torch.tensor([0, 1, 0]); a['xy'] = torch.tensor([[.1, .1], [.9, .1], [.1, .9]])
        b = copy.deepcopy(a); b['assignment'][2] = 1; b['xy'][1] = torch.tensor([.9, .9]); b['xy'][2] = torch.tensor([.9, .1])
        va, vb = f.objective(data, a, timing_model=timing()), f.objective(data, b, timing_model=timing())
        self.assertAlmostEqual(va['wirelength'], vb['wirelength'], places=6)
        self.assertGreater(vb['timing'], va['timing'])

    def test_high_fanout_delay_extrema_equal_brute_force(self):
        rng = np.random.default_rng(37)
        data, base = case(211)
        xy = rng.uniform(.01, .99, (211, 2)).astype(np.float32)
        z = np.arange(211)%2
        leaves = (xy[:, 0] >= .5).astype(np.int64)+2*z
        base = dict(xy=torch.from_numpy(xy), assignment=torch.from_numpy(leaves), layer=torch.from_numpy(z))
        set_nets(data, [list(range(211))])
        summary = portals(tuple(map(tuple, rng.uniform(.01, .99, (71, 2)))))
        p = f.prepare_problem(data, base, summary, timing())
        points = p.profiles[0][1].points
        _, actual = p._at_portals(0, p.base, points)
        expected = np.zeros(len(points))
        h, v, terminal = p.timing
        for sink in range(1, 211):
            if z[sink] == z[0]:
                delay = h[z[0]]*np.abs(xy[0]*p.scale-xy[sink]*p.scale).sum()+terminal
            else:
                delay = h[z[0]]*np.abs(points-xy[0]*p.scale).sum(1)+h[z[sink]]*np.abs(points-xy[sink]*p.scale).sum(1)+v[z[0], z[sink]]+terminal
            expected = np.maximum(expected, delay)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)

    def test_unknown_driver_has_no_fabricated_timing_path(self):
        data, base = case(2)
        data['risk_nets'][0]['drivers'] = []
        with self.assertRaisesRegex(ValueError, 'known driver/sink'):
            f.refine(data, base, timing_model=timing(), objective_mode='joint', config=dict(max_moves=0))
        result = f.refine(data, base, timing_model=timing(), config=dict(max_moves=0))
        diag = result['refinement_diagnostics']
        self.assertEqual(diag['initial_timing'], 0.)
        self.assertEqual(diag['timing_known_fraction'], 0.)
        self.assertFalse(diag['timing_improvement_claimed'])

    def test_atomic_exchange_escapes_full_capacity_and_costs_two_touches(self):
        data, base = case(4)
        base['assignment'] = torch.tensor([0, 2, 0, 2]); base['layer'] = torch.tensor([0, 1, 0, 1])
        base['xy'] = torch.tensor([[.1, .2], [.1, .2], [.4, .2], [.4, .2]])
        data['capacity'][:, 0] = torch.tensor([2., 0., 2., 0.])
        data['fixed_xy'] = base['xy'].clone()
        set_nets(data, [[0, 3], [2, 1]])
        summary = portals(((.9, .9),))
        summary['pools'][0]['orientations'].extend(portals(reverse_only=True)['pools'][0]['orientations'])
        p = f.prepare_problem(data, base, summary, timing())
        single = f.refine(data, base, objective_mode='wl', problem=p, config=dict(max_moves=2))
        swap = f.refine(data, base, objective_mode='wl', allow_swaps=True, problem=p, config=dict(max_moves=2))
        self.assertEqual(action_digest(single), action_digest(base))
        diag = swap['refinement_diagnostics']
        self.assertEqual(diag['counters']['accepted_swaps'], 1)
        self.assertEqual(diag['counters']['node_touches'], 2)
        self.assertLess(diag['final_objective'], diag['initial_objective'])
        self.assertTrue(f.validate(data, swap, base=base)['valid'])
        for move in diag['accepted_moves']:
            self.assertEqual(len(move['affected_nets']), len(set(move['affected_nets'])))
        blocked = f.refine(data, base, allow_swaps=True, problem=p, config=dict(max_moves=1))
        self.assertEqual(action_digest(blocked), action_digest(base))

    def test_four_arms_share_reference_without_mutation(self):
        data, base = case(10)
        old = copy.deepcopy(base)
        p = f.prepare_problem(data, base, portals(), timing())
        initial = {mode: copy.deepcopy(c) for mode, c in p.baseline.items()}
        original_access_keys = set(p.access_cache)
        original_coordinate_keys = set(p.coordinate_indices)
        norms = []
        for mode in ('wl', 'joint'):
            for swaps in (False, True):
                action = f.refine(data, base, objective_mode=mode, allow_swaps=swaps, problem=p,
                    config=dict(max_moves=7, max_seconds=4.))
                diag = action['refinement_diagnostics']; norms.append(diag['normalizers'])
                self.assertTrue(diag['setup_complete'])
                self.assertLessEqual(diag['counters']['node_touches'], 7)
                self.assertTrue(all(b < a for a, b in zip(diag['history'], diag['history'][1:])))
                exact = f.objective(data, action, objective_mode=mode, problem=p)
                self.assertAlmostEqual(diag['final_objective'], exact['value'], places=11)
        self.assertTrue(all(n == norms[0] for n in norms))
        self.assertEqual(p.baseline, initial)
        self.assertEqual(set(p.access_cache), original_access_keys)
        self.assertEqual(set(p.coordinate_indices), original_coordinate_keys)
        for key in old:
            self.assertTrue(torch.equal(base[key], old[key]))

    def test_setup_budget_failure_is_explicit_and_search_budget_is_separate(self):
        data, base = case()
        with self.assertRaises(f.SetupBudgetExceeded):
            f.prepare_problem(data, base, timing_model=timing(), config=dict(max_setup_seconds=0.))
        p = f.prepare_problem(data, base, timing_model=timing(), config=dict(max_seconds=0.))
        out = f.refine(data, base, problem=p, config=dict(max_seconds=0.))
        self.assertTrue(out['refinement_diagnostics']['setup_complete'])
        self.assertEqual(out['refinement_diagnostics']['stop_reason'], 'time_budget')

    def test_equivalent_pool_profiles_share_coordinate_index(self):
        data, base = case(2)
        summary = portals(((.1, .1), (.2, .2)))
        summary['pools'] *= 200
        p = f.prepare_problem(data, base, summary, timing())
        self.assertEqual(p.raw_pool_count, 200)
        self.assertEqual(len(p.profiles), 1)
        self.assertEqual(len(p.coordinate_indices), 1)

    def test_fixed_immovable_and_original_trust_remain_hard_constraints(self):
        data, base = case(12)
        data['movable'][0] = False
        data['fixed_xy'][1, 0] = base['xy'][1, 0]
        data['fixed_layer'][2] = 0
        data['fixed_region'][3] = base['assignment'][3]
        action = f.refine(data, base, portals(), timing(), 'joint', True,
                         dict(max_moves=12, max_seconds=5.))
        self.assertTrue(f.validate(data, action, base=base)['valid'])
        self.assertTrue(torch.equal(action['xy'][0], base['xy'][0]))
        self.assertEqual(float(action['xy'][1, 0]), float(base['xy'][1, 0]))
        self.assertEqual(int(action['layer'][2]), 0)
        self.assertEqual(int(action['assignment'][3]), int(base['assignment'][3]))


if __name__ == '__main__':
    unittest.main()
