"""Capacity/fixed/fragmentation and finite-work tests for the 010 allocator."""
import copy
import math
import time
import unittest

import torch

from model import decode
from structured_decoder import (allocate_regions, validate_leaf_capacity,
                                StructuredCapacityError)
from test_model import fixture


def graph(n=6, r=4):
    data = fixture(n)
    cells = r//2
    data['capacity'] = torch.tensor([[math.ceil(n/r)+1., math.ceil(n/r)+1., 1., 1., 2.]]*r)
    data['region_layer'] = torch.arange(r)//cells
    data['region_bounds'] = torch.tensor([[i/cells, (i+1)/cells, 0., 1.]
                                         for _ in range(2) for i in range(cells)])
    data['region_x'] = torch.zeros(r, 5)
    return data


class StructuredDecoderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_collapsed_scores_no_longer_overfill_a_leaf(self):
        data = fixture()
        logits = torch.tensor([[100., 0., 100., 0.]]*6)
        old = decode(data, lambda nodes: logits[nodes], density_passes=0)
        self.assertGreater(old['capacity_diagnostics']['overflow_region_count'], 0)
        new = allocate_regions(data, torch.tensor([[100., 0.]]*6), logits)
        self.assertEqual(new['capacity_diagnostics']['overflow_region_count'], 0)
        self.assertTrue(new['capacity_diagnostics']['hard_leaf_capacity_enforced'])
        self.assertTrue((new['usage'] <= data['capacity']+1e-6).all())
        self.assertEqual(len(new['assignment']), 6)

    def test_fixed_xy_layer_and_heterogeneous_resource_obligations(self):
        data = fixture()
        data['capacity'][:, 2:] = 0
        data['capacity'][2, 2] = 2  # memory only exists on the upper-left leaf
        data['capacity'][1, 3] = 2  # DSP only exists on the lower-right leaf
        data['demand'][0] = torch.tensor([0., 0., 1., 0., 0.])
        data['demand'][1] = torch.tensor([0., 0., 0., 1., 0.])
        data['fixed_layer'][0] = 1
        data['fixed_xy'][0, 0] = .2
        data['fixed_region'][2] = 3
        result = allocate_regions(data, torch.tensor([[100., 0.]]*6), torch.zeros(6, 4))
        self.assertEqual(int(result['assignment'][0]), 2)
        self.assertEqual(int(result['assignment'][1]), 1)
        self.assertEqual(int(result['assignment'][2]), 3)
        self.assertTrue(validate_leaf_capacity(data, result['assignment'])['valid'])

    def test_aggregate_failure_stops_without_calling_scorer(self):
        data = fixture(9)
        calls = []
        with self.assertRaises(StructuredCapacityError) as error:
            allocate_regions(data, torch.zeros(9, 2), lambda ids: calls.append(ids))
        self.assertEqual(calls, [])
        self.assertEqual(error.exception.diagnostics['attempts'], 0)

    def test_feasible_fragmentation_is_explicit_and_bounded(self):
        data = graph(4, 2)
        data['capacity'] = torch.tensor([[4., 4., 0., 0., 0.]]*2)
        data['demand'] = torch.tensor([[3., 1., 0., 0., 0.], [1., 3., 0., 0., 0.],
                                       [2., 2., 0., 0., 0.], [2., 2., 0., 0., 0.]])
        self.assertTrue(validate_leaf_capacity(data, [0, 0, 1, 1])['valid'])
        with self.assertRaisesRegex(StructuredCapacityError, 'not a physical infeasibility') as error:
            allocate_regions(data, torch.tensor([[100., 0.], [0., 100.], [0., 0.], [0., 0.]]),
                             torch.zeros(4, 2), chunk_size=2, load_weight=0., max_restarts=2)
        self.assertEqual(error.exception.diagnostics['attempts'], 2)
        self.assertEqual(len(error.exception.diagnostics['failures']), 2)
        self.assertFalse(error.exception.diagnostics['fallback_used'])

    def test_more_regions_do_not_reduce_an_equal_best_layer_utility(self):
        data = graph(1)
        data['region_layer'] = torch.tensor([0, 0, 0, 1])
        result = allocate_regions(data, torch.tensor([[.1, 0.]]), torch.zeros(1, 4))
        self.assertEqual(int(result['layer'][0]), 0)

    def test_assignment_audit_rejects_float_truncation_and_overflow(self):
        data = fixture()
        with self.assertRaisesRegex(ValueError, 'integers'):
            validate_leaf_capacity(data, [.5, 0., 0., 0., 0., 0.])
        with self.assertRaisesRegex(StructuredCapacityError, 'capacity proxy'):
            validate_leaf_capacity(data, [0]*6)

    def test_callback_and_dense_scores_match_without_mutating_inputs(self):
        data = fixture()
        original = copy.deepcopy(data)
        layer_logits = torch.tensor([[.2, -.2]]*6)
        leaf_logits = torch.randn(6, 4)
        calls = []
        def score(nodes):
            calls.append(len(nodes))
            return leaf_logits[nodes]
        a = allocate_regions(data, layer_logits, score, chunk_size=2)
        b = allocate_regions(data, layer_logits, leaf_logits, chunk_size=6)
        self.assertTrue(torch.equal(a['assignment'], b['assignment']))
        self.assertLessEqual(sum(calls), 2*len(data['demand']))
        for key in ('demand', 'capacity', 'fixed_xy', 'fixed_layer'):
            self.assertTrue(torch.equal(data[key], original[key]))

    def test_nonfinite_allowed_scores_are_rejected(self):
        data = fixture()
        scores = torch.zeros(6, 4); scores[0, 0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'nonfinite compatible leaf'):
            allocate_regions(data, torch.zeros(6, 2), scores)
        with self.assertRaisesRegex(ValueError, 'max_restarts'):
            allocate_regions(data, torch.zeros(6, 2), torch.zeros(6, 4), max_restarts=1000)

    def test_twenty_thousand_nodes_use_bounded_chunked_work(self):
        n, r = 20000, 128
        data = graph(n, r)
        calls = []
        def score(nodes):
            calls.append(len(nodes))
            return torch.zeros(len(nodes), r)
        started = time.perf_counter()
        result = allocate_regions(data, torch.tensor([[10., 0.]]*n), score,
                                  chunk_size=256, max_restarts=1)
        self.assertEqual(sum(calls), n)
        self.assertLessEqual(max(calls), 256)
        self.assertEqual(result['capacity_diagnostics']['allocation_attempts'], 1)
        self.assertEqual(result['capacity_diagnostics']['overflow_region_count'], 0)
        self.assertFalse(result['capacity_diagnostics']['convergence_loop'])
        # Generous CPU bound catches accidental Python region x resource loops.
        self.assertLess(time.perf_counter()-started, 30.)


if __name__ == '__main__':
    unittest.main()
