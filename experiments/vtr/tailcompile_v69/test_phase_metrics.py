"""Regression fixtures copied in shape from VPR 40f8d197 log output."""
import json
import unittest

from phase_metrics import parse_phase_metrics


class PhaseMetricsTest(unittest.TestCase):
    def test_ap_phases_are_not_added_to_enclosing_timers(self):
        value = parse_phase_metrics('''
## AP Global Placer took 3.02 seconds (max_rss 552.2 MiB)
### Creating Clusters took 10.57 seconds (max_rss 552.2 MiB)
## AP Full Legalizer took 10.64 seconds (max_rss 552.2 MiB)
### Placement Quench took 0.06 seconds (max_rss 552.2 MiB)
Placement Quench timing analysis took 0.0163586 seconds (0.0138015 STA)
## AP Detailed Placer took 4.34 seconds (max_rss 552.2 MiB)
# Analytical Placement took 27.31 seconds (max_rss 552.2 MiB)
# Routing took 5.79 seconds (max_rss 552.2 MiB)
The entire flow of VPR took 34.97 seconds (max_rss 552.2 MiB)
''')
        self.assertEqual(value['seconds']['full_legalization_seconds'], 10.64)
        self.assertEqual(value['seconds']['analytical_placement_seconds'], 27.31)
        self.assertEqual(value['seconds']['vpr_reported_seconds'], 34.97)
        self.assertEqual(len(value['evidence']), 6)
        self.assertIsNone(value['timing_semantics']['cpu_seconds'])

    def test_actual_tables_and_explicit_counters(self):
        value = parse_phase_metrics('''
Iter  Lower Bound HPWL  Upper Bound HPWL  UB CPD (ns)  UB sTNS (ns)  Solver Time  Legalizer Time  Timing Update Time  Total Time
                                                                           (sec)           (sec)               (sec)       (sec)
----  ----------------  ----------------  -----------
   0              0.00         251911.83        7.678      -42843.7        0.000  0.021  0.027  0.049
   1             25.60         155026.81        7.258      -38823.7        0.029  0.022  0.023  0.074
   2             59.88         149691.03        6.996      -37745.9        0.029  0.022  0.023  0.074
B2B Solver Statistics:
    Total number of CG iterations: 6061
Bi-Partitioning Partial Legalizer Statistics:
    Total number of windows partitioned: 153479
    Total number of blocks partitioned: 3125117
## AP Global Placer took 3.02 seconds (max_rss 552.2 MiB)
Placement number of temperatures: 75
Placement total # of swap attempts: 453326
Iter   Time    pres  BBs    Heap  Re-Rtd  Re-Rtd Overused RR Nodes      Wirelength      CPD
      (sec)     fac Updt    push    Nets   Conns                                       (ns)
---- ------ ------- ---- ------- ------- ------- ----------------- --------------- --------
   1    0.5     0.0    0 2715513    9592   16918    9236 ( 0.691%)  214568 (15.8%)    7.245
   2    0.5     0.5    0 2383193    7103   13690    4957 ( 0.371%)  214814 (15.8%)    7.245
Restoring best routing
Successfully routed after 2 routing iterations.
# Routing took 1.1 seconds (max_rss 552.2 MiB)
''')
        counts = value['counters']
        self.assertEqual(counts['gp_iteration_rows_observed'], 3)
        self.assertEqual(counts['gp_optimizer_steps_observed'], 2)
        self.assertEqual(counts['gp_cg_iterations'], 6061)
        self.assertEqual(counts['placement_swap_attempts'], 453326)
        self.assertEqual(counts['routing_iteration_rows_observed'], 2)
        self.assertEqual(counts['routing_iterations_reported_success'], 2)
        json.dumps(value, allow_nan=False)

    def test_timeout_keeps_partial_iterations_but_not_fabricated_runtime(self):
        value = parse_phase_metrics('''
## AP Global Placer took 3.00 seconds (max_rss 552.2 MiB)
Iter   Time    pres  BBs    Heap  Re-Rtd  Re-Rtd Overused RR Nodes
      (sec)     fac Updt    push    Nets   Conns
---- ------ ------- ---- ------- ------- -------
   1  350.0     0.0    0 2715513    9592   16918    9236 ( 0.691%)
   2  400.0     0.5    0 2383193    7103   13690    4957 ( 0.371%)
''')
        self.assertIsNone(value['seconds']['routing_seconds'])
        self.assertIsNone(value['seconds']['vpr_reported_seconds'])
        self.assertIsNone(value['counters']['routing_iterations_reported_success'])
        self.assertEqual(value['counters']['routing_last_iteration_index'], 2)
        self.assertFalse(value['completed_phase_timers']['routing_seconds'])

    def test_empty_and_scientific_time(self):
        self.assertTrue(all(x is None for x in parse_phase_metrics('')['seconds'].values()))
        value = parse_phase_metrics('## AP Global Placer took 1.2e-03 seconds\n'
                                    '# Routing took 0.00 seconds\n')
        self.assertEqual(value['seconds']['global_placement_seconds'], .0012)
        self.assertEqual(value['seconds']['routing_seconds'], 0.)
        self.assertEqual(value['completed_timer_occurrences']['routing_seconds'], 1)


if __name__ == '__main__':
    unittest.main()
