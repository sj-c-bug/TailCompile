from pathlib import Path
import os
import subprocess
import sys
import tempfile
import unittest

from campaign import (ARCHES, campaign_lock, check_resource_upper_bound, hard_ap_counts,
                      identity, pilot_tasks, select_widths, within)


class CampaignTests(unittest.TestCase):
    def conditions(self):
        return [dict(id=f'{family}/{arch}/w{width}',family=family,architecture=arch,
                     width=width,width_role=role)
                for family in ('a','b','c','d') for arch in ARCHES
                for role,width in (('stress',64),('reference',128))]

    def test_exact_480_registered_tasks_and_missing_cells_rejected(self):
        conditions = self.conditions()
        tasks = pilot_tasks(conditions,['a','b','c','d'])
        self.assertEqual(len(tasks),480)
        self.assertEqual(len({(t['condition']['id'],t['seed'],t['method']) for t in tasks}),480)
        with self.assertRaisesRegex(ValueError,'complete'):
            pilot_tasks(conditions[:-1],['a','b','c','d'])
        with self.assertRaisesRegex(ValueError,'four independent'):
            pilot_tasks(conditions,['a','b','c','c'])

    def test_widths_use_all_eight_normal_results_and_no_timeout_as_boundary(self):
        def batch(width,successes):
            return [dict(width=width,architecture=arch,seed=seed,
                         status='success' if i<successes else 'unroutable')
                    for i,(arch,seed) in enumerate((a,s) for a in ARCHES for s in (501,502))]
        rows=batch(32,0)+batch(64,4)+batch(96,8)
        chosen=select_widths(rows,[32,64,96,128])
        self.assertEqual((chosen['W_stress'],chosen['W_ref']),(64,96))
        rows[8]['status']='timeout'
        chosen=select_widths(rows,[32,64,96,128])
        self.assertEqual(chosen['W_stress'],32)
        self.assertEqual(select_widths(batch(32,8),[32])['status'],'left_censored')
        self.assertEqual(select_widths(batch(32,0),[32])['status'],'right_censored')

    def test_signature_is_order_stable_and_output_paths_are_bounded(self):
        self.assertEqual(identity(dict(a=1,b=2)),identity(dict(b=2,a=1)))
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            self.assertEqual(within(Path(root)/'run',root),Path(root).resolve()/'run')
            with self.assertRaisesRegex(ValueError,'outside'):
                within(Path(root).parent/'unrelated',root)

    def test_cohort_hard_count_does_not_double_when_dsp_mode_demand_doubles(self):
        logic = dict(clusters=[dict(resource='dsp_top',demand=[0,0,0,2,0]) for _ in range(11)])
        self.assertEqual(hard_ap_counts(logic),dict(memory=0,dsp_top=11))
        self.assertLess(sum(hard_ap_counts(logic).values()),16)

    def test_known_bram_shortage_stops_before_width_scan(self):
        with self.assertRaisesRegex(ValueError, r'memory.*294.*248'):
            check_resource_upper_bound('mcml', [0,0,294,102,392], [71920,71920,248,496,3968])
        check_resource_upper_bound('mcml84', [0,0,294,204,392], [127920,127920,328,656,5248])

    def test_upper_bound_check_preserves_dense_but_not_overfull_candidate(self):
        check_resource_upper_bound('spmv', [100,100,232,64,99], [71920,71920,248,496,3968])
        with self.assertRaisesRegex(ValueError, 'io'):
            check_resource_upper_bound('io_overflow', [0,0,0,0,9], [10,10,1,2,8])

    @unittest.skipUnless(os.name=='posix','server stage locking uses Linux flock')
    def test_second_stage_process_is_rejected_and_release_allows_resume(self):
        code='from campaign import campaign_lock; import sys\ntry:\n with campaign_lock(sys.argv[1]): pass\nexcept RuntimeError:\n sys.exit(3)\n'
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as root:
            def attempt():
                return subprocess.run([sys.executable,'-c',code,root],cwd=Path(__file__).parent,
                                      capture_output=True,text=True,timeout=30).returncode
            with campaign_lock(root,'first'):
                self.assertEqual(attempt(),3)
            self.assertEqual(attempt(),0)


if __name__=='__main__':
    unittest.main()
