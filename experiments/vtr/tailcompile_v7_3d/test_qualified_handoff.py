"""Exercise metadata handoff and cold comparison without launching VPR."""
from collections import Counter
import itertools
import os
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import qualified_learning as q
from io_utils import read, sha, write

HERE = Path(__file__).resolve().parent


class QualifiedHandoffTests(unittest.TestCase):
    def temporary(self):
        (HERE/'runs').mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='qualified-integration-', dir=HERE/'runs')

    def fixture(self, root):
        """Six-input source, five qualified panels, one cache borrowed from 003."""
        source, out, borrowed = root/'source004', root/'learning005', root/'source003'
        backend, mapping = root/'readonly_backend', root/'readonly_mapping'
        binary = backend/'build_tailcompile_v7/vpr/vpr'
        binary.parent.mkdir(parents=True)
        binary.write_bytes(b'fake-pinned-vpr')
        mapping.mkdir()
        write(backend/'backend_manifest.json', {'binary_sha256': sha(binary)})
        designs, records, conditions, boundaries = [], [], [], []
        for design, family in {**q.TRAIN, **q.VALIDATION, 'or1200': 'or1200'}.items():
            split = 'validation' if design in q.VALIDATION else 'train'
            original = dict(design=design, family=family, split=split, grid=[16, 16])
            designs.append(original)
            folder = source/'inputs'/design
            folder.mkdir(parents=True)
            rtl = folder/'input.v'
            rtl.write_text('module '+design+'; endmodule\n')
            blif, prefix = folder/'design.blif', folder/'structure'
            blif.write_text('.model '+design+'\n.end\n')
            suffixes = ('.blocks.tsv', '.nets.tsv', '.meta.tsv')
            for suffix in suffixes:
                Path(str(prefix)+suffix).write_text(design+suffix)
            record = dict(original, blif=str(blif), mapped_blif_sha256=sha(blif),
                structure_prefix=str(prefix), snapshot_sha256={s: sha(str(prefix)+s) for s in suffixes})
            records.append(record)
            write(folder/'source.json', dict(source=str(rtl), sha256=sha(rtl),
                family=family, split=split, fixed_grid=[16, 16]))
            trials = [dict(design=design, family=family, architecture=a, seed=s, width=w,
                method='default', status='success' if w == 64 and design != 'or1200' else 'unroutable')
                for a, s, w in itertools.product(q.ARCHES, (501, 502), (32, 64))]
            boundary = dict(q.select_widths(trials, [32, 64]), design=design, family=family, record=record)
            boundaries.append(boundary)
            write(source/'calibration'/design/'trials.json', trials)
            write(source/'calibration'/design/'boundary.json', boundary)
            if design != 'or1200':
                for arch, role in itertools.product(q.ARCHES, ('stress', 'reference')):
                    width = 32 if role == 'stress' else 64
                    conditions.append(dict(id=f'{design}/{arch}/w{width}', design=design,
                        family=family, split=split, architecture=arch, width=width,
                        width_role=role, record=record))
        devices = {}
        for index, (arch, width) in enumerate(itertools.product(q.ARCHES, (32, 64))):
            arch_file = source/'architectures/16x16'/arch/'arch.xml'
            arch_file.parent.mkdir(parents=True, exist_ok=True)
            arch_file.write_text('<architecture name="'+arch+'"/>')
            write(arch_file.parent/'architecture.json', dict(architecture_sha256=sha(arch_file)))
            key = q.identity(dict(architecture=sha(arch_file), binary=sha(binary), width=width))[:20]
            folder = (borrowed if index == 0 else source)/'device_cache'/key
            folder.mkdir(parents=True)
            (folder/'rr_graph.xml').write_text('<rr_graph width="'+str(width)+'"/>')
            write(folder/'device_graph.json', dict(width=width))
            write(folder/'manifest.json', dict(architecture_sha256=sha(arch_file),
                binary_sha256=sha(binary), channel_width=width,
                rr_graph_sha256=sha(folder/'rr_graph.xml'), graph_sha256=sha(folder/'device_graph.json')))
            if index == 0:
                devices[key] = dict(folder=str(folder), manifest_sha256=sha(folder/'manifest.json'))
        write(source/'architectures/16x16/actual_audit.json', dict(four_cell_matches=True,
            width_coupling={arch: dict(z_capacity_invariant=True) for arch in q.ARCHES}))
        catalog = source/'capacity_recovery_catalog.json'
        write(catalog, dict(devices=devices))
        config = dict(designs=designs, width_grid=[32, 64], primary_runtime='cold',
            input_recovery=dict(catalog=str(catalog), sha256=sha(catalog)))
        write(source/'campaign.json', config)
        write(source/'protocol.json', dict(config=config, mapping_vtr_root=str(mapping),
            backend_manifest_sha256=sha(backend/'backend_manifest.json')))
        write(source/'prepared.json', dict(records=records, complete=True))
        write(source/'conditions.json', dict(records=conditions, censored=['or1200']))
        write(source/'boundaries.json', dict(records=boundaries))
        write(source/'conditions_manifest.json', dict(conditions_sha256=sha(source/'conditions.json'),
            prepared_sha256=sha(source/'prepared.json'), boundaries_sha256=sha(source/'boundaries.json')))
        write(source/'binary_identity.json', dict(binary=str(binary), sha256=sha(binary)))
        write(source/'qualification/protocol.json', dict(binary_sha256=sha(binary)))
        write(source/'qualification/qualification_report.json', dict(complete=True, passed=True))
        context = dict(binary=binary, binary_sha256=sha(binary),
            source_hashes={str(HERE/'complete_device_cache.py'): sha(HERE/'complete_device_cache.py')})
        return SimpleNamespace(root=root, source=source, out=out, borrowed=borrowed,
            backend=backend, binary=binary, context=context, conditions=conditions)

    @staticmethod
    def snapshot(*roots):
        return {str(p): (sha(p), p.stat().st_mtime_ns)
                for root in roots for p in root.rglob('*') if p.is_file()}

    def handoff(self, f):
        # Frozen-code/binary qualification is tested separately. All metadata,
        # cache lookup, file pinning, output ownership and freeze IO below is real.
        with patch.object(q, '_validate_protocol', return_value=f.context):
            return q.prepare_handoff(f.source, f.out)

    def reader(self, f, catalog):
        instance = q.QualifiedCampaign.__new__(q.QualifiedCampaign)
        instance.out = f.out
        instance.catalog = catalog
        instance.binary_hash = f.context['binary_sha256']
        instance.check_binary = lambda: None
        instance._lock = lambda key: threading.Lock()
        instance.verified_rr = set()
        return instance

    def test_handoff_reuses_payloads_and_never_writes_source(self):
        with self.temporary() as tmp:
            f = self.fixture(Path(tmp))
            before = self.snapshot(f.source, f.borrowed, f.backend)
            catalog = self.handoff(f)
            self.assertEqual(catalog['backend'], str(f.backend))
            self.assertEqual(catalog['binary'], str(f.binary))
            self.assertEqual(len(catalog['devices']), 8)
            self.assertTrue(any(str(f.borrowed) in path for path in catalog['devices'].values()))
            self.assertFalse((f.out/'device_cache').exists())
            self.assertEqual(len(read(f.out/'prepared.json')['records']), 5)
            self.assertEqual(len(read(f.out/'conditions.json')['records']), 40)
            self.assertNotIn('input_recovery', read(f.out/'campaign.json'))
            reader = self.reader(f, catalog)
            self.assertEqual(len(reader.records()), 5)
            for c in f.conditions:
                reader.rr(c['record'], c['architecture'], c['width'])
            output_before = self.snapshot(f.out)
            self.assertEqual(catalog, self.handoff(f))
            self.assertEqual(output_before, self.snapshot(f.out))
            self.assertEqual(before, self.snapshot(f.source, f.borrowed, f.backend))

    def test_changed_borrowed_blif_and_structure_rejected_by_stage_reader(self):
        with self.temporary() as tmp:
            f = self.fixture(Path(tmp))
            reader = self.reader(f, self.handoff(f))
            record = reader.records()[0]
            blif = Path(record['blif'])
            original = blif.read_bytes()
            blif.write_bytes(original+b'changed')
            with self.assertRaisesRegex(ValueError, 'mapped BLIF changed'):
                reader.records()
            blif.write_bytes(original)
            Path(record['structure_prefix']+'.blocks.tsv').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'structural snapshot changed'):
                reader.records()

    def test_changed_source_metadata_rejects_handoff_resume(self):
        with self.temporary() as tmp:
            f = self.fixture(Path(tmp))
            self.handoff(f)
            write(f.source/'inputs/sha/source.json', dict(source='different source'))
            with self.assertRaisesRegex(ValueError, 'frozen input changed'):
                self.handoff(f)

    def test_missing_large_payload_does_not_launch_or_publish_handoff(self):
        with self.temporary() as tmp:
            f = self.fixture(Path(tmp))
            next(f.borrowed.glob('device_cache/*/rr_graph.xml')).unlink()
            with self.assertRaisesRegex(ValueError, 'raw server RRG/device payloads missing'):
                self.handoff(f)
            self.assertFalse((f.out/'qualified_source_manifest.json').exists())

    @unittest.skipUnless(hasattr(os, 'sched_setaffinity') and Path('/proc/sys/kernel/random/boot_id').exists(),
                         'controlled comparison requires Linux thread affinity')
    def test_cold_panel_all_outcomes_and_whole_wave_resume(self):
        with self.temporary() as tmp:
            f = self.fixture(Path(tmp))
            campaign = q.QualifiedCampaign.__new__(q.QualifiedCampaign)
            campaign.out = f.out
            checkpoint = f.root/'checkpoint.pt'
            checkpoint.write_bytes(b'fake-deployed-checkpoint')
            campaign.selected = lambda: checkpoint
            campaign.conditions = lambda: f.conditions
            original_affinity = os.sched_getaffinity(0)
            cores = sorted(original_affinity)[:4]
            campaign.budget = dict(physical_cores=cores)
            campaign.args = SimpleNamespace(timeout=10, threads=1)
            campaign.binary = f.binary
            campaign.binary_reference = f.source/'binary_identity.json'
            campaign.guard = lambda: None
            campaign.rr = lambda *args: {}
            campaign.architecture = lambda record, arch: f.source/'architectures/16x16'/arch/'arch.xml'
            write(f.out/'selection.json', dict(status='selected_by_real_vpr_validation'))
            calls, lock = Counter(), threading.Lock()
            def prepared(folder, command, **kwargs):
                self.assertEqual(os.sched_getaffinity(0), {kwargs['affinity']})
                with lock:
                    calls['prepare'] += 1
                    first = calls['prepare'] == 1
                return dict(status='infrastructure_failure' if first else 'complete')
            def graph(condition):
                self.assertEqual(len(os.sched_getaffinity(0)), 1)
                self.assertEqual(condition['record']['structure_export_seconds'], 0.)
                self.assertIn('cold_graph', condition['graph_root'])
                with lock:
                    calls['graph'] += 1
                return {}
            def execute(condition, seed, folder, *, method='default', checkpoint=None, affinity=None):
                self.assertEqual(os.sched_getaffinity(0), {affinity})
                with lock:
                    calls['execute'] += 1
                status = 'unroutable' if condition['width_role'] == 'stress' else 'success'
                return dict(condition_id=condition['id'], seed=seed, method=method, status=status,
                    cpd_ns=1. if status == 'success' else None,
                    wirelength=100. if status == 'success' else None,
                    vpr_manifest='fake/vpr/manifest.json', **{k: condition[k] for k in
                        ('design', 'family', 'architecture', 'width', 'width_role')})
            campaign.graph, campaign.execute = graph, execute
            with patch.object(q, 'run_trial', side_effect=prepared), patch.object(q, 'emit'):
                result = campaign.compare()
            self.assertEqual(os.sched_getaffinity(0), original_affinity)
            self.assertEqual(result['completed'], 600)
            self.assertEqual(calls, Counter(prepare=400, graph=399, execute=599))
            self.assertTrue(all(r['measured_cold_total'] and r['total_seconds'] > 0 for r in result['rows']))
            self.assertEqual(len({(r['condition_id'], r['seed'], r['method']) for r in result['rows']}), 600)
            summary = q.comparison_summary(result['rows'], f.conditions)
            self.assertTrue(summary['complete'])
            self.assertEqual(summary['unknown_failure_count'], 1)
            self.assertFalse(any(r['provisional_runtime_target'] for r in summary['split_comparisons']))
            missing = q.comparison_summary(result['rows'][:-1], f.conditions)
            self.assertFalse(missing['complete'])
            completed_before = {str(p): (sha(p), p.stat().st_mtime_ns)
                for p in (f.out/'comparison').glob('wave_*/completed.json')}
            campaign.graph = lambda *a, **k: self.fail('completed waves must not rebuild graphs')
            campaign.execute = lambda *a, **k: self.fail('completed waves must not execute VPR')
            with patch.object(q, 'run_trial', side_effect=AssertionError('unexpected subprocess')), patch.object(q, 'emit'):
                resumed = campaign.compare()
            self.assertEqual(resumed, result)
            self.assertEqual(completed_before, {str(p): (sha(p), p.stat().st_mtime_ns)
                for p in (f.out/'comparison').glob('wave_*/completed.json')})
            with patch.object(q.os, 'uname', return_value=('Linux', 'changed-host', '', '', '')):
                with self.assertRaisesRegex(ValueError, 'comparison host/cores changed'):
                    campaign.compare()


if __name__ == '__main__':
    unittest.main()
