"""Input eligibility and read-only path/provenance checks for 015."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from attribution_panel import ARCHES, SCHEMA, load_panel, remap_path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


class PanelTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name).resolve()
        self.panel = self.repo/'panel.json'
        self.binary = self.repo/'payload/vpr'
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(b'qualified binary')
        self.blif = self.repo/'payload/design.blif'
        self.blif.write_bytes(b'.model example\n.end\n')
        prefix = self.repo/'payload/structure'
        snapshots = {}
        for suffix in ('.blocks.tsv', '.nets.tsv', '.meta.tsv'):
            path = Path(str(prefix)+suffix)
            path.write_bytes(suffix.encode())
            snapshots[suffix] = sha(path)
        self.identity = self.repo/'prior/binary_identity.json'
        save(self.identity, dict(path=str(self.binary), sha256=sha(self.binary)))
        record = dict(design='example', family='example_family', prepared=True,
                      grid=[16, 16], blif=str(self.blif), structure_prefix=str(prefix),
                      mapped_blif_sha256=sha(self.blif), snapshot_sha256=snapshots,
                      hard_ap_counts=dict(memory=16, dsp_top=0),
                      resource_upper_utilization=[.1, .1, .5, 0, .1])
        conditions, rows, pins = [], [], {}
        for arch_id in ARCHES:
            folder = self.repo/'payload'/arch_id
            folder.mkdir()
            arch = folder/'arch.xml'
            arch.write_bytes(arch_id.encode())
            rr = folder/'rr_graph.xml'
            rr.write_bytes((arch_id+' RRG').encode())
            dev = folder/'device_graph.json'
            dev.write_text('{}')
            rr_manifest = folder/'manifest.json'
            # Producer binary intentionally differs: successful consumer is
            # the proof of compatibility, not the old RRG producer label.
            save(rr_manifest, dict(architecture_sha256=sha(arch),
                binary_sha256='f'*64, channel_width=128,
                rr_graph_sha256=sha(rr), graph_sha256=sha(dev)))
            consumer = self.repo/'prior'/arch_id/'native.json'
            command = [str(self.binary), str(arch), str(self.blif),
                       '--analytical_place', '--route', '--analysis',
                       '--route_chan_width', '128', '--seed', '101',
                       '--read_rr_graph', str(rr), '--ap_full_legalizer', 'appack',
                       '--ap_detailed_placer', 'annealer']
            save(consumer, dict(status='success', returncode=0, signature=dict(
                command=command, env=dict(TC_GP_LAYER_PRIOR='0'),
                inputs={str(p): sha(p) for p in (arch, self.blif, rr_manifest, self.identity)})))
            cid = 'example/'+arch_id+'/w128'
            conditions.append(dict(id=cid, design='example', family='example_family',
                architecture=arch_id, width=128, width_role='reference', split='train', record=record))
            rows.append(dict(condition_id=cid, method='B', seed=101, status='success',
                             vpr_manifest=str(consumer)))
            for path in (rr_manifest, consumer):
                pins[path.relative_to(self.repo).as_posix()] = sha(path)
        self.conditions = self.repo/'prior/conditions.json'
        self.report = self.repo/'prior/report.json'
        save(self.conditions, dict(records=conditions))
        save(self.report, dict(rows=rows))
        for path in (self.conditions, self.report, self.identity):
            pins[path.relative_to(self.repo).as_posix()] = sha(path)
        self.manifest = dict(schema=SCHEMA, minimum_families=1, minimum_heavy_families=1,
            backend_binary_sha256=sha(self.binary), source_files=pins, designs=[dict(
                design='example', family='example_family', reference_width=128,
                conditions='prior/conditions.json', native_report='prior/report.json',
                binary_identity='prior/binary_identity.json', source_output='prior',
                default_method='B', default_seed=101, vertical_layout='007',
                exposure='training_exposed_development', lineage='Independent fixture source.')])
        save(self.panel, self.manifest)

    def load(self, **kwargs):
        return load_panel(self.panel, repo=self.repo, **kwargs)

    def repin(self, path):
        self.manifest['source_files'][path.relative_to(self.repo).as_posix()] = sha(path)
        save(self.panel, self.manifest)

    def test_payload_and_old_producer_are_readonly(self):
        before = {p: p.read_bytes() for p in self.repo.rglob('*') if p.is_file()}
        result = self.load(verify_payload=True)
        self.assertEqual(result['family_count'], 1)
        self.assertEqual(result['conditions_count'], 4)
        self.assertEqual(result['resource_group_counts']['hard_resource_heavy'], 1)
        self.assertTrue(result['payload_verified'])
        self.assertEqual(result['conditions'][0]['rrg_producer_binary_sha256'], 'f'*64)
        self.assertEqual(before, {p: p.read_bytes() for p in self.repo.rglob('*') if p.is_file()})

    def test_lightweight_mode_does_not_claim_missing_payload_verified(self):
        self.blif.unlink()
        self.assertFalse(self.load()['payload_verified'])
        with self.assertRaises(FileNotFoundError):
            self.load(verify_payload=True)

    def test_payload_corruption_rejected(self):
        self.blif.write_bytes(b'changed mapped circuit')
        self.assertEqual(self.load()['family_count'], 1)
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            self.load(verify_payload=True)

    def test_source_manifest_corruption_rejected(self):
        self.report.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            self.load()

    def test_same_family_not_multiple_independent_designs(self):
        self.manifest['designs'].append(dict(self.manifest['designs'][0], design='variant'))
        save(self.panel, self.manifest)
        with self.assertRaisesRegex(ValueError, 'independent families'):
            self.load()

    def test_missing_architecture_rejected_without_silent_drop(self):
        value = json.loads(self.conditions.read_text())
        value['records'].pop()
        save(self.conditions, value)
        self.repin(self.conditions)
        with self.assertRaisesRegex(ValueError, 'reference condition'):
            self.load()

    def test_failed_default_is_not_qualified(self):
        value = json.loads(self.report.read_text())
        value['rows'][0]['status'] = 'unroutable'
        save(self.report, value)
        self.repin(self.report)
        with self.assertRaisesRegex(ValueError, 'successful Default'):
            self.load()

    def test_incomplete_structure_provenance_rejected(self):
        value = json.loads(self.conditions.read_text())
        for condition in value['records']:
            del condition['record']['snapshot_sha256']['.nets.tsv']
        save(self.conditions, value)
        self.repin(self.conditions)
        with self.assertRaisesRegex(ValueError, 'complete AP structure'):
            self.load()

    def test_warm_reference_is_not_cold_default(self):
        path = self.repo/'prior/H0Z0/native.json'
        value = json.loads(path.read_text())
        value['signature']['env']['TC_GP_INIT'] = '/some/initialization'
        save(path, value)
        self.repin(path)
        with self.assertRaisesRegex(ValueError, 'cold backend'):
            self.load()

    def test_unpinned_consumer_rejected(self):
        del self.manifest['source_files']['prior/H0Z0/native.json']
        save(self.panel, self.manifest)
        with self.assertRaisesRegex(ValueError, 'unpinned'):
            self.load()

    def test_insufficient_heavy_coverage_rejected(self):
        self.manifest['minimum_heavy_families'] = 2
        save(self.panel, self.manifest)
        with self.assertRaisesRegex(ValueError, 'heavy families'):
            self.load()

    def test_aliases_are_canonical_and_external_paths_rejected(self):
        suffix = 'experiments/vtr/file.json'
        for prefix in (
                '/inspire/hdd/global_user/u/workspace/DAC2027/TailCompile/',
                '/inspire/hdd/project/project-public/p/u/workspace/DAC2027/TailCompile/'):
            self.assertEqual(remap_path(prefix+suffix, self.repo), self.repo/suffix)
        for bad in ('../other.json', '/inspire/hdd/global_user/u/workspace/DAC2027/other/file',
                    '/inspire/hdd/global_user/u/workspace/DAC2027/TailCompile/../other'):
            with self.assertRaises(ValueError):
                remap_path(bad, self.repo)

    def test_native_rr_manifest_pin_cannot_be_replaced(self):
        path = self.repo/'payload/H0Z0/manifest.json'
        value = json.loads(path.read_text())
        value['rr_graph_sha256'] = '0'*64
        save(path, value)
        self.repin(path)
        with self.assertRaisesRegex(ValueError, 'RRG manifest bytes changed'):
            self.load()


if __name__ == '__main__':
    unittest.main()
