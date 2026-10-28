"""Capacity correction, evidence-preserving reuse and refusal contracts."""
from pathlib import Path
import os
import hashlib
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import recover_capacity as recovery
from campaign import select_widths
from io_utils import COMMIT, read, sha, write

HERE = Path(__file__).resolve().parent


class CapacityRecoveryTests(unittest.TestCase):
    def temporary(self):
        (HERE/'runs').mkdir(exist_ok=True)
        return tempfile.TemporaryDirectory(prefix='capacity-recovery-', dir=HERE/'runs')

    def fixture(self, root):
        repo = root/'repo'
        old = repo/'experiments/vtr/tailcompile_v7_3d/runs/old'
        new = old.parent/'new'
        backend = root/'read_only_backend'
        (backend/'vpr').mkdir(parents=True)
        (backend/'vpr/vpr').write_bytes(b'qualified-binary')
        binary_hash = sha(backend/'vpr/vpr')
        source_hashes = {}
        for name in (*recovery.install_hook.FILES, *recovery.install_hook.HETERO_FILES):
            p = backend/name
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(name)
            source_hashes[name] = sha(p)
        write(backend/'backend_manifest.json', dict(vtr_commit=COMMIT, binary_sha256=binary_hash,
              hook_sha256='verified-hook', source_sha256=source_hashes))
        config = dict(tile_size=4, width_grid=[32, 64], designs=[
            dict(design='sha', family='sha', split='train', source='sha.v', grid=[16, 16]),
            dict(design='mcml', family='mcml', split='train', source='mcml.v', grid=[64, 64])])
        write(old/'campaign.json', config)
        write(old/'protocol.json', dict(config=config, binary_sha256=binary_hash,
              backend_manifest_sha256=sha(backend/'backend_manifest.json')))
        write(old/'binary_identity.json', dict(binary_sha256=binary_hash))
        records = []
        for original in config['designs']:
            record = dict(original)
            folder = old/'inputs'/record['design']
            source = root/(record['design']+'.v')
            source.write_text('module '+record['design']+'; endmodule')
            write(folder/'source.json', dict(source=str(source), sha256=sha(source),
                  family=record['family'], split=record['split'], fixed_grid=record['grid']))
            blif = folder/'design.blif'
            blif.write_text('.model '+record['design']+'\n.end\n')
            prefix = folder/'structure'
            suffixes = ('.blocks.tsv', '.nets.tsv', '.meta.tsv')
            for suffix in suffixes:
                Path(str(prefix)+suffix).write_text(suffix)
            record.update(blif=str(blif), mapped_blif_sha256=sha(blif), structure_prefix=str(prefix),
                          snapshot_sha256={s: sha(str(prefix)+s) for s in suffixes}, prepared=True)
            records.append(record)
        write(old/'prepared.json', dict(complete=True, records=records))
        cases, seeds = ['aligned', 'shifted', 'fixed_mixed', 'sha'], [701, 702]
        write(old/'qualification/protocol.json', dict(commit=COMMIT, binary_sha256=binary_hash, cases=cases, seeds=seeds))
        write(old/'qualification/qualification_report.json', dict(complete=True, passed=True, harness_version=2,
              conditions=[dict(case=c, seed=s, passed=True) for c in cases for s in seeds]))
        mcml_raw = old/'inputs/mcml/mapping/attempts/attempt_0001/flow/mcml.abc.blif'
        mcml_raw.parent.mkdir(parents=True)
        mcml_raw.write_text('.model raw_mcml\n.end\n')
        write(old/'inputs/mcml/mapping/manifest.json', dict(status='complete',workdir=str(mcml_raw.parent.parent)))
        write(old/'inputs/mcml/mapping_adapter.json', dict(source=str(mcml_raw), source_sha256=sha(mcml_raw),
              derived_sha256=records[1]['mapped_blif_sha256']))
        arch = old/'architectures/16x16/H0Z0/arch.xml'
        arch.parent.mkdir(parents=True)
        arch.write_text('<architecture/>\n')
        rows = []
        for width in (32, 64):
            for h in range(2):
                for z in range(2):
                    for seed in (501, 502):
                        condition = f'H{h}Z{z}'
                        folder = old/f'calibration/sha/{condition}/w{width}/s{seed}'
                        status = 'unroutable' if width == 32 else 'success'
                        manifest = folder/'vpr/manifest.json'
                        command = [str(backend/'vpr/vpr'), str(arch), records[0]['blif'],
                                   '--route_chan_width', str(width), '--seed', str(seed)]
                        signature = dict(command=command, timeout=600, inputs={str(arch):sha(arch),
                            records[0]['blif']:records[0]['mapped_blif_sha256'],
                            str(old/'binary_identity.json'):sha(old/'binary_identity.json')},
                            metadata=dict(condition_id=f'sha/{condition}/w{width}',method='default',seed=seed,checkpoint_sha256=None))
                        write(manifest, dict(status=status,returncode=2 if width == 32 else 0,signature=signature))
                        rows.append(dict(design='sha',family='sha',split='train',architecture=condition,
                            condition_id=f'sha/{condition}/w{width}',width=width,seed=seed,method='default',
                            status=status,vpr_manifest=str(manifest)))
        boundary = select_widths(rows, config['width_grid'])
        boundary.update(design='sha',family='sha',split='train',record=records[0])
        write(old/'calibration/sha/boundary.json', boundary)
        write(old/'calibration/sha/trials.json', rows)
        for h in range(2):
            for z in range(2):
                for seed in (501, 502):
                    folder = old/f'calibration/mcml/H{h}Z{z}/w32/s{seed}/vpr'
                    log = folder/'attempts/attempt_0001/vpr.out'
                    log.parent.mkdir(parents=True)
                    log.write_text('Message: Failed to find device which satisfies resource requirements required: '
                        'io: 392, clb: 5502, dsp_top: 102, memory: 294 '
                        '(available io: 3968, clb: 7192, dsp_top: 248, memory: 248)\n')
                    write(folder/'manifest.json', dict(status='packing_or_placement_failure',returncode=1,log=str(log)))
        device = old/'device_cache/shared'
        device.mkdir(parents=True)
        (device/'rr_graph.xml').write_text('read-only-rrg')
        (device/'device_graph.json').write_text('{}')
        write(device/'manifest.json', dict(binary_sha256=binary_hash,architecture_sha256=sha(arch),
              channel_width=64,rr_graph_sha256=sha(device/'rr_graph.xml'),graph_sha256=sha(device/'device_graph.json')))
        return repo, old, new, backend, arch, mcml_raw

    def do_recover(self, old, new, backend, repo):
        with patch.object(recovery, 'compatible_implementation'), patch.object(recovery.install_hook, 'verify', return_value='verified-hook'):
            return recovery.recover(old, new, backend, repo=repo)

    def campaign(self, new, backend, arch):
        return SimpleNamespace(config=read(new/'campaign.json'),out=new,binary_hash=sha(backend/'vpr/vpr'),
                               args=SimpleNamespace(timeout=600),architecture=lambda record, condition:arch)

    def test_exact_grid_margin_and_resource_error_refuses_noncapacity_messages(self):
        required = dict(io=392,clb=5502,dsp_top=102,memory=294)
        available = dict(io=3968,clb=7192,dsp_top=248,memory=248)
        grid, capacity = recovery.choose_grid(required, [64,64], available)
        self.assertEqual(grid, [84,84])
        self.assertEqual(capacity['memory'],328)
        self.assertLessEqual(required['memory']/capacity['memory'], .9)
        self.assertGreater(required['memory']/(4*(80-2)), .9)
        with self.assertRaisesRegex(ValueError,'explicit VPR'):
            recovery.resource_error('routing failed from congestion')

    def test_registered_old_campaign_and_current_unchanged_runtime_are_compatible(self):
        repo = HERE.parents[2]
        inherited = repo/'experiments/vtr/tailrisk_core.py'
        for commit, digest in recovery.APPROVED_CAMPAIGNS.items():
            with self.subTest(commit=commit):
                baseline = subprocess.check_output(['git','-C',str(repo),'show',
                    commit+':experiments/vtr/tailcompile_v7_3d/campaign.py'])
                self.assertEqual(hashlib.sha256(baseline).hexdigest(),digest)
                protocol = dict(implementation={'campaign.py':digest,
                                **{name:sha(HERE/name) for name in recovery.CORE}},
                                inherited_implementation={str(inherited):sha(inherited)})
                recovery.compatible_implementation(protocol)

    def test_recovery_reuses_only_unchanged_evidence_and_keeps_all_originals_readonly(self):
        with self.temporary() as tmp:
            repo, old, new, backend, arch, raw = self.fixture(Path(tmp).resolve())
            before = {str(p):sha(p) for base in (old,backend) for p in base.rglob('*') if p.is_file()}
            catalog = self.do_recover(old,new,backend,repo)
            self.assertEqual(catalog['correction']['grid'],[84,84])
            self.assertEqual(list(catalog['calibration']),['sha'])
            config = read(new/'campaign.json')
            self.assertEqual(config['designs'][1]['mapped_blif'],str(raw))
            self.assertFalse((new/'protocol.json').exists())
            campaign = self.campaign(new,backend,arch)
            with patch.object(recovery,'compatible_implementation'):
                record = recovery.reused_prepared(campaign,config['designs'][0])
                self.assertEqual(record['blif'],str(old/'inputs/sha/design.blif'))
                self.assertIsNone(recovery.reused_prepared(campaign,config['designs'][1]))
                boundary,rows = recovery.reused_calibration(campaign,record)
                self.assertEqual((boundary['W_stress'],boundary['W_ref']),(32,64))
                self.assertEqual(len(rows),16)
                self.assertTrue(all(str(old) in r['vpr_manifest'] for r in rows))
                self.assertEqual(recovery.reused_device_folder(campaign,'shared',arch,64),old/'device_cache/shared')
            self.assertEqual(before,{p:sha(p) for p in before})

    def test_changed_old_artifact_and_changed_timeout_cannot_be_reused(self):
        with self.temporary() as tmp:
            repo, old, new, backend, arch, _ = self.fixture(Path(tmp).resolve())
            self.do_recover(old,new,backend,repo)
            campaign = self.campaign(new,backend,arch)
            with patch.object(recovery,'compatible_implementation'):
                record = recovery.reused_prepared(campaign,campaign.config['designs'][0])
                campaign.args.timeout=100
                with self.assertRaisesRegex(ValueError,'same timeout'):
                    recovery.reused_calibration(campaign,record)
                campaign.args.timeout=600
                (old/'calibration/sha/trials.json').write_text('[]')
                with self.assertRaisesRegex(ValueError,'artifact changed'):
                    recovery.reused_calibration(campaign,record)

    def test_missing_failure_panel_and_unregistered_implementation_reject_before_writes(self):
        with self.temporary() as tmp:
            repo, old, new, backend, _, _ = self.fixture(Path(tmp).resolve())
            (old/'calibration/mcml/H1Z1/w32/s502/vpr/manifest.json').unlink()
            with self.assertRaisesRegex(ValueError,'all eight'):
                self.do_recover(old,new,backend,repo)
            self.assertFalse(new.exists())
            with self.assertRaisesRegex(ValueError,'unregistered'):
                recovery.compatible_implementation(dict(implementation={'campaign.py':'unknown'}))

    def test_incomplete_unchanged_calibration_is_not_imported(self):
        with self.temporary() as tmp:
            repo, old, new, backend, arch, _ = self.fixture(Path(tmp).resolve())
            (old/'calibration/sha/boundary.json').unlink()
            catalog=self.do_recover(old,new,backend,repo)
            self.assertEqual(catalog['calibration'],{})
            campaign=self.campaign(new,backend,arch)
            with patch.object(recovery,'compatible_implementation'):
                self.assertIsNone(recovery.reused_calibration(campaign,catalog['old_records']['sha']))

    @unittest.skipUnless(os.name=='posix','directory symlink contract')
    def test_output_symlink_escape_is_rejected_without_external_writes(self):
        with self.temporary() as tmp:
            root=Path(tmp).resolve()
            repo, old, new, backend, _, _=self.fixture(root)
            outside=root/'outside'
            outside.mkdir()
            new.mkdir()
            (new/'qualification').symlink_to(outside,target_is_directory=True)
            with self.assertRaisesRegex(ValueError,'symlink'):
                self.do_recover(old,new,backend,repo)
            self.assertEqual(list(outside.iterdir()),[])
            self.assertFalse((new/'capacity_recovery_catalog.json').exists())


if __name__=='__main__':
    unittest.main()
