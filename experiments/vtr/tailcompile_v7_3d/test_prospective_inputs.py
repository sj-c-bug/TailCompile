"""014 prospective registration, actual-RRG and common-width qualification contracts."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import prospective_inputs as inputs
from io_utils import read, sha, write


def audit(arch):
    h, z = int(arch[1]), int(arch[3])
    return dict(grid=[16,16,2], total_resource_capacity=dict(memory=20,dsp_top=20),
        layer_resource_capacity=[dict(memory=10-5*h,dsp_top=10+5*h),dict(memory=10+5*h,dsp_top=10-5*h)],
        z_capacity_by_class={'Z':4}, z_switch_classes={'S':4}, z_actual_adjacent_switch_classes={'S':4},
        z_mean_squared_distance_to_center=10-z, z_capacity_by_xy={'place'+str(z):4})


class Fixture:
    """Real audit and pinning with deterministic stand-ins for actual P&R."""
    def __init__(self, root):
        self.out=root
        self.inputs=[dict(design=n, family=n, grid=[16,16], split='validation') for n in ('alpha','beta')]
        write(root/'prepared_inputs_manifest.json',dict(files={}))
        write(root/'registered_panel.json',dict(designs=self.inputs))
        self.calls=[]
        self.outcome=lambda c:'unroutable' if c['design']=='alpha' and c['width']==128 and c['architecture']=='H0Z1' else 'success'

    def records(self): return self.inputs
    def gate(self): return True
    def parallel(self, tasks, fn): return [fn(t) for t in reversed(tasks)]
    condition=inputs.Campaign.condition

    def rr(self, record, arch, width):
        folder=self.out/'device_cache'/arch/f'w{width}'
        graph, manifest=folder/'device.json',folder/'manifest.json'
        write(graph, dict(channel_width=width,audit=audit(arch)))
        write(manifest, dict(graph_sha256=sha(graph)))
        return dict(device=str(graph),manifest=str(manifest))

    def run(self, campaign, condition, seed, target, method):
        assert campaign is self and method=='Default'
        check=self.out/'calibration'/condition['design']/f'w{condition["width"]}'/'actual_audit.json'
        assert read(check)['four_cell_matches']
        status=self.outcome(condition)
        self.calls.append((condition['design'],condition['width'],condition['architecture'],seed))
        path=target/'vpr/manifest.json'
        write(path,dict(status=status,condition_id=condition['id'],seed=seed))
        return dict(design=condition['design'],architecture=condition['architecture'],width=condition['width'],
                    seed=seed,status=status,vpr_manifest=str(path))


class ProspectiveInputTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)

    def registration(self):
        designs=[]
        for name in ('fir','alpha'):
            p=self.root/f'vtr_flow/benchmarks/arithmetic/{name}.v'
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text('module '+name+'; endmodule')
            designs.append(dict(design=name,family=name,source=p.relative_to(self.root).as_posix(),
                source_sha256=sha(p),grid=[16,16],lineage='independently reviewed '+name,aliases=[name]))
        panel=self.root/'panel.json'
        write(panel,dict(schema='tailcompile-014-source-panel-1',designs=designs))
        return panel

    def test_register_absolute_paths_and_pin_manifest_without_outputs(self):
        panel=self.registration()
        with patch.object(inputs,'historical_exposure',return_value=(set(),set(),{'history':'hash'})):
            value=inputs.register_panel(panel,self.root)
        self.assertEqual(value['manifest_sha256'],sha(panel))
        for record in value['designs']:
            self.assertEqual(record['source_sha256'],sha(record['source']))
            self.assertEqual(Path(record['source']),self.root/record['source_relative'])
        self.assertFalse((self.root/'runs').exists())

    def test_prior_alias_and_source_bytes_excluded_independently(self):
        panel=self.registration()
        for prior in [({'fir'},set(),{}), (set(),{read(panel)['designs'][1]['source_sha256']},{})]:
            with patch.object(inputs,'historical_exposure',return_value=prior), self.assertRaisesRegex(ValueError,'exposed'):
                inputs.register_panel(panel,self.root)

    def test_mutated_source_or_duplicate_family_rejected(self):
        panel=self.registration()
        before=read(panel)
        with patch.object(inputs,'historical_exposure',return_value=(set(),set(),{})):
            for key,value,error in [('source_sha256','d'*64,'bytes changed'),('family','fir','duplicated'),
                                    ('source','../elsewhere.v','must be relative')]:
                modified=copy.deepcopy(before)
                modified['designs'][1][key]=value
                write(panel,modified)
                with self.assertRaisesRegex(ValueError,error):
                    inputs.register_panel(panel,self.root)

    def test_both_new_mapping_stages_checked_against_old_digest_and_other_family(self):
        history = self.root/'old.json'
        write(history,dict(contexts=[dict(design='old',mapped_blif_sha256='a'*64)]*2))
        inventory=self.root/'experiments/vtr/tailcompile_v610/runs/koios_sweep_002/inventory.json'
        write(inventory,dict(designs=[dict(design='old'),dict(design='unmapped')]))
        new=dict(x=dict(before_adapter='b'*64,deployed_blif='c'*64),y=dict(before_adapter='d'*64,deployed_blif='e'*64))
        with patch.object(inputs,'MAPPED_LEDGER',{'old.json':'contexts'}):
            result=inputs.audit_mapped_blifs(new,self.root)
            self.assertTrue(result['passed'])
            self.assertEqual(result['distinct_historical_mapped_sha256'],1)
            self.assertEqual(result['sources']['old.json']['rows'],2)
            self.assertEqual(result['v610_inventory_without_digest_in_checked_ledgers'],['unmapped'])
            for field,value in [('before_adapter','a'*64),('deployed_blif','a'*64),('deployed_blif','b'*64)]:
                changed=copy.deepcopy(new);changed['y'][field]=value
                self.assertFalse(inputs.audit_mapped_blifs(changed,self.root)['passed'])

    def test_missing_complete_stage_output_preserves_attempt_and_stops_after_second(self):
        folder=self.root/'build';work=folder/'attempts/attempt_0001';work.mkdir(parents=True)
        src=self.root/'input.v';src.write_text('pinned')
        value=dict(workdir=str(work),status='complete',signature=dict(inputs={str(src):sha(src)}))
        write(folder/'manifest.json',value)
        output=folder/'rr.xml'
        output.write_text('retained');inputs.rearm_missing(folder,lambda w:[output])
        self.assertTrue((folder/'manifest.json').is_file())
        output.unlink();inputs.rearm_missing(folder,lambda w:[output])
        self.assertEqual(read(work/'incomplete_payload_manifest.json'),value)
        self.assertFalse((folder/'manifest.json').exists())
        next_work=work.parent/'attempt_0002';next_work.mkdir()
        write(folder/'manifest.json',dict(value,workdir=str(next_work)))
        with self.assertRaisesRegex(RuntimeError,'retry exhausted'):
            inputs.rearm_missing(folder,lambda w:[output])

    def test_mapping_cleanup_removes_only_identical_staged_copies(self):
        folder=self.root/'input';work=folder/'mapping/attempts/attempt_0001'
        flow=work/'flow';flow.mkdir(parents=True)
        source=self.root/'foo.v';source.write_text('keep source')
        arch=self.root/'arch.xml';arch.write_text('keep architecture')
        (flow/'foo.v').write_bytes(source.read_bytes())
        (flow/'arch.xml').write_text('different, retain')
        raw=flow/'foo.abc.blif';raw.write_text('keep mapped BLIF')
        blif=folder/'design.blif';blif.write_text('keep adapted BLIF')
        write(folder/'mapping/manifest.json',dict(workdir=str(work)))
        write(folder/'mapping_adapter.json',dict(source=str(raw),architecture=str(arch)))
        row=dict(source=str(source),blif=str(blif))
        for _ in range(2):
            inputs.cleanup_mapping_copies(folder,row)
        self.assertFalse((flow/'foo.v').exists())
        self.assertTrue(all(p.is_file() for p in [source,arch,raw,blif,flow/'arch.xml']))
        self.assertEqual(read(folder/'payload_pruned.json')['bytes'],source.stat().st_size)

    def test_balanced_actual_supply_and_fixed_vertical_pool(self):
        graphs={a:[dict(channel_width=w,audit=audit(a)) for w in (128,192)] for a in inputs.ARCHES}
        self.assertTrue(inputs.check_actual(graphs)['four_cell_matches'])
        for a in inputs.ARCHES:
            graphs[a][1]['audit']['z_capacity_by_class']['Z']=5
        with self.assertRaisesRegex(ValueError,'vertical capacity changed'):
            inputs.check_actual(graphs)
        same={a:[dict(channel_width=128,audit=audit('H0'+a[2:]))] for a in inputs.ARCHES}
        with self.assertRaisesRegex(ValueError,'per-layer resource'):
            inputs.check_actual(same)

    def test_select_first_all_four_successes_per_family_and_verified_resume(self):
        engine=Fixture(self.root)
        with patch('prospective_native.run_native',side_effect=engine.run):
            result=inputs.calibrate_inputs(engine)
            self.assertEqual(len(result),8)
            self.assertEqual({(r['design'],r['width']) for r in result},{('alpha',192),('beta',128)})
            self.assertEqual(len(engine.calls),12)
            self.assertEqual(inputs.calibrate_inputs(engine),result)
            self.assertEqual(len(engine.calls),12)
            path=next((self.root/'device_cache').rglob('manifest.json'));path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'input changed'):
                inputs.qualified_conditions(engine)

    def test_no_qualified_panel_on_exhaustion(self):
        engine=Fixture(self.root)
        engine.outcome=lambda c:'unroutable'
        with patch('prospective_native.run_native',side_effect=engine.run):
            with self.assertRaisesRegex(RuntimeError,'qualification failed'):
                inputs.calibrate_inputs(engine)
            report=read(self.root/'input_qualification_report.json')
            self.assertEqual(report['native_count'],24)
            self.assertFalse(report['passed'])
            self.assertFalse((self.root/'conditions_manifest.json').exists())
            with self.assertRaisesRegex(RuntimeError,'already failed'):
                inputs.calibrate_inputs(engine)
            self.assertEqual(len(engine.calls),24)


if __name__=='__main__': unittest.main()
