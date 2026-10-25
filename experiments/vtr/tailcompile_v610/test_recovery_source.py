import contextlib
import copy
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import recovery_source as recovery
from bridge import read,write,sha
from prepare import boundary_grid


class RecoverySourceTest(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'old';self.source.mkdir()
        (self.source/'.lock').write_bytes(b'')
        self.child=self.root/'child'
        self.checkpoint=self.root/'checkpoint.pt';self.checkpoint.write_bytes(b'frozen')
        self.binary=self.root/'vpr';self.binary.write_bytes(b'fixed fake binary')
        self.arch=self.root/'architecture.xml';self.arch.write_bytes(b'fixed fake architecture')
        self.identity=dict(vtr_commit='fixed',binary_sha256=sha(self.binary),architecture_sha256=sha(self.arch),hook={'sha':'hook'})
        self.a=SimpleNamespace(recovery_source=self.source,out=self.child,identity=self.identity,
            checkpoint=self.checkpoint,tile_size=4,calibration_seeds=[401,402],
            width_start=48,width_max=128,width_step=4,half_band=3)
        self.row=dict(design='toy',source='koios/toy.v',source_sha256='source',include_sha256='include',
            family='toy',split='sweep',original_split='new',checkpoint_exposure='unknown',inventory_origin='koios')
        self.inventory=dict(designs=[self.row])
        self.protocol=dict(schema=recovery.SOURCE_SCHEMA,training_permitted=False,identity=self.identity,
            checkpoint_sha256=sha(self.checkpoint),calibration_seeds=[401,402],width_start=48,
            width_max=128,width_step=4,half_band=3,timeout=7200)
        write(self.source/'protocol.json',self.protocol)
        write(self.source/'inventory.json',self.inventory)
        write(self.source/'preparation_report.json',dict(failures=[dict(design='other',status='timeout')]))
        folder=self.source/'inputs/toy';folder.mkdir(parents=True)
        self.blif=folder/'toy.pre-vpr.blif';self.blif.write_text('.model toy\n.end\n')
        source={k:self.row[k] for k in recovery.SOURCE_FIELDS}
        write(folder/'source.json',source)
        write(folder/'mapping.json',dict(source,mapped_blif_sha256=sha(self.blif)))
        self.prefix=folder/'structure'
        Path(str(self.prefix)+'.meta.tsv').write_text('20 20\n')
        Path(str(self.prefix)+'.blocks.tsv').write_text('0\ta0\t0\t0\t-1\t-1\t0\t0\t.names\t0\n')
        Path(str(self.prefix)+'.nets.tsv').write_text('')
        self.input=dict(self.row,mapped_blif=str(self.blif),mapped_blif_sha256=sha(self.blif),
                        structure_prefix=str(self.prefix),device_width=20,device_height=20)
        self.input['split']='old_split'
        write(folder/'prepared.json',self.input)
        write(self.source/'prepared_inputs.json',dict(designs=[self.input]))
        command=self.command(48,401)
        identity=dict(self.identity,stage='structure',blif_sha256=sha(self.blif),command=command,
            hook_env={'TC_GP_STRUCTURE':str(self.prefix),'TC_GP_STRUCTURE_ONLY':'1'})
        write(folder/'structure_run/trial.json',dict(identity=identity,command=command,status='complete',returncode=0))
        self.lock_patch=patch.object(recovery,'_source_lock',side_effect=lambda source:contextlib.nullcontext())
        self.lock_patch.start();self.addCleanup(self.lock_patch.stop)

    def command(self,width,seed):
        return [str(self.binary),str(self.arch),str(self.blif),'--analytical_place','--route','--analysis',
            '--ap_full_legalizer','appack','--ap_detailed_placer','annealer','--num_workers','1',
            '--seed',str(seed),'--route_chan_width',str(width),'--device_width','20']

    def observation(self,width,seed,status):
        folder=self.source/f'screen/toy/w{width}/s{seed}'
        command=self.command(width,seed)
        identity=dict(self.identity,stage='route',design='toy',width=width,seed=seed,
            mapped_blif_sha256=sha(self.blif),initializer_sha256=None,teacher=False,
            device_width=20,command=command,hook_env={})
        code=0 if status=='success' else 2 if status=='unroutable' else 124
        trial=dict(identity=identity,command=command,status=status,returncode=code,wall_seconds=12)
        write(folder/'trial.json',trial)
        write(folder/'metrics.json',dict(trial,design='toy',width=width,seed=seed,teacher_prefix=None))
        (folder/'vpr.out').write_text('VPR succeeded' if status=='success' else
            'Circuit is unroutable with a channel width factor' if status=='unroutable' else 'partial log')
        return folder

    def boundary(self):
        observations=[]
        for seed in (401,402):
            for width,status in ((76,'unroutable'),(80,'success')):
                self.observation(width,seed,status)
                observations.append(dict(width=width,seed=seed,status=status))
        value=dict(calibration=[dict(seed=s,lowest_success=80,adjacent_failure=76,right_censored=False) for s in (401,402)],
            reference_width=80,reference_is_success=True,widths=boundary_grid(80),
            calibration_status='reference_found',calibration_observations=observations)
        signature=dict(identity=self.identity,source_sha256='source',mapped_blif_sha256=sha(self.blif),
            device_width=20,seeds=[401,402],start=48,maximum=128,step=4,half_band=3,timeout=7200)
        write(self.source/'screen/toy/boundary.json',dict(signature=signature,boundary=value))
        write(self.source/'boundary_plan.json',dict(designs=[dict(self.input,**value)]))

    def graph(self,width=80,widths=None):
        folder=self.source/'old_graphs'/f'w{width}';folder.mkdir(parents=True)
        context=dict(self.input,width=width,widths=widths or [width],reference_width=80)
        for key in ('graph','logic','device','supply'):
            path=folder/f'{key}.json';path.write_text('{}');context[key]=str(path)
        context['audit']=str(folder/'audit.json')
        write(context['audit'],dict(signature=dict(tile_size=4,channel_width=width,architecture_sha256=sha(self.arch),
            design='toy',mapped_blif_sha256=sha(self.blif)),
            output_sha256={k:sha(context[k]) for k in ('graph','logic','device','supply')},
            movable_coordinates_in_features=False))
        return context

    def test_known_inputs_boundaries_and_observations_are_recovered_read_only(self):
        self.boundary()
        self.observation(48,401,'timeout')
        before={str(p.relative_to(self.source)):sha(p) for p in self.source.rglob('*') if p.is_file()}
        value=recovery.load_recovery(self.a,self.inventory)
        self.assertEqual(len(value['inputs']),1)
        self.assertEqual(value['inputs']['toy']['split'],'sweep')
        self.assertEqual(len(value['observations']),4)
        self.assertEqual(value['boundaries']['toy']['reference_width'],80)
        self.assertNotIn(('toy',48,401),value['observations'])
        self.assertEqual(before,{str(p.relative_to(self.source)):sha(p) for p in self.source.rglob('*') if p.is_file()})
        self.assertTrue((self.child/'recovery_manifest.json').exists())
        self.assertTrue(any(f.get('status')=='timeout' for f in value['failures']))

    def test_complete_graph_grid_reused_and_missing_grid_skipped(self):
        context=self.graph()
        write(self.source/'contexts.partial.json',dict(contexts=[context]))
        first=recovery.load_recovery(self.a,self.inventory)
        self.assertEqual(len(first['contexts']),1)
        Path(context['graph']).unlink()
        self.a.out=self.root/'child_second'
        second=recovery.load_recovery(self.a,self.inventory)
        self.assertEqual(second['contexts'],[])
        self.assertIn('toy',second['inputs'])

    def test_partial_declared_graph_grid_is_skipped_for_rebuild(self):
        context=self.graph(widths=[80,84])
        write(self.source/'contexts.partial.json',dict(contexts=[context]))
        value=recovery.load_recovery(self.a,self.inventory)
        self.assertEqual(value['contexts'],[])
        self.assertIn('toy',value['inputs'])
        self.assertTrue(any(f.get('reason')=='partial_declared_graph_grid' for f in value['failures']))

    def test_existing_graph_or_mapped_input_corruption_is_rejected(self):
        context=self.graph()
        write(self.source/'contexts.partial.json',dict(contexts=[context]))
        Path(context['graph']).write_text('corrupt')
        with self.assertRaisesRegex(ValueError,'Corrupted graph'):
            recovery.load_recovery(self.a,self.inventory)
        write(self.source/'contexts.partial.json',dict(contexts=[]))
        self.blif.write_text('changed')
        with self.assertRaisesRegex(ValueError,'mapped BLIF hash'):
            recovery.load_recovery(self.a,self.inventory)

    def test_backend_checkpoint_and_source_changes_are_rejected(self):
        protocol=copy.deepcopy(self.protocol);protocol['identity']['binary_sha256']='other'
        write(self.source/'protocol.json',protocol)
        with self.assertRaisesRegex(ValueError,'identity differs'):
            recovery.load_recovery(self.a,self.inventory)
        write(self.source/'protocol.json',self.protocol)
        self.checkpoint.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError,'checkpoint differs'):
            recovery.load_recovery(self.a,self.inventory)

    def test_missing_canonical_artifact_skips_but_wrong_dimensions_reject(self):
        Path(str(self.prefix)+'.meta.tsv').write_text('21 20\n')
        with self.assertRaisesRegex(ValueError,'dimensions differ'):
            recovery.load_recovery(self.a,self.inventory)
        Path(str(self.prefix)+'.meta.tsv').unlink()
        value=recovery.load_recovery(self.a,self.inventory)
        self.assertEqual(value['inputs'],{})

    def test_nondefault_observation_is_rejected(self):
        folder=self.observation(80,401,'success')
        trial=read(folder/'trial.json');trial['identity']['initializer_sha256']='gnn'
        write(folder/'trial.json',trial)
        with self.assertRaisesRegex(ValueError,'uninitialized Default'):
            recovery.load_recovery(self.a,self.inventory)

    def test_command_cannot_use_a_different_netlist_with_same_declared_identity(self):
        folder=self.observation(80,401,'success')
        wrong=self.root/'wrong.blif';wrong.write_text('different netlist')
        trial=read(folder/'trial.json')
        trial['command'][2]=str(wrong)
        trial['identity']['command']=trial['command']
        write(folder/'trial.json',trial)
        row=read(folder/'metrics.json');row.update(trial)
        write(folder/'metrics.json',row)
        with self.assertRaisesRegex(ValueError,'command BLIF content differs'):
            recovery.load_recovery(self.a,self.inventory)

    def test_structure_command_cannot_use_another_architecture(self):
        path=self.source/'inputs/toy/structure_run/trial.json'
        wrong=self.root/'wrong.xml';wrong.write_text('another architecture')
        trial=read(path);trial['command'][1]=str(wrong)
        trial['identity']['command']=trial['command']
        write(path,trial)
        with self.assertRaisesRegex(ValueError,'command architecture content differs'):
            recovery.load_recovery(self.a,self.inventory)

    def test_boundary_may_not_claim_a_successful_unknown_width(self):
        self.boundary()
        path=self.source/'screen/toy/boundary.json';value=read(path)
        value['boundary']['reference_width']=84
        write(path,value)
        with self.assertRaisesRegex(ValueError,'grid/reference differs'):
            recovery.load_recovery(self.a,self.inventory)

    def test_fingerprint_is_stable_and_source_mutation_blocks_resume(self):
        one=recovery.fingerprint_source(self.source)
        self.assertEqual(one,recovery.fingerprint_source(self.source))
        recovery.load_recovery(self.a,self.inventory)
        write(self.source/'preparation_report.json',dict(failures=[]))
        with self.assertRaisesRegex(ValueError,'Frozen inputs changed'):
            recovery.load_recovery(self.a,self.inventory)

    def test_incomplete_active_trial_and_same_output_rejected(self):
        write(self.source/'inputs/toy/new/started.json',dict(child_pid=1))
        with self.assertRaisesRegex(ValueError,'Interrupted source'):
            recovery.load_recovery(self.a,self.inventory)
        self.a.out=self.source
        with self.assertRaisesRegex(ValueError,'separate cohort'):
            recovery.load_recovery(self.a,self.inventory)

    @unittest.skipUnless(os.name=='posix','Linux flock')
    def test_read_only_source_lock_rejects_running_pipeline(self):
        self.lock_patch.stop()
        with recovery._source_lock(self.source):
            with self.assertRaisesRegex(RuntimeError,'still running'):
                recovery.fingerprint_source(self.source)


if __name__=='__main__':
    unittest.main()
