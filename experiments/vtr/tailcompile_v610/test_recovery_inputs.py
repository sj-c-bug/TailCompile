from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from bridge import read,sha,write
import recovery_inputs as ri


class RecoveryInputsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        vtr = self.root/'vtr'
        source = vtr/'vtr_flow/benchmarks/verilog/toy.v'
        source.parent.mkdir(parents=True)
        source.write_text('module toy(input a, output b); assign b=a; endmodule\n')
        arch = vtr/'architecture.xml';arch.write_text('<architecture/>')
        self.a = SimpleNamespace(out=self.root/'new_cohort',vtr=vtr,arch=arch,binary=vtr/'vpr',
            identity={'vtr_commit':'frozen','architecture_sha256':sha(arch),'binary_sha256':'binary'},
            timeout=7200,synthesis_timeout=21600,calibration_timeout=14400,
            calibration_seeds=[401,402],width_start=48,retry_failed=True,retry_timeouts=True,heartbeat=60)
        self.record = dict(design='toy',family='toy',source='toy.v',source_sha256=sha(source),
            include_sha256=None,split='sweep',original_split='new',checkpoint_exposure='unseen')
        self.calls=[]

    def write_structure(self,prefix,width=10,leaked=False):
        prefix=Path(prefix);prefix.parent.mkdir(parents=True,exist_ok=True)
        Path(str(prefix)+'.meta.tsv').write_text(f'{width} {width}\n')
        coordinates='1\t1' if leaked else '-1\t-1'
        Path(str(prefix)+'.blocks.tsv').write_text(f'0\ta0\t.names\t0\t{coordinates}\t0\t0\n')
        Path(str(prefix)+'.nets.tsv').write_text('')

    def fake_call(self,a,command,folder,signature,**kwargs):
        self.calls.append((a.timeout,list(map(str,command)),signature,kwargs))
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        if signature['stage']=='synthesis':
            flow=Path(command[command.index('-temp_dir')+1]);flow.mkdir(parents=True,exist_ok=True)
            (flow/'toy.abc.blif').write_text('.model toy\n.inputs a\n.outputs b\n.end\n')
            status='success'
        else:
            width=int(command[command.index('--device_width')+1]) if '--device_width' in command else 10
            self.write_structure(kwargs['env']['TC_GP_STRUCTURE'],width)
            status='complete'
        (folder/'vpr.out').write_text('fixture success\n')
        row=dict(returncode=0,status=status,identity=signature)
        write(folder/'trial.json',row)
        return row

    def imported(self):
        folder=self.root/'old_cohort/inputs/toy';folder.mkdir(parents=True)
        blif=folder/'toy.pre-vpr.blif';blif.write_text('.model toy\n.end\n')
        prefix=folder/'structure';self.write_structure(prefix)
        return dict(self.record,mapped_blif=str(blif),mapped_blif_sha256=sha(blif),
            structure_prefix=str(prefix),device_width=10,device_height=10,
            structure_sha256={s:sha(str(prefix)+s) for s in ri.STRUCTURE_SUFFIXES})

    def test_extended_budgets_preserve_final_timeout_and_resume_mapping(self):
        with patch.object(ri.e,'call',side_effect=self.fake_call):
            result=ri.prepare_input(self.a,self.record)
            resumed=ri.prepare_input(self.a,self.record)
        self.assertEqual(result,resumed)
        self.assertEqual([c[0] for c in self.calls],[21600,14400])
        self.assertEqual(self.a.timeout,7200)
        command=self.calls[0][1]
        self.assertEqual(command[command.index('-timeout')+1],'21600.0')
        self.assertEqual(command[command.index('-ending_stage')+1],'abc')
        self.assertEqual(result['device_width'],10)
        self.assertTrue(Path(result['mapped_blif']).is_relative_to(self.a.out))
        self.assertEqual(self.calls[1][3]['env']['TC_GP_STRUCTURE_ONLY'],1)

    def test_import_references_are_read_only_and_metadata_is_current(self):
        imported=self.imported()
        folder=Path(imported['mapped_blif']).parent
        before={p.name:p.read_bytes() for p in folder.iterdir()}
        inventory=dict(self.record,checkpoint_exposure='checkpoint_training_family')
        with patch.object(ri.e,'call') as call:
            result=ri.prepare_input(self.a,inventory,imported)
        call.assert_not_called()
        self.assertEqual(before,{p.name:p.read_bytes() for p in folder.iterdir()})
        self.assertFalse(self.a.out.exists())
        self.assertEqual(result['mapped_blif'],imported['mapped_blif'])
        self.assertEqual(result['checkpoint_exposure'],'checkpoint_training_family')

    def test_resize_only_exports_new_structure_and_discards_stale_graph_grid(self):
        imported=self.imported()
        original=Path(imported['mapped_blif']).read_bytes()
        stale=dict(imported,graph='old_graph',widths=[48,52],reference_width=52,
                   input_origin='verified_read_only_recovery_input',requested_device_width=None)
        with patch.object(ri.e,'call',side_effect=self.fake_call):
            resized=ri.resize_input(self.a,stale,14)
            resumed=ri.resize_input(self.a,stale,14)
        self.assertEqual(resized,resumed)
        self.assertEqual(len(self.calls),1)
        self.assertEqual(self.calls[0][2]['stage'],'structure')
        self.assertEqual(resized['mapped_blif'],imported['mapped_blif'])
        self.assertEqual(Path(imported['mapped_blif']).read_bytes(),original)
        self.assertEqual(resized['device_width'],14)
        self.assertEqual(resized['device_height'],14)
        self.assertNotIn('graph',resized);self.assertNotIn('widths',resized)
        self.assertTrue(Path(resized['structure_prefix']).is_relative_to(self.a.out/'inputs/toy/structures/w14'))
        with self.assertRaisesRegex(ValueError,'increase'):
            ri.resize_input(self.a,imported,10)

    def test_import_corruption_and_teacher_coordinate_leakage_are_rejected(self):
        imported=self.imported()
        Path(imported['mapped_blif']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'BLIF'):
            ri.prepare_input(self.a,self.record,imported)
        imported['mapped_blif_sha256']=sha(imported['mapped_blif'])
        self.write_structure(imported['structure_prefix'],leaked=True)
        with self.assertRaisesRegex(ValueError,'structure changed'):
            ri.prepare_input(self.a,self.record,imported)
        imported.pop('structure_sha256')
        with self.assertRaisesRegex(ValueError,'teacher leakage'):
            ri.prepare_input(self.a,self.record,imported)

    def test_canonical_mapping_hash_must_match_before_resume(self):
        with patch.object(ri.e,'call',side_effect=self.fake_call):
            result=ri.prepare_input(self.a,self.record)
        Path(result['mapped_blif']).write_text('changed')
        with patch.object(ri.e,'call') as call:
            with self.assertRaisesRegex(ValueError,'BLIF'):
                ri.prepare_input(self.a,self.record)
        call.assert_not_called()

    def test_timeout_status_is_not_relabelled_capacity_or_unroutable(self):
        with patch.object(ri.e,'call',return_value={'status':'timeout','returncode':124}):
            with self.assertRaises(ri.PreparationFailure) as caught:
                ri.prepare_input(self.a,self.record)
        self.assertEqual(caught.exception.status,'timeout')
        self.assertFalse((self.a.out/'inputs/toy/mapping.json').exists())

    def test_only_exact_capacity_errors_trigger_resize(self):
        folder=self.root/'failed';folder.mkdir()
        log=folder/'vpr.out'
        row={'status':'other_failure','returncode':1}
        pack='Error 1:\nType: Other\nMessage: Failed to find device which satisfies resource requirements required: clb: 1, dsp_top: 133 (available clb: 3, dsp_top: 132)\n'
        log.write_text(pack)
        self.assertTrue(ri.is_device_capacity_failure(row,folder))
        with self.assertRaises(ri.PreparationFailure) as caught:
            ri._fail_result(row,folder,'structure','Structure export failed')
        self.assertTrue(caught.exception.device_capacity_failure)
        place='Message: 32 blocks could not be placed during initial placement; no spaces were available for them on the grid.\nIf VPR was run with floorplan constraints, the constraints may be too tight.\n'
        log.write_text(place)
        self.assertTrue(ri.is_device_capacity_failure(row,folder))
        self.assertFalse(ri.is_device_capacity_failure(dict(row,status='timeout',returncode=124),folder))
        self.assertFalse(ri.is_device_capacity_failure(dict(row,status='unroutable'),folder))
        self.assertFalse(ri.is_device_capacity_failure(dict(row,returncode=-9),folder))
        for unrelated in ('Packing failed to fit on device. Re-packing with unrelated_logic_clustering',
                          'Message: SIGSEGV inside packing', 'Created clusters could not fit on device.',
                          place+'Disk quota exceeded'):
            log.write_text(unrelated)
            self.assertFalse(ri.is_device_capacity_failure(row,folder))


if __name__=='__main__':
    unittest.main()
