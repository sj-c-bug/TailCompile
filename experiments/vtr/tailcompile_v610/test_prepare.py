from pathlib import Path
import pickle
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import prepare
from bridge import read,sha,write


class PreparationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.a = SimpleNamespace(out=self.root,identity={'architecture_sha256':'arch','vtr_commit':'commit'},
            calibration_seeds=[401,402],seeds=[411,412,413,414,415],width_start=48,width_max=128,
            width_step=4,half_band=3,timeout=7200,tile_size=4)
        self.record = dict(design='toy',family='toy',source='toy.v',source_sha256='source',
            include_sha256=None,device_width=20,device_height=20,mapped_blif_sha256='blif',
            original_split='new',checkpoint_exposure='no_prior_exposure_recorded',inventory_origin='koios')

    def graph(self,width=80):
        folder = self.root/'graphs'/str(width)
        folder.mkdir(parents=True,exist_ok=True)
        context = dict(self.record,width=width,widths=[width])
        for key in prepare.GRAPH_KEYS:
            path=folder/key
            path.write_text(key)
            context[key]=str(path)
        context['audit']=str(folder/'audit.json')
        audit = dict(signature={'tile_size':4,'channel_width':width,'architecture_sha256':'arch',
            'design':'toy','mapped_blif_sha256':'blif'},
            output_sha256={key:sha(context[key]) for key in prepare.GRAPH_KEYS},
            movable_coordinates_in_features=False)
        write(context['audit'],audit)
        return context

    def test_bounded_grid_and_censoring(self):
        self.assertEqual(prepare.boundary_grid(80),[68,72,76,80,84,88,92,104])
        self.assertEqual(prepare.boundary_grid(128,maximum=128),[116,120,124,128])
        self.assertEqual(prepare.boundary_grid(4),[4,8,12,16,28])
        for center in (0,-4,81):
            with self.assertRaises(ValueError): prepare.boundary_grid(center)

    def test_cached_graph_survives_raw_rrg_reclamation_and_checks_integrity(self):
        context=self.graph()
        self.assertTrue(prepare.verify_graph_outputs(context,tile_size=4,architecture_sha256='arch',mapped_sha256='blif'))
        Path(context['graph']).write_text('changed')
        with self.assertRaisesRegex(ValueError,'Corrupted'):
            prepare.verify_graph_outputs(context,tile_size=4,architecture_sha256='arch')
        Path(context['graph']).unlink()
        self.assertFalse(prepare.verify_graph_outputs(context,tile_size=4,architecture_sha256='arch'))

    def test_teacher_features_and_wrong_width_identity_are_rejected(self):
        context=self.graph()
        audit=read(context['audit']);audit['movable_coordinates_in_features']=True
        write(context['audit'],audit)
        with self.assertRaisesRegex(ValueError,'teacher'):
            prepare.verify_graph_outputs(context,tile_size=4,architecture_sha256='arch')
        audit['movable_coordinates_in_features']=False
        audit['signature']['channel_width']=84
        write(context['audit'],audit)
        with self.assertRaisesRegex(ValueError,'identity'):
            prepare.verify_graph_outputs(context,tile_size=4,architecture_sha256='arch')

    def test_default_calibration_is_frozen_and_resume_does_not_rerun(self):
        calls=[]
        def route(a,record,width,seed,folder):
            calls.append((width,seed))
            return {'status':'success' if width >= (80 if seed==401 else 84) else 'unroutable'}
        with patch.object(prepare.e,'route',side_effect=route),patch.object(prepare,'prune_trial') as prune:
            result=prepare.calibrate(self.a,self.record)
            self.assertEqual(result['reference_width'],84)
            self.assertEqual(result['calibration_status'],'reference_found')
            self.assertEqual(prune.call_count,len(calls))
            count=len(calls)
            self.assertEqual(prepare.calibrate(self.a,self.record),result)
            self.assertEqual(len(calls),count)
        self.assertTrue(all(seed in (401,402) for _,seed in calls))

    def test_right_censored_design_retained_with_maximum_width_band(self):
        with patch.object(prepare.e,'route',return_value={'status':'unroutable'}),patch.object(prepare,'prune_trial'):
            result=prepare.calibrate(self.a,self.record)
        self.assertEqual(result['calibration_status'],'right_censored')
        self.assertFalse(result['reference_is_success'])
        self.assertEqual(result['widths'],[116,120,124,128])
        self.assertTrue(all(r['lowest_success'] is None for r in result['calibration']))

    def test_timeout_remains_failure_and_exception_is_pickleable(self):
        with patch.object(prepare.e,'route',return_value={'status':'timeout'}),patch.object(prepare,'prune_trial'):
            with self.assertRaises(prepare.PreparationFailure) as caught:
                prepare.calibrate(self.a,self.record)
        self.assertEqual(caught.exception.status,'timeout')
        self.assertFalse((self.root/'screen/toy/boundary.json').exists())
        restored=pickle.loads(pickle.dumps(caught.exception))
        self.assertEqual(restored.status,'timeout')

    def test_shared_rrg_deleted_only_with_complete_validated_consumer_proof(self):
        context=self.graph()
        folder=self.root/'device_cache/20x20/w80'
        folder.mkdir(parents=True)
        raw=folder/'rr_graph.xml.gz';raw.write_bytes(b'fixture compressed XML')
        write(folder/'rr_graph.json',dict(sha256=sha(raw),identity=self.a.identity))
        build=folder/'build';build.mkdir()
        write(build/'trial.json',dict(status='success'))
        packed=build/'toy.net';packed.write_text('<packed-net/>')
        log=build/'vpr.out';log.write_text('VPR succeeded')
        import storage
        with patch.object(storage,'tracked_files',return_value=set()):
            prepare._prune_rrg(self.a,folder,[context])
            self.assertFalse(raw.exists())
            self.assertFalse(packed.exists())
            self.assertTrue(log.exists())
            prepare._prune_rrg(self.a,folder,[context])
        self.assertEqual(read(folder/'rrg_pruned.json')['consumers'],1)
        self.assertTrue((folder/'consumers_complete.json').is_file())

    def test_invalid_consumer_prevents_rrg_reclamation(self):
        context=self.graph()
        folder=self.root/'device_cache/20x20/w80';folder.mkdir(parents=True)
        raw=folder/'rr_graph.xml.gz';raw.write_bytes(b'rrg')
        write(folder/'rr_graph.json',dict(sha256=sha(raw),identity=self.a.identity))
        Path(context['device']).unlink()
        with self.assertRaisesRegex(ValueError,'incomplete consumer'):
            prepare._prune_rrg(self.a,folder,[context])
        self.assertTrue(raw.exists())

    def test_existing_graph_reuse_validates_source_and_does_not_read_teachers(self):
        context=self.graph()
        source=self.root/'old_input';source.mkdir()
        blif=source/'toy.pre-vpr.blif';blif.write_text('.model toy\n.end\n')
        context['mapped_blif']=str(blif);context['mapped_blif_sha256']=sha(blif)
        write(source/'source.json',{k:self.record[k] for k in ('design','source','source_sha256','include_sha256')})
        audit=read(context['audit']);audit['signature']['mapped_blif_sha256']=sha(blif)
        write(context['audit'],audit)
        result=prepare.reusable_contexts(self.record,[context],self.a)
        self.assertEqual(result[0]['context_origin'],'verified_v68_deployment_cache')
        self.assertEqual(result[0]['split'],'sweep')
        modified=dict(self.record,source_sha256='different')
        self.assertIsNone(prepare.reusable_contexts(modified,[context],self.a))
        blif.write_text('corrupted')
        with self.assertRaisesRegex(ValueError,'BLIF changed'):
            prepare.reusable_contexts(self.record,[context],self.a)

    def test_qualification_failure_does_not_prune_a_partial_shared_group(self):
        repo=self.root/'repo'
        self.a.out=repo/'experiments/vtr/tailcompile_v610/runs/test'
        self.a.source=repo/'old'
        self.a.vtr=repo/'vtr'
        self.a.jobs=self.a.prepare_jobs=self.a.rr_jobs=1
        row=dict(self.record,input_status='available',missing_inputs=[])
        with patch.object(prepare,'REPO',repo), \
             patch.object(prepare,'discover',return_value={'designs':[row]}), \
             patch.object(prepare.e,'input_structure',return_value=row), \
             patch.object(prepare.e,'route',return_value={'status':'timeout'}), \
             patch.object(prepare,'prune_trial'), \
             patch.object(prepare.cf,'ProcessPoolExecutor') as pool:
            with self.assertRaisesRegex(RuntimeError,'qualification incomplete'):
                prepare.prepare(self.a)
        pool.assert_not_called()
        self.assertEqual(read(self.a.out/'preparation_report.json')['failures'][0]['status'],'timeout')
        self.assertFalse((self.a.out/'contexts.json').exists())

    def test_completed_graph_group_resume_does_not_recreate_rrg(self):
        context=self.graph()
        with patch.object(prepare,'_cached_graph',return_value=context), \
             patch.dict(sys.modules,torch=SimpleNamespace(set_num_threads=lambda n:None)), \
             patch.object(prepare.e,'rr_graph') as rrg:
            result=prepare.graph_group_worker((self.a,[self.record],80))
        self.assertEqual(result,[context])
        rrg.assert_not_called()


if __name__ == '__main__':
    unittest.main()
