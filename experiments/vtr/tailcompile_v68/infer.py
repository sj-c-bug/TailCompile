"""Deploy the actual hierarchical initializer, with measured component costs."""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

os.environ.setdefault('KMP_DUPLICATE_LIB_OK','TRUE')
import torch

from common import emit, freeze, read, sha, write
from model import HierarchicalInitializer, SCHEMA, to_device


class Generator:
    def __init__(self, checkpoint, device='cpu'):
        started = time.perf_counter()
        self.device = torch.device(device)
        if self.device.type=='cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA was requested but is unavailable')
        self.checkpoint = Path(checkpoint).resolve()
        self.checkpoint_sha256 = sha(self.checkpoint)
        self.saved = torch.load(self.checkpoint,map_location=self.device,weights_only=True)
        if self.saved.get('schema') != SCHEMA or 'model_config' not in self.saved:
            raise ValueError('v6.8 needs its trained hierarchical GENERATOR, not a v3 generator/v6 scorer')
        self.model = HierarchicalInitializer(**self.saved['model_config']).to(self.device)
        self.model.load_state_dict(self.saved['model_state']);self.model.eval()
        self.device_cache = {}
        self.logic_cache = None
        self.graph_cache = None
        self.synchronize()
        self.load_seconds = time.perf_counter()-started

    def synchronize(self):
        if self.device.type=='cuda':
            torch.cuda.synchronize(self.device)

    def measured(self,function):
        self.synchronize();started=time.perf_counter();value=function();self.synchronize()
        return value,time.perf_counter()-started

    def generate(self, context, folder, *, candidate=0, seed=0,
                 score_noise_std=0., coordinate_noise_std=0.):
        generation_started=time.perf_counter()
        folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
        signature=dict(schema=SCHEMA,checkpoint_sha256=self.checkpoint_sha256,
            graph_sha256=sha(context['graph']),logic_sha256=sha(context['logic']),
            device_sha256=sha(context['device']),supply_sha256=sha(context['supply']),
            candidate=candidate,seed=seed,score_noise_std=score_noise_std,
            coordinate_noise_std=coordinate_noise_std,
            implementation={p.name:sha(p) for p in [Path(__file__),Path(__file__).with_name('model.py')]})
        manifest=folder/'generation.json'
        if manifest.exists():
            old=read(manifest)
            if (old['signature']!=signature or sha(folder/'init.ap.tsv')!=old['ap_init_sha256'] or
                sha(folder/'init.fplace')!=old['fplace_sha256']):
                raise ValueError(f'Generated initializer provenance changed: {folder}')
            return old
        timing={'model_load':self.load_seconds}
        if self.graph_cache is not None and self.graph_cache[0]==signature['graph_sha256']:
            data=self.graph_cache[1];timing['graph_load']=0.;timing['tensor_transfer']=0.
        else:
            data,timing['graph_load']=self.measured(lambda:torch.load(context['graph'],map_location='cpu',weights_only=True))
            data,timing['tensor_transfer']=self.measured(lambda:to_device(data,self.device))
            self.graph_cache=(signature['graph_sha256'],data)
        cache_key=(signature['device_sha256'],signature['supply_sha256'],self.checkpoint_sha256)
        with torch.no_grad():
            hit=cache_key in self.device_cache
            if not hit:
                encoded,timing['rrg_encoding']=self.measured(lambda:self.model.encode_device(data))
                self.device_cache[cache_key]=encoded
            else:
                timing['rrg_encoding']=0.
            if self.logic_cache is None or self.logic_cache[0]!=signature['graph_sha256']:
                logic_embedding,timing['logic_forward']=self.measured(lambda:self.model.encode_logic(data))
                self.logic_cache=(signature['graph_sha256'],logic_embedding)
            else:
                timing['logic_forward']=0.
            prediction,timing['logic_matching_decode']=self.measured(lambda:self.model.predict(data,
                device_embedding=self.device_cache[cache_key],logic_embedding=self.logic_cache[1],score_noise_std=score_noise_std,
                coordinate_noise_std=coordinate_noise_std,seed=seed))
        logic,physical=read(context['logic']),read(context['device'])
        target=dict(teacher_region=prediction['assignment'].cpu().tolist(),
                    teacher_xy=prediction['xy'].detach().cpu().tolist(),
                    movable=data['movable'].cpu().tolist())
        def lower():
            rows=[];flat_rows=[];observed=set()
            for index,block in enumerate(logic['clusters']):
                if not target['movable'][index]:
                    continue
                x,y=target['teacher_xy'][index]
                if not all(math.isfinite(v) and 0<=v<=1 for v in (x,y)):
                    raise ValueError('Initializer contains nonfinite/out-of-grid coordinates')
                # Native AP coordinates are [0,W), not [0,W-1], and are not
                # necessarily half-tile centers. Keep exact continuous priors
                # in a separate file instead of silently applying a .5 shift.
                x=min(x*physical['grid_width'],math.nextafter(float(physical['grid_width']),0.))
                y=min(y*physical['grid_height'],math.nextafter(float(physical['grid_height']),0.))
                for atom in block['atoms']:
                    if atom in observed or any(c.isspace() for c in atom):
                        raise ValueError('Duplicate or whitespace-containing AP atom name')
                    observed.add(atom)
                    rows.append(f'{atom}\t{x:.17g}\t{y:.17g}\t0\t0\n')
                    flat_rows.append(f'{atom}\t{math.floor(x)}\t{math.floor(y)}\t0\t0\n')
            native=folder/'init.ap.tsv'
            native.write_text('# Exact v6.8 native continuous AP initialization\n'+''.join(rows),encoding='utf-8')
            output=folder/'init.fplace'
            output.write_text('# Diagnostic floor(AP) flat placement; not site-legalized or used to bypass GP\n'+''.join(flat_rows),encoding='utf-8')
            return native,output
        (native,output),timing['fplace_lowering']=self.measured(lower)
        write(folder/'target.json',target)
        warm=sum(timing[k] for k in ('tensor_transfer','logic_forward','logic_matching_decode','fplace_lowering'))
        elapsed=time.perf_counter()-generation_started
        timing['provenance_and_io']=max(0.,elapsed-sum(v for k,v in timing.items() if k!='model_load'))
        cold=elapsed+self.load_seconds
        result=dict(signature=signature,ap_init=str(native.resolve()),ap_init_sha256=sha(native),
            fplace=str(output.resolve()),fplace_sha256=sha(output),
            target=str((folder/'target.json').resolve()),timing=timing,
            warm_inference_lowering_seconds=warm,cold_from_prepared_graph_seconds=cold,
            rrg_cache_hit=hit,leaf_count=len(physical['regions']),
            hierarchy_level_count=len(data['parents']),model_device=str(self.device),
            candidate_region_count=prediction['candidate_region_count'],
            capacity_usage=prediction['usage'].tolist(),regional_capacity_is_proxy=True,
            source_placement_used=False, native_gp_required=True,
            timing_limit='Raw RRG preprocessing and AP structure extraction are separate; not measured here.')
        write(manifest,result)
        emit('generated',design=context['design'],width=context['width'],candidate=candidate,
             leaf_count=result['leaf_count'],hierarchy_levels=result['hierarchy_level_count'],timing=timing)
        return result
