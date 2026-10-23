"""Fast, resource-compatible AP priors; no claim of final placement legality."""
from __future__ import annotations

import math
import os
from pathlib import Path
import time

from bridge import V68, emit, fingerprint, read, sha, write
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
import torch
from model import SCHEMA, to_device
from fast_model import FastHierarchicalInitializer


def candidate_options(index):
    return dict(seed=690000+index, score_noise_std=.15 if index else 0.,
                coordinate_noise_std=.025 if index else 0.)


def signature(context, checkpoint, options, candidate):
    return dict(schema='tailcompile-v69-fast-generation-1',
        checkpoint_sha256=sha(checkpoint), options=options, candidate=candidate,
        sampling=candidate_options(candidate),
        inputs={key: sha(context[key]) for key in ('graph','logic','device','supply')},
        implementation={str(p.name):sha(p) for p in (Path(__file__),
            Path(__file__).with_name('fast_model.py'), V68/'model.py')})


def validate_generation(folder, expected):
    folder=Path(folder)
    if not (folder/'generation.json').exists():
        return None
    saved=read(folder/'generation.json')
    if saved['signature'] != expected:
        raise ValueError(f'Generation identity changed: {folder}; use a new output directory')
    for key in ('ap_init','fplace','target'):
        if sha(saved[key]) != saved[key+'_sha256']:
            raise ValueError(f'Changed generation artifact: {saved[key]}')
    return saved


def generate(context, checkpoint, folder, options, candidate=0, reuse=True):
    """Fresh model/logic encoding per call: no free cached logic in runtime claims."""
    torch.set_num_threads(1)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    expected=signature(context,checkpoint,options,candidate)
    if reuse:
        old=validate_generation(folder,expected)
        if old is not None:
            return old
    elif (folder/'generation.json').exists():
        raise ValueError('Timed generation requires a fresh measured directory')
    start=time.perf_counter();timing={}
    def measured(key, fn):
        begin=time.perf_counter();value=fn();timing[key]=time.perf_counter()-begin
        return value
    saved=measured('checkpoint_load',lambda:torch.load(checkpoint,map_location='cpu',weights_only=True))
    if saved.get('schema')!=SCHEMA:
        raise ValueError('Expected the v68 hierarchical coordinate generator checkpoint')
    if context['split']=='final_test' and context['family'] in set(
            saved.get('training_families',[])+saved.get('validation_families',[])):
        raise ValueError('Evaluation family leaked into checkpoint training/validation')
    def load_model():
        m=FastHierarchicalInitializer(**saved['model_config'])
        m.load_state_dict(saved['model_state']);m.eval();return m
    model=measured('model_setup',load_model)
    data=measured('graph_load',lambda:torch.load(context['graph'],map_location='cpu',weights_only=True))
    with torch.no_grad():
        device=measured('rrg_encoding',lambda:model.encode_device(data))
        logic=measured('logic_forward',lambda:model.encode_logic(data))
        prediction=measured('matching_decode_coordinate',lambda:model.predict(data,
            device_embedding=device,logic_embedding=logic,**options,**candidate_options(candidate)))
    logical,physical=measured('metadata_load',lambda:(read(context['logic']),read(context['device'])))
    target=dict(teacher_region=prediction['assignment'].tolist(),
        teacher_xy=prediction['xy'].detach().cpu().tolist(),movable=data['movable'].tolist())
    def lower():
        rows=[];flat=[];seen=set()
        if len(logical['clusters'])!=len(target['movable']):
            raise ValueError('Logic/AP graph node count mismatch')
        # Validate the complete AP partition before omitting fixed atoms from
        # the initializer. Otherwise a broken IR could place a fixed atom in a
        # movable molecule, or silently emit an empty/comment-only molecule.
        for i,block in enumerate(logical['clusters']):
            atoms=block.get('atoms') if isinstance(block,dict) else None
            if not isinstance(atoms,list) or not atoms:
                raise ValueError(f'AP cluster {i} must contain a nonempty atoms list')
            for atom in atoms:
                if (not isinstance(atom,str) or not atom or atom.startswith('#') or
                        any(c.isspace() or not c.isprintable() for c in atom)):
                    raise ValueError(f'Invalid AP atom name in cluster {i}; expected a nonempty row-safe string')
                if atom in seen:
                    raise ValueError(f'Duplicate AP atom {atom!r}; all fixed/movable molecules must be disjoint')
                seen.add(atom)
        for i,block in enumerate(logical['clusters']):
            if not target['movable'][i]:
                continue  # Fixed AP blocks stay under native VPR control.
            x,y=target['teacher_xy'][i]
            if not all(math.isfinite(v) and 0<=v<=1 for v in (x,y)):
                raise ValueError('Invalid native AP coordinate')
            x=min(x*physical['grid_width'],math.nextafter(float(physical['grid_width']),0.))
            y=min(y*physical['grid_height'],math.nextafter(float(physical['grid_height']),0.))
            for atom in block['atoms']:
                rows.append(f'{atom}\t{x:.17g}\t{y:.17g}\t0\t0\n')
                flat.append(f'{atom}\t{math.floor(x)}\t{math.floor(y)}\t0\t0\n')
        (folder/'init.ap.tsv').write_text('# v69 native GP initialization; aggregate capacity is soft\n'+''.join(rows),encoding='utf-8')
        (folder/'init.fplace').write_text('# Diagnostic only, not a legalized placement or GP bypass\n'+''.join(flat),encoding='utf-8')
        write(folder/'target.json',target)
    measured('lowering_and_target_io',lower)
    elapsed=time.perf_counter()-start
    result=dict(signature=expected,timing=timing,phase_seconds=prediction.get('phase_seconds',{}),
        capacity_diagnostics=prediction['capacity_diagnostics'],
        capacity_usage=prediction['usage'].tolist(),candidate_region_count=prediction['candidate_region_count'],
        measured_generation_seconds=elapsed,
        generation_seconds_excluding_rrg_encoding=elapsed-timing['rrg_encoding'],
        hard_regional_capacity_enforced=False,regional_capacity_is_proxy=True,
        native_gp_required=True,final_legality='Certified only by VPR, not this initializer',
        leaf_count=len(physical['regions']),hierarchy_level_count=len(data['parents']),
        generation_scope='Fresh CPU model/graph/inference/lowering; excludes interpreter startup; RRG preprocessing reused')
    for name,value in prediction.items():
        if name.startswith('overflow') or name=='decoder_kind':
            result[name]=value
    for key,name in (('ap_init','init.ap.tsv'),('fplace','init.fplace'),('target','target.json')):
        result[key]=str((folder/name).resolve());result[key+'_sha256']=sha(folder/name)
    write(folder/'generation.json',result)
    emit('generated',design=context['design'],width=context['width'],candidate=candidate,
         seconds=elapsed,timing=timing,hard_capacity=False)
    return result


def worker(task):
    return generate(*task)
