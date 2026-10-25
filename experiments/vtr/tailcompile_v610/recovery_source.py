"""Read-only, content-audited imports from an unfinished v6.10 preparation.

Timeouts and tool failures remain diagnostics. Only complete canonical inputs,
whole graph grids and Default routing observations can become reusable caches.
"""
from __future__ import annotations

from contextlib import contextmanager
import os
from pathlib import Path

from bridge import emit, freeze, read, sha

SOURCE_SCHEMA = 'tailcompile-v610-frozen-sweep-1'
SOURCE_FIELDS = ('design','source','source_sha256','include_sha256')
BACKEND_FIELDS = ('vtr_commit','binary_sha256','architecture_sha256','hook')
STRUCTURE_SUFFIXES = ('.blocks.tsv','.nets.tsv','.meta.tsv')
KNOWN = {'success','unroutable'}


@contextmanager
def _source_lock(source):
    if os.name != 'posix':
        raise RuntimeError('Recovery source locking requires the Linux server')
    import fcntl
    lock = source/'.lock'
    if not lock.is_file():
        raise ValueError(f'Recovery source has no cohort lock: {lock}')
    # Read-only open: importing a cache must never create a file in its source.
    with lock.open('rb') as stream:
        try:
            fcntl.flock(stream,fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f'Recovery source is still running: {source}') from error
        try:
            yield
        finally:
            fcntl.flock(stream,fcntl.LOCK_UN)


def _metadata_files(source):
    for name in ('protocol.json','inventory.json','preparation_report.json','contexts.json',
                 'contexts.partial.json','prepared_inputs.json','boundary_plan.json'):
        path=source/name
        if path.is_file():
            yield path
    for pattern in ('inputs/*/source.json','inputs/*/mapping.json','inputs/*/prepared.json',
                    'screen/*/protocol.json','screen/*/boundary.json',
                    'screen/*/w*/s*/trial.json','screen/*/w*/s*/metrics.json'):
        yield from (p for p in source.glob(pattern) if p.is_file())


def _fingerprint_unlocked(source):
    return dict(source=str(source),files={str(p.relative_to(source)):sha(p)
        for p in sorted(set(_metadata_files(source)))})


def fingerprint_source(path):
    """Small metadata fingerprint for the child protocol; no tensor loading."""
    if path is None:
        return None
    source=Path(path).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f'Recovery source missing: {source}')
    with _source_lock(source):
        return _fingerprint_unlocked(source)


def _same_source(current, old):
    return all(current.get(k)==old.get(k) for k in SOURCE_FIELDS)


def _flag(command, flag):
    if command.count(flag)!=1:
        raise ValueError(f'Expected exactly one {flag} in cached VPR command')
    index=command.index(flag)
    if index+1>=len(command):
        raise ValueError(f'Cached VPR command has no value for {flag}')
    return command[index+1]


def _trial_identity(trial, identity, stage):
    saved=trial.get('identity',{})
    if saved.get('stage')!=stage or any(saved.get(k)!=identity.get(k) for k in BACKEND_FIELDS):
        raise ValueError('Cached trial backend/stage identity changed')
    command=trial.get('command')
    if not isinstance(command,list) or not command or command!=saved.get('command'):
        raise ValueError('Cached trial command ledger differs from its identity')
    return saved,command


def _check_stopped(source):
    for marker in source.rglob('started.json'):
        if {'_attempts','_interrupted','_repairs'} & set(marker.parts):
            continue
        if not (marker.parent/'trial.json').is_file():
            raise ValueError(f'Interrupted source trial requires recovery first: {marker.parent}')


def load_recovery(a, inventory):
    source_arg=getattr(a,'recovery_source',None)
    empty=dict(contexts=[],inputs={},boundaries={},observations={},failures=[],manifest=None)
    if source_arg is None:
        return empty
    source,out=Path(source_arg).resolve(),Path(a.out).resolve()
    if source==out or source in out.parents or out in source.parents:
        raise ValueError('Recovery source and child output must be separate cohort directories')
    if not source.is_dir():
        raise FileNotFoundError(f'Recovery source missing: {source}')
    with _source_lock(source):
        return _load_locked(a,inventory,source,out)


def _load_locked(a, inventory, source, out):
    from data import read_ap_snapshot
    from prepare import boundary_grid, reusable_contexts
    from common import classify
    _check_stopped(source)
    signature=_fingerprint_unlocked(source)
    protocol=read(source/'protocol.json')
    if protocol.get('schema')!=SOURCE_SCHEMA or protocol.get('training_permitted') is not False:
        raise ValueError('Recovery requires a frozen, no-training v610 source')
    if protocol.get('identity')!=a.identity:
        raise ValueError('Recovery source VTR/architecture/hook identity differs')
    if sha(a.checkpoint)!=protocol.get('checkpoint_sha256'):
        raise ValueError('Recovery source checkpoint differs from frozen child checkpoint')
    for name,expected in (('seeds',a.calibration_seeds),('start',a.width_start),
                          ('maximum',a.width_max),('step',a.width_step),('half_band',a.half_band)):
        key={'seeds':'calibration_seeds','start':'width_start','maximum':'width_max',
             'step':'width_step'}.get(name,name)
        if protocol.get(key)!=expected:
            raise ValueError(f'Recovery calibration settings differ: {key}')
    old_inventory=read(source/'inventory.json')
    old_rows={r['design']:r for r in old_inventory['designs']}
    current={r['design']:r for r in inventory['designs']}
    if len(old_rows)!=len(old_inventory['designs']) or len(current)!=len(inventory['designs']):
        raise ValueError('Duplicate recovery inventory design')
    for name,row in current.items():
        if name not in old_rows or not _same_source(row,old_rows[name]):
            raise ValueError(f'Recovery source/include identity differs: {name}')
    extras=[r for name,r in old_rows.items() if name not in current]
    if any(r['source'] not in {'koios/hard_mem_include.v','koios/complex_dsp_include.v'} for r in extras):
        raise ValueError('Recovery source and child inventories differ beyond the two macro includes')

    refs={}
    snapshots={}
    diagnostics=[]
    def snapshot(path):
        stat=Path(path).stat()
        return (stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns)
    def proof(path):
        path=Path(path).resolve()
        if str(path) not in refs:
            before=snapshot(path)
            refs[str(path)]=sha(path)
            if snapshot(path)!=before:
                raise ValueError(f'Recovery artifact changed while hashing: {path}')
            snapshots[str(path)]=before
        return refs[str(path)]
    def skip(design,reason,path=None):
        diagnostics.append(dict(design=design,status='cache_not_reused',reason=reason,
                                path=str(path) if path else None))
    for relative,digest in signature['files'].items():
        refs[str((source/relative).resolve())]=digest

    def command_payloads(name, command, mapped_hash):
        if len(command)<3:
            raise ValueError(f'Cached VPR command lacks binary/architecture/BLIF: {name}')
        for index,expected,label in ((0,a.identity['binary_sha256'],'binary'),
                                     (1,a.identity['architecture_sha256'],'architecture'),
                                     (2,mapped_hash,'BLIF')):
            path=Path(command[index])
            if not path.is_file():
                skip(name,f'missing_command_{label}_artifact',path)
                return False
            if proof(path)!=expected:
                raise ValueError(f'Cached VPR command {label} content differs: {name}')
        return True

    def canonical(row):
        name=row['design']
        if name not in current:
            return None
        record=current[name]
        if not _same_source(record,row):
            raise ValueError(f'Cached input source differs: {name}')
        required=('mapped_blif','mapped_blif_sha256','structure_prefix','device_width','device_height')
        if any(k not in row for k in required):
            skip(name,'incomplete_input_metadata')
            return None
        blif,prefix=Path(row['mapped_blif']),str(row['structure_prefix'])
        folder=Path(prefix).parent
        source_path=blif.parent/'source.json'
        mapping_path=blif.parent/'mapping.json'
        trial_path=folder/'structure_run/trial.json'
        structure_paths=[Path(prefix+s) for s in STRUCTURE_SUFFIXES]
        paths=[blif,source_path,mapping_path,trial_path,*structure_paths]
        missing=[str(p) for p in paths if not p.is_file()]
        if missing:
            skip(name,'missing_canonical_input_artifact',missing[0])
            return None
        if not _same_source(record,read(source_path)) or not _same_source(record,read(mapping_path)):
            raise ValueError(f'Canonical source/mapping signature differs: {name}')
        mapped=proof(blif)
        if mapped!=row['mapped_blif_sha256'] or read(mapping_path).get('mapped_blif_sha256')!=mapped:
            raise ValueError(f'Canonical mapped BLIF hash differs: {name}')
        trial=read(trial_path)
        saved,command=_trial_identity(trial,a.identity,'structure')
        if trial.get('status')!='complete' or trial.get('returncode')!=0 or saved.get('blif_sha256')!=mapped:
            raise ValueError(f'Canonical structure export was not complete: {name}')
        if not command_payloads(name,command,mapped):
            return None
        env=saved.get('hook_env',{})
        if (str(Path(env.get('TC_GP_STRUCTURE','')).resolve())!=str(Path(prefix).resolve()) or
                str(env.get('TC_GP_STRUCTURE_ONLY'))!='1' or env.get('TC_GP_INIT')):
            raise ValueError(f'Cached structure export hook differs: {name}')
        if '--device_width' in command and int(_flag(command,'--device_width'))!=row['device_width']:
            raise ValueError(f'Cached structure device width differs: {name}')
        snapshot=read_ap_snapshot(prefix,structure=True)
        if snapshot['size']!=[row['device_width'],row['device_height']]:
            raise ValueError(f'Cached AP snapshot dimensions differ: {name}')
        hashes={s:proof(prefix+s) for s in STRUCTURE_SUFFIXES}
        if row.get('structure_sha256') is not None and row['structure_sha256']!=hashes:
            raise ValueError(f'Cached AP structure hash differs: {name}')
        for p in (source_path,mapping_path,trial_path):
            proof(p)
        return dict(record,mapped_blif=str(blif.resolve()),mapped_blif_sha256=mapped,
            structure_prefix=str(Path(prefix).resolve()),device_width=row['device_width'],
            device_height=row['device_height'],structure_sha256=hashes,
            recovery_origin=dict(source=str(source),structure_trial=str(trial_path.resolve())))

    contexts=[]
    cached=[]
    for name in ('contexts.json','contexts.partial.json'):
        path=source/name
        if path.is_file():
            value=read(path)['contexts']
            if cached and value!=cached:
                raise ValueError('Final and partial source context manifests differ')
            cached=value
    for name,record in sorted(current.items()):
        group=[r for r in cached if r['design']==name]
        if not group:
            continue
        declared=group[0].get('widths',[])
        if len({r['width'] for r in group})!=len(group) or any(r.get('widths')!=declared for r in group):
            raise ValueError(f'Duplicate/inconsistent graph grid: {name}')
        if sorted(r['width'] for r in group)!=sorted(declared):
            skip(name,'partial_declared_graph_grid')
            continue
        imported=reusable_contexts(record,group,a)
        if imported is None:
            skip(name,'incomplete_graph_grid_or_source_provenance')
            continue
        for item in imported:
            # reusable_contexts already validates every graph audit/hash.
            for key in ('graph','logic','device','supply','audit','mapped_blif'):
                proof(item[key])
        contexts.extend(imported)
        emit('recovery_progress',phase='graphs',design=name,recovered_contexts=len(contexts))

    candidates=list(cached)
    for name in ('prepared_inputs.json','boundary_plan.json'):
        path=source/name
        if path.is_file():
            candidates.extend(read(path)['designs'])
    candidates.extend(read(p) for p in sorted((source/'inputs').glob('*/prepared.json')))
    inputs={}
    seen=set()
    for row in candidates:
        key=(row['design'],row.get('mapped_blif'),row.get('structure_prefix'))
        if key in seen:
            continue
        seen.add(key)
        value=canonical(row)
        if value is None:
            continue
        old=inputs.get(value['design'])
        if old is not None and any(old[k]!=value[k] for k in ('mapped_blif_sha256','structure_sha256','device_width','device_height')):
            raise ValueError(f'Conflicting canonical inputs in recovery manifests: {value["design"]}')
        inputs[value['design']]=value
        emit('recovery_progress',phase='canonical_inputs',design=value['design'],recovered_inputs=len(inputs))

    observations={}
    for trial_path in sorted((source/'screen').glob('*/w*/s*/trial.json')):
        folder=trial_path.parent
        name=folder.parent.parent.name
        if name not in inputs:
            continue
        record=inputs[name]
        trial=read(trial_path)
        if trial.get('status') not in KNOWN:
            diagnostics.append(dict(design=name,status=trial.get('status'),returncode=trial.get('returncode'),
                directory=str(folder),reason='unknown_source_outcome_not_imported'))
            continue
        metrics_path,log_path=folder/'metrics.json',folder/'vpr.out'
        if not metrics_path.is_file() or not log_path.is_file():
            skip(name,'missing_observation_evidence',folder)
            continue
        saved,command=_trial_identity(trial,a.identity,'route')
        if not command_payloads(name,command,record['mapped_blif_sha256']):
            continue
        row=read(metrics_path)
        width,seed=int(folder.parent.name[1:]),int(folder.name[1:])
        if seed not in a.calibration_seeds:
            continue
        expected=dict(design=name,width=width,seed=seed,device_width=record['device_width'],
                      mapped_blif_sha256=record['mapped_blif_sha256'],initializer_sha256=None,teacher=False)
        if any(saved.get(k)!=v for k,v in expected.items()) or saved.get('hook_env'):
            raise ValueError(f'Source observation is not the matching uninitialized Default: {folder}')
        if (row.get('status')!=trial['status'] or row.get('returncode')!=trial.get('returncode') or
                any(row.get(k)!=v for k,v in dict(design=name,width=width,seed=seed).items()) or
                row.get('identity')!=saved or row.get('teacher_prefix') is not None):
            raise ValueError(f'Source observation metrics/trial differ: {folder}')
        for flag,value in (('--seed',seed),('--route_chan_width',width),('--device_width',record['device_width']),
                           ('--ap_full_legalizer','appack'),('--ap_detailed_placer','annealer'),('--num_workers',1)):
            if _flag(command,flag)!=str(value):
                raise ValueError(f'Source observation command condition differs: {folder}')
        if not all(flag in command for flag in ('--analytical_place','--route','--analysis')):
            raise ValueError(f'Source observation did not run the native flow: {folder}')
        if classify(trial['returncode'],log_path.read_text(errors='replace'))!=trial['status']:
            raise ValueError(f'Source observation log no longer certifies its outcome: {folder}')
        key=(name,width,seed)
        if key in observations:
            raise ValueError(f'Duplicate recovery observation: {key}')
        evidence={str(p.resolve()):proof(p) for p in (trial_path,metrics_path,log_path)}
        observations[key]=dict(row,recovery_grid=[record['device_width'],record['device_height']],
            recovery_mapped_blif_sha256=record['mapped_blif_sha256'],recovery_evidence_sha256=evidence,
            recovery_directory=str(folder.resolve()))

    boundaries={}
    for path in sorted((source/'screen').glob('*/boundary.json')):
        name=path.parent.name
        if name not in inputs:
            continue
        record=inputs[name]
        saved=read(path)
        expected=dict(identity=a.identity,source_sha256=record['source_sha256'],
            mapped_blif_sha256=record['mapped_blif_sha256'],device_width=record['device_width'],
            seeds=a.calibration_seeds,start=a.width_start,maximum=a.width_max,
            step=a.width_step,half_band=a.half_band)
        if ({k:v for k,v in saved['signature'].items() if k!='timeout'}!=expected or
                saved['signature'].get('timeout')!=protocol.get('timeout')):
            raise ValueError(f'Completed source boundary settings differ: {name}')
        value=saved['boundary']
        if {r['seed'] for r in value['calibration']}!=set(a.calibration_seeds) or len(value['calibration'])!=2:
            raise ValueError(f'Completed source boundary must have both calibration seeds: {name}')
        for observation in value['calibration_observations']:
            key=(name,observation['width'],observation['seed'])
            if key not in observations:
                skip(name,'incomplete_boundary_observation_evidence',path)
                break
            if observations[key]['status']!=observation['status']:
                raise ValueError(f'Boundary observation differs from its source trial: {name}')
        else:
            censored=False
            for entry in value['calibration']:
                minimum,adjacent=entry['lowest_success'],entry.get('adjacent_failure')
                if minimum is None:
                    censored=True
                    if (not entry.get('right_censored') or adjacent!=a.width_max or
                            observations.get((name,a.width_max,entry['seed']),{}).get('status')!='unroutable'):
                        raise ValueError(f'Invalid right-censored source boundary: {name}')
                elif (observations.get((name,minimum,entry['seed']),{}).get('status')!='success' or
                      (adjacent is not None and observations.get((name,adjacent,entry['seed']),{}).get('status')!='unroutable')):
                    raise ValueError(f'Invalid successful source boundary: {name}')
            center=a.width_max if censored else max(r['lowest_success'] for r in value['calibration'])
            if (value['reference_width']!=center or value.get('reference_is_success')!=(not censored) or
                    value['widths']!=boundary_grid(center,a.width_step,a.half_band,
                                                  maximum=a.width_max if censored else None)):
                raise ValueError(f'Completed source boundary grid/reference differs: {name}')
            proof(path)
            boundaries[name]=dict(record,**value,recovery_boundary_path=str(path.resolve()),
                                  recovery_boundary_sha256=proof(path))
    if _fingerprint_unlocked(source)!=signature:
        raise ValueError('Recovery source changed while importing caches')
    # The source lock excludes cooperative writers. Avoid rehashing all tensor
    # payloads a second time; detect any external replacement/write by snapshot.
    for path,previous in snapshots.items():
        if snapshot(path)!=previous:
            raise ValueError(f'Recovery artifact changed while importing: {path}')
    report=read(source/'preparation_report.json') if (source/'preparation_report.json').is_file() else {}
    failures=list(report.get('failures',[]))+diagnostics
    manifest=dict(schema='tailcompile-v610-recovery-source-1',source=str(source),source_fingerprint=signature,
        source_protocol_sha256=proof(source/'protocol.json'),source_inventory_sha256=proof(source/'inventory.json'),
        checkpoint_sha256=protocol['checkpoint_sha256'],identity=a.identity,
        referenced_sha256=dict(sorted(refs.items())),
        imported=dict(contexts=len(contexts),inputs=sorted(inputs),boundaries=sorted(boundaries),
            observations=[dict(design=d,width=w,seed=s,status=row['status']) for (d,w,s),row in sorted(observations.items())]),
        failures=failures,policy='Source read-only; known Default observations only; no timeout or failure becomes an unroutable label')
    freeze(out/'recovery_manifest.json',manifest)
    emit('recovery_imported',contexts=len(contexts),inputs=len(inputs),boundaries=len(boundaries),observations=len(observations))
    return dict(contexts=contexts,inputs=inputs,boundaries=boundaries,observations=observations,
                failures=failures,manifest=manifest)
