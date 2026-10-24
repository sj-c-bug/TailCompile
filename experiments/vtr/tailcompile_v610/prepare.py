"""Prepare a frozen-weight sweep without collecting any training teachers.

Existing v68 deployment graphs are read-only. New device RRGs live only until
their complete group of graph/IR consumers has been checked and persisted.
"""
from __future__ import annotations

import concurrent.futures as cf
import math
import multiprocessing as mp
import os
from pathlib import Path
import threading

from bridge import (V68, REPO, e, emit, freeze, read, sha, write,
                    parallel, check_free_space, StoragePressureError,
                    storage_error, log_storage_failure, prune_trial)
from inventory import discover

GRAPH_KEYS = ('graph', 'logic', 'device', 'supply')
STRUCTURE_SUFFIXES = ('.blocks.tsv', '.nets.tsv', '.meta.tsv')


class PreparationFailure(RuntimeError):
    """A known status that cannot be silently excluded as unroutability."""
    def __init__(self, status, message, directory=None):
        super().__init__(message)
        self.status = status
        self.directory = str(directory) if directory else None

    def __reduce__(self):
        return (type(self), (self.status, str(self), self.directory))


def boundary_grid(center, step=4, half_band=3, maximum=None):
    if min(center, step) <= 0 or half_band < 0 or center % step:
        raise ValueError('Positive quantized center/step and nonnegative half_band required')
    points = {max(step, center+i*step) for i in range(-half_band, half_band+1)} | {center+6*step}
    if maximum is not None:
        points = {min(maximum, width) for width in points}
    return sorted(points)


def verify_graph_outputs(context, *, tile_size, architecture_sha256, mapped_sha256=None):
    """Validate a deployment graph without needing a retained raw RRG XML."""
    audit_path = Path(context['audit'])
    if not audit_path.is_file() or any(not Path(context[k]).is_file() for k in GRAPH_KEYS):
        return False
    audit = read(audit_path)
    signature = audit['signature']
    if (signature.get('tile_size') != tile_size or
            signature.get('channel_width') != context['width'] or
            signature.get('architecture_sha256') != architecture_sha256 or
            signature.get('design') != context['design'] or
            (mapped_sha256 is not None and signature.get('mapped_blif_sha256') != mapped_sha256)):
        raise ValueError(f'Deployment graph input identity differs: {audit_path}')
    for key in GRAPH_KEYS:
        if sha(context[key]) != audit['output_sha256'][key]:
            raise ValueError(f'Corrupted graph/IR output: {context[key]}')
    if audit.get('movable_coordinates_in_features') is not False:
        raise ValueError(f'Deployment graph may contain teacher coordinates: {audit_path}')
    return True


def _source_match(record, old_context):
    folder = Path(old_context['mapped_blif']).parent
    source_path = folder/'source.json'
    if not source_path.is_file():
        return False
    saved = read(source_path)
    return all(saved.get(key) == record.get(key)
               for key in ('design', 'source', 'source_sha256', 'include_sha256'))


def _context_metadata(record, context, origin):
    return dict(context, design=record['design'], family=record['family'], split='sweep',
                original_split=record['original_split'], checkpoint_exposure=record['checkpoint_exposure'],
                source=record['source'], source_sha256=record['source_sha256'],
                include_sha256=record['include_sha256'], inventory_origin=record['inventory_origin'],
                context_origin=origin)


def reusable_contexts(record, old_contexts, a):
    group = sorted((r for r in old_contexts if r['design'] == record['design']), key=lambda r:r['width'])
    if not group or not _source_match(record, group[0]):
        return None
    mapped = Path(group[0]['mapped_blif'])
    if not mapped.is_file():
        return None
    mapped_hash = sha(mapped)
    if mapped_hash != group[0]['mapped_blif_sha256']:
        raise ValueError(f'Canonical reused BLIF changed: {mapped}')
    declared = group[0]['widths']
    if sorted({r['width'] for r in group}) != sorted(declared) or len(group) != len(set(declared)):
        raise ValueError(f'Incomplete/duplicate recorded context grid: {record["design"]}')
    if not all(verify_graph_outputs(c, tile_size=a.tile_size,
                architecture_sha256=a.identity['architecture_sha256'], mapped_sha256=mapped_hash)
                for c in group):
        return None
    result = []
    for c in group:
        if c['mapped_blif_sha256'] != mapped_hash or c['widths'] != declared:
            raise ValueError(f'Inconsistent reused input/grid: {record["design"]}')
        result.append(_context_metadata(record, c, 'verified_v68_deployment_cache'))
    return result


def reusable_input(record, old_contexts):
    """A missing graph need not force synthesis of an unchanged source again."""
    old = next((c for c in old_contexts if c['design'] == record['design']), None)
    if old is None or not _source_match(record, old):
        return None
    blif, prefix = Path(old['mapped_blif']), old['structure_prefix']
    if not blif.is_file() or any(not Path(prefix+s).is_file() for s in STRUCTURE_SUFFIXES):
        return None
    if sha(blif) != old['mapped_blif_sha256']:
        raise ValueError(f'Canonical reused BLIF changed: {blif}')
    audit_path = Path(old['audit'])
    if not audit_path.is_file():
        return None
    structure_hashes = read(audit_path)['signature'].get('structure_sha256')
    if not structure_hashes:
        return None
    if structure_hashes != {suffix:sha(prefix+suffix) for suffix in STRUCTURE_SUFFIXES}:
        raise ValueError(f'Canonical reused AP structure changed: {prefix}')
    return dict(record, mapped_blif=str(blif), mapped_blif_sha256=old['mapped_blif_sha256'],
                structure_prefix=prefix, device_width=old['device_width'],
                device_height=old['device_height'])


def _failure(record, stage, error):
    return dict(design=record['design'], family=record['family'], source=record['source'],
                stage=stage, status=getattr(error, 'status', 'preparation_failure'),
                error=str(error), directory=getattr(error, 'directory', None))


def _check_trial_failure(folder, error):
    path = Path(folder)/'trial.json'
    if not path.is_file():
        return
    trial = read(path)
    text = (Path(folder)/'vpr.out').read_text(encoding='utf-8', errors='replace')
    if log_storage_failure(text, trial['returncode']):
        raise StoragePressureError(f'Storage exhausted during preparation: {folder}') from error
    if trial['status'] in {'timeout', 'killed'}:
        raise PreparationFailure(trial['status'], str(error), folder) from error
    # A tool issue remains visible and blocks prepare acceptance. No arbitrary
    # synthesis crash is relabelled an unsupported benchmark to improve scores.
    if trial['returncode'] != 0:
        raise PreparationFailure('preparation_failure', str(error), folder) from error


def calibrate(a, record):
    root = a.out/'screen'/record['design']
    signature = dict(identity=a.identity, source_sha256=record['source_sha256'],
        mapped_blif_sha256=record['mapped_blif_sha256'], device_width=record['device_width'],
        seeds=a.calibration_seeds, start=a.width_start, maximum=a.width_max,
        step=a.width_step, half_band=a.half_band, timeout=a.timeout)
    freeze(root/'protocol.json', signature)
    saved = root/'boundary.json'
    if saved.is_file():
        value = read(saved)
        if value['signature'] != signature:
            raise ValueError(f'Calibration identity changed: {root}')
        return dict(record, **value['boundary'])
    observations = {}

    def outcome(width, seed):
        key = (width, seed)
        if key not in observations:
            folder = root/f'w{width}'/f's{seed}'
            row = e.route(a, record, width, seed, folder)
            prune_trial(folder, a.out)
            observations[key] = row['status']
            if row['status'] not in {'success', 'unroutable'}:
                raise PreparationFailure(row['status'], f'Cannot calibrate from {row["status"]}: {folder}', folder)
        return observations[key] == 'success'

    boundaries = []
    for seed in a.calibration_seeds:
        low, high = 0, a.width_start
        while not outcome(high, seed):
            low = high
            if high >= a.width_max:
                boundaries.append(dict(seed=seed, lowest_success=None, adjacent_failure=low,
                                       right_censored=True))
                break
            high = min(a.width_max, math.ceil(high*1.5/a.width_step)*a.width_step)
        else:
            # Only a cheap Default reference search. The actual five-seed
            # evaluation scans every frozen grid point and retains reversals.
            while high-low > a.width_step:
                middle = max(low+a.width_step, ((low+high)//(2*a.width_step))*a.width_step)
                if outcome(middle, seed):
                    high = middle
                else:
                    low = middle
            boundaries.append(dict(seed=seed, lowest_success=high, adjacent_failure=low or None,
                                   right_censored=False))
    censored = any(row['right_censored'] for row in boundaries)
    center = a.width_max if censored else max(row['lowest_success'] for row in boundaries)
    boundary = dict(calibration=boundaries, reference_width=center,
        reference_is_success=not censored,
        calibration_status='right_censored' if censored else 'reference_found',
        widths=boundary_grid(center, a.width_step, a.half_band,
                            maximum=a.width_max if censored else None),
        calibration_observations=[dict(width=w, seed=s, status=status)
                                  for (w,s),status in sorted(observations.items())],
        boundary_selection='Default-only expansion/bisection reference, not a monotonicity or global Wmin claim; full evaluation grid frozen before GNN results')
    freeze(saved, dict(signature=signature, boundary=boundary))
    return dict(record, **boundary)


def _graph_input(record, width, a):
    return dict(identity=a.identity, design=record['design'], width=width, tile_size=a.tile_size,
                mapped_blif_sha256=sha(record['mapped_blif']),
                structure_sha256={s:sha(record['structure_prefix']+s) for s in STRUCTURE_SUFFIXES},
                graph_implementation_sha256={p.name:sha(p) for p in
                    (V68/'data.py', V68.parent/'prepare_tailcompile_v61.py',
                     V68.parent/'tailcompile_v61.py', V68.parent/'tailcompile_v3.py',
                     V68.parent/'tailcompile_v4_risk.py', V68.parent/'tailrisk_core.py')})


def _cached_graph(record, width, a):
    folder = a.out/'graphs'/record['design']/f'w{width}'
    manifest = folder/'sweep_graph.json'
    if not manifest.is_file():
        return None
    saved = read(manifest)
    if saved['inputs'] != _graph_input(record, width, a):
        raise ValueError(f'Frozen deployment graph changed: {folder}')
    context = saved['context']
    if not verify_graph_outputs(context, tile_size=a.tile_size,
            architecture_sha256=a.identity['architecture_sha256'],
            mapped_sha256=record['mapped_blif_sha256']):
        raise ValueError(f'Previously complete graph artifacts missing: {folder}; restore artifacts or use a new output directory')
    return context


def _prune_rrg(a, folder, contexts):
    """Only this sweep's closed shared cache, after every consumer is durable."""
    if os.environ.get('TAILCOMPILE_V610_PRUNE_COMPLETED', '1') == '0':
        return
    folder = Path(folder).resolve()
    folder.relative_to(a.out.resolve()/'device_cache')
    for context in contexts:
        if not verify_graph_outputs(context, tile_size=a.tile_size,
                architecture_sha256=a.identity['architecture_sha256'],
                mapped_sha256=context['mapped_blif_sha256']):
            raise ValueError(f'Cannot reclaim RRG with incomplete consumer: {context["design"]}')
    proof = dict(identity=a.identity, consumers=[dict(design=c['design'], width=c['width'],
                 graph_audit_sha256=sha(c['audit']), graph_sha256=sha(c['graph'])) for c in contexts])
    freeze(folder/'consumers_complete.json', proof)
    audit = read(folder/'rr_graph.json')
    if audit['identity'] != a.identity:
        raise ValueError(f'RRG backend identity changed: {folder}')
    target = folder/'rr_graph.xml.gz'
    # Source cache path must not be tracked and must not be any input retained
    # for another stage. The consumer list is the complete device-key group.
    import storage
    tracked = storage.tracked_files(REPO, folder=folder)
    if target.exists():
        if target.is_symlink() or sha(target) != audit['sha256']:
            raise ValueError(f'RRG changed before reclamation: {target}')
        if target in tracked:
            emit('rrg_retained', path=str(target), reason='git_tracked')
        else:
            released = target.stat().st_size
            write(folder/'rrg_prune_pending.json', dict(proof=proof, sha256=audit['sha256'], bytes=released))
            target.unlink()
            write(folder/'rrg_pruned.json', dict(raw_sha256=audit['sha256'], bytes=released,
                consumers=len(contexts), note='All immutable graph/IR outputs validated; resume verifies graph manifests without raw RRG'))
            (folder/'rrg_prune_pending.json').unlink()
            emit('rrg_pruned', directory=str(folder), bytes=released, consumers=len(contexts))
    # Pack-only RRG export also duplicates packed netlists per width. These
    # outputs are not the canonical mapped BLIF/AP structure used by the model.
    build = folder/'build'
    trial_path = build/'trial.json'
    if not trial_path.is_file() or read(trial_path).get('status') != 'success':
        return
    trial_hash = sha(trial_path)
    entries = []
    for path in build.iterdir():
        if (not path.is_file() or path.is_symlink() or path in tracked or
                path.suffix not in {'.net', '.route', '.place', '.post_routing', '.net_postrouting', '.postrouting'}):
            continue
        entries.append(dict(path=str(path), stat=storage.snapshot(path), trial_sha256=trial_hash))
    if entries:
        write(folder/'pack_prune_pending.json', dict(entries=entries))
    removed = []
    for entry in entries:
        path = Path(entry['path'])
        if storage.snapshot(path) != entry['stat'] or sha(trial_path) != trial_hash:
            raise ValueError(f'Pack output changed before reclamation: {path}')
        path.unlink()
        removed.append(entry)
    if removed:
        previous = read(folder/'pack_pruned.json') if (folder/'pack_pruned.json').is_file() else {'cleanups':[]}
        previous['cleanups'].append(dict(removed=removed, consumer_proof_sha256=sha(folder/'consumers_complete.json')))
        write(folder/'pack_pruned.json', previous)
        (folder/'pack_prune_pending.json').unlink()
        emit('pack_outputs_pruned', directory=str(build), files=len(removed),
             bytes=sum(entry['stat']['allocated_bytes'] for entry in removed))


def graph_group_worker(task):
    a, records, width = task
    import torch
    torch.set_num_threads(1)
    contexts, pending = [], []
    for record in records:
        old = _cached_graph(record, width, a)
        if old is not None:
            contexts.append(old)
        else:
            pending.append(record)
    if pending:
        check_free_space(a.out)
        rrg = e.rr_graph(a, records[0], width)
        from data import build_cached_graph
        for record in pending:
            request = dict(record, channel_width=width, rr_graph=str(rrg), architecture=str(a.arch),
                           vtr_commit=a.identity['vtr_commit'])
            folder = a.out/'graphs'/record['design']/f'w{width}'
            built = build_cached_graph(request, folder, tile_size=a.tile_size)
            context = dict(record, width=width, **built)
            context = _context_metadata(record, context, 'new_frozen_weight_sweep_graph')
            if not verify_graph_outputs(context, tile_size=a.tile_size,
                    architecture_sha256=a.identity['architecture_sha256'],
                    mapped_sha256=record['mapped_blif_sha256']):
                raise ValueError(f'Graph build did not persist every output: {folder}')
            freeze(folder/'sweep_graph.json', dict(inputs=_graph_input(record, width, a), context=context))
            contexts.append(context)
            emit('graph_ready', design=record['design'], width=width, tile_size=a.tile_size)
    cache = a.out/'device_cache'/f'{records[0]["device_width"]}x{records[0]["device_height"]}'/f'w{width}'
    if (cache/'rr_graph.json').is_file():
        _prune_rrg(a, cache, sorted(contexts, key=lambda c:c['design']))
    return contexts


def prepare(a):
    """Return {'contexts': [...], 'report': {...}}; never train or export teachers."""
    a.out, a.source = Path(a.out).resolve(), Path(a.source).resolve()
    a.out.relative_to(REPO/'experiments/vtr/tailcompile_v610/runs')
    if len(a.calibration_seeds) != 2 or len(set(a.calibration_seeds)) != 2:
        raise ValueError('Exactly two distinct Default calibration seeds required')
    if set(a.calibration_seeds) & set(a.seeds):
        raise ValueError('Calibration and evaluation seeds must be disjoint')
    if (min(a.width_start,a.width_max,a.width_step) <= 0 or a.width_start > a.width_max or
            a.width_start % a.width_step or a.width_max % a.width_step):
        raise ValueError('Width limits must be positive multiples of width_step')
    metadata = getattr(a, 'checkpoint_metadata', None)
    inventory = discover(a.vtr, a.source, metadata, getattr(a, 'designs', None),
                         scope=getattr(a, 'scope', 'all'))
    freeze(a.out/'inventory.json', inventory)
    records = inventory['designs']
    old_contexts = []
    if (a.source/'protocol.json').is_file() and (a.source/'contexts.json').is_file():
        old_protocol = read(a.source/'protocol.json')
        if old_protocol['identity'] == a.identity:
            old_contexts = read(a.source/'contexts.json')['contexts']
        else:
            emit('source_reuse_disabled', reason='VPR/architecture/hook identity differs; current sweep inputs rebuilt')
    contexts, prepared, failures, excluded = [], [], [], []
    lock = threading.Lock()

    def progress(stage):
        with lock:
            report = dict(schema='tailcompile-v610-preparation-1', complete=False, stage=stage,
                inventory_design_count=len(records), ready_contexts=len(contexts),
                prepared_designs=len(prepared), failures=list(failures), excluded=list(excluded))
            write(a.out/'preparation_report.json', report)
            emit('preparation_progress', stage_name=stage, prepared_designs=len(prepared),
                 ready_contexts=len(contexts), failures=len(failures), excluded=len(excluded))

    def input_one(record):
        try:
            if record['input_status'] != 'available':
                with lock:
                    excluded.append(dict(design=record['design'], family=record['family'], status='missing_input',
                                         reason='source/include not present in pinned checkout', paths=record['missing_inputs']))
                return
            reused = reusable_contexts(record, old_contexts, a)
            if reused is not None:
                with lock:
                    contexts.extend(reused)
                emit('reuse_design_graphs', design=record['design'], contexts=len(reused))
            else:
                ready = reusable_input(record, old_contexts)
                if ready is None:
                    try:
                        ready = e.input_structure(a, record)
                    except Exception as error:
                        for stage in ('synthesis', 'structure_run'):
                            _check_trial_failure(a.out/'inputs'/record['design']/stage, error)
                        raise
                with lock:
                    prepared.append(ready)
        except Exception as error:
            with lock:
                failures.append(_failure(record, 'input', error))
            if storage_error(error):
                raise
        finally:
            progress('input')

    parallel(input_one, records, a.prepare_jobs)
    write(a.out/'prepared_inputs.json', dict(designs=sorted(prepared,key=lambda r:r['design'])))
    screened = []
    def screen_one(record):
        try:
            value = calibrate(a, record)
            with lock:
                screened.append(value)
        except Exception as error:
            with lock:
                failures.append(_failure(record, 'calibration', error))
            if storage_error(error):
                raise
        finally:
            progress('calibration')
    parallel(screen_one, prepared, a.jobs)
    # The manifest is written after each stage; failures stay visible and do
    # not erase already completed peers. Resume reconstructs the same groups.
    write(a.out/'boundary_plan.json', dict(designs=sorted(screened,key=lambda r:r['design'])))
    if failures:
        # A later resume must not add a newly qualified design to a shared
        # device group whose RRG has already been pruned. Finish all input and
        # calibration qualifications before any shared cache is reclaimed.
        write(a.out/'contexts.partial.json', dict(schema='tailcompile-v610-contexts-1', contexts=contexts))
        progress('qualification_incomplete')
        raise RuntimeError(f'Preparation qualification incomplete: {len(failures)} failures; inspect preparation_report.json and resume. Completed inputs and calibration are retained.')
    groups = {}
    for record in screened:
        for width in record['widths']:
            groups.setdefault((record['device_width'],record['device_height'],width), []).append(record)
    tasks = [(a,sorted(group,key=lambda r:r['design']),key[2]) for key,group in sorted(groups.items())]
    if tasks:
        jobs = min(a.rr_jobs, a.prepare_jobs, len(tasks))
        emit('graph_groups_start', groups=len(tasks), jobs=jobs,
             note='Each process builds one RRG and all dependent graph/IR outputs, then reclaims that RRG')
        with cf.ProcessPoolExecutor(max_workers=jobs, mp_context=mp.get_context('spawn')) as pool:
            pending = {}; iterator = iter(tasks); stopped = False
            def submit():
                try:
                    task = next(iterator)
                except StopIteration:
                    return False
                pending[pool.submit(graph_group_worker, task)] = task
                return True
            for _ in range(jobs): submit()
            while pending:
                done, _ = cf.wait(pending, timeout=5, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    task = pending.pop(future)
                    try:
                        contexts.extend(future.result())
                    except Exception as error:
                        for record in task[1]:
                            failures.append(dict(_failure(record,'graph',error), width=task[2]))
                        if storage_error(error):
                            stopped = True
                    if not stopped:
                        submit()
                progress('graph')
            if stopped:
                raise StoragePressureError('Graph dispatch stopped on storage pressure; completed graph groups retained')
    contexts.sort(key=lambda c:(c['design'],c['width']))
    ready_designs = {c['design'] for c in contexts}
    complete_designs = {d for d in ready_designs if
        sorted(c['width'] for c in contexts if c['design']==d) ==
        sorted(next(c['widths'] for c in contexts if c['design']==d))}
    report = dict(schema='tailcompile-v610-preparation-1', complete=bool(contexts) and not failures,
        stage='complete' if contexts and not failures else 'incomplete', inventory_design_count=len(records),
        ready_design_count=len(complete_designs), context_count=len(contexts),
        failures=failures, excluded=excluded,
        reused_context_count=sum(c['context_origin']=='verified_v68_deployment_cache' for c in contexts),
        benchmark_status=[dict(design=r['design'],family=r['family'],source=r['source'],
            exposure=r['checkpoint_exposure'],status='ready' if r['design'] in complete_designs else
            'excluded_missing_input' if any(x['design']==r['design'] for x in excluded) else 'incomplete') for r in records],
        no_training=True, teacher_trials=0,
        storage_policy='Read-only reused inputs; new shared RRG removed only after every dependent graph is validated')
    write(a.out/'preparation_report.json', report)
    write(a.out/'contexts.partial.json', dict(schema='tailcompile-v610-contexts-1', contexts=contexts))
    if failures:
        raise RuntimeError(f'Preparation incomplete: {len(failures)} recorded failures; inspect preparation_report.json and resume. Successful graphs retained.')
    if not contexts:
        raise RuntimeError('No runnable benchmark contexts; inspect inventory and preparation reports')
    freeze(a.out/'contexts.json', dict(schema='tailcompile-v610-contexts-1', contexts=contexts))
    emit('prepare_complete', designs=len(complete_designs), contexts=len(contexts), excluded=len(excluded), teacher_trials=0)
    return dict(contexts=contexts, report=report)
