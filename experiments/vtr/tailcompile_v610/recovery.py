"""Preparation continuation in a new cohort: shared old data, qualified grids.

Only Default calibration seeds choose the device. No GNN results, final-test
seeds, weight updates or VTR edits enter preparation decisions.
"""
from __future__ import annotations

from copy import copy
import math
from pathlib import Path
import threading

from bridge import (e, emit, execute, freeze, read, write, sha, fingerprint,
                    parallel, prune_trial, check_free_space, storage_error,
                    log_storage_failure, StoragePressureError)
from prepare import (PreparationFailure, boundary_grid, graph_group_worker,
                     _failure)
from recovery_inputs import prepare_input, resize_input, is_device_capacity_failure
from recovery_source import load_recovery
from preparation_retry import (load_retry_plan, override_for,
    audit as retry_audit, calibration_destination)


def policy(a):
    return dict(schema='tailcompile-v610-qualified-preparation-1',
        synthesis_timeout=a.synthesis_timeout, calibration_timeout=a.calibration_timeout,
        final_route_timeout=a.timeout, device_width_steps=a.device_width_steps,
        selection='First Default-qualified grid on frozen width ladder; no GNN/final seeds',
        reuse='Read-only exact source/BLIF/grid/seed/command outcomes; timeouts remain unknown',
        graphs='Regenerate AP structure and all device IR/graphs after any grid change')


def same_grid(record, imported):
    return bool(imported and all(record[k]==imported[k] for k in
        ('mapped_blif_sha256','device_width','device_height')))


def old_observation(a, recovery, record, width, seed):
    if not same_grid(record,recovery['inputs'].get(record['design'])):
        return None
    row=recovery['observations'].get((record['design'],width,seed))
    return row if row and row['status'] in {'success','unroutable'} else None


def resource_failure_before(recovery, record):
    if not same_grid(record,recovery['inputs'].get(record['design'])):
        return None
    for failure in recovery.get('failures',[]):
        if failure.get('design')!=record['design'] or failure.get('stage')!='calibration' or not failure.get('directory'):
            continue
        folder=Path(failure['directory'])
        if not (folder/'trial.json').is_file():
            continue
        row=read(folder/'trial.json');identity=row.get('identity',{})
        if (identity.get('device_width')==record['device_width'] and
            identity.get('mapped_blif_sha256')==record['mapped_blif_sha256'] and
            is_device_capacity_failure(row,folder)):
            return dict(directory=str(folder),trial_sha256=sha(folder/'trial.json'),
                        log_sha256=sha(folder/'vpr.out'),status=row['status'])
    return None


def qualify_placement(a,record,root):
    """Run native AP + APPack + detailed placement, with routing disabled."""
    rows=[]
    for seed in a.calibration_seeds:
        folder=root/'qualification'/f's{seed}'
        command=[x for x in e.vpr_command(a,record['mapped_blif'],a.width_start,seed,record['device_width'])
                 if x not in {'--route','--analysis'}]
        if not (folder/'trial.json').exists() or a.retry_failed or a.retry_timeouts:
            check_free_space(a.out)
        result=execute(command,folder,dict(a.identity,stage='route',role='default_placement_qualification',
            design=record['design'],width=a.width_start,seed=seed,
            mapped_blif_sha256=record['mapped_blif_sha256'],device_width=record['device_width'],
            device_height=record['device_height'],preparation_policy_sha256=fingerprint(policy(a))),
            timeout=a.calibration_timeout,heartbeat=a.heartbeat,
            retry_failed=a.retry_failed,retry_timeouts=a.retry_timeouts)
        text=(folder/'vpr.out').read_text(encoding='utf-8',errors='replace')
        row=dict(result,design=record['design'],width=a.width_start,seed=seed,
                 qualification='Default native analytical placement; routing not requested')
        write(folder/'metrics.json',row)
        prune_trial(folder,a.out)
        if log_storage_failure(text,result['returncode']):
            raise StoragePressureError(f'Disk quota/full during device qualification: {folder}')
        if result['status']!='success':
            error=PreparationFailure(result['status'],f'Default placement qualification failed: {folder}',folder)
            error.device_capacity_failure=is_device_capacity_failure(result,folder)
            raise error
        rows.append(dict(seed=seed,trial=str(folder/'trial.json'),trial_sha256=sha(folder/'trial.json')))
    freeze(root/'qualification.json',dict(complete=True,grid=[record['device_width'],record['device_height']],
        mapped_blif_sha256=record['mapped_blif_sha256'],seeds=a.calibration_seeds,trials=rows))


def calibrate_record(a,record,recovery,root):
    signature=dict(identity=a.identity,mapped_blif_sha256=record['mapped_blif_sha256'],
        grid=[record['device_width'],record['device_height']],source_sha256=record['source_sha256'],
        seeds=a.calibration_seeds,start=a.width_start,maximum=a.width_max,step=a.width_step,
        half_band=a.half_band,timeout=a.calibration_timeout)
    freeze(root/'calibration_protocol.json',signature)
    if (root/'boundary.json').is_file():
        saved=read(root/'boundary.json')
        if saved['signature']!=signature:
            raise ValueError('Recovery calibration signature changed')
        return dict(record,**saved['boundary'])
    observations={}
    budget_args=copy(a);budget_args.timeout=a.calibration_timeout
    retry=override_for(a,record['design'],'calibration')
    retried_observations=[]
    def outcome(width,seed):
        key=(width,seed)
        if key not in observations:
            folder=root/'calibration'/f'w{width}'/f's{seed}'
            row=old_observation(a,recovery,record,width,seed)
            if row is not None:
                freeze(folder/'imported_observation.json',dict(source_row=row,
                    grid=signature['grid'],mapped_blif_sha256=record['mapped_blif_sha256']))
                emit('reuse_calibration_trial',design=record['design'],width=width,seed=seed,status=row['status'])
            else:
                folder,extended=calibration_destination(root,width,seed,retry)
                if extended:
                    actual_args=copy(budget_args);actual_args.timeout=float(retry['new_timeout'])
                    row=e.route(actual_args,record,width,seed,folder,
                        extra={'preparation_retry':retry_audit(retry)})
                    retried_observations.append(dict(width=width,seed=seed,directory=str(folder),
                        actual_timeout=actual_args.timeout,status=row['status']))
                else:
                    row=e.route(budget_args,record,width,seed,folder)
                prune_trial(folder,a.out)
            observations[key]=row['status']
            if row['status'] not in {'success','unroutable'}:
                error=PreparationFailure(row['status'],f'Cannot calibrate from {row["status"]}: {folder}',folder)
                error.device_capacity_failure=is_device_capacity_failure(row,folder)
                raise error
        return observations[key]=='success'
    boundaries=[]
    for seed in a.calibration_seeds:
        low,high=0,a.width_start
        while not outcome(high,seed):
            low=high
            if high>=a.width_max:
                boundaries.append(dict(seed=seed,lowest_success=None,adjacent_failure=low,right_censored=True))
                break
            high=min(a.width_max,math.ceil(high*1.5/a.width_step)*a.width_step)
        else:
            while high-low>a.width_step:
                middle=max(low+a.width_step,((low+high)//(2*a.width_step))*a.width_step)
                if outcome(middle,seed):high=middle
                else:low=middle
            boundaries.append(dict(seed=seed,lowest_success=high,adjacent_failure=low or None,right_censored=False))
    censored=any(row['right_censored'] for row in boundaries)
    center=a.width_max if censored else max(row['lowest_success'] for row in boundaries)
    boundary=dict(calibration=boundaries,reference_width=center,reference_is_success=not censored,
        calibration_status='right_censored' if censored else 'reference_found',
        widths=boundary_grid(center,a.width_step,a.half_band,maximum=a.width_max if censored else None),
        calibration_observations=[dict(width=w,seed=s,status=status) for (w,s),status in sorted(observations.items())],
        boundary_selection='Default-only reference search; all frozen grid points scanned with five new seeds')
    if retry:
        boundary['preparation_retry']=dict(authorization=retry_audit(retry),
            observations=retried_observations)
    freeze(root/'boundary.json',dict(signature=signature,boundary=boundary))
    return dict(record,**boundary)


def prepare_design(a,record,recovery):
    imported=recovery['inputs'].get(record['design'])
    current=prepare_input(a,record,imported)
    inherited=recovery['boundaries'].get(record['design'])
    if inherited and same_grid(current,inherited):
        value=dict(current,**{k:inherited[k] for k in ('widths','reference_width','calibration','calibration_status','reference_is_success','calibration_observations','boundary_selection') if k in inherited})
        freeze(a.out/'screen'/record['design']/'adopted_boundary.json',value)
        emit('reuse_complete_boundary',design=record['design'],grid=[current['device_width'],current['device_height']])
        return value
    selected=a.out/'screen'/record['design']/'qualified_input.json'
    if selected.is_file():
        saved=read(selected)
        if saved['policy']!=policy(a) or saved['mapped_blif_sha256']!=current['mapped_blif_sha256']:
            raise ValueError('Frozen device qualification policy/input changed')
        validated=prepare_input(a,record,saved['record'])
        if not same_grid(validated,saved['record']) or validated['structure_prefix']!=saved['record']['structure_prefix']:
            raise ValueError('Qualified AP structure changed')
        return saved['record']
    initial=current['device_width'];prior=resource_failure_before(recovery,current)
    attempts=[]
    for increment in a.device_width_steps:
        width=initial+increment
        if increment==0 and prior:
            attempts.append(dict(requested_width=width,status='known_device_capacity_failure',source=prior))
            emit('reuse_device_capacity_failure',design=record['design'],width=width)
            continue
        try:
            candidate=current if increment==0 else resize_input(a,current,width)
            root=a.out/'screen'/record['design']/f'g{candidate["device_width"]}x{candidate["device_height"]}'
            qualify_placement(a,candidate,root)
            value=calibrate_record(a,candidate,recovery,root)
            attempts.append(dict(requested_width=width,status='qualified',grid=[candidate['device_width'],candidate['device_height']]))
            freeze(selected,dict(policy=policy(a),mapped_blif_sha256=current['mapped_blif_sha256'],
                initial_grid=[current['device_width'],current['device_height']],attempts=attempts,record=value))
            emit('device_qualified',design=record['design'],grid=[candidate['device_width'],candidate['device_height']],
                 width_increase=candidate['device_width']-initial)
            return value
        except PreparationFailure as error:
            attempts.append(dict(requested_width=width,status=error.status,directory=error.directory,
                                 capacity_failure=getattr(error,'device_capacity_failure',False)))
            write(a.out/'screen'/record['design']/'qualification_attempts.json',dict(policy=policy(a),attempts=attempts))
            if not getattr(error,'device_capacity_failure',False):
                raise
    raise PreparationFailure('device_capacity_failure',
        f'No Default-qualified grid within frozen device width ladder: {record["design"]}',
        a.out/'screen'/record['design'])


def prepare(a):
    from inventory import discover
    retry_plan=load_retry_plan(a)
    if retry_plan:
        emit('preparation_retry_authorized',overrides=retry_plan['overrides'],
            note='Base preparation policy and formal scan/timing budgets unchanged')
    inventory=discover(a.vtr,a.source,a.checkpoint_metadata,a.designs,scope=a.scope)
    freeze(a.out/'inventory.json',inventory)
    recovery=load_recovery(a,inventory)
    contexts=list(recovery['contexts']);ready=[];failures=[];lock=threading.Lock()
    cached={c['design'] for c in contexts}
    records=[r for r in inventory['designs'] if r['design'] not in cached]
    def progress(stage):
        with lock:
            report=dict(schema='tailcompile-v610-recovery-preparation-1',complete=False,stage=stage,
                inventory_design_count=len(inventory['designs']),reused_design_count=len(cached),
                ready_contexts=len(contexts),prepared_designs=len(ready),failures=list(failures),excluded=[],
                policy=policy(a),recovery_source=str(a.recovery_source))
            write(a.out/'preparation_report.json',report)
            write(a.out/'prepared_inputs.json',dict(designs=sorted(ready,key=lambda r:r['design'])))
            emit('recovery_progress',stage_name=stage,reused_designs=len(cached),ready_contexts=len(contexts),
                 qualified_designs=len(ready),failed=len(failures))
    def one(record):
        try:
            value=prepare_design(a,record,recovery)
            with lock:ready.append(value)
        except Exception as error:
            with lock:failures.append(_failure(record,'recovery_qualification',error))
            if storage_error(error):raise
        finally:progress('qualification')
    parallel(one,records,a.prepare_jobs)
    write(a.out/'boundary_plan.json',dict(designs=sorted(ready,key=lambda r:r['design'])))
    if failures:
        write(a.out/'contexts.partial.json',dict(contexts=contexts))
        raise RuntimeError(f'Recovery qualification incomplete: {len(failures)} failures; reusable successes retained')
    groups={}
    for record in ready:
        for width in record['widths']:
            groups.setdefault((record['device_width'],record['device_height'],width),[]).append(record)
    # The original bounded worker validates all consumers before reclaiming RRG.
    # Processes limit memory independently from VPR qualification concurrency.
    import concurrent.futures as cf
    import multiprocessing as mp
    graph_args=copy(a);graph_args.timeout=a.calibration_timeout
    tasks=[(graph_args,sorted(g,key=lambda r:r['design']),key[2]) for key,g in sorted(groups.items())]
    if tasks:
        from generation import _terminate_pool
        pool=cf.ProcessPoolExecutor(max_workers=min(a.rr_jobs,len(tasks)),mp_context=mp.get_context('spawn'))
        iterator=iter(tasks);pending={};pressure=False
        def submit():
            try:task=next(iterator)
            except StopIteration:return
            check_free_space(a.out);pending[pool.submit(graph_group_worker,task)]=task
        try:
            for _ in range(min(a.rr_jobs,len(tasks))):submit()
            while pending:
                done,_=cf.wait(pending,timeout=5,return_when=cf.FIRST_COMPLETED)
                for future in done:
                    task=pending.pop(future)
                    try:contexts.extend(future.result())
                    except Exception as error:
                        failures.extend(dict(_failure(r,'graph',error),width=task[2]) for r in task[1])
                        pressure=pressure or storage_error(error)
                    if not pressure:submit()
                progress('graphs')
            pool.shutdown()
        except BaseException:
            _terminate_pool(pool);raise
        if pressure:raise StoragePressureError('Recovery graph dispatch stopped on disk pressure')
    contexts.sort(key=lambda c:(c['design'],c['width']))
    complete={r['design'] for r in inventory['designs'] if
        (group:=[c for c in contexts if c['design']==r['design']]) and
        sorted(c['width'] for c in group)==sorted(group[0]['widths'])}
    report=dict(schema='tailcompile-v610-recovery-preparation-1',complete=not failures and len(complete)==len(inventory['designs']),
        stage='complete' if not failures else 'incomplete',inventory_design_count=len(inventory['designs']),
        ready_design_count=len(complete),context_count=len(contexts),failures=failures,excluded=[],
        reused_context_count=len(recovery['contexts']),policy=policy(a),no_training=True,teacher_trials=0,
        recovery_source=str(a.recovery_source),
        benchmark_status=[dict(design=r['design'],family=r['family'],status='ready' if r['design'] in complete else 'incomplete') for r in inventory['designs']])
    write(a.out/'preparation_report.json',report)
    write(a.out/'contexts.partial.json',dict(contexts=contexts))
    if not report['complete']:
        raise RuntimeError('Recovery graph preparation incomplete; completed consumers retained')
    freeze(a.out/'contexts.json',dict(schema='tailcompile-v610-contexts-1',contexts=contexts))
    emit('prepare_complete',designs=len(complete),contexts=len(contexts),reused_contexts=len(recovery['contexts']))
    return dict(contexts=contexts,report=report)
