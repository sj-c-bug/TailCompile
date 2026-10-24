"""Frozen checkpoint sweep: current benchmarks + Koios, with inference-start timing.

No training entrypoint. The sweep is exploratory and retains checkpoint exposure.
"""
from __future__ import annotations

import argparse
from collections import Counter
import concurrent.futures as cf
import csv
import math
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import sys
import time

from bridge import (HERE, V68, V69, REPO, e, pt, emit, execute, fingerprint,
    freeze, read, recover, sha, write, load_module, metrics, check_free_space,
    log_storage_failure, StoragePressureError, storage_error, parallel, prune_trial)

helpers = load_module('v610_v69_helpers', V69/'run.py')
RUNTIME_SCOPE = ('GNN inference (logic AND device encoders), matching/repair/coordinates, '
                 'initializer serialization/writing, plus VPR; excludes Python/Torch startup, '
                 'model construction/loading and graph/device/metadata loading/transfer. '
                 'Reusable RRG/graph preparation separately reported.')


def implementation():
    paths = list(HERE.glob('*.py'))
    paths += [V69/name for name in ('run.py','metrics.py','phase_metrics.py','fast_model.py')]
    paths += [V68/name for name in ('model.py','data.py','experiment.py','common.py',
                                  'parallel_timing.py','lifecycle.py','storage.py','generation.py')]
    return {str(p.relative_to(REPO)):sha(p) for p in sorted(paths) if not p.name.startswith('test_')}


def options(a):
    return dict(decoder=a.decoder, soft_density_passes=2, density_price_weight=1.,
                density_step=1., repair_passes=a.repair_passes, repair_topk=a.repair_topk)


def prepare(a):
    from prepare import prepare as prepare_inputs
    e.identity(a)
    if not a.checkpoint.is_file():
        raise FileNotFoundError(f'Frozen checkpoint not found: {a.checkpoint}')
    import torch
    saved_model = torch.load(a.checkpoint, map_location='cpu', weights_only=True)
    a.checkpoint_metadata = {k:saved_model.get(k,[]) for k in ('training_families','validation_families')}
    del saved_model
    protocol = dict(schema='tailcompile-v610-frozen-sweep-1', identity=a.identity,
        checkpoint=str(a.checkpoint.resolve()), checkpoint_sha256=sha(a.checkpoint),
        source=str(a.source.resolve()), designs=sorted(a.designs) if a.designs else None,
        scope=a.scope, options=options(a), seeds=a.seeds, calibration_seeds=a.calibration_seeds,
        tile_size=a.tile_size, width_start=a.width_start, width_max=a.width_max,
        width_step=a.width_step, half_band=a.half_band, timeout=a.timeout,
        methods=['default','fast'], code=implementation(), training_permitted=False,
        checkpoint_metadata=a.checkpoint_metadata,
        budgets=dict(metrics.DEFAULT_BUDGETS), runtime_metric=RUNTIME_SCOPE,
        evaluation_role='Exploratory frozen-weight coverage sweep; seen and unseen families reported separately',
        timing_selection='Default-only reference width and highest frozen scan width; no GNN-success filtering')
    freeze(a.out/'protocol.json', protocol)
    result = prepare_inputs(a)
    contexts = result['contexts']
    if not contexts:
        raise RuntimeError('No usable contexts; inspect preparation_report.json')
    digests={str(Path(c[k]).resolve()):sha(c[k]) for c in contexts for k in ('graph','logic','device','supply')}
    freeze(a.out/'prepared_signature.json', dict(contexts_sha256=sha(a.out/'contexts.json'),
        artifact_sha256=digests,
        inventory_sha256=sha(a.out/'inventory.json'), checkpoint_sha256=protocol['checkpoint_sha256']))
    emit('prepared', designs=len({c['design'] for c in contexts}), conditions=len(contexts),
         scan_trials=len(contexts)*len(a.seeds)*2, checkpoint_sha256=protocol['checkpoint_sha256'])
    return result


def load(a, vpr=False):
    p = read(a.out/'protocol.json')
    for name, digest in p['code'].items():
        if sha(REPO/name) != digest:
            raise ValueError(f'Frozen implementation changed: {name}; use a new cohort')
    a.checkpoint = Path(p['checkpoint'])
    if sha(a.checkpoint) != p['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed; training/weight substitution is prohibited')
    saved = read(a.out/'prepared_signature.json')
    if sha(a.out/'contexts.json') != saved['contexts_sha256'] or sha(a.out/'inventory.json') != saved['inventory_sha256']:
        raise ValueError('Prepared sweep inputs changed')
    contexts = read(a.out/'contexts.json')['contexts']
    for path,digest in saved['artifact_sha256'].items():
        if sha(path)!=digest:
            raise ValueError(f'Prepared graph/IR changed: {path}')
    keys = [(c['design'], c['width']) for c in contexts]
    if len(set(keys)) != len(keys):
        raise ValueError('Duplicate sweep context')
    a.timeout = p['timeout']
    if vpr:
        e.identity(a)
        if a.identity != p['identity']:
            raise ValueError('Backend/architecture differs from prepared sweep')
        checked = {}
        for c in contexts:
            path = c['mapped_blif']
            if path not in checked:
                checked[path] = sha(path)
            if checked[path] != c['mapped_blif_sha256']:
                raise ValueError(f'Canonical mapped input changed: {path}')
    return p, contexts


def generate_all(a, p, contexts, *, report_name='generation_report.json'):
    from generate import signature, validate_generation, worker
    from generation import _terminate_pool
    results, tasks, failures = {}, [], []
    root = a.out/'generation'
    for c in contexts:
        folder = root/c['design']/f'w{c["width"]}'/'c0'
        expected = signature(c, a.checkpoint, p['options'], 0)
        old = validate_generation(folder, expected)
        if old:
            results[c['design'],c['width'],0] = old
        elif (folder/'generation_error.json').exists() and not a.retry_failed:
            failed = read(folder/'generation_error.json')
            if failed['signature'] != expected:
                raise ValueError(f'Failed generation identity changed: {folder}')
            failures.append(failed)
        else:
            tasks.append((c, str(a.checkpoint), str(folder), p['options'], 0))
    emit('generation_start', reused=len(results), pending=len(tasks), failed=len(failures), jobs=a.generation_jobs)
    if tasks:
        pool = cf.ProcessPoolExecutor(max_workers=min(a.generation_jobs,len(tasks)),
                                      mp_context=mp.get_context('spawn'))
        iterator = iter(tasks)
        pending = {}
        old_handler = signal.getsignal(signal.SIGTERM)
        def stop(signum, frame):
            raise KeyboardInterrupt('Stopping generation workers; complete artifacts retained')
        signal.signal(signal.SIGTERM, stop)
        def submit():
            try:
                task = next(iterator)
            except StopIteration:
                return
            check_free_space(a.out)
            pending[pool.submit(worker, task)] = task
        try:
            for _ in range(min(a.generation_jobs,len(tasks))):
                submit()
            while pending:
                done, _ = cf.wait(pending, timeout=5, return_when=cf.FIRST_COMPLETED)
                for future in done:
                    task = pending.pop(future)
                    c, _, folder, _, _ = task
                    try:
                        value = future.result()
                        results[c['design'],c['width'],0] = value
                        error_file = Path(folder)/'generation_error.json'
                        if error_file.exists():
                            error_file.unlink()
                    except Exception as error:
                        if storage_error(error) or log_storage_failure(str(error),1):
                            raise StoragePressureError(str(error)) from error
                        value = dict(design=c['design'],width=c['width'],status='generation_failure',
                            error=str(error),signature=signature(c,a.checkpoint,p['options'],0))
                        write(Path(folder)/'generation_error.json',value)
                        failures.append(value)
                        emit('generation_failure', design=c['design'],width=c['width'],error=str(error))
                    submit()
                progress=dict(expected=len(contexts),completed=len(results),failed=len(failures),
                              active=len(pending),complete=len(results)+len(failures)==len(contexts))
                write(root/('generation_progress.json' if report_name=='generation_report.json' else 'timing_generation_progress.json'),progress)
                emit('generation_progress',**progress)
            pool.shutdown()
        except BaseException:
            _terminate_pool(pool)
            raise
        finally:
            signal.signal(signal.SIGTERM,old_handler)
    write(root/report_name,dict(complete=True,successful=len(results),failures=failures,
        expected=len(contexts),checkpoint_sha256=p['checkpoint_sha256'],
        diagnostics=[dict(design=d,width=w,before=g.get('capacity_diagnostics_before'),
                          after=g.get('capacity_diagnostics'),repair=g.get('repair_diagnostics'),
                          compute_generation_seconds=g.get('compute_generation_seconds'))
                     for (d,w,k),g in sorted(results.items())]))
    return results


def evaluate(a):
    p, contexts = load(a, vpr=True)
    generated = generate_all(a,p,contexts)
    freeze(a.out/'evaluation_plan.json',dict(checkpoint_sha256=p['checkpoint_sha256'],
        contexts=[dict(design=c['design'],family=c['family'],width=c['width']) for c in contexts],
        seeds=p['seeds'],methods=p['methods']))
    completed = 0
    def one(task):
        nonlocal completed
        c,seed,method = task
        folder = a.out/'evaluation/runs'/c['design']/f'w{c["width"]}'/f's{seed}'/method
        generation = generated.get((c['design'],c['width'],0))
        if method=='fast' and not generation:
            row=dict(design=c['design'],family=c['family'],width=c['width'],seed=seed,method=method,
                     status='generation_failure',cpd=None,wirelength=None)
            write(folder/'metrics.json',row)
        else:
            init = generation['ap_init'] if method=='fast' else None
            if not (folder/'trial.json').exists() or a.retry_failed or a.retry_timeouts:
                check_free_space(a.out)
            result=execute(e.vpr_command(a,c['mapped_blif'],c['width'],seed,c['device_width']),folder,
                dict(a.identity,stage='route',protocol_sha256=fingerprint(p),design=c['design'],
                     width=c['width'],seed=seed,method=method,
                     blif_sha256=c['mapped_blif_sha256'],initializer_sha256=sha(init) if init else None),
                env={'TC_GP_INIT':init} if init else None,timeout=a.timeout,heartbeat=a.heartbeat,
                retry_failed=a.retry_failed,retry_timeouts=a.retry_timeouts)
            row=helpers.parse_row(result,folder,c,seed,method)
            prune_trial(folder,a.out)
        completed += 1
        emit('scan_progress',completed=completed,expected=len(contexts)*len(p['seeds'])*2,
             design=c['design'],width=c['width'],seed=seed,method=method,status=row['status'])
        return row
    try:
        parallel(one,[(c,s,m) for c in contexts for s in p['seeds']
                      for m in (p['methods'] if s%2 else p['methods'][::-1])],a.jobs)
    finally:
        scan_report(a)


def scan_report(a):
    p,contexts=load(a)
    grids={}
    rows=[]
    for c in contexts:
        grids.setdefault(c['design'],[]).append(c['width'])
        for seed in p['seeds']:
            for method in p['methods']:
                path=a.out/'evaluation/runs'/c['design']/f'w{c["width"]}'/f's{seed}'/method/'metrics.json'
                if path.exists():
                    rows.append(read(path))
    result=metrics.aggregate_widths(rows,seeds=p['seeds'],widths_by_design=grids,methods=p['methods'])
    metadata={c['design']:c for c in contexts}
    for group in result['groups']:
        c=metadata[group['design']]
        group.update(family=c['family'],checkpoint_exposure=c.get('checkpoint_exposure'),
                     original_split=c.get('original_split'))
    result.update(schema='tailcompile-v610-width-1',execution_complete=len(rows)==len(contexts)*len(p['seeds'])*2,
        status_counts=dict(Counter(r['status'] for r in rows)),recorded_trials=len(rows),
        expected_trials=len(contexts)*len(p['seeds'])*2,
        checkpoint_sha256=p['checkpoint_sha256'],evaluation_role=p['evaluation_role'])
    write(a.out/'evaluation_report.json',result)
    return result


def timing_contexts(contexts):
    selected=[]
    for design in sorted({c['design'] for c in contexts}):
        group=sorted((c for c in contexts if c['design']==design),key=lambda c:c['width'])
        reference=group[0]['reference_width']
        near=next((c for c in group if c['width']>=reference),group[-1])
        selected.append(near)
        if group[-1]['width']!=near['width']:
            selected.append(group[-1])
    return selected


def cost_fields(result, cost):
    if cost is None:
        return dict(vpr_seconds=None,generation_seconds=None,total_seconds=None,
                    excluded_setup_seconds=None,raw_worker_seconds=result['wall_seconds'])
    vpr,compute=cost['vpr_seconds'],cost['compute_generation_seconds']
    if not all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in (vpr,compute)):
        raise ValueError('Invalid measured inference/VPR cost')
    return dict(vpr_seconds=vpr,generation_seconds=compute,total_seconds=vpr+compute,
                excluded_setup_seconds=cost.get('excluded_setup_seconds'),
                raw_worker_seconds=result['wall_seconds'],runtime_scope=RUNTIME_SCOPE)


def timed_trial(a,p,c,g,lane,method,folder,protocol,gate):
    command=[str(x) for x in e.vpr_command(a,c['mapped_blif'],c['width'],lane['seed'],c['device_width'])]
    spec=dict(cpu=lane['cpu'],context=c,method=method,checkpoint=str(a.checkpoint),candidate=0,
        options=p['options'],expected_ap_sha256=g['ap_init_sha256'],vpr_command=command,
        start_gate=str(gate),ready_file=str(folder/'ready.json'),setup_timeout=a.setup_timeout,
        measured_timeout=a.timeout)
    freeze(folder/'spec.json',spec)
    result=execute([sys.executable,HERE/'timed_worker.py','--spec',folder/'spec.json'],folder,
        dict(a.identity,stage='route',protocol_sha256=fingerprint(protocol),spec_sha256=fingerprint(spec)),
        timeout=a.timeout+a.setup_timeout+60,heartbeat=a.heartbeat)
    row=helpers.parse_row(result,folder,c,lane['seed'],method)
    affinity=read(folder/'affinity.json') if (folder/'affinity.json').exists() else None
    if affinity is None or affinity['allowed_cpus']!=[lane['cpu']] or set(affinity['thread_env'].values())!={'1'}:
        raise RuntimeError(f'CPU/thread isolation failed: {folder}')
    cost=read(folder/'cost.json') if (folder/'cost.json').exists() else None
    row.update(cost_fields(result,cost),cpu=lane['cpu'],timing_protocol=fingerprint(protocol),cost=cost)
    write(folder/'metrics.json',row)
    return row


def timing(a):
    p,contexts=load(a,vpr=True)
    width_report=scan_report(a)
    if not width_report['execution_complete']:
        raise ValueError('Finish the planned scan before formal timing (timeouts remain censored, not rerun automatically)')
    selected=timing_contexts(contexts)
    generated=generate_all(a,p,selected,report_name='timing_generation_report.json')
    units=[dict(design=c['design'],width=c['width'],seed=s,candidate=0)
           for c in selected if (c['design'],c['width'],0) in generated for s in p['seeds']]
    if not units:
        raise ValueError('No generated candidates available for paired timing')
    hw=pt.hardware(a.timing_jobs,a.cpu_list,len(units))
    root=a.out/'timing'/a.run_name
    protocol=dict(schema='tailcompile-v610-preloaded-pairs-1',hardware=hw,
        parent_protocol_sha256=fingerprint(p),checkpoint_sha256=p['checkpoint_sha256'],
        timeout=a.timeout,setup_timeout=a.setup_timeout,seeds=p['seeds'],units=units,
        scope=RUNTIME_SCOPE,policy='Load all lanes, then start-gate; same physical core AB/BA; partial waves rerun',
        generation_sha256={f'{d}/w{w}':g['ap_init_sha256'] for (d,w,k),g in generated.items()})
    freeze(root/'protocol.json',protocol)
    waves=helpers.make_waves(units,hw['selected_cores'])
    freeze(root/'schedule.json',dict(waves=waves))
    cases={(c['design'],c['width']):c for c in selected}
    all_rows=[]
    for wave in waves:
        old=pt.completed_wave(root,wave)
        if old and not pt.retry_wave(a,old):
            all_rows.extend(old['results'].values())
            emit('reuse_wave',wave=wave['index'])
            continue
        pt.archive_wave(root,wave,a.confirm_remote_stopped)
        folder=pt.wave_folder(root,wave)
        results={};evidence={};before=pt.snapshot();started=time.monotonic()
        for half in (0,1):
            half_root=folder/f'half{half}'
            gate=half_root/'go.json'
            check_free_space(a.out)
            emit('timing_loading',wave=wave['index'],waves=len(waves),half=half,lanes=len(wave['lanes']))
            def one(lane):
                c=cases[lane['design'],lane['width']]
                directory=half_root/f'lane{lane["lane"]:02d}'
                row=timed_trial(a,p,c,generated[c['design'],c['width'],0],lane,
                    lane['order'][half],directory,protocol,gate)
                return lane,row,directory
            with cf.ThreadPoolExecutor(max_workers=len(wave['lanes'])) as pool:
                futures=[pool.submit(one,l) for l in wave['lanes']]
                try:
                    deadline=time.monotonic()+a.setup_timeout
                    while not all((half_root/f'lane{l["lane"]:02d}'/'ready.json').exists() for l in wave['lanes']):
                        if any(f.done() for f in futures):
                            raise RuntimeError(f'Worker failed during preload; inspect {half_root}')
                        if time.monotonic()>deadline:
                            raise TimeoutError(f'Preload barrier timed out: {half_root}')
                        time.sleep(.2)
                    write(gate,dict(go=True,ready_lanes=len(wave['lanes']),unix_time=time.time()))
                    emit('timing_measured_start',wave=wave['index'],half=half,lanes=len(wave['lanes']))
                    errors=[]
                    for future in cf.as_completed(futures):
                        try:
                            lane,row,directory=future.result()
                            results[f'{half}:{lane["lane"]}']=row
                            for name in ('trial.json','metrics.json','affinity.json','spec.json','ready.json','cost.json'):
                                if (directory/name).exists():
                                    evidence[str((directory/name).relative_to(folder))]=sha(directory/name)
                        except Exception as error:
                            errors.append(str(error))
                    if errors:
                        raise RuntimeError(f'Incomplete timing wave retained: {errors[0]}')
                except BaseException:
                    if not gate.exists():
                        write(gate,dict(go=False,abort=True))
                    raise
                finally:
                    # Cleanup only after every measured worker in this half drained.
                    for future in futures:
                        try:
                            future.result()
                        except BaseException:
                            pass
                    for lane in wave['lanes']:
                        directory=half_root/f'lane{lane["lane"]:02d}'
                        if (directory/'metrics.json').exists():
                            prune_trial(directory,a.out)
        record=dict(complete=True,wave=wave,results=results,evidence_sha256=evidence,
            wall_seconds=time.monotonic()-started,load_before=before,load_after=pt.snapshot())
        write(folder/'wave.json',record)
        all_rows.extend(results.values())
        emit('wave_complete',wave=wave['index'],waves=len(waves),trials=len(results))
    write(root/'rows.json',dict(complete=True,rows=all_rows,protocol_sha256=fingerprint(protocol)))
    return runtime_report(a,p,contexts,all_rows,protocol)


def grouped_summaries(rows, field):
    grouped=[]
    for key in sorted({str(r.get(field,'unknown')) for r in rows}):
        members=[r for r in rows if str(r.get(field,'unknown'))==key]
        ratios={}
        for metric in ('vpr','total','cpd','wirelength'):
            values=[r['candidate_over_baseline_geomean'][metric] for r in members]
            ratios[metric]=(math.exp(sum(math.log(v) for v in values)/len(values))
                if all(v is not None and v>0 for v in values) else None)
        grouped.append(dict(group=key,designs=[r['design'] for r in members],
            candidate_over_baseline_geomean=ratios,
            successes_lost=sum(r['success_loss_pairs'] for r in members),
            successes_gained=sum(r['success_gain_pairs'] for r in members),
            designs_passed=sum(r['passed'] for r in members),
            note='Descriptive equal-design average within group, not independent-family significance'))
    return grouped


def runtime_report(a,p=None,contexts=None,rows=None,protocol=None):
    if p is None:
        p,contexts=load(a)
    root=a.out/'timing'/a.run_name
    if rows is None:
        rows=read(root/'rows.json')['rows']
        protocol=read(root/'protocol.json')
    expected=[dict(design=c['design'],width=c['width'],seed=s)
              for c in timing_contexts(contexts) for s in p['seeds']]
    report=metrics.compare_runtime(rows,expected_conditions=expected,budgets=p['budgets'],width_report=scan_report(a))
    designs={c['design']:c for c in contexts}
    for row in report['by_design']:
        c=designs[row['design']]
        row.update(family=c['family'],original_split=c.get('original_split'),
                   checkpoint_exposure=c.get('checkpoint_exposure',c.get('exposure')))
    report.update(schema='tailcompile-v610-runtime-1',runtime_scope=RUNTIME_SCOPE,protocol=protocol,
        checkpoint_sha256=p['checkpoint_sha256'],evaluation_role=p['evaluation_role'],
        preparation_report=read(a.out/'preparation_report.json'),raw_rows=rows,
        startup_and_loading_included=False,device_gnn_inference_included=True,
        note='Exploratory per-benchmark coverage. Related Koios variants are not independent families; no training performed.')
    report['by_family']=grouped_summaries(report['by_design'],'family')
    report['by_checkpoint_exposure']=grouped_summaries(report['by_design'],'checkpoint_exposure')
    report['execution_complete']=len(rows)==len(expected)*2
    write(root/'runtime_report.json',report)
    columns=['design','family','original_split','checkpoint_exposure','default_mean_wmin','gnn_mean_wmin',
             'wmin_ratio','vpr_ratio','compute_plus_vpr_ratio','cpd_ratio','wirelength_ratio',
             'joint_success_pairs','success_gain_pairs','success_loss_pairs','passed']
    with (a.out/'benchmark_summary.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns);writer.writeheader()
        for r in report['by_design']:
            g=r['width_guard'];ratios=r['candidate_over_baseline_geomean']
            writer.writerow(dict(design=r['design'],family=r['family'],original_split=r['original_split'],
                checkpoint_exposure=r['checkpoint_exposure'],default_mean_wmin=g.get('baseline_mean'),
                gnn_mean_wmin=g.get('candidate_mean'),wmin_ratio=g['ratio'],vpr_ratio=ratios['vpr'],
                compute_plus_vpr_ratio=ratios['total'],cpd_ratio=ratios['cpd'],wirelength_ratio=ratios['wirelength'],
                joint_success_pairs=r['joint_success_pairs'],success_gain_pairs=r['success_gain_pairs'],
                success_loss_pairs=r['success_loss_pairs'],passed=r['passed']))
    write(a.out/'benchmark_summary.json',report)
    emit('runtime_report',path=str(root/'runtime_report.json'),complete=report['complete'],passed=report['passed'])
    return report


def status(a):
    result={}
    for key,name in [('prepare','preparation_report.json'),('generation','generation/generation_report.json'),
                     ('scan','evaluation_report.json'),('timing',f'timing/{a.run_name}/runtime_report.json')]:
        path=a.out/name
        if path.exists():
            value=read(path)
            result[key]={k:value.get(k) for k in ('complete','execution_complete','passed','expected_trials','recorded_trials') if k in value}
        else:
            result[key]={'exists':False}
    root=a.out/'timing'/a.run_name
    if (root/'schedule.json').exists():
        waves=read(root/'schedule.json')['waves']
        result['timing'].update(waves_expected=len(waves),waves_complete=sum(pt.completed_wave(root,w) is not None for w in waves))
    emit('status',**result)


def parse_args(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=['inventory','prepare','generate','evaluate','timing','report','status','recover','all'])
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--source',type=Path,default=V68/'runs/hier_init_001')
    parser.add_argument('--checkpoint',type=Path,default=V69/'runs/runtime_001/models/runtime/generator.pt')
    parser.add_argument('--vtr-root',dest='vtr',type=Path,default=os.environ.get('VTR_ROOT'))
    parser.add_argument('--scope',choices=['all','koios','existing'],default='all')
    parser.add_argument('--designs',nargs='+')
    for name in ('jobs','prepare-jobs','rr-jobs','generation-jobs','timing-jobs'):
        parser.add_argument('--'+name,type=int,default=32)
    parser.add_argument('--cpu-list')
    parser.add_argument('--run-name',default='preloaded_001')
    parser.add_argument('--timeout',type=float,default=7200)
    parser.add_argument('--setup-timeout',type=float,default=1800)
    parser.add_argument('--heartbeat',type=float,default=60)
    parser.add_argument('--seeds',type=int,nargs='+',default=[411,412,413,414,415])
    parser.add_argument('--calibration-seeds',type=int,nargs='+',default=[401,402])
    parser.add_argument('--tile-size',type=int,default=4)
    parser.add_argument('--width-start',type=int,default=48)
    parser.add_argument('--width-max',type=int,default=512)
    parser.add_argument('--width-step',type=int,default=4)
    parser.add_argument('--half-band',type=int,default=3)
    parser.add_argument('--decoder',choices=['balanced','legacy_soft'],default='balanced')
    parser.add_argument('--repair-passes',type=int,default=2)
    parser.add_argument('--repair-topk',type=int,default=32)
    for name in ('retry-failed','retry-timeouts','confirm-remote-stopped'):
        parser.add_argument('--'+name,action='store_true')
    a=parser.parse_args(argv);a.out=a.out.resolve()
    a.out.relative_to(REPO/'experiments/vtr/tailcompile_v610/runs')
    if len(a.seeds)!=5 or len(set(a.seeds))!=5 or set(a.seeds)&set(a.calibration_seeds):
        parser.error('Use five distinct evaluation seeds disjoint from calibration')
    if not a.calibration_seeds or len(set(a.calibration_seeds))!=len(a.calibration_seeds):
        parser.error('Calibration seeds must be nonempty and unique')
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.run_name):
        parser.error('Unsafe timing run name')
    if not all(math.isfinite(x) for x in (a.timeout,a.setup_timeout,a.heartbeat)):
        parser.error('Timeouts and heartbeat must be finite')
    if min(a.jobs,a.prepare_jobs,a.rr_jobs,a.generation_jobs,a.timing_jobs,a.timeout,a.setup_timeout,
           a.heartbeat,a.tile_size,a.width_start,a.width_max,a.width_step,a.repair_topk)<=0:
        parser.error('Budgets must be positive')
    if a.width_max<a.width_start or a.half_band<1 or not 0<=a.repair_passes<=8:
        parser.error('Invalid width/repair budget')
    return a


def main():
    a=parse_args()
    emit('parallelism',vpr=a.jobs,prepare=a.prepare_jobs,rr=a.rr_jobs,generation=a.generation_jobs,
         timing_requested=a.timing_jobs,vpr_workers=1,training=False)
    if a.phase=='recover':
        pt.assert_workers_stopped(a.out,a.confirm_remote_stopped);recover(a.out,a.confirm_remote_stopped)
    elif a.phase=='inventory':
        from inventory import discover
        write(a.out/'inventory_preview.json',discover(a.vtr,a.source,designs=a.designs,scope=a.scope))
    elif a.phase=='generate':
        p,c=load(a);generate_all(a,p,c)
    elif a.phase=='report':
        scan_report(a)
        if (a.out/'timing'/a.run_name/'rows.json').exists():
            runtime_report(a)
    elif a.phase=='all':
        prepare(a);evaluate(a);timing(a)
    else:
        {'prepare':prepare,'evaluate':evaluate,'timing':timing,'status':status}[a.phase](a)


if __name__=='__main__':
    main()
