"""v6.8: CPU preparation, teacher collection, real feedback and paired VPR.

Final-test teachers never enter a tensor bundle. The native AP optimizer is
retained in both arms; TC_GP_INIT changes its initialization, not its result.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import gzip
import math
import os
from pathlib import Path
import re
import shutil
import socket
import statistics
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
from common import (COMMIT, classify, emit, execute, fingerprint, freeze,
                    read, recover, sha, write)
from analyze_tailcompile_runtime import parse_vpr_log
from scheduling import DEFAULT_JOBS, compatible_protocol, resolve_concurrency
from lifecycle import (StoragePressureError, after_metrics, check_free_space,
                       log_storage_failure, parallel, record_policy)


def identity(a):
    from install_hook import verify
    if not a.vtr:
        raise ValueError('Set VTR_ROOT or pass --vtr-root')
    a.vtr = a.vtr.resolve()
    hook = verify(a.vtr)
    a.binary = a.vtr / 'vpr/vpr'
    a.arch = a.vtr / 'vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml'
    preflight=a.out/'preflight';preflight.mkdir(parents=True,exist_ok=True)
    check = subprocess.run([str(a.binary), '--help'], cwd=preflight,capture_output=True, text=True, timeout=30)
    if check.returncode:
        raise RuntimeError(f'VPR cannot start. Check shared libraries: {check.stderr[-2000:]}')
    a.identity = dict(vtr_commit=COMMIT, binary_sha256=sha(a.binary),
                      architecture_sha256=sha(a.arch), hook=hook)


def cohort(a):
    from data import load_config, validate_splits
    config = load_config(a.cohort)
    validate_splits(config)
    if a.designs:
        selected = [r for r in config['designs'] if r['design'] in a.designs]
        if {r['design'] for r in selected} != set(a.designs):
            raise ValueError('Unknown design in --designs')
        config = dict(config, designs=selected, exploratory_subset=True)
    freeze(a.out / 'cohort.json', config)
    return config['designs']


def protocol(a):
    designs = cohort(a)
    frozen = dict(schema='tailcompile-v68-protocol-1', identity=a.identity,
        cohort_sha256=sha(a.out/'cohort.json'), tile_size=a.tile_size,
        calibration_seeds=a.calibration_seeds, teacher_seeds=a.teacher_seeds,
        feedback_seeds=a.feedback_seeds, evaluation_seeds=a.seeds,
        width_step=a.width_step, width_start=a.width_start, width_max=a.width_max,
        half_band=a.half_band, timing_guard=a.timing_guard, wire_guard=a.wire_guard,
        vpr_threads=1, legalizer='appack',
        source={p.name:sha(p) for p in [HERE/'experiment.py', HERE/'data.py', HERE/'common.py', HERE/'scheduling.py']},
        timing_note='Parallel throughput wall times are not serial speedup measurements.')
    all_seeds = a.calibration_seeds+a.teacher_seeds+a.feedback_seeds+a.seeds
    if len(set(all_seeds)) != len(all_seeds):
        raise ValueError('Calibration, teacher, feedback and final evaluation seed sets must be disjoint')
    if (a.out/'protocol.json').exists():
        frozen, compatible = compatible_protocol(read(a.out/'protocol.json'), frozen)
        if compatible:
            emit('compatible_scheduler_update', frozen_protocol_preserved=True,
                 note='Only audited concurrency changes; completed inputs and VPR trials reused')
    freeze(a.out/'protocol.json', frozen)
    write(a.out/'execution_settings.json', dict(jobs=a.jobs, prepare_jobs=a.prepare_jobs,
        rr_jobs=a.rr_jobs, graph_jobs=a.prepare_jobs, generation_jobs=a.generation_jobs,
        vpr_workers=1,
        actual_experiment_sha256=sha(Path(__file__)), scheduling_sha256=sha(HERE/'scheduling.py'),
        recorded_unix=time.time()))
    return designs


def call(a, command, folder, signature, *, env=None, marker=None):
    previous = Path(folder)/'trial.json'
    saved = read(previous) if previous.exists() else None
    retry = saved and ((saved['status']=='timeout' and a.retry_timeouts) or
        (saved['status'] not in {'success','unroutable','complete','timeout'} and a.retry_failed))
    # Cached trials can still be parsed/pruned on a low-space filesystem.
    if saved is None or retry:
        check_free_space(a.out)
    return execute(command, folder, dict(a.identity, **signature), env=env,
        timeout=a.timeout, heartbeat=a.heartbeat, retry_failed=a.retry_failed,
        retry_timeouts=a.retry_timeouts, required_marker=marker)


def vpr_command(a, blif, width, seed, device_width=None):
    command = [a.binary, a.arch, blif, '--analytical_place', '--route', '--analysis',
        '--ap_full_legalizer', 'appack', '--ap_detailed_placer', 'annealer',
        '--num_workers', '1', '--seed', str(seed), '--route_chan_width', str(width)]
    if device_width is not None:
        command += ['--device_width', str(device_width)]
    return command


def paths(a, design):
    folder = a.out/'inputs'/design
    return folder, folder/(design+'.pre-vpr.blif'), folder/'structure'


def input_structure(a, record):
    name = record['design']
    folder, blif, prefix = paths(a, name)
    source = a.vtr/'vtr_flow/benchmarks/verilog'/record['source']
    if not source.is_file():
        raise FileNotFoundError(f'Frozen cohort source missing: {source}; do not silently replace family')
    signature = dict(design=name, source=record['source'], source_sha256=sha(source),
        include_sha256=sha(source.parent/'hard_block_include.v') if record['source'].startswith('koios/') else None)
    freeze(folder/'source.json', signature)
    if not (folder/'mapping.json').exists():
        flow = folder/'mapping'
        command = [sys.executable, a.vtr/'vtr_flow/scripts/run_vtr_flow.py', source, a.arch,
            '-temp_dir', flow, '-ending_stage', 'abc', '-timeout', str(a.timeout)]
        if record['source'].startswith('koios/'):
            command += ['-include', source.parent/'hard_block_include.v']
        result = call(a, command, folder/'synthesis', dict(signature, stage='synthesis'))
        generated = flow/(name+'.abc.blif')
        if result['returncode'] != 0 or not generated.is_file():
            raise RuntimeError(f'Synthesis failed: {folder}/synthesis/vpr.out')
        shutil.copy2(generated, blif)
        write(folder/'mapping.json', dict(signature, mapped_blif_sha256=sha(blif)))
    if read(folder/'mapping.json')['mapped_blif_sha256'] != sha(blif):
        raise ValueError('Mapped BLIF changed')
    result = call(a, vpr_command(a, blif, a.width_start, a.calibration_seeds[0], record.get('device_width')),
        folder/'structure_run', dict(stage='structure', blif_sha256=sha(blif)),
        env=dict(TC_GP_STRUCTURE=prefix, TC_GP_STRUCTURE_ONLY=1),
        marker='TAILCOMPILE_V68_STRUCTURE_ONLY_COMPLETE')
    if result['status'] != 'complete':
        raise RuntimeError(f'Pre-GP structure export failed: {folder}/structure_run/vpr.out')
    from data import read_ap_snapshot
    snapshot = read_ap_snapshot(prefix, structure=True)
    # Width/height are architectural dimensions, not post-route features.
    width, height = snapshot['size']
    value = dict(record, mapped_blif=str(blif), structure_prefix=str(prefix),
                 device_width=width, device_height=height, mapped_blif_sha256=sha(blif))
    freeze(folder/'prepared.json', value)
    emit('structure_ready', design=name, grid=[width, height])
    return value


def route(a, record, width, seed, directory, *, init=None, teacher=False, extra=None):
    directory = Path(directory)
    env = {}
    if teacher:
        if record['split'] == 'final_test':
            raise ValueError('Final-test GP teachers are forbidden')
        env['TC_GP_EXPORT'] = directory/'gp'
    if init:
        env['TC_GP_INIT'] = init
    result = call(a, vpr_command(a, record['mapped_blif'], width, seed, record['device_width']), directory,
        dict(stage='route', design=record['design'], width=width, seed=seed,
             mapped_blif_sha256=sha(record['mapped_blif']),
             initializer_sha256=sha(init) if init else None,
             teacher=teacher, device_width=record['device_width'], **(extra or {})), env=env)
    text = (directory/'vpr.out').read_text(encoding='utf-8', errors='replace')
    if init and result['status'] in {'success','unroutable'}:
        if ('TAILCOMPILE_V68_GP_INIT_CONSUMED' not in text or
            'TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1' not in text or
            'skipping the Global Placement' in text):
            raise RuntimeError(f'Initializer was not consumed by native GP: {directory}')
    if teacher and result['status'] in {'success','unroutable'} and 'TAILCOMPILE_V68_EXPORTED stage=post_gp_teacher' not in text:
        raise RuntimeError(f'Missing GP teacher export: {directory}')
    def metric(pattern):
        values = re.findall(pattern, text)
        return float(values[-1]) if values else None
    row = dict(result, design=record['design'], family=record['family'], split=record['split'],
        width=width, seed=seed, cpd=metric(r'Final critical path delay \(least slack\):\s*([\d.eE+-]+)'),
        wirelength=metric(r'Total wirelength:\s*([\d.eE+-]+)'),
        native_gp_continued=bool(init and 'TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1' in text),
        phases=parse_vpr_log(text),
        teacher_prefix=str(directory/'gp') if teacher else None)
    write(directory/'metrics.json', row)
    # These raw outputs are no longer consumed: resume/feedback use the log,
    # trial and metrics, while teacher gp.* and generated inputs remain intact.
    # Cleanup happens AFTER the VPR stopwatch and never changes its command.
    after_metrics(directory, repo=HERE.parents[2])
    if log_storage_failure(text, result['returncode']):
        raise StoragePressureError(f'VPR reported disk quota/full filesystem: {directory}; '
            'complete evidence retained; no further tasks should be dispatched')
    return row


def screen(a, record):
    """Default-only geometric expansion then quantized boundary bisection."""
    root = a.out/'screen'/record['design']
    def one(width, seed):
        result = route(a, record, width, seed, root/f'w{width}'/f's{seed}')
        if result['status'] not in {'success','unroutable'}:
            raise RuntimeError(f'Cannot calibrate a boundary from {result["status"]}: {root}/w{width}/s{seed}')
        return result['status'] == 'success'
    boundaries = []
    for seed in a.calibration_seeds:
        low, high = 0, a.width_start
        while not one(high, seed):
            low = high
            if high >= a.width_max:
                raise RuntimeError(f'No Default success <= {a.width_max} for {record["design"]}; '
                                   'right-censored, preserve evidence and revise cohort explicitly')
            high = min(a.width_max, math.ceil(high*1.5/a.width_step)*a.width_step)
        while high-low > a.width_step:
            middle = ((low+high)//(2*a.width_step))*a.width_step
            middle = max(low+a.width_step, middle)
            if one(middle, seed):
                high = middle
            else:
                low = middle
        boundaries.append(dict(seed=seed, lowest_success=high, adjacent_failure=low or None))
    center = max(row['lowest_success'] for row in boundaries)
    widths = sorted({max(a.width_step, center+i*a.width_step)
                     for i in range(-a.half_band, a.half_band+1)} | {center+6*a.width_step})
    value = dict(record, calibration=boundaries, reference_width=center, widths=widths,
                 boundary_selection='Default only, calibration seeds disjoint from final evaluation')
    freeze(root/'boundary.json', value)
    return value


def rr_graph(a, record, width):
    # Architecture + fixed grid + width: reusable across designs with that device.
    key = f'{record["device_width"]}x{record["device_height"]}/w{width}'
    folder = a.out/'device_cache'/key
    compressed, raw = folder/'rr_graph.xml.gz', folder/'rr_graph.xml'
    if compressed.exists():
        audit = read(folder/'rr_graph.json')
        if sha(compressed) != audit['sha256'] or audit['identity'] != a.identity:
            raise ValueError('Shared device RRG provenance changed')
        return compressed
    # A pack-only VPR flow can export RRG; it does not run placement or routing.
    command = [a.binary, a.arch, record['mapped_blif'], '--pack', '--num_workers', '1',
        '--device_width', str(record['device_width']), '--route_chan_width', str(width),
        '--place_chan_width', str(width), '--write_rr_graph', raw]
    result = call(a, command, folder/'build', dict(stage='rr_graph', grid=key,
        mapped_blif_sha256=sha(record['mapped_blif'])))
    if result['status'] != 'success' or not raw.is_file():
        raise RuntimeError(f'Pack-only RRG export failed: {folder}/build/vpr.out')
    temporary = compressed.with_suffix('.gz.tmp')
    with raw.open('rb') as src, gzip.open(temporary, 'wb') as dst:
        shutil.copyfileobj(src, dst)
    temporary.replace(compressed)
    write(folder/'rr_graph.json', dict(identity=a.identity, sha256=sha(compressed),
        width=width, grid=[record['device_width'],record['device_height']],
        placement_or_route_used=False, build_wall_seconds=result['wall_seconds']))
    # Only this newly generated, checked XML is removed; compressed source remains.
    raw.unlink()
    return compressed


def graph_worker(task):
    """Top-level entry for independent CPU processes, not GIL-bound threads."""
    request, output, tile_size = task
    from data import build_cached_graph
    built = build_cached_graph(request, output, tile_size=tile_size)
    emit('graph_ready', design=request['design'], width=request['channel_width'], tile_size=tile_size)
    return dict(request, width=request['channel_width'], **built)


def build_graphs(tasks, jobs):
    results, failures = [], []
    with cf.ProcessPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(graph_worker, task) for task in tasks]
        for future in cf.as_completed(futures):
            try:
                results.append(future.result())
            except Exception as error:
                failures.append(str(error));emit('graph_failed', error=str(error))
    if failures:
        raise RuntimeError(f'{len(failures)} graph builds failed; valid caches retained. First: {failures[0]}')
    return sorted(results, key=lambda row:(row['design'], row['width']))


def prepare(a):
    records = protocol(a)
    prepared = parallel(lambda d:input_structure(a,d), records, a.prepare_jobs)
    write(a.out/'prepared_inputs.json', dict(designs=sorted(prepared,key=lambda d:d['design'])))
    screened = parallel(lambda d:screen(a,d), prepared, a.jobs)
    screened.sort(key=lambda d:d['design'])
    freeze(a.out/'boundary_plan.json', dict(designs=screened))
    # Build shared RRGs once, avoiding concurrent writes for identical device keys.
    representatives = {}
    for record in screened:
        for width in record['widths']:
            representatives.setdefault((record['device_width'],record['device_height'],width), (record,width))
    parallel(lambda task:rr_graph(a,*task), representatives.values(), a.rr_jobs)
    graph_tasks = []
    for record in screened:
        for width in record['widths']:
            request = dict(record, channel_width=width, rr_graph=str(rr_graph(a,record,width)),
                           architecture=str(a.arch), vtr_commit=COMMIT)
            graph_tasks.append((request, a.out/'graphs'/record['design']/f'w{width}', a.tile_size))
    contexts = build_graphs(graph_tasks, a.prepare_jobs)
    # Keep the exact pre-update context schema/order, so a scheduling-only
    # upgrade can reuse an existing frozen contexts.json as well as its caches.
    original={(r['design'],w):dict(r,width=w) for r in screened for w in r['widths']}
    contexts=[dict(original[c['design'],c['width']], **{k:c[k] for k in ('graph','logic','device','supply','audit')})
              for c in contexts]
    freeze(a.out/'contexts.json', dict(schema='tailcompile-v68-contexts-1', contexts=contexts))
    teacher_tasks = [(c,s) for c in contexts if c['split'] in ('train','validation') for s in a.teacher_seeds]
    parallel(lambda task:route(a,task[0],task[0]['width'],task[1],
        a.out/'teachers'/task[0]['design']/f'w{task[0]["width"]}'/f's{task[1]}',teacher=True), teacher_tasks,a.jobs)
    make_dataset(a, contexts)


def make_dataset(a, contexts):
    from data import parse_teacher
    examples, missing = [], []
    for c in contexts:
        if c['split'] == 'final_test':
            continue
        records = []
        for seed in a.teacher_seeds:
            folder = a.out/'teachers'/c['design']/f'w{c["width"]}'/f's{seed}'
            metric = read(folder/'metrics.json')
            if metric['status'] in {'success','unroutable'}:
                records.append(metric)
        if not records:
            missing.append(dict(design=c['design'],width=c['width']))
            continue
        # One teacher per condition, never average unrelated/symmetric placements.
        selected = min(records,key=lambda r:(r['status']!='success',r['cpd'] or math.inf,r['wirelength'] or math.inf))
        target = parse_teacher(selected['teacher_prefix'], read(c['logic']), read(c['device']),split=c['split'])
        examples.append(dict(design=c['design'],family=c['family'],split=c['split'],width=c['width'],
            reference_width=c['reference_width'], graph=c['graph'], graph_sha256=sha(c['graph']),
            target=target, teacher_status=selected['status'], teacher_seed=selected['seed'],
            teacher_trial_sha256=fingerprint(selected), weight=2. if abs(c['width']/c['reference_width']-1)<=.1 else 1.))
    report = dict(schema='tailcompile-v68-dataset-1',examples=examples,
        tile_size=a.tile_size,
        source_protocol_sha256=sha(a.out/'protocol.json'), missing_conditions=missing,
        final_test_teacher_count=0, target_note='One actual native-GP teacher, no coordinate averaging.')
    write(a.out/'dataset.json',report)
    emit('prepare_complete',examples=len(examples),families=len({e['family'] for e in examples}),
         final_test_teacher_count=0,missing_conditions=len(missing))
    if missing:
        raise RuntimeError('Incomplete teacher conditions; timeouts remain censored, rerun with explicit retry after diagnosis')


def load_contexts(a):
    if read(a.out/'protocol.json')['identity'] != a.identity:
        raise ValueError('VPR executable/architecture/hook changed; start a separate cohort')
    return read(a.out/'contexts.json')['contexts']


def feedback_quality(baseline, candidate, *, timing_guard, wire_guard, reference=None):
    """Frozen deployment-label preference, with no failed-run QoR imputation."""
    if any(r['status'] not in {'success','unroutable'} for r in baseline+candidate):
        return dict(acceptable=False,reason='censored_or_infrastructure_failure')
    gains=sum(r['status']=='success' for r in candidate)-sum(r['status']=='success' for r in baseline)
    if gains<0:
        return dict(acceptable=False,reason='success_loss',net_success_gain=gains)
    joint=[];used_higher_reference=False
    for b,c in zip(baseline,candidate):
        if c['status']!='success':
            continue
        if b['status']=='success':
            joint.append((b,c))
        elif reference is not None:
            joint.append((reference,c));used_higher_reference=True
        else:
            return dict(acceptable=False,reason='rescued_case_has_no_qor_reference',net_success_gain=gains)
    if not joint:
        return dict(acceptable=False,reason='no_comparable_success_qor',net_success_gain=gains)
    comparison='includes_nearest_higher_width_training_teacher' if used_higher_reference else 'same_width_paired_seed'
    ratios={key:[c[key]/b[key] for b,c in joint if b.get(key) and c.get(key)]
            for key in ('cpd','wirelength')}
    if any(len(v)!=len(joint) for v in ratios.values()):
        return dict(acceptable=False,reason='missing_success_qor',net_success_gain=gains)
    if max(ratios['cpd'])>1+timing_guard or max(ratios['wirelength'])>1+wire_guard:
        return dict(acceptable=False,reason='qor_guard',net_success_gain=gains)
    cpd=math.exp(statistics.mean(map(math.log,ratios['cpd'])))
    wire=math.exp(statistics.mean(map(math.log,ratios['wirelength'])))
    quality=math.log(cpd)+.2*math.log(wire)
    acceptable=gains>0 or quality < -1e-4
    return dict(acceptable=acceptable,reason='accepted' if acceptable else 'no_gain',
        net_success_gain=gains,cpd_ratio=cpd,wirelength_ratio=wire,
        utility=4*gains-quality,qor_comparison=comparison)


def feedback(a):
    from generation import generate_contexts
    contexts=[c for c in load_contexts(a) if c['split']=='train']
    checkpoint=a.checkpoint or a.out/'models/pretrain/generator.pt'
    root=a.out/'feedback'
    freeze(root/'protocol.json',dict(checkpoint_sha256=sha(checkpoint),candidates=a.candidates,
        seeds=a.feedback_seeds,score_noise_std=.25,coordinate_noise_std=.05,
        context_sha256=sha(a.out/'contexts.json'),timing_guard=a.timing_guard,wire_guard=a.wire_guard))
    # Each candidate is an independent CPU process-pool task.  This also
    # parallelizes the costly decoders within one design/width.  Each worker
    # retains its own model/most-recent graph embeddings when applicable.
    # Existing candidate signatures and seed/noise policy remain unchanged.
    generated=generate_contexts(contexts,checkpoint,root/'generation',
        jobs=getattr(a,'generation_jobs',a.jobs),candidates=a.candidates,
        mode='feedback',device=a.device,progress_path=root/'generation_progress.json')
    tasks=[(c,s,k) for c in contexts for s in a.feedback_seeds for k in range(-1,a.candidates)]
    def one(task):
        c,seed,candidate=task
        generation=generated.get((c['design'],c['width'],candidate))
        return route(a,c,c['width'],seed,root/'runs'/c['design']/f'w{c["width"]}'/
            ('default' if candidate<0 else f'c{candidate}')/f's{seed}',
            init=generation['ap_init'] if generation else None,
            extra=dict(feedback_checkpoint_sha256=sha(checkpoint)))
    results=parallel(one,tasks,a.jobs)
    selected,diagnostics=[],[]
    for c in contexts:
        folder=root/'runs'/c['design']/f'w{c["width"]}'
        baseline=[read(folder/'default'/f's{s}'/'metrics.json') for s in a.feedback_seeds]
        reference=None
        for other in sorted((r for r in contexts if r['design']==c['design'] and r['width']>=c['width']),key=lambda r:r['width']):
            possible=[a.out/'teachers'/other['design']/f'w{other["width"]}'/f's{s}'/'metrics.json' for s in a.teacher_seeds]
            successes=[read(p) for p in possible if p.exists() and read(p)['status']=='success']
            if successes:
                reference=min(successes,key=lambda r:r['cpd'] or math.inf);break
        choices=[]
        for candidate in range(a.candidates):
            rows=[read(folder/f'c{candidate}'/f's{s}'/'metrics.json') for s in a.feedback_seeds]
            quality=feedback_quality(baseline,rows,timing_guard=a.timing_guard,wire_guard=a.wire_guard,reference=reference)
            diagnostics.append(dict(design=c['design'],width=c['width'],candidate=candidate,**quality))
            if quality['acceptable']:
                choices.append((quality['utility'],-candidate,candidate,quality))
        if choices:
            _,_,candidate,quality=max(choices)
            selected.append(dict(design=c['design'],family=c['family'],split='train',width=c['width'],
                target=generated[c['design'],c['width'],candidate]['target'],
                target_sha256=sha(generated[c['design'],c['width'],candidate]['target']),
                candidate=candidate,quality=quality))
    write(root/'feedback_dataset.json',dict(schema='tailcompile-v68-real-feedback-1',complete=True,
        generator_checkpoint_sha256=sha(checkpoint),selected=selected,diagnostics=diagnostics,
        final_test_examples=0,candidate_trials=len(tasks),
        all_route_outcomes_known=all(r['status'] in ('success','unroutable') for r in results),
        limitations=['Candidate best is a train-only hindsight teacher, not deployable test selection.',
                    'No failed-run CPD imputation; rescues may use a higher-width training reference.']))
    emit('feedback_complete',candidate_trials=len(tasks),improved_initializer_targets=len(selected))
    if any(r['status'] not in ('success','unroutable') for r in results):
        raise RuntimeError('Feedback batch persisted, but censored/infrastructure outcomes remain; diagnose and explicitly retry before fine-tuning')


def evaluation_plan(a, contexts, checkpoint):
    saved=read(a.out/'protocol.json')
    freeze(a.out/'evaluation_plan.json',dict(schema='tailcompile-v68-evaluation-1',
        checkpoint_sha256=sha(checkpoint),checkpoint=str(checkpoint.resolve()),
        contexts_sha256=sha(a.out/'contexts.json'),seeds=saved['evaluation_seeds'],
        designs=[dict(design=c['design'],family=c['family'],width=c['width']) for c in contexts],
        methods=['default','gnn_warmstart'],
        no_test_teacher_or_candidate_selection=True,
        runtime_scope='scan wall time measures throughput; use separate serial timing phase for speedup'))
    return read(a.out/'evaluation_plan.json')


def evaluate(a):
    from infer import Generator
    from generation import generate_contexts
    contexts=[c for c in load_contexts(a) if c['split']=='final_test']
    if not contexts:
        raise ValueError('No frozen final-test contexts; a development-only subset is not final validation')
    checkpoint=a.checkpoint or a.out/'models/finetune/generator.pt'
    if not checkpoint.is_file():
        raise FileNotFoundError(f'{checkpoint}; train then fine-tune before final evaluation')
    plan=evaluation_plan(a,contexts,checkpoint)
    generator=Generator(checkpoint,a.device)
    forbidden=set(generator.saved['training_families'])|set(generator.saved['validation_families'])
    if forbidden & {c['family'] for c in contexts}:
        raise ValueError('Final-test family participated in generator development')
    # Drop the metadata-only generator before starting CPU worker processes.
    del generator
    all_generated=generate_contexts(contexts,checkpoint,a.out/'evaluation/generation',
        jobs=getattr(a,'generation_jobs',a.jobs),candidates=1,mode='evaluation',
        device=a.device,progress_path=a.out/'evaluation/generation_progress.json')
    generated={(design,width):value for (design,width,candidate),value in all_generated.items()}
    tasks=[]
    for c in contexts:
        for seed in plan['seeds']:
            for method in (['default','gnn_warmstart'] if seed%2 else ['gnn_warmstart','default']):
                tasks.append((c,seed,method))
    def one(task):
        c,seed,method=task
        return route(a,c,c['width'],seed,a.out/'evaluation/runs'/c['design']/f'w{c["width"]}'/f's{seed}'/method,
            init=generated[c['design'],c['width']]['ap_init'] if method=='gnn_warmstart' else None,
            extra=dict(evaluation_checkpoint_sha256=sha(checkpoint)))
    parallel(one,tasks,a.jobs)
    report(a)


def width_observation(grid, rows):
    observed={r['width']:r['status'] for r in rows}
    successful=[w for w in grid if observed.get(w)=='success']
    unknown=[w for w in grid if observed.get(w) not in ('success','unroutable')]
    minimum=min(successful) if successful else None
    lower_failures=[w for w in grid if observed.get(w)=='unroutable' and (minimum is None or w<minimum)]
    return dict(minimum_success_in_grid=minimum,
        greatest_known_failure_below= max(lower_failures) if lower_failures else None,
        left_censored=bool(minimum==min(grid)),right_censored=minimum is None,
        unknown_widths=unknown,
        nonmonotone=bool(minimum is not None and any(observed.get(w)=='unroutable' for w in grid if w>minimum)),
        exact_wmin_proven=False)


def report(a):
    if not (a.out/'evaluation_plan.json').exists():
        raise ValueError('No frozen evaluation plan')
    plan=read(a.out/'evaluation_plan.json')
    comparisons=[];all_rows=[];missing=[];width_rows=[]
    for design in sorted({r['design'] for r in plan['designs']}):
        grid=sorted({r['width'] for r in plan['designs'] if r['design']==design})
        for seed in plan['seeds']:
            by_method={}
            for method in plan['methods']:
                rows=[]
                for width in grid:
                    file=a.out/'evaluation/runs'/design/f'w{width}'/f's{seed}'/method/'metrics.json'
                    if file.exists():rows.append(read(file))
                    else:missing.append(str(file.relative_to(a.out)))
                by_method[method]=width_observation(grid,rows)
                all_rows.extend(dict(r,method=method) for r in rows)
                width_rows.append(dict(design=design,seed=seed,method=method,**by_method[method]))
            left,right=by_method['default'],by_method['gnn_warmstart']
            # Never count a timeout as evidence that a larger width was needed.
            eligible=(not left['unknown_widths'] and not right['unknown_widths'] and
                left['minimum_success_in_grid'] is not None and right['minimum_success_in_grid'] is not None)
            comparisons.append(dict(design=design,seed=seed,comparable_observed_grid=eligible,
                delta_width=(right['minimum_success_in_grid']-left['minimum_success_in_grid']) if eligible else None,
                censored=left['left_censored'] or right['left_censored']))
    per_family=[]
    for design in sorted({r['design'] for r in comparisons}):
        pairs=[r for r in comparisons if r['design']==design and r['comparable_observed_grid']]
        per_family.append(dict(design=design,paired_seeds=len(pairs),
            improved=sum(r['delta_width']<0 for r in pairs),same=sum(r['delta_width']==0 for r in pairs),
            degraded=sum(r['delta_width']>0 for r in pairs),
            mean_delta_width=statistics.mean(r['delta_width'] for r in pairs) if pairs else None))
    protocol_record=read(a.out/'protocol.json')
    joint=[]
    index={(r['design'],r['width'],r['seed'],r['method']):r for r in all_rows}
    for key,b in index.items():
        design,width,seed,method=key
        g=index.get((design,width,seed,'gnn_warmstart'))
        if method=='default' and b['status']=='success' and g and g['status']=='success':
            joint.append(dict(design=design,width=width,seed=seed,
                cpd_ratio=g['cpd']/b['cpd'] if b.get('cpd') and g.get('cpd') else None,
                wirelength_ratio=g['wirelength']/b['wirelength'] if b.get('wirelength') and g.get('wirelength') else None,
                throughput_wall_ratio=g['wall_seconds']/b['wall_seconds']))
    result=dict(schema='tailcompile-v68-evaluation-report-1',complete=not missing,
        all_route_outcomes_known=not missing and all(r['status'] in ('success','unroutable') for r in all_rows),
        expected_trials=len(plan['designs'])*len(plan['seeds'])*2,recorded_trials=len(all_rows),
        missing=missing,width_results=width_rows,paired_width_deltas=comparisons,by_design=per_family,
        joint_success_qor=joint,
        timing_guard=protocol_record['timing_guard'],wire_guard=protocol_record['wire_guard'],
        timeout_count=sum(r['status']=='timeout' for r in all_rows),
        infrastructure_failure_count=sum(r['status'] not in ('success','unroutable','timeout') for r in all_rows),
        caveats=['Width minimum is the smallest observed success on the frozen grid, not an exhaustive mathematical minimum.',
                 'Multiple seeds/widths are not independent design families.',
                 'Parallel scan wall times cannot establish per-task runtime speedup; use timing_report.json.',
                 'Inspect CPD and wirelength guards separately; no gain is claimed automatically.'])
    write(a.out/'evaluation_report.json',result)
    emit('evaluation_report',complete=result['complete'],all_route_outcomes_known=result['all_route_outcomes_known'],
         trials=len(all_rows),by_design=per_family)


def timing(a):
    """A small serial, alternating AB timing pass on joint-success conditions.

    Heavy preparation/width scans/feedback remain parallel. These measurements
    have their own directories and never mix with throughput-run wall times.
    """
    contexts=load_contexts(a);plan=read(a.out/'evaluation_plan.json')
    timing_root=a.out/'timing'/a.timing_name
    quota_path=Path('/sys/fs/cgroup/cpu.max')
    hardware=dict(host=socket.gethostname(),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else None,
        cpu_quota=quota_path.read_text().strip() if quota_path.exists() else None)
    report_record=read(a.out/'evaluation_report.json')
    if not report_record['all_route_outcomes_known']:
        raise ValueError('Finish/classify scan outcomes before selecting timing conditions')
    selected=[]
    for design in sorted({r['design'] for r in plan['designs']}):
        seeds=plan['seeds'][:3]
        for width in sorted({r['width'] for r in plan['designs'] if r['design']==design}):
            if all(read(a.out/'evaluation/runs'/design/f'w{width}'/f's{s}'/method/'metrics.json')['status']=='success'
                   for s in seeds for method in plan['methods']):
                selected.append(dict(design=design,width=width,seeds=seeds));break
    timing_protocol=dict(conditions=selected,concurrency=1,vpr_threads=1,hardware=hardware,
        checkpoint_sha256=plan['checkpoint_sha256'],selection='Lowest frozen grid width successful for both methods/all timing seeds')
    freeze(timing_root/'timing_plan.json',timing_protocol)
    write(a.out/'timing_plan.json',dict(timing_protocol,run_name=a.timing_name))
    rows=[]
    for condition in selected:
        c=next(c for c in contexts if c['design']==condition['design'] and c['width']==condition['width'])
        generation=read(a.out/'evaluation/generation'/c['design']/f'w{c["width"]}'/'generation.json')
        for seed in condition['seeds']:
            pair={}
            for method in (plan['methods'] if seed%2 else list(reversed(plan['methods']))):
                pair[method]=route(a,c,c['width'],seed,timing_root/c['design']/f'w{c["width"]}'/f's{seed}'/method,
                    init=generation['ap_init'] if method=='gnn_warmstart' else None,
                    extra=dict(timing_checkpoint_sha256=plan['checkpoint_sha256']))
            if not all(r['status']=='success' for r in pair.values()):
                rows.append(dict(design=c['design'],width=c['width'],seed=seed,comparable=False));continue
            baseline=pair['default'];gnn=pair['gnn_warmstart']
            structure=read(a.out/'inputs'/c['design']/'structure_run/trial.json')['wall_seconds']
            audit=read(c['audit'])
            logical_cost=sum(v for key,v in audit.get('timing',{}).items()
                             if key not in ('device_preprocessing','device_provenance'))
            scopes=dict(vpr_only=gnn['wall_seconds'],
                cached_graph_warm=gnn['wall_seconds']+generation['warm_inference_lowering_seconds'],
                new_design_prepared_device=gnn['wall_seconds']+generation['cold_from_prepared_graph_seconds']+structure+logical_cost)
            rows.append(dict(design=c['design'],width=c['width'],seed=seed,comparable=True,
                speedup={k:baseline['wall_seconds']/v for k,v in scopes.items()},method_seconds=scopes,
                initializer_components=generation['timing'],structure_extraction_seconds=structure,
                logic_feature_seconds=logical_cost,cpd_ratio=gnn['cpd']/baseline['cpd'],
                wirelength_ratio=gnn['wirelength']/baseline['wirelength']))
    timing_result=dict(complete=True,run_name=a.timing_name,hardware=hardware,concurrency=1,vpr_threads=1,rows=rows,
        reusable_rrg_preprocessing_excluded=True,
        python_process_startup_seconds=None,
        scope_note='new_design_prepared_device is a component-based estimate in an already running Python process, including AP structure extraction, logic preparation, generator and VPR. Logic/structure costs were measured during preparation, not remeasured as one isolated process. Reusable device RRG preparation separate.',
        excluded_common_costs=['RTL synthesis is common and excluded from both arms.'],
        warning='Cached graph warm speedup is not a first-use/new-design speedup.')
    write(timing_root/'timing_report.json',timing_result)
    write(a.out/'timing_report.json',timing_result)
    emit('timing_complete',paired_conditions=len(rows))


def status(a):
    files={name:(a.out/name).exists() for name in ('inventory.json','contexts.json','dataset.json',
        'models/pretrain/training_report.json','feedback/feedback_dataset.json',
        'models/finetune/training_report.json','evaluation_report.json','timing_report.json')}
    records=list((a.out/'evaluation/runs').rglob('metrics.json'))
    from collections import Counter
    result=dict(files=files,evaluation_status_counts=dict(Counter(read(p)['status'] for p in records)),
        interrupted_tasks=sum(not (p.parent/'trial.json').exists() for p in a.out.rglob('started.json')
                              if '_interrupted' not in p.parts and '_attempts' not in p.parts),
        evaluation_complete=read(a.out/'evaluation_report.json').get('all_route_outcomes_known',False)
                            if files['evaluation_report.json'] else False)
    result['generation_progress']={}
    for phase in ('feedback','evaluation'):
        path=a.out/phase/'generation_progress.json'
        if path.exists():
            progress=read(path)
            result['generation_progress'][phase]={key:progress.get(key) for key in (
                'complete','total_candidates','completed_candidates','reused_candidates',
                'pending_candidates','failed_contexts','failed_candidates','workers','elapsed_seconds')}
            result['generation_progress'][phase]['active_candidates']=len(progress.get('running_contexts',[]))
    emit('status',**result)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('phase',choices=['inventory','prepare','feedback','evaluate','report','timing','status','recover'])
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--vtr-root',dest='vtr',type=Path,default=os.environ.get('VTR_ROOT'))
    p.add_argument('--cohort',type=Path,default=HERE/'cohort.json')
    p.add_argument('--designs',nargs='+')
    p.add_argument('--tile-size',type=int,default=4)
    p.add_argument('--jobs',type=int,default=DEFAULT_JOBS,help='Concurrent VPR tasks (default: 16)')
    p.add_argument('--prepare-jobs',type=int,default=None,help='Synthesis/structure and graph processes; defaults to --jobs')
    p.add_argument('--rr-jobs',type=int,default=None,help='RRG build processes; defaults to --jobs, not restricted to 1')
    p.add_argument('--generation-jobs',type=int,default=None,
                   help='CPU initializer generation processes; defaults to --jobs')
    p.add_argument('--timeout',type=float,default=7200)
    p.add_argument('--heartbeat',type=float,default=60)
    p.add_argument('--width-start',type=int,default=100)
    p.add_argument('--width-max',type=int,default=640)
    p.add_argument('--width-step',type=int,default=4)
    p.add_argument('--half-band',type=int,default=3)
    p.add_argument('--calibration-seeds',type=int,nargs='+',default=[301,302])
    p.add_argument('--teacher-seeds',type=int,nargs='+',default=[311,312])
    p.add_argument('--feedback-seeds',type=int,nargs='+',default=[321,322])
    p.add_argument('--seeds',type=int,nargs='+',default=[341,342,343,344,345])
    p.add_argument('--timing-guard',type=float,default=.03)
    p.add_argument('--wire-guard',type=float,default=.03)
    p.add_argument('--checkpoint',type=Path)
    p.add_argument('--candidates',type=int,default=8)
    p.add_argument('--timing-name',default='serial_001')
    p.add_argument('--device',default='cpu')
    p.add_argument('--retry-failed',action='store_true')
    p.add_argument('--retry-timeouts',action='store_true')
    p.add_argument('--confirm-remote-stopped',action='store_true',help='Recover only after explicitly verifying old remote-host workers have stopped')
    a=p.parse_args(argv)
    try:
        resolve_concurrency(a)
    except ValueError as error:
        p.error(str(error))
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.timing_name):
        p.error('timing-name must be a simple run identifier')
    if (a.width_step < 2 or a.width_step%2 or a.width_start%a.width_step or
        a.width_max%a.width_step or a.width_max<a.width_start or a.half_band<1 or a.tile_size<1 or
        min(a.jobs,a.prepare_jobs,a.rr_jobs,a.generation_jobs,a.candidates)<1 or a.timeout<=0 or a.heartbeat<=0):
        p.error('Invalid width grid, positive task budget or tile size')
    return a


def main():
    a=parse_args()
    a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=True)
    emit('parallelism', vpr_jobs=a.jobs, prepare_jobs=a.prepare_jobs, rr_jobs=a.rr_jobs,
         graph_jobs=a.prepare_jobs, generation_jobs=a.generation_jobs, vpr_workers=1,
         note='These are sequential pipeline stages, not summed simultaneous budgets')
    if a.phase=='recover':
        recover(a.out,confirm_remote_stopped=a.confirm_remote_stopped);return
    if a.phase=='status':
        status(a);return
    if a.phase=='report':
        report(a);return
    identity(a)
    if a.phase=='inventory':
        from data import inventory
        result=inventory(read(a.cohort),a.vtr);write(a.out/'inventory.json',result)
        emit('inventory_complete',report=str(a.out/'inventory.json'));return
    record_policy(a.out)
    {'prepare':prepare,'feedback':feedback,'evaluate':evaluate,'timing':timing}[a.phase](a)


if __name__=='__main__':
    main()
