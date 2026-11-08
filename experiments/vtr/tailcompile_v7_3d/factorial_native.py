"""017 frozen-D factorial candidate generation and matched native observations."""
from pathlib import Path
from contextlib import contextmanager
import signal
import math
import sys
import time

from campaign import identity
from io_utils import read, write, freeze, sha, run_trial, vpr_command, prune_derived
from residual_campaign import classify_native_failure, scientific_result
from layer_observations import collect_layers
from residual_observations import collect_residual_observations
from qor_route_metrics import summarize_route
from factorial_timing import validate_timing_model

HERE = Path(__file__).resolve().parent
METHODS = ('Default', 'D', 'W', 'WS', 'T', 'TS')
SEEDS = (17101, 17102, 17103)
ARMS = {'W': ('wl', False), 'WS': ('wl', True),
        'T': ('joint', False), 'TS': ('joint', True)}


@contextmanager
def refinement_deadline(seconds):
    """Unix hard deadline complements cooperative core budgets and parent timeout.

    An overrun aborts the worker with evidence; it is never relabeled as D or
    as mathematical convergence. The platform launcher always runs on Linux.
    """
    if isinstance(seconds,bool) or not isinstance(seconds,(int,float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError('017 refinement hard deadline must be positive finite seconds')
    if not hasattr(signal,'setitimer'):
        raise RuntimeError('017 hard refinement deadline requires POSIX setitimer')
    def expired(signum,frame):
        raise TimeoutError('017 refinement exceeded its hard deadline; no fallback action was emitted')
    old=signal.signal(signal.SIGALRM,expired)
    previous=signal.setitimer(signal.ITIMER_REAL,float(seconds))
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        signal.signal(signal.SIGALRM,old)
        if previous[0]:
            signal.setitimer(signal.ITIMER_REAL,*previous)


def pinned(path, digest):
    path = Path(path).resolve()
    if sha(path) != digest:
        raise ValueError('017 frozen payload changed: '+str(path))
    return path


def device_signature(condition, binary_hash):
    fields = condition['fields']
    # Compact reports can be re-evaluated from a relocated Git checkout; the
    # immutable large-payload digests are already recorded in this manifest.
    from attribution_panel import remap_path
    rr = read(remap_path(fields['device_manifest']))
    # Older RRG producers are allowed only after panel qualification verifies
    # a successful consumer with this exact current backend and frozen RRG.
    if rr['channel_width'] != condition['width']:
        raise ValueError('017 RRG/width mismatch')
    return identity(dict(binary=binary_hash, architecture=rr['architecture_sha256'],
        width=condition['width'], rr_graph=rr['rr_graph_sha256'], graph=rr['graph_sha256'],
        blif=condition.get('record',{}).get('mapped_blif_sha256') or sha(fields['blif'])))


def refine_arms(graph,base,summary,timing_model,config,hard_timeout,condition_id):
    """Prepare once, run the complete factorial, and reject partial setup."""
    from factorial_refinement import refine, prepare_problem
    started=time.perf_counter()
    print('V8_017_COMMON_SETUP_START condition='+condition_id,flush=True)
    with refinement_deadline(hard_timeout):
        problem=prepare_problem(graph,base,vertical_summary=summary,timing_model=timing_model,config=config)
    setup_seconds=time.perf_counter()-started
    actions={'D':base};arm_seconds={}
    for name,(mode,swaps) in ARMS.items():
        arm_started=time.perf_counter()
        print('V8_017_REFINEMENT_START condition='+condition_id+' arm='+name,flush=True)
        with refinement_deadline(hard_timeout):
            actions[name]=refine(graph,base,vertical_summary=summary,timing_model=timing_model,
                objective_mode=mode,allow_swaps=swaps,config=config,problem=problem)
        arm_seconds[name]=time.perf_counter()-arm_started
        if not actions[name].get('refinement_diagnostics',{}).get('setup_complete'):
            raise RuntimeError('017 incomplete refinement setup; no P&R dispatch: '+name)
        print('V8_017_REFINEMENT_COMPLETE condition='+condition_id+' arm='+name,flush=True)
    normalizers=[actions[n]['refinement_diagnostics']['normalizers'] for n in ARMS]
    if any(n!=normalizers[0] for n in normalizers[1:]):
        raise RuntimeError('017 factorial arms did not share D normalizers')
    return actions,dict(common_setup=setup_seconds,arms=arm_seconds,
                        refinement=time.perf_counter()-started)


def prediction_worker(request_path, output):
    """One bounded CPU process; all five initializations freeze before P&R."""
    import torch
    from data import build_logic
    from fast_graph import make_graph, semantic_digest
    from layer_features import augment_graph
    from train_residual import from_source
    from structured_model import generate
    from generation import lower_initialization
    from aligned_model import action_digest
    request = read(request_path)
    output = Path(output).resolve()
    cohort = Path(request['out']).resolve()
    if cohort.parent != HERE/'runs' or not cohort.name.startswith('development_017_') or cohort/'predictions' not in output.parents:
        raise ValueError('017 worker output outside its own cohort')
    def verify_inputs():
        for path, digest in request['inputs'].items():
            pinned(path, digest)
        for name,digest in request['code'].items():
            if Path(name).name != name:
                raise ValueError('017 malformed code reference')
            pinned(HERE/name,digest)
    verify_inputs()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    c = request['condition']; f = c['fields']; r = c['record']
    logical = build_logic(r['structure_prefix'], c['design'], f['blif'], f['arch'],
                          family=c['family'], split='validation')
    device = read(f['device']); summary=read(request['vertical_summary'])
    timing_model = validate_timing_model(read(request['timing_model']))
    graph_started=time.perf_counter()
    graph = augment_graph(make_graph(logical, device), logical, device,
                          summary, f['blif'], r['structure_prefix'])
    graph_seconds=time.perf_counter()-graph_started
    print('V8_017_GRAPH_COMPLETE condition='+c['id']+' seconds='+str(graph_seconds),flush=True)
    source = request['model']
    model_started=time.perf_counter()
    selector = from_source(source['path'], expected_sha256=source['sha256'])
    with torch.no_grad():
        base = generate(selector.backbone, graph, method='D', score_noise_std=0., diagnostics=False)
        model_seconds=time.perf_counter()-model_started
        actions,refine_phases=refine_arms(graph,base,summary,timing_model,
            request['refinement_config'],request['refinement_hard_timeout'],c['id'])
    candidates = {}
    for name, action in actions.items():
        lowered = lower_initialization(logical, device, action, output/name, method=name)
        changed = (base['layer'] != action['layer']) | (base['xy'] != action['xy']).any(1)
        edits = [dict(name=atom, base_layer=int(base['layer'][i]), target_layer=int(action['layer'][i]),
                      base_xy=base['xy'][i].tolist(), target_xy=action['xy'][i].tolist())
                 for i in changed.nonzero().flatten().tolist() for atom in logical['clusters'][i]['atoms']]
        edit_path = output/name/'residual_edits.json'
        write(edit_path, dict(grid=[device['grid_width'], device['grid_height']], atoms=edits))
        diag = dict(action.get('refinement_diagnostics', {}))
        receipt={}
        if name in ARMS:
            receipt_path=output/name/'refinement_receipt.json'
            write(receipt_path,diag)
            receipt=dict(refinement_receipt=str(receipt_path),refinement_receipt_sha256=sha(receipt_path))
            if isinstance(diag.get('accepted_moves'),list):
                diag['accepted_move_count']=len(diag['accepted_moves'])
            diag={k:v for k,v in diag.items() if k not in ('history','moves','accepted_moves') or not isinstance(v,(list,dict))}
        candidates[name] = dict(ap_init=lowered['ap_init'], ap_init_sha256=lowered['ap_init_sha256'],
            deployed_action_sha256=action_digest(action), residual_edits=str(edit_path),
            residual_edits_sha256=sha(edit_path), diagnostics=diag,
            edit_counts=action.get('capacity_diagnostics', {}), **receipt)
    verify_inputs()
    write(output/'prediction_report.json', dict(condition_id=c['id'],
        device_signature=request['device_signature'], graph_sha256=semantic_digest(graph),
        model=source, candidates=candidates,vertical_cache_provenance=request.get('vertical_cache_provenance'),
        timing_model_sha256=sha(request['timing_model']),
        timing_provenance=timing_model.get('provenance'),
        generation_seconds=time.perf_counter()-started,
        phase_seconds=dict(graph=graph_seconds,model=model_seconds,**refine_phases)))
    print('TAILCOMPILE_V8_017_PREDICTION_COMPLETE', flush=True)


def validate_candidate(candidate, out):
    for key in ('ap_init', 'residual_edits') + (('refinement_receipt',) if 'refinement_receipt' in candidate else ()):
        path = pinned(candidate[key], candidate[key+'_sha256'])
        if Path(out)/'predictions' not in path.parents:
            raise ValueError('017 initialization escapes its owning prediction folder')


def run_native(controller, condition, seed, folder, method, candidate=None):
    if method not in METHODS or (method=='Default') != (candidate is None):
        raise ValueError('017 method/initializer mismatch')
    controller.guard()
    f = condition['fields']
    devsig = device_signature(condition, controller.binary_hash)
    env = dict(TC_GP_LAYER_PRIOR='0', TC_GP_EXPORT='post_gp')
    command = vpr_command(controller.binary, f['arch'], f['blif'], condition['width'], seed,
        rr_graph=f['rr_graph'])+['--device','pilot','--write_flat_place','post_route.fplace',
        '--write_legalized_flat_place','post_legalizer.fplace','--flat_place_verbosity','0']
    signature = identity(dict(device_signature=devsig, command=list(map(str,command)),
        condition_id=condition['id'], native_env=env.copy(), timeout=controller.args.timeout))
    inputs = [controller.binary_reference, f['arch'], f['blif'], f['device_manifest']]
    if candidate is not None:
        validate_candidate(candidate, controller.out)
        env['TC_GP_INIT'] = candidate['ap_init']
        inputs.append(candidate['ap_init'])
    native = run_trial(Path(folder)/'vpr', command, inputs=inputs, env=env,
        timeout=controller.args.timeout, metadata=dict(method=method, execution_signature=signature), prune=False)
    manifest = Path(folder)/'vpr/manifest.json'
    cache = Path(folder)/'metrics.json'; work = Path(native['workdir'])
    if cache.is_file() and read(cache)['vpr_manifest_sha256'] == sha(manifest):
        row = read(cache)
    else:
        log = Path(native['log']).read_text(errors='replace')
        status, confirmed = classify_native_failure(native['status'], native['returncode'], log)
        markers = [s for s in log.splitlines() if 'TAILCOMPILE_V7_GP_INIT_CONSUMED' in s]
        consumed = bool(markers)
        continued = 'TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' in log
        if (status in ('success','unroutable') or confirmed) and (consumed != (candidate is not None) or
            candidate and (not any(s.endswith(' path='+candidate['ap_init']) for s in markers) or
                           not continued or 'skipping the Global Placement' in log)):
            status = 'interface_failure'
        m = native['metrics']; routes = list(work.glob('*.route'))
        if status=='success' and (len(routes)!=1 or any(not isinstance(m.get(k),(int,float)) or
                not math.isfinite(m[k]) or m[k]<=0 for k in ('wirelength','cpd_ns'))):
            status = 'interface_failure'
        row = dict(condition_id=condition['id'], design=condition['design'], family=condition['family'],
            architecture=condition['architecture'], width=condition['width'], seed=seed, method=method,
            status=status, confirmed_candidate_infeasibility=confirmed, execution_signature=signature,
            device_signature=devsig, input_consumed=consumed, native_solver_continued=continued,
            vpr_manifest=str(manifest), vpr_manifest_sha256=sha(manifest),
            wirelength=m['wirelength'], cpd_ns=m['cpd_ns'], vpr_seconds=m['wall_seconds'],
            phases=m['phases'], unique_chanz_nodes=m['unique_chanz_nodes'],
            ap_init_sha256=candidate['ap_init_sha256'] if candidate else None,
            deployed_action_sha256=candidate['deployed_action_sha256'] if candidate else None)
        if scientific_result(row):
            row['final_route_usage'] = summarize_route(routes[0]) if status=='success' else None
            row['layer_observations'] = collect_layers(env.get('TC_GP_INIT'), work/'post_gp',
                final_fplace=work/'post_route.fplace', legalized_fplace=work/'post_legalizer.fplace')
            row['residual_observations'] = collect_residual_observations(candidate, work, work)
        write(cache, row)
    if not scientific_result(row):
        raise RuntimeError(f'017 native trial needs diagnosis ({row["status"]}): '+native['log'])
    if not (work/'payload_pruned.json').is_file():
        prune_derived(work, protected=inputs)
    for path in [*work.glob('post_gp.*'),work/'post_route.fplace',work/'post_legalizer.fplace']:
        if path.is_file() and not path.is_symlink() and path.resolve().parent==work.resolve():
            path.unlink()
    return row


if __name__ == '__main__':
    if len(sys.argv)!=4 or sys.argv[1]!='worker':
        raise SystemExit('usage: factorial_native.py worker REQUEST OUTPUT')
    prediction_worker(sys.argv[2], sys.argv[3])
