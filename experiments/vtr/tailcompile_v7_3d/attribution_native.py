"""015 frozen-D candidate generation and matched native VPR observations."""
from pathlib import Path
import math
import sys
import time

from campaign import identity
from io_utils import read, write, freeze, sha, run_trial, vpr_command, prune_derived
from residual_campaign import classify_native_failure, scientific_result
from layer_observations import collect_layers
from residual_observations import collect_residual_observations
from qor_route_metrics import summarize_route

HERE = Path(__file__).resolve().parent
METHODS = ('Default', 'D', 'L', 'Y', 'J', 'R1', 'R2')
SEEDS = (15101, 15102, 15103)


def pinned(path, digest):
    path = Path(path).resolve()
    if sha(path) != digest:
        raise ValueError('015 frozen payload changed: '+str(path))
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
        raise ValueError('015 RRG/width mismatch')
    return identity(dict(binary=binary_hash, architecture=rr['architecture_sha256'],
        width=condition['width'], rr_graph=rr['rr_graph_sha256'], graph=rr['graph_sha256'],
        blif=condition.get('record',{}).get('mapped_blif_sha256') or sha(fields['blif'])))


def prediction_worker(request_path, output):
    """One isolated CPU graph/model process; no teacher or measured QoR input."""
    import torch
    from data import build_logic
    from fast_graph import make_graph, semantic_digest
    from layer_features import augment_graph
    from train_residual import from_source
    from structured_model import generate
    from attribution_actions import make_attribution_candidates
    from generation import lower_initialization
    from aligned_model import action_digest
    request = read(request_path)
    output = Path(output).resolve()
    cohort = Path(request['out']).resolve()
    if cohort.parent != HERE/'runs' or not cohort.name.startswith('development_015_') or cohort/'predictions' not in output.parents:
        raise ValueError('015 worker output outside its own cohort')
    for path, digest in request['inputs'].items():
        pinned(path, digest)
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    started = time.perf_counter()
    c = request['condition']; f = c['fields']; r = c['record']
    logical = build_logic(r['structure_prefix'], c['design'], f['blif'], f['arch'],
                          family=c['family'], split='validation')
    device = read(f['device'])
    graph = augment_graph(make_graph(logical, device), logical, device,
                          read(request['vertical_summary']), f['blif'], r['structure_prefix'])
    source = request['model']
    selector = from_source(source['path'], expected_sha256=source['sha256'])
    with torch.no_grad():
        base = generate(selector.backbone, graph, method='D', score_noise_std=0., diagnostics=False)
        actions = make_attribution_candidates(graph, base)
    candidates = {}
    for name, action in actions.items():
        lowered = lower_initialization(logical, device, action, output/name, method=name)
        changed = (base['layer'] != action['layer']) | (base['xy'] != action['xy']).any(1)
        edits = [dict(name=atom, base_layer=int(base['layer'][i]), target_layer=int(action['layer'][i]),
                      base_xy=base['xy'][i].tolist(), target_xy=action['xy'][i].tolist())
                 for i in changed.nonzero().flatten().tolist() for atom in logical['clusters'][i]['atoms']]
        edit_path = output/name/'residual_edits.json'
        write(edit_path, dict(grid=[device['grid_width'], device['grid_height']], atoms=edits))
        diag = dict(action.get('attribution_diagnostics', {}))
        matching = action.get('matching_diagnostics', diag.get('matching_diagnostics', {}))
        if name.startswith('R'):
            diag.update(matching)
            diag['matched_control'] = bool(matching.get('matched_control', diag.get('matched_control', False)))
        else:
            diag['matched_control'] = True
        candidates[name] = dict(ap_init=lowered['ap_init'], ap_init_sha256=lowered['ap_init_sha256'],
            deployed_action_sha256=action_digest(action), residual_edits=str(edit_path),
            residual_edits_sha256=sha(edit_path), diagnostics=diag,
            edit_counts=action.get('capacity_diagnostics', {}))
    for path, digest in request['inputs'].items():
        pinned(path, digest)
    write(output/'prediction_report.json', dict(condition_id=c['id'],
        device_signature=request['device_signature'], graph_sha256=semantic_digest(graph),
        model=source, candidates=candidates, generation_seconds=time.perf_counter()-started))
    print('TAILCOMPILE_V8_015_PREDICTION_COMPLETE', flush=True)


def validate_candidate(candidate, out):
    for key in ('ap_init', 'residual_edits'):
        path = pinned(candidate[key], candidate[key+'_sha256'])
        if Path(out)/'predictions' not in path.parents:
            raise ValueError('015 initialization escapes its owning prediction folder')


def run_native(controller, condition, seed, folder, method, candidate=None):
    if method not in METHODS or (method=='Default') != (candidate is None):
        raise ValueError('015 method/initializer mismatch')
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
        raise RuntimeError(f'015 native trial needs diagnosis ({row["status"]}): '+native['log'])
    if not (work/'payload_pruned.json').is_file():
        prune_derived(work, protected=inputs)
    for path in [*work.glob('post_gp.*'),work/'post_route.fplace',work/'post_legalizer.fplace']:
        if path.is_file() and not path.is_symlink() and path.resolve().parent==work.resolve():
            path.unlink()
    return row


if __name__ == '__main__':
    if len(sys.argv)!=4 or sys.argv[1]!='worker':
        raise SystemExit('usage: attribution_native.py worker REQUEST OUTPUT')
    prediction_worker(sys.argv[2], sys.argv[3])
