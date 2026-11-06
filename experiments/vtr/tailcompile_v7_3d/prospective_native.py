"""014 locked joint-edit predictions and paired native 3D VPR measurements."""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import time

from campaign import identity
from io_utils import freeze, read, write, sha, run_trial, vpr_command, prune_derived
from residual_campaign import classify_native_failure, scientific_result
from layer_observations import collect_layers
from qor_route_metrics import summarize_route
from residual_observations import collect_residual_observations

HERE = Path(__file__).resolve().parent
METHODS = ('Default', 'D', 'j_s1', 'j_s2')
SEEDS = (14101, 14102, 14103)


def _inside(path, root):
    path = Path(path).resolve()
    if Path(root).resolve() not in path.parents:
        raise ValueError(f'prospective artifact outside its owning directory: {path}')
    return path


def _cohort(out):
    out = Path(out).resolve()
    if out.parent != HERE/'runs' or not out.name.startswith('development_014'):
        raise ValueError('014 native execution requires its own cohort')
    return out


def _pinned(path, expected):
    if sha(path) != expected:
        raise ValueError(f'prospective input signature differs: {path}')
    return str(Path(path).resolve())


def model_sources(source012, source013):
    """Bind each epoch-80 checkpoint to its completed archived full-fit row."""
    sources = [Path(p).resolve() for p in (source012, source013)]
    for path, label in zip(sources, ('012','013')):
        if path.parent != HERE/'runs' or not path.name.startswith('development_'+label):
            raise ValueError('014 requires canonical source cohorts')
    protocol = read(sources[0]/'edit_context_protocol.json')
    base = Path(protocol['base_checkpoint']).resolve()
    if HERE/'runs' not in base.parents or not base.relative_to(HERE/'runs').parts[0].startswith('development_010'):
        raise ValueError('014 D must come from the frozen010 cohort')
    models = dict(D=dict(path=_pinned(base, protocol['base_checkpoint_sha256']),
                         sha256=protocol['base_checkpoint_sha256']))
    for name, source, variant in (('E012',sources[0],'E'),('G012',sources[0],'G'),
                                  ('E013',sources[1],'E_h8_train_std_l2p01')):
        index = read(source/'trained_report.json')
        key = 'config_id' if name == 'E013' else 'mode'
        rows = [e for e in index['fits'] if e['split_id']=='full_fit' and e[key]==variant and e['seed']==12301]
        if not index['complete'] or len(rows)!=1:
            raise ValueError('014 fixed scoring checkpoint absent or ambiguous: '+name)
        entry, = rows
        path = _inside(entry['path'], source)
        fit = read(_pinned(path, entry['sha256']))
        training = fit['training']
        actual = training['factor_config']['id'] if name=='E013' else training['mode']
        if (actual != variant or fit['seed'] != 12301 or fit['split']['id'] != 'full_fit'
                or training['epochs'] != 80 or training['source_checkpoint_sha256'] != models['D']['sha256']):
            raise ValueError('014 scoring checkpoint provenance differs: '+name)
        checkpoint = _inside(training['checkpoint'],path.parent)
        expected = training['checkpoint_sha256'][training['checkpoint']]
        if checkpoint.name != 'epoch_0080.pt':
            raise ValueError('014 requires the frozen epoch 80')
        models[name] = dict(path=_pinned(checkpoint, expected), sha256=expected,
                            fit_sha256=entry['sha256'], backbone_sha256=training['backbone_sha256'],
                            cache_sha256=training['cache_sha256'], train_ids=training['train_ids'])
    for key in ('backbone_sha256','cache_sha256','train_ids'):
        if any(models[name][key] != models['E012'][key] for name in ('G012','E013')):
            raise ValueError('014 scorer source folds, features or backbone differ')
    return models


def _device(campaign, condition):
    record, arch_id, width = condition['record'], condition['architecture'], condition['width']
    campaign.guard()
    campaign.check_binary()
    arch = campaign.architecture(record, arch_id)
    rr = campaign.rr(record, arch_id, width)
    info = read(rr['manifest'])
    if info['binary_sha256'] != campaign.binary_hash or info['channel_width'] != width:
        raise ValueError('014 device does not match backend and width')
    _pinned(arch, info['architecture_sha256'])
    fields = dict(arch=str(arch), blif=record['blif'], device=rr['device'], rr_graph=rr['rr_graph'],
                  device_manifest=rr['manifest'])
    signature = identity(dict(binary=campaign.binary_hash, architecture=info['architecture_sha256'],
                              width=width, rr_graph=info['rr_graph_sha256'], graph=info['graph_sha256'],
                              blif=sha(record['blif'])))
    return fields, signature


def _check_output(candidate, owner):
    path = _inside(candidate['ap_init'], owner)
    _pinned(path, candidate['ap_init_sha256'])
    if candidate.get('residual_edits'):
        _pinned(_inside(candidate['residual_edits'],owner),candidate['residual_edits_sha256'])
    if not candidate.get('deployed_action_sha256'):
        raise ValueError('missing action identity')


def _score_and_lower(graph, logical, device, models, output):
    """All heads score exactly the unique original bank, including identity ties."""
    import torch
    from train import state_sha256
    from train_residual import from_source
    from edit_context_model import feature_blocks
    from train_edit_context import load_model as load012
    from train_generalization import load_model as load013
    from residual_actions import make_candidates
    from aligned_model import action_digest
    from generation import lower_initialization

    selector = from_source(models['D']['path'], expected_sha256=models['D']['sha256'])
    base, bank = selector.candidate_bank(graph, 'J')
    backbone_sha = state_sha256(selector.backbone)
    ids = ['D' if a['candidate_id']=='identity' else a['candidate_id'] for a in bank]
    if ids[0] != 'D' or set(ids)-set(METHODS[1:]):
        raise ValueError('014 unexpected joint candidate bank')
    glob, local = feature_blocks(selector.backbone, graph, base, bank)
    candidates = {}
    for method, action in zip(ids, bank):
        action_sha = action_digest(action)
        lowered = lower_initialization(logical,device,action,output/method,method=method)
        changed = (base['layer'] != action['layer']) | (base['xy'] != action['xy']).any(1)
        edits = [dict(name=atom,base_layer=int(base['layer'][i]),target_layer=int(action['layer'][i]),
                      base_xy=base['xy'][i].tolist(),target_xy=action['xy'][i].tolist())
                 for i in changed.nonzero().flatten().tolist() for atom in logical['clusters'][i]['atoms']]
        edit_path = output/method/'residual_edits.json'
        write(edit_path,dict(grid=[device['grid_width'],device['grid_height']],atoms=edits))
        candidates[method] = dict(ap_init=lowered['ap_init'], ap_init_sha256=lowered['ap_init_sha256'],
            deployed_action_sha256=action_sha, canonical_method=method,
            residual_edits=str(edit_path),residual_edits_sha256=sha(edit_path),
            edit_counts={k: action['capacity_diagnostics'][k] for k in ('changed_nodes','changed_layers','changed_xy')})
    # The original generator removes duplicate actions before scoring. Restore
    # physical trial labels for those duplicates, without creating extra scores.
    if 'j_s1' not in candidates:
        candidates['j_s1'] = dict(candidates['D'])
    if 'j_s2' not in candidates:
        cfg = selector.config
        single = make_candidates(graph, base, 'J', strengths=(cfg['strengths'][1],),
                                 max_moves=cfg['max_moves'], fraction=cfg['fraction'], seed=cfg['seed'])
        strong_sha = action_digest(single[-1])
        canonical = next((m for m in ids if candidates[m]['deployed_action_sha256']==strong_sha), None)
        if canonical is None:
            raise ValueError('014 cannot certify generator-deduplicated strong edit')
        candidates['j_s2'] = dict(candidates[canonical])
    choices = {}
    for name in ('E012','G012','E013'):
        start = time.perf_counter()
        binding = models[name]
        model = (load013 if name=='E013' else load012)(binding['path'], device='cpu').eval()
        cfg, metadata = model.config, model.training_metadata
        if (cfg['mode'] != name[0] or cfg['source_checkpoint_sha256'] != models['D']['sha256']
                or cfg['backbone_sha256'] != backbone_sha or backbone_sha != binding['backbone_sha256']
                or cfg['cache_sha256'] != binding['cache_sha256'] or metadata['epoch'] != 80
                or metadata['train_ids'] != binding['train_ids']):
            raise ValueError('014 model checkpoint/graph source mismatch: '+name)
        if name=='E013' and metadata['factor_config']['id']!='E_h8_train_std_l2p01':
            raise ValueError('014 incorrect 013 factor configuration')
        scores = model.score_features(glob, local, 'J').detach().cpu()
        if scores.shape != (len(ids),) or not torch.isfinite(scores).all():
            raise ValueError('014 invalid joint scorer output')
        method = ids[int(scores.argmax())]
        candidate = candidates[method]
        choices[name] = dict(method=method, candidate_ids=ids, scores=scores.tolist(),
            ap_init_sha256=candidate['ap_init_sha256'], deployed_action_sha256=candidate['deployed_action_sha256'],
            scoring_seconds=time.perf_counter()-start, checkpoint_sha256=binding['sha256'])
    return candidates, choices


def prediction_worker(request, output):
    import torch
    from data import build_logic
    from fast_graph import make_graph, semantic_digest
    from layer_features import augment_graph
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cohort = _cohort(request['out'])
    _inside(output, _inside(cohort/'predictions',cohort))
    pins_path = Path(request['prefix']+'.pins.tsv').resolve()
    if pins_path.is_file() != (str(pins_path) in request['inputs']):
        raise ValueError('014 optional AP pin-direction input changed')
    for path, expected in request['inputs'].items():
        _pinned(path, expected)
    started = time.perf_counter()
    logical = build_logic(request['prefix'],request['design'],request['blif'],request['arch'],
                          family=request['family'],split=request['split'])
    device = read(request['device'])
    graph = augment_graph(make_graph(logical,device),logical,device,read(request['vertical_summary']),
                          request['blif'],request['prefix'])
    graph_sha = semantic_digest(graph)
    with torch.no_grad():
        candidates, choices = _score_and_lower(graph, logical, device, request['models'], output)
    for path, expected in request['inputs'].items():
        _pinned(path, expected)
    elapsed = time.perf_counter()-started
    value = dict(condition_id=request['condition_id'], device_signature=request['device_signature'],
                 graph_sha256=graph_sha, models=request['models'], candidates=candidates, choices=choices,
                 generation_seconds=elapsed-sum(c['scoring_seconds'] for c in choices.values()),
                 prediction_worker_internal_seconds=elapsed, generation_seconds_excludes_scorer_times=True)
    write(output/'prediction_report.json', value)
    print('TAILCOMPILE_V8_014_PREDICTION_COMPLETE', flush=True)


def predict_panel(campaign, conditions, models):
    from vertical_risk import prepare_device
    out = _cohort(campaign.out)
    target = _inside(out/'predictions',out)
    if len(conditions)!=8 or len({c['id'] for c in conditions})!=8:
        raise ValueError('014 predictions require eight qualified conditions')
    locked = target/'prediction_report.json'
    if locked.exists():
        panel = read(locked)
        if panel['models'] != models:
            raise ValueError('014 frozen model bindings changed')
        _validate_panel(campaign, conditions, panel)
        return panel
    if (out/'comparison').exists():
        raise ValueError('014 cannot generate predictions after primary dispatch')
    def predict(c):
        fields, device_signature = _device(campaign,c)
        folder = _inside(target/c['id'],target)
        summary = folder/'vertical_summary.json'
        folder.mkdir(parents=True,exist_ok=True)
        prepare_device(fields['rr_graph'],read(fields['device']),summary,
                       max_seconds=campaign.args.timeout)
        prefix = str(c['record']['structure_prefix'])
        files = [fields[k] for k in ('arch','device','device_manifest','blif')] + [summary,Path(__file__)]
        files.extend(prefix+s for s in ('.blocks.tsv','.nets.tsv','.meta.tsv'))
        pins_file = Path(prefix+'.pins.tsv')
        if pins_file.is_file():
            files.append(pins_file)
        pins = {str(Path(p).resolve()):sha(p) for p in files}
        for binding in models.values():
            pins[_pinned(binding['path'],binding['sha256'])] = binding['sha256']
        req = dict(fields, condition_id=c['id'], design=c['design'], family=c['family'], split=c['split'],
                   models=models, prefix=prefix, inputs=pins, device_signature=device_signature,
                   out=str(out), vertical_summary=str(summary))
        freeze(folder/'request.json',req)
        work = _inside(folder/'payload',folder)
        native = run_trial(folder/'worker',[sys.executable,HERE/'prospective_native.py','worker',
            folder/'request.json',work],inputs=[folder/'request.json',*pins],timeout=campaign.args.timeout,
            marker='TAILCOMPILE_V8_014_PREDICTION_COMPLETE',prune=False)
        if native['status']!='complete':
            raise RuntimeError('014 predictions incomplete: '+native['log'])
        entry = read(work/'prediction_report.json')
        if entry['condition_id']!=c['id'] or entry['device_signature']!=device_signature or entry['models']!=models:
            raise ValueError('014 worker evidence disagrees with its input')
        for candidate in entry['candidates'].values():
            _check_output(candidate, work)
        entry['prediction_worker_manifest']=str(folder/'worker/manifest.json')
        entry['prediction_request_sha256']=sha(folder/'request.json')
        entry['prediction_payload_sha256']=sha(work/'prediction_report.json')
        return entry
    rows = campaign.parallel(conditions,predict)
    result = dict(schema='v8-014-prediction-panel-1', models=models,
                  conditions={row['condition_id']:row for row in sorted(rows,key=lambda r:r['condition_id'])})
    _validate_panel(campaign, conditions, result)
    freeze(locked,result)
    return result


def run_native(campaign, condition, seed, target, method, generated=None, affinity=None):
    """Run one Default calibration or one native trial with consumer evidence."""
    out = _cohort(campaign.out)
    target = _inside(target,out)
    if method not in METHODS or (method=='Default') != (generated is None):
        raise ValueError('014 unexpected method/initializer pairing')
    dev, device_signature = _device(campaign,condition)
    env = dict(TC_GP_LAYER_PRIOR='0',TC_GP_EXPORT='post_gp')
    inputs = [campaign.binary_reference,dev['arch'],dev['blif'],dev['device_manifest']]
    command = vpr_command(campaign.binary,dev['arch'],dev['blif'],condition['width'],seed,
                          rr_graph=dev['rr_graph'])+['--device','pilot','--write_flat_place','post_route.fplace',
                          '--write_legalized_flat_place','post_legalizer.fplace','--flat_place_verbosity','0']
    execution_signature = identity(dict(device_signature=device_signature, command=list(map(str,command)),
        condition_id=condition['id'], native_env=env.copy(), timeout=campaign.args.timeout))
    if generated is not None:
        _check_output(generated, out/'predictions')
        env['TC_GP_INIT']=generated['ap_init']
        inputs.append(generated['ap_init'])
    native = run_trial(target/'vpr',command,inputs=inputs,env=env,timeout=campaign.args.timeout,
                       affinity=affinity,metadata=dict(method=method,execution_signature=execution_signature),prune=False)
    manifest = target/'vpr/manifest.json'
    cached = target/'metrics.json'
    work = Path(native['workdir'])
    native_sha = sha(manifest)
    if cached.is_file() and read(cached)['vpr_manifest_sha256']==native_sha:
        row = read(cached)
    else:
        log = Path(native['log']).read_text(errors='replace')
        status,confirmed = classify_native_failure(native['status'],native['returncode'],log)
        markers = [line for line in log.splitlines() if 'TAILCOMPILE_V7_GP_INIT_CONSUMED' in line]
        consumed = bool(markers)
        continued = 'TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' in log
        if (status in ('success','unroutable') or confirmed) and (consumed != (generated is not None) or
                generated and (not any(line.endswith(' path='+str(generated['ap_init'])) for line in markers)
                               or not continued or 'skipping the Global Placement' in log)):
            status='interface_failure'
        metrics = native['metrics']
        routes = list(work.glob('*.route'))
        if status=='success' and (len(routes)!=1 or any(
                not isinstance(metrics.get(m),(float,int)) or not math.isfinite(metrics[m]) or metrics[m]<=0
                for m in ('wirelength','cpd_ns'))):
            status='interface_failure'
        row = dict(condition_id=condition['id'],design=condition['design'],family=condition['family'],
            architecture=condition['architecture'],split=condition['split'],width_role=condition.get('width_role'),
            width=condition['width'],seed=seed,method=method,status=status,
            confirmed_candidate_infeasibility=confirmed, execution_signature=execution_signature,
            device_signature=device_signature, input_consumed=consumed, native_solver_continued=continued,
            vpr_manifest=str(manifest), vpr_manifest_sha256=native_sha,
            wirelength=metrics['wirelength'],cpd_ns=metrics['cpd_ns'], phases=metrics['phases'],
            vpr_seconds=metrics['wall_seconds'],unique_chanz_nodes=metrics['unique_chanz_nodes'],
            ap_init_sha256=generated['ap_init_sha256'] if generated else None,
            deployed_action_sha256=generated['deployed_action_sha256'] if generated else None)
        if scientific_result(row):
            row['final_route_usage'] = summarize_route(routes[0]) if status=='success' else None
            row['layer_observations'] = collect_layers(env.get('TC_GP_INIT'),work/'post_gp',
                final_fplace=work/'post_route.fplace',legalized_fplace=work/'post_legalizer.fplace')
            row['residual_observations'] = collect_residual_observations(generated,work,work)
        write(cached,row)
    if not scientific_result(row):
        raise RuntimeError(f'014 native trial needs diagnosis ({row["status"]}): '+native['log'])
    if not (work/'payload_pruned.json').is_file():
        prune_derived(work,protected=inputs)
    for path in [*work.glob('post_gp.*'),work/'post_route.fplace',work/'post_legalizer.fplace']:
        if path.is_file() and not path.is_symlink() and path.resolve().parent==work.resolve():
            path.unlink()
    return row


def _validate_panel(campaign, conditions, predictions):
    out = _cohort(campaign.out)
    catalog = predictions['conditions']
    if len(catalog)!=8 or len(conditions)!=8 or set(catalog)!={c['id'] for c in conditions}:
        raise ValueError('014 incomplete prediction coverage')
    for c in conditions:
        entry = catalog[c['id']]
        folder = _inside(out/'predictions'/c['id'],out/'predictions')
        _pinned(folder/'request.json',entry['prediction_request_sha256'])
        payload = read(_pinned(folder/'payload/prediction_report.json',entry['prediction_payload_sha256']))
        if entry['condition_id'] != c['id'] or entry['models'] != predictions['models'] or any(
                entry[k] != payload[k] for k in ('choices','candidates','graph_sha256','condition_id','models','device_signature')):
            raise ValueError('014 frozen predictions differ from original worker output')
        request = read(folder/'request.json')
        for pinned_path, expected in request['inputs'].items():
            _pinned(pinned_path,expected)
        if entry['device_signature']!=_device(campaign,c)[1]:
            raise ValueError('014 condition changed after predictions')
        by_method = entry['candidates']
        if set(by_method)!=set(METHODS[1:]) or set(entry['choices'])!={'E012','G012','E013'}:
            raise ValueError('014 incomplete predictions or candidate pool')
        for candidate in by_method.values():
            _check_output(candidate, out/'predictions')
        for choice in entry['choices'].values():
            scores, ids = choice['scores'], choice['candidate_ids']
            if (len(scores)!=len(ids) or not ids or ids[0]!='D' or len(ids)!=len(set(ids))
                    or not all(math.isfinite(v) for v in scores)
                    or choice['method']!=ids[max(range(len(ids)),key=scores.__getitem__)]):
                raise ValueError('014 frozen choice differs from scores')
            actual = by_method[choice['method']]
            if any(choice[k]!=actual[k] for k in ('ap_init_sha256','deployed_action_sha256')):
                raise ValueError('014 choice did not bind the measured action')


def run_panel(campaign, conditions, predictions):
    out = _cohort(campaign.out)
    path = out/'predictions/prediction_report.json'
    if read(path)!=predictions:
        raise ValueError('014 primary trials require the frozen full prediction panel')
    _validate_panel(campaign,conditions,predictions)
    catalog = predictions['conditions']
    comparison = _inside(out/'comparison',out)
    freeze(comparison/'protocol.json',dict(predictions_sha256=sha(path), methods=list(METHODS), seeds=list(SEEDS)))
    def measure(task):
        condition,seed = task
        bank = catalog[condition['id']]['candidates']
        results = []
        by_input = {}
        for method in METHODS:
            candidate = bank.get(method)
            init_sha = candidate['ap_init_sha256'] if candidate else None
            if init_sha in by_input:
                canonical = by_input[init_sha]
                row = dict(canonical, method=method, measured_method=canonical['method'],
                    deployed_action_sha256=candidate['deployed_action_sha256'], identical_input_reuse=True)
            else:
                folder = _inside(comparison/condition['id']/f's{seed}'/method,comparison)
                row = run_native(campaign,condition,seed,folder,method,candidate)
                if row['device_signature'] != catalog[condition['id']]['device_signature']:
                    raise ValueError('014 native device changed after prediction freeze')
                by_input[init_sha] = row
            results.append(row)
        return results
    jobs = [(c,s) for c in conditions for s in SEEDS]
    rows = [row for result in campaign.parallel(jobs,measure) for row in result]
    rows.sort(key=lambda r:(r['condition_id'],r['seed'],METHODS.index(r['method'])))
    freeze(comparison/'trials_report.json',dict(complete=True, predictions_sha256=sha(path), rows=rows,
        physical_runs=len({r['vpr_manifest'] for r in rows}), planned_rows=96))
    return rows


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('worker',))
    parser.add_argument('request', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    prediction_worker(read(args.request),args.output)
