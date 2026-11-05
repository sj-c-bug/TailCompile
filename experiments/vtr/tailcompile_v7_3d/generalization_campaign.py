"""013 offline factorial study using the signed and frozen 012 feature cache."""
from __future__ import annotations

import argparse
from collections import defaultdict
import itertools
import json
import os
from pathlib import Path
import queue
import sys
import time

from campaign import campaign_lock, resource_budget
from edit_context_campaign import (
    CODE as SOURCE_CODE, aggregate, execution_options as source_execution_options,
    inspect_source, parallel_fit_map, splits, _stop_fit_processes,
)
from generalization_spec import CONFIGS, EPOCHS, LEARNING_RATE, PRIMARY_CONFIG, SNAPSHOTS, TRAIN_SEEDS, config_by_id, plan
from io_utils import freeze, install_signal_handlers, raise_if_cancelled, read, run_trial, sha, write
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
STAGES = ('plan', 'check', 'prepare', 'train', 'report')
CODE = tuple(dict.fromkeys(SOURCE_CODE + ('generalization_spec.py', 'generalization_model.py',
    'train_generalization.py', 'generalization_campaign.py', 'run_generalization_campaign.sh')))


def owned(path):
    path = Path(path).resolve()
    runs = HERE/'runs'
    if runs not in path.parents or not path.relative_to(runs).parts[0].startswith('development_013'):
        raise ValueError('013 output requires its own development_013 cohort inside v7 runs')
    validate_output_roots((path,))
    return path


def origin_path(path):
    path = Path(path).resolve()
    if path.parent != HERE/'runs' or not path.name.startswith('development_012'):
        raise ValueError('013 source must be a signed 012 cohort inside v7 runs')
    return path


def execution_options(device, jobs=None, fit_timeout=1800):
    default_jobs = 16 if device == 'cpu' else 1
    return source_execution_options(device, default_jobs if jobs is None else jobs, fit_timeout)


def implementation():
    return {name: sha(HERE/name) for name in CODE}


def inspect_origin(origin):
    """Read-only input qualification, including the original measured actions."""
    origin = origin_path(origin)
    required = [origin/'edit_context_protocol.json', origin/'prepared_report.json']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(ready=False, missing=missing)
    protocol, prepared = [read(p) for p in required]
    if not prepared.get('complete'):
        raise ValueError('013 source012 feature preparation is incomplete')
    source = inspect_source(protocol['source'])
    if not source['ready']:
        return source
    cache = Path(prepared['cache']).resolve()
    if HERE/'runs' not in cache.parents or not cache.relative_to(HERE/'runs').parts[0].startswith('development_012'):
        raise ValueError('013 feature cache must remain in the source012 namespace')
    if not cache.is_file():
        return dict(ready=False, missing=[str(cache)])
    if sha(cache) != prepared['cache_file_sha256']:
        raise ValueError('013 original012 cache file signature differs')
    pins = dict(source['source_inputs'])
    pins.update({str(p): sha(p) for p in required + [cache]})
    original_index = origin/'trained_report.json'
    if original_index.is_file():
        pins[str(original_index)] = sha(original_index)
        for entry in read(original_index)['fits']:
            path = Path(entry['path']).resolve()
            if origin not in path.parents or sha(path) != entry['sha256']:
                raise ValueError('013 original012 fit evidence changed')
            pins[str(path)] = entry['sha256']
    return dict(ready=True, missing=[], origin=str(origin), source_info=source,
                source_inputs=pins, cache=str(cache), cache_file_sha256=prepared['cache_file_sha256'],
                cache_sha256=prepared['cache_sha256'], base_checkpoint_sha256=source['checkpoint_sha256'])


def verify_request(request):
    cache = Path(request['cache']).resolve()
    if HERE/'runs' not in cache.parents or not cache.relative_to(HERE/'runs').parts[0].startswith('development_012'):
        raise ValueError('013 worker cache escapes the read-only012 namespace')
    if sha(cache) != request['cache_file_sha256']:
        raise ValueError('013 worker cache bytes changed')
    if request['implementation'] != implementation():
        raise ValueError('013 worker implementation signatures differ')
    if request['config'] != config_by_id(request['config']['id']) or request['seed'] not in TRAIN_SEEDS:
        raise ValueError('013 worker uses an unfrozen configuration or seed')
    execution_options(request['device'], 1, request['fit_timeout'])


def fit_worker(request, out):
    out = owned(out)
    verify_request(request)
    import torch
    from train_edit_context import evaluate_cache, load_cache
    from train_generalization import fit_cache, load_model
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    cache = load_cache(request['cache'])
    split = request['split']
    if cache['cache_sha256'] != request['cache_sha256'] or split not in splits(cache['banks']):
        raise ValueError('013 worker cache content or fold differs')
    out.mkdir(parents=True, exist_ok=True)
    training = fit_cache(cache, split['train_ids'], out, config=request['config'], epochs=EPOCHS,
                         snapshots=SNAPSHOTS, seed=request['seed'], learning_rate=LEARNING_RATE,
                         device=request['device'])
    for path, expected in training['checkpoint_sha256'].items():
        if sha(path) != expected:
            raise ValueError('013 trained checkpoint changed')
    model = load_model(training['checkpoint'], device=request['device'])
    if model.training_metadata['epoch'] != EPOCHS or set(model.training_metadata['train_ids']) != set(split['train_ids']) or \
            model.training_metadata['factor_config'] != request['config']:
        raise ValueError('013 evaluated checkpoint uses different epoch or fold')
    evaluation = evaluate_cache(model, cache, split['evaluation_ids'])
    seed_evaluation = {str(s): evaluate_cache(model, cache, split['evaluation_ids'], seed_subset={s})
                       for s in (11401, 11402)}
    curves = []
    if not split['heldout']:
        for epoch, path in zip(SNAPSHOTS, training['checkpoints']):
            snapshot = load_model(path, device=request['device'])
            curves.append(dict(epoch=epoch, training_evaluation=evaluate_cache(snapshot, cache, split['train_ids'])))
    verify_request(request)
    value = dict(split=split, config=request['config'], seed=request['seed'], training=training,
                 evaluation=evaluation, seed_evaluation=seed_evaluation, full_fit_curves=curves,
                 epoch_selection_used_holdout=False, new_pnr=0, worker_pid=os.getpid(),
                 worker_threads=torch.get_num_threads(), cache_file_sha256=request['cache_file_sha256'])
    freeze(out/'fit_report.json', value)
    print('TAILCOMPILE_GENERALIZATION_FIT_COMPLETE', flush=True)


def verify_fit(value, split, config, seed, cache, device, *, checkpoints=True):
    train = value['training']
    if value['split'] != split or value['config'] != config or value['seed'] != seed or \
            train['seed'] != seed or train['epochs'] != EPOCHS or train['device'] != device or \
            train['cache_sha256'] != cache['cache_sha256'] or train['train_ids'] != sorted(split['train_ids']) or \
            train['source_checkpoint_sha256'] != cache['source_checkpoint_sha256'] or value['worker_threads'] != 1:
        raise ValueError('013 completed fit differs from the frozen request')
    if train['factor_config'] != config or train['mode'] != config['mode'] or train['test_checkpoint_epoch_fixed'] != EPOCHS or \
            train['normalization']['kind'] != config['normalization'] or \
            train['normalization']['train_ids'] != sorted(split['train_ids']) or \
            train['optimizer'] != dict(name='Adam', learning_rate=LEARNING_RATE,
                weight_decay=config['weight_decay'], penalty='coupled L2 on head parameters only'):
        raise ValueError('013 actual training factors or normalization fold differ')
    if checkpoints:
        for path, expected in train['checkpoint_sha256'].items():
            if sha(path) != expected:
                raise ValueError('013 completed checkpoint bytes changed')


def compare_rows(method, baseline):
    lookup = {(r['condition_id'], r['arm']): r for r in baseline}
    if len(lookup) != len(baseline) or {(r['condition_id'], r['arm']) for r in method} != set(lookup):
        raise ValueError('013 contrast populations differ')
    rows = [dict(r, D_WL=lookup[r['condition_id'], r['arm']]['selected_WL'],
                 D_CPD=lookup[r['condition_id'], r['arm']]['selected_CPD'],
                 D_success_count=lookup[r['condition_id'], r['arm']]['selected_success_count']) for r in method]
    return aggregate(rows)


def summarize_fits(fits):
    """Fixed-fold summaries and factorial contrasts, without choosing a winner."""
    grouped = defaultdict(list)
    historical = defaultdict(list)
    rankings = defaultdict(lambda: dict(correct=0, pairs=0, ties=0))
    matrix = set()
    for fit in fits:
        split, config, seed = fit['split'], fit['config'], fit['seed']
        identity = split['id'], config['id'], seed
        if identity in matrix:
            raise ValueError('013 duplicate fit evidence')
        matrix.add(identity)
        if split['heldout'] and set(split['train_ids']) & set(split['evaluation_ids']):
            raise ValueError('013 held-out condition leaked into training')
        training = fit['training']
        if training['calibration']['train_ids'] != sorted(split['train_ids']):
            raise ValueError('013 calibration used a different training fold')
        if training['factor_config'] != config or training['normalization']['train_ids'] != sorted(split['train_ids']):
            raise ValueError('013 factor or normalizer evidence used a different fold')
        key = split['group'], config['id'], seed
        grouped[key].extend(fit['evaluation']['rows'])
        for arm, rank in fit['evaluation']['ranking'].items():
            for name in ('correct', 'pairs', 'ties'):
                rankings[key+(arm,)][name] += rank[name]
        predictions = {(r['condition_id'], r['arm']): r['selected_digest'] for r in fit['evaluation']['rows']}
        for vpr_seed, values in fit['seed_evaluation'].items():
            if {(r['condition_id'], r['arm']): r['selected_digest'] for r in values['rows']} != predictions:
                raise ValueError('013 measured-seed evaluation changed the prediction')
            historical[key+(vpr_seed,)].extend(values['rows'])
    summaries = {'/'.join(map(str, key)): aggregate(rows) for key, rows in grouped.items()}
    contrasts = {}
    for group, config_id, seed in grouped:
        config = config_by_id(config_id)
        if config['mode'] == 'G':
            peer_id = config_id.replace('G_', 'E_', 1)
            contrasts[f'{group}/s{seed}/local/{config_id}'] = compare_rows(
                grouped[group, peer_id, seed], grouped[group, config_id, seed])
            continue
        for axis, treated in (('hidden', 8), ('normalization', 'train_std'), ('weight_decay', .01)):
            if config[axis] != treated:
                continue
            peer = dict(config, **{axis: {'hidden': 32, 'normalization': 'raw', 'weight_decay': 0.}[axis]})
            peer_id = next(c['id'] for c in CONFIGS if all(c[k] == peer[k] for k in ('mode','hidden','normalization','weight_decay')))
            contrasts[f'{group}/s{seed}/{axis}/{config_id}'] = compare_rows(grouped[group, config_id, seed],
                                                                                        grouped[group, peer_id, seed])
    return dict(schema='generalization-evaluation-013-1', complete=True, offline_fits=len(fits), new_pnr=0,
                summaries=summaries, factorial_and_local_contrasts=contrasts,
                rankings={'/'.join(map(str, key)): value for key, value in rankings.items()},
                historical_vpr_seed_summaries={'/'.join(map(str, key)): aggregate(value) for key, value in historical.items()},
                primary_configuration=PRIMARY_CONFIG, primary_selected_before_results=True,
                heldout_used_for_model_selection=False, independent_family_test=False, Wmin_measured=False,
                frozen_backbone_family_exposure=True, original_actions_and_labels_unchanged=True,
                historical_measurements_are_not_new_pnr=True, fits=fits)


def baseline_reproduction(fits, source):
    """Compare raw controls to the archived012 fit, without importing its weights."""
    checks = []
    for fit in fits:
        config = fit['config']
        if (config['hidden'], config['normalization'], config['weight_decay']) != (32, 'raw', 0.):
            continue
        path = source/'models'/fit['split']['id']/config['mode']/f's{fit["seed"]}'/'fit_report.json'
        if not path.is_file():
            checks.append(dict(split=fit['split']['id'], config=config['id'], seed=fit['seed'], available=False))
            continue
        old = read(path)
        signature = lambda f: [(r['condition_id'], r['arm'], r['selected_digest']) for r in f['evaluation']['rows']]
        same_device = fit['training']['device'] == old['training']['device']
        checks.append(dict(split=fit['split']['id'], config=config['id'], seed=fit['seed'], available=True,
            same_device=same_device, old_report_sha256=sha(path),
            initial_weights_identical=fit['training']['initial_state_sha256'] == old['training']['initial_state_sha256'],
            final_weights_identical=fit['training']['final_state_sha256'] == old['training']['final_state_sha256'],
            predictions_identical=signature(fit) == signature(old),
            ranking_identical=fit['evaluation']['ranking'] == old['evaluation']['ranking']))
    same_device_checks = [c for c in checks if c['available'] and c['same_device']]
    exact = all(
        c['initial_weights_identical'] and c['final_weights_identical'] and c['predictions_identical'] and c['ranking_identical']
        for c in same_device_checks) if same_device_checks else None
    return dict(checks=checks, exact_reproduction_all_available_same_device=exact,
        checked_same_device=len(same_device_checks),
        available=sum(c['available'] for c in checks), planned=28,
        role='control reproducibility, not a new feature benefit')


class Campaign:
    def __init__(self, args):
        self.args, self.out = args, owned(args.out)

    def setup(self):
        self.source = inspect_origin(self.args.source)
        if not self.source['ready']:
            raise ValueError('013 server input missing: '+str(self.source['missing']))
        freeze(self.out/'generalization_protocol.json', dict(plan=plan(), source=str(self.args.source),
            source_inputs=self.source['source_inputs'], implementation=implementation(),
            base_checkpoint_sha256=self.source['base_checkpoint_sha256'],
            execution=execution_options(self.args.device, self.args.jobs, self.args.fit_timeout)))
        freeze(self.out/'execution'/self.args.stage/'protocol.json', dict(device=self.args.device,
            **execution_options(self.args.device, self.args.jobs, self.args.fit_timeout)))

    def prepare(self):
        report = self.out/'prepared_report.json'
        if report.exists():
            self.cache()
            return
        from edit_context_cache_import import import_cache
        from train_edit_context import load_cache
        imported = import_cache(self.source['source_info'], self.args.source,
                                current_implementation={n: sha(HERE/n) for n in SOURCE_CODE})
        cache = load_cache(imported['cache'])
        freeze(report, dict(imported, splits=splits(cache['banks']), original_feature_bytes_read_only=True))

    def cache(self):
        from train_edit_context import load_cache
        prepared = read(self.out/'prepared_report.json')
        if sha(prepared['cache']) != prepared['cache_file_sha256']:
            raise ValueError('013 source012 cache changed')
        cache = load_cache(prepared['cache'])
        if cache['cache_sha256'] != prepared['cache_sha256']:
            raise ValueError('013 source012 cache content differs')
        return cache

    def train(self):
        cache = self.cache()
        options = execution_options(self.args.device, self.args.jobs, self.args.fit_timeout)
        tasks = list(itertools.product(splits(cache['banks']), CONFIGS, TRAIN_SEEDS))
        budget = resource_budget(min(options['jobs'], len(tasks)), 1024, reserve_mib=1024)
        cores = queue.Queue()
        for core in budget['physical_cores']:
            cores.put(core)
        prepared = read(self.out/'prepared_report.json')
        code = implementation()
        freeze(self.out/'execution/train/parallel_protocol.json', dict(**options, fit_count=len(tasks),
            matrix=[[s['id'], c['id'], seed] for s, c, seed in tasks],
            cache=prepared['cache'], cache_file_sha256=prepared['cache_file_sha256']))
        observations = self.out/'execution/train/parallel_runs'
        write(observations/f'attempt_{len(list(observations.glob("attempt_*")))+1:04d}'/'metrics.json',
              dict(effective_jobs=budget['effective'], budget=budget, started_unix=time.time(), pid=os.getpid()))
        print(json.dumps(dict(event='generalization_parallel_start', planned=len(tasks),
                              effective_jobs=budget['effective'], threads_per_fit=1)), flush=True)

        def run_task(task):
            raise_if_cancelled()
            split, config, seed = task
            folder = self.out/'models'/split['id']/config['id']/f's{seed}'
            durable = folder/'fit_report.json'
            request = dict(cache=prepared['cache'], cache_file_sha256=prepared['cache_file_sha256'],
                           cache_sha256=cache['cache_sha256'], split=split, config=config, seed=seed,
                           device=self.args.device, fit_timeout=options['fit_timeout'], implementation=code)
            value = read(durable) if durable.exists() else None
            attempts = sorted(p for p in folder.glob('attempt_*') if p.is_dir())
            if value is None:
                for attempt in attempts:
                    if (attempt/'fit_report.json').is_file() and (attempt/'worker/manifest.json').is_file() and \
                            read(attempt/'worker/manifest.json')['status'] == 'complete':
                        if read(attempt/'request.json') != request:
                            raise ValueError('013 resumed request differs')
                        value = dict(read(attempt/'fit_report.json'), worker_manifest=str(attempt/'worker/manifest.json'))
                        break
            if value is None:
                if len(attempts) >= 2:
                    raise RuntimeError('013 retry budget exhausted for '+str(folder))
                attempt = folder/f'attempt_{len(attempts)+1:04d}'
                freeze(attempt/'request.json', request)
                core = cores.get()
                try:
                    result = run_trial(attempt/'worker', [sys.executable, HERE/'generalization_campaign.py',
                        'fit-worker', '--request', attempt/'request.json', '--out', attempt],
                        inputs=[attempt/'request.json', prepared['cache']]+[HERE/n for n in CODE],
                        affinity=core, timeout=options['fit_timeout'],
                        marker='TAILCOMPILE_GENERALIZATION_FIT_COMPLETE', prune=False)
                finally:
                    cores.put(core)
                if result['status'] != 'complete':
                    raise RuntimeError('013 isolated fit failed: '+result['log'])
                value = dict(read(attempt/'fit_report.json'), worker_manifest=str(attempt/'worker/manifest.json'))
            verify_fit(value, split, config, seed, cache, self.args.device)
            manifest = read(value['worker_manifest'])
            if manifest['status'] != 'complete' or manifest['returncode'] != 0:
                raise ValueError('013 completed fit has no successful process evidence')
            freeze(durable, value)
            return dict(path=str(durable), sha256=sha(durable), split_id=split['id'], config_id=config['id'], seed=seed)

        progress = {}
        def completed(index, value):
            progress[index] = value
            write(self.out/'train_progress_report.json', dict(completed=len(progress), planned=len(tasks),
                effective_jobs=budget['effective'], fits=[progress[k] for k in sorted(progress)]))
            print(json.dumps(dict(event='generalization_fit_complete', completed=len(progress),
                planned=len(tasks), split=value['split_id'], config=value['config_id'], seed=value['seed'])), flush=True)
        fits = parallel_fit_map(tasks, jobs=budget['effective'], run_task=run_task,
                                on_complete=completed, cancel_running=_stop_fit_processes)
        summary = self.out/'trained_report.json'
        first_jobs = read(summary)['effective_jobs'] if summary.exists() else budget['effective']
        freeze(summary, dict(complete=len(fits) == 140, planned=140, fits=fits, effective_jobs=first_jobs,
                             isolated_processes=True, threads_per_fit=1, new_pnr=0,
                             no_holdout_checkpoint_selection=True))

    def report(self):
        trained = read(self.out/'trained_report.json')
        prepared = read(self.out/'prepared_report.json')
        expected = {(s['id'], c['id'], seed) for s, c, seed in itertools.product(prepared['splits'], CONFIGS, TRAIN_SEEDS)}
        actual = {(e['split_id'], e['config_id'], e['seed']) for e in trained['fits']}
        if not trained['complete'] or len(trained['fits']) != 140 or actual != expected:
            raise ValueError('013 fit matrix is incomplete')
        fits = []
        for entry in trained['fits']:
            if sha(entry['path']) != entry['sha256']:
                raise ValueError('013 fit report bytes changed')
            fit = read(entry['path'])
            if (fit['split']['id'], fit['config']['id'], fit['seed']) != (entry['split_id'], entry['config_id'], entry['seed']):
                raise ValueError('013 fit index identity differs')
            fits.append(fit)
        report = summarize_fits(fits)
        report['baseline012_reproduction'] = baseline_reproduction(fits, self.args.source)
        report['control_reproducibility_issue'] = report['baseline012_reproduction']['exact_reproduction_all_available_same_device'] is False
        write(self.out/'generalization_report.json', report)
        print(json.dumps(dict(complete=True, offline_fits=140, new_pnr=0,
                              report=str(self.out/'generalization_report.json'))), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES+('fit-worker',))
    parser.add_argument('--source', type=Path, default=HERE/'runs/development_012_edit_context_parallel')
    parser.add_argument('--out', type=Path, default=HERE/'runs/development_013_generalization')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--jobs', type=int)
    parser.add_argument('--fit-timeout', type=int, default=1800)
    parser.add_argument('--request', type=Path)
    args = parser.parse_args()
    try:
        args.jobs = execution_options(args.device, args.jobs, args.fit_timeout)['jobs']
        args.out, args.source = owned(args.out), origin_path(args.source)
    except ValueError as exc:
        parser.error(str(exc))
    install_signal_handlers()
    if args.stage == 'fit-worker':
        if args.request is None:
            parser.error('fit-worker requires --request')
        fit_worker(read(args.request), args.out)
        return
    if args.stage == 'plan':
        print(json.dumps(plan(), indent=2))
        return
    if args.stage == 'check':
        result = inspect_origin(args.source)
        print(json.dumps({k: v for k, v in result.items() if k not in ('source_info', 'source_inputs')}, indent=2))
        raise SystemExit(0 if result['ready'] else 2)
    with campaign_lock(args.out, 'generalization-'+args.stage):
        runner = Campaign(args)
        runner.setup()
        getattr(runner, args.stage)()


if __name__ == '__main__':
    main()
