"""013 parallel engineering replay on the existing tiny measured012 cache."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

from campaign import resource_budget
from generalization_spec import CONFIGS, TRAIN_SEEDS
from io_utils import freeze, read, sha, write
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent


def initialize_worker():
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)


def run_fit(request):
    import torch
    from train_edit_context import load_cache
    from train_generalization import fit_cache, load_model, evaluate_cache
    cache = load_cache(request['cache'])
    ids = sorted({b['condition_id'] for b in cache['banks']})
    output = Path(request['out'])/request['config']['id']/f's{request["seed"]}'
    existing = output/'training_report.json'
    if existing.is_file():
        training = read(existing)
        if training['factor_config'] != request['config'] or training['seed'] != request['seed'] or \
                training['cache_sha256'] != cache['cache_sha256']:
            raise ValueError('013 resumed local diagnostic has different frozen inputs')
        for path, expected in training['checkpoint_sha256'].items():
            if sha(path) != expected:
                raise ValueError('013 resumed local diagnostic checkpoint changed')
    else:
        training = fit_cache(cache, ids, output, config=request['config'], seed=request['seed'])
    model = load_model(training['checkpoint'])
    evaluation = evaluate_cache(model, cache, ids)
    return dict(config=request['config'], seed=request['seed'], training=training, evaluation=evaluation,
                pid=os.getpid(), threads=torch.get_num_threads())


def run(output, jobs):
    output = Path(output).resolve()
    if output.parent != HERE/'runs' or not output.name.startswith('development_013_generalization_local'):
        raise ValueError('013 local output requires an isolated ignored013 local cohort')
    source = HERE/'runs/development_012_edit_context_local/feature_cache.pt'
    validate_output_roots((output,), dependency_roots=(source.parent,))
    budget = resource_budget(min(jobs, len(CONFIGS)*len(TRAIN_SEEDS)), 1024, reserve_mib=1024)
    before = sha(source)
    freeze(output/'protocol.json', dict(source=str(source), source_sha256=before, configs=CONFIGS,
        seeds=list(TRAIN_SEEDS), planned=20, new_pnr=0, natural_design_claim=False))
    tasks = [dict(cache=str(source), out=str(output), config=config, seed=seed)
             for config in CONFIGS for seed in TRAIN_SEEDS]
    with ProcessPoolExecutor(max_workers=budget['effective'], mp_context=multiprocessing.get_context('spawn'),
                             initializer=initialize_worker) as pool:
        fits = list(pool.map(run_fit, tasks))
    assert sha(source) == before
    assert len(fits) == 20 and all(f['threads'] == 1 for f in fits)
    comparisons = []
    for fit in fits:
        config = fit['config']
        if config['hidden'] != 32 or config['normalization'] != 'raw' or config['weight_decay'] != 0:
            continue
        old = HERE/'runs/development_012_edit_context_local/fits/full_fit'/f's{fit["seed"]}'/config['mode']/'training_report.json'
        if old.is_file():
            original = read(old)
            assert original['final_state_sha256'] == fit['training']['final_state_sha256']
            assert original['training_evaluation'] == fit['training']['training_evaluation']
            comparisons.append(dict(config=config['id'], seed=fit['seed'], original012_exact=True))
    report = dict(schema='generalization-local-smoke-013-1', passed=True, new_pnr=0,
        offline_fits=20, effective_jobs=budget['effective'], distinct_processes=len({f['pid'] for f in fits}),
        source_unchanged=True, original012_checks=comparisons, fits=fits,
        natural_design_claim=False, performance_claim=False, full_server_matrix_completed=False)
    write(output/'generalization_smoke_report.json', report)
    # These local diagnostic models are reproducible and are not deployed.
    # Remove only the exact checkpoint files recorded by this smoke run.
    removed = []
    for fit in fits:
        for path, expected in fit['training']['checkpoint_sha256'].items():
            path = Path(path).resolve()
            if output not in path.parents or path.suffix != '.pt' or sha(path) != expected:
                raise ValueError('013 local cleanup checkpoint path/signature differs')
            removed.append(dict(path=str(path), bytes=path.stat().st_size))
            path.unlink()
    write(output/'payload_pruned.json', dict(checkpoints=removed, bytes=sum(p['bytes'] for p in removed),
        source_feature_cache_preserved=True, reason='completed local engineering diagnostics'))
    print(dict(passed=True, offline_fits=20, jobs=budget['effective'], new_pnr=0,
               original012_checks=len(comparisons), report=str(output/'generalization_smoke_report.json')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'runs/development_013_generalization_local')
    parser.add_argument('--jobs', type=int, default=4)
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error('jobs must be a positive integer')
    run(args.out, args.jobs)
