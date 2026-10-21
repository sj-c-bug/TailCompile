"""Process-parallel candidate throughput, without changing generator semantics.

Each candidate is an independent task, so the eight costly decodes of one
context may run concurrently. Workers cache their most recent graph/embeddings.
Completed manifests retain the exact legacy infer.py/model.py
signature and are checked before reuse.  This scheduler does not alter scores,
sampling seeds, capacities, lowering, or formal isolated runtime measurement.
"""
from __future__ import annotations

import concurrent.futures as cf
import multiprocessing as mp
import os
from pathlib import Path
import queue
import signal
import socket
import time

from common import emit, read, sha, write

HERE = Path(__file__).resolve().parent
SCHEMA = 'tailcompile-v68-hierarchical-initializer-1'
PROGRESS_SCHEMA = 'tailcompile-v68-generation-progress-1'
_GENERATOR = None
_CONFIG = None
_EVENTS = None
_LAST_CONTEXT = None


def candidate_folder(root, context, candidate, mode):
    design = str(context['design'])
    if design in ('.', '..') or '/' in design or '\\' in design:
        raise ValueError('candidate design must be a single directory name')
    width = int(context['width'])
    if width < 1:
        raise ValueError('candidate width must be positive')
    folder = Path(root) / design / f'w{width}'
    return folder / f'c{candidate}' if mode == 'feedback' else folder


def candidate_options(candidate, mode):
    if mode == 'feedback':
        return dict(candidate=candidate, seed=680000 + candidate,
                    score_noise_std=.25 if candidate else 0.,
                    coordinate_noise_std=.05 if candidate else 0.)
    if mode == 'evaluation':
        if candidate != 0:
            raise ValueError('evaluation has one deterministic deployed candidate')
        return dict(candidate=0, seed=0, score_noise_std=0., coordinate_noise_std=0.)
    raise ValueError(f'unknown candidate mode: {mode}')


def generation_signature(context, checkpoint_sha256, candidate, mode, digest_cache=None):
    """Keep exactly the signature produced by unchanged Generator.generate."""
    cache = {} if digest_cache is None else digest_cache

    def digest(path):
        key = str(Path(path).resolve())
        if key not in cache:
            cache[key] = sha(path)
        return cache[key]

    return dict(schema=SCHEMA, checkpoint_sha256=checkpoint_sha256,
        graph_sha256=digest(context['graph']), logic_sha256=digest(context['logic']),
        device_sha256=digest(context['device']), supply_sha256=digest(context['supply']),
        **candidate_options(candidate, mode),
        implementation={p.name: digest(p) for p in (HERE/'infer.py', HERE/'model.py')})


def validate_cached_generation(folder, expected_signature):
    """Return a verified existing result; never overwrite incompatible evidence."""
    folder = Path(folder)
    manifest = folder/'generation.json'
    if not manifest.exists():
        return None
    old = read(manifest)
    if old.get('signature') != expected_signature:
        raise ValueError(f'Generated initializer provenance changed: {folder}')
    for key, name, checksum in (('ap_init', 'init.ap.tsv', 'ap_init_sha256'),
                                ('fplace', 'init.fplace', 'fplace_sha256')):
        artifact = folder/name
        if (not artifact.is_file() or Path(old.get(key, '')).resolve() != artifact.resolve()
                or sha(artifact) != old.get(checksum)):
            raise ValueError(f'Generated initializer output missing/corrupted: {artifact}')
    target = folder/'target.json'
    if not target.is_file() or Path(old.get('target', '')).resolve() != target.resolve():
        raise ValueError(f'Generated initializer target missing/changed path: {target}')
    # Legacy manifests have no target hash; do not invent one and invalidate all
    # existing candidates.  The generation protocol and AP/fplace hashes stay strict.
    if old.get('target_sha256') and sha(target) != old['target_sha256']:
        raise ValueError(f'Generated initializer target corrupted: {target}')
    target_data = read(target)
    if (any(not isinstance(target_data.get(k), list)
            for k in ('teacher_region', 'teacher_xy', 'movable')) or
            len({len(target_data[k]) for k in ('teacher_region', 'teacher_xy', 'movable')}) != 1):
        raise ValueError(f'Generated initializer target shape corrupted: {target}')
    return old


def _worker_init(checkpoint, device, events):
    global _GENERATOR, _CONFIG, _EVENTS, _LAST_CONTEXT
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '1'
    os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
    import torch
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    _GENERATOR = None
    _LAST_CONTEXT = None
    _CONFIG = (checkpoint, device)
    _EVENTS = events


def _generate_candidate(task):
    """Spawn-picklable worker, never calls VPR or shares a Generator across threads."""
    global _GENERATOR, _LAST_CONTEXT
    context, entry, mode = task
    candidate, folder, signature = entry
    context_key = context['design'], context['width']
    key = (*context_key, candidate)
    if _EVENTS is not None:
        _EVENTS.put(('started', key, os.getpid()))
    old = validate_cached_generation(folder, signature)
    reused = old is not None
    if old is None:
        if _GENERATOR is None:
            from infer import Generator
            _GENERATOR = Generator(*_CONFIG)
        if _LAST_CONTEXT != context_key:
            # Bound graph residency, but reuse encodings if the next task happens
            # to have this same context. Expensive decode is parallel regardless.
            _GENERATOR.graph_cache = None
            _GENERATOR.logic_cache = None
            _GENERATOR.device_cache.clear()
            _LAST_CONTEXT = context_key
        old = _GENERATOR.generate(context, folder, **candidate_options(candidate, mode))
        if old['signature'] != signature:
            raise ValueError(f'Generation inputs changed while worker ran: {folder}')
    if _EVENTS is not None:
        _EVENTS.put(('candidate', key, reused))
    return candidate, old, reused, os.getpid()


def _terminate_pool(executor):
    """Python 3.10 equivalent of terminate_workers, limited to our own children."""
    processes = list((getattr(executor, '_processes', None) or {}).values())
    executor.shutdown(wait=False, cancel_futures=True)
    for process in processes:
        if process.is_alive():
            process.terminate()
    for process in processes:
        process.join(timeout=5)
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def generate_contexts(contexts, checkpoint, root, *, jobs=16, candidates=8,
                      mode='feedback', device='cpu', progress_path=None):
    """Generate/resume candidates with deterministic outputs and CPU parallelism.

    CPU uses spawned processes; each process has one Torch/BLAS thread.  CUDA
    generation is permitted only with jobs=1, never 16/32 replicas on one GPU.
    Existing valid candidates are checked without loading a checkpoint/model.
    """
    if jobs < 1 or candidates < 1:
        raise ValueError('generation jobs and candidates must be positive')
    if mode not in ('feedback', 'evaluation'):
        raise ValueError(f'unknown candidate mode: {mode}')
    if mode == 'evaluation' and candidates != 1:
        raise ValueError('evaluation requires exactly one deterministic candidate')
    if str(device).split(':')[0] != 'cpu' and jobs != 1:
        raise ValueError('parallel generation is CPU-only; CUDA requires jobs=1')
    contexts = list(contexts)
    context_keys = [(c['design'], c['width']) for c in contexts]
    if len(set(context_keys)) != len(context_keys):
        raise ValueError('duplicate generation context would share candidate output paths')
    root = Path(root).resolve()
    progress_path = Path(progress_path) if progress_path else root.parent/'generation_progress.json'
    began = time.monotonic()
    checkpoint = str(Path(checkpoint).resolve())
    checkpoint_hash = sha(checkpoint)
    digest_cache, generated, expected, tasks = {}, {}, {}, []
    completed, reused, running, failed, worker_pids = set(), set(), {}, [], set()
    total = len(contexts)*candidates
    for context in contexts:
        for candidate in range(candidates):
            key = context['design'], context['width'], candidate
            folder = candidate_folder(root, context, candidate, mode)
            signature = generation_signature(context, checkpoint_hash, candidate, mode, digest_cache)
            expected[key] = folder, signature
            old = validate_cached_generation(folder, signature)
            if old is not None:
                generated[key] = old
                completed.add(key)
                reused.add(key)
            else:
                tasks.append((context, (candidate, str(folder), signature), mode))
    workers = min(jobs, len(tasks))
    execution = dict(schema='tailcompile-v68-generation-execution-1', mode=mode,
        host=socket.gethostname(), parent_pid=os.getpid(), requested_jobs=jobs, workers=workers,
        inner_threads=1, process_start_method='spawn', device=str(device),
        checkpoint_sha256=checkpoint_hash, contexts=len(contexts), total_candidates=total,
        reused_candidates=len(reused), pending_tasks=len(tasks), task_granularity='candidate',
        implementation_sha256=sha(Path(__file__)),
        scope='Parallel throughput collection only; not isolated per-task runtime/speedup evidence.',
        generator_algorithm_changed=False)
    write(progress_path.with_name('generation_execution.json'), execution)

    def progress(complete=False, interrupted=False):
        completed_contexts = sum(all((d, w, k) in completed for k in range(candidates))
                                 for d, w in context_keys)
        record = dict(schema=PROGRESS_SCHEMA, complete=complete, interrupted=interrupted,
            mode=mode, jobs=jobs, workers=workers, device=str(device),
            expected_contexts=len(contexts), completed_contexts=completed_contexts,
            total_candidates=total, completed_candidates=len(completed),
            reused_candidates=len(reused), new_candidates=len(completed-reused),
            pending_candidates=total-len(completed),
            failed_contexts=len({(r['design'], r['width']) for r in failed}),
            failed_candidates=len(failed),
            worker_pids=sorted(worker_pids), task_granularity='candidate',
            failures=failed, running_contexts=[dict(design=k[0], width=k[1], candidate=k[2], pid=pid)
                                             for k, pid in sorted(running.items())],
            elapsed_seconds=round(time.monotonic()-began, 3),
            note='Candidate throughput only; VPR evaluation is a separate following stage.')
        write(progress_path, record)
        emit('generation_progress', **{k:v for k,v in record.items()
             if k not in ('schema', 'note', 'failures', 'running_contexts')})
        return record

    progress()
    if not tasks:
        progress(complete=True)
        return generated
    context = mp.get_context('spawn')
    events = context.Queue()
    executor = None
    previous_handler = None
    handler_installed = False

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'generation interrupted by signal {signum}; completed candidates retained')

    def drain_events():
        while True:
            try:
                event = events.get_nowait()
            except queue.Empty:
                break
            kind, key, value = event
            if kind == 'started':
                if tuple(key) not in completed:
                    running[tuple(key)] = value
                worker_pids.add(value)
            elif kind == 'candidate':
                completed.add(tuple(key))
                running.pop(tuple(key), None)
                if value:
                    reused.add(tuple(key))

    try:
        # Signal registration is only valid in the main thread. Library callers
        # in another thread still get KeyboardInterrupt/error cleanup below.
        try:
            previous_handler = signal.signal(signal.SIGTERM, interrupted)
            handler_installed = True
        except ValueError:
            pass
        executor = cf.ProcessPoolExecutor(max_workers=workers, mp_context=context,
            initializer=_worker_init, initargs=(checkpoint, str(device), events))
        futures = {executor.submit(_generate_candidate, task): (task[0]['design'], task[0]['width'], task[1][0])
                   for task in tasks}
        pending_futures = set(futures)
        while pending_futures:
            done, pending_futures = cf.wait(pending_futures, timeout=5,
                                           return_when=cf.FIRST_COMPLETED)
            drain_events()
            for future in done:
                key = futures[future]
                running.pop(key, None)
                try:
                    candidate, result, was_reused, pid = future.result()
                    if candidate != key[2] or result['signature'] != expected[key][1]:
                        raise ValueError(f'Worker generation signature changed: {key}')
                    generated[key] = result
                    completed.add(key)
                    worker_pids.add(pid)
                    if was_reused:
                        reused.add(key)
                except Exception as error:
                    failed.append(dict(design=key[0], width=key[1], candidate=key[2], error=str(error)))
                    emit('generation_candidate_failed', design=key[0], width=key[1], candidate=key[2], error=str(error))
            progress()
        executor.shutdown(wait=True)
        executor = None
        drain_events()
        running.clear()
        progress(complete=not failed and len(completed) == total)
        if failed:
            raise RuntimeError(f'{len(failed)} candidate generation tasks failed; completed files retained. '
                               f'First: {failed[0]}')
        return generated
    except BaseException as error:
        # A stopped parent must not leave 16/32 orphan writers generating more
        # candidates after the pipeline lock has been released.
        if executor is not None:
            _terminate_pool(executor)
            executor = None
        running.clear()
        progress(interrupted=isinstance(error, (KeyboardInterrupt, SystemExit)))
        raise
    finally:
        if handler_installed:
            signal.signal(signal.SIGTERM, previous_handler)
        events.close()
        events.join_thread()
