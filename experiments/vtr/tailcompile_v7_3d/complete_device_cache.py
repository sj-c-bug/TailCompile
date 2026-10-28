"""Complete interrupted device-graph preparation with the frozen extractor.

This preparation-only recovery never runs VPR, changes a campaign protocol, or
relabels a P&R trial. It publishes the same cache manifest as Campaign.rr after
checking the complete producer and graph provenance.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import math
from pathlib import Path
import signal
import threading
import time

from campaign import campaign_lock, identity
from device_graph import build_device_graph
from io_utils import COMMIT, emit, read, sha, write
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
IMPLEMENTATION = ('campaign.py', 'architectures.py', 'device_graph.py', 'data.py',
                  'model.py', 'generation.py', 'supervision.py', 'train.py',
                  'io_utils.py', 'repository_guard.py', 'recover_capacity.py')


def _inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f'device cache recovery path escapes its owner: {path}')
    return path


def _require_hash(path, digest):
    if not isinstance(digest, str) or sha(path) != digest:
        raise ValueError(f'device cache recovery input changed: {path}')


def _validate_protocol(out, repo):
    protocol = read(out/'protocol.json')
    if protocol.get('schema') != 'tailcompile-v7-campaign-1':
        raise ValueError('unsupported frozen campaign protocol')
    if protocol.get('config') != read(out/'campaign.json'):
        raise ValueError('campaign configuration differs from its frozen protocol')
    if set(protocol.get('implementation', {})) != set(IMPLEMENTATION):
        raise ValueError('unregistered device cache recovery implementation set')
    sources = {}
    for name, digest in protocol['implementation'].items():
        source = HERE/name
        _require_hash(source, digest)
        sources[str(source.resolve())] = digest
    inherited = protocol.get('inherited_implementation', {})
    if not inherited:
        raise ValueError('campaign lacks inherited implementation provenance')
    for name, digest in inherited.items():
        source = _inside(name, repo)
        _require_hash(source, digest)
        sources[str(source)] = digest
    sources[str(Path(__file__).resolve())] = sha(__file__)
    reference = out/'binary_identity.json'
    binary = read(reference)
    digest = protocol.get('binary_sha256')
    if (binary.get('commit') != COMMIT or binary.get('sha256') != digest or
            binary.get('backend_manifest_sha256') != protocol.get('backend_manifest_sha256')):
        raise ValueError('frozen backend reference does not match the campaign')
    executable = Path(binary['binary']).resolve()
    _require_hash(executable, digest)
    qualification = read(out/'qualification/qualification_report.json')
    qualified_protocol = read(out/'qualification/protocol.json')
    if (not qualification.get('complete') or not qualification.get('passed') or
            qualified_protocol.get('commit') != COMMIT or qualified_protocol.get('binary_sha256') != digest):
        raise ValueError('qualified backend evidence does not match the campaign')
    tile_size = protocol['config'].get('tile_size', 4)
    if isinstance(tile_size, bool) or not isinstance(tile_size, int) or tile_size < 1:
        raise ValueError('invalid frozen tile size')
    return dict(binary=executable, binary_sha256=digest, binary_reference=reference,
                tile_size=tile_size, source_hashes=sources,
                protocol_sha256=sha(out/'protocol.json'))


def _owned_cache(folder, out):
    cache_root = _inside(out/'device_cache', out)
    folder = Path(folder).resolve()
    if folder.parent != cache_root or not folder.is_dir():
        raise ValueError('device cache must be a direct owned cache directory')
    for name in ('manifest.json', 'device_graph.json', 'rr_graph.xml',
                 'extraction_completion_report.json'):
        if (folder/name).resolve().parent != folder:
            raise ValueError(f'device cache target escapes through a symlink: {name}')
    return folder


def _producer(folder, context, repo):
    build_path = folder/'build/manifest.json'
    if not build_path.is_file():
        return None
    build = read(build_path)
    if build.get('status') != 'complete' or build.get('returncode') != 0:
        return None
    command = build['signature']['command']
    if len(command) != 14:
        raise ValueError('unexpected RRG producer command')
    binary = Path(command[0]).resolve()
    architecture, blif = (_inside(command[i], repo) for i in (1, 2))
    try:
        width = int(command[7])
    except (ValueError, TypeError) as error:
        raise ValueError('invalid RRG channel width') from error
    rrg = folder/'rr_graph.xml'
    expected = [str(command[0]), str(command[1]), str(command[2]), '--pack', '--device', 'pilot',
                '--route_chan_width', str(width), '--place_chan_width', str(width),
                '--num_workers', '1', '--write_rr_graph', str(rrg)]
    if (binary != context['binary'] or command != expected or width < 1 or not rrg.is_file()):
        raise ValueError('RRG producer command/executable/output does not match the frozen inputs')
    signature = build['signature']
    if (signature.get('env') != {} or signature.get('marker') is not None or
            signature.get('affinity') is not None or
            signature.get('metadata') != dict(stage='reusable_rrg', width=width)):
        raise ValueError('unexpected RRG producer environment or metadata')
    required = {str(Path(context['binary_reference']).resolve()), str(architecture), str(blif)}
    inputs = signature.get('inputs', {})
    if set(inputs) != required:
        raise ValueError('RRG producer input set does not match its command')
    for name, digest in inputs.items():
        _inside(name, repo)
        _require_hash(name, digest)
    architecture_hash = sha(architecture)
    key = identity(dict(architecture=architecture_hash, binary=context['binary_sha256'], width=width))[:20]
    if key != folder.name:
        raise ValueError('RRG cache key does not match its architecture/binary/width')
    return dict(architecture=architecture, architecture_sha256=architecture_hash,
                blif=blif, width=width, rrg=rrg, build_manifest=build_path,
                build_manifest_sha256=sha(build_path), input_hashes=dict(inputs))


def _validate_complete(folder, producer, context):
    manifest = read(folder/'manifest.json')
    if (manifest.get('architecture_sha256') != producer['architecture_sha256'] or
            manifest.get('binary_sha256') != context['binary_sha256'] or
            manifest.get('channel_width') != producer['width']):
        raise ValueError('complete device cache identity mismatch; no overwrite permitted')
    if not (folder/'device_graph.json').is_file():
        raise ValueError('complete device cache graph is missing; no overwrite permitted')
    for name in ('rr_graph_sha256', 'graph_sha256'):
        value = manifest.get(name)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError('complete device cache lacks durable content hashes')
    # Existing caches stay immutable. Campaign.rr checks their full SHA before
    # use; repeated recovery passes must not reread tens of GB needlessly.


@contextmanager
def _extraction_deadline(seconds):
    """Bound otherwise late-checked Python extraction phases in the Linux CLI."""
    enabled = (threading.current_thread() is threading.main_thread() and
               hasattr(signal, 'setitimer') and hasattr(signal, 'SIGALRM'))
    if not enabled or signal.getitimer(signal.ITIMER_REAL) != (0.0, 0.0):
        yield
        return
    previous = signal.getsignal(signal.SIGALRM)
    def expired(signum, frame):
        raise TimeoutError(f'device graph recovery exceeded its {seconds}s wall-clock preparation budget')
    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def complete_pending(out, max_seconds=3600, *, repo=REPO):
    if isinstance(max_seconds, bool):
        raise ValueError('positive finite extraction budget required')
    try:
        max_seconds = float(max_seconds)
    except (TypeError, ValueError) as error:
        raise ValueError('positive finite extraction budget required') from error
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('positive finite extraction budget required')
    out, repo = Path(out).resolve(), Path(repo).resolve()
    runs = repo/'experiments/vtr/tailcompile_v7_3d/runs'
    if runs not in out.parents:
        raise ValueError('device cache completion must stay inside canonical v7 runs')
    validate_output_roots((out,), repo=repo)
    # Validate before opening a stage lock, which can create an output directory.
    context = _validate_protocol(out, repo)
    if (out/'.campaign-stage.lock').resolve().parent != out:
        raise ValueError('device preparation stage lock escapes through a symlink')
    completed = []
    with campaign_lock(out, 'complete-device-cache'):
        cache_root = _inside(out/'device_cache', out)
        for candidate in (sorted(cache_root.iterdir()) if cache_root.is_dir() else []):
            if not candidate.is_dir():
                continue
            folder = _owned_cache(candidate, out)
            producer = _producer(folder, context, repo)
            if producer is None:
                if (folder/'manifest.json').exists():
                    raise ValueError('complete cache lacks a completed auditable producer')
                emit('device_cache_completion_skipped', key=folder.name, reason='producer_not_complete')
                continue
            if (folder/'manifest.json').exists():
                _validate_complete(folder, producer, context)
                emit('device_cache_completion_cached', key=folder.name,
                     verification='identity_and_presence; full_content_verified_by_Campaign_before_use')
                continue
            rrg_hash = sha(producer['rrg'])
            report = dict(schema='tailcompile-v7-device-cache-completion-1', status='preparing',
                preparation_only=True, no_vpr_execution=True, original_budget_seconds=300,
                recovery_budget_seconds=max_seconds, protocol_sha256=context['protocol_sha256'],
                producer_manifest_sha256=producer['build_manifest_sha256'],
                rr_graph_sha256=rrg_hash, architecture_sha256=producer['architecture_sha256'],
                binary_sha256=context['binary_sha256'], parser_sha256=sha(HERE/'device_graph.py'),
                recovery_utility_sha256=sha(__file__),
                previous_uncommitted_graph_sha256=sha(folder/'device_graph.json') if (folder/'device_graph.json').is_file() else None)
            report_path = folder/'extraction_completion_report.json'
            write(report_path, report)
            emit('device_cache_completion_started', key=folder.name, max_seconds=max_seconds,
                 rr_graph_bytes=producer['rrg'].stat().st_size, serial=True)
            started = time.perf_counter()
            try:
                with _extraction_deadline(max_seconds):
                    graph = build_device_graph(producer['rrg'], producer['architecture'],
                        context['tile_size'], max_seconds=max_seconds)
                _require_hash(producer['rrg'], rrg_hash)
                _require_hash(producer['build_manifest'], producer['build_manifest_sha256'])
                for name, digest in {**context['source_hashes'], **producer['input_hashes']}.items():
                    _require_hash(name, digest)
                _require_hash(context['binary'], context['binary_sha256'])
                if (graph.get('schema') != 'tailcompile-v7-layered-device-1' or
                        graph.get('tile_size') != context['tile_size'] or
                        graph.get('channel_width') != producer['width'] or
                        graph.get('provenance', {}).get('rr_graph_sha256') != rrg_hash or
                        graph.get('provenance', {}).get('architecture_sha256') != producer['architecture_sha256']):
                    raise ValueError('recovered graph provenance/schema does not match the frozen extraction')
                write(folder/'device_graph.json', graph)
                write(folder/'manifest.json', dict(
                    architecture_sha256=producer['architecture_sha256'], binary_sha256=context['binary_sha256'],
                    channel_width=producer['width'], rr_graph_sha256=rrg_hash,
                    graph_sha256=sha(folder/'device_graph.json'), producer_blif_sha256=sha(producer['blif']),
                    reusable_across_designs=True, graph_is_not_a_placement_or_routability_certificate=True))
                report.update(status='complete', extraction_seconds=time.perf_counter()-started,
                              graph_sha256=sha(folder/'device_graph.json'))
                write(report_path, report)
                completed.append(folder.name)
                emit('device_cache_completion_finished', key=folder.name, seconds=report['extraction_seconds'])
            except BaseException as error:
                report.update(status='failed', extraction_seconds=time.perf_counter()-started,
                              error_type=type(error).__name__, error=str(error),
                              rr_graph_preserved=producer['rrg'].is_file())
                write(report_path, report)
                raise
    return completed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=3600)
    args = parser.parse_args()
    keys = complete_pending(args.out, args.max_seconds)
    print(dict(completed_device_caches=keys, preparation_only=True, vpr_runs_started=0), flush=True)


if __name__ == '__main__':
    main()
