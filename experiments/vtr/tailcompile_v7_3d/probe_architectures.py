"""Finite real-RRG qualification of H x Z and channel-width coupling."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import subprocess
import time

from architectures import COMMIT, SOURCE_ARCH, _module, audit_family, generate_architecture, sha
from device_graph import build_device_graph


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def run(args):
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(['git', '-C', str(args.vtr_root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != COMMIT:
        raise ValueError(f'expected {COMMIT}, found {head}')
    binary = args.binary or args.vtr_root/'vpr/vpr'
    blif = out/'tiny.blif'
    if not blif.exists():
        _module('probe_layer_heterogeneity').circuit(blif)
    arches = {}
    for h in (0, 1):
        for z in (0, 1):
            key = f'H{h}Z{z}'
            arch = out/key/'arch.xml'
            meta = arch.parent/'architecture.json'
            if not arch.exists():
                save(meta, generate_architecture(args.vtr_root/SOURCE_ARCH, arch, h, z,
                                                 args.width, args.height))
            elif sha(arch) != json.loads(meta.read_text())['architecture_sha256']:
                raise ValueError(f'architecture signature changed: {arch}')
            arches[key] = arch
    def trial(condition, channel):
        folder = out/condition/f'w{channel}'
        folder.mkdir(parents=True, exist_ok=True)
        rrg = folder/'rr_graph.xml'
        signature = dict(binary_sha256=sha(binary), architecture_sha256=sha(arches[condition]),
                         blif_sha256=sha(blif), channel_width=channel,
                         extractor_sha256=sha(Path(__file__).parent/'device_graph.py'),
                         generator_sha256=sha(Path(__file__).parent/'architectures.py'))
        graph_file = folder/'device_graph.json'
        old = folder/'manifest.json'
        if old.exists():
            record = json.loads(old.read_text())
            if record['signature'] != signature:
                raise ValueError(f'run signature changed: {folder}')
            if record.get('returncode') == 0 and graph_file.exists():
                value = json.loads(graph_file.read_text())
                if sha(graph_file) != record.get('device_graph_sha256'):
                    raise ValueError(f'graph checksum changed: {folder}')
                return condition, value
        command = [str(binary), str(arches[condition]), str(blif), '--pack',
                   '--device', 'pilot', '--num_workers', '1', '--seed', '701',
                   '--route_chan_width', str(channel), '--place_chan_width', str(channel),
                   '--write_rr_graph', str(rrg)]
        started = time.perf_counter()
        result = subprocess.run(command, cwd=folder, capture_output=True, text=True, timeout=args.timeout)
        (folder/'vpr.out').write_text(result.stdout+result.stderr)
        record = dict(signature=signature, command=command, returncode=result.returncode,
                      seconds=time.perf_counter()-started, placement_or_route_executed=False)
        save(old, record)
        if result.returncode or not rrg.exists():
            raise RuntimeError(f'RRG export failed: {folder}/vpr.out')
        value = build_device_graph(rrg, arches[condition], tile_size=args.tile_size)
        save(graph_file, value)
        record.update(device_graph_sha256=sha(graph_file), rr_graph_sha256=sha(rrg))
        save(old, record)
        return condition, value
    results = {key: [] for key in arches}
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(trial, key, width) for key in arches for width in args.channels]
        for future in as_completed(futures):
            key, value = future.result()
            results[key].append(value)
            print(json.dumps(dict(condition=key, width=value['channel_width'],
                                   z=value['audit']['z_capacity_by_class'], regions=len(value['regions']))), flush=True)
    audit = audit_family(results)
    audit['source_vtr_commit'] = head
    audit['binary_sha256'] = sha(binary)
    audit['source_files_sha256'] = {name: sha(Path(__file__).parent/name)
                                    for name in ('architectures.py', 'device_graph.py', 'probe_architectures.py')}
    save(out/'architecture_audit.json', audit)
    return audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vtr-root', type=Path, required=True)
    p.add_argument('--binary', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--width', type=int, default=12)
    p.add_argument('--height', type=int, default=12)
    p.add_argument('--channels', type=int, nargs='+', default=[64, 128])
    p.add_argument('--tile-size', type=int, default=4)
    p.add_argument('--jobs', type=int, default=2)
    p.add_argument('--timeout', type=float, default=120)
    args = p.parse_args()
    print(json.dumps(run(args), indent=2))


if __name__ == '__main__':
    main()
