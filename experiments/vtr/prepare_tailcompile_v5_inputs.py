#!/usr/bin/env python3
"""Prepare untouched Koios design inputs for the 2D v5 confirmation campaign."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from tailcompile_v4_risk import VTR_COMMIT, digest, save


HERE = Path(__file__).resolve().parent
DEFAULT_DESIGNS = ('bnn', 'conv_layer', 'eltwise_layer', 'lenet', 'lstm',
                   'robot_rl', 'tiny_darknet_like')
DEVELOPMENT_DESIGNS = ('sha', 'attention_layer', 'spmv')


def call(command, cwd=None, allowed=(0,)):
    print(json.dumps({'stage': 'command', 'command': [str(x) for x in command]}), flush=True)
    result = subprocess.run([str(x) for x in command], cwd=cwd)
    if result.returncode not in allowed:
        raise RuntimeError(f'command failed with code {result.returncode}: {command}')
    return result.returncode


def normalized_blif_hash(path):
    content = Path(path).read_bytes().replace(b'\r\n', b'\n')
    return hashlib.sha256(content).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--vtr-root', type=Path, default=os.environ.get('VTR_ROOT'))
    p.add_argument('--out', type=Path,
                   default=HERE / 'tailcompile_v5/inputs')
    p.add_argument('--scratch', type=Path,
                   default=os.environ.get('TAILCOMPILE_SCRATCH', HERE / 'vtr_work'))
    p.add_argument('--designs', nargs='+', default=list(DEFAULT_DESIGNS))
    p.add_argument('--flow-width', type=int, default=300)
    p.add_argument('--profile-width', type=int, default=100)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--vtr-python', default=os.environ.get('VTR_PYTHON', sys.executable))
    a = p.parse_args()
    if not a.vtr_root:
        p.error('set VTR_ROOT or pass --vtr-root')
    if set(a.designs) & set(DEVELOPMENT_DESIGNS):
        p.error('v5 holdout inputs must exclude sha, attention_layer and spmv')
    if len(set(a.designs)) != len(a.designs):
        p.error('duplicate designs')
    vtr = a.vtr_root.resolve()
    actual = subprocess.run(['git', '-C', str(vtr), 'rev-parse', 'HEAD'], check=True,
                            capture_output=True, text=True).stdout.strip()
    if actual != VTR_COMMIT:
        raise ValueError(f'VTR commit mismatch: expected {VTR_COMMIT}, actual {actual}')
    bench = vtr / 'vtr_flow/benchmarks/verilog/koios'
    flow = vtr / 'vtr_flow/scripts/run_vtr_flow.py'
    include = bench / 'hard_block_include.v'
    arch = vtr / 'vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml'
    vpr = vtr / 'vpr/vpr'
    for path in (flow, include, arch, vpr):
        if not path.is_file():
            raise FileNotFoundError(path)
    available = {path.stem for path in bench.glob('*.v')}
    missing = sorted(set(a.designs) - available)
    if missing:
        raise ValueError(f'Koios designs not found: {missing}; available={sorted(available)}')
    a.out.mkdir(parents=True, exist_ok=True)
    a.scratch.mkdir(parents=True, exist_ok=True)
    records = []
    for design in a.designs:
        root = a.out / design
        full = root / 'full'
        profile = root / f'rr_w{a.profile_width}'
        ir = root / 'ir'
        full.mkdir(parents=True, exist_ok=True)
        profile.mkdir(parents=True, exist_ok=True)
        source = bench / f'{design}.v'
        blif = full / f'{design}.pre-vpr.blif'
        packed = full / f'{design}.net'
        place = full / f'{design}.place'
        if not all(path.is_file() and path.stat().st_size for path in (blif, packed, place)):
            work = Path(tempfile.mkdtemp(prefix=f'tailcompile-v5-{design}-', dir=a.scratch))
            call([a.vtr_python, flow, source, arch, '-include', include,
                  '-temp_dir', work, '-timeout', a.timeout, '--seed', 1,
                  '--route_chan_width', a.flow_width])
            for name, target in ((f'{design}.pre-vpr.blif', blif),
                                 (f'{design}.net', packed), (f'{design}.place', place)):
                generated = work / name
                if not generated.is_file() or not generated.stat().st_size:
                    raise FileNotFoundError(generated)
                shutil.copy2(generated, target)
        rr_xml = profile / f'{design}.rr_graph.xml'
        rr_gz = profile / f'{design}.rr_graph.xml.gz'
        if not rr_gz.is_file() or not rr_gz.stat().st_size:
            command = [vpr, arch, blif, '--route', '--analysis', '--net_file', packed,
                       '--place_file', place, '--route_file', profile / f'{design}.route',
                       '--seed', 1, '--route_chan_width', a.profile_width,
                       '--write_rr_graph', rr_xml]
            code = call(command, cwd=profile, allowed=(0, 2))
            if not rr_xml.is_file() or not rr_xml.stat().st_size:
                raise RuntimeError(f'VPR code {code} did not produce {rr_xml}')
            with rr_xml.open('rb') as src, gzip.open(rr_gz, 'wb') as dst:
                shutil.copyfileobj(src, dst)
            rr_xml.unlink()
        call([sys.executable, HERE / 'tailcompile_v2.py', '--design', design,
              '--blif', blif, '--packed-net', packed, '--rr-graph', rr_gz,
              '--out', ir, '--region-grid', '4x4'])
        logic = json.loads((ir / 'logic_ir.json').read_text())
        device = json.loads((ir / 'device_ir.json').read_text())
        if logic['design'] != design or logic['packed_net_sha256'] != digest(packed):
            raise ValueError(f'IR provenance mismatch for {design}')
        records.append({
            'design': design,
            'family': design,
            'source_verilog': str(source.resolve()),
            'source_verilog_sha256': digest(source),
            'logic_ir': str((ir / 'logic_ir.json').resolve()),
            'device_ir': str((ir / 'device_ir.json').resolve()),
            'mapped_blif': str(blif.resolve()),
            'mapped_blif_normalized_sha256': normalized_blif_hash(blif),
            'packed_net': str(packed.resolve()),
            'device_width': int(device['grid_width']),
            'screen_widths': [20, 30, 40, 50, 60, 80, 100, 120, 140,
                              160, 180, 200, 240, 280, 320],
        })
    payload = {
        'schema': 'tailcompile-v5-holdout-inputs-v1',
        'vtr_commit': VTR_COMMIT,
        'development_designs_excluded': list(DEVELOPMENT_DESIGNS),
        'profile_width': a.profile_width,
        'designs': records,
    }
    save(a.out / 'holdout_inputs.json', payload)
    print(json.dumps({'stage': 'complete', 'design_count': len(records),
                      'config': str(a.out / 'holdout_inputs.json')}), flush=True)


if __name__ == '__main__':
    main()
