#!/usr/bin/env python3
"""Run one paired analytical-placement TailCompile v0 experiment under WSL."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

VTR = Path(os.environ.get('VTR_ROOT', '/home/phoenix/tools/vtr-e422b088'))
ARCH = VTR / 'vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml'
SHARED = Path(os.environ.get(
    'TAILCOMPILE_SHARED', '/mnt/c/Users/phoenix/Documents/ChatGPT/DAC2027/experiments/vtr'))
COMMIT = os.environ.get('VTR_COMMIT', 'e422b08861dfc8500874f04105ba2a7eb2f11ccd')


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=['v0', 'v1', 'v2', 'v3'], default='v0')
    parser.add_argument('--design', default='sha')
    parser.add_argument('--region-grid', choices=['2x2', '2x4', '4x4'], default='2x2')
    parser.add_argument('--method', choices=['default_ap', 'rrg_only', 'mean_rrg', 'tail_rrg',
                                             'gnn_candidate'], required=True)
    parser.add_argument('--fplace', type=Path,
                        help='Explicit fplace input, required for a v3 gnn_candidate run')
    parser.add_argument('--packed-net', type=Path,
                        help='Reuse this packed netlist instead of allowing VPR to repack')
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--width', type=int, required=True)
    parser.add_argument('--legalizer', choices=['appack', 'flat-recon'], default='appack')
    parser.add_argument('--detailed-placer', choices=['annealer', 'none', 'windowed_bi_matching'], default='annealer')
    parser.add_argument('--device-size', help='Freeze VPR device grid, for example 88x88')
    parser.add_argument('--device-width', type=int,
                        help='Freeze auto-layout width; square layouts obtain the same height')
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--skip-existing', action='store_true',
                        help='Return success when the exact run archive already exists')
    args = parser.parse_args()
    if args.fplace is not None:
        args.fplace = args.fplace.resolve()
    if args.packed_net is not None:
        args.packed_net = args.packed_net.resolve()
    if args.variant == 'v0' and args.method == 'rrg_only':
        parser.error('rrg_only is only available in v1')
    if args.variant in ('v0', 'v1') and args.design != 'sha':
        parser.error('v0/v1 archived runners only support sha; use v2 for other designs')
    if args.variant == 'v3' and (args.method != 'gnn_candidate' or args.fplace is None
                                 or args.packed_net is None):
        parser.error('v3 requires --method gnn_candidate, --fplace, and --packed-net')
    inputs = (SHARED / f'tailcompile_{args.variant}' / args.design
              / args.region_grid if args.variant in ('v2', 'v3')
              else SHARED / f'tailcompile_{args.variant}/sha')
    archive_root = inputs / 'runs'
    scratch_base = Path(os.environ.get('TAILCOMPILE_SCRATCH', '/home/phoenix/vtr_work'))
    scratch_root = (scratch_base / f'tailcompile_{args.variant}'
                    / args.design / (args.region_grid if args.variant in ('v2', 'v3') else ''))
    blif = SHARED / 'runs/pilot_v1' / args.design / 'full-seed1-w300' / f'{args.design}.pre-vpr.blif'
    if args.seed < 0 or (args.width != -1 and (args.width <= 0 or args.width % 2)):
        parser.error('seed must be nonnegative; width must be -1 or a positive even number')
    fplace = (args.fplace if args.fplace is not None else
              None if args.method == 'default_ap' else inputs / f'{args.method}.fplace')
    for path in (ARCH, blif, *((fplace,) if fplace else ()),
                 *((args.packed_net,) if args.packed_net else ())):
        if not path.is_file():
            parser.error(f'missing input: {path}')
    input_tag = '' if fplace is None else f'-fp{digest(fplace)[:8]}'
    legalizer_tag = 'appack' if args.legalizer == 'appack' else 'flat-recon'
    key = (f'{args.method}{input_tag}-{legalizer_tag}-{args.detailed_placer}'
           f'-seed{args.seed}-w{"min" if args.width == -1 else args.width}')
    if args.device_size:
        key += f'-dev{args.device_size}'
    if args.device_width:
        key += f'-devw{args.device_width}'
    scratch = scratch_root / key
    archive = archive_root / key
    if scratch.exists() or archive.exists():
        if args.skip_existing:
            print(f'{key}: status=skipped_existing archive={archive}')
            return 0
        parser.error(f'run already exists: {scratch} or {archive}')
    scratch.mkdir(parents=True)
    archive.mkdir(parents=True)
    packed_net_work = None
    packed_net_input_sha256 = None
    if args.packed_net is not None:
        packed_net_input_sha256 = digest(args.packed_net)
        packed_net_work = scratch / f'{args.design}.locked-input.net'
        shutil.copy2(args.packed_net, packed_net_work)
    command = [str(VTR / 'vpr/vpr'), str(ARCH), str(blif),
               '--analytical_place', '--route', '--analysis',
               '--ap_full_legalizer', args.legalizer,
               '--ap_detailed_placer', args.detailed_placer,
               '--seed', str(args.seed), '--route_chan_width', str(args.width),
               '--write_flat_place', str(scratch / 'post_route.fplace'),
               '--write_legalized_flat_place', str(scratch / 'post_legalizer.fplace'),
               '--flat_place_verbosity', '1']
    if args.device_size:
        command += ['--device', args.device_size]
    if args.device_width:
        command += ['--device_width', str(args.device_width)]
    if packed_net_work is not None:
        command += ['--net_file', str(packed_net_work)]
    if fplace is not None:
        command += ['--read_flat_place', str(fplace)]
    started = datetime.now(timezone.utc).isoformat()
    tick = time.monotonic()
    try:
        result = subprocess.run(command, cwd=scratch, capture_output=True, text=True,
                                timeout=args.timeout)
        code, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as exc:
        code = 124
        stdout = (exc.stdout or b'').decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        stderr = (exc.stderr or b'').decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
    elapsed = round(time.monotonic() - tick, 3)
    (scratch / 'vpr.out').write_text(stdout + '\n' + stderr)
    status = ('success' if code == 0 and 'VPR succeeded' in stdout else
              'unroutable' if 'Circuit is unroutable with a channel width factor' in stdout + stderr else
              'other_failure')
    for item in scratch.iterdir():
        if item.is_file():
            shutil.copy2(item, archive / item.name)
    shutil.copy2(blif, archive / blif.name)
    shutil.copy2(ARCH, archive / ARCH.name)
    if fplace is not None:
        shutil.copy2(fplace, archive / fplace.name)
    if args.packed_net is not None:
        shutil.copy2(args.packed_net, archive / f'{args.design}.input-packed.net')
    metadata = {
        'schema': f'tailcompile-{args.variant}-run', 'variant': args.variant,
        'design': args.design, 'region_grid': args.region_grid, 'method': args.method,
        'mapped_blif': str(blif), 'mapped_blif_sha256': digest(blif),
        'architecture': str(ARCH), 'architecture_sha256': digest(ARCH),
        'input_fplace': str(fplace) if fplace else None,
        'input_fplace_sha256': digest(fplace) if fplace else None,
        'input_packed_net': str(args.packed_net) if args.packed_net else None,
        'input_packed_net_sha256': packed_net_input_sha256,
        'scratch_packed_net': str(packed_net_work) if packed_net_work else None,
        'vtr_commit': COMMIT, 'seed': args.seed, 'route_chan_width': args.width,
        'ap_full_legalizer': args.legalizer, 'ap_detailed_placer': args.detailed_placer,
        'device_size': args.device_size,
        'device_width': args.device_width,
        'started_utc': started, 'wall_seconds': elapsed,
        'process_returncode': code, 'vpr_status': status,
        'command': command, 'scratch': str(scratch), 'archive': str(archive),
    }
    (archive / 'manifest.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'{key}: status={status} code={code} elapsed={elapsed}s archive={archive}', flush=True)
    return 0 if status in ('success', 'unroutable') else 1


if __name__ == '__main__':
    raise SystemExit(main())
