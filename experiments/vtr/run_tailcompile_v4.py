#!/usr/bin/env python3
"""Resumable, version-locked 2D supply/build/VPR paired campaign.

Windows can launch a WSL VPR while performing Python candidate construction
on the host. Linux/server runs use the same entry point without --wsl-distro.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

from tailcompile_v4_risk import VTR_COMMIT, build_supply, digest, region_at, save
from tailcompile_v4 import load

HERE = Path(__file__).resolve().parent
DEFAULTS = {'sha': (14, [90, 100]), 'attention_layer': (88, [110, 120]), 'spmv': (84, [90, 100])}


class Backend:
    def __init__(self, root, distro=None):
        if not root:
            raise ValueError('set VTR_ROOT or pass --vtr-root')
        self.root, self.distro = str(root).rstrip('/'), distro
        actual = self.call(['git', '-C', self.root, 'rev-parse', 'HEAD']).stdout.strip()
        if actual != VTR_COMMIT:
            raise ValueError(f'VTR commit mismatch: expected {VTR_COMMIT}, actual {actual}')
        self.binary = self.root + '/vpr/vpr'
        self.arch = self.root + '/vtr_flow/arch/COFFE_22nm/k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml'
        self.binary_hash = self.call(['sha256sum', self.binary]).stdout.split()[0]
        self.arch_hash = self.call(['sha256sum', self.arch]).stdout.split()[0]
        changes = self.call(['git', '-C', self.root, 'diff', '--name-only', '--ignore-submodules', 'HEAD']).stdout.strip()
        if changes:
            raise ValueError('VTR tracked sources differ from pinned commit: ' + changes)

    def path(self, path):
        path = str(path)
        if self.distro and len(path) > 2 and path[1] == ':':
            return '/mnt/' + path[0].lower() + '/' + path[3:].replace('\\', '/')
        return path

    def command(self, args, cwd=None):
        command = [self.path(arg) for arg in args]
        if self.distro:
            command = ['wsl', '-d', self.distro, *(['--cd', self.path(cwd)] if cwd else []), '--', *command]
        return command

    def call(self, args, cwd=None):
        return subprocess.run(self.command(args, cwd), cwd=None if self.distro else cwd,
                              check=True, capture_output=True, text=True, encoding='utf-8', errors='replace')


def run_one(backend, directory, blif, packed_net, width, device_width, seed,
            fplace=None, write_rr=False, timeout=1200):
    directory = Path(directory).resolve()
    signature = {'vtr_commit': VTR_COMMIT, 'vpr_binary_sha256': backend.binary_hash,
                 'architecture_sha256': backend.arch_hash, 'mapped_blif_sha256': digest(blif),
                 'input_packed_net_sha256': digest(packed_net),
                 'input_fplace_sha256': digest(fplace) if fplace else None,
                 'seed': seed, 'route_chan_width': width, 'device_width': device_width,
                 'write_rr': write_rr, 'timeout': timeout,
                 'runner_sha256': digest(Path(__file__))}
    manifest = directory / 'manifest.json'
    if manifest.is_file():
        old = load(manifest)
        if old['signature'] != signature:
            raise ValueError(f'run inputs changed: {directory}; choose a new --out')
        if old['vpr_status'] not in ('success', 'unroutable'):
            raise ValueError(f'previous run failed: {directory}; preserve for diagnosis and choose a new --out')
        return old
    if directory.exists() and any(directory.iterdir()):
        raise ValueError(f'interrupted/incomplete run: {directory}; preserve and choose a new --out')
    directory.mkdir(parents=True, exist_ok=True)
    packed_copy = directory / 'input.net'
    shutil.copy2(packed_net, packed_copy)
    command = [backend.binary, backend.arch, str(blif), '--analytical_place', '--route', '--analysis',
               '--ap_full_legalizer', 'flat-recon', '--ap_detailed_placer', 'annealer',
               '--seed', str(seed), '--route_chan_width', str(width), '--device_width', str(device_width),
               '--net_file', str(packed_copy), '--flat_place_verbosity', '1',
               '--write_flat_place', str(directory / 'post_route.fplace'),
               '--write_legalized_flat_place', str(directory / 'post_legalizer.fplace')]
    if fplace:
        command += ['--read_flat_place', str(fplace)]
    if write_rr:
        command += ['--write_rr_graph', str(directory / 'device.rr_graph.xml')]
    start = time.monotonic()
    with (directory / 'vpr.out').open('w', encoding='utf-8') as log:
        try:
            # Direct log streaming survives interruption and avoids hidden buffered failures.
            # Enforce timeout inside WSL too: killing wsl.exe alone can leave VPR running.
            launched = (['timeout', '--signal=TERM', '--kill-after=15s', str(timeout), *command]
                        if backend.distro else command)
            completed = subprocess.run(backend.command(launched, directory),
                                       cwd=None if backend.distro else directory,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       timeout=timeout + 30 if backend.distro else timeout)
            code = completed.returncode
        except subprocess.TimeoutExpired:
            code = 124
    text = (directory / 'vpr.out').read_text(errors='replace')
    status = ('success' if code == 0 and 'VPR succeeded' in text else
              'unroutable' if code == 2 and 'Circuit is unroutable with a channel width factor' in text
              else 'other_failure')
    result = {'schema': 'tailcompile-v4-run-v1', 'signature': signature,
              'vtr_commit': VTR_COMMIT, 'vpr_status': status, 'process_returncode': code,
              'seed': seed, 'route_chan_width': width, 'command': command,
              'wall_seconds': time.monotonic() - start}
    if digest(packed_net) != signature['input_packed_net_sha256']:
        raise AssertionError('archived input packed net was modified')
    save(manifest, result)
    print(json.dumps({'stage': 'vpr', 'run': str(directory), 'status': status,
                      'seconds': result['wall_seconds']}), flush=True)
    if status == 'other_failure':
        raise RuntimeError(f'VPR failed ({code}), see {directory / "vpr.out"}')
    return result


def retention(path, logic, device, binding):
    if not path.is_file():
        return None
    atoms = {atom: c['id'] for c in logic['clusters'] if c['resource'] != 'io' for atom in c['atoms']}
    seen, matched = set(), 0
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        fields = line.split()
        if len(fields) >= 5 and fields[0] in atoms:
            if fields[0] in seen:
                raise ValueError('duplicate atom in flat placement')
            seen.add(fields[0])
            matched += int(int(fields[3]) == 0 and region_at(device, float(fields[1]), float(fields[2]))
                           == binding[atoms[fields[0]]])
    return {'expected_atoms': len(atoms), 'observed_atoms': len(seen),
            'retained_fraction': matched / len(seen) if seen else None}


def summarize(root):
    from prepare_tailcompile_v3_training import success_metrics
    plan = load(root / 'plan.json')
    rows, missing = [], []
    for spec in plan['runs']:
        directory = root / spec['directory']
        if not (directory / 'manifest.json').is_file():
            missing.append(spec['directory'])
            continue
        meta = load(directory / 'manifest.json')
        manifest_hash = digest(directory / 'manifest.json')
        if (directory / 'metrics.json').is_file():
            cached = load(directory / 'metrics.json')
            if (cached.get('manifest_sha256') == manifest_hash and 'metric_error' not in cached
                    and all(cached.get(k) == v for k, v in spec.items())):
                rows.append(cached)
                continue
        row = {**spec, 'status': meta['vpr_status'], 'seconds': meta['wall_seconds']}
        row['manifest_sha256'] = manifest_hash
        row['total_seconds'] = row['seconds'] + spec.get('preplacement_seconds', 0.0)
        if row['status'] == 'success':
            try:
                row.update(success_metrics(directory, spec['design']))
                if any(row.get(k) is None for k in ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')):
                    raise ValueError('successful run has missing QoR metrics')
            except (FileNotFoundError, ValueError, ZeroDivisionError) as exc:
                row['metric_error'] = str(exc)
        if spec.get('candidate'):
            inputs = root / spec['inputs']
            candidate = root / spec['candidate']
            row['retention'] = retention(directory / 'post_route.fplace', load(inputs / 'logic_ir.json'),
                                          load(inputs / 'device_ir.json'),
                                          load(candidate.with_name(candidate.stem + '_binding.json'))['assignment'])
        save(directory / 'metrics.json', row)
        rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['design'], row['width'], row['method']].append(row)
    metrics = ('region_p99', 'channel_p99', 'wirelength', 'final_cpd_ns')
    aggregates = []
    for (design, width, method), values in sorted(grouped.items()):
        aggregates.append({'design': design, 'width': width, 'method': method, 'trials': len(values),
                           'success': sum(v['status'] == 'success' for v in values),
                           'status_counts': dict(Counter(v['status'] for v in values)),
                           'median_vpr_seconds': statistics.median(v['seconds'] for v in values),
                           'median_total_seconds': statistics.median(v['total_seconds'] for v in values),
                           **{m: statistics.median([v[m] for v in values if v.get(m) is not None])
                              if any(v.get(m) is not None for v in values) else None for m in metrics}})
    pairs = []
    index = {(r['design'], r['width'], r['method'], r['seed']): r for r in rows}
    for row in rows:
        method = row['method']
        if method == 'default' or method.endswith('_base'):
            continue
        base = index.get((row['design'], row['width'], method.split('_')[0] + '_base', row['seed']))
        if base:
            pairs.append({'design': row['design'], 'width': row['width'], 'seed': row['seed'],
                          'method': method, 'before_status': base['status'], 'after_status': row['status'],
                          'joint_success': base['status'] == row['status'] == 'success',
                          'deltas': {m: row[m] - base[m] for m in metrics
                                     if row.get(m) is not None and base.get(m) is not None}})
    payload = {'schema': 'tailcompile-v4-summary-v1', 'vtr_commit': VTR_COMMIT,
               'expected': len(plan['runs']), 'observed': len(rows), 'missing': missing,
               'status_counts': dict(Counter(r['status'] for r in rows)),
               'metric_errors': sum('metric_error' in r for r in rows),
               'aggregates': aggregates, 'paired_changes': pairs,
               'interpretation': '2D development experiment; spmv is not an untouched test set; proxy improvement is not a QoR claim'}
    save(root / 'summary.json', payload)
    if aggregates:
        with (root / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(aggregates[0]))
            writer.writeheader()
            writer.writerows(aggregates)
    print(json.dumps({k: payload[k] for k in ('expected', 'observed', 'status_counts', 'metric_errors')}), flush=True)
    return payload


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', choices=('supply', 'build', 'run', 'summarize', 'all'), default='all')
    p.add_argument('--vtr-root', default=os.environ.get('VTR_ROOT'))
    p.add_argument('--wsl-distro')
    p.add_argument('--input-root', type=Path, default=HERE)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--designs', nargs='+', choices=list(DEFAULTS), default=list(DEFAULTS))
    p.add_argument('--widths', nargs='+', type=int)
    p.add_argument('--seeds', nargs='+', type=int, default=[11, 12])
    p.add_argument('--budget', type=int, default=512)
    p.add_argument('--timeout', type=int, default=1200)
    p.add_argument('--checkpoint', type=Path, default=HERE / 'tailcompile_v3_training/model_width_conditioned/ranker.pt')
    a = p.parse_args()
    if a.budget <= 0 or a.timeout <= 0 or any(s < 0 for s in a.seeds):
        p.error('budget/timeout must be positive and seeds nonnegative')
    if len(set(a.seeds)) != len(a.seeds) or len(set(a.designs)) != len(a.designs):
        p.error('duplicate seeds or designs')
    if a.widths and len(set(a.widths)) != len(a.widths):
        p.error('duplicate widths')
    root = a.out.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if a.stage == 'summarize':
        summarize(root)
        return
    if a.widths and any(w <= 0 or w % 2 for w in a.widths):
        p.error('widths must be positive even integers')
    backend = Backend(a.vtr_root, a.wsl_distro) if a.stage in ('supply', 'run', 'all') else None
    config = {'designs': a.designs, 'widths': a.widths, 'seeds': a.seeds, 'budget': a.budget,
              'checkpoint_sha256': digest(a.checkpoint), 'vtr_commit': VTR_COMMIT}
    if (root / 'config.json').exists() and load(root / 'config.json') != config:
        raise ValueError('campaign config changed; use a new --out')
    save(root / 'config.json', config)
    plan = []
    for design in a.designs:
        bundle = HERE / 'tailcompile_v3_training/bundle/designs' / design
        old = a.input_root.resolve() / 'runs/pilot_v1' / design / 'full-seed1-w300'
        packed = old / f'{design}.net'
        if digest(packed) != load(bundle / 'logic_ir.json')['packed_net_sha256']:
            raise ValueError(f'{design} packed net hash differs from bundled Logic IR')
        inputs = root / design / 'inputs'
        inputs.mkdir(parents=True, exist_ok=True)
        for src, name in ((bundle / 'logic_ir.json', 'logic_ir.json'),
                          (bundle / 'device_ir.json', 'device_ir.json'),
                          (bundle / 'mapped.blif', f'{design}.pre-vpr.blif'), (packed, f'{design}.net')):
            target = inputs / name
            content = src.read_bytes()
            if src.name == 'mapped.blif':
                expected = load(bundle / 'logic_ir.json')['mapped_blif_sha256']
                if hashlib.sha256(content).hexdigest() != expected:
                    content = content.replace(b'\r\n', b'\n')
                    if hashlib.sha256(content).hexdigest() != expected:
                        raise ValueError('BLIF content changed beyond Git newline conversion')
            target_hash = hashlib.sha256(content).hexdigest()
            if target.exists() and digest(target) != target_hash:
                raise ValueError(f'campaign input changed: {target}')
            if not target.exists():
                target.write_bytes(content)
        blif, packed = inputs / f'{design}.pre-vpr.blif', inputs / f'{design}.net'
        for width in a.widths or DEFAULTS[design][1]:
            work = root / design / f'w{width}'
            supply_path = work / 'supply.json'
            if a.stage in ('supply', 'all'):
                provenance = run_one(backend, work / 'supply_run', blif, packed, width,
                                     DEFAULTS[design][0], a.seeds[0], write_rr=True, timeout=a.timeout)
                if not supply_path.exists():
                    supply = build_supply(work / 'supply_run/device.rr_graph.xml', load(inputs / 'device_ir.json'))
                    if supply['channel_width'] != width:
                        raise ValueError('RRG actual width differs from requested width')
                    supply['provenance'] = provenance['signature']
                    save(supply_path, supply)
            candidates = work / 'candidates'
            if a.stage in ('build', 'all'):
                from tailcompile_v4 import build
                # Same frozen checkpoint, one predetermined model seed, no label-based selection.
                gnn_dir = work / 'gnn_initial'
                if not (gnn_dir / 'build_summary.json').exists():
                    import sys
                    init_clock = time.monotonic()
                    subprocess.run([sys.executable, str(HERE / 'tailcompile_v3.py'),
                                    '--logic-ir', str(inputs / 'logic_ir.json'), '--device-ir', str(inputs / 'device_ir.json'),
                                    '--blif', str(blif), '--checkpoint', str(a.checkpoint.resolve()),
                                    '--channel-width', str(width), '--candidates', '1', '--seed', '2027',
                                    '--out', str(gnn_dir)], check=True, stdout=subprocess.DEVNULL)
                    save(gnn_dir / 'v4_initialization.json', {'seconds': time.monotonic() - init_clock,
                                                            'checkpoint_sha256': config['checkpoint_sha256']})
                initial_summary = load(gnn_dir / 'build_summary.json')
                if initial_summary['checkpoint_sha256'] != config['checkpoint_sha256']:
                    raise ValueError('initializer checkpoint changed')
                gnn_lower = gnn_dir / initial_summary['candidates'][0]['lowering']
                from tailcompile_v2 import solve_binding
                from tailcompile_v1 import lower_fplace
                det_dir = work / 'det_initial'
                det_dir.mkdir(parents=True, exist_ok=True)
                det_lower = det_dir / 'tail_rrg_lowering.json'
                if not det_lower.is_file():
                    init_clock = time.monotonic()
                    logic, device = load(inputs / 'logic_ir.json'), load(inputs / 'device_ir.json')
                    binding = solve_binding(logic, device, 'tail_rrg')
                    lowering = lower_fplace(logic, device, binding, det_dir / 'tail_rrg.fplace')
                    save(det_lower, lowering)
                    save(det_dir / 'tail_rrg_binding.json', binding)
                    save(det_dir / 'v4_initialization.json', {'seconds': time.monotonic() - init_clock})
                build(inputs / 'logic_ir.json', inputs / 'device_ir.json', blif, supply_path,
                      det_lower, gnn_lower, candidates, a.budget)
            if a.stage in ('run', 'all'):
                built = load(candidates / 'build_summary.json')
                for row in [{'method': 'default', 'fplace': None}] + built['candidates']:
                    fplace = candidates / row['fplace'] if row['fplace'] else None
                    if fplace and digest(fplace) != row['fplace_sha256']:
                        raise ValueError(f'candidate differs from immutable build summary: {fplace}')
                    for seed in a.seeds:
                        directory = work / 'runs' / f'{row["method"]}_s{seed}'
                        spec = {'design': design, 'width': width, 'method': row['method'], 'seed': seed,
                                'directory': directory.relative_to(root).as_posix(),
                                'inputs': inputs.relative_to(root).as_posix(),
                                'candidate': fplace.relative_to(root).as_posix() if fplace else None}
                        if fplace:
                            init_name = row['method'].split('_')[0]
                            init_seconds = load(work / f'{init_name}_initial/v4_initialization.json')['seconds']
                            spec['preplacement_seconds'] = init_seconds + row['search_seconds'] + row['lowering_seconds']
                            spec['search_evaluations'] = row['evaluations']
                        plan.append(spec)
    # Save the entire intended trial inventory before running any paired trials.
    if a.stage in ('run', 'all'):
        save(root / 'plan.json', {'schema': 'tailcompile-v4-plan-v1', 'config': config, 'runs': plan})
        for spec in plan:
            inputs = root / spec['inputs']
            run_one(backend, root / spec['directory'], inputs / f'{spec["design"]}.pre-vpr.blif',
                    inputs / f'{spec["design"]}.net', spec['width'], DEFAULTS[spec['design']][0], spec['seed'],
                    root / spec['candidate'] if spec['candidate'] else None, timeout=a.timeout)
        result = summarize(root)
        if result['missing'] or result['metric_errors'] or result['status_counts'].get('other_failure'):
            raise RuntimeError('campaign did not pass completeness checks')


if __name__ == '__main__':
    main()
