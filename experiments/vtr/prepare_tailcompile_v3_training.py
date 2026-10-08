#!/usr/bin/env python3
"""Build a portable TailCompile v3 structured-ranking dataset bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from confirmatory_skew import region_metrics
from extract_congestion import channels, percentile


HERE = Path(__file__).resolve().parent


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field(pattern: str, text: str, cast=float):
    match = re.search(pattern, text)
    return cast(match.group(1)) if match else None


def success_metrics(archive: Path, design: str) -> dict:
    log = (archive / 'vpr.out').read_text(errors='replace')
    congestion = channels(archive / 'chanx_occupancy.txt', 'x')
    congestion += channels(archive / 'chany_occupancy.txt', 'y')
    ratios = [occupancy / capacity for _, _, _, occupancy, capacity, _ in congestion]
    region_p99, region_max, _ = region_metrics(archive, f'{design}.pre-vpr')
    return {
        'channel_p99': percentile(ratios, 0.99),
        'channel_max': max(ratios),
        'region_p99': region_p99,
        'region_max': region_max,
        'wirelength': field(r'Total wirelength:\s*(\d+)', log, int),
        'final_cpd_ns': field(
            r'Final critical path delay \(least slack\):\s*([0-9.]+) ns', log),
    }


def candidate_index(root: Path) -> dict[str, dict]:
    found = {}
    for summary_path in sorted(root.rglob('build_summary.json')):
        if 'runs' in summary_path.parts:
            continue
        summary = json.loads(summary_path.read_text())
        for row in summary.get('candidates', []):
            candidate = dict(row)
            candidate['root'] = summary_path.parent
            candidate['build_summary'] = summary_path
            previous = found.get(row['fplace_sha256'])
            if previous and previous['root'] != candidate['root']:
                raise ValueError(f'duplicate fplace hash in two pools: {row["fplace_sha256"]}')
            found[row['fplace_sha256']] = candidate
    return found


def copy_source(source: Path, destination: Path, bundle_root: Path) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return {'path': destination.relative_to(bundle_root).as_posix(),
            'sha256': digest(destination)}


def build_design(design: str, output: Path) -> dict:
    v2_root = HERE / 'tailcompile_v2' / design / '4x4'
    v3_root = HERE / 'tailcompile_v3' / design / '4x4'
    logic_path, device_path = v2_root / 'logic_ir.json', v2_root / 'device_ir.json'
    design_out = output / 'designs' / design
    if logic_path.is_file() and device_path.is_file():
        logic = json.loads(logic_path.read_text())
        blif_path = Path(logic['mapped_blif'])
        if not blif_path.is_absolute():
            blif_path = HERE.parent.parent / blif_path
    else:
        # A cloned portable bundle already contains the immutable graph inputs,
        # even when the larger v2 construction directory was intentionally not
        # committed.  Reuse those sources while rebuilding trial labels.
        logic_path = design_out / 'logic_ir.json'
        device_path = design_out / 'device_ir.json'
        blif_path = design_out / 'mapped.blif'
        for source in (logic_path, device_path, blif_path):
            if not source.is_file():
                raise FileNotFoundError(
                    f'missing v2 source and portable bundle fallback: {source}')
        logic = json.loads(logic_path.read_text())
    candidates = candidate_index(v3_root)
    sources = {
        'logic_ir': copy_source(logic_path, design_out / 'logic_ir.json', output),
        'device_ir': copy_source(device_path, design_out / 'device_ir.json', output),
        'mapped_blif': copy_source(blif_path, design_out / 'mapped.blif', output),
    }
    actions = {}
    for fplace_hash, candidate in candidates.items():
        binding = json.loads((candidate['root'] / candidate['binding']).read_text())
        lowering = json.loads((candidate['root'] / candidate['lowering']).read_text())
        action = {
            'id': fplace_hash,
            'generator_seed': candidate['seed'],
            'assignment': binding['assignment'],
            'relative_xy': [row['relative_xy'] for row in lowering['clusters']],
            'source_pool': candidate['root'].relative_to(v3_root).as_posix() or '.',
            'trials': [],
        }
        actions[fplace_hash] = action
    expected_packed_hash = logic['packed_net_sha256']
    rejected = []
    for manifest_path in sorted((v3_root / 'runs').glob('*/manifest.json')):
        meta = json.loads(manifest_path.read_text())
        action = actions.get(meta.get('input_fplace_sha256'))
        if action is None:
            continue
        if meta.get('input_packed_net_sha256') != expected_packed_hash:
            rejected.append({'manifest': str(manifest_path), 'reason': 'packed_net_hash_mismatch'})
            continue
        status = meta['vpr_status']
        trial = {
            'placement_seed': meta['seed'],
            'channel_width': meta['route_chan_width'],
            'status': status,
            'wall_seconds': meta['wall_seconds'],
            'metrics': None,
        }
        if status == 'success':
            trial['metrics'] = success_metrics(manifest_path.parent, design)
        action['trials'].append(trial)
    usable = [row for row in actions.values() if row['trials']]
    return {
        'design': design,
        'sources': sources,
        'cluster_count': len(logic['clusters']),
        'action_count': len(usable),
        'trial_count': sum(len(row['trials']) for row in usable),
        'rejected_trial_count': len(rejected),
        'rejected_trials': rejected,
        'actions': usable,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--designs', nargs='+', default=['sha', 'attention_layer'])
    parser.add_argument('--out', type=Path,
                        default=HERE / 'tailcompile_v3_training' / 'bundle')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    payload = {
        'schema': 'tailcompile-v3-structured-ranking-dataset-v2',
        'metric_direction': {
            'route_failure': 'lower', 'channel_p99': 'lower', 'region_p99': 'lower',
            'wirelength': 'lower', 'final_cpd_ns': 'lower',
        },
        'pairing_rule': ('same design and channel_width; aggregate placement seeds before '
                         'constructing candidate pairs'),
        'placement_seed_policy': {
            'input_feature': False,
            'role': 'nuisance repetition',
            'route_failure': 'failure rate across seeds',
            'successful_metrics': 'median across successful seeds',
        },
        'channel_width_policy': {
            'input_feature': True,
            'normalization': 'log1p(width)/log1p(300)',
        },
        'designs': [build_design(design, args.out) for design in args.designs],
    }
    target = args.out / 'dataset.json'
    target.write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({
        'dataset': str(target),
        'design_count': len(payload['designs']),
        'action_count': sum(row['action_count'] for row in payload['designs']),
        'trial_count': sum(row['trial_count'] for row in payload['designs']),
        'rejected_trial_count': sum(row['rejected_trial_count'] for row in payload['designs']),
    }, indent=2))


if __name__ == '__main__':
    main()
