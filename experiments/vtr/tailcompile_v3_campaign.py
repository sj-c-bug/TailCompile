#!/usr/bin/env python3
"""Generate candidate pools and emit a resumable WSL VPR labeling campaign."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_REPO = '/mnt/c/Users/phoenix/Documents/ChatGPT/DAC2027'


def local_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO / path


def portable_argument(path: Path) -> str:
    relative = path.resolve().relative_to(REPO.resolve()).as_posix()
    return f'"${{repo_root}}/{relative}"'


def generate(config: dict) -> None:
    for row in config['designs']:
        output = local_path(row['candidate_output'])
        command = [
            sys.executable, str(HERE / 'tailcompile_v3.py'),
            '--logic-ir', str(local_path(row['logic_ir'])),
            '--device-ir', str(local_path(row['device_ir'])),
            '--blif', str(local_path(row['blif'])),
            '--out', str(output), '--candidates', str(row['candidate_count']),
            '--seed', str(row['candidate_seed']), '--hidden', str(config['model']['hidden']),
            '--partition-prior-weight', str(config['model']['partition_prior_weight']),
        ]
        checkpoint = config['model'].get('checkpoint')
        if checkpoint:
            command += ['--checkpoint', str(local_path(checkpoint)),
                        '--score-noise-std', str(config['model']['score_noise_std']),
                        '--coordinate-noise-std', str(config['model']['coordinate_noise_std'])]
        subprocess.run(command, check=True)


def emit(config: dict, output: Path) -> dict:
    lines = [
        '#!/usr/bin/env bash', 'set -u',
        f'repo_root="${{TAILCOMPILE_REPO:-{DEFAULT_REPO}}}"',
        'export TAILCOMPILE_SHARED="${TAILCOMPILE_SHARED:-${repo_root}/experiments/vtr}"',
        'cd "$repo_root"', '',
    ]
    count = 0
    for row in config['designs']:
        pool = local_path(row['candidate_output'])
        build = json.loads((pool / 'build_summary.json').read_text())
        for candidate in build['candidates']:
            fplace = pool / candidate['fplace']
            for placement_seed in row['placement_seeds']:
                for width in row['channel_widths']:
                    command = [
                        'python3', 'experiments/vtr/run_tailcompile_v0.py',
                        '--variant', 'v3', '--design', row['design'],
                        '--region-grid', '4x4', '--method', 'gnn_candidate',
                        '--fplace', portable_argument(fplace),
                        '--packed-net', portable_argument(local_path(row['packed_net'])),
                        '--seed', str(placement_seed), '--width', str(width),
                        '--legalizer', 'flat-recon', '--detailed-placer', 'annealer',
                        '--device-width', str(row['device_width']),
                        '--timeout', str(config['vpr_timeout_seconds']),
                    ]
                    rendered = [token if token.startswith('"${repo_root}') else shlex.quote(token)
                                for token in command]
                    lines.append(' '.join(rendered) + ' || true')
                    count += 1
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('\n'.join(lines) + '\n', encoding='utf-8', newline='\n')
    return {'script': str(output), 'vpr_job_count': count}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=['generate', 'emit', 'all'], default='all')
    parser.add_argument('--script', type=Path,
                        default=HERE / 'tailcompile_v3_training' / 'run_labels.sh')
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.stage in ('generate', 'all'):
        generate(config)
    result = None
    if args.stage in ('emit', 'all'):
        result = emit(config, args.script)
    print(json.dumps({'stage': args.stage, 'label_campaign': result}, indent=2))


if __name__ == '__main__':
    main()
