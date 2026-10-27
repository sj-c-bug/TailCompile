"""Start a new preparation cohort using verified, read-only qualification evidence.

Never retries or deletes an old failed mapping, moves a backend, or builds VTR.
The original qualification files remain the evidence referenced by the copied report.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import install_hook
from io_utils import COMMIT, freeze, read, sha
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def recover(previous, out, backend, *, repo=REPO):
    repo = Path(repo).resolve()
    previous, out, backend = (Path(p).resolve() for p in (previous, out, backend))
    runs = repo/'experiments/vtr/tailcompile_v7_3d/runs'
    for candidate in (previous, out):
        if runs not in candidate.parents:
            raise ValueError('both cohorts must be inside the canonical v7 runs directory')
    if previous == out or previous in out.parents or out in previous.parents:
        raise ValueError('recovery requires a separate sibling cohort; old evidence is preserved')
    validate_output_roots((out,), repo=repo, dependency_roots=(backend,))

    manifest = read(backend/'backend_manifest.json')
    binary = backend/'vpr/vpr'
    digest = sha(binary)
    if manifest.get('vtr_commit') != COMMIT or manifest.get('binary_sha256') != digest:
        raise ValueError('backend manifest does not match the pinned executable')
    os.environ['GIT_OPTIONAL_LOCKS'] = '0'
    if install_hook.verify(backend) != manifest.get('hook_sha256'):
        raise ValueError('read-only backend source/hook does not match the manifest')
    sources = manifest.get('source_sha256', {})
    required = (*install_hook.FILES, *install_hook.HETERO_FILES)
    if any(rel not in sources or sha(backend/rel) != sources[rel] for rel in required):
        raise ValueError('backend source provenance is incomplete or changed')

    qualification = previous/'qualification'
    protocol = read(qualification/'protocol.json')
    report = read(qualification/'qualification_report.json')
    if protocol.get('commit') != COMMIT or protocol.get('binary_sha256') != digest:
        raise ValueError('qualification belongs to another binary/commit')
    if protocol.get('cases') != ['aligned', 'shifted', 'fixed_mixed', 'sha'] or protocol.get('seeds') != [701, 702]:
        raise ValueError('full four-case/two-seed qualification is required')
    if not report.get('complete') or not report.get('passed') or report.get('harness_version') != 2:
        raise ValueError('only complete passed current-harness qualification may be reused')
    expected = {(case, seed) for case in protocol['cases'] for seed in protocol['seeds']}
    conditions = report.get('conditions', [])
    if len(conditions) != 8 or {(r['case'], r['seed']) for r in conditions} != expected or not all(r.get('passed') for r in conditions):
        raise ValueError('qualification evidence lacks all eight passed conditions')
    config = read(previous/'campaign.json')
    if read(previous/'protocol.json').get('config') != config:
        raise ValueError('previous campaign configuration does not match its frozen protocol')

    ledger = dict(schema='tailcompile-v7-prepare-recovery-1', previous_out=str(previous),
        backend_read_only=str(backend), binary_sha256=digest,
        original_files={str(path): sha(path) for path in
                        (previous/'campaign.json', qualification/'protocol.json', qualification/'qualification_report.json')},
        reason='absolute attempt-local mapping paths; new code requires a new cohort',
        original_failed_attempt_preserved=True, qualification_runs_repeated=False)
    if (out/'protocol.json').exists() and not (out/'recovery.json').is_file():
        raise ValueError('target already contains a different campaign; choose a new cohort')
    targets = [out/name for name in ('recovery.json', 'campaign.json',
               'qualification/protocol.json', 'qualification/qualification_report.json')]
    if any(out not in target.resolve().parents for target in targets):
        raise ValueError('recovery target escapes its cohort through a symlink')
    # Everything above this line is read-only. Freeze makes exact recovery resumable.
    freeze(out/'recovery.json', ledger)
    freeze(out/'campaign.json', config)
    freeze(out/'qualification/protocol.json', protocol)
    freeze(out/'qualification/qualification_report.json', report)
    return ledger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-out', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--backend', type=Path, required=True)
    args = parser.parse_args()
    ledger = recover(args.previous_out, args.out, args.backend)
    print(dict(recovery='ready', out=str(args.out.resolve()), binary_sha256=ledger['binary_sha256'],
               backend_read_only=True, repeated_qualification=False), flush=True)


if __name__ == '__main__':
    main()
