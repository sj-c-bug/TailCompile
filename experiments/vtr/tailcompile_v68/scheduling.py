"""Execution budgets, separate from frozen scientific inputs/model settings."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from common import sha

DEFAULT_JOBS = 16
HERE = Path(__file__).resolve().parent


def resolve_concurrency(args):
    """Unset per-stage budgets inherit --jobs, including explicit values >16."""
    if args.prepare_jobs is None:
        args.prepare_jobs = args.jobs
    if args.rr_jobs is None:
        args.rr_jobs = args.jobs
    if min(args.jobs, args.prepare_jobs, args.rr_jobs) < 1:
        raise ValueError('All process concurrency budgets must be positive')
    return args


def compatible_protocol(previous, expected):
    """Accept ONLY the audited scheduler-only update from the original v6.8.

    Exact old/new code hashes live in a small, reviewable manifest. Every other
    protocol field must remain identical. The frozen old protocol itself is
    not rewritten, so already-collected tensors/teachers retain their source
    identity. Unknown code, data, VPR, seeds or objective changes still fail.
    """
    if previous == expected:
        return expected, False
    manifest = json.loads((HERE/'scheduling_compatibility.json').read_text())
    if (previous.get('source', {}).get('experiment.py') != manifest['legacy_experiment_sha256'] or
        expected.get('source', {}).get('experiment.py') != manifest['updated_experiment_sha256'] or
        sha(Path(__file__)) != manifest['scheduling_module_sha256']):
        return expected, False
    candidate = copy.deepcopy(expected)
    candidate['source']['experiment.py'] = manifest['legacy_experiment_sha256']
    candidate['source'].pop('scheduling.py', None)
    if candidate != previous:
        return expected, False
    return candidate, True
