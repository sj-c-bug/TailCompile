"""Run the bounded v7 contract tests and persist the exact verification result."""
from __future__ import annotations

import argparse
import contextlib
import io
from pathlib import Path
import platform
import sys
import time
import unittest

from io_utils import sha, write


def verify(output):
    import torch
    torch.set_num_threads(1)
    here = Path(__file__).resolve().parent
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(str(here), pattern='test_*.py')
    started = time.perf_counter()
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    report = dict(schema='tailcompile-v7-contract-verification-1',
        passed=result.wasSuccessful(), tests_run=result.testsRun,
        failures=len(result.failures), errors=len(result.errors), skipped=len(result.skipped),
        seconds=time.perf_counter()-started, python=sys.version,
        platform=platform.platform(), torch=torch.__version__,
        source_sha256={p.name:sha(p) for p in sorted(here.glob('*.py'))},
        scope='unit/contract tests only; no new P&R or performance acceptance')
    output = Path(output)
    write(output, report)
    output.with_suffix('.log').write_text(stream.getvalue(), encoding='utf-8')
    print({k:report[k] for k in ('passed','tests_run','failures','errors','skipped','seconds')})
    if not report['passed']:
        print(stream.getvalue())
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    raise SystemExit(0 if verify(args.out)['passed'] else 1)
