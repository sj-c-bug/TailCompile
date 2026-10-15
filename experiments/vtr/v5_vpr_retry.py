"""Bounded timeout retry without changing the underlying VPR runner signature."""
import json
import os
import shutil
import time
from pathlib import Path

import run_tailcompile_v4 as runner
from tailcompile_v4 import load
from tailcompile_v4_risk import digest, VTR_COMMIT


def run_one(backend, directory, blif, packed_net, width, device_width, seed,
            fplace=None, write_rr=False, timeout=1800, full_legalizer='appack'):
    limit = int(os.environ.get('TAILCOMPILE_VPR_RETRY_TIMEOUT', '7200'))
    if timeout <= 0 or limit < timeout:
        raise ValueError('retry timeout must be at least the positive initial timeout')
    directory = Path(directory).resolve()
    manifest = directory / 'manifest.json'
    args = (backend, directory, blif, packed_net, width, device_width, seed)
    kwargs = dict(fplace=fplace, write_rr=write_rr, full_legalizer=full_legalizer)
    if not manifest.exists():
        try:
            return runner.run_one(*args, timeout=timeout, **kwargs)
        except RuntimeError:
            if not manifest.exists() or load(manifest).get('process_returncode') != 124:
                raise
    old = load(manifest)
    previous = old['signature']['timeout']
    if old['vpr_status'] in ('success', 'unroutable'):
        # The original runner checks all input hashes, parameters and its own
        # implementation hash. Preserve the recorded budget and manifest.
        return runner.run_one(*args, timeout=previous, **kwargs)
    if old.get('process_returncode') != 124:
        return runner.run_one(*args, timeout=previous, **kwargs)
    expected = dict(vtr_commit=VTR_COMMIT, vpr_binary_sha256=backend.binary_hash,
                    architecture_sha256=backend.arch_hash, mapped_blif_sha256=digest(blif),
                    input_packed_net_sha256=digest(packed_net),
                    input_fplace_sha256=digest(fplace) if fplace else None,
                    seed=seed, route_chan_width=width, device_width=device_width,
                    ap_full_legalizer=full_legalizer, write_rr=write_rr,
                    timeout=previous, runner_sha256=digest(Path(runner.__file__)))
    if old['signature'] != expected:
        raise ValueError(f'timeout retry inputs changed: {directory}')
    if previous >= limit:
        raise RuntimeError(f'timeout retry budget exhausted ({previous}s): {directory}')
    archive = directory.parent / '_timeout_history' / (directory.name + '-' + str(time.time_ns()))
    archive.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(directory), str(archive))
    print(json.dumps(dict(stage='timeout_retry', run=str(directory),
                          archive=str(archive), previous_timeout=previous,
                          timeout=limit)), flush=True)
    return runner.run_one(*args, timeout=limit, **kwargs)
