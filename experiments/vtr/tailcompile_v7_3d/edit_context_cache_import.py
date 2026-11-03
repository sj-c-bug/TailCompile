"""Read-only reuse of a signed 012 feature cache after scheduling changes.

Only the controller and launcher may differ from the producing cohort. The
feature/model/training implementation, source actions and labels remain fixed.
Torch is imported only after the cheap protocol and file checks succeed.
"""
from __future__ import annotations

from pathlib import Path
import time

try:
    from .io_utils import read, sha
except ImportError:
    from io_utils import read, sha

HERE = Path(__file__).resolve().parent
SCHEDULING_FILES = frozenset(('edit_context_campaign.py', 'run_edit_context_campaign.sh'))
FEATURE_FILES = frozenset(('edit_context_features.py', 'edit_context_model.py', 'train_edit_context.py'))


def _pins(values):
    if not isinstance(values, dict):
        raise ValueError('012 cache import requires source input signatures')
    result = {}
    for path, value in values.items():
        resolved = str(Path(path).resolve())
        if resolved in result and result[resolved] != value:
            raise ValueError('012 conflicting canonical source input signatures')
        result[resolved] = value
    return result


def import_cache(source_info, origin, *, current_implementation):
    """Return prepared-cache metadata pointing to the original bytes, no copy.

    The caller must keep the destination cohort disjoint from ``origin`` and
    write this returned metadata only into its new output. No path is created,
    rewritten, pruned or locked by this function.
    """
    started = time.perf_counter()
    origin = Path(origin).resolve()
    runs = HERE/'runs'
    if origin.parent != runs or not origin.name.startswith('development_012'):
        raise ValueError('012 cache origin must be a canonical development_012 cohort')
    pp, rp = origin/'edit_context_protocol.json', origin/'prepared_report.json'
    if not pp.is_file() or not rp.is_file():
        raise ValueError('012 cache origin has no completed signed preparation')
    protocol, prepared = read(pp), read(rp)
    if Path(protocol['source']).resolve() != Path(source_info['source']).resolve():
        raise ValueError('012 cache origin uses a different source cohort')
    fresh_pins = _pins(source_info['source_inputs'])
    if _pins(protocol['source_inputs']) != fresh_pins:
        raise ValueError('012 cache origin source input signatures differ')
    wanted = source_info['checkpoint_sha256']
    if protocol.get('base_checkpoint_sha256') != wanted:
        raise ValueError('012 cache origin frozen D checkpoint differs')
    old_code = protocol.get('implementation', {})
    if not FEATURE_FILES <= old_code.keys():
        raise ValueError('012 cache origin has no complete feature implementation signature')
    for name, digest in old_code.items():
        if name not in current_implementation or (name not in SCHEDULING_FILES and current_implementation[name] != digest):
            raise ValueError('012 cache feature implementation changed: '+name)
    path = Path(prepared['cache']).resolve()
    # Chained imports are allowed, but their immutable payload must remain in a
    # 012 cohort. This cannot redirect an import to another version or /tmp.
    if runs not in path.parents or not path.relative_to(runs).parts[0].startswith('development_012'):
        raise ValueError('012 reused cache must remain inside a canonical 012 cohort')
    provenance = {str(pp): sha(pp), str(rp): sha(rp), str(path): sha(path)}
    if not prepared.get('complete') or provenance[str(path)] != prepared.get('cache_file_sha256'):
        raise ValueError('012 cache origin preparation/file signature mismatch')

    try:
        from .train_edit_context import load_cache, _digest
        from .train_structured import load_model
        from .train import state_sha256
        from .edit_context_features import edit_feature_dim
    except ImportError:
        from train_edit_context import load_cache, _digest
        from train_structured import load_model
        from train import state_sha256
        from edit_context_features import edit_feature_dim
    cache = load_cache(path)
    if cache['cache_sha256'] != prepared.get('cache_sha256') or cache['source_checkpoint_sha256'] != wanted:
        raise ValueError('012 reused cache content/source checkpoint differs')
    if cache['feedback_sha256'] != _digest(source_info['feedback']):
        raise ValueError('012 reused cache measured feedback differs')
    dependencies = {str(Path(source_info['checkpoint']).resolve())}
    dependencies.update(str(Path(r['graph']).resolve()) for r in source_info['records'])
    dependencies.update(str(Path(r['action']).resolve()) for r in source_info['feedback'])
    cached_pins = _pins(cache['source_inputs'])
    if set(cached_pins) != dependencies or any(fresh_pins.get(p) != h or sha(p) != h for p, h in cached_pins.items()):
        raise ValueError('012 reused cache graph/action/checkpoint signatures differ')
    ids = {r['condition_id'] for r in source_info['records']}
    if {(b['condition_id'], b['arm']) for b in cache['banks']} != {(c, a) for c in ids for a in ('L','Y','J')}:
        raise ValueError('012 reused cache condition/arm coverage differs')
    backbone = load_model(source_info['checkpoint'], device='cpu')
    if sha(source_info['checkpoint']) != wanted or state_sha256(backbone) != cache['backbone_sha256']:
        raise ValueError('012 reused cache frozen D backbone differs')
    hidden = backbone.layer_query.in_features
    if cache['global_dim'] != 4*hidden+32 or cache['local_dim'] != edit_feature_dim(hidden):
        raise ValueError('012 reused cache feature dimensions differ from frozen D')
    if any(sha(p) != h for p, h in provenance.items()):
        raise ValueError('012 cache origin changed during read-only import')
    return dict(complete=True, cache=str(path), cache_file_sha256=provenance[str(path)],
        cache_sha256=cache['cache_sha256'], conditions=len(ids), banks=len(cache['banks']),
        global_dim=cache['global_dim'], local_dim=cache['local_dim'], cache_file_bytes=path.stat().st_size,
        feature_cache_seconds=0., cache_import_seconds=time.perf_counter()-started,
        cache_reused_from=str(origin), cache_reuse_provenance=provenance,
        source_actions_unchanged=True, new_pnr=0, original_cache_read_only=True)
