"""Read-only, frozen six-family inputs for the 015 mechanism experiment.

Historical native consumers establish that each architecture/RRG/netlist pair
was usable with the shared layer backend. RRG producer hashes can predate that
backend; neither old manifests nor payloads are rewritten by this loader.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
ARCHES = ('H0Z0', 'H0Z1', 'H1Z0', 'H1Z1')
SCHEMA = 'tailcompile-attribution-panel-015-1'
_SERVER = re.compile(r'^/inspire/hdd/(?:global_user/[^/]+|project/project-public/[^/]+/[^/]+)/workspace/DAC2027/TailCompile(?:/|$)')


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def remap_path(value, repo=REPO):
    """Resolve only current-repo paths or registered historical server aliases."""
    repo = Path(repo).resolve()
    value = str(value).replace('\\', '/')
    server = _SERVER.match(value)
    if server:
        relative = value[server.end():]
        path = repo / relative
    elif not Path(value).is_absolute():
        path = repo / value
    else:
        path = Path(value)
    if '..' in Path(value).parts:
        raise ValueError('015 archived path contains parent traversal: ' + value)
    path = path.resolve()
    if path != repo and repo not in path.parents:
        raise ValueError('015 archived input escapes the canonical repository: ' + value)
    return path


def _one_option(command, name):
    if command.count(name) != 1:
        raise ValueError('015 consumer option missing or repeated: ' + name)
    index = command.index(name) + 1
    if index >= len(command):
        raise ValueError('015 consumer option lacks its value: ' + name)
    return command[index]


def _relative(path, repo):
    return Path(path).resolve().relative_to(repo).as_posix()


def _verify(path, expected, cache):
    if not _digest(expected):
        raise ValueError('015 malformed archived payload digest: ' + str(path))
    path = Path(path)
    if path not in cache:
        cache[path] = _sha(path)
    if cache[path] != expected:
        raise ValueError('015 frozen input bytes differ: ' + str(path))


def _resource_group(record):
    hard = record.get('hard_ap_counts', {})
    utilization = record.get('resource_upper_utilization', [])
    if (sum(hard.get(k, 0) for k in ('memory', 'dsp_top')) >= 16
            and len(utilization) == 5 and max(utilization[2:4]) >= .1):
        return 'hard_resource_heavy'
    return 'other'


def load_panel(manifest_path=HERE/'attribution_panel_015.json', *, repo=REPO,
               verify_payload=False, minimum_families=None):
    """Return remapped input conditions; verify_payload hashes large server files.

    Every source JSON is checked even in the lightweight mode. The optional
    minimum_families override only supports small engineering fixtures; it
    cannot change which designs are selected from a registered manifest.
    """
    repo = Path(repo).resolve()
    manifest_path = remap_path(manifest_path, repo)
    manifest = _read(manifest_path)
    if manifest.get('schema') != SCHEMA:
        raise ValueError('015 unrecognized frozen panel schema')
    entries = manifest.get('designs', [])
    minimum = manifest['minimum_families'] if minimum_families is None else minimum_families
    if (type(minimum) is not int or minimum < 1 or len(entries) < minimum
            or len({r['family'] for r in entries}) != len(entries)
            or len({r['design'] for r in entries}) != len(entries)):
        raise ValueError('015 needs distinct registered designs and independent families')
    binary_sha = manifest['backend_binary_sha256']
    if not _digest(binary_sha):
        raise ValueError('015 malformed qualified backend hash')
    pins = manifest.get('source_files', {})
    if not pins:
        raise ValueError('015 panel requires frozen source evidence')
    hashes = {}
    for relative, expected in pins.items():
        _verify(remap_path(relative, repo), expected, hashes)
    documents = {}
    def document(relative):
        path = remap_path(relative, repo)
        if _relative(path, repo) not in pins:
            raise ValueError('015 unpinned source evidence: ' + str(path))
        if path not in documents:
            documents[path] = _read(path)
        return documents[path]
    designs, conditions, payloads = [], [], {}
    circuit_hashes = set()
    for entry in entries:
        design, family = entry['design'], entry['family']
        if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', s) for s in (design, family)):
            raise ValueError('015 design/family must be safe path components')
        source_conditions = document(entry['conditions'])
        available = source_conditions.get('records', source_conditions.get('conditions', []))
        selected = [r for r in available if r['design'] == design and r['family'] == family
                    and r['width_role'] == 'reference' and r['width'] == entry['reference_width']]
        if len(selected) != 4 or {r['architecture'] for r in selected} != set(ARCHES):
            raise ValueError('015 missing or duplicate registered reference condition: ' + design)
        native_rows = document(entry['native_report'])['rows']
        native_identity = document(entry['binary_identity'])
        if native_identity.get('sha256') != binary_sha:
            raise ValueError('015 historical consumer uses another qualified backend')
        base_record = deepcopy(selected[0]['record'])
        if any(r['record'] != base_record for r in selected):
            raise ValueError('015 design input changes between architectures: ' + design)
        if base_record['design'] != design or base_record['family'] != family or not base_record.get('prepared'):
            raise ValueError('015 selected condition lacks its prepared design identity')
        circuit_sha = base_record.get('mapped_blif_sha256')
        if not _digest(circuit_sha) or circuit_sha in circuit_hashes:
            raise ValueError('015 duplicated or invalid mapped circuit across design families')
        circuit_hashes.add(circuit_sha)
        if not {'.blocks.tsv', '.nets.tsv', '.meta.tsv'}.issubset(base_record.get('snapshot_sha256', {})):
            raise ValueError('015 prepared input lacks complete AP structure hashes')
        base_record['blif'] = str(remap_path(base_record['blif'], repo))
        base_record['structure_prefix'] = str(remap_path(base_record['structure_prefix'], repo))
        group = _resource_group(base_record)
        base_record.update(exposure=entry['exposure'], resource_group=group,
                           registered_lineage=entry['lineage'])
        designs.append(base_record)
        for selected_condition in sorted(selected, key=lambda r: r['architecture']):
            cid = selected_condition['id']
            expected_cid = f'{design}/{selected_condition["architecture"]}/w{entry["reference_width"]}'
            if cid != expected_cid:
                raise ValueError('015 noncanonical registered condition identifier')
            candidates = [r for r in native_rows if r['condition_id'] == cid
                          and r['method'] == entry['default_method'] and r['seed'] == entry['default_seed']]
            if len(candidates) != 1 or candidates[0]['status'] != 'success':
                raise ValueError('015 reference lacks its predeclared successful Default consumer: ' + cid)
            consumer_path = remap_path(candidates[0]['vpr_manifest'], repo)
            native = document(consumer_path)
            if native.get('status') != 'success' or native.get('returncode') != 0:
                raise ValueError('015 source Default native execution was not successful')
            command = native['signature']['command']
            if (len(command) < 3 or not {'--analytical_place', '--route', '--analysis'}.issubset(command)
                    or int(_one_option(command, '--route_chan_width')) != entry['reference_width']
                    or int(_one_option(command, '--seed')) != entry['default_seed']
                    or _one_option(command, '--ap_full_legalizer') != 'appack'
                    or _one_option(command, '--ap_detailed_placer') != 'annealer'
                    or native['signature']['env'].get('TC_GP_LAYER_PRIOR') != '0'
                    or 'TC_GP_INIT' in native['signature']['env']):
                raise ValueError('015 source consumer does not establish the frozen cold backend')
            arch, blif = (remap_path(v, repo) for v in command[1:3])
            binary = remap_path(command[0], repo)
            if binary != remap_path(native_identity.get('path', native_identity.get('binary')), repo):
                raise ValueError('015 consumer command binary differs from its pinned identity')
            if str(blif) != base_record['blif']:
                raise ValueError('015 consumer BLIF differs from prepared input')
            rr = remap_path(_one_option(command, '--read_rr_graph'), repo)
            rr_manifest_path = rr.parent/'manifest.json'
            rr_manifest = document(rr_manifest_path)
            native_pins = {str(remap_path(k, repo)): v for k, v in native['signature']['inputs'].items()}
            for pinned in (arch, blif, rr_manifest_path, remap_path(entry['binary_identity'], repo)):
                if str(pinned) not in native_pins:
                    raise ValueError('015 consumer input lacks hash evidence: ' + str(pinned))
            if native_pins[str(rr_manifest_path)] != _sha(rr_manifest_path):
                raise ValueError('015 consumer RRG manifest bytes changed')
            if native_pins[str(remap_path(entry['binary_identity'], repo))] != _sha(remap_path(entry['binary_identity'], repo)):
                raise ValueError('015 consumer backend identity evidence changed')
            if (rr_manifest['architecture_sha256'] != native_pins[str(arch)]
                    or rr_manifest['channel_width'] != entry['reference_width']
                    or native_pins[str(blif)] != base_record['mapped_blif_sha256']):
                raise ValueError('015 RRG architecture/width or prepared BLIF provenance differs')
            fields = dict(arch=str(arch), blif=str(blif), device=str(rr.parent/'device_graph.json'),
                          rr_graph=str(rr), device_manifest=str(rr_manifest_path))
            source_output = remap_path(entry['source_output'], repo)
            if entry['vertical_layout'] == '007':
                summary = source_output/'vertical_devices'/rr.parent.name/'summary.json'
            elif entry['vertical_layout'] == '014':
                summary = source_output/'predictions'/cid/'vertical_summary.json'
            else:
                raise ValueError('015 unsupported vertical-summary archive layout')
            fields['vertical_summary'] = str(summary)
            payloads.update({str(binary): binary_sha, str(arch): native_pins[str(arch)],
                             str(blif): base_record['mapped_blif_sha256'],
                             str(rr): rr_manifest['rr_graph_sha256'],
                             fields['device']: rr_manifest['graph_sha256']})
            for suffix, expected in base_record['snapshot_sha256'].items():
                if suffix not in ('.blocks.tsv', '.nets.tsv', '.meta.tsv', '.pins.tsv'):
                    raise ValueError('015 unexpected AP snapshot suffix')
                payloads[base_record['structure_prefix']+suffix] = expected
            condition = deepcopy(selected_condition)
            condition.update(record=deepcopy(base_record), fields=fields,
                consumer_manifest=str(consumer_path), source_output=str(source_output),
                input_source=entry['conditions'], exposure=entry['exposure'], resource_group=group,
                consumer_binary_sha256=binary_sha,
                rrg_producer_binary_sha256=rr_manifest['binary_sha256'],
                source_files={_relative(consumer_path, repo): pins[_relative(consumer_path, repo)],
                              _relative(rr_manifest_path, repo): pins[_relative(rr_manifest_path, repo)]})
            conditions.append(condition)
    if sum(d['resource_group'] == 'hard_resource_heavy' for d in designs) < manifest['minimum_heavy_families']:
        raise ValueError('015 registered panel lacks its two hard-resource-heavy families')
    if verify_payload:
        for path, expected in payloads.items():
            _verify(path, expected, hashes)
    return dict(schema=SCHEMA, designs=designs, conditions=conditions,
                design_count=len(designs), family_count=len(designs),
                conditions_count=len(conditions), backend_binary_sha256=binary_sha,
                sources=dict(pins), manifest_sha256=_sha(manifest_path),
                payload_files=payloads, payload_verified=bool(verify_payload),
                resource_group_counts={group: sum(d['resource_group'] == group for d in designs)
                                       for group in ('hard_resource_heavy', 'other')},
                final_blind_test=False, teacher_recollection=False,
                family_exposure={d['family']: d['exposure'] for d in designs},
                excluded_designs=manifest.get('excluded_designs', []))
