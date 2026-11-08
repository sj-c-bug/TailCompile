"""Bounded, device-derived SB routing-delay template for experiment 017.

This is a connection delay proxy in ns, not STA, path delay, or CPD.  Horizontal
cost amortizes resolved RR switch delay and a one-load Elmore wire estimate over
sampled CHANX/CHANY spans.  Cross-layer connections use the caller's portal
detour, then the directional vertical coefficient.  No RR edges are traversed.
Only the audited two-layer Hecate SB adapter is supported; CB pin crossings and
arbitrary scatter/gather architectures deliberately fail qualification.
"""
from __future__ import annotations

from collections import defaultdict
import gzip
import hashlib
import json
import math
from pathlib import Path
import statistics
import time
import xml.etree.ElementTree as ET


SCHEMA = 'tailcompile-017-device-delay-proxy-1'
PINNED_VTR_COMMIT = '40f8d19708b0b60e2aa04452deb1cfed68d96b0b'
MAX_SAMPLE_NODES = 2_000_000
MAX_SAMPLE_BYTES = 512 * 1024 * 1024  # decompressed XML, independently of hashing
MAX_ARCH_BYTES = 16 * 1024 * 1024
MAX_SWITCHES = 4096
SAMPLES_PER_STRATUM = 32
# Once this many prefix nodes have been visited, missing strata may be mirrored
# only under the explicit globally shared, isotropic-segment qualification.
EARLY_SAMPLE_NODES = 100_000
SEGMENTS = ('L4', 'L16')


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _check_time(deadline):
    if time.perf_counter() >= deadline:
        raise TimeoutError('017 timing extraction exceeded max_seconds; stage timing preparation separately with a larger budget')


def _file_sha(path, deadline):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            _check_time(deadline)
            block = stream.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _load(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text(encoding='utf-8'))
    if not isinstance(value, dict):
        raise ValueError('017 device and device_manifest must be JSON dictionaries or file paths')
    return value


def _finite(value, label, positive=False):
    if isinstance(value, bool):
        raise ValueError(f'017 timing invalid {label}')
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f'017 timing missing/invalid {label}') from error
    if not math.isfinite(number) or number < 0 or positive and number == 0:
        raise ValueError(f'017 timing invalid {label}: finite {"positive" if positive else "nonnegative"} value required')
    return number


def _sha_string(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _architecture(path, deadline):
    if path.stat().st_size > MAX_ARCH_BYTES:
        raise ValueError('017 timing architecture exceeds metadata size cap')
    root = ET.parse(path).getroot()
    _check_time(deadline)
    patterns = root.findall('./scatter_gather_list/sg_pattern')
    if len(patterns) != 1 or patterns[0].get('type') != 'bidir':
        raise ValueError('017 timing supports the audited bidirectional Hecate SB pattern only; CB adapter unsupported')
    directions = {}
    z_segments = set()
    for link in patterns[0].findall('./sg_link_list/sg_link'):
        offset = link.get('z_offset')
        if offset not in ('1', '-1') or offset in directions:
            raise ValueError('017 timing requires exactly one adjacent-layer SB link per direction')
        directions[offset] = link.get('mux')
        z_segments.add(link.get('seg_type'))
    if set(directions) != {'1', '-1'} or None in z_segments:
        raise ValueError('017 timing architecture lacks directional SB mux declarations')
    # A CB architecture permits pins to connect directly to Z.  This model has
    # no such topology, even when it happens to contain an SG pattern as well.
    subtiles = root.findall('./tiles/tile/sub_tile')
    if not subtiles:
        raise ValueError('017 timing cannot qualify SB topology without tile pin fc declarations')
    for sub in subtiles:
        fc = sub.find('fc')
        if fc is None:
            raise ValueError('017 timing cannot qualify missing tile pin fc declarations')
        inactive = (fc.get('in_type') == fc.get('out_type') == 'abs' and
                    _finite(fc.get('in_val'), 'fc in_val') == _finite(fc.get('out_val'), 'fc out_val') == 0)
        for segment in z_segments:
            overrides = [x for x in fc.findall('fc_override') if x.get('segment_name') == segment]
            if not inactive and (not overrides or any(_finite(x.get('fc_val'), 'Z fc override') != 0 for x in overrides)):
                raise ValueError('017 timing CB/direct Z pin access unsupported; require explicit zero Z fc overrides')
    segments = {}
    for name in SEGMENTS:
        segment = root.find(f"./segmentlist/segment[@name='{name}']")
        mux = None if segment is None else segment.find('mux')
        if segment is None or mux is None or not mux.get('name'):
            raise ValueError(f'017 timing requires named {name} segment and mux')
        segments[name] = dict(length=int(segment.get('length')), mux=mux.get('name'),
                              frequency=_finite(segment.get('freq'), f'{name} frequency', positive=True))
        if segments[name]['length'] <= 0:
            raise ValueError('017 timing segment length must be positive')
    cb = root.find('./device/connection_block')
    if cb is None or not cb.get('input_switch_name'):
        raise ValueError('017 timing requires local ipin connection-block switch declaration')
    zero_z_metal = True
    for name in z_segments:
        segment = root.find(f"./segmentlist/segment[@name='{name}']")
        zero_z_metal = zero_z_metal and segment is not None and all(
            segment.get(key) is not None and _finite(segment.get(key), 'Z '+key) == 0
            for key in ('Rmetal', 'Cmetal'))
    # These architecture properties are shared by both layers. They authorize
    # an explicit modeling assumption, never a claim of per-layer RR sampling.
    mirror_allowed = all(root.find(f"./segmentlist/segment[@name='{name}']").get('axis') is None
                         and all(root.find(f"./segmentlist/segment[@name='{name}']").get(key) is not None
                                 for key in ('Rmetal', 'Cmetal')) for name in SEGMENTS)
    return dict(directions=directions, z_segments=sorted(z_segments), segments=segments,
                ipin_switch=cb.get('input_switch_name'), zero_z_metal=zero_z_metal,
                mirror_allowed=mirror_allowed)


class _LimitedReader:
    def __init__(self, stream, deadline):
        self.stream, self.deadline, self.bytes = stream, deadline, 0

    def read(self, size=-1):
        _check_time(self.deadline)
        remaining = MAX_SAMPLE_BYTES - self.bytes
        if remaining <= 0:
            raise ValueError('017 timing RR node sampling hit decompressed byte cap before complete layer/axis/segment coverage; prepare a qualified sampled artifact')
        # iterparse normally requests 16 KiB; keep any parser request bounded.
        value = self.stream.read(min(size if size >= 0 else 65536, remaining, 65536))
        self.bytes += len(value)
        return value


def _node_timing(item):
    timing = item.find('timing')
    if timing is None or 'R' not in timing.attrib or 'C' not in timing.attrib:
        raise ValueError('017 timing sampled RR channel node lacks actual R/C; export a timing-enabled RR graph')
    return (_finite(timing.get('R'), 'RR node R [ohm]'),
            _finite(timing.get('C'), 'RR node C [farad]'))


def _sample(rr_graph, config, deadline):
    switches, rr_segments = defaultdict(list), {}
    samples = defaultdict(list)
    z_samples = []
    nodes = 0
    keys = {(z, axis, name) for z in (0, 1) for axis in ('CHANX', 'CHANY') for name in SEGMENTS}
    metadata_complete = set()
    stop_reason = 'rr_nodes_end'
    opener = gzip.open if str(rr_graph).endswith('.gz') else open
    with opener(rr_graph, 'rb') as stream:
        bounded = _LimitedReader(stream, deadline)
        stack = []
        for event, item in ET.iterparse(bounded, events=('start', 'end')):
            if event == 'start':
                if item.tag == 'rr_edges':
                    raise ValueError('017 timing RR nodes missing/incomplete before rr_edges; no edge traversal attempted')
                stack.append(item)
                continue
            parent = stack[-2] if len(stack) > 1 else None
            tag = parent.tag if parent is not None else None
            if item.tag == 'switch' and tag == 'switches':
                if sum(map(len, switches.values())) >= MAX_SWITCHES:
                    raise ValueError('017 timing RR switch metadata exceeds cap')
                timing = item.find('timing')
                # VPR's XML serializer may omit zero R/Cin/Cout.  An omitted
                # Tdel is *not* accepted for any switch used by this adapter.
                record = dict(id=int(item.get('id')), name=item.get('name'),
                              type=item.get('type'), Tdel=None)
                if timing is not None:
                    record.update({key: _finite(timing.get(key, '0'), f'switch {key}') for key in ('R', 'Cin', 'Cout')})
                    if timing.get('Tdel') is not None:
                        record['Tdel'] = _finite(timing.get('Tdel'), 'resolved RR switch Tdel')
                switches[record['name']].append(record)
            elif item.tag == 'segment' and tag == 'segments':
                rr_segments[int(item.get('id'))] = item.get('name')
            elif item.tag in ('switches', 'segments'):
                metadata_complete.add(item.tag)
            elif item.tag == 'node' and tag == 'rr_nodes':
                nodes += 1
                if nodes > MAX_SAMPLE_NODES:
                    raise ValueError('017 timing RR node sampling hit node cap before complete coverage; prepare a qualified sampled artifact')
                kind = item.get('type')
                if kind in ('CHANX', 'CHANY', 'CHANZ'):
                    loc, segment = item.find('loc'), item.find('segment')
                    if loc is None or segment is None:
                        raise ValueError('017 timing RR channel node missing loc/segment')
                    name = rr_segments.get(int(segment.get('segment_id')))
                    z = int(loc.get('layer_low', loc.get('layer', '-1')))
                    high = int(loc.get('layer_high', str(z)))
                    if kind in ('CHANX', 'CHANY') and name in SEGMENTS and z == high and z in (0, 1):
                        key = z, kind, name
                        values = samples[key]
                        axis = 'x' if kind == 'CHANX' else 'y'
                        length = int(loc.get(axis+'high')) - int(loc.get(axis+'low')) + 1
                        if length <= 0:
                            raise ValueError('017 timing invalid sampled RR span')
                        nominal = config['segments'][name]['length']
                        # Replace prefix boundary clips with full spans when
                        # available, retaining only a bounded number of nodes.
                        if len(values) < SAMPLES_PER_STRATUM or length == nominal and any(v['length_tiles'] != nominal for v in values):
                            r, c = _node_timing(item)
                            record = dict(node_id=int(item.get('id')), length_tiles=length, R_ohm=r, C_farad=c)
                            if len(values) == SAMPLES_PER_STRATUM:
                                values[next(i for i, v in enumerate(values) if v['length_tiles'] != nominal)] = record
                            else:
                                values.append(record)
                    elif kind == 'CHANZ' and name in config['z_segments'] and len(z_samples) < SAMPLES_PER_STRATUM:
                        r, c = _node_timing(item)
                        z_samples.append(dict(node_id=int(item.get('id')), layer_low=z, layer_high=high, R_ohm=r, C_farad=c))
                if nodes % 512 == 0:
                    _check_time(deadline)
                z_ready = z_samples or config['zero_z_metal']
                if metadata_complete == {'switches', 'segments'} and all(len(samples[k]) >= SAMPLES_PER_STRATUM for k in keys) and z_ready:
                    stop_reason = 'stratum_quotas_met'
                    break
                if (nodes >= EARLY_SAMPLE_NODES and config['mirror_allowed'] and z_ready and
                    all(any(len(v) >= SAMPLES_PER_STRATUM for k, v in samples.items() if k[2] == name) for name in SEGMENTS)):
                    stop_reason = 'early_prefix_with_explicit_shared_segment_mirroring'
                    break
            elif item.tag == 'rr_nodes':
                break
            if parent is not None and tag in ('rr_nodes', 'switches', 'segments', 'grid', 'block_types', 'channels'):
                parent.remove(item)
            stack.pop()
    missing = sorted(keys - {key for key, values in samples.items() if values})
    mirror_sources = {}
    if missing and config['mirror_allowed']:
        observed_keys = [key for key, values in samples.items() if values]
        for key in missing:
            candidates = [k for k in observed_keys if k[2] == key[2]]
            if candidates:
                source = max(candidates, key=lambda k: (k[1] == key[1], len(samples[k]), k))
                mirror_sources[key] = source
                samples[key] = samples[source]
    remaining = sorted(keys - {key for key, values in samples.items() if values})
    if metadata_complete != {'switches', 'segments'} or remaining:
        raise ValueError(f'017 timing RR metadata/sample coverage incomplete (missing {missing}); require actual L4/L16 CHANX/CHANY samples on both layers')
    if not z_samples and not config['zero_z_metal']:
        raise ValueError('017 timing no actual CHANZ R/C sample before caps; export a qualified SB RR graph')
    return dict(switches=dict(switches), samples=samples, z_samples=z_samples,
                mirror_sources=mirror_sources,
                nodes_visited=nodes, decompressed_bytes=bounded.bytes, stop_reason=stop_reason)


def _switch_records(sample, name, audit_names):
    if name not in audit_names:
        raise ValueError(f'017 timing SB switch {name!r} is not certified by device actual-adjacent-switch audit')
    records = sample['switches'].get(name, [])
    if not records or any(x['Tdel'] is None for x in records):
        raise ValueError(f'017 timing missing resolved scalar RR switch Tdel for {name!r}; architecture fanin tables are not a fallback')
    if any(x['type'] not in ('mux', 'buffer') for x in records):
        raise ValueError(f'017 timing unsupported RR switch type for {name!r}; this SB template requires buffered muxes')
    return records


def _switch_delay(records, wire_r, wire_c, load_c=0.):
    # Seconds: intrinsic switch delay + its output resistance charging the
    # segment and one receiver; distributed wire contributes R*C/2.  Cin/Cout
    # enter as capacitance, never as an extra intrinsic Tdel or R*Tdel term.
    return statistics.median(x['Tdel'] + x['R']*(wire_c + load_c + x['Cout']) +
                             wire_r*(wire_c/2 + load_c) for x in records) * 1e9


def _seal(data):
    data['model_sha256'] = _digest({k: v for k, v in data.items() if k != 'model_sha256'})
    return data


def validate_timing_model(data):
    """Validate a complete persisted adapter artifact; return it or raise."""
    if not isinstance(data, dict) or data.get('schema') != SCHEMA or data.get('units') != 'ns':
        raise ValueError('017 timing model schema/units mismatch')
    horizontal, vertical = data.get('horizontal_delay_per_tile_ns'), data.get('vertical_delay_ns')
    if not isinstance(horizontal, list) or len(horizontal) != 2 or not isinstance(vertical, list) or len(vertical) != 2:
        raise ValueError('017 timing model requires exactly two layers')
    for value in horizontal:
        _finite(value, 'horizontal delay per tile [ns]', positive=True)
    for z, row in enumerate(vertical):
        if not isinstance(row, list) or len(row) != 2:
            raise ValueError('017 timing model directional matrix malformed')
        for other, value in enumerate(row):
            number = _finite(value, 'vertical delay [ns]', positive=(z != other))
            if z == other and number != 0:
                raise ValueError('017 timing diagonal vertical delay must equal zero')
    _finite(data.get('terminal_delay_ns'), 'terminal delay [ns]')
    provenance = data.get('provenance', {})
    for key in ('rr_graph_sha256', 'architecture_sha256', 'device_content_sha256', 'implementation_sha256', 'signature_sha256'):
        if not _sha_string(provenance.get(key)):
            raise ValueError(f'017 timing model missing provenance {key}')
    if not _sha_string(data.get('model_sha256')) or data['model_sha256'] != _digest({k: v for k, v in data.items() if k != 'model_sha256'}):
        raise ValueError('017 timing model content digest mismatch; cached artifact modified')
    return data


def build_timing_model(rr_graph, architecture, device, device_manifest=None, out=None, max_seconds=60):
    """Read actual RR timing and return an immutable, self-checking JSON model.

    Paths or parsed JSON dictionaries are accepted for device/manifest. Passing
    a device path additionally verifies the manifest's original graph file SHA.
    Every invocation checks artifact presence, hashes and frozen device pins,
    including cache hits. Hashing streams compressed bytes when RR XML is gz.
    max_seconds covers hashing, metadata parsing and sampling, without loading
    a whole RRG or retaining node/edge tables.
    """
    seconds = _finite(max_seconds, 'max_seconds', positive=True)
    started = time.perf_counter()
    deadline = started + seconds
    rr_graph, architecture = Path(rr_graph), Path(architecture)
    device_path = Path(device) if isinstance(device, (str, Path)) else None
    # Missing input cannot be concealed by a previously materialized cache.
    for path in (rr_graph, architecture, device_path):
        if path is not None and not path.is_file():
            raise FileNotFoundError(f'017 timing requires existing input artifact: {path}')
    data = _load(device)
    manifest = _load(device_manifest) if device_manifest is not None else None
    if data.get('layers', data.get('grid', {}).get('layers')) != 2:
        raise ValueError('017 timing adapter requires a two-layer audited SB device')
    pins = data.get('provenance', {})
    if pins.get('vtr_commit') != PINNED_VTR_COMMIT:
        raise ValueError('017 timing requires device provenance for the pinned 40f8 Hecate backend')
    hashes = dict(rr_graph_sha256=_file_sha(rr_graph, deadline),
                  architecture_sha256=_file_sha(architecture, deadline))
    for key, value in hashes.items():
        if pins.get(key) != value:
            raise ValueError(f'017 timing frozen device provenance mismatch: {key}')
        if manifest is not None and manifest.get(key) != value:
            raise ValueError(f'017 timing frozen device manifest mismatch: {key}')
    device_file_sha = _file_sha(device_path, deadline) if device_path else None
    if manifest is not None:
        if manifest.get('channel_width') != data.get('channel_width'):
            raise ValueError('017 timing device manifest channel_width mismatch')
        if device_file_sha is not None and manifest.get('graph_sha256') != device_file_sha:
            raise ValueError('017 timing frozen device manifest graph_sha256 mismatch')
    config = _architecture(architecture, deadline)
    audit = data.get('audit', {})
    audited = audit.get('z_actual_adjacent_switch_classes')
    if not isinstance(audited, list) or not audited:
        raise ValueError('017 timing requires actual device adjacent-CHANZ switch-class audit')
    if set(config['directions'].values()) - set(audited):
        raise ValueError('017 timing SG direction classes are absent from actual device switch audit')
    provenance = dict(**hashes, rr_graph_bytes=rr_graph.stat().st_size,
                      device_content_sha256=_digest(data), device_file_sha256=device_file_sha,
                      device_manifest_sha256=_digest(manifest) if manifest is not None else None,
                      graph_file_pin_verified=device_file_sha is not None and manifest is not None,
                      implementation_sha256=_file_sha(Path(__file__), deadline),
                      vtr_commit=PINNED_VTR_COMMIT)
    provenance['signature_sha256'] = _digest(provenance)
    if out is not None and Path(out).exists():
        cached = validate_timing_model(_load(out))
        if cached['provenance'] != provenance:
            raise ValueError('017 timing cache input/implementation signature mismatch; use a new cache path')
        _check_time(deadline)
        return cached
    sample = _sample(rr_graph, config, deadline)
    # Horizontal drivers need not all be adjacent to CHANZ (L16 is usually not),
    # but their identities are certified by architecture segment declarations.
    allowed = set(audited) | {x['mux'] for x in config['segments'].values()} | {config['ipin_switch']}
    drivers = {name: _switch_records(sample, item['mux'], allowed) for name, item in config['segments'].items()}
    ipins = _switch_records(sample, config['ipin_switch'], allowed)
    vertical_records = {offset: _switch_records(sample, name, audited) for offset, name in config['directions'].items()}
    details, horizontal = [], []
    for layer in (0, 1):
        layer_details = []
        weighted = weight = 0.
        for axis in ('CHANX', 'CHANY'):
            for name in SEGMENTS:
                nodes = sample['samples'][layer, axis, name]
                full = [n for n in nodes if n['length_tiles'] == config['segments'][name]['length']]
                used = full or nodes
                records = drivers[name]
                load_c = statistics.median(x['Cin'] for x in records)
                per_tile = statistics.median(_switch_delay(records, n['R_ohm'], n['C_farad'], load_c)/n['length_tiles'] for n in used)
                _finite(per_tile, f'{layer}/{axis}/{name} delay per tile', positive=True)
                frequency = config['segments'][name]['frequency']
                weighted += frequency * per_tile
                weight += frequency
                source = sample['mirror_sources'].get((layer, axis, name))
                layer_details.append(dict(axis=axis, segment=name, samples=nodes,
                    sample_count=len(nodes), full_length_count=len(full), used_full_length_only=bool(full),
                    actual_stratum_sample_count=0 if source else len(nodes),
                    mirrored_from=list(source) if source else None,
                    horizontal_delay_per_tile_ns=per_tile, nominal_length_tiles=config['segments'][name]['length'],
                    nominal_frequency=frequency))
        horizontal.append(weighted/weight)
        details.append(dict(layer=layer, strata=layer_details))
    zr = statistics.median(x['R_ohm'] for x in sample['z_samples']) if sample['z_samples'] else 0.
    zc = statistics.median(x['C_farad'] for x in sample['z_samples']) if sample['z_samples'] else 0.
    # Vertical coefficient includes the Z entry mux and sampled Z wire. Exit
    # horizontal drivers are already amortized into horizontal traversal and
    # are not charged a second time at the portal.
    exit_c = statistics.median(x['Cin'] for x in drivers['L4'])
    up = _switch_delay(vertical_records['1'], zr, zc, exit_c)
    down = _switch_delay(vertical_records['-1'], zr, zc, exit_c)
    terminal = _switch_delay(ipins, 0., 0.)
    used_names = sorted({item['mux'] for item in config['segments'].values()} |
                        set(config['directions'].values()) | {config['ipin_switch']})
    model = dict(schema=SCHEMA, units='ns', proxy_kind='SB device-derived routed-connection delay template',
        horizontal_delay_per_tile_ns=horizontal, vertical_delay_ns=[[0., up], [down, 0.]],
        vertical_hop_delay_ns={'0->1': up, '1->0': down}, terminal_delay_ns=terminal,
        pin_access_delay_ns=terminal, provenance=provenance, layers=details,
        resolved_rr_switches={name: sample['switches'][name] for name in used_names},
        vertical_switch_names={'0->1': config['directions']['1'], '1->0': config['directions']['-1']},
        sampled_vertical_nodes=sample['z_samples'],
        vertical_wire_rc_source='actual sampled CHANZ nodes' if sample['z_samples'] else 'explicit zero Rmetal/Cmetal architecture declaration; no CHANZ RC samples',
        sampling=dict(method='deterministic bounded RR node prefix, stratified by layer/axis/L4/L16; prefer full spans',
            nodes_visited=sample['nodes_visited'], decompressed_bytes=sample['decompressed_bytes'],
            stop_reason=sample['stop_reason'], node_cap=MAX_SAMPLE_NODES, decompressed_byte_cap=MAX_SAMPLE_BYTES,
            samples_per_stratum_cap=SAMPLES_PER_STRATUM, rr_edges_visited=0),
        formula=dict(horizontal='frequency-weighted mean over L4/L16 and both axes of median(hop_delay_ns / actual inclusive span_tiles)',
            hop_delay_ns='1e9 * median(Tdel + R_switch*(C_wire + C_receiver + Cout_switch) + R_wire*(C_wire/2 + C_receiver))',
            vertical='resolved directional SG entry mux + sampled CHANZ wire, loaded by one L4-driver Cin',
            terminal='one local ipin_cblock delay; source access included by horizontal driver amortization',
            connection='terminal + sum(layer horizontal_distance_tiles * horizontal_delay_per_tile_ns[layer]) + directional vertical_delay_ns for a portal crossing',
            raw_units=dict(Tdel='second', R='ohm', C='farad', span='physical FPGA tile; inclusive RR channel coordinates')),
        assumptions=[
            'Only the pinned two-layer Hecate SB pattern with explicit zero Z pin fc is qualified; CB is unsupported.',
            'A straight route is approximated by amortized switched wire hops; placement-time portal geometry supplies detours.',
            'One receiver Cin per hop; no full fanout RC tree, switch loading census, congestion timing, or setup/clock/cell delay.',
            'Each named RR mux may have several resolved fanin variants: median scalar delay is used, not an architecture Tdel table interpolation.',
            'Distinct directional SG mux classes retain their distinct delays; a shared name uses the same resolved-variant median in both directions.',
            'A prefix node sample is not an unbiased chip census; full spans are preferred. Missing layer/axis strata may be explicitly mirrored only for shared global isotropic segment R/C declarations; actual_stratum_sample_count and mirrored_from disclose this.',
            'CHANZ wire R/C can be taken as zero without sampling only when the architecture explicitly declares both Z metal values zero; resolved nonzero SG switch delay is still mandatory.',
            'VPR omitted zero-valued switch R/Cin/Cout fields are interpreted as zero; absent resolved Tdel and absent node R/C are errors.',
            'Exit-driver cost is amortized into horizontal distance; zero-XY cross-layer connections omit that exit-driver floor.',
            'Depth-based net criticality, when applied by the caller, is a heuristic and is not STA; this artifact never estimates path CPD.'
        ], extraction_seconds=time.perf_counter()-started)
    _seal(model)
    validate_timing_model(model)
    _check_time(deadline)
    if out is not None:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Exclusive creation preserves immutable preparation caches on races.
        with path.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(model, sort_keys=True, indent=2, allow_nan=False)+'\n')
    return model
