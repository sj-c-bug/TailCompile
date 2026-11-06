"""Register fresh 014 families, prepare inputs and qualify one Default width per design."""
from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
import os
from pathlib import Path
import re
import subprocess
import tempfile

from architectures import COMMIT, SOURCE_ARCH, audit_family, generate_architecture
from campaign import ARCHES, Campaign, freeze as freeze_json_form, identity, within
from io_utils import freeze, read, sha, write
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WIDTHS = (128, 192, 256)
CALIBRATION_SEED = 14001
EXPOSURE_LEDGER = (
    'experiments/vtr/tailcompile_v66/cohort.json',
    'experiments/vtr/tailcompile_v68/cohort.json',
    'experiments/vtr/tailcompile_v610/runs/koios_sweep_002/inventory.json',
    'experiments/vtr/tailcompile_v610/runs/koios_sweep_002/prepared_inputs.json',
    'experiments/vtr/tailcompile_v7_3d/runs/development_003_paths/protocol.json',
    'experiments/vtr/tailcompile_v7_3d/runs/development_004_capacity/prepared.json',
)
MAPPED_LEDGER = {
    'experiments/vtr/tailcompile_v68/runs/hier_init_001/contexts.json':'contexts',
    'experiments/vtr/tailcompile_v610/runs/koios_sweep_002/contexts.json':'contexts',
    'experiments/vtr/tailcompile_v610/runs/koios_sweep_002/prepared_inputs.json':'designs',
    **{f'experiments/vtr/tailcompile_v7_3d/runs/{cohort}/prepared.json':'records' for cohort in (
        'development_003_paths', 'development_004_capacity', 'development_005_qualified_learning',
        'development_006_learning_repair', 'development_007_layer/borrowed')},
}


def audit_mapped_blifs(new, repo=REPO):
    """Use archived digests even when historical large netlists have been pruned."""
    observed, covered, sources = defaultdict(set), set(), {}
    for relative, field in MAPPED_LEDGER.items():
        path = repo/relative
        rows = read(path)[field]
        missing, hashes = set(), set()
        for row in rows:
            design = row['design']
            digest = row.get('mapped_blif_sha256')
            if not digest:
                missing.add(design)
                continue
            if not re.fullmatch('[0-9a-f]{64}',digest):
                raise ValueError('invalid historical mapped BLIF signature: '+relative)
            observed[digest].add(design)
            covered.add(design)
            hashes.add(digest)
        sources[relative] = dict(sha256=sha(path), rows=len(rows), distinct_sha256=len(hashes),
                                 missing_hash_designs=sorted(missing))
    if not observed:
        raise ValueError('014 no historical mapped BLIF signatures available')
    inventory = read(repo/'experiments/vtr/tailcompile_v610/runs/koios_sweep_002/inventory.json')['designs']
    prior_new, collisions = {}, []
    for design, values in new.items():
        for digest in sorted(set(values.values())):
            if digest in observed or digest in prior_new:
                collisions.append(dict(design=design, sha256=digest,
                    historical_designs=sorted(observed.get(digest,())), other_014_design=prior_new.get(digest)))
            prior_new[digest]=design
    return dict(passed=not collisions, collisions=collisions, new_mappings=new, sources=sources,
                distinct_historical_mapped_sha256=len(observed), historical_design_count=len(covered),
                v610_inventory_without_digest_in_checked_ledgers=sorted({r['design'] for r in inventory}-covered),
                scope='byte digests in listed historical mapping records; other unsynchronized mappings and structurally equivalent encodings require separate lineage review')


def rearm_missing(folder, outputs):
    """Recover only incomplete unpublished stage payloads under the owned cohort lock."""
    manifest = folder/'manifest.json'
    if not manifest.is_file():
        return
    trial = read(manifest)
    if trial['status']!='complete':
        return
    work = within(trial['workdir'],folder/'attempts')
    if all(p.is_file() and p.stat().st_size for p in outputs(work)):
        return
    if len(list((folder/'attempts').glob('attempt_*')))>=2:
        raise RuntimeError('014 preparation retry exhausted; missing stage payload at '+str(folder))
    verify_pins(trial['signature']['inputs'])
    archived = work/'incomplete_payload_manifest.json'
    if archived.exists():
        raise ValueError('014 incomplete preparation attempt already archived')
    manifest.rename(archived)


def name_key(value):
    return re.sub('[^a-z0-9]', '', value.casefold())


def historical_exposure(repo):
    """Match declared lineages, known RTL hashes and tracked historical run names."""
    names, hashes, pins = set(), set(), {}
    for relative in EXPOSURE_LEDGER:
        path = repo/relative
        doc = read(path)
        pins[relative] = sha(path)
        source = doc.get('config', doc)
        rows = source.get('designs', source.get('records'))
        if not isinstance(rows, list) or not rows:
            raise ValueError('014 unrecognized or empty exposure registry: '+relative)
        names.update(name_key(x) for x in source.get('previously_used_families', []))
        for row in rows:
            for field in ('design', 'family', 'source'):
                if row.get(field):
                    names.add(name_key(Path(row[field]).stem if field == 'source' else row[field]))
            if row.get('source_sha256'):
                hashes.add(row['source_sha256'])
    # Inspect tracked run directories as well, including older 2D/3D cohorts
    # without a common inventory schema. Untracked scratch is not evidence.
    paths = subprocess.check_output(['git', '-C', str(repo), 'ls-tree', '-dr', '--name-only',
                                     'HEAD', '--', 'experiments/vtr', 'experiments/vtr3d'], text=True)
    for path in paths.splitlines():
        parts = path.split('/')
        if 'runs' in parts:
            suffix = parts[parts.index('runs')+1:]
            if suffix and not suffix[0].startswith('development_014'):
                names.update(name_key(part) for part in suffix)
    return names, hashes, pins


def register_panel(manifest, mapping_root, *, repo=REPO):
    """Read-only registration; source membership and immutable bytes precede QoR."""
    manifest = Path(manifest).resolve()
    root = Path(mapping_root).resolve()
    value = read(manifest)
    rows = value.get('designs')
    if value.get('schema') != 'tailcompile-014-source-panel-1' or not isinstance(rows,list) or len(rows) != 2:
        raise ValueError('014 requires two registered independent families')
    known, source_hashes, history = historical_exposure(Path(repo))
    used_names, used_hashes, designs = set(), set(), []
    for row in rows:
        design, family = row['design'], row['family']
        aliases = row['aliases']
        if not isinstance(aliases,list) or not all(isinstance(a,str) and a.strip() for a in aliases):
            raise ValueError('014 invalid family alias list')
        if not all(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', x) for x in (design, family)):
            raise ValueError('014 design and family must be safe single path components')
        if not isinstance(row.get('lineage'),str) or not row['lineage'].strip():
            raise ValueError('014 each design requires reviewed RTL lineage')
        relative = Path(row['source'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('014 source must be relative to the mapping VTR checkout')
        path = within(root/relative, root/'vtr_flow/benchmarks')
        if path.suffix.lower() not in ('.v', '.blif') or not path.is_file():
            raise ValueError('014 expected a registered single-file Verilog or BLIF source: '+str(path))
        digest = sha(path)
        if digest != row['source_sha256']:
            raise ValueError('014 registered source bytes changed: '+design)
        tags = {name_key(s) for s in (design, family, path.stem, *aliases)}
        hit = tags & (known | used_names)
        if hit or digest in source_hashes or digest in used_hashes:
            raise ValueError(f'014 source/lineage already exposed or duplicated: {design} {sorted(hit)}')
        grid = row['grid']
        if not isinstance(grid,list) or len(grid)!=2 or any(type(x) is not int or x<12 for x in grid):
            raise ValueError('014 source requires a supported predeclared fixed grid')
        used_names.update(tags)
        used_hashes.add(digest)
        designs.append(dict(design=design, family=family, aliases=aliases, lineage=row['lineage'], grid=grid,
                            source=str(path), source_relative=relative.as_posix(), source_sha256=digest))
    return dict(schema=value['schema'], manifest=str(manifest), manifest_sha256=sha(manifest),
                exposure_sources=history, designs=designs,
                exposure_rule='reviewed source lineage, registered design/family/aliases, available source hashes and tracked run directory names',
                stage='prospective development; preserve full-history independence only until method results are inspected')


class InputCampaign(Campaign):
    """Reuse the fixed 004 mapper/RRG pipeline with the measured 007 backend."""

    def gate(self):
        self.check_binary()
        report_path = self.args.qualification_report
        report = read(report_path)
        if report.get('schema') != 'tailcompile-layer-qualification-007' or not report.get('passed') or not report.get('complete') or \
                report['binary_sha256'] != self.binary_hash or report.get('pnr_count') != 12:
            raise ValueError('014 requires passed 007 XYZ/backend qualification for this binary')
        proof = read(report_path.parent/'qualification_protocol.json')
        if proof['binary_sha256'] != self.binary_hash or proof['schema'] != report['schema']:
            raise ValueError('014 qualification protocol mismatch')
        if not self.args.allow_local:
            earlier = read(HERE/'runs/development_007_layer/layer_protocol.json')
            if earlier['qualification_sha256'] != sha(report_path) or earlier['backend_manifest_sha256'] != sha(self.args.backend_manifest):
                raise ValueError('014 source backend differs from the frozen natural-design 007 backend')
        return report

    def architecture(self, record, condition):
        width,height=record['grid']
        folder=self.out/'architectures'/f'{width}x{height}'/condition
        with self._lock('014-create:'+str(folder)):
            if not (folder/'architecture.json').is_file():
                folder.mkdir(parents=True,exist_ok=True)
                arch=folder/'arch.xml'
                with tempfile.TemporaryDirectory(dir=folder) as tmp:
                    staged=Path(tmp)/'arch.xml'
                    meta=generate_architecture(self.vtr/SOURCE_ARCH,staged,int(condition[1]),int(condition[3]),width,height)
                    meta['architecture']=str(arch)
                    staged.replace(arch)
                    write(folder/'architecture.json',meta)
            return super().architecture(record,condition)

    def rr(self, record, condition, width):
        arch=self.architecture(record,condition)
        key=identity(dict(architecture=sha(arch),binary=self.binary_hash,width=width))[:20]
        folder=self.out/'device_cache'/key
        with self._lock('014-build:'+key):
            if not (folder/'manifest.json').exists():
                rearm_missing(folder/'build',lambda work:[folder/'rr_graph.xml'])
            return super().rr(record,condition,width)


def make_campaign(args, panel):
    out = Path(args.out).resolve()
    scratch = Path(args.scratch).resolve()
    if out.parent != HERE/'runs' or not out.name.startswith('development_014'):
        raise ValueError('014 input outputs must use their own v7 development_014 namespace')
    within(scratch, REPO/'vtr_work/tailcompile_v7_3d/v8_014')
    mapping = Path(args.mapping_vtr_root or args.vtr_root).resolve()
    validate_output_roots((out,scratch), dependency_roots=(mapping, args.binary.parent))
    arch_source = mapping/SOURCE_ARCH
    if not arch_source.is_file():
        raise FileNotFoundError('014 full mapping VTR checkout lacks the pinned Hecate XML: '+str(arch_source))
    if sha(panel['manifest']) != panel['manifest_sha256'] or any(sha(r['source'])!=r['source_sha256'] for r in panel['designs']):
        raise ValueError('014 registration changed before input preparation')
    freeze(out/'registered_panel.json', panel)
    # The reused campaign's loader accepts validation/train tags and requires
    # a legacy timing-policy field. 014 has no teacher/training phases.
    config = dict(designs=[dict(r, split='validation') for r in panel['designs']], width_grid=list(WIDTHS),
                  primary_runtime='cold', primary014_metrics=['wirelength','cpd_ns'], tile_size=4,
                  source_arch_sha256=sha(arch_source), exposure='014 frozen prospective panel')
    config_path = out/'input_config.json'
    freeze(config_path, config)
    options = Namespace(**vars(args))
    options.out, options.scratch, options.config = out, scratch, config_path
    options.mapping_vtr_root = options.vtr_root = mapping
    instance = InputCampaign(options)
    instance.gate()
    # io_utils enforces at most two attempts for a completed infrastructure failure.
    os.environ['TAILCOMPILE_V7_RETRY_INFRASTRUCTURE']='1'
    return instance


def prepare_inputs(campaign):
    if not (campaign.out/'prepared.json').exists():
        for row in campaign.config['designs']:
            folder=campaign.out/'inputs'/row['design']
            source=Path(row['source'])
            if source.suffix.lower()!='.blif':
                rearm_missing(folder/'mapping',lambda work:[work/'flow'/(source.stem+'.abc.blif')])
            rearm_missing(folder/'structure_run',lambda work:[folder/('structure'+s) for s in ('.blocks.tsv','.nets.tsv','.meta.tsv')])
            # The legacy adapter writes BLIF then its receipt. Regenerate an
            # unpublished BLIF after an interruption without trusting partial bytes.
            adapted=folder/'design.blif'
            if adapted.exists() and not (folder/'mapping_adapter.json').is_file():
                adapted.replace(folder/'design.unpublished.blif')
    records = campaign.prepare()
    if len(records)!=2 or len({r['family'] for r in records})!=2:
        raise ValueError('014 prepared input families differ from two-family registration')
    mapping_hashes = {r['mapped_blif_sha256'] for r in records}
    mappings={}
    inputs = {str(campaign.out/'prepared.json'):sha(campaign.out/'prepared.json')}
    for row in campaign.records():
        if sha(row['source']) != row['source_sha256']:
            raise ValueError('014 source bytes differ after preparation: '+row['design'])
        folder = campaign.out/'inputs'/row['design']
        adapter_path = folder/'mapping_adapter.json'
        adapter = read(adapter_path)
        if adapter['derived_sha256'] != row['mapped_blif_sha256'] or sha(adapter['source']) != adapter['source_sha256']:
            raise ValueError('014 mapper/adapter netlist identity differs: '+row['design'])
        mappings[row['design']] = dict(before_adapter=adapter['source_sha256'], deployed_blif=row['mapped_blif_sha256'])
        inputs[str(adapter_path)] = sha(adapter_path)
        path = folder/'logic.json'
        inputs[str(path)] = sha(path)
    audit_path=campaign.out/'mapped_exposure_audit.json'
    audit=audit_mapped_blifs(mappings)
    freeze(audit_path,audit)
    if not audit['passed']:
        raise ValueError('014 mapped netlist matched a prior design: '+str(audit['collisions']))
    inputs[str(audit_path)]=sha(audit_path)
    freeze(campaign.out/'prepared_inputs_manifest.json', dict(files=inputs, mapped_blif_hashes=sorted(mapping_hashes)))
    for row in records:
        cleanup_mapping_copies(campaign.out/'inputs'/row['design'],row)
    return records


def cleanup_mapping_copies(folder, record):
    """Keep mapper output and deployed BLIF, release only exact duplicate inputs."""
    mapper = folder/'mapping/manifest.json'
    if not mapper.is_file():
        return
    flow = within(read(mapper)['workdir'],mapper.parent/'attempts')/'flow'
    if flow.is_symlink() or flow != flow.resolve():
        raise ValueError('014 mapping cleanup cannot follow a redirected flow directory')
    adapter=read(folder/'mapping_adapter.json')
    originals=[Path(record['source']),Path(adapter['architecture'])]
    protected={p.resolve() for p in originals} | {Path(adapter['source']).resolve(),Path(record['blif']).resolve()}
    receipt=folder/'mapping_copy_prune_plan.json'
    if not receipt.exists():
        copies=[]
        for origin in originals:
            duplicate=flow/origin.name
            if duplicate.is_file() and not duplicate.is_symlink() and duplicate.resolve() not in protected and sha(duplicate)==sha(origin):
                copies.append(dict(path=str(duplicate),origin=str(origin),sha256=sha(origin),bytes=duplicate.stat().st_size))
        freeze(receipt,dict(files=copies))
    copies=read(receipt)['files']
    for entry in copies:
        duplicate=Path(entry['path'])
        origin=Path(entry['origin'])
        if origin not in originals or duplicate!=flow/origin.name or duplicate.is_symlink() or duplicate.resolve() in protected:
            raise ValueError('014 mapping duplicate cleanup path differs')
        if duplicate.exists():
            if sha(origin)!=entry['sha256'] or sha(duplicate)!=entry['sha256']:
                raise ValueError('014 mapping duplicate changed')
            duplicate.unlink()
    freeze(folder/'payload_pruned.json',dict(verified_copies=copies, bytes=sum(e['bytes'] for e in copies),
        retained=['mapped BLIF before and after adapter','logic/structure','mapping logs']))


def verify_pins(pins):
    for path, expected in pins.items():
        if sha(path) != expected:
            raise ValueError('014 frozen qualification input changed: '+str(path))


def check_actual(graphs):
    result = audit_family(graphs)
    for width, rows in result['by_width'].items():
        for z in (0,1):
            balanced = rows[f'H0Z{z}']['layer_resource_capacity']
            unbalanced = rows[f'H1Z{z}']['layer_resource_capacity']
            if len(balanced)!=2 or len(unbalanced)!=2 or balanced[0]!=balanced[1] or any(
                    unbalanced[0][r] == unbalanced[1][r] for r in ('memory','dsp_top')):
                raise ValueError(f'014 actual per-layer resource factor missing at width {width}')
        for h in (0,1):
            if rows[f'H{h}Z0']['layer_resource_capacity'] != rows[f'H{h}Z1']['layer_resource_capacity']:
                raise ValueError('014 vertical-channel factor unexpectedly changes resource supply')
    if not all(c['z_capacity_invariant'] for c in result['width_coupling'].values()):
        raise ValueError('014 vertical capacity changed across the attempted routing widths')
    return result


def qualified_conditions(campaign):
    campaign.gate()
    records = campaign.records()
    verify_pins(read(campaign.out/'prepared_inputs_manifest.json')['files'])
    receipt = read(campaign.out/'conditions_manifest.json')
    verify_pins(receipt['files'])
    rows = read(campaign.out/'conditions.json')['records']
    expected = {(r['design'], a) for r in records for a in ARCHES}
    if len(rows)!=8 or {(r['design'],r['architecture']) for r in rows}!=expected:
        raise ValueError('014 frozen condition panel must contain two families and four architectures')
    return rows


def calibrate_inputs(campaign):
    """At most one 4-cell Default wave per design/width, with audits before P&R."""
    from prospective_native import run_native
    if (campaign.out/'conditions_manifest.json').is_file():
        return qualified_conditions(campaign)
    old_report = campaign.out/'input_qualification_report.json'
    if old_report.exists() and not read(old_report)['passed']:
        raise RuntimeError('014 registered panel already failed Default qualification: '+str(old_report))
    campaign.gate()
    records = campaign.records()
    verify_pins(read(campaign.out/'prepared_inputs_manifest.json')['files'])
    active = {r['design']:r for r in records}
    finished, failed, files = {}, {}, {}
    graphs = defaultdict(lambda:defaultdict(list))
    all_rows = []
    for width in WIDTHS:
        if not active:
            break
        tasks = [campaign.condition(r,a,width,'qualification') for r in active.values() for a in ARCHES]
        def build(c):
            rr = campaign.rr(c['record'], c['architecture'], width)
            dev = read(rr['device'])
            if dev['channel_width'] != width or dev['audit']['grid'] != list(c['record']['grid'])+[2]:
                raise ValueError('014 cached actual RRG width or device dimensions mismatch')
            return c['design'], c['architecture'], dict(channel_width=width,audit=dev['audit']), rr['manifest']
        for design, arch, summary, manifest in campaign.parallel(tasks, build):
            graphs[design][arch].append(summary)
            files[manifest] = sha(manifest)
        for design in active:
            audit_path = campaign.out/'calibration'/design/f'w{width}'/'actual_audit.json'
            # The existing audit returns integer width keys; use the same JSON
            # representation for first publication and crash/resume comparison.
            freeze_json_form(audit_path, check_actual(graphs[design]))
            files[str(audit_path)] = sha(audit_path)
        def run(c):
            target = campaign.out/'calibration'/c['design']/c['architecture']/f'w{width}'/f's{CALIBRATION_SEED}'
            return run_native(campaign,c,CALIBRATION_SEED,target,'Default')
        rows = campaign.parallel(tasks, run)
        for design, record in list(active.items()):
            batch = sorted((r for r in rows if r['design']==design), key=lambda r:r['architecture'])
            if len(batch)!=4 or {r['architecture'] for r in batch}!=set(ARCHES) or any(
                    r['seed']!=CALIBRATION_SEED or r['width']!=width for r in batch):
                raise ValueError('014 incomplete or mismatched four-architecture calibration wave')
            for row in batch:
                files[str(row['vpr_manifest'])] = sha(row['vpr_manifest'])
            all_rows.extend(batch)
            if all(r['status']=='success' for r in batch):
                finished[design] = width
                del active[design]
            elif any(r['status']=='packing_or_placement_failure' and r.get('confirmed_candidate_infeasibility') for r in batch):
                failed[design] = 'confirmed device capacity failure'
                del active[design]
            elif any(r['status'] not in ('success','unroutable') for r in batch):
                raise RuntimeError('014 infrastructure or unclassified calibration failure: '+design)
        write(campaign.out/'calibration_progress_report.json',dict(width=width, reference_widths=finished,
              capacity_failures=failed, rows=all_rows))
    passed = len(finished)==2
    report_path = campaign.out/'input_qualification_report.json'
    freeze(report_path,dict(complete=True, passed=passed, reference_widths=finished,
        ineligible={**failed, **{d:'no common successful reference width' for d in active}}, rows=all_rows,
        native_count=len(all_rows), maximum_native_count=24, calibration_seed=CALIBRATION_SEED))
    if not passed:
        raise RuntimeError('014 prospective input qualification failed: '+str(report_path))
    rows = [campaign.condition(r,a,finished[r['design']],'reference') for r in records for a in ARCHES]
    path = campaign.out/'conditions.json'
    freeze(path, dict(records=rows, purpose='prospective real-P&R development panel'))
    for p in (path, report_path,campaign.out/'prepared_inputs_manifest.json',campaign.out/'registered_panel.json'):
        files[str(p)] = sha(p)
    freeze(campaign.out/'conditions_manifest.json',dict(files=files, complete=True))
    return rows
