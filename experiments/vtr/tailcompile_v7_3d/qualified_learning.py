"""Five-design development learning, reusing immutable qualified 004 inputs.

This entry never invokes calibration, builds an RRG, or changes the formal
four-family pilot. New results are not an unseen-family or Wmin acceptance.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import copy
import itertools
import math
import os
from pathlib import Path
import re
import threading
import time
from types import SimpleNamespace

from campaign import (ARCHES, METHODS, ORDERS, Campaign, campaign_lock, freeze,
                      identity, select_widths, within)
from complete_device_cache import _validate_protocol
from io_utils import emit, install_signal_handlers, read, run_trial, sha, vpr_command, write
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
TRAIN = {'sha': 'sha', 'spmv': 'spmv', 'mcml': 'mcml'}
VALIDATION = {'blob_merge': 'blob_merge', 'diffeq1': 'diffeq'}
SEEDS = (551, 552, 553, 554, 555)
STAGES = ('plan', 'teachers', 'pretrain', 'feedback', 'finetune', 'validate', 'compare', 'report')


def require_hash(path, digest):
    if not isinstance(digest, str) or sha(path) != digest:
        raise ValueError(f'qualified learning input changed: {path}')


def qualified_panel(source):
    """Audit light evidence, usable on the local Git copy without raw payloads."""
    source = Path(source)
    config = read(source/'campaign.json')
    if read(source/'protocol.json')['config'] != config:
        raise ValueError('source configuration differs from protocol')
    manifest = read(source/'conditions_manifest.json')
    for name, key in (('conditions.json', 'conditions_sha256'),
                      ('boundaries.json', 'boundaries_sha256'), ('prepared.json', 'prepared_sha256')):
        require_hash(source/name, manifest[key])
    prepared = read(source/'prepared.json')
    records = {r['design']: r for r in prepared['records']}
    if not prepared.get('complete') or len(records) != 6 or set(records) != {*TRAIN, *VALIDATION, 'or1200'}:
        raise ValueError('expected the complete six-input source cohort')
    boundaries = {r['design']: r for r in read(source/'boundaries.json')['records']}
    conditions = read(source/'conditions.json')['records']
    if len(conditions) != 40 or len({c['id'] for c in conditions}) != 40:
        raise ValueError('expected exactly forty unique qualified conditions')
    if (boundaries['or1200']['status'] != 'right_censored' or
            read(source/'conditions.json')['censored'] != ['or1200']):
        raise ValueError('this revision explicitly retains OR1200 as right censored')
    for design, family in {**TRAIN, **VALIDATION}.items():
        record = records[design]
        split = 'train' if design in TRAIN else 'validation'
        rows = [c for c in conditions if c['design'] == design]
        if (record['family'] != family or record['split'] != split or len(rows) != 8 or
                {(c['architecture'], c['width_role']) for c in rows} !=
                set(itertools.product(ARCHES, ('stress', 'reference')))):
            raise ValueError(f'incomplete panel or changed family split: {design}')
        trials = read(source/'calibration'/design/'trials.json')
        if any(r['status'] not in ('success', 'unroutable') or r['method'] != 'default' for r in trials):
            raise ValueError('qualification must use normal Default outcomes only')
        boundary = select_widths(trials, config['width_grid'])
        if boundary['status'] != 'paired':
            raise ValueError(f'not a qualified design: {design}')
        for key in ('status', 'W_ref', 'W_stress', 'success_rates'):
            if _json_value(boundary[key]) != boundaries[design][key]:
                raise ValueError(f'boundary disagrees with Default trials: {design}')
        for c in rows:
            expected_width = boundary['W_ref' if c['width_role'] == 'reference' else 'W_stress']
            if (c['record'] != record or c['width'] != expected_width or
                    c['family'] != family or c['split'] != split):
                raise ValueError('condition disagrees with prepared record or boundary')
        audit = read(source/'architectures'/f'{record["grid"][0]}x{record["grid"][1]}'/'actual_audit.json')
        if not audit.get('four_cell_matches'):
            raise ValueError('actual H/Z supply controls failed')
        if not all(r.get('z_capacity_invariant') for r in audit['width_coupling'].values()):
            raise ValueError('vertical capacity changes with channel width')
    return config, [records[d] for d in (*TRAIN, *VALIDATION)], conditions


def _json_value(value):
    import json
    return json.loads(json.dumps(value))


def queue_plan():
    return dict(scope='five-design development; active validation, not final test',
        train_designs=TRAIN, validation_designs=VALIDATION,
        excluded_but_retained={'or1200': 'Default right-censored; parallel diagnosis'},
        new_calibration_runs=0, new_rrg_builds=0, teacher_runs=80,
        training_conditions=24, feedback_candidate_runs=192, feedback_default_runs=48,
        validation_runs_upper=128, comparison_runs=600,
        comparison_train_runs=360, comparison_validation_runs=240,
        comparison_seeds=list(SEEDS), total_new_pnr_upper=1048,
        comparison_runtime='balanced same-core cold end-to-end; reusable RRG preparation separate',
        original_480_completed=False, Wmin_measured=False, final_generalization_claim=False)


def validate_paths(source, out, repo=REPO):
    source, out, repo = (Path(p).resolve() for p in (source, out, repo))
    runs = repo/'experiments/vtr/tailcompile_v7_3d/runs'
    if (runs not in source.parents or runs not in out.parents or source == out or
            source in out.parents or out in source.parents):
        raise ValueError('source and output must be separate cohorts inside v7 runs')
    validate_output_roots((out,), repo=repo)
    if out.exists() and any(p.is_symlink() for p in out.rglob('*')):
        raise ValueError('output cohort must not contain symlink write targets')
    if (out/'protocol.json').exists() and not (out/'qualified_source_manifest.json').is_file():
        raise ValueError('cannot write into an existing non-learning cohort')
    return source, out


def prepare_handoff(source, out):
    """Pin only light metadata; reference, never copy, large prior payloads."""
    source, out = validate_paths(source, out)
    config, records, conditions = qualified_panel(source)
    for record in records:
        for path in (record['blif'], record['structure_prefix']):
            resolved = Path(path).resolve()
            if resolved == out or out in resolved.parents:
                raise ValueError('output overlaps a borrowed input cohort')
    context = _validate_protocol(source, REPO)
    binary = context['binary']
    roots = [p for p in binary.parents if (p/'backend_manifest.json').is_file()]
    roots = [p for p in roots if sha(p/'backend_manifest.json') == read(source/'protocol.json')['backend_manifest_sha256']]
    if len(roots) != 1:
        raise ValueError('cannot uniquely locate the qualified read-only backend')
    backend = roots[0]
    mapping = Path(read(source/'protocol.json')['mapping_vtr_root']).resolve()
    validate_output_roots((out,), dependency_roots=(backend, mapping))
    files = {}
    def pin(path):
        path = within(path, REPO)
        files[str(path)] = sha(path)
        return path
    for name in ('protocol.json', 'campaign.json', 'prepared.json', 'conditions.json',
                 'conditions_manifest.json', 'boundaries.json', 'binary_identity.json',
                 'qualification/protocol.json', 'qualification/qualification_report.json'):
        pin(source/name)
    borrowed = {}
    if config.get('input_recovery'):
        pointer = config['input_recovery']
        require_hash(pointer['catalog'], pointer['sha256'])
        recovery = read(pin(pointer['catalog']))
        borrowed = recovery['devices']
    devices, arches = {}, {}
    for condition in conditions:
        record = condition['record']
        grid = f'{record["grid"][0]}x{record["grid"][1]}'
        arch = pin(source/'architectures'/grid/condition['architecture']/'arch.xml')
        meta = read(pin(arch.parent/'architecture.json'))
        require_hash(arch, meta['architecture_sha256'])
        pin(source/'architectures'/grid/'actual_audit.json')
        arches[f'{grid}/{condition["architecture"]}'] = str(arch)
        key = identity(dict(architecture=sha(arch), binary=context['binary_sha256'], width=condition['width']))[:20]
        folder = source/'device_cache'/key
        if not (folder/'manifest.json').exists() and key in borrowed:
            folder = within(borrowed[key]['folder'], REPO)
            require_hash(folder/'manifest.json', borrowed[key]['manifest_sha256'])
        manifest = read(pin(folder/'manifest.json'))
        if (manifest['architecture_sha256'] != sha(arch) or
                manifest['binary_sha256'] != context['binary_sha256'] or
                manifest['channel_width'] != condition['width']):
            raise ValueError('borrowed RRG identity mismatch')
        if not all((folder/name).is_file() for name in ('rr_graph.xml', 'device_graph.json')):
            raise ValueError('raw server RRG/device payloads missing; Git evidence alone cannot run training')
        devices[key] = str(folder)
    for record in records:
        pin(source/'inputs'/record['design']/'source.json')
        pin(source/'calibration'/record['design']/'trials.json')
        pin(source/'calibration'/record['design']/'boundary.json')
    config = copy.deepcopy(config)
    config.pop('input_recovery', None)
    config.update(note='Qualified five-design learning development; OR1200 retained separately. Not the formal480.',
                  designs=[r for r in config['designs'] if r['design'] in {*TRAIN, *VALIDATION}],
                  pilot_families=[], qualified_development=queue_plan())
    catalog = dict(source=str(source), files=files, devices=devices, architectures=arches,
        implementation={name: sha(HERE/name) for name in ('qualified_learning.py', 'run_qualified_learning.sh')},
        reuse_validation_sources=context['source_hashes'],
        backend=str(backend), mapping=str(mapping), binary=str(binary))
    freeze(out/'qualified_source_manifest.json', catalog)
    freeze(out/'campaign.json', config)
    freeze(out/'prepared.json', dict(records=records, complete=True))
    for record in records:
        freeze(out/'inputs'/record['design']/'source.json', read(source/'inputs'/record['design']/'source.json'))
    freeze(out/'conditions.json', dict(records=conditions, censored=['or1200']))
    freeze(out/'boundaries.json', read(source/'boundaries.json'))
    freeze(out/'conditions_manifest.json', dict(conditions_sha256=sha(out/'conditions.json'),
        boundaries_sha256=sha(out/'boundaries.json'), prepared_sha256=sha(out/'prepared.json')))
    freeze(out/'qualified_protocol.json', dict(schema='qualified-development-1',
        catalog_sha256=sha(out/'qualified_source_manifest.json'), plan=queue_plan(),
        no_model_or_method_outcomes_used_to_choose_designs=True,
        selection_seeds=[531, 532], comparison_seeds=list(SEEDS),
        thresholds=dict(runtime_ratio=.95, cpd_ratio=1.03, wirelength_ratio=1.03),
        formal_acceptance_requires_separate_Wmin_and_full_original_panel=True))
    return catalog


class QualifiedCampaign(Campaign):
    def __init__(self, args, catalog):
        self.catalog = catalog
        super().__init__(args)
        self.records()

    def architecture(self, record, condition):
        grid = f'{record["grid"][0]}x{record["grid"][1]}'
        path = Path(self.catalog['architectures'][f'{grid}/{condition}'])
        require_hash(path, self.catalog['files'][str(path)])
        return path

    def rr(self, record, condition, width):
        self.check_binary()
        arch = self.architecture(record, condition)
        key = identity(dict(architecture=sha(arch), binary=self.binary_hash, width=width))[:20]
        if key not in self.catalog['devices']:
            raise ValueError('unregistered RRG requested; this stage never rebuilds devices')
        folder = Path(self.catalog['devices'][key])
        manifest = folder/'manifest.json'
        with self._lock(key):
            require_hash(manifest, self.catalog['files'][str(manifest.resolve())])
            old = read(manifest)
            paths = (folder/'rr_graph.xml', folder/'device_graph.json')
            if key not in self.verified_rr:
                require_hash(paths[0], old['rr_graph_sha256'])
                require_hash(paths[1], old['graph_sha256'])
                self.verified_rr.add(key)
                self._rr_stats_local = getattr(self, '_rr_stats_local', {})
                self._rr_stats_local[key] = tuple(self._stat(p) for p in paths)
            elif self._rr_stats_local[key] != tuple(self._stat(p) for p in paths):
                raise ValueError('borrowed RRG changed during stage')
        return dict(rr_graph=str(paths[0]), device=str(paths[1]), manifest=str(manifest))

    def compare(self):
        """All planned conditions, including failures; cold balanced whole waves."""
        checkpoint = self.selected()
        conditions = sorted(self.conditions(), key=lambda c: c['id'])
        cases = [(c, s) for c in conditions for s in SEEDS]
        cores = self.budget['physical_cores']
        # Equal wave sizes avoid comparing mixtures of different machine load.
        count = max(n for n in range(1, len(cores)+1) if len(cases) % n == 0)
        cores = cores[:count]
        root = self.out/'comparison'
        host = dict(uname=list(os.uname()), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                    scheduler_job=os.environ.get('SLURM_JOB_ID'), cores=cores)
        freeze(root/'plan.json', dict(cases=[(c['id'], s) for c, s in cases],
            methods=list(METHODS), orders=ORDERS, checkpoint_sha256=sha(checkpoint),
            timeout=self.args.timeout, lane_count=count, threads=self.args.threads,
            eligibility='all registered cases, never filtered by learned success', expected=600))
        # Shared integrity checks excluded equally, before any measured method.
        for condition in conditions:
            self.rr(condition['record'], condition['architecture'], condition['width'])
        rows = []
        for start in range(0, len(cases), count):
            self.guard()
            wave = cases[start:start+count]
            folder = root/f'wave_{start//count:04d}'
            signature = dict(cases=[(c['id'], s) for c, s in wave], host=host,
                             plan_sha256=sha(root/'plan.json'))
            complete = folder/'completed.json'
            if complete.exists():
                old = read(complete)
                if old['signature'] != _json_value(signature):
                    raise ValueError('comparison host/cores changed; use a new output cohort, no mixed timing')
                rows.extend(old['rows'])
                continue
            attempt = folder/f'attempt_{len(list(folder.glob("attempt_*")))+1:04d}'
            attempt.mkdir(parents=True)
            barrier = threading.Barrier(count)
            def lane(item):
                index, (condition, seed) = item
                # Linux affinity is per thread. Bind controller-side graph
                # construction as well as its VPR/generator child processes.
                os.sched_setaffinity(0, {cores[index]})
                if os.sched_getaffinity(0) != {cores[index]}:
                    raise RuntimeError('failed to pin the cold comparison lane')
                order = ORDERS[(start+index) % len(ORDERS)]
                result = []
                try:
                    for method in order:
                        self.guard()
                        barrier.wait(timeout=self.args.timeout*4)
                        target = attempt/f'lane_{index:03d}'/method
                        working = dict(condition)
                        begun = time.perf_counter()
                        if method != 'default':
                            record = dict(condition['record'])
                            prefix = target/'cold_structure'
                            arch = self.architecture(record, condition['architecture'])
                            run = run_trial(target/'structure_run',
                                vpr_command(self.binary, arch, record['blif'], condition['width'], 701)+['--device', 'pilot'],
                                inputs=[self.binary_reference, arch, record['blif']],
                                env=dict(TC_GP_STRUCTURE=prefix, TC_GP_STRUCTURE_ONLY=1),
                                marker='TAILCOMPILE_V7_STRUCTURE_ONLY_COMPLETE', affinity=cores[index],
                                timeout=self.args.timeout, prune=True)
                            if run['status'] != 'complete':
                                # Preserve pre-VPR failure in its registered method denominator.
                                row = dict(condition_id=condition['id'], design=condition['design'],
                                    family=condition['family'], architecture=condition['architecture'],
                                    width=condition['width'], width_role=condition['width_role'],
                                    method=method, seed=seed, status=run['status'],
                                    vpr_manifest=None, cpd_ns=None, wirelength=None,
                                    preparation_manifest=str(target/'structure_run/manifest.json'))
                            else:
                                record.update(structure_prefix=str(prefix), structure_export_seconds=0.)
                                working.update(record=record, graph_root=str(target/'cold_graph'))
                                self.graph(working)
                                row = self.execute(working, seed, target/'execution', method=method,
                                    checkpoint=checkpoint if method == 'gnn' else None, affinity=cores[index])
                        else:
                            row = self.execute(working, seed, target/'execution', affinity=cores[index])
                        row.update(total_seconds=time.perf_counter()-begun, measured_cold_total=True,
                                   split=condition['split'], physical_core=cores[index], wave_load=count,
                                   order=list(order), host=host, reusable_RRG_preparation_excluded=True)
                        write(target/'timed_result.json', row)
                        result.append(row)
                        barrier.wait(timeout=self.args.timeout*4)
                except BaseException:
                    barrier.abort()
                    raise
                return result
            with ThreadPoolExecutor(max_workers=count) as pool:
                groups = list(pool.map(lane, enumerate(wave)))
            completed = [r for group in groups for r in group]
            freeze(complete, dict(signature=signature, rows=completed, attempt=str(attempt)))
            rows.extend(completed)
            write(root/'report.json', dict(planned=600, completed=len(rows), rows=rows,
                selection_status=read(self.out/'selection.json')['status']))
            emit('qualified_comparison_progress', completed=len(rows), planned=600)
        value = dict(planned=600, completed=len(rows), rows=rows,
                     selection_status=read(self.out/'selection.json')['status'])
        write(root/'report.json', value)
        return value


def comparison_summary(rows, conditions):
    """Report all failure denominators; QoR only on declared paired successes."""
    expected = {(c['id'], s, m) for c in conditions for s in SEEDS for m in METHODS}
    actual = [(r['condition_id'], r['seed'], r['method']) for r in rows]
    if len(actual) != len(set(actual)) or not set(actual) <= expected:
        raise ValueError('duplicate or unregistered comparison row')
    lookup = dict(zip(actual, rows))
    strata = []
    for condition, method, baseline in itertools.product(conditions, ('capacity_only', 'gnn'), ('default', 'capacity_only')):
        if method == baseline:
            continue
        pairs = [(lookup.get((condition['id'], s, method)), lookup.get((condition['id'], s, baseline))) for s in SEEDS]
        present = [(a, b) for a, b in pairs if a is not None and b is not None]
        success = [(a, b) for a, b in present if a['status'] == b['status'] == 'success']
        def ratio(key):
            values = [a.get(key, 0)/b[key] for a, b in success if
                      isinstance(a.get(key), (float, int)) and isinstance(b.get(key), (float, int)) and
                      math.isfinite(a[key]) and math.isfinite(b[key]) and a[key] > 0 and b[key] > 0]
            return math.exp(sum(math.log(v) for v in values)/len(values)) if values and len(values) == len(success) else None
        strata.append(dict(design=condition['design'], family=condition['family'], split=condition['split'],
            architecture=condition['architecture'], width_role=condition['width_role'], method=method, baseline=baseline,
            planned=5, observed_pairs=len(present), jointly_successful=len(success),
            method_successes=sum(a['status'] == 'success' for a, b in present),
            baseline_successes=sum(b['status'] == 'success' for a, b in present),
            runtime_ratio=ratio('total_seconds'), cpd_ratio=ratio('cpd_ns'), wirelength_ratio=ratio('wirelength'),
            missing_qor_reason=None if success else 'no jointly successful pair; not zero or a win'))
    split_reports = []
    for split, method, baseline in itertools.product(('train', 'validation'), ('capacity_only', 'gnn'), ('default', 'capacity_only')):
        if method == baseline:
            continue
        group = [r for r in strata if (r['split'], r['method'], r['baseline']) == (split, method, baseline)]
        refs = [r for r in group if r['width_role'] == 'reference']
        def mean(key):
            values = [r[key] for r in refs]
            return math.exp(sum(math.log(v) for v in values)/len(values)) if values and all(v is not None for v in values) else None
        runtime, cpd, wire = (mean(k) for k in ('runtime_ratio', 'cpd_ratio', 'wirelength_ratio'))
        fully_observed = all(r['observed_pairs'] == 5 for r in group)
        success_ok = fully_observed and all(r['method_successes'] >= r['baseline_successes'] for r in group)
        paired_refs = all(r['jointly_successful'] == 5 for r in refs)
        quality_ok = paired_refs and all(r['cpd_ratio'] is not None and r['cpd_ratio'] <= 1.03 and
                                         r['wirelength_ratio'] is not None and r['wirelength_ratio'] <= 1.03 for r in refs)
        split_reports.append(dict(split=split, method=method, baseline=baseline,
            reference_runtime_ratio=runtime, reference_cpd_ratio=cpd, reference_wirelength_ratio=wire,
            success_protection=success_ok, reference_quality_protection=quality_ok,
            provisional_runtime_target=bool(success_ok and quality_ok and runtime is not None and runtime <= .95)))
    unknown = sum(r['status'] not in ('success', 'unroutable', 'timeout', 'decode_failure',
                                     'resource_budget_exceeded', 'packing_or_placement_failure') for r in rows)
    if unknown:
        for group in split_reports:
            group['provisional_runtime_target'] = False
    learned_validation = [g for g in split_reports if g['split'] == 'validation' and g['method'] == 'gnn']
    status = ('incomplete_comparison' if set(actual) != expected else
              'infrastructure_requires_diagnosis' if unknown else
              'provisional_active_validation_targets_met' if all(g['provisional_runtime_target'] for g in learned_validation) else
              'learning_target_not_demonstrated')
    return dict(scope='five-design development; validation already used for checkpoint selection',
        status=status,
        planned=len(expected), completed=len(rows), complete=set(actual) == expected,
        actual_vpr_executions=sum(r.get('vpr_manifest') is not None for r in rows),
        status_counts=dict(Counter(r['status'] for r in rows)), unknown_failure_count=unknown,
        denominators_include_failed_methods=True, strata=strata, split_comparisons=split_reports,
        runtime_population='reference width; all five paired successes required for provisional target',
        Wmin_measured=False, formal_480_completed=False, full_acceptance_passed=False,
        generalization_claim=False)


def main():
    install_signal_handlers()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=STAGES)
    parser.add_argument('--source', type=Path, default=HERE/'runs/development_004_capacity')
    parser.add_argument('--out', type=Path, default=HERE/'runs/development_005_qualified_learning')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--per-job-mib', type=int, default=4096)
    parser.add_argument('--timeout', type=int, default=3600)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    if min(args.jobs, args.per_job_mib, args.timeout) < 1 or args.threads != 1:
        parser.error('positive resources and exactly one thread per VPR/generator required')
    if not re.fullmatch(r'cpu|cuda(?::[0-9]+)?', args.device):
        parser.error('device must be cpu, cuda, or cuda:N')
    source, out = validate_paths(args.source, args.out)
    scratch = within(REPO/'vtr_work/tailcompile_v7_3d/qualified_learning', REPO)
    validate_output_roots((scratch,))
    with campaign_lock(out, 'qualified-'+args.stage):
        catalog = prepare_handoff(source, out)
        freeze(out/'execution_protocol.json', dict(timeout=args.timeout, threads=args.threads))
        runtime = SimpleNamespace(**{k: v for k, v in vars(args).items() if k not in ('source', 'out')},
            out=out, config=out/'campaign.json', workspace=REPO.parent, scratch=scratch,
            vtr_root=Path(catalog['backend']), mapping_vtr_root=Path(catalog['mapping']),
            binary=Path(catalog['binary']), backend_manifest=Path(catalog['backend'])/'backend_manifest.json',
            qualification_report=source/'qualification/qualification_report.json',
            rrg_jobs=1, rrg_mib=args.per_job_mib, allow_local=False)
        campaign = QualifiedCampaign(runtime, catalog)
        campaign.gate()
        if args.stage == 'plan':
            write(out/'queue_plan.json', queue_plan())
            emit('qualified_queue_plan', **queue_plan())
        elif args.stage in ('pretrain', 'finetune'):
            campaign.train(feedback=args.stage == 'finetune')
        elif args.stage == 'report':
            rows = read(out/'comparison/report.json')['rows']
            report = comparison_summary(rows, campaign.conditions())
            report['selection_status'] = read(out/'selection.json')['status']
            write(out/'development_report.json', report)
            emit('qualified_development_report', **{k: v for k, v in report.items() if k != 'strata'})
        else:
            getattr(campaign, args.stage)()
        emit('qualified_stage_complete', stage=args.stage, out=str(out))


if __name__ == '__main__':
    main()
