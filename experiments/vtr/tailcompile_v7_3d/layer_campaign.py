"""007 explicit-layer experiment: immutable inputs, bounded stages, cold timing."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import math
import os
import random
import shutil
from pathlib import Path
import sys
import threading
import time

from campaign import campaign_lock, identity, physical_cores, resource_budget
from io_utils import read, write, freeze, sha, run_trial, vpr_command, prune_derived, install_signal_handlers
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
METHODS = ('B', 'D-I', 'L-I', 'T-I', 'D-S', 'L-S', 'T-S')
STAGES = ('plan', 'check', 'prepare', 'pretrain', 'feedback', 'train', 'validate', 'compare', 'report')
CODE = ('layer_campaign.py', 'layer_model.py', 'train_layers.py', 'vertical_risk.py',
        'layer_features.py', 'install_layer_hook.py', 'layer_prior_hook.inc', 'layer_route_hook.inc', 'run_layer_campaign.sh',
        'model.py', 'data.py', 'fast_graph.py', 'generation.py', 'io_utils.py', 'train.py', 'device_graph.py',
        'layer_observations.py',
        'architectures.py', 'campaign.py', 'qualified_learning.py', 'repository_guard.py', 'install_hook.py',
        'supervision.py', '../tailcompile_v68/model.py', '../tailcompile_v68/data.py',
        '../tailcompile_v68/install_hook.py', '../tailcompile_v69/phase_metrics.py',
        '../tailrisk_core.py', '../tailcompile_v2.py')


def queue_plan():
    return dict(schema='explicit-layer-007-1', policy='mean-positive-20261031',
        train_families=['spmv', 'mcml'], active_validation_families=['blob_merge', 'diffeq'],
        methods=list(METHODS), primary_comparison=['T-S', 'B'], primary_width_role='reference',
        architectures=['H0Z0', 'H0Z1', 'H1Z0', 'H1Z1'], width_roles=['reference', 'stress'],
        feedback_seeds=[2721, 2722], validation_seeds=[2741, 2742], comparison_seeds=[2761, 2762],
        training_seed=2801, shared_epochs=20, candidate_epochs=5, feedback_epochs=10,
        checkpoint_epochs=[5, 10], tail_weight=.1, feedback_runs_upper=160,
        validation_runs_upper=80, comparison_runs=224, local_qualification_upper=12,
        research_runs_upper=464, total_runs_upper=476, teacher_recollection=0,
        current_stage='development_attribution', Wmin_measured=False, generalization_claim=False,
        raw_negative_outcomes_retained=True, per_case_qor_veto=False,
        all_hosts_pnr_upper=488, local_engineering_pnr_upper=12,
        normalization='capacity fraction, normalized-XY weighted mean distance, weighted cut mean; Tail scale=train teacher positive-CVaR median fixed before paired training',
        timing='cold total including structure, graph, model, decode, lowering and VPR; device preprocessing separate')


def mean(values):
    return math.exp(sum(math.log(v) for v in values)/len(values)) if values else None


def evaluate(rows, *, methods=METHODS, expected=None):
    """Family-equal paired estimates; failures never become a runtime win."""
    keys = [(r['condition_id'], r['seed'], r['method']) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError('duplicate measured task')
    by = dict(zip(keys, rows))
    contrasts = [('T-S', 'B'), ('T-I', 'B'), ('L-I', 'D-I'), ('T-I', 'L-I'),
                 ('D-S', 'D-I'), ('L-S', 'L-I'), ('T-S', 'T-I'), ('L-S', 'D-S'), ('T-S', 'L-S')]
    reports = []
    for method, baseline in contrasts:
        if method not in methods or baseline not in methods:
            continue
        for role in ('reference', 'stress'):
            selected = [r for r in rows if r['method'] == method and r['width_role'] == role]
            strata = defaultdict(list)
            failures = Counter()
            paired = []
            for a in selected:
                b = by.get((a['condition_id'], a['seed'], baseline))
                if b is None:
                    failures['missing_control'] += 1
                    continue
                failures['method_'+a['status']] += 1
                failures['baseline_'+b['status']] += 1
                if a['status'] != 'success' or b['status'] != 'success':
                    continue
                record = dict(condition_id=a['condition_id'], family=a['family'], seed=a['seed'])
                for metric in ('total_seconds', 'cpd_ns', 'wirelength'):
                    x, y = a.get(metric), b.get(metric)
                    ratio = x/y if (isinstance(x, (int, float)) and isinstance(y, (int, float))
                                     and math.isfinite(x) and math.isfinite(y) and x > 0 and y > 0) else None
                    record[metric+'_ratio'] = ratio
                paired.append(record)
                strata[(a['family'], a['architecture'], a['condition_id'])].append(record)
            aggregate = {}
            for metric in ('total_seconds', 'cpd_ns', 'wirelength'):
                family = defaultdict(list)
                for (fam, arch, cid), group in strata.items():
                    values = [v[metric+'_ratio'] for v in group]
                    if all(v is not None for v in values):
                        family[fam].append(mean(values))
                family_values = {f: mean(v) for f, v in family.items()}
                ratio = mean(list(family_values.values()))
                bootstrap = []
                if family_values:
                    rng = random.Random(2801)
                    values = list(family_values.values())
                    bootstrap = sorted(mean([rng.choice(values) for _ in values]) for _ in range(1024))
                interval = [1-bootstrap[998], 1-bootstrap[25]] if bootstrap else None
                aggregate[metric] = dict(ratio=ratio, gain=1-ratio if ratio is not None else None,
                    family_ratios=family_values, family_count=len(family_values),
                    descriptive_family_bootstrap_gain95=interval, bootstrap_seed=2801,
                    small_family_interval_is_not_generalization_proof=True,
                    average_positive=(ratio < 1 if ratio is not None else None))
            complete_success = bool(selected) and len(paired) == len(selected)
            reports.append(dict(method=method, baseline=baseline, width_role=role,
                observed_method_tasks=len(selected), jointly_successful=len(paired), status_counts=dict(failures),
                estimate_population='all paired tasks' if complete_success else 'conditional common-success subset',
                full_population_runtime_gain_claim=complete_success and (set(keys) == set(expected) if expected is not None else False)
                    and bool(aggregate['total_seconds']['average_positive']),
                metrics=aggregate, pairs=paired,
                runtime_wins=sum(r['total_seconds_ratio'] < 1 for r in paired if r['total_seconds_ratio'] is not None),
                runtime_losses=sum(r['total_seconds_ratio'] > 1 for r in paired if r['total_seconds_ratio'] is not None)))
    present = set(keys)
    return dict(schema='explicit-layer-007-report-1', policy='mean-positive-20261031',
        complete=(present == set(expected) if expected is not None else None), completed=len(rows),
        missing=sorted(set(expected)-present) if expected is not None else None,
        unexpected=sorted(present-set(expected)) if expected is not None else None,
        status_counts=dict(Counter(r['status'] for r in rows)), comparisons=reports,
        primary_comparison=['T-S', 'B', 'reference'], Wmin_measured=False,
        generalization_claim=False, five_percent_is_stretch_target=True)


def choose_epoch(rows, epochs=(5, 10)):
    lookup = {(r['epoch'], r['method'], r['condition_id'], r['seed']): r for r in rows}
    conditions = {(r['condition_id'], r['seed']) for r in rows if r['method'] == 'B'}
    expected = {(e, m, c, s) for e in epochs for m in ('L', 'T') for c, s in conditions}
    if not conditions or not expected <= lookup.keys():
        raise ValueError('checkpoint selection incomplete; missing is not a loss')
    common = [(c, s) for c, s in conditions if all(lookup[e, m, c, s]['status'] == 'success'
              for e in epochs for m in ('L', 'T')) and lookup[0, 'B', c, s]['status'] == 'success']
    scores = []
    for e in epochs:
        ratios = defaultdict(list)
        for c, s in common:
            base = lookup[0, 'B', c, s]['total_seconds']
            for m in ('L', 'T'):
                row = lookup[e, m, c, s]
                ratios[row['family']].append(row['total_seconds']/base)
        score = mean([mean(v) for v in ratios.values()])
        successes = sum(lookup[e, m, c, s]['status'] == 'success' for m in ('L', 'T') for c, s in conditions)
        scores.append(dict(epoch=e, successes=successes, common_success_ratio=score))
    best = min(scores, key=lambda r: (-r['successes'], r['common_success_ratio'] or float('inf'), r['epoch']))
    if not common:
        best = scores[0]
    return dict(epoch=best['epoch'], scores=scores, common_success_conditions=len(common),
        performance_selection_completed=bool(common), selection='shared epoch; success count then family-equal cold time',
        no_common_success_fallback='earlier checkpoint' if not common else None)


def owned(out):
    out = Path(out).resolve()
    runs = HERE/'runs'
    if runs not in out.parents or not out.relative_to(runs).parts[0].startswith('development_007'):
        raise ValueError('new layer output requires a separate 007 cohort inside v7 runs')
    validate_output_roots((out,))
    return out


def panel(source):
    from qualified_learning import qualified_panel
    _, _, conditions = qualified_panel(source)
    selected = [c for c in conditions if c['family'] in ('spmv', 'mcml') or
                c['family'] in ('blob_merge', 'diffeq') and c['width_role'] == 'reference']
    if len(selected) != 24:
        raise ValueError('007 needs sixteen train and eight active validation conditions')
    return selected


def check(args):
    """Read-only payload check: Git light evidence alone does not satisfy it."""
    conditions = panel(args.source)
    missing = []
    for name in ('backend_manifest.json',):
        if not (args.backend/name).is_file():
            missing.append(str(args.backend/name))
    for name in ('teacher_records.json',):
        if not (args.prior/name).is_file():
            missing.append(str(args.prior/name))
    if (args.prior/'teacher_records.json').is_file():
        teachers = {r['condition_id']: r for r in read(args.prior/'teacher_records.json')['records']}
        for c in conditions:
            for name in ('graph', 'teacher'):
                path = Path(teachers[c['id']][name])
                if not path.is_file():
                    missing.append(str(path))
    catalog_path = args.prior/'qualified_source_manifest.json'
    if catalog_path.is_file():
        catalog = read(catalog_path)
        old_binary_hash = read(args.source/'binary_identity.json')['sha256']
        for c in conditions:
            grid = c['record']['grid']
            arch = Path(catalog['architectures'][f'{grid[0]}x{grid[1]}/{c["architecture"]}'])
            if not arch.is_file():
                missing.append(str(arch))
                continue
            key = identity(dict(architecture=sha(arch), binary=old_binary_hash, width=c['width']))[:20]
            folder = Path(catalog['devices'][key])
            missing.extend(str(folder/n) for n in ('rr_graph.xml', 'device_graph.json', 'manifest.json') if not (folder/n).is_file())
    else:
        missing.append(str(catalog_path))
    for c in conditions:
        r = c['record']
        for path in [Path(r['blif'])] + [Path(r['structure_prefix']+s) for s in ('.blocks.tsv', '.nets.tsv', '.meta.tsv')]:
            if not path.is_file():
                missing.append(str(path))
    return dict(ready=not missing, missing=sorted(set(missing)), conditions=len(conditions), plan=queue_plan())


def worker(request, out):
    """Fresh interpreter reconstructs logical graph, loads model, lowers XYZ."""
    import torch
    from data import build_logic
    from fast_graph import make_graph
    from layer_features import augment_graph
    from generation import lower_initialization
    from train_layers import load_model
    from layer_model import generate
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    validate_output_roots((out,))
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    r = request
    logical = build_logic(r['prefix'], r['design'], r['blif'], r['architecture'], family=r['family'], split=r['split'])
    device = read(r['device'])
    graph = augment_graph(make_graph(logical, device), logical, device, read(r['vertical_summary']), r['blif'], r['prefix'])
    torch.save(graph, out/'graph.pt')
    write(out/'logic.json', logical)
    result = dict(graph=str(out/'graph.pt'), logic=str(out/'logic.json'), graph_sha256=sha(out/'graph.pt'),
                  logic_sha256=sha(out/'logic.json'), graph_seconds=time.perf_counter()-started)
    if r.get('checkpoint'):
        model = load_model(r['checkpoint'], device='cpu')
        from model import CapacityError
        try:
            with torch.no_grad():
                prediction = generate(model, graph, method=r['method'], seed=r.get('generation_seed', 2801),
                                      score_noise_std=r.get('score_noise_std', 0.))
        except CapacityError as exc:
            result.update(status='decode_failure', reason=str(exc), packing_infeasibility_claimed=False)
            write(out/'generation_report.json', result)
            print('TAILCOMPILE_LAYER_WORKER_COMPLETE', flush=True)
            return result
        lowered = lower_initialization(logical, device, prediction, out/'initialization', method=r['method'])
        result.update(lowered)
        result['soft_vertical_risk'] = prediction.get('soft_vertical_risk')
        result['hard_vertical_risk'] = prediction.get('hard_vertical_risk')
    result['worker_internal_seconds'] = time.perf_counter()-started
    write(out/'generation_report.json', result)
    print('TAILCOMPILE_LAYER_WORKER_COMPLETE', flush=True)
    return result


class LayerCampaign:
    def __init__(self, args):
        self.args, self.out = args, owned(args.out)
        self.conditions = panel(args.source)
        self.train = [c for c in self.conditions if c['split'] == 'train']
        self.validation = [c for c in self.conditions if c['split'] == 'validation']
        self.budget = resource_budget(args.jobs, args.per_job_mib)
        self.binary = None
        self._devices = {}

    def setup(self):
        from qualified_learning import prepare_handoff
        from install_layer_hook import verify
        self.backend_verification = verify(self.args.backend)
        manifest = read(self.args.backend/'backend_manifest.json')
        self.binary = Path(manifest['binary'])
        if sha(self.binary) != manifest['binary_sha256']:
            raise ValueError('layer executable changed')
        stat = self.binary.stat()
        self.binary_stat = (stat.st_size, stat.st_mtime_ns, stat.st_ino)
        qualification = read(self.args.qualification)
        if not qualification.get('passed') or qualification.get('binary_sha256') != sha(self.binary):
            raise ValueError('new layer binary needs its own successful qualification')
        self.catalog = prepare_handoff(self.args.source, self.out/'borrowed')
        self.old_binary_hash = sha(self.catalog['binary'])
        self.teacher_rows = {r['condition_id']: r for r in read(self.args.prior/'teacher_records.json')['records']}
        files = {str(self.args.prior/'teacher_records.json'): sha(self.args.prior/'teacher_records.json')}
        for c in self.conditions:
            r = self.teacher_rows[c['id']]
            if r['family'] != c['family'] or r['split'] != c['split']:
                raise ValueError('teacher family/split mismatch')
            for k in ('graph', 'teacher'):
                files[r[k]] = sha(r[k])
        freeze(self.out/'layer_protocol.json', dict(plan=queue_plan(), source=str(self.args.source),
            prior=str(self.args.prior), backend=str(self.args.backend), binary_sha256=sha(self.binary),
            backend_manifest_sha256=sha(self.args.backend/'backend_manifest.json'),
            qualification_sha256=sha(self.args.qualification), teacher_files=files,
            implementation={n: sha(HERE/n) for n in CODE}, timeout=self.args.timeout))
        freeze(self.out/'queue_plan.json', queue_plan())
        self.binary_reference = self.out/'binary_identity.json'
        freeze(self.binary_reference, dict(path=str(self.binary), sha256=sha(self.binary)))
        freeze(self.out/'execution'/self.args.stage/'protocol.json', dict(device=self.args.device,
            jobs=self.args.jobs, per_job_mib=self.args.per_job_mib, timeout=self.args.timeout,
            physical_cores=self.budget['physical_cores'], effective_lanes=self.budget['effective']))

    def guard(self):
        if shutil.disk_usage(self.out).free < 8*1024**3:
            raise OSError('less than 8 GiB free; stop dispatch and retain completed evidence')
        stat = self.binary.stat()
        if (stat.st_size, stat.st_mtime_ns, stat.st_ino) != self.binary_stat:
            raise ValueError('frozen layer executable changed during dispatch')

    def device(self, c):
        if c['id'] in self._devices:
            return self._devices[c['id']]
        arch = Path(self.catalog['architectures'][f'{c["record"]["grid"][0]}x{c["record"]["grid"][1]}/{c["architecture"]}'])
        key = identity(dict(architecture=sha(arch), binary=self.old_binary_hash, width=c['width']))[:20]
        folder = Path(self.catalog['devices'][key])
        value = dict(architecture=str(arch), rr_graph=str(folder/'rr_graph.xml'), device=str(folder/'device_graph.json'),
                    device_manifest=str(folder/'manifest.json'),
                    vertical_summary=str(self.out/'vertical_devices'/key/'summary.json'))
        self._devices[c['id']] = value
        return value

    def request(self, c, checkpoint=None, method='L', prefix=None, noise=0.):
        return dict(self.device(c), prefix=prefix or c['record']['structure_prefix'], design=c['design'],
            family=c['family'], split=c['split'], blif=c['record']['blif'], checkpoint=str(checkpoint) if checkpoint else None,
            method=method, generation_seed=2801, score_noise_std=noise)

    def generate(self, c, target, checkpoint=None, method='L', prefix=None, noise=0., core=None):
        self.guard()
        request = self.request(c, checkpoint, method, prefix, noise)
        freeze(target/'request.json', request)
        trial = run_trial(target/'worker', [sys.executable, HERE/'layer_campaign.py', 'worker',
            '--request', target/'request.json', '--out', target/'payload'],
            inputs=[target/'request.json'] + [HERE/n for n in CODE], timeout=self.args.timeout, affinity=core,
            marker='TAILCOMPILE_LAYER_WORKER_COMPLETE', prune=False)
        if trial['status'] != 'complete':
            raise RuntimeError('layer graph/generator worker failed (not a routability label): '+trial['log'])
        value = read(target/'payload/generation_report.json')
        value['generation_process_seconds'] = trial['metrics']['wall_seconds']
        return value

    def prepare(self):
        from vertical_risk import prepare_device
        done = set()
        for c in self.conditions:
            dev = self.device(c)
            target = Path(dev['vertical_summary'])
            if target not in done:
                begun = time.perf_counter()
                summary = prepare_device(dev['rr_graph'], read(dev['device']), out=target,
                                         max_seconds=min(900, self.args.timeout))
                device_manifest = read(dev['device_manifest'])
                if summary['signature']['rr_graph_sha256'] != device_manifest['rr_graph_sha256']:
                    raise ValueError('raw RRG differs from the immutable qualified device')
                if sha(dev['device']) != device_manifest['graph_sha256']:
                    raise ValueError('raw device graph differs from the qualified manifest')
                stat = Path(dev['rr_graph']).stat()
                cost = target.parent/'device_preparation_report.json'
                if cost.exists():
                    previous = read(cost)
                    if previous['summary_sha256'] != sha(target) or previous['rr_stat'] != [stat.st_size, stat.st_mtime_ns, stat.st_ino]:
                        raise ValueError('prepared device dependency changed')
                    write(target.parent/'cache_checks'/f'{time.time_ns()}_report.json',
                          dict(seconds=time.perf_counter()-begun, first_cost_report=str(cost)))
                else:
                    write(cost, dict(seconds=time.perf_counter()-begun,
                        reusable_device_cost=True, summary_sha256=sha(target), rr_graph_sha256=device_manifest['rr_graph_sha256'],
                        rr_stat=[stat.st_size, stat.st_mtime_ns, stat.st_ino]))
                done.add(target)
        rows = []
        for c in self.conditions:
            generated = self.generate(c, self.out/'graphs'/c['id'])
            if generated.get('status'):
                raise RuntimeError('graph preparation failed: '+str(generated))
            teacher = self.teacher_rows[c['id']]
            rows.append(dict(condition_id=c['id'], graph=generated['graph'], teacher=teacher['teacher'],
                             family=c['family'], split=c['split']))
        freeze(self.out/'teacher_records.json', dict(records=rows))

    def pretrain(self):
        from train_layers import train_shared, prepare_arms
        records = [r for r in read(self.out/'teacher_records.json')['records'] if r['split'] == 'train']
        if (self.out/'shared_report.json').exists():
            shared = read(self.out/'shared_report.json')
        else:
            folder = self.fresh_attempt(self.out/'models/shared')
            shared = train_shared(records, folder, epochs=20, seed=2801, device=self.args.device)
            freeze(self.out/'shared_report.json', shared)
        if not (self.out/'prepared_models_report.json').exists():
            prepared = prepare_arms(shared['checkpoint'], records, self.fresh_attempt(self.out/'models/prepared'), epochs=5,
                                    seed=2801, device=self.args.device)
            freeze(self.out/'prepared_models_report.json', prepared)

    @staticmethod
    def fresh_attempt(folder):
        folder.mkdir(parents=True, exist_ok=True)
        target = folder/f'attempt_{len(list(folder.glob("attempt_*")))+1:04d}'
        target.mkdir()
        return target

    def pnr(self, c, seed, target, method, generated=None, core=None):
        self.guard()
        dev = self.device(c)
        frozen_device = read(Path(dev['vertical_summary']).parent/'device_preparation_report.json')
        stat = Path(dev['rr_graph']).stat()
        if frozen_device['rr_stat'] != [stat.st_size, stat.st_mtime_ns, stat.st_ino]:
            raise ValueError('frozen RRG changed after one-time payload verification')
        env = dict(TC_GP_LAYER_PRIOR='1' if method.endswith('-S') else '0', TC_GP_EXPORT=target/'post_gp')
        env['TC_GP_Z_USAGE'] = str(target/'vertical_usage')
        # The immutable device manifest pins the large RRG; verify the raw
        # payload once at preparation, not by rereading gigabytes per timed run.
        inputs = [self.binary_reference, dev['architecture'], c['record']['blif'], dev['device_manifest']]
        if generated is not None:
            env.update(TC_GP_INIT=generated['ap_init'])
            inputs.append(generated['ap_init'])
        result = run_trial(target/'vpr', vpr_command(self.binary, dev['architecture'], c['record']['blif'], c['width'], seed,
            rr_graph=dev['rr_graph'])+['--device', 'pilot', '--write_flat_place', 'post_route.fplace',
            '--write_legalized_flat_place', 'post_legalizer.fplace', '--flat_place_verbosity', '0'], inputs=inputs, env=env, affinity=core,
            timeout=self.args.timeout, prune=False)
        if (target/'metrics.json').exists():
            # Completed native outputs may already have been pruned. Preserve
            # their durable observations rather than replace them by "missing".
            cached = read(target/'metrics.json')
            if (cached['condition_id'], cached['seed'], cached['method']) != (c['id'], seed, method):
                raise ValueError('cached native result identity changed')
            return cached
        status = result['status']
        text = Path(result['log']).read_text(errors='replace')
        if generated and status in ('success', 'unroutable') and ('TAILCOMPILE_V7_GP_INIT_CONSUMED' not in text or
            'TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' not in text):
            status = 'interface_failure'
        row = dict(condition_id=c['id'], design=c['design'], family=c['family'], split=c['split'],
            architecture=c['architecture'], width=c['width'], width_role=c['width_role'], method=method, seed=seed,
            status=status, vpr_seconds=result['metrics']['wall_seconds'], cpd_ns=result['metrics']['cpd_ns'],
            wirelength=result['metrics']['wirelength'], phases=result['metrics']['phases'],
            unique_chanz_nodes=result['metrics']['unique_chanz_nodes'], vpr_manifest=str(target/'vpr/manifest.json'),
            ap_init_sha256=generated.get('ap_init_sha256') if generated else None)
        from vertical_risk import route_pool_summary
        row['vertical_routing'] = {stage: route_pool_summary(read(dev['vertical_summary']), path)
            for stage in ('first', 'last') if (path := target/f'vertical_usage.{stage}.tsv').is_file()}
        row['last_vertical_snapshot_semantics'] = 'last completed routing iteration; final best-route restoration may differ'
        from layer_observations import collect_layers
        row['layer_observations'] = collect_layers(generated['ap_init'] if generated else None, target/'post_gp',
            final_fplace=Path(result['workdir'])/'post_route.fplace',
            legalized_fplace=Path(result['workdir'])/'post_legalizer.fplace')
        write(target/'metrics.json', row)
        if status in ('success', 'unroutable', 'complete'):
            prune_derived(result['workdir'], protected=inputs)
            traces = list(target.glob('post_gp.*')) + [Path(result['workdir'])/name for name in ('post_route.fplace', 'post_legalizer.fplace')]
            removed = []
            for trace in traces:
                if trace.is_file() and not trace.is_symlink() and target.resolve() in trace.resolve().parents:
                    removed.append(dict(path=str(trace), sha256=sha(trace), bytes=trace.stat().st_size))
                    trace.unlink()
            write(target/'trace_pruned_report.json', dict(removed=removed, layer_observations_durable=True))
        return row

    def feedback(self):
        prepared = read(self.out/'prepared_models_report.json')
        shared = read(self.out/'shared_report.json')['checkpoint']
        actions = []
        cores = self.budget['physical_cores'][:self.budget['effective']]
        for c in self.train:
            self.device(c)
        def condition_job(item):
            i, c = item
            core = cores[i]
            os.sched_setaffinity(0, {core})
            pool, seen = [], {}
            sources = [('D', shared, 'D', 0.), ('L', prepared['arms']['L']['checkpoint'], 'L', 0.),
                       ('T', prepared['arms']['T']['checkpoint'], 'T', 0.),
                       ('sample', prepared['arms']['L']['checkpoint'], 'L', .25)]
            for name, checkpoint, method, noise in sources:
                folder = self.out/'feedback'/c['id']/name
                generated = self.generate(c, folder/'generation', checkpoint, method, noise=noise, core=core)
                if not generated.get('ap_init'):
                    pool.append(dict(candidate_id=name, condition_id=c['id'], action=None, trials=[], generation=generated))
                    continue
                digest = generated['ap_init_sha256']
                if digest in seen:
                    write(folder/'duplicate_report.json', dict(alias_of=seen[digest], ap_init_sha256=digest))
                    continue
                seen[digest] = name
                trials = [self.pnr(c, s, folder/f's{s}', name+'-I', generated, core) for s in (2721, 2722)]
                pool.append(dict(candidate_id=name, condition_id=c['id'], action=generated['target'], trials=trials,
                                 generation=generated))
            controls = [self.pnr(c, s, self.out/'feedback'/c['id']/'B'/f's{s}', 'B', core=core) for s in (2721, 2722)]
            for action in pool:
                action['default_trials'] = controls
            return pool
        for start in range(0, len(self.train), len(cores)):
            wave = self.train[start:start+len(cores)]
            with ThreadPoolExecutor(max_workers=len(cores)) as executor:
                groups = list(executor.map(condition_job, enumerate(wave)))
            actions.extend(a for group in groups for a in group)
            write(self.out/'feedback_progress_report.json', dict(completed_conditions=min(start+len(cores), len(self.train)), actions=len(actions)))
        write(self.out/'feedback.json', actions)
        write(self.out/'feedback_report.json', dict(actions=len(actions), planned_pnr_upper=160,
            measured_pnr=sum(len(r['trials']) for r in actions)+2*len(self.train),
            successful=sum(t['status'] == 'success' for a in actions for t in a['trials'])))

    def train_models(self):
        if (self.out/'trained_models_report.json').exists():
            return
        from train_layers import train_feedback
        rows = [r for r in read(self.out/'teacher_records.json')['records'] if r['split'] == 'train']
        result = train_feedback(rows, read(self.out/'prepared_models_report.json'), self.fresh_attempt(self.out/'models/feedback'),
            read(self.out/'feedback.json'), epochs=10, seed=2801, device=self.args.device)
        freeze(self.out/'trained_models_report.json', result)

    def cold(self, c, seed, method, checkpoint, target, core=None):
        saved = target/'timed_report.json'
        if saved.exists():
            return read(saved)
        target.mkdir(parents=True, exist_ok=True)
        begun = time.perf_counter()
        generated = None
        structure_seconds = 0.
        if method != 'B':
            prefix = target/'cold_structure'
            dev = self.device(c)
            export = run_trial(target/'structure', vpr_command(self.binary, dev['architecture'], c['record']['blif'], c['width'], 2701)
                +['--device', 'pilot'], inputs=[self.binary_reference, dev['architecture'], c['record']['blif']],
                env=dict(TC_GP_STRUCTURE=prefix, TC_GP_STRUCTURE_ONLY=1), affinity=core, timeout=self.args.timeout,
                marker='TAILCOMPILE_V7_STRUCTURE_ONLY_COMPLETE')
            structure_seconds = export['metrics']['wall_seconds']
            if export['status'] != 'complete':
                generated = dict(status=export['status'])
            else:
                generated = self.generate(c, target/'generation', checkpoint, method[0], prefix=str(prefix), core=core)
        if generated is not None and not generated.get('ap_init'):
            row = dict(condition_id=c['id'], design=c['design'], family=c['family'], split=c['split'], architecture=c['architecture'],
                width=c['width'], width_role=c['width_role'], method=method, seed=seed,
                status=generated.get('status', 'decode_failure'), cpd_ns=None, wirelength=None, vpr_manifest=None)
        else:
            row = self.pnr(c, seed, target, method, generated, core)
        row.update(total_seconds=time.perf_counter()-begun, physical_core=core, cold_total=True)
        row['structure_export_seconds'] = structure_seconds
        row['generation_process_seconds'] = generated.get('generation_process_seconds') if generated else 0.
        row['graph_seconds_within_generator'] = generated.get('graph_seconds') if generated else 0.
        write(saved, row)
        # Only this completed request's duplicate graph/snapshot is disposable.
        removed = []
        for p in [target/'generation/payload/graph.pt', target/'generation/payload/logic.json'] + list(target.glob('cold_structure.*')):
            if p.is_file() and not p.is_symlink() and target.resolve() in p.resolve().parents:
                removed.append(dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size))
                p.unlink()
        write(target/'payload_pruned.json', dict(files=removed, removed_bytes=sum(r['bytes'] for r in removed)))
        return row

    def validate(self):
        trained = read(self.out/'trained_models_report.json')
        arms = trained['arms']
        checkpoints = {m: arms[m]['checkpoints'] for m in ('L', 'T')}
        epochs = (5, 10) if len(checkpoints['L']) == 2 else (0,)
        core = self.budget['physical_cores'][0]
        freeze(self.out/'validation/timing_protocol.json', dict(core=core, sequential=True,
            uname=list(os.uname()), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            checkpoints={m: {str(p): sha(p) for p in paths} for m, paths in checkpoints.items()}))
        rows = []
        for c, s in itertools.product(self.validation, (2741, 2742)):
            b = self.cold(c, s, 'B', None, self.out/'validation'/c['id']/f's{s}'/'B', core)
            rows.append(dict(b, epoch=0))
            for index, e in enumerate(epochs):
                for m in ('L', 'T'):
                    row = self.cold(c, s, m+'-I', checkpoints[m][index], self.out/'validation'/c['id']/f's{s}'/f'{m}_e{e}', core)
                    rows.append(dict(row, method=m, epoch=e))
        selection = choose_epoch(rows, epochs)
        index = list(epochs).index(selection['epoch'])
        selection['checkpoints'] = {m: checkpoints[m][index] for m in ('L', 'T')}
        selection['checkpoint_sha256'] = {m: sha(p) for m, p in selection['checkpoints'].items()}
        freeze(self.out/'selection.json', selection)
        write(self.out/'validation/report.json', dict(rows=rows, selection=selection))

    def compare(self):
        selection = read(self.out/'selection.json')
        checkpoints = dict(selection['checkpoints'], D=read(self.out/'shared_report.json')['checkpoint'])
        cases = list(itertools.product(self.train, (2761, 2762)))
        available = self.budget['physical_cores'][:self.budget['effective']]
        lanes = max(n for n in range(1, len(available)+1) if len(cases) % n == 0)
        cores = available[:lanes]
        host = dict(uname=list(os.uname()), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), cores=cores)
        freeze(self.out/'comparison/timing_protocol.json', dict(host=host, cases=[[c['id'], s] for c, s in cases],
            methods=list(METHODS), seed_rotation='case index modulo seven', checkpoints={m: sha(p) for m, p in checkpoints.items()},
            fixed_lane_barriers=True))
        rows = []
        for start in range(0, len(cases), lanes):
            wave = self.out/'comparison'/f'wave_{start//lanes:04d}'
            signature = dict(host=host, cases=[[c['id'], s] for c, s in cases[start:start+lanes]],
                             timing_protocol_sha256=sha(self.out/'comparison/timing_protocol.json'))
            completed = wave/'completed.json'
            if completed.exists():
                cached = read(completed)
                if cached['signature'] != signature:
                    raise ValueError('timing wave host/input signature changed; use new cohort')
                rows.extend(cached['rows'])
                continue
            # Never mix cached partial-wave calls with newly measured calls.
            # An interrupted wave gets a fresh whole-wave attempt and an audit.
            if len(list(wave.glob('attempt_*'))) >= 2:
                raise RuntimeError('timing wave retry exhausted; inspect retained attempts')
            attempt = self.fresh_attempt(wave)
            barrier = threading.Barrier(lanes)
            def lane(item):
                i, (c, s) = item
                core = cores[i]
                os.sched_setaffinity(0, {core})
                offset = (start+i) % len(METHODS)
                order = METHODS[offset:]+METHODS[:offset]
                result = []
                try:
                    for m in order:
                        barrier.wait(timeout=self.args.timeout*4)
                        row = self.cold(c, s, m, checkpoints.get(m[0]),
                            attempt/f'lane_{i:03d}'/c['id']/f's{s}'/m, core)
                        result.append(dict(row, order=list(order), wave_load=lanes, host=host))
                        barrier.wait(timeout=self.args.timeout*4)
                except BaseException:
                    barrier.abort()
                    raise
                return result
            with ThreadPoolExecutor(max_workers=lanes) as pool:
                groups = list(pool.map(lane, enumerate(cases[start:start+lanes])))
            wave_rows = [r for group in groups for r in group]
            freeze(completed, dict(signature=signature, rows=wave_rows, actual_attempt=str(attempt),
                previous_incomplete_attempts=len(list(wave.glob('attempt_*')))-1,
                previous_incomplete_costs_retained=True))
            rows.extend(wave_rows)
            write(self.out/'comparison/report.json', dict(planned=224, completed=len(rows), rows=rows))
        for c, s in cases:
            for m in ('D', 'L', 'T'):
                pair = [r for r in rows if r['condition_id'] == c['id'] and r['seed'] == s and r['method'] in (m+'-I', m+'-S')]
                hashes = [r.get('ap_init_sha256') for r in pair]
                if len(hashes) != 2 or hashes[0] != hashes[1]:
                    raise ValueError('I/S did not reuse the same initializer: '+c['id']+'/'+m)

    def report(self):
        rows = read(self.out/'comparison/report.json')['rows']
        expected = [(c['id'], s, m) for c, s, m in itertools.product(self.train, (2761, 2762), METHODS)]
        result = evaluate(rows, expected=expected)
        primary = next(r for r in result['comparisons'] if (r['method'], r['baseline'], r['width_role']) == ('T-S', 'B', 'reference'))
        if not result['complete']:
            result['conclusion'] = 'incomplete_comparison'
        elif any(r['status'] in ('infrastructure_failure', 'interface_failure', 'interrupted') for r in rows):
            result['conclusion'] = 'infrastructure_or_interface_requires_diagnosis'
        elif primary['metrics']['total_seconds']['ratio'] is None:
            result['conclusion'] = 'insufficient_successful_pairs'
        elif primary['metrics']['total_seconds']['average_positive']:
            result['conclusion'] = ('average_runtime_gain' if primary['full_population_runtime_gain_claim']
                                    else 'conditional_runtime_gain_reported_with_failures')
        else:
            result['conclusion'] = 'no_average_runtime_gain'
        result['actual_comparison_vpr_tasks'] = sum(r.get('vpr_manifest') is not None for r in rows)
        result['comparison_decode_or_preparation_failures'] = sum(r.get('vpr_manifest') is None for r in rows)
        write(self.out/'layer_report.json', result)
        print(result['conclusion'], result['status_counts'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=STAGES+('worker',))
    p.add_argument('--source', type=Path, default=HERE/'runs/development_004_capacity')
    p.add_argument('--prior', type=Path, default=HERE/'runs/development_005_qualified_learning')
    p.add_argument('--out', type=Path, default=HERE/'runs/development_007_layer')
    p.add_argument('--backend', type=Path, default=REPO/'.deps/tailcompile_v7_3d_layer/vtr')
    p.add_argument('--qualification', type=Path, default=HERE/'runs/development_007_layer/qualification/qualification_report.json')
    p.add_argument('--request', type=Path)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--per-job-mib', type=int, default=4096)
    p.add_argument('--timeout', type=int, default=3600)
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    if min(args.jobs, args.per_job_mib, args.timeout) < 1:
        p.error('positive resource limits required')
    install_signal_handlers()
    args.out = owned(args.out)
    if args.stage == 'worker':
        worker(read(args.request), args.out)
        return
    if args.stage == 'plan':
        print(__import__('json').dumps(queue_plan(), indent=2))
        return
    if args.stage == 'check':
        result = check(args)
        print(__import__('json').dumps(result, indent=2))
        raise SystemExit(0 if result['ready'] else 2)
    with campaign_lock(args.out, 'layer-'+args.stage):
        campaign = LayerCampaign(args)
        campaign.setup()
        getattr(campaign, 'train_models' if args.stage == 'train' else args.stage)()


if __name__ == '__main__':
    main()
