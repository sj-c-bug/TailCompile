"""CPU/GPU stage hand-off for the registered three-method 3D experiment.

Every completed stage has immutable input signatures. No held-out teacher or
test-time candidate selection is permitted. P&R feedback and controlled timing
are separate queues. Long stages are intended for the server, not local smoke.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, as_completed, wait
from contextlib import contextmanager
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

from architectures import COMMIT, SOURCE_ARCH, LEGACY, _module, audit_family, generate_architecture
from device_graph import build_device_graph
from repository_guard import validate_output_roots
from io_utils import (emit, freeze as freeze_json, install_signal_handlers,
                      raise_if_cancelled, read, run_trial, sha, vpr_command, write)

HERE = Path(__file__).resolve().parent
SERVER_ROOT = Path('/inspire/hdd/project/project-public/changshunjie-253114010247/changshunjie-253114010247/workspace/DAC2027')
ARCHES = ('H0Z0', 'H0Z1', 'H1Z0', 'H1Z1')
METHODS = ('default', 'capacity_only', 'gnn')
ORDERS = tuple(itertools.permutations(METHODS))
INFRA = {'infrastructure_failure', 'interrupted', 'unknown', 'not_run'}


class InitializationFailure(RuntimeError):
    def __init__(self, status, seconds, log):
        super().__init__(f'{status}: {log}')
        self.status, self.seconds, self.log = status, seconds, log


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def freeze(path,value):
    # JSON has lists/string keys, not tuples/integer object keys. Compare in the
    # persisted representation so an identical resumed plan remains identical.
    return freeze_json(path,json.loads(json.dumps(value,allow_nan=False)))


def within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path != root and root not in path.parents:
        raise ValueError(f'path outside the declared experiment workspace: {path}')
    return path


@contextmanager
def campaign_lock(out, stage='unknown'):
    """One CLI controller per output root; child generators bypass this lock."""
    if os.name != 'posix':
        raise RuntimeError('campaign stage locking requires the Linux server/WSL environment')
    import fcntl
    out=Path(out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    path=out/'.campaign-stage.lock'
    with path.open('a+',encoding='utf-8') as handle:
        try:
            fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'another campaign stage is running for output root: {out}') from exc
        try:
            handle.seek(0)
            handle.truncate()
            json.dump(dict(pid=os.getpid(),stage=stage,started_unix=time.time()),handle)
            handle.flush()
            yield
        finally:
            # Keep the inode: unlinking a lock file permits split locks during restart.
            fcntl.flock(handle.fileno(),fcntl.LOCK_UN)


def physical_cores():
    allowed = set(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else set(range(os.cpu_count() or 1))
    path = Path('/proc/cpuinfo')
    cores = {}
    if path.exists():
        for block in path.read_text().split('\n\n'):
            row = dict(line.split(':', 1) for line in block.splitlines() if ':' in line)
            row = {key.strip(): value.strip() for key, value in row.items()}
            if 'processor' in row and int(row['processor']) in allowed:
                cores.setdefault((row.get('physical id', '0'), row.get('core id', row['processor'])), int(row['processor']))
    return sorted(cores.values()) or sorted(allowed)


def resource_budget(requested, per_job_mib, reserve_mib=2048):
    cores = physical_cores()
    quota = Path('/sys/fs/cgroup/cpu.max')
    quota_cores = len(cores)
    if quota.exists():
        amount, period = quota.read_text().split()[:2]
        if amount != 'max':
            quota_cores = max(1, int(int(amount)/int(period)))
    memory = 1 << 60
    meminfo = Path('/proc/meminfo')
    if meminfo.exists():
        value = next((line.split()[1] for line in meminfo.read_text().splitlines()
                      if line.startswith('MemAvailable:')), None)
        if value:
            memory = int(value)*1024
    maximum, current = Path('/sys/fs/cgroup/memory.max'), Path('/sys/fs/cgroup/memory.current')
    if maximum.exists() and current.exists() and maximum.read_text().strip() != 'max':
        memory = min(memory, int(maximum.read_text())-int(current.read_text()))
    memory_jobs = max(0, int((memory/(1024*1024)-reserve_mib)//per_job_mib))
    jobs = min(requested, len(cores), quota_cores, memory_jobs)
    if jobs < 1:
        raise RuntimeError('insufficient memory headroom for one declared job budget')
    return dict(requested=requested, effective=jobs, physical_cores=cores[:jobs],
                affinity_physical_count=len(cores), quota_cores=quota_cores,
                available_mib=round(memory/(1024*1024)), per_job_mib=per_job_mib,
                reserve_mib=reserve_mib,
                reduced=jobs < requested,
                rule='min(requested, physical affinity, CPU quota, declared RSS headroom)')


def select_widths(rows, grid):
    """Use only all four architectures x both calibration seeds at each point."""
    complete = {}
    for width in grid:
        batch = [row for row in rows if row['width'] == width]
        if len(batch) != 8 or len({(r['architecture'], r['seed']) for r in batch}) != 8:
            continue
        if any(row['status'] not in ('success', 'unroutable') for row in batch):
            continue
        complete[width] = sum(row['status'] == 'success' for row in batch)/8
    refs = [width for width in grid if complete.get(width) == 1]
    if not refs:
        return dict(status='right_censored', W_ref=None, W_stress=None, success_rates=complete)
    reference = min(refs)
    lower = [width for width in grid if width < reference and width in complete and complete[width] < 1]
    if not lower:
        return dict(status='left_censored', W_ref=reference, W_stress=None, success_rates=complete)
    stress = min(lower, key=lambda width: (abs(complete[width]-.5), -width))
    return dict(status='paired', W_ref=reference, W_stress=stress, success_rates=complete)


def pilot_tasks(conditions, pilot_families, seeds=(531, 532, 533, 534, 535)):
    if len(pilot_families) != 4 or len(set(pilot_families)) != 4:
        raise ValueError('pilot must pre-register four independent development families')
    selected = [row for row in conditions if row['family'] in pilot_families]
    for family in pilot_families:
        rows = [row for row in selected if row['family'] == family]
        cells = {(row['architecture'], row['width_role']) for row in rows}
        if len(rows) != 8 or cells != set(itertools.product(ARCHES, ('stress', 'reference'))):
            raise ValueError(f'pilot family {family} lacks its complete four-architecture/two-width panel')
    return [dict(condition=row, seed=seed, method=method) for row in selected
            for seed in seeds for method in METHODS]


def _gmean(values):
    return math.exp(sum(map(math.log, values))/len(values)) if values else None


def hard_ap_counts(logic):
    """Frozen cohort thresholds count hard APBlocks, not DSP mode units."""
    return {kind: sum(block['resource'] == kind for block in logic['clusters'])
            for kind in ('memory', 'dsp_top')}


class Campaign:
    def __init__(self, args):
        self.args = args
        self.config = read(args.config)
        self.out, self.scratch = args.out.resolve(), args.scratch.resolve()
        self.vtr = args.vtr_root.resolve()
        self.mapping = (args.mapping_vtr_root or args.vtr_root).resolve()
        validate_output_roots((self.out,self.scratch),dependency_roots=(self.vtr,self.mapping))
        self.binary = args.binary.resolve()
        self.root = args.workspace.resolve()
        for path in (self.out, self.scratch, self.vtr, self.mapping, self.binary):
            within(path, self.root)
        if not args.allow_local and self.root != SERVER_ROOT.resolve():
            raise ValueError('server mode requires the registered workspace root; local smoke needs --allow-local')
        for path in (self.out, self.scratch, self.scratch/'tmp', self.scratch/'cache'):
            path.mkdir(parents=True, exist_ok=True)
        for key, value in dict(TMPDIR=self.scratch/'tmp', TMP=self.scratch/'tmp',
                               TEMP=self.scratch/'tmp', XDG_CACHE_HOME=self.scratch/'cache').items():
            os.environ[key] = str(value)
        self.backend = read(args.backend_manifest)
        self.binary_hash=sha(self.binary)
        self.binary_stat=self._stat(self.binary)
        if self.backend.get('vtr_commit') != COMMIT or self.backend.get('binary_sha256') != self.binary_hash:
            raise ValueError('backend manifest does not match the required commit/binary')
        self.binary_reference=self.out/'binary_identity.json'
        freeze(self.binary_reference,dict(binary=str(self.binary),sha256=self.binary_hash,
            commit=COMMIT,backend_manifest_sha256=sha(args.backend_manifest)))
        records = self.config['designs']
        if len({r['design'] for r in records}) != len(records):
            raise ValueError('duplicate design record')
        splits = defaultdict(set)
        for row in records:
            splits[row['family']].add(row['split'])
            if len(row['grid']) != 2 or min(row['grid']) < 12:
                raise ValueError('each development input needs an explicit supported fixed grid')
        if any(len(value) > 1 for value in splits.values()):
            raise ValueError('design-family leakage between splits')
        if any(r['split'] not in ('train', 'validation') for r in records):
            raise ValueError('this development campaign forbids inherited/final test records')
        if self.config.get('primary_runtime') != 'cold':
            raise ValueError('this protocol implementation freezes cold-start total runtime as primary')
        self.budget = resource_budget(args.jobs, args.per_job_mib)
        self.rrg_budget = resource_budget(args.rrg_jobs, args.rrg_mib)
        self.rrg_slots = threading.Semaphore(self.rrg_budget['effective'])
        self.cache_locks = defaultdict(threading.Lock)
        self.cache_locks_guard = threading.Lock()
        self.verified_rr = set()
        self.rr_stats={}
        inherited = [HERE.parent/'tailcompile_v68'/name for name in ('model.py','data.py')]
        inherited += [HERE.parent/'tailcompile_v69/phase_metrics.py']
        inherited += [HERE.parent/'tailrisk_core.py']
        inherited += [LEGACY/name for name in ('heterogeneity_fixtures.py','pilot_architectures.py',
                                              'device_supply_3d.py','prepare_natural_blif.py')]
        frozen = dict(schema='tailcompile-v7-campaign-1', config=self.config,
                      backend_manifest_sha256=sha(args.backend_manifest),
                      binary_sha256=self.binary_hash,
                      implementation={name: sha(HERE/name) for name in
                          ('campaign.py','architectures.py','device_graph.py','data.py','model.py',
                           'generation.py','supervision.py','train.py','io_utils.py','repository_guard.py')})
        frozen['inherited_implementation']={str(path):sha(path) for path in inherited}
        mapping_commit=subprocess.check_output(['git','-C',str(self.mapping),'rev-parse','HEAD'],text=True).strip()
        if mapping_commit!=COMMIT:
            raise ValueError('mapping VTR root must use the same pinned commit')
        frozen['mapping_vtr_root']=str(self.mapping)
        frozen['mapping_vtr_commit']=mapping_commit
        freeze(self.out/'protocol.json', frozen)
        write(self.out/'resources.json', dict(cpu=self.budget, rrg=self.rrg_budget,
                                            stage=args.stage, pid=os.getpid()))
        write(self.out/'resource_records'/f'{args.stage}_{time.time_ns()}_{os.getpid()}.json',
              dict(cpu=self.budget,rrg=self.rrg_budget,stage=args.stage,pid=os.getpid()))
        emit('campaign_resources', stage=args.stage, cpu=self.budget, rrg=self.rrg_budget)

    def guard(self):
        raise_if_cancelled()
        minimum = self.config.get('minimum_free_gib', 8)*(1024**3)
        if min(shutil.disk_usage(path).free for path in (self.out, self.scratch)) < minimum:
            raise OSError('space preflight failed; new dispatch stopped, active tasks will drain')

    def gate(self):
        self.check_binary()
        report = read(self.args.qualification_report)
        if not report.get('complete') or not report.get('passed'):
            raise RuntimeError('full XYZ/native-GP/APPack qualification must pass before this stage')
        protocol = read(self.args.qualification_report.parent/'protocol.json')
        if protocol.get('binary_sha256') != self.binary_hash or protocol.get('commit') != COMMIT:
            raise ValueError('qualification evidence belongs to another backend')
        return report

    @staticmethod
    def _stat(path):
        stat=Path(path).stat()
        return (stat.st_size,stat.st_mtime_ns,stat.st_ino)

    def check_binary(self):
        if self._stat(self.binary)!=self.binary_stat:
            raise ValueError('frozen executable changed during campaign dispatch')

    def parallel(self, items, function):
        """Bound submitted work; a quota/disk exception stops further dispatch."""
        items = iter(items)
        pending, results = {}, []
        with ThreadPoolExecutor(max_workers=self.budget['effective']) as pool:
            def submit():
                raise_if_cancelled()
                try:
                    item = next(items)
                except StopIteration:
                    return False
                self.guard()
                pending[pool.submit(function, item)] = item
                return True
            for _ in range(self.budget['effective']):
                if not submit():
                    break
            try:
                while pending:
                    done, _ = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        pending.pop(future)
                        results.append(future.result())
                        emit('campaign_progress', completed=len(results), active=len(pending))
                        submit()
            except BaseException:
                for future in pending:
                    future.cancel()
                raise
        return results

    def architecture(self, record, condition):
        width, height = record['grid']
        folder = self.out/'architectures'/f'{width}x{height}'/condition
        arch, meta = folder/'arch.xml', folder/'architecture.json'
        with self._lock(str(folder)):
            if not meta.exists():
                write(meta, generate_architecture(self.vtr/SOURCE_ARCH, arch,
                    int(condition[1]), int(condition[3]), width, height))
            if sha(arch) != read(meta)['architecture_sha256']:
                raise ValueError('frozen architecture changed')
        return arch

    def _lock(self, key):
        with self.cache_locks_guard:
            return self.cache_locks[key]

    def rr(self, record, condition, width):
        self.check_binary()
        arch = self.architecture(record, condition)
        key = identity(dict(architecture=sha(arch), binary=self.binary_hash, width=width))[:20]
        folder = self.out/'device_cache'/key
        manifest, graph, rrg = folder/'manifest.json', folder/'device_graph.json', folder/'rr_graph.xml'
        with self._lock(key):
            if manifest.exists():
                old = read(manifest)
                if not rrg.is_file():
                    raise ValueError('RRG/device cache is missing or corrupt')
                if key not in self.verified_rr:
                    if sha(rrg) != old['rr_graph_sha256'] or sha(graph) != old['graph_sha256']:
                        raise ValueError('RRG/device cache checksum mismatch')
                    self.verified_rr.add(key)
                    self.rr_stats[key]=(self._stat(rrg),self._stat(graph))
                elif self.rr_stats[key]!=(self._stat(rrg),self._stat(graph)):
                    raise ValueError('immutable shared device cache changed during dispatch')
                return dict(rr_graph=str(rrg), device=str(graph), manifest=str(manifest))
            with self.rrg_slots:
                self.guard()
                folder.mkdir(parents=True, exist_ok=True)
                command = [self.binary, arch, record['blif'], '--pack', '--device','pilot',
                    '--route_chan_width',str(width),'--place_chan_width',str(width),
                    '--num_workers','1','--write_rr_graph',rrg]
                row = run_trial(folder/'build', command, inputs=[self.binary_reference,arch,record['blif']],
                    timeout=self.args.timeout, metadata=dict(stage='reusable_rrg', width=width), prune=True)
                if row['status'] != 'complete' or not rrg.is_file():
                    raise RuntimeError(f'RRG generation failed: {row["log"]}')
                value = build_device_graph(rrg, arch, self.config.get('tile_size',4))
                write(graph, value)
                write(manifest, dict(architecture_sha256=sha(arch), binary_sha256=self.binary_hash,
                    channel_width=width, rr_graph_sha256=sha(rrg), graph_sha256=sha(graph),
                    producer_blif_sha256=sha(record['blif']), reusable_across_designs=True,
                    graph_is_not_a_placement_or_routability_certificate=True))
                self.verified_rr.add(key)
                self.rr_stats[key]=(self._stat(rrg),self._stat(graph))
        return dict(rr_graph=str(rrg), device=str(graph), manifest=str(manifest))

    def prepare(self):
        from data import build_logic
        self.gate()
        if (self.out/'prepared.json').exists():
            return self.records()
        rows = []
        for original in self.config['designs']:
            self.guard()
            record = dict(original)
            folder = self.out/'inputs'/record['design']
            folder.mkdir(parents=True, exist_ok=True)
            arch = self.architecture(record, 'H0Z0')
            for condition in ARCHES:
                self.architecture(record, condition)
            source = Path(record.get('mapped_blif') or self.mapping/'vtr_flow/benchmarks/verilog'/record['source']).resolve()
            within(source, self.root)
            if not source.is_file():
                raise FileNotFoundError(f'pre-registered source missing: {source}')
            include=source.parent/'hard_block_include.v'
            freeze(folder/'source.json', dict(source=str(source), sha256=sha(source), family=record['family'],
                include=str(include) if source.suffix!='.blif' and include.exists() else None,
                include_sha256=sha(include) if source.suffix!='.blif' and include.exists() else None,
                split=record['split'], fixed_grid=record['grid']))
            if source.suffix.lower() == '.blif':
                mapped = source
            else:
                runner=self.mapping/'vtr_flow/scripts/run_vtr_flow.py'
                command = [sys.executable,runner,source,arch,
                           '-temp_dir','flow','-ending_stage','abc','-timeout',str(self.args.timeout)]
                mapping_inputs=[source,arch,runner]
                for executable in (self.mapping/'abc/abc',self.mapping/'parmys/yosys/yosys',
                                   self.mapping/'odin_ii/odin_II'):
                    if executable.is_file():
                        mapping_inputs.append(executable)
                if record['source'].startswith('koios/'):
                    include = source.parent/'hard_block_include.v'
                    if include.exists():
                        command += ['-include',include]
                        mapping_inputs.append(include)
                trial = run_trial(folder/'mapping', command, inputs=mapping_inputs,
                                  timeout=self.args.timeout, prune=False)
                # run_trial owns a fresh attempt cwd, including retries. A relative
                # flow avoids sharing stale ABC intermediates between campaigns.
                mapped = Path(trial['workdir'])/'flow'/(source.stem+'.abc.blif')
                if trial['status'] != 'complete' or not mapped.is_file():
                    raise RuntimeError(f'RTL mapping failed: {trial["log"]}')
            blif = folder/'design.blif'
            if not blif.exists():
                conversion = _module('prepare_natural_blif').prepare_blif(mapped,blif,arch)
                write(folder/'mapping_adapter.json',conversion)
            record['blif'] = str(blif)
            prefix = folder/'structure'
            structure = run_trial(folder/'structure_run',
                vpr_command(self.binary,arch,blif,self.config['width_grid'][0],701)+['--device','pilot'],
                inputs=[self.binary_reference,arch,blif], env=dict(TC_GP_STRUCTURE=prefix,TC_GP_STRUCTURE_ONLY=1),
                timeout=self.args.timeout, marker='TAILCOMPILE_V7_STRUCTURE_ONLY_COMPLETE', prune=True)
            if structure['status'] != 'complete':
                raise RuntimeError(f'pre-GP structural export failed: {structure["log"]}')
            logic = build_logic(prefix,record['design'],blif,arch,family=record['family'],split=record['split'])
            demands = [sum(c['demand'][a] for c in logic['clusters']) for a in range(5)]
            hard_counts=hard_ap_counts(logic)
            record.update(structure_prefix=str(prefix), primitive_demand=demands,
                          hard_ap_counts=hard_counts,
                          resource_group_semantics='one count per memory/DSP APBlock; mode demand remains separate',
                          structure_export_seconds=structure['metrics']['wall_seconds'],
                          mapped_blif_sha256=sha(blif),
                          snapshot_sha256={suffix:sha(str(prefix)+suffix) for suffix in
                                           ('.blocks.tsv','.nets.tsv','.meta.tsv')},
                          node_count=len(logic['clusters']), prepared=True)
            metadata=read(arch.parent/'architecture.json')
            from device_graph import resource_units
            units=resource_units(arch)
            roots={kind:sum(len({tuple(p[:2]) for p in layer['slots'][kind]})
                            for layer in metadata['layers']) for kind in ('clb','memory','dsp_top','io')}
            upper=[sum(roots[kind]*units.get(kind,{}).get(resource,0) for kind in roots)
                   for resource in ('lut','ff','memory','dsp_top','io')]
            record['resource_upper_capacity']=upper
            record['resource_upper_utilization']=[d/c if c else None for d,c in zip(demands,upper)]
            record['physical_root_counts']=roots
            if record.get('pilot_group') == 'heavy' and sum(hard_counts.values()) < self.config.get('heavy_min_hard_blocks',16):
                raise ValueError(f'{record["design"]} did not meet pre-registered heavy demand threshold')
            if record.get('pilot_group') == 'light' and sum(hard_counts.values()) > self.config.get('light_max_hard_blocks',8):
                raise ValueError(f'{record["design"]} did not meet pre-registered light demand threshold')
            if record.get('pilot_group')=='heavy' and max(v or 0 for v in record['resource_upper_utilization'][2:4]) < self.config.get('heavy_min_peak_utilization',.1):
                raise ValueError(f'{record["design"]} hard-resource utilization proxy is below its pre-registered heavy threshold')
            write(folder/'logic.json',logic)
            rows.append(record)
            write(self.out/'prepare_progress.json',dict(records=rows,complete=len(rows)==len(self.config['designs'])))
        freeze(self.out/'prepared.json',dict(records=rows,complete=True))
        return rows

    def records(self):
        value = read(self.out/'prepared.json')
        if not value.get('complete'):
            raise RuntimeError('prepare has not completed')
        for record in value['records']:
            if sha(record['blif'])!=record['mapped_blif_sha256']:
                raise ValueError('prepared mapped BLIF changed')
            for suffix,digest in record['snapshot_sha256'].items():
                if sha(record['structure_prefix']+suffix)!=digest:
                    raise ValueError('prepared structural snapshot changed')
            source=read(self.out/'inputs'/record['design']/'source.json')
            if sha(source['source'])!=source['sha256'] or (
                    source.get('include') and sha(source['include'])!=source['include_sha256']):
                raise ValueError('prepared RTL/BLIF/include dependency changed')
        return value['records']

    def execute(self, condition, seed, folder, *, method='default', checkpoint=None,
                candidate=0, affinity=None, teacher=False):
        record = condition['record']
        arch = self.architecture(record,condition['architecture'])
        rr = self.rr(record,condition['architecture'],condition['width'])
        folder = Path(folder)
        env, generation_seconds = {}, 0.
        inputs = [self.binary_reference,arch,record['blif'],rr['manifest']]
        generation = None
        if method != 'default':
            try:
                generation = self.generate(condition,folder/'initializer',method=method,
                                           checkpoint=checkpoint,candidate=candidate,affinity=affinity)
            except InitializationFailure as exc:
                result = dict(condition_id=condition['id'],design=condition['design'],family=condition['family'],
                    architecture=condition['architecture'],width=condition['width'],width_role=condition.get('width_role'),
                    seed=seed,method=method,status=exc.status,cpd_ns=None,wirelength=None,
                    vpr_seconds=None,generation_seconds=exc.seconds,total_seconds=exc.seconds,
                    generation=None,vpr_manifest=None,error_log=exc.log)
                write(folder/'result.json',result)
                return result
            env['TC_GP_INIT'] = generation['ap_init']
            inputs.append(generation['ap_init'])
            generation_seconds = generation['cold_generation_seconds']+generation['nonreusable_logical_seconds']
        if teacher:
            env['TC_GP_EXPORT'] = str(folder/'teacher_gp')
        row = run_trial(folder/'vpr',vpr_command(self.binary,arch,record['blif'],condition['width'],seed,
                            rr_graph=rr['rr_graph'])+['--device','pilot',
                            '--write_flat_place','post_route.fplace'],
            inputs=inputs,env=env,timeout=self.args.timeout,affinity=affinity,
            metadata=dict(condition_id=condition['id'],method=method,seed=seed,
                          checkpoint_sha256=sha(checkpoint) if checkpoint else None),prune=True)
        if method != 'default' and row['status'] in ('success','unroutable'):
            text = Path(row['log']).read_text(errors='replace')
            if ('TAILCOMPILE_V7_GP_INIT_CONSUMED' not in text or
                'TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' not in text or
                'skipping the Global Placement' in text):
                raise RuntimeError('warm initial state was not consumed by the native GP chain')
        metrics = row['metrics']
        result = dict(condition_id=condition['id'],design=condition['design'],family=condition['family'],
            architecture=condition['architecture'],width=condition['width'],width_role=condition.get('width_role'),
            seed=seed,method=method,status=row['status'],cpd_ns=metrics.get('cpd_ns'),
            wirelength=metrics.get('wirelength'),vpr_seconds=metrics['wall_seconds'],
            generation_seconds=generation_seconds,total_seconds=metrics['wall_seconds']+generation_seconds,
            generation_process_seconds=generation['cold_generation_seconds'] if generation else 0.,
            extra_logical_preparation_seconds=generation['nonreusable_logical_seconds'] if generation else 0.,
            phases=metrics.get('phases'),unique_chanz_nodes=metrics.get('unique_chanz_nodes'),
            vpr_manifest=str(folder/'vpr/manifest.json'),generation=generation)
        flat = Path(row['workdir'])/'post_route.fplace'
        if flat.exists():
            counts = Counter()
            with flat.open() as stream:
                for line in stream:
                    fields = line.split()
                    if not fields or fields[0].startswith('#') or len(fields)<5:
                        continue
                    try:
                        counts[str(int(float(fields[3])))] += 1
                    except ValueError:
                        continue
            result['post_route_layer_atom_counts'] = dict(counts)
            result['post_route_fplace_sha256'] = sha(flat)
        else:
            previous = read(folder/'result.json') if (folder/'result.json').exists() else {}
            result['post_route_layer_atom_counts'] = previous.get('post_route_layer_atom_counts')
            result['post_route_fplace_sha256'] = previous.get('post_route_fplace_sha256')
        result['critical_path_cross_layer_delay_ns'] = None
        result['critical_path_cross_layer_delay_status'] = 'not_extracted; no geometric delay substitution'
        write(folder/'result.json',result)
        if flat.exists():
            # This is an explicitly requested, derived diagnostic output. The
            # initializer, teachers, main .place and layer summary are retained.
            ledger = dict(file=str(flat),bytes=flat.stat().st_size,sha256=sha(flat),
                reason='post-route layer counts and hash durable; no teacher or replay input dependency')
            flat.unlink()
            write(folder/'post_route_trace_pruned.json',ledger)
        return result

    def calibrate(self):
        self.gate()
        results, boundaries = [], []
        # Each record advances one eight-run width wave at a time. Different
        # designs can fill the queue; no method outcome is used for selection.
        grid = self.config['width_grid']
        def one_design(record):
            report_path = self.out/'calibration'/record['design']/'boundary.json'
            rows = []
            for width in grid:
                conditions = [self.condition(record,arch,width) for arch in ARCHES]
                tasks = [(c,seed) for c in conditions for seed in (501,502)]
                batch = self.parallel(tasks,lambda item:self.execute(item[0],item[1],
                    self.out/'calibration'/record['design']/item[0]['architecture']/f'w{width}'/f's{item[1]}'))
                rows.extend(batch)
                write(self.out/'calibration'/record['design']/'trials.json',rows)
                if any(row['status'] not in ('success','unroutable') for row in batch):
                    raise RuntimeError(f'Default calibration has a non-routing failure: {record["design"]}, w{width}')
                boundary = select_widths(rows,grid)
                if boundary['status'] == 'paired' or (boundary['status']=='left_censored' and boundary['W_ref']==grid[0]):
                    break
            boundary.update(design=record['design'],family=record['family'],split=record['split'],record=record)
            freeze(report_path,boundary)
            return boundary,rows
        records = self.records()
        design_jobs = max(1,min(len(records),self.budget['effective']//8))
        emit('calibration_parallelism',design_jobs=design_jobs,runs_per_width_wave=8,
             maximum_active_runs=min(self.budget['effective'],8*design_jobs))
        with ThreadPoolExecutor(max_workers=design_jobs) as pool:
            for boundary,rows in pool.map(one_design,records):
                boundaries.append(boundary);results.extend(rows)
        freeze(self.out/'boundaries.json',dict(records=boundaries))
        conditions = []
        for boundary in boundaries:
            if boundary['status'] != 'paired':
                continue
            for arch in ARCHES:
                for role,key in (('stress','W_stress'),('reference','W_ref')):
                    conditions.append(self.condition(boundary['record'],arch,boundary[key],role))
        freeze(self.out/'conditions.json',dict(records=conditions,
            censored=[b['design'] for b in boundaries if b['status'] != 'paired']))
        freeze(self.out/'conditions_manifest.json',dict(
            conditions_sha256=sha(self.out/'conditions.json'),boundaries_sha256=sha(self.out/'boundaries.json'),
            prepared_sha256=sha(self.out/'prepared.json')))
        # Audit every actual per-width cache, keeping same-width four-cell matches.
        for size in sorted({tuple(r['grid']) for r in self.records()}):
            selected = [r for r in self.records() if tuple(r['grid']) == size]
            widths = sorted({row['width'] for row in results if any(row['design']==r['design'] for r in selected)})
            graphs = {arch:[read(self.rr(selected[0],arch,width)['device']) for width in widths] for arch in ARCHES}
            write(self.out/'architectures'/f'{size[0]}x{size[1]}'/'actual_audit.json',audit_family(graphs))
        return conditions

    def condition(self, record, arch, width, role=None):
        return dict(id=f'{record["design"]}/{arch}/w{width}', design=record['design'],
                    family=record['family'],split=record['split'],architecture=arch,
                    width=int(width),width_role=role,record=record)

    def conditions(self):
        self.records()
        expected=read(self.out/'conditions_manifest.json')
        if any(sha(self.out/name)!=expected[key] for name,key in
               (('conditions.json','conditions_sha256'),('boundaries.json','boundaries_sha256'),
                ('prepared.json','prepared_sha256'))):
            raise ValueError('frozen condition catalog changed')
        return read(self.out/'conditions.json')['records']

    def graph(self, condition):
        from data import build_cached_graph
        record = condition['record']
        rr = self.rr(record,condition['architecture'],condition['width'])
        folder = Path(condition.get('graph_root',self.out/'graphs'/condition['id']))
        with self._lock('graph:'+str(folder)):
            cost=folder/'context_cost.json'
            if (folder/'graph_manifest.json').exists() and not cost.exists():
                raise RuntimeError('graph cache lacks original logical cost evidence; do not replace it with cached-read time')
            started = time.perf_counter()
            row = build_cached_graph(record['structure_prefix'],record['design'],record['blif'],
                self.architecture(record,condition['architecture']),rr['device'],folder,
                family=record['family'],split=record['split'])
            if not cost.exists():
                write(cost,dict(measured_graph_build_seconds=time.perf_counter()-started,
                    structure_export_seconds=record.get('structure_export_seconds'),
                    reusable_rrg_preprocessing_excluded=True))
        return row

    def teachers(self):
        self.gate()
        from data import raw_teacher
        from supervision import project_teacher
        from train import load_graph
        def collect(condition):
            graph = self.graph(condition)
            logic = read(graph['logic'])
            choices = []
            for seed in (511,512):
                folder = self.out/'teachers'/condition['id']/f's{seed}'
                row = self.execute(condition,seed,folder,teacher=True)
                prefix = folder/'teacher_gp'
                if not Path(str(prefix)+'.meta.tsv').exists():
                    raise RuntimeError(f'teacher GP export missing after {row["status"]}: {folder}')
                raw = raw_teacher(prefix,logic,split=condition['split'])
                raw_file, target_file = folder/'raw.json',folder/'projected.json'
                write(raw_file,raw)
                target,audit = project_teacher(load_graph(graph['graph']),raw)
                write(target_file,target);write(folder/'supervision_audit.json',audit)
                choices.append((seed,target_file))
            seed,target = choices[0]  # Pre-registered seed order, never best QoR or mean coordinates.
            return dict(condition_id=condition['id'],family=condition['family'],split=condition['split'],
                        graph=graph['graph'],teacher=str(target),teacher_seed=seed)
        records = self.parallel(self.conditions(),collect)
        records.sort(key=lambda r:r['condition_id'])
        write(self.out/'teacher_records.json',dict(records=records,raw_teacher_seeds=[511,512],
                                                   supervision_seed_selection='first fixed seed, no QoR selection'))
        return records

    def train(self, feedback=False):
        import torch
        from train import train
        torch.set_num_threads(self.args.threads)
        records = [r for r in read(self.out/'teacher_records.json')['records'] if r['split']=='train']
        name = 'runtime' if feedback else 'pretrain'
        root = self.out/'models'/name
        root.mkdir(parents=True,exist_ok=True)
        teacher_report = read(self.out/'models/pretrain/completed.json')['report'] if feedback else None
        source = teacher_report['checkpoints'][-1] if feedback else None
        signature = dict(records_sha256=sha(self.out/'teacher_records.json'),config=self.config['training'],
                         record_payloads={r['condition_id']:dict(graph=sha(r['graph']),teacher=sha(r['teacher'])) for r in records},
                         source_checkpoint_sha256=sha(source) if source else None,
                         feedback=sha(self.out/'feedback.json') if feedback else None)
        freeze(root/'input_signature.json',signature)
        completed = root/'completed.json'
        if completed.exists():
            old = read(completed)
            if any(sha(path)!=digest for path,digest in old['checkpoint_sha256'].items()):
                raise ValueError('trained checkpoint changed')
            return old['report']
        attempt = root/f'attempt_{len(list(root.glob("attempt_*")))+1:04d}'
        config = self.config['training']
        report = train(records,attempt,epochs=config['feedback_epochs' if feedback else 'pretrain_epochs'],
            hidden=config.get('hidden',64),learning_rate=config.get('learning_rate',.001),
            seed=config.get('seed',601),checkpoint=source,
            feedback=read(self.out/'feedback.json')['records'] if feedback else None,
            feedback_weight=config.get('feedback_weight',.2),device=self.args.device)
        write(completed,dict(report=report,checkpoint_sha256={path:sha(path) for path in report['checkpoints']}))
        return report

    def generate(self, condition, folder, *, method, checkpoint=None, candidate=0, affinity=None):
        folder = Path(folder)
        folder.mkdir(parents=True,exist_ok=True)
        graph = self.graph(condition)
        rr = self.rr(condition['record'],condition['architecture'],condition['width'])
        noise = self.config['candidate_noise'][candidate] if candidate else dict(score=0.,coordinate=0.)
        request = dict(graph=graph['graph'],logic=graph['logic'],device=rr['device'],method=method,
                       checkpoint=str(checkpoint) if checkpoint else None,
                       seed=int(identity(dict(condition=condition['id'],candidate=candidate))[:8],16),
                       score_noise_std=noise['score'],coordinate_noise_std=noise['coordinate'],
                       density_passes=self.config.get('density_passes',2))
        freeze(folder/'request.json',request)
        payload = folder/'payload'
        command = [sys.executable,HERE/'campaign.py','generate-worker','--request',folder/'request.json','--out',payload]
        row = run_trial(folder/'execution',command,
            inputs=[folder/'request.json',graph['graph'],graph['logic'],rr['device'],HERE/'model.py',HERE/'generation.py']+
                   ([checkpoint] if checkpoint else []),timeout=self.config.get('generation_timeout',120),
            marker='TAILCOMPILE_V7_GENERATION_COMPLETE',affinity=affinity,prune=False)
        if row['status'] != 'complete':
            status='timeout' if row['status']=='timeout' else 'infrastructure_failure'
            if status!='timeout':
                for line in Path(row['log']).read_text(errors='replace').splitlines():
                    marker='TAILCOMPILE_V7_INITIALIZATION_FAILURE '
                    if line.startswith(marker):
                        status=json.loads(line[len(marker):])['status']
            raise InitializationFailure(status,row['metrics']['wall_seconds'],row['log'])
        generated = read(payload/'generation.json')
        if sha(generated['ap_init']) != generated['ap_init_sha256']:
            raise ValueError('initializer payload changed')
        if sha(payload/'target.json') != generated['target_sha256']:
            raise ValueError('initializer action payload changed')
        generated.update(cold_generation_seconds=row['metrics']['wall_seconds'],
                         nonreusable_logical_seconds=self._logical_cost(condition),
                         target=str(payload/'target.json'),generation_manifest=str(payload/'generation.json'))
        return generated

    def _logical_cost(self,condition):
        folder=Path(condition.get('graph_root',self.out/'graphs'/condition['id']))
        cost=read(folder/'context_cost.json')
        structure=cost.get('structure_export_seconds')
        if structure is None:
            raise ValueError('cold total requires measured AP structural preparation cost')
        return structure+cost['measured_graph_build_seconds']

    def feedback(self):
        source = read(self.out/'models/pretrain/completed.json')['report']['checkpoints'][-1]
        conditions = [c for c in self.conditions() if c['split']=='train']
        def collect(condition):
            defaults = [self.execute(condition,seed,self.out/'feedback'/condition['id']/'default'/f's{seed}')
                        for seed in (521,522)]
            entries = []
            for candidate in range(4):
                trials = [self.execute(condition,seed,self.out/'feedback'/condition['id']/f'c{candidate}'/f's{seed}',
                            method='gnn',checkpoint=source,candidate=candidate) for seed in (521,522)]
                targets = [read(row['generation']['target']) for row in trials if row.get('generation')]
                if len(targets)==2 and targets[0] != targets[1]:
                    raise ValueError('same feedback action changed across paired backend seeds')
                entries.append(dict(condition_id=condition['id'],candidate_id=candidate,action=targets[0] if targets else None,
                                    trials=trials,default_trials=defaults))
            write(self.out/'feedback'/condition['id']/'all_actions.json',entries)
            return entries
        groups = self.parallel(conditions,collect)
        value = dict(records=[r for group in groups for r in group],
                     all_candidate_statuses_retained=True,candidates_per_condition=4,
                     checkpoint_sha256=sha(source),test_labels_used=False)
        value['records'].sort(key=lambda r:(r['condition_id'],r['candidate_id']))
        write(self.out/'feedback.json',value)
        return value

    def validation_report(self, rows, defaults, checkpoint, epoch, signature):
        baseline = {(r['condition_id'],r['seed']):r for r in defaults}
        paired = [(r,baseline[(r['condition_id'],r['seed'])]) for r in rows]
        groups = defaultdict(list)
        unknown = 0
        for row,default in paired:
            unknown += int(row['status'] in INFRA or default['status'] in INFRA)
            if row['status']=='success' and default['status']=='success':
                groups[row['family']].append((row,default))
        family_ratios = {}
        for family,values in groups.items():
            family_ratios[family] = dict(
                runtime=_gmean([a['total_seconds']/b['total_seconds'] for a,b in values]),
                cpd=_gmean([a['cpd_ns']/b['cpd_ns'] for a,b in values]),
                wire=_gmean([a['wirelength']/b['wirelength'] for a,b in values]))
        quality = defaultdict(list)
        strata_success=defaultdict(int)
        for row,default in paired:
            strata_success[(row['family'],row['architecture'],row['width_role'])] += (
                int(row['status']=='success')-int(default['status']=='success'))
            if row['status']=='success' and default['status']=='success':
                quality[(row['family'],row['architecture'],row['width_role'])].append((row,default))
        strata = [dict(family=key[0],architecture=key[1],width_role=key[2],
                    cpd_ratio=_gmean([a['cpd_ns']/b['cpd_ns'] for a,b in values]),
                    wirelength_ratio=_gmean([a['wirelength']/b['wirelength'] for a,b in values]))
                  for key,values in quality.items()]
        missing = len({r['family'] for r in defaults})-len(family_ratios)
        return dict(evidence='real_vpr_validation',validation_signature=signature,
            checkpoint=str(checkpoint),epoch=epoch,unknown_count=unknown+missing,
            strata_success_protection_passed=all(delta>=0 for delta in strata_success.values()),
            strata_success_deltas=[dict(family=k[0],architecture=k[1],width_role=k[2],delta=v) for k,v in strata_success.items()],
            success_count=sum(r['status']=='success' for r in rows),
            default_success_count=sum(r['status']=='success' for r in defaults),
            total_runtime_ratio=_gmean([v['runtime'] for v in family_ratios.values()]) or 1e30,
            cpd_ratio=max([v['cpd'] for v in family_ratios.values()]+
                          [v['cpd_ratio'] for v in strata],default=1e30),
            wirelength_ratio=max([v['wire'] for v in family_ratios.values()]+
                                 [v['wirelength_ratio'] for v in strata],default=1e30),
            family_ratios=family_ratios,strata=strata,rows=rows,
            timing_scope='development real cold generation plus VPR; formal controlled timing is separate')

    def validate(self):
        from supervision import select_checkpoint
        report = read(self.out/'models/runtime/completed.json')['report']
        checkpoints = report['checkpoints']
        if not 1 <= len(checkpoints) <= 3:
            raise ValueError('validation may inspect at most three frozen checkpoints')
        conditions = [r for r in self.conditions() if r['split']=='validation']
        if not conditions:
            raise ValueError('no qualified validation conditions')
        seeds = self.config.get('validation_seeds',[531,532])
        tasks = [(c,seed) for c in conditions for seed in seeds]
        signature = identity([(c['id'],seed) for c,seed in tasks])
        freeze(self.out/'validation/plan.json',dict(conditions=[(c['id'],s) for c,s in tasks],
            checkpoints={str(p):sha(p) for p in checkpoints},max_candidates=3))
        if (self.out/'selection.json').exists():
            previous=read(self.out/'selection.json')
            if previous.get('checkpoint') and sha(previous['checkpoint'])!=previous['checkpoint_sha256']:
                raise ValueError('selected checkpoint changed')
            return previous
        defaults = self.parallel(tasks,lambda item:self.execute(item[0],item[1],
            self.out/'validation/default'/item[0]['id']/f's{item[1]}'))
        reports = []
        for checkpoint in checkpoints:
            # Epoch is read from its declared filename; no teacher-loss ranking.
            epoch = int(Path(checkpoint).stem.split('_')[-1])
            rows = self.parallel(tasks,lambda item:self.execute(item[0],item[1],
                self.out/'validation'/f'e{epoch}'/item[0]['id']/f's{item[1]}',
                method='gnn',checkpoint=checkpoint))
            result = self.validation_report(rows,defaults,checkpoint,epoch,signature)
            write(self.out/'validation'/f'epoch_{epoch}.json',result)
            reports.append(result)
        selection = select_checkpoint(reports)
        if selection['checkpoint'] is None and self.config.get('exploratory_pilot_if_no_protected_checkpoint',False):
            available = [r for r in reports if r['unknown_count']==0 and r['total_runtime_ratio']<1e29]
            if available:
                chosen = min(available,key=lambda r:(-r['success_count'],r['total_runtime_ratio'],r['epoch']))
                selection = dict(status='exploratory_checkpoint_did_not_pass_protection',
                    checkpoint=chosen['checkpoint'],report=chosen,runtime_target_met=False)
        selection.update(checkpoint_sha256=sha(selection['checkpoint']) if selection['checkpoint'] else None,
                         all_reports=reports,teacher_loss_used_for_selection=False)
        freeze(self.out/'selection.json',selection)
        return selection

    def selected(self):
        selection = read(self.out/'selection.json')
        checkpoint = selection.get('checkpoint')
        if not checkpoint:
            raise RuntimeError('no validation-selected or pre-authorized exploratory checkpoint')
        if sha(checkpoint) != selection['checkpoint_sha256']:
            raise ValueError('selected checkpoint changed')
        return checkpoint

    def pilot(self):
        checkpoint = self.selected()
        tasks = pilot_tasks(self.conditions(),self.config['pilot_families'])
        freeze(self.out/'pilot/plan.json',dict(tasks=[dict(condition_id=t['condition']['id'],
            seed=t['seed'],method=t['method']) for t in tasks],count=len(tasks),
            checkpoint_sha256=sha(checkpoint),expected=480,
            development_only=True,selection_status=read(self.out/'selection.json')['status']))
        rows = self.parallel(tasks,lambda t:self.execute(t['condition'],t['seed'],
            self.out/'pilot'/t['condition']['id']/f's{t["seed"]}'/t['method'],
            method=t['method'],checkpoint=checkpoint if t['method']=='gnn' else None))
        rows.sort(key=lambda r:(r['condition_id'],r['seed'],r['method']))
        report = dict(planned=480,completed=len(rows),status_counts=dict(Counter(r['status'] for r in rows)),
                      actual_vpr_executions=sum(r.get('vpr_manifest') is not None for r in rows),rows=rows,
                      full_performance_acceptance='pending_independent_scan_and_controlled_timing')
        write(self.out/'pilot/report.json',report)
        return report

    def scan(self):
        checkpoint = self.selected()
        boundaries = [b for b in read(self.out/'boundaries.json')['records']
                      if b['family'] in self.config['pilot_families']]
        if len(boundaries)!=4 or any(b['status']!='paired' for b in boundaries):
            raise ValueError('five-seed scanning requires four pre-registered paired development designs')
        step = self.config.get('scan',{}).get('step',4)
        padding = self.config.get('scan',{}).get('padding',8)
        tasks,grids = [],{}
        for b in boundaries:
            low = max(step,((b['W_stress']-padding)//step)*step)
            high = math.ceil((b['W_ref']+padding)/step)*step
            grid = list(range(low,high+1,step))
            grids[b['design']] = grid
            for arch,width,seed,method in itertools.product(ARCHES,grid,range(531,536),METHODS):
                tasks.append(dict(condition=self.condition(b['record'],arch,width),seed=seed,method=method))
        if len(tasks)>self.config.get('scan',{}).get('max_tasks',20000):
            raise ValueError('frozen scan grid exceeds its explicit task budget')
        freeze(self.out/'scan/plan.json',dict(grids=grids,seeds=list(range(531,536)),
            methods=METHODS,count=len(tasks),checkpoint_sha256=sha(checkpoint),
            purpose='development grid-observed Wmin, not an unseen-family final test'))
        rows = self.parallel(tasks,lambda t:self.execute(t['condition'],t['seed'],
            self.out/'scan'/t['condition']['id']/f's{t["seed"]}'/t['method'],
            method=t['method'],checkpoint=checkpoint if t['method']=='gnn' else None))
        rows.sort(key=lambda r:(r['condition_id'],r['seed'],r['method']))
        summaries = []
        for b,arch,method in itertools.product(boundaries,ARCHES,METHODS):
            values,per_seed = [],[]
            for seed in range(531,536):
                trial = [r for r in rows if (r['design'],r['architecture'],r['seed'],r['method'])==
                         (b['design'],arch,seed,method)]
                successes = [r['width'] for r in trial if r['status']=='success']
                minimum = min(successes) if successes else None
                unknown = any(r['status'] in INFRA or r['status']=='timeout' for r in trial)
                per_seed.append(dict(seed=seed,minimum_observed_success=minimum,
                    left_censored=minimum==min(grids[b['design']]),no_success=minimum is None,
                    unknown_or_timeout=unknown,states={str(r['width']):r['status'] for r in trial}))
                if minimum is not None and not unknown:
                    values.append(minimum)
            summaries.append(dict(design=b['design'],architecture=arch,method=method,per_seed=per_seed,
                five_seed_mean=sum(values)/5 if len(values)==5 else None,
                all_five_observed=len(values)==5,physical_exact_Wmin_claimed=False))
        report = dict(rows=rows,summaries=summaries,mean_rule='arithmetic mean over all five seeds; missing seeds never dropped')
        write(self.out/'scan/report.json',report)
        return report

    def timing(self):
        """Balanced same-core three-method waves; cold context is really rebuilt."""
        checkpoint = self.selected()
        pilot = read(self.out/'pilot/report.json')['rows']
        conditions = {r['id']:r for r in self.conditions() if r['width_role']=='reference' and
                      r['family'] in self.config['pilot_families']}
        cases = []
        for condition in conditions.values():
            for seed in range(531,536):
                panel = [r for r in pilot if r['condition_id']==condition['id'] and r['seed']==seed]
                if len(panel)==3 and all(r['status']=='success' for r in panel):
                    cases.append((condition,seed))
        if not cases:
            raise RuntimeError('no complete three-method reference cases for controlled timing')
        root=self.out/'timing'/self.args.timing_id
        preflight=time.perf_counter()
        for condition in conditions.values():
            self.rr(condition['record'],condition['architecture'],condition['width'])
        write(root/'rrg_preflight.json',dict(measured_seconds=time.perf_counter()-preflight,
            scope='shared RRG integrity verification; before all three-method measurements'))
        cores=self.budget['physical_cores']
        divisors=[n for n in range(1,len(cores)+1) if len(cases)%n==0]
        chosen=max(divisors)
        if chosen>=min(8,len(cores)):
            cores=cores[:chosen]
        host = dict(uname=list(os.uname()) if hasattr(os,'uname') else [sys.platform],
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
                         if Path('/proc/sys/kernel/random/boot_id').exists() else None,
                    scheduler_job=os.environ.get('SLURM_JOB_ID'),
                    resource_budget={key:value for key,value in self.budget.items()
                                     if key not in ('available_mib','reduced')})
        freeze(root/'plan.json',dict(cases=[(c['id'],s) for c,s in cases],
            eligibility='predeclared reference width, all three pilot methods succeeded',
            omitted_case_count=80-len(cases),orders=[list(o) for o in ORDERS],
            primary='actual cold context + generation + native VPR',
            checkpoint_sha256=sha(checkpoint),cores=cores,host=host,
            equal_wave_choice='largest sufficient divisor; otherwise tail load reported separately'))
        rows = []
        for start in range(0,len(cases),len(cores)):
            wave = cases[start:start+len(cores)]
            wave_root=root/f'wave_{start//len(cores):04d}'
            wave_root.mkdir(parents=True,exist_ok=True)
            signature=dict(cases=[(c['id'],s) for c,s in wave],cores=cores[:len(wave)],
                           checkpoint=sha(checkpoint),host=host,timeout=self.args.timeout)
            complete=wave_root/'completed.json'
            if complete.exists():
                previous=read(complete)
                if previous['signature']!=signature:
                    raise ValueError('timing wave changed host/budget/input; use a new timing-id')
                rows.extend(previous['rows'])
                emit('timing_wave_cached',wave=start//len(cores),whole_wave_only=True)
                continue
            attempt=wave_root/f'attempt_{len(list(wave_root.glob("attempt_*")))+1:04d}'
            attempt.mkdir()
            barrier = threading.Barrier(len(wave))
            def lane(item):
                index,(condition,seed) = item
                core = cores[index]
                order = ORDERS[(start+index)%len(ORDERS)]
                lane_rows = []
                try:
                    for slot,method in enumerate(order):
                        barrier.wait(timeout=self.args.timeout*3)
                        folder = attempt/f'lane_{index:03d}'/method
                        working = dict(condition)
                        if method!='default':
                            # The extra AP extraction/graph is design work, never
                            # hidden inside a reusable architecture cache.
                            record = dict(condition['record'])
                            prefix = folder/'cold_structure'
                            arch = self.architecture(record,condition['architecture'])
                            self.rr(record,condition['architecture'],condition['width'])
                        begun = time.perf_counter()
                        if method!='default':
                            run = run_trial(folder/'structure_run',
                                vpr_command(self.binary,arch,record['blif'],condition['width'],701)+['--device','pilot'],
                                inputs=[self.binary_reference,arch,record['blif']],
                                env=dict(TC_GP_STRUCTURE=prefix,TC_GP_STRUCTURE_ONLY=1),
                                marker='TAILCOMPILE_V7_STRUCTURE_ONLY_COMPLETE',affinity=core,
                                timeout=self.args.timeout,prune=True)
                            if run['status']!='complete':
                                barrier.abort()
                                raise RuntimeError('cold logical preparation failed')
                            record.update(structure_prefix=str(prefix),structure_export_seconds=0.)
                            working['record']=record
                            working['graph_root']=str(folder/'cold_graph')
                            self.graph(working)
                        row = self.execute(working,seed,folder/'execution',method=method,
                            checkpoint=checkpoint if method=='gnn' else None,affinity=core)
                        row.update(total_seconds=time.perf_counter()-begun,
                            measured_cold_total=True,physical_core=core,wave=start//len(cores),
                            wave_load=len(wave),slot=slot,order=list(order),host=host,
                            reusable_RRG_preprocessing_excluded=True,
                            controller_signature_IO_included=True)
                        write(folder/'timed_result.json',row)
                        lane_rows.append(row)
                        barrier.wait(timeout=self.args.timeout*3)
                except BaseException:
                    barrier.abort()
                    raise
                return lane_rows
            with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                groups = list(pool.map(lane,enumerate(wave)))
            wave_rows=[r for group in groups for r in group]
            freeze(complete,dict(signature=signature,attempt=str(attempt),rows=wave_rows,
                                 incomplete_attempts_preserved=True))
            rows.extend(wave_rows)
            write(root/'report.json',dict(rows=rows,primary_runtime='cold',
                completed_cases=len(rows)//3,planned_cases=len(cases),timing_id=self.args.timing_id,
                wave_loads=sorted({r['wave_load'] for r in rows}),
                pilot_failure_counts=read(self.out/'pilot/report.json')['status_counts']))
        result=dict(rows=rows,primary_runtime='cold',completed_cases=len(rows)//3,
                    planned_cases=len(cases),timing_id=self.args.timing_id,
                    wave_loads=sorted({r['wave_load'] for r in rows}),
                    pilot_failure_counts=read(self.out/'pilot/report.json')['status_counts'])
        write(root/'report.json',result)
        write(self.out/'timing/report.json',result)
        return rows

    def resident(self):
        """Actual one-process generation requests; not a synthetic import subtraction."""
        checkpoint = self.selected()
        conditions = [r for r in self.conditions() if r['family'] in self.config['pilot_families']]
        count = self.config.get('resident_requests',16)
        if count<1 or not conditions:
            raise ValueError('positive resident request count and prepared conditions required')
        for method in ('capacity_only','gnn'):
            folder = self.out/'resident'/method
            requests = []
            for index in range(count):
                condition = conditions[index % len(conditions)]
                graph = self.graph(condition)
                device = self.rr(condition['record'],condition['architecture'],condition['width'])['device']
                requests.append(dict(condition_id=condition['id'],graph=graph['graph'],logic=graph['logic'],
                    device=device,method=method,checkpoint=checkpoint if method=='gnn' else None,
                    seed=0,density_passes=self.config.get('density_passes',2)))
            freeze(folder/'requests.json',dict(requests=requests))
            row = run_trial(folder/'worker',
                [sys.executable,HERE/'campaign.py','generate-worker','--batch',folder/'requests.json','--out',folder/'payload'],
                inputs=[folder/'requests.json',checkpoint,HERE/'model.py',HERE/'generation.py'],
                timeout=max(120,count*self.config.get('generation_timeout',120)),
                marker='TAILCOMPILE_V7_RESIDENT_COMPLETE',prune=False)
            if row['status']!='complete':
                raise RuntimeError(f'resident cost experiment failed: {row["log"]}')
            report = read(folder/'payload/resident_report.json')
            report.update(actual_process_wall_seconds=row['metrics']['wall_seconds'],
                per_request_amortized_process_seconds=row['metrics']['wall_seconds']/count,
                primary_runtime_acceptance=False,
                scope='actual resident generation only; extra AP preparation and VPR timing remain separate')
            write(folder/'report.json',report)

    def report(self):
        pilot = read(self.out/'pilot/report.json')
        rows = pilot['rows']
        default = {(r['condition_id'],r['seed']):r for r in rows if r['method']=='default'}
        learned = [r for r in rows if r['method']=='gnn']
        quality = defaultdict(list)
        success_delta = defaultdict(int)
        for row in learned:
            base = default[row['condition_id'],row['seed']]
            key = (row['family'],row['architecture'],row['width_role'])
            success_delta[key] += int(row['status']=='success')-int(base['status']=='success')
            if row['status']=='success' and base['status']=='success':
                quality[key].append((row,base))
        protection = [dict(family=k[0],architecture=k[1],width_role=k[2],
            cpd_ratio=_gmean([a['cpd_ns']/b['cpd_ns'] for a,b in v]),
            wirelength_ratio=_gmean([a['wirelength']/b['wirelength'] for a,b in v]))
            for k,v in quality.items()]
        success_ok = all(delta>=0 for delta in success_delta.values())
        quality_ok = len(protection)==32 and all(r['cpd_ratio']<=1.03 and r['wirelength_ratio']<=1.03 for r in protection)
        unknown = any(r['status'] in INFRA for r in rows)
        runtime_ratio,wmin_ratio = None,None
        runtime_by_family = {}
        learned_over_capacity={}
        timing_file = self.out/'timing/report.json'
        if timing_file.exists():
            timed = read(timing_file)['rows']
            base = {(r['condition_id'],r['seed']):r for r in timed if r['method']=='default'}
            capacity_base={(r['condition_id'],r['seed']):r for r in timed if r['method']=='capacity_only'}
            grouped = defaultdict(list)
            capacity_grouped=defaultdict(list)
            for row in timed:
                if row['method']=='gnn' and (row['condition_id'],row['seed']) in base:
                    other = base[row['condition_id'],row['seed']]
                    if row['status']=='success' and other['status']=='success':
                        grouped[(row['family'],row['wave_load'])].append(row['total_seconds']/other['total_seconds'])
                    simple=capacity_base.get((row['condition_id'],row['seed']))
                    if simple and row['status']=='success' and simple['status']=='success':
                        capacity_grouped[(row['family'],row['wave_load'])].append(row['total_seconds']/simple['total_seconds'])
            loads = {load for family,load in grouped}
            if len(loads)==1:
                runtime_by_family={family:_gmean(values) for (family,load),values in grouped.items()}
                if len(runtime_by_family)==4:
                    runtime_ratio=_gmean(list(runtime_by_family.values()))
                learned_over_capacity={family:_gmean(values) for (family,load),values in capacity_grouped.items()}
        scan_file = self.out/'scan/report.json'
        wmin_unknown=True
        if scan_file.exists():
            scan=read(scan_file)['summaries']
            base={(r['design'],r['architecture']):r for r in scan if r['method']=='default'}
            ratios=[];wmin_unknown=False
            for row in [r for r in scan if r['method']=='gnn']:
                other=base[row['design'],row['architecture']]
                if not row['all_five_observed'] or not other['all_five_observed']:
                    wmin_unknown=True;continue
                ratios.append(row['five_seed_mean']/other['five_seed_mean'])
                if any(r['left_censored'] for r in row['per_seed']+other['per_seed']):
                    wmin_unknown=True
            if len(ratios)==16:
                wmin_ratio=max(ratios)
        passed=(not unknown and success_ok and quality_ok and runtime_ratio is not None and
                runtime_ratio<=.95 and not wmin_unknown and wmin_ratio is not None and wmin_ratio<=1.03)
        report=dict(scope='four-family development pilot; not new-family generalization',
            success_protection_passed=success_ok,quality_protection_passed=quality_ok,
            unknown_infrastructure=unknown,runtime_ratio=runtime_ratio,
            runtime_by_family=runtime_by_family,wmin_max_design_architecture_ratio=wmin_ratio,
            learned_over_capacity_runtime_by_family=learned_over_capacity,
            learned_over_capacity_runtime_ratio=_gmean(list(learned_over_capacity.values()))
                if len(learned_over_capacity)==4 else None,
            wmin_protection_unknown=wmin_unknown,quality_strata=protection,
            complete_acceptance_passed=passed,
            status='development_targets_passed' if passed else
                   'development_targets_not_passed' if not success_ok or (len(protection)==32 and not quality_ok) else
                   'pending_independent_timing_or_scan' if runtime_ratio is None or wmin_unknown else 'development_targets_not_passed',
            thresholds=dict(total_runtime_ratio=.95,cpd_ratio=1.03,wirelength_ratio=1.03,mean_Wmin_ratio=1.03))
        write(self.out/'acceptance_report.json',report)
        emit('development_acceptance',**report)
        return report


def generation_worker():
    parser = argparse.ArgumentParser()
    group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--request',type=Path)
    group.add_argument('--batch',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args(sys.argv[2:])
    args.out.mkdir(parents=True,exist_ok=True)
    if args.batch:
        return resident_worker(args.batch,args.out)
    try:
        import torch
        from model import CapacityError, Hierarchical3DInitializer, SCHEMA, capacity_only
        from train import load_graph
        from generation import lower_initialization
        torch.set_num_threads(1)
        request = read(args.request)
        graph = load_graph(request['graph'])
        if request['method']=='capacity_only':
            prediction = capacity_only(graph,density_passes=request['density_passes'])
        else:
            checkpoint = torch.load(request['checkpoint'],map_location='cpu',weights_only=False)
            if checkpoint.get('schema')!=SCHEMA:
                raise ValueError('checkpoint is not the registered 3D generator')
            model = Hierarchical3DInitializer(**checkpoint['config'])
            model.load_state_dict(checkpoint['state_dict']);model.eval()
            prediction = model.predict(graph,seed=request['seed'],
                score_noise_std=request['score_noise_std'],coordinate_noise_std=request['coordinate_noise_std'],
                density_passes=request['density_passes'])
        generated=lower_initialization(read(request['logic']),read(request['device']),prediction,args.out,
                                       method=request['method'])
        generated['target_sha256']=sha(args.out/'target.json')
        write(args.out/'generation.json',generated)
    except Exception as exc:
        status = 'decode_failure' if type(exc).__name__ in ('CapacityError','ValueError') else 'infrastructure_failure'
        write(args.out/'initialization_failure.json',dict(status=status,type=type(exc).__name__,error=str(exc)))
        print('TAILCOMPILE_V7_INITIALIZATION_FAILURE '+json.dumps(dict(status=status,type=type(exc).__name__,error=str(exc))),flush=True)
        raise
    print('TAILCOMPILE_V7_GENERATION_COMPLETE',flush=True)


def resident_worker(batch,output):
    begun=time.perf_counter()
    import torch
    from model import Hierarchical3DInitializer,capacity_only
    from train import load_graph
    from generation import lower_initialization
    torch.set_num_threads(1)
    requests=read(batch)['requests']
    first=requests[0]
    model=None
    if first['method']=='gnn':
        saved=torch.load(first['checkpoint'],map_location='cpu',weights_only=False)
        model=Hierarchical3DInitializer(**saved['config'])
        model.load_state_dict(saved['state_dict']);model.eval()
    startup=time.perf_counter()-begun
    embeddings,devices={},{}
    results=[]
    for index,request in enumerate(requests):
        if (request['method'],request.get('checkpoint'))!=(first['method'],first.get('checkpoint')):
            raise ValueError('one resident worker must keep the same method/checkpoint')
        start=time.perf_counter()
        graph=load_graph(request['graph'])
        key=request['device']
        if key not in devices:
            devices[key]=read(key)
        device_hit=key in embeddings
        encode_seconds=0.
        if model is not None:
            if not device_hit:
                encode_start=time.perf_counter()
                with torch.no_grad():
                    embeddings[key]=model.encode_device(graph)
                encode_seconds=time.perf_counter()-encode_start
            prediction=model.predict(graph,seed=request['seed'],density_passes=request['density_passes'],
                                     device_embedding=embeddings[key])
        else:
            prediction=capacity_only(graph,density_passes=request['density_passes'])
        generated=lower_initialization(read(request['logic']),devices[key],prediction,
                                       output/f'request_{index:04d}',method=request['method'])
        results.append(dict(condition_id=request['condition_id'],request=index,
            request_seconds=time.perf_counter()-start,device_embedding_cache_hit=device_hit,
            new_device_encoding_seconds=encode_seconds,initializer=generated))
    write(output/'resident_report.json',dict(startup_seconds=startup,requests=results,
        request_count=len(results),sum_request_seconds=sum(r['request_seconds'] for r in results),
        independent_process_start_per_request=False,teacher_coordinates_read=False))
    print('TAILCOMPILE_V7_RESIDENT_COMPLETE',flush=True)


def main():
    install_signal_handlers()
    if len(sys.argv)>1 and sys.argv[1]=='generate-worker':
        return generation_worker()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('plan','prepare','calibrate','teachers','pretrain','feedback',
                                        'finetune','validate','pilot','scan','timing','resident','report'))
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--workspace',type=Path,default=SERVER_ROOT)
    parser.add_argument('--vtr-root',type=Path,required=True)
    parser.add_argument('--mapping-vtr-root',type=Path,
                        help='read-only same-commit full VTR with ABC/Yosys; defaults to vtr-root')
    parser.add_argument('--binary',type=Path,required=True)
    parser.add_argument('--backend-manifest',type=Path,required=True)
    parser.add_argument('--qualification-report',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--scratch',type=Path,required=True)
    parser.add_argument('--jobs',type=int,default=32)
    parser.add_argument('--rrg-jobs',type=int,default=4)
    parser.add_argument('--per-job-mib',type=int,default=4096)
    parser.add_argument('--rrg-mib',type=int,default=4096)
    parser.add_argument('--timeout',type=int,default=600)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--timing-id',default='paired_001')
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--allow-local',action='store_true')
    args = parser.parse_args()
    if min(args.jobs,args.rrg_jobs,args.per_job_mib,args.rrg_mib,args.timeout)<1:
        parser.error('positive resources and timeout required')
    if not args.timing_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.timing_id):
        parser.error('timing-id must be a plain alphanumeric label')
    within(args.out,args.workspace)
    validate_output_roots((args.out,args.scratch),
        dependency_roots=(args.vtr_root,args.mapping_vtr_root or args.vtr_root))
    with campaign_lock(args.out,args.stage):
        dispatch(args)


def dispatch(args):
    campaign = Campaign(args)
    if args.stage=='plan':
        ntrain=sum(r['split']=='train' for r in campaign.config['designs'])
        nvalidation=sum(r['split']=='validation' for r in campaign.config['designs'])
        validation_conditions=nvalidation*4*2*len(campaign.config.get('validation_seeds',[531,532]))
        plan=dict(train_families=ntrain,teacher_runs=len(campaign.config['designs'])*4*2*2,
            feedback_candidate_runs=ntrain*4*2*4*2,feedback_default_runs=ntrain*4*2*2,
            default_calibration_upper=len(campaign.config['designs'])*4*2*len(campaign.config['width_grid']),
            validation_candidate_upper=3*validation_conditions,validation_default_runs=validation_conditions,
            pilot_runs=480,formal_wmin_and_timing_separate=True,
            scan_task_count='derived from Default-only frozen grid before dispatch',
            controlled_timing_pnr_upper=4*4*5*3,qualification_runs_separate=True,
            all_records_require_qualification=True,current_phase='planned_not_executed')
        write(campaign.out/'queue_plan.json',plan)
        emit('queue_plan',**plan)
    elif args.stage=='pretrain':
        campaign.train(False)
    elif args.stage=='finetune':
        campaign.train(True)
    else:
        campaign.gate()
        getattr(campaign,args.stage)()


if __name__=='__main__':
    main()
