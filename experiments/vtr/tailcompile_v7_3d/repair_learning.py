"""006: exact graph repair, matched reward ablation and bounded real-VPR retest.

004 supplies qualified inputs; 005 supplies immutable teachers/checkpoints.
No calibration, architecture change, new teacher, or 2D modification.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import math
import os
from pathlib import Path
import re
import sys
import threading
import time
from types import SimpleNamespace

import qualified_learning as qualified
from campaign import campaign_lock, freeze, identity, within
from io_utils import (emit, install_signal_handlers, read, run_trial, sha,
                      vpr_command, write)
from repository_guard import validate_output_roots

HERE, REPO = qualified.HERE, qualified.REPO
STAGES = ('prepare', 'feedback', 'train', 'validate', 'compare', 'report')
POLICIES = ('clipped', 'guarded')
METHODS = ('default', 'capacity_only', 'frozen_gnn', 'clipped', 'guarded')
COMPARE_SEEDS = (561, 562)
VALIDATION_SEEDS = (541, 542)


def plan():
    return dict(schema='learning-repair-006', epochs_per_arm=3, arms=list(POLICIES),
        same_pretrained_checkpoint=True, same_feedback=True, same_training_order=True,
        graph_inputs_and_model_schema_unchanged=True, teacher_recollection=0, calibration_runs=0,
        feedback_runs=240, validation_runs=80, comparison_runs=200,
        comparison='5 designs x 4 architectures x reference width x 2 seeds x 5 methods',
        compare_seeds=list(COMPARE_SEEDS), validation_seeds=list(VALIDATION_SEEDS),
        thresholds=dict(runtime=.95, cpd=1.03, wirelength=1.03),
        scope='development repair retest; no stress/Wmin/final acceptance',
        old_005_results_rewritten=False, full_acceptance_passed=False)


def new_paths(source, prior, out):
    source, out = qualified.validate_paths(source, out)
    prior = Path(prior).resolve()
    runs = HERE/'runs'
    if runs not in prior.parents:
        raise ValueError('prior learning cohort must be inside v7 runs')
    if any(a == b or a in b.parents or b in a.parents
           for a, b in itertools.combinations((source, prior, out), 2)):
        raise ValueError('004, 005 and new output must be disjoint')
    if out.exists() and (out/'protocol.json').exists() and not (out/'repair_protocol.json').exists():
        raise ValueError('cannot adopt an existing non-repair output cohort')
    return source, prior, out


def prepare_catalog(source, prior, out):
    source, prior, out = new_paths(source, prior, out)
    # Existing helper validates the unchanged frozen core, binary and 004 H/Z panel.
    catalog = qualified.prepare_handoff(source, out)
    selection = read(prior/'selection.json')
    pretrain = read(prior/'models/pretrain/completed.json')
    pretrained = Path(pretrain['report']['checkpoints'][-1])
    frozen_model = Path(selection['checkpoint'])
    qualified.require_hash(pretrained, pretrain['checkpoint_sha256'][str(pretrained)])
    qualified.require_hash(frozen_model, selection['checkpoint_sha256'])
    # Verify the 005 dependency belongs to this exact qualified input/binary.
    old_catalog = read(prior/'qualified_source_manifest.json')
    if (old_catalog['source'] != str(source) or old_catalog['devices'] != catalog['devices'] or
            old_catalog['files'] != catalog['files'] or
            old_catalog['architectures'] != catalog['architectures']):
        raise ValueError('005 is not the same qualified source panel')
    teacher_records = read(prior/'teacher_records.json')['records']
    conditions = read(out/'conditions.json')['records']
    expected = {c['id']: c for c in conditions}
    if len(teacher_records) != len(expected) or {r['condition_id'] for r in teacher_records} != expected.keys():
        raise ValueError('005 teachers do not cover exactly the qualified conditions')
    teachers = {}
    for row in teacher_records:
        c = expected[row['condition_id']]
        if row['split'] != c['split'] or row['family'] != c['family'] or row['teacher_seed'] != 511:
            raise ValueError('teacher split/family/seed changed')
        for key in ('graph', 'teacher'):
            within(row[key], REPO)
        teachers[row['condition_id']] = dict(row, graph_sha256=sha(row['graph']),
                                            teacher_sha256=sha(row['teacher']))
    # Check pretraining data against its recorded payload hashes, not just newly
    # hashing whatever happens to be at the old path today.
    signature = read(prior/'models/pretrain/input_signature.json')
    qualified.require_hash(prior/'teacher_records.json', signature['records_sha256'])
    for cid, hashes in signature['record_payloads'].items():
        for field in ('graph', 'teacher'):
            if teachers[cid][field+'_sha256'] != hashes[field]:
                raise ValueError('pretrained teacher/graph payload changed')
    names = ('repair_learning.py', 'run_learning_repair.sh', 'fast_graph.py',
             'reward_v2.py', 'train_repair.py')
    protocol = dict(plan=plan(), prior=str(prior), source=str(source),
        pretrained=str(pretrained), pretrained_sha256=sha(pretrained),
        frozen_model=str(frozen_model), frozen_model_sha256=sha(frozen_model),
        prior_selection_sha256=sha(prior/'selection.json'), teachers=teachers,
        qualified_catalog_sha256=sha(out/'qualified_source_manifest.json'),
        implementation={name: sha(HERE/name) for name in names})
    freeze(out/'repair_protocol.json', protocol)
    freeze(out/'queue_plan.json', plan())
    return catalog, protocol


def graph_request(campaign, condition, engine='fast'):
    r = condition['record']
    rr = campaign.rr(r, condition['architecture'], condition['width'])
    return dict(prefix=r['structure_prefix'], design=r['design'], family=r['family'],
        split=r['split'], blif=r['blif'], arch=str(campaign.architecture(r, condition['architecture'])),
        device=rr['device'], engine=engine)


def graph_process(request, folder, *, affinity, timeout):
    """Independent interpreter, with startup and integrity IO inside wall time."""
    folder = Path(folder)
    if (folder/'worker/manifest.json').exists() and not (folder/'context_cost.json').exists():
        raise RuntimeError('graph worker completed without durable cost; use a fresh output, never time a cache hit')
    begun = time.perf_counter()
    freeze(folder/'request.json', request)
    inputs = [folder/'request.json', HERE/'fast_graph.py', HERE/'data.py',
              request['blif'], request['arch'], request['device']]
    inputs += [str(request['prefix'])+s for s in ('.blocks.tsv', '.nets.tsv', '.meta.tsv')]
    trial = run_trial(folder/'worker', [sys.executable, HERE/'fast_graph.py',
        '--request', folder/'request.json', '--out', folder/'payload'],
        inputs=inputs, affinity=affinity, timeout=timeout, prune=False,
        marker='TAILCOMPILE_V7_FAST_GRAPH_COMPLETE')
    if trial['status'] != 'complete':
        raise RuntimeError(f'graph worker {trial["status"]}; log={trial["log"]}')
    row = read(folder/'payload/graph_manifest.json')
    qualified.require_hash(row['graph'], row['graph_sha256'])
    qualified.require_hash(row['logic'], row['logic_sha256'])
    cost = folder/'context_cost.json'
    if not cost.exists():
        write(cost, dict(measured_graph_build_seconds=time.perf_counter()-begun,
            worker_process_seconds=trial['metrics']['wall_seconds'],
            reusable_rrg_preprocessing_excluded=True, profile=row['profile']))
    return row


def prune_temporary_graphs(root, owner):
    """Only redundant profile/cold-measurement payloads, after durable reports.

    Training graphs, teacher targets, actions, checkpoints and 004/005 remain.
    The deletion ledger is written before unlinking so interrupted cleanup can
    be safely completed without losing the list of intended files.
    """
    root,owner=Path(root).resolve(),Path(owner).resolve()
    if owner not in root.parents or root.relative_to(owner).parts[0] not in ('profile','comparison'):
        raise ValueError('cleanup must target owned profile/comparison subtree')
    ledger=root/'payload_pruned.json'
    if ledger.exists():
        rows=read(ledger)['files']
    else:
        paths=[p for p in root.rglob('*') if p.is_file() and
               ((p.parent.name=='payload' and p.name in ('graph.pt','logic.json')) or
                p.name in ('cold_structure.blocks.tsv','cold_structure.nets.tsv','cold_structure.meta.tsv'))]
        rows=[]
        for path in paths:
            if path.is_symlink() or root not in path.resolve().parents:
                raise ValueError('cleanup payload escapes owner')
            rows.append(dict(path=str(path),bytes=path.stat().st_size,sha256=sha(path)))
        write(ledger,dict(files=rows,removed_bytes=sum(r['bytes'] for r in rows),
            reason='durable graph profile/equivalence or completed timing wave; duplicate graph/snapshot only'))
    for row in rows:
        path=Path(row['path'])
        if path.is_symlink() or root not in path.resolve().parents:
            raise ValueError('cleanup ledger path escapes owner')
        if path.exists():
            qualified.require_hash(path,row['sha256'])
            path.unlink()


class RepairCampaign(qualified.QualifiedCampaign):
    def __init__(self, args, catalog, protocol):
        self.repair = protocol
        self.graph_memory = {}
        super().__init__(args, catalog)

    def graph(self, condition):
        folder = Path(condition.get('graph_root', self.out/'graphs'/condition['id']))
        request = graph_request(self, condition)
        with self._lock('repair-graph:'+str(folder)):
            key = str(folder)
            if key in self.graph_memory:
                row, stats, old_request = self.graph_memory[key]
                if request != old_request or stats != tuple(self._stat(p) for p in (row['graph'], row['logic'])):
                    raise ValueError('graph payload/request changed during stage')
                return row
            row = graph_process(request, folder, affinity=condition.get('_affinity'), timeout=self.args.timeout)
            cost = read(folder/'context_cost.json')
            structure = condition['record'].get('structure_export_seconds')
            if structure is None:
                raise ValueError('missing AP structure export cost')
            if 'structure_export_seconds' in cost and cost['structure_export_seconds'] != structure:
                raise ValueError('graph cost source changed')
            cost['structure_export_seconds'] = structure
            write(folder/'context_cost.json', cost)
            self.graph_memory[key] = (row, tuple(self._stat(p) for p in (row['graph'], row['logic'])), request)
            return row

    def prepare(self):
        import torch
        from fast_graph import semantic_digest
        torch.set_num_threads(1)
        def check(condition):
            new = self.graph(condition)
            old = self.repair['teachers'][condition['id']]
            graph = torch.load(old['graph'], map_location='cpu', weights_only=False)
            same = semantic_digest(graph) == new['semantic_sha256']
            del graph
            if not same:
                raise ValueError('optimized graph differs from 005: '+condition['id'])
            return dict(condition_id=condition['id'], family=condition['family'], split=condition['split'],
                graph=new['graph'], teacher=old['teacher'], teacher_seed=511,
                graph_semantics_equal=True, old_graph_sha256=old['graph_sha256'],
                graph_sha256=new['graph_sha256'], teacher_sha256=old['teacher_sha256'])
        records = sorted(self.parallel(self.conditions(), check), key=lambda r: r['condition_id'])
        freeze(self.out/'teacher_records.json', dict(records=records, borrowed_immutable_005_targets=True))
        write(self.out/'graph_equivalence_report.json', dict(complete=len(records)==40,
            checked_conditions=len(records), all_equal=all(r['graph_semantics_equal'] for r in records),
            checkpoint_and_decoder_schema_unchanged=True))
        # Pin the same 005 teacher anchor; never recollect or choose a new one.
        freeze(self.out/'models/pretrain/completed.json', dict(report=dict(checkpoints=[self.repair['pretrained']]),
            checkpoint_sha256={self.repair['pretrained']:self.repair['pretrained_sha256']}))
        if (self.out/'graph_profile_report.json').exists():
            prune_temporary_graphs(self.out/'profile',self.out)
            return records
        # Five predeclared workloads; paired engines on the same core, then
        # concurrent lanes. Old 005 timings are NOT a controlled baseline here.
        workload = [c for c in sorted(self.conditions(), key=lambda c:c['id'])
                    if c['architecture']=='H0Z0' and c['width_role']=='reference']
        rows = []
        cores = self.budget['physical_cores']
        host = dict(uname=list(os.uname()), boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        freeze(self.out/'profile/plan.json', dict(host=host, cores=cores[:min(4,len(cores))],
            conditions=[c['id'] for c in workload], repetitions=2, engines=['legacy','fast']))
        def profile_item(item):
            c, mode, repetition, core, order, load = item
            result = []
            for engine in order:
                folder = self.out/'profile'/mode/c['design']/f'r{repetition}'/engine
                row = graph_process(graph_request(self,c,engine), folder, affinity=core, timeout=self.args.timeout)
                if row['semantic_sha256'] != read(self.out/'graphs'/c['id']/'payload/graph_manifest.json')['semantic_sha256']:
                    raise ValueError('profile engine graph mismatch')
                result.append(dict(design=c['design'], mode=mode, repetition=repetition,
                    engine=engine, core=core, wave_load=load, **read(folder/'context_cost.json')))
            return result
        for rep in range(2):
            for i,c in enumerate(workload):
                rows += profile_item((c,'serial',rep,cores[0],('legacy','fast') if (i+rep)%2==0 else ('fast','legacy'),1))
        count=min(4,len(cores))
        for rep in range(2):
            for start in range(0,len(workload),count):
                wave=workload[start:start+count]
                items=[(c,'parallel',rep,cores[i],('legacy','fast') if (i+rep)%2==0 else ('fast','legacy'),len(wave))
                       for i,c in enumerate(wave)]
                with ThreadPoolExecutor(max_workers=len(wave)) as pool:
                    rows += [r for group in pool.map(profile_item,items) for r in group]
        write(self.out/'graph_profile_report.json', dict(rows=rows,
            note='both engines isolated in subprocesses; concurrency workload is recorded, not old threaded 005',
            graph_equivalence_passed=True))
        prune_temporary_graphs(self.out/'profile',self.out)

    def feedback(self):
        value = super().feedback()
        from train import outcome_reward as clipped
        from reward_v2 import outcome_reward, pairs
        by_condition = defaultdict(list)
        audit = []
        for entry in value['records']:
            old = clipped(entry['trials'], entry['default_trials'])
            new = outcome_reward(entry['trials'], entry['default_trials'])
            if entry.get('action') is None:
                old = dict(eligible=False,reason='missing_action')
                new = dict(eligible=False,reason='missing_action')
            audit.append(dict(condition_id=entry['condition_id'],candidate_id=entry['candidate_id'],clipped=old,guarded=new))
            by_condition[entry['condition_id']].append(new)
        count=sum(len(pairs(group)) for group in by_condition.values())
        write(self.out/'feedback_signal_report.json',dict(records=audit,guarded_preference_pairs=count,
            raw_runtime_preserved=True,failure_runtime_imputed=False,training_ready=count>0))
        if not count:
            raise RuntimeError('no candidate preferences after guard/deadband; inspect feedback_signal_report.json')
        return value

    def train_arms(self):
        import torch
        from train_repair import train
        torch.set_num_threads(1)
        records=[r for r in read(self.out/'teacher_records.json')['records'] if r['split']=='train']
        for record in records:
            for key in ('graph','teacher'):
                qualified.require_hash(record[key],record[key+'_sha256'])
        feedback=read(self.out/'feedback.json')['records']
        if not read(self.out/'feedback_signal_report.json')['training_ready']:
            raise ValueError('feedback failed signal gate')
        for policy in POLICIES:
            root=self.out/'models'/policy
            signature=dict(records_sha256=sha(self.out/'teacher_records.json'),feedback_sha256=sha(self.out/'feedback.json'),
                checkpoint_sha256=self.repair['pretrained_sha256'],policy=policy,epochs=3,seed=601,
                payloads={r['condition_id']:{k:sha(r[k]) for k in ('graph','teacher')} for r in records})
            freeze(root/'input_signature.json',signature)
            if (root/'completed.json').exists():
                for p,digest in read(root/'completed.json')['checkpoint_sha256'].items():
                    qualified.require_hash(p,digest)
                continue
            attempt=root/f'attempt_{len(list(root.glob("attempt_*")))+1:04d}'
            config=self.config['training']
            report=train(records,attempt,epochs=3,seed=601,checkpoint=self.repair['pretrained'],
                hidden=config.get('hidden',64),learning_rate=config.get('learning_rate',.001),
                feedback=feedback,feedback_weight=config.get('feedback_weight',.2),device=self.args.device,policy=policy)
            write(root/'completed.json',dict(report=report,checkpoint_sha256={p:sha(p) for p in report['checkpoints']}))

    def validate(self):
        from supervision import select_checkpoint
        conditions=[c for c in self.conditions() if c['split']=='validation' and c['width_role']=='reference']
        tasks=[(c,s) for c in conditions for s in VALIDATION_SEEDS]
        signature=identity([(c['id'],s) for c,s in tasks])
        defaults=self.parallel(tasks,lambda item:self.execute(item[0],item[1],
            self.out/'validation/default'/item[0]['id']/f's{item[1]}'))
        for policy in POLICIES:
            reports=[]
            completed=read(self.out/'models'/policy/'completed.json')
            checkpoints=completed['report']['checkpoints']
            freeze(self.out/'validation'/policy/'plan.json',dict(tasks=[(c['id'],s) for c,s in tasks],
                checkpoints=completed['checkpoint_sha256'],reference_only=True))
            for checkpoint in checkpoints:
                qualified.require_hash(checkpoint,completed['checkpoint_sha256'][checkpoint])
                epoch=int(Path(checkpoint).stem.split('_')[-1])
                rows=self.parallel(tasks,lambda item:self.execute(item[0],item[1],
                    self.out/'validation'/policy/f'e{epoch}'/item[0]['id']/f's{item[1]}',method='gnn',checkpoint=checkpoint))
                report=self.validation_report(rows,defaults,checkpoint,epoch,signature)
                write(self.out/'validation'/policy/f'epoch_{epoch}_report.json',report)
                reports.append(report)
            selected=select_checkpoint(reports)
            if selected['checkpoint'] is None:
                valid=[r for r in reports if r['unknown_count']==0 and r['total_runtime_ratio']<1e29]
                if valid:
                    chosen=min(valid,key=lambda r:(-r['success_count'],r['total_runtime_ratio'],r['epoch']))
                    selected=dict(status='exploratory_checkpoint_did_not_pass_protection',
                                  checkpoint=chosen['checkpoint'],report=chosen,runtime_target_met=False)
            selected.update(checkpoint_sha256=sha(selected['checkpoint']) if selected['checkpoint'] else None,
                            all_reports=reports,teacher_loss_used_for_selection=False,reference_only=True)
            freeze(self.out/'validation'/policy/'selection.json',selected)

    def checkpoints(self):
        value={'frozen_gnn': self.repair['frozen_model']}
        qualified.require_hash(value['frozen_gnn'],self.repair['frozen_model_sha256'])
        for policy in POLICIES:
            selected=read(self.out/'validation'/policy/'selection.json')
            if selected['checkpoint'] is None:
                raise ValueError('no measurable checkpoint for '+policy)
            qualified.require_hash(selected['checkpoint'],selected['checkpoint_sha256'])
            value[policy]=selected['checkpoint']
        return value

    def compare(self):
        checkpoints=self.checkpoints()
        conditions=sorted([c for c in self.conditions() if c['width_role']=='reference'],key=lambda c:c['id'])
        cases=[(c,s) for c in conditions for s in COMPARE_SEEDS]
        cores=self.budget['physical_cores']
        count=max(n for n in range(1,min(len(cores),len(cases))+1) if len(cases)%n==0)
        cores=cores[:count]
        host=dict(uname=list(os.uname()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cores=cores)
        root=self.out/'comparison'
        # Each of five methods occurs equally often in each order position.
        orders=[list(METHODS[i:]+METHODS[:i]) for i in range(len(METHODS))]
        orders += [list(reversed(order)) for order in orders]
        freeze(root/'plan.json',dict(cases=[(c['id'],s) for c,s in cases],methods=list(METHODS),orders=orders,
            checkpoints={k:sha(v) for k,v in checkpoints.items()},host=host,lanes=count,expected=len(cases)*len(METHODS)))
        for c in conditions:
            self.rr(c['record'],c['architecture'],c['width'])
        rows=[]
        for start in range(0,len(cases),count):
            wave=cases[start:start+count]
            folder=root/f'wave_{start//count:04d}'
            signature=dict(plan_sha256=sha(root/'plan.json'),cases=[(c['id'],s) for c,s in wave])
            if (folder/'completed.json').exists():
                done=read(folder/'completed.json')
                if done['signature']!=qualified._json_value(signature):
                    raise ValueError('comparison signature changed')
                prune_temporary_graphs(folder,self.out)
                rows+=done['rows'];continue
            attempt=folder/f'attempt_{len(list(folder.glob("attempt_*")))+1:04d}'
            attempt.mkdir(parents=True)
            barrier=threading.Barrier(count)
            def lane(item):
                i,(condition,seed)=item
                os.sched_setaffinity(0,{cores[i]})
                result=[]
                order=orders[(start+i)%len(orders)]
                try:
                    for method in order:
                        self.guard();barrier.wait(timeout=self.args.timeout*4)
                        target=attempt/f'lane_{i:03d}'/method
                        begun=time.perf_counter()
                        working=dict(condition,_affinity=cores[i])
                        row=None
                        if method!='default':
                            prefix=target/'cold_structure'
                            record=dict(condition['record'])
                            arch=self.architecture(record,condition['architecture'])
                            run=run_trial(target/'structure_run',vpr_command(self.binary,arch,record['blif'],condition['width'],701)+['--device','pilot'],
                                inputs=[self.binary_reference,arch,record['blif']],env=dict(TC_GP_STRUCTURE=prefix,TC_GP_STRUCTURE_ONLY=1),
                                marker='TAILCOMPILE_V7_STRUCTURE_ONLY_COMPLETE',affinity=cores[i],timeout=self.args.timeout,prune=True)
                            if run['status']!='complete':
                                row=dict(condition_id=condition['id'],design=condition['design'],family=condition['family'],
                                    architecture=condition['architecture'],width=condition['width'],width_role=condition['width_role'],
                                    seed=seed,status=run['status'],vpr_manifest=None,cpd_ns=None,wirelength=None)
                            else:
                                record.update(structure_prefix=str(prefix),structure_export_seconds=0.)
                                working.update(record=record,graph_root=str(target/'cold_graph'))
                        if row is None:
                            row=self.execute(working,seed,target/'execution',
                                method=method if method in ('default','capacity_only') else 'gnn',
                                checkpoint=checkpoints.get(method),affinity=cores[i])
                        row.update(method=method,split=condition['split'],total_seconds=time.perf_counter()-begun,
                            measured_cold_total=True,physical_core=cores[i],wave_load=count,order=order,host=host,
                            reusable_RRG_preparation_excluded=True)
                        write(target/'timed_result.json',row)
                        result.append(row)
                        barrier.wait(timeout=self.args.timeout*4)
                except BaseException:
                    barrier.abort();raise
                return result
            with ThreadPoolExecutor(max_workers=count) as pool:
                wave_rows=[r for group in pool.map(lane,enumerate(wave)) for r in group]
            freeze(folder/'completed.json',dict(signature=signature,rows=wave_rows,attempt=str(attempt)))
            rows+=wave_rows
            write(root/'report.json',dict(planned=200,completed=len(rows),rows=rows))
            prune_temporary_graphs(folder,self.out)
            emit('repair_comparison_progress',completed=len(rows),planned=200)
        return rows


def summarize(rows, conditions):
    conditions=[c for c in conditions if c['width_role']=='reference']
    expected={(c['id'],s,m) for c in conditions for s in COMPARE_SEEDS for m in METHODS}
    keys=[(r['condition_id'],r['seed'],r['method']) for r in rows]
    if len(keys)!=len(set(keys)) or not set(keys)<=expected:
        raise ValueError('duplicate/unregistered repair comparison')
    lookup=dict(zip(keys,rows)); contrasts=[]
    for method,base in [('capacity_only','default'),('frozen_gnn','default'),('clipped','default'),
                        ('guarded','default'),('guarded','capacity_only'),('guarded','clipped'),('guarded','frozen_gnn')]:
        strata=[]
        for c in conditions:
            pairs=[(lookup.get((c['id'],s,method)),lookup.get((c['id'],s,base))) for s in COMPARE_SEEDS]
            present=[(a,b) for a,b in pairs if a and b]
            good=[(a,b) for a,b in present if a['status']==b['status']=='success']
            def ratio(key):
                values=[a.get(key,0)/b[key] for a,b in good
                        if isinstance(a.get(key),(int,float)) and isinstance(b.get(key),(int,float))
                        and math.isfinite(a[key]) and math.isfinite(b[key]) and min(a[key],b[key])>0]
                return math.exp(sum(map(math.log,values))/len(values)) if values and len(values)==len(good) else None
            strata.append(dict(condition_id=c['id'],split=c['split'],design=c['design'],
                observed=len(present),joint_success=len(good),method_success=sum(a['status']=='success' for a,b in present),
                baseline_success=sum(b['status']=='success' for a,b in present),
                runtime_ratio=ratio('total_seconds'),cpd_ratio=ratio('cpd_ns'),wirelength_ratio=ratio('wirelength')))
        for split in ('train','validation'):
            group=[s for s in strata if s['split']==split]
            def avg(key):
                vals=[s[key] for s in group]
                return math.exp(sum(map(math.log,vals))/len(vals)) if vals and all(v is not None for v in vals) else None
            ok=bool(group) and all(s['observed']==2 and s['method_success']>=s['baseline_success'] and s['joint_success']==2
                and s['cpd_ratio'] is not None and s['wirelength_ratio'] is not None
                and s['cpd_ratio']<=1.03 and s['wirelength_ratio']<=1.03 for s in group)
            runtime=avg('runtime_ratio')
            contrasts.append(dict(method=method,baseline=base,split=split,strata=group,
                runtime_ratio=runtime,cpd_ratio=avg('cpd_ratio'),wirelength_ratio=avg('wirelength_ratio'),
                protection_passed=ok,development_runtime_target=bool(ok and runtime is not None and runtime<=.95)))
    unknown=sum(r['status'] not in ('success','unroutable','timeout','decode_failure',
                                  'packing_or_placement_failure','resource_budget_exceeded') for r in rows)
    if unknown:
        for contrast in contrasts:
            contrast['development_runtime_target']=False
    primary=next(c for c in contrasts if (c['method'],c['baseline'],c['split'])==('guarded','default','validation'))
    reward_control=next(c for c in contrasts if (c['method'],c['baseline'],c['split'])==('guarded','clipped','validation'))
    status=('incomplete_comparison' if set(keys)!=expected else 'infrastructure_requires_diagnosis' if unknown else
            'reference_development_target_supported' if primary['development_runtime_target'] else
            'reference_development_target_not_met')
    return dict(schema='learning-repair-report-006',complete=set(keys)==expected,planned=len(expected),completed=len(rows),
        status=status,
        reward_change_helped_active_validation=bool(not unknown and reward_control['protection_passed']
            and reward_control['runtime_ratio'] is not None and reward_control['runtime_ratio']<1),
        status_counts=dict(Counter(r['status'] for r in rows)),contrasts=contrasts,
        infrastructure_or_unknown_count=unknown,
        scope='reference-only two-seed development; active validation, not held-out acceptance',
        Wmin_measured=False,stress_evaluated=False,full_acceptance_passed=False)


def main():
    install_signal_handlers()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=STAGES)
    parser.add_argument('--source',type=Path,default=HERE/'runs/development_004_capacity')
    parser.add_argument('--prior',type=Path,default=HERE/'runs/development_005_qualified_learning')
    parser.add_argument('--out',type=Path,default=HERE/'runs/development_006_learning_repair')
    parser.add_argument('--jobs',type=int,default=4)
    parser.add_argument('--per-job-mib',type=int,default=4096)
    parser.add_argument('--timeout',type=int,default=3600)
    parser.add_argument('--device',default='cpu')
    args=parser.parse_args()
    if min(args.jobs,args.per_job_mib,args.timeout)<1 or not re.fullmatch(r'cpu|cuda(?::[0-9]+)?',args.device):
        parser.error('invalid resource/device argument')
    source,prior,out=new_paths(args.source,args.prior,args.out)
    with campaign_lock(out,'repair-'+args.stage):
        catalog,protocol=prepare_catalog(source,prior,out)
        freeze(out/'execution_protocol.json',dict(timeout=args.timeout,threads=1))
        runtime=SimpleNamespace(stage=args.stage,out=out,config=out/'campaign.json',workspace=REPO.parent,
            scratch=REPO/'vtr_work/tailcompile_v7_3d/learning_repair',vtr_root=Path(catalog['backend']),
            mapping_vtr_root=Path(catalog['mapping']),binary=Path(catalog['binary']),
            backend_manifest=Path(catalog['backend'])/'backend_manifest.json',
            qualification_report=source/'qualification/qualification_report.json',
            jobs=args.jobs,per_job_mib=args.per_job_mib,rrg_jobs=1,rrg_mib=args.per_job_mib,
            timeout=args.timeout,threads=1,device=args.device,allow_local=False)
        campaign=RepairCampaign(runtime,catalog,protocol)
        campaign.gate()
        if args.stage=='train':
            campaign.train_arms()
        elif args.stage=='report':
            report=summarize(read(out/'comparison/report.json')['rows'],campaign.conditions())
            report['selection_statuses']={p:read(out/'validation'/p/'selection.json')['status'] for p in POLICIES}
            write(out/'repair_report.json',report)
        else:
            getattr(campaign,args.stage)()
        emit('repair_stage_complete',stage=args.stage,out=str(out))


if __name__=='__main__':
    main()
