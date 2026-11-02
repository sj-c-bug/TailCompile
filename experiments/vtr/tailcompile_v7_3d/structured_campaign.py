"""010: bounded resource/density decoding and netlist/geometry GNN learning.

Reads frozen 008/009 cohorts. Evaluation reports stay local/server-only; raw
compact metrics may be synchronized on request, separately from source commits.
No backend rebuilds, packing pass before inference, or runtime-based selection.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import itertools
import json
import math
import os
from pathlib import Path
import statistics
import sys
import threading
import time

from qor_campaign import QorCampaign, CODE as QOR_CODE, panel, legacy, check as check008
from campaign import campaign_lock, resource_budget
from io_utils import read, write, freeze, sha, run_trial, install_signal_handlers
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
METHODS = ('B', 'X', 'D', 'N')
STAGES = ('plan', 'check', 'prepare', 'feedback', 'train', 'validate', 'compare', 'report')
CODE = tuple(dict.fromkeys(QOR_CODE + ('structured_campaign.py', 'structured_model.py',
    'structured_decoder.py', 'train_structured.py', 'run_structured_campaign.sh')))


def queue_plan():
    return dict(schema='structured-campaign-010-1', methods=list(METHODS),
        factors=dict(B='native Default', X='frozen selected009X',
                     D='same009X weights + strict010 decoder', N='same start +010 training + strict010 decoder'),
        primary_comparison=['N', 'B'], contrasts=[['D', 'X'], ['N', 'D'], ['N', 'X'], ['N', 'B']],
        tail_weight=0., backend_layer_prior=0, source009_read_only=True,
        train_families=['spmv', 'mcml'], active_validation_families=['blob_merge', 'diffeq'],
        reference_widths=dict(spmv=128, mcml=224), training_seed=10301,
        feedback_seed=10401, feedback_noise=[0., .25], feedback_runs_upper=16,
        feedback_relabel_after_projection=False, bootstrap_epochs=0, training_epochs=5,
        checkpoint_epochs=[2, 5], validation_seed=10201, validation_runs=24,
        comparison_seeds=[10101, 10102], comparison_runs=64, research_runs_upper=104,
        new_backend_builds=0, new_rrg_extractions=0, Wmin_measured=False,
        independent_family_test=False, runtime_role='cost only',
        metric_rule='arithmetic same-successful-seed means, then equal architecture/family geometric ratios',
        WL_units='native nonglobal CHANX/CHANY only; final CHANZ usage separately')


def owned(path):
    path = Path(path).resolve()
    runs = HERE/'runs'
    if runs not in path.parents or not path.relative_to(runs).parts[0].startswith('development_010'):
        raise ValueError('010 writes require a separate development_010 cohort inside v7 runs')
    validate_output_roots((path,))
    return path


def source008(source009):
    recorded = Path(read(Path(source009)/'qor_protocol.json')['source'])
    result = HERE/'runs'/recorded.name
    if recorded.exists() and recorded.resolve() != result.resolve():
        raise ValueError('009 source008 must belong to this canonical repository')
    return result


def check(args):
    required = ('qor_protocol.json', 'qor_report.json', 'selection.json', 'feedback.json')
    missing = [str(args.source/n) for n in required if not (args.source/n).is_file()]
    if missing:
        return dict(ready=False, missing=missing, plan=queue_plan())
    old = source008(args.source)
    from types import SimpleNamespace
    result = check008(SimpleNamespace(source=old))
    if not read(args.source/'qor_report.json').get('complete'):
        raise ValueError('010 requires completed009 comparison evidence')
    checkpoint = Path(read(args.source/'selection.json')['checkpoints']['X'])
    if not checkpoint.is_file():
        result['missing'].append(str(checkpoint))
    for row in read(args.source/'feedback.json'):
        if row.get('action') and not Path(row['action']).is_file():
            result['missing'].append(row['action'])
    result.update(ready=not result['missing'], plan=queue_plan(), source008=str(old))
    return result


def gm(values):
    values = list(values)
    return math.exp(statistics.mean(math.log(v) for v in values)) if values else None


def verify_checkpoint(path, expected):
    """Require the identity recorded when a model was created/selected."""
    if not isinstance(expected,str) or len(expected)!=64 or sha(path)!=expected:
        raise ValueError('010 checkpoint differs from its frozen identity: '+str(path))
    return str(path)


def contrast(rows, method, baseline):
    lookup = {(r['condition_id'], r['seed'], r['method']): r for r in rows}
    groups = defaultdict(list)
    for r in rows:
        if r['method'] != method:
            continue
        b = lookup.get((r['condition_id'], r['seed'], baseline))
        if b is not None and r['status'] == b['status'] == 'success':
            groups[r['family'], r['architecture'], r['condition_id']].append((r, b))
    metrics = {}
    for name in ('wirelength', 'cpd_ns', 'total_seconds', 'vpr_seconds', 'final_unique_chanz_nodes'):
        families = defaultdict(list); conditions = {}
        for (family, arch, cid), pairs in sorted(groups.items()):
            def metric(r):
                return (r.get('final_route_usage') or {}).get('unique_chanz_nodes') if name=='final_unique_chanz_nodes' else r[name]
            values=[[metric(p[i]) for p in pairs] for i in (0,1)]
            if any(v is None for group in values for v in group):
                continue
            a, b = [statistics.mean(group) for group in values]
            if not all(math.isfinite(v) and v > 0 for v in (a, b)):
                if name=='final_unique_chanz_nodes' and min(a,b)==0:
                    # Zero occupancy is valid; its multiplicative ratio is undefined.
                    conditions[cid]=dict(method_seed_mean=a,baseline_seed_mean=b,ratio=None,
                        common_success_seeds=[p[0]['seed'] for p in pairs],ratio_undefined_due_to_zero=True)
                    continue
                raise ValueError('invalid successful QoR/cost metric')
            conditions[cid] = dict(method_seed_mean=a, baseline_seed_mean=b, ratio=a/b,
                                   common_success_seeds=[p[0]['seed'] for p in pairs])
            families[family].append(a/b)
        fr = {f: gm(v) for f, v in families.items()}
        ratio = gm(fr.values())
        metrics[name] = dict(ratio=ratio, gain=None if ratio is None else 1-ratio,
                             family_ratios=fr, conditions=conditions)
    w, c = [metrics[n]['ratio'] for n in ('wirelength', 'cpd_ns')]
    classification = 'insufficient_qor' if w is None or c is None else (
        'average_joint_improvement' if w <= 1 and c <= 1 and min(w, c) < 1 else
        'average_joint_regression' if w >= 1 and c >= 1 and max(w, c) > 1 else
        'no_average_change' if w == c == 1 else 'qor_tradeoff')
    return dict(method=method, baseline=baseline, metrics=metrics,
                jointly_successful=sum(map(len, groups.values())),
                classification=classification, joint_q=None if w is None or c is None else .5*(math.log(w)+math.log(c)))


def choose_epoch(rows):
    lookup = {(r['epoch'], r['method'], r['condition_id']): r for r in rows}
    controls = {r['condition_id'] for r in rows if r['method'] == 'B'}
    expected = {(0, 'B', c) for c in controls} | {(e, 'N', c) for e in (2, 5) for c in controls}
    if len(controls) != 8 or len(lookup) != len(rows) or set(lookup) != expected:
        raise ValueError('010 selection requires all8 active-validation controls and epoch2/5')
    common = {c for c in controls if all(lookup[e, m, c]['status'] == 'success' for e, m in ((0, 'B'), (2, 'N'), (5, 'N')))}
    scores = []
    for e in (2, 5):
        subset = [lookup[0, 'B', c] for c in common] + [lookup[e, 'N', c] for c in common]
        score = contrast(subset, 'N', 'B')['joint_q'] if subset else None
        scores.append(dict(epoch=e, successes=sum(lookup[e, 'N', c]['status'] == 'success' for c in controls),
                           common_success_q=score, common_success_conditions=len(common)))
    best = min(scores, key=lambda x: (-x['successes'], x['common_success_q'] if x['common_success_q'] is not None else float('inf'), x['epoch']))
    return dict(epoch=best['epoch'], scores=scores, performance_selection_completed=bool(common),
                selection='success count, then shared-common-population WL/CPD q, earlier tie', runtime_used_for_selection=False)


def worker(request, out):
    # Reject a valid-but-replaced checkpoint before graph construction or writes.
    verify_checkpoint(request['checkpoint'],request.get('checkpoint_sha256'))
    import torch
    from data import build_logic
    from fast_graph import make_graph
    from layer_features import augment_graph
    from generation import lower_initialization
    from train import state_sha256
    from aligned_teacher import constraints_digest
    from model import CapacityError
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    validate_output_roots((out,)); out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); r = request
    logical = build_logic(r['prefix'], r['design'], r['blif'], r['architecture'], family=r['family'], split=r['split'])
    device = read(r['device'])
    graph = augment_graph(make_graph(logical, device), logical, device, read(r['vertical_summary']), r['blif'], r['prefix'])
    torch.save(graph, out/'graph.pt'); write(out/'logic.json', logical)
    result = dict(graph=str(out/'graph.pt'), logic=str(out/'logic.json'), graph_sha256=sha(out/'graph.pt'),
                  logic_sha256=sha(out/'logic.json'), graph_seconds=time.perf_counter()-started)
    if r['method'] == 'X':
        from train_qor import load_model
        from qor_model import generate
    else:
        from train_structured import load_model
        from structured_model import generate
    model = load_model(r['checkpoint'], device='cpu')
    try:
        with torch.no_grad():
            prediction = generate(model, graph, method=r['method'], seed=r.get('generation_seed', 10301),
                                  score_noise_std=r.get('score_noise_std', 0.), diagnostics=False)
    except CapacityError as exc:
        result.update(status='decode_failure', reason=str(exc), packing_infeasibility_claimed=False,
                      decoder_diagnostics=getattr(exc,'diagnostics',{}))
    else:
        verify_checkpoint(r['checkpoint'],r['checkpoint_sha256'])
        lowered = lower_initialization(logical, device, prediction, out/'initialization', method=r['method'])
        action = read(lowered['target'])
        action.update(schema='tailcompile-v7-qor-action-009-1' if r['method']=='X' else 'tailcompile-v7-structured-action-010-1',
            decoder_kind=prediction['decoder_kind'], shared_xy_sha256=state_sha256(model.shared_xy),
            constraints_sha256=constraints_digest(graph), collection_checkpoint_sha256=r['checkpoint_sha256'], collection_arm=r['method'],
            deployed_action_sha256=prediction['deployed_action_sha256'])
        write(lowered['target'], action)
        lowered.update(target_sha256=sha(lowered['target']), shared_xy_sha256=action['shared_xy_sha256'],
            constraints_sha256=action['constraints_sha256'], collection_checkpoint_sha256=action['collection_checkpoint_sha256'],
            deployed_action_sha256=prediction['deployed_action_sha256'], risk_diagnostics_computed=False)
        write(out/'initialization/generation.json', lowered); result.update(lowered)
    result['worker_internal_seconds'] = time.perf_counter()-started
    write(out/'generation_report.json', result); print('TAILCOMPILE_STRUCTURED_WORKER_COMPLETE', flush=True)
    return result


class StructuredCampaign(QorCampaign):
    def __init__(self, args):
        self.args, self.out = args, owned(args.out)
        self.source009 = args.source; self.source008 = source008(args.source)
        self.conditions = panel(self.source008)
        self.train = [c for c in self.conditions if c['split']=='train']
        self.reference = [c for c in self.train if c['width_role']=='reference']
        self.validation = [c for c in self.conditions if c['split']=='validation']
        self.previous = legacy(self.source008)
        self.budget = resource_budget(args.jobs, args.per_job_mib); self.binary=None; self._devices={}

    def setup(self):
        from install_layer_hook import verify
        for source in (self.source008, self.source009, self.previous):
            if source == self.out or source in self.out.parents or self.out in source.parents:
                raise ValueError('010 output must be disjoint from all borrowed cohorts')
        p = read(self.source008/'aligned_protocol.json'); p009 = read(self.source009/'qor_protocol.json')
        if not read(self.source009/'qor_report.json').get('complete'):
            raise ValueError('009 comparison incomplete')
        self.args.backend = Path(p['backend']).resolve()
        verify(self.args.backend); manifest=read(self.args.backend/'backend_manifest.json')
        self.binary=Path(manifest['binary'])
        if sha(self.binary) != p['binary_sha256'] or p009['binary_sha256'] != p['binary_sha256']:
            raise ValueError('frozen backend identity mismatch')
        for frozen in (p, p009):
            for name, wanted in frozen['implementation'].items():
                if sha(HERE/name) != wanted: raise ValueError('frozen008/009 code changed: '+name)
        for name in ('layer_protocol.json','borrowed/conditions.json','borrowed/qualified_source_manifest.json'):
            path=self.previous/name
            if sha(path) != p['source_inputs'].get(str(path.resolve())):
                raise ValueError('frozen007 metadata changed: '+name)
        qpath=self.previous/'qualification/qualification_report.json'; q=read(qpath)
        if not q.get('passed') or q['binary_sha256'] != p['binary_sha256'] or sha(qpath) != read(self.previous/'layer_protocol.json')['qualification_sha256']:
            raise ValueError('borrowed backend qualification mismatch')
        self.catalog=read(self.previous/'borrowed/qualified_source_manifest.json')
        old_reference=Path(self.catalog['source'])/'binary_identity.json'
        if sha(old_reference) != self.catalog['files'][str(old_reference.resolve())]: raise ValueError('original binary identity changed')
        self.old_binary_hash=read(old_reference)['sha256']
        selection=read(self.source009/'selection.json'); self.source_checkpoint=Path(selection['checkpoints']['X'])
        if sha(self.source_checkpoint) != selection['checkpoint_sha256']['X']: raise ValueError('selected009X changed')
        self.teacher_rows={r['condition_id']:r for r in read(self.source008/'teacher_records.json')['records']}
        inputs={str(self.source009/n):sha(self.source009/n) for n in ('qor_protocol.json','qor_report.json','selection.json','feedback.json')}
        inputs[str(self.source_checkpoint)]=sha(self.source_checkpoint)
        for n in ('aligned_protocol.json','teacher_records.json','feedback.json','teacher_alignment_report.json'):
            path=self.source008/n
            if sha(path) != p009['source_inputs'].get(str(path.resolve())):
                raise ValueError('source008 metadata differs from frozen009 input: '+n)
            inputs[str(path)]=sha(path)
        for c in self.conditions:
            row=self.teacher_rows[c['id']]
            if row['split'] != c['split'] or row['family'] != c['family']: raise ValueError('teacher split changed')
            if sha(row['graph']) != p['source_inputs'].get(row['graph']): raise ValueError('source graph changed')
            if sha(row['teacher']) != p009['source_inputs'].get(row['teacher']): raise ValueError('source aligned teacher changed')
            inputs[row['graph']]=sha(row['graph']); inputs[row['teacher']]=sha(row['teacher'])
        freeze(self.out/'structured_protocol.json', dict(plan=queue_plan(), source009=str(self.source009),
            source008=str(self.source008), source_checkpoint=str(self.source_checkpoint), source_inputs=inputs,
            backend=str(self.args.backend), binary_sha256=p['binary_sha256'],
            implementation={n:sha(HERE/n) for n in CODE}, timeout=self.args.timeout))
        freeze(self.out/'queue_plan.json', queue_plan())
        self.binary_reference=self.out/'binary_identity.json'; freeze(self.binary_reference,dict(path=str(self.binary),sha256=p['binary_sha256']))
        st=self.binary.stat(); self.binary_stat=(st.st_size,st.st_mtime_ns,st.st_ino)
        freeze(self.out/'execution'/self.args.stage/'protocol.json',dict(device=self.args.device,jobs=self.args.jobs,
            per_job_mib=self.args.per_job_mib,timeout=self.args.timeout,physical_cores=self.budget['physical_cores'],effective_lanes=self.budget['effective']))

    def request(self, c, checkpoint=None, method='N', prefix=None, noise=0.):
        checkpoint=str(checkpoint)
        if method=='X':
            trusted=read(self.source009/'selection.json')
            if checkpoint!=trusted['checkpoints']['X']:raise ValueError('010 X must use the selected009 checkpoint')
            expected=trusted['checkpoint_sha256']['X']
        elif method=='D':
            trusted=read(self.out/'prepared_report.json')
            if checkpoint!=trusted['checkpoint']:raise ValueError('010 D must use the prepared control checkpoint')
            expected=trusted['checkpoint_sha256']
        elif method=='N':
            expected=read(self.out/'trained_report.json').get('checkpoint_sha256',{}).get(checkpoint)
        else:raise ValueError('010 initialization method must be X/D/N')
        verify_checkpoint(checkpoint,expected)
        return dict(self.device(c), prefix=prefix or c['record']['structure_prefix'], design=c['design'], family=c['family'],
                    split=c['split'], blif=c['record']['blif'], checkpoint=checkpoint,checkpoint_sha256=expected,method=method,
                    generation_seed=10301, score_noise_std=noise)

    def generate(self,c,target,checkpoint=None,method='N',prefix=None,noise=0.,core=None):
        self.guard(); freeze(target/'request.json',self.request(c,checkpoint,method,prefix,noise))
        result=run_trial(target/'worker',[sys.executable,HERE/'structured_campaign.py','worker','--request',target/'request.json',
            '--out',target/'payload'],inputs=[target/'request.json',checkpoint]+[HERE/n for n in CODE],timeout=self.args.timeout,
            affinity=core,marker='TAILCOMPILE_STRUCTURED_WORKER_COMPLETE',prune=False)
        if result['status']!='complete': raise RuntimeError('010 generator worker failed: '+result['log'])
        value=read(target/'payload/generation_report.json'); value['generation_process_seconds']=result['metrics']['wall_seconds']
        return value

    def prepare(self):
        if (self.out/'prepared_report.json').exists(): return
        from train_structured import from009, save_model
        records=[self.teacher_rows[c['id']] for c in self.conditions]
        # Migration changes neither source weights nor the D control's weights.
        model=from009(self.source_checkpoint, device='cpu')
        checkpoint=save_model(model,self.out/'models/decoder_control/epoch_0000.pt',epoch=0)
        freeze(self.out/'teacher_records.json',dict(records=records))
        freeze(self.out/'source_feedback.json',read(self.source008/'feedback.json')+read(self.source009/'feedback.json'))
        freeze(self.out/'prepared_report.json',dict(complete=True,checkpoint=checkpoint,checkpoint_sha256=sha(checkpoint),
            source_checkpoint=str(self.source_checkpoint),source_checkpoint_sha256=sha(self.source_checkpoint),
            train_conditions=len(self.train),active_validation_conditions=len(self.validation),new_training_steps=0))

    def feedback(self):
        if (self.out/'feedback_report.json').exists(): return
        prepared=read(self.out/'prepared_report.json')
        checkpoint=verify_checkpoint(prepared['checkpoint'],prepared['checkpoint_sha256'])
        cores=self.budget['physical_cores'][:min(self.budget['effective'],len(self.reference))]
        freeze(self.out/'feedback/timing_protocol.json',dict(cores=cores,seed=10401,
            candidate_noise=queue_plan()['feedback_noise'],each_lane_sequential=True))
        def lane(item):
            lane_index,core=item; actions=[]; failures=[]; duplicates=[]
            for c in self.reference[lane_index::len(cores)]:
                seen=set()
                for index,noise in enumerate(queue_plan()['feedback_noise']):
                    folder=self.out/'feedback'/c['id']/f'candidate_{index}'
                    gen=self.generate(c,folder/'generation',checkpoint,'D',noise=noise,core=core)
                    if not gen.get('ap_init'):
                        failures.append(dict(condition_id=c['id'],candidate=index,generation=gen))
                        self.prune_feedback_graph(folder); continue
                    if gen['ap_init_sha256'] in seen:
                        duplicates.append(dict(condition_id=c['id'],candidate=index,sha256=gen['ap_init_sha256']))
                        self.prune_feedback_graph(folder); continue
                    seen.add(gen['ap_init_sha256'])
                    trial=self.pnr(c,10401,folder/'native','D',gen,core)
                    actions.append(dict(condition_id=c['id'],family=c['family'],split='train',candidate_id=f'010-D-{index}',
                        action=gen['target'],generation=gen,trials=[trial]))
                    self.prune_feedback_graph(folder)
            return actions,failures,duplicates
        with ThreadPoolExecutor(max_workers=len(cores)) as pool:
            groups=list(pool.map(lane,enumerate(cores)))
        actions=[r for g in groups for r in g[0]]
        failures=[r for g in groups for r in g[1]]
        duplicates=[r for g in groups for r in g[2]]
        freeze(self.out/'feedback.json',actions)
        freeze(self.out/'feedback_report.json',dict(actions=len(actions),native_runs=len(actions),budget_upper=16,
            duplicates=duplicates,decode_failures=failures,old_labels_reused_after_projection=False))

    def train_models(self):
        if (self.out/'trained_report.json').exists(): return
        from train_structured import fit
        rows=[r for r in read(self.out/'teacher_records.json')['records'] if r['split']=='train']
        feedback=read(self.out/'source_feedback.json')+read(self.out/'feedback.json')
        result=fit(rows,self.source_checkpoint,self.fresh_attempt(self.out/'models/trained'),feedback=feedback,
            epochs=5,snapshots=(2,5),seed=10301,device=self.args.device)
        result['checkpoint_sha256']={str(p):sha(p) for p in result['checkpoints']}
        freeze(self.out/'trained_report.json',result)
        if result.get('feedback_pairs',result.get('effective_pairs',0)) <= 0:
            raise RuntimeError('010 has no effective deployable QoR pairs; keep diagnostics and stop before validation/comparison')

    def validate(self):
        trained=read(self.out/'trained_report.json')
        if trained.get('feedback_pairs',trained.get('effective_pairs',0)) <= 0:
            raise ValueError('010 validation requires effective deployable QoR feedback')
        checkpoints=trained['checkpoints']; epochs=(2,5)
        if len(checkpoints)!=2: raise ValueError('010 requires epoch2/5 snapshots')
        for checkpoint in checkpoints:
            verify_checkpoint(checkpoint,trained.get('checkpoint_sha256',{}).get(str(checkpoint)))
        core=self.budget['physical_cores'][0]; rows=[]
        freeze(self.out/'validation/timing_protocol.json',dict(core=core,sequential=True,seed=10201,
            checkpoints={str(p):sha(p) for p in checkpoints}))
        for c in self.validation:
            rows.append(dict(self.cold(c,10201,'B',None,self.out/'validation'/c['id']/'B',core),epoch=0))
            for epoch,checkpoint in zip(epochs,checkpoints):
                rows.append(dict(self.cold(c,10201,'N',checkpoint,self.out/'validation'/c['id']/f'N_e{epoch}',core),epoch=epoch))
        selection=choose_epoch(rows); idx=epochs.index(selection['epoch'])
        selection['checkpoint']=checkpoints[idx]; selection['checkpoint_sha256']=sha(checkpoints[idx])
        freeze(self.out/'selection.json',selection); write(self.out/'validation/report.json',dict(rows=rows,selection=selection))

    def compare(self):
        selection=read(self.out/'selection.json')
        if not selection['performance_selection_completed']: raise ValueError('no common successful validation QoR; stop comparison')
        prepared=read(self.out/'prepared_report.json'); trained=read(self.out/'trained_report.json')
        d=verify_checkpoint(prepared['checkpoint'],prepared['checkpoint_sha256'])
        n=verify_checkpoint(selection['checkpoint'],selection['checkpoint_sha256'])
        if trained.get('checkpoint_sha256',{}).get(n)!=selection['checkpoint_sha256']:
            raise ValueError('selected010N was not a frozen trained snapshot')
        checkpoints=dict(X=str(self.source_checkpoint),D=d,N=n)
        cases=list(itertools.product(self.reference,(10101,10102)))
        available=self.budget['physical_cores'][:self.budget['effective']]
        lanes=max(n for n in range(1,len(available)+1) if len(cases)%n==0); cores=available[:lanes]
        host=dict(uname=list(os.uname()),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),cores=cores)
        freeze(self.out/'comparison/timing_protocol.json',dict(host=host,cases=[[c['id'],s] for c,s in cases],methods=list(METHODS),
            seed_rotation='case index modulo four',checkpoints={m:sha(p) for m,p in checkpoints.items()},fixed_lane_barriers=True))
        rows=[]
        for start in range(0,len(cases),lanes):
            wave=self.out/'comparison'/f'wave_{start//lanes:04d}'
            signature=dict(host=host,cases=[[c['id'],s] for c,s in cases[start:start+lanes]],
                           timing_protocol_sha256=sha(self.out/'comparison/timing_protocol.json'))
            if (wave/'completed.json').exists():
                cached=read(wave/'completed.json')
                if cached['signature']!=signature: raise ValueError('timing host/input signature changed; use a new cohort')
                rows.extend(cached['rows']); continue
            if len(list(wave.glob('attempt_*')))>=2: raise RuntimeError('010 whole-wave retry budget exhausted')
            attempt=self.fresh_attempt(wave); barrier=threading.Barrier(lanes)
            def lane(item):
                i,(c,s)=item; core=cores[i]; os.sched_setaffinity(0,{core})
                offset=(start+i)%len(METHODS); order=METHODS[offset:]+METHODS[:offset]; result=[]
                try:
                    for m in order:
                        barrier.wait(timeout=self.args.timeout*4)
                        row=self.cold(c,s,m,checkpoints.get(m),attempt/f'lane_{i:03d}'/c['id']/f's{s}'/m,core)
                        result.append(dict(row,order=list(order),wave_load=lanes,host=host)); barrier.wait(timeout=self.args.timeout*4)
                except BaseException: barrier.abort(); raise
                return result
            with ThreadPoolExecutor(max_workers=lanes) as pool: groups=list(pool.map(lane,enumerate(cases[start:start+lanes])))
            wave_rows=[r for group in groups for r in group]
            freeze(wave/'completed.json',dict(signature=signature,rows=wave_rows,actual_attempt=str(attempt),
                previous_incomplete_attempts=len(list(wave.glob('attempt_*')))-1))
            rows.extend(wave_rows); write(self.out/'comparison/report.json',dict(planned=64,completed=len(rows),rows=rows))

    def report(self):
        rows=read(self.out/'comparison/report.json')['rows']
        expected={(c['id'],s,m) for c,s,m in itertools.product(self.reference,(10101,10102),METHODS)}
        keys=[(r['condition_id'],r['seed'],r['method']) for r in rows]
        if len(keys)!=len(set(keys)): raise ValueError('duplicate010 native task')
        result=dict(schema='structured-evaluation-010-1',complete=set(keys)==expected,completed=len(rows),
            missing=sorted(expected-set(keys)),unexpected=sorted(set(keys)-expected),status_counts=dict(Counter(r['status'] for r in rows)),
            status_by_method={m:dict(Counter(r['status'] for r in rows if r['method']==m)) for m in METHODS},
            comparisons=[contrast(rows,m,b) for m,b in (('N','B'),('X','B'),('D','B'),('D','X'),('N','D'),('N','X'))],
            tail_weight=0.,runtime_is_cost=True,Wmin_measured=False,independent_family_test=False)
        primary=result['comparisons'][0]
        result['conclusion']=('incomplete' if not result['complete'] else
            ('' if primary['jointly_successful']==16 else 'conditional_')+primary['classification'])
        write(self.out/'structured_report.json',result)
        print(result['conclusion'],result['status_counts'],flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES+('worker',))
    p.add_argument('--source',type=Path,default=HERE/'runs/development_009_qor')
    p.add_argument('--out',type=Path,default=HERE/'runs/development_010_structured')
    p.add_argument('--request',type=Path);p.add_argument('--jobs',type=int,default=4)
    p.add_argument('--per-job-mib',type=int,default=8192);p.add_argument('--timeout',type=int,default=3600)
    p.add_argument('--device',default='cpu');args=p.parse_args()
    if min(args.jobs,args.per_job_mib,args.timeout)<1:p.error('positive resource limits required')
    install_signal_handlers();args.source=args.source.resolve();args.out=owned(args.out)
    if args.stage=='worker':worker(read(args.request),args.out);return
    if args.stage=='plan':print(json.dumps(queue_plan(),indent=2));return
    if args.stage=='check':
        result=check(args);print(json.dumps(result,indent=2));raise SystemExit(0 if result['ready'] else 2)
    with campaign_lock(args.out,'structured-'+args.stage):
        runner=StructuredCampaign(args);runner.setup();getattr(runner,'train_models' if args.stage=='train' else args.stage)()


if __name__=='__main__':main()
