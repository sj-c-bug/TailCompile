"""009 tiny teacher/XY/native smoke; at most four new P&R, no QoR claim."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import time

from io_utils import read, write, freeze, sha, run_trial, vpr_command, prune_derived
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
FILES = ('qor_local_smoke.py','qor_model.py','train_qor.py','qor_preferences.py','qor_route_metrics.py')


def run(source, aligned, output, binary, *, timeout=180):
    import torch
    from aligned_local_smoke import native_fixed_check
    from aligned_teacher import verify_teacher, constraints_digest
    from aligned_model import validate_action
    from train_aligned_layers import load_model as load_008
    from train import state_sha256
    from train_qor import train_bootstrap, train_feedback, load_model
    from qor_model import generate, validate_collected_action, ACTION_SCHEMA, DECODER_SCHEMA
    from qor_route_metrics import summarize_route
    from generation import lower_initialization
    from vertical_risk import hard_risk
    from layer_observations import collect_layers

    source, aligned, output, binary = [Path(x).resolve() for x in (source,aligned,output,binary)]
    validate_output_roots((output,),dependency_roots=(source,aligned,binary))
    if any(output == p or output in p.parents or p in output.parents for p in (source,aligned)):
        raise ValueError('009 local output must be disjoint from immutable fixture inputs')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    cases = {r['name']:r for r in read(source/'qualification_inputs.json')}
    selected=('balanced','asymmetric')
    records = read(aligned/'teacher_records.json')['records']
    records = [r for r in records if any('/'+name+'/' in r['condition_id'] for name in selected)]
    if len(records) != 2 or any(r['split'] != 'train' for r in records):
        raise ValueError('expected two engineering train fixtures')
    checkpoint = Path(read(aligned/'models/prepared/preparation_report.json')['arms']['L']['checkpoint'])
    inputs = [binary, source/'qualification_inputs.json',checkpoint,aligned/'teacher_records.json']
    for record in records: inputs.extend((Path(record['graph']),Path(record['teacher'])))
    for name in selected:
        case=cases[name]
        inputs.extend((Path(case['arch']),Path(case['blif'])))
        inputs.extend(source/'model_smoke'/name/n for n in ('logic.json','device_graph.json'))
        inputs.extend(Path(case['structure']+s) for s in ('.blocks.tsv','.nets.tsv','.meta.tsv'))
        if case.get('constraints'): inputs.append(Path(case['constraints']))
    signatures = {str(p):sha(p) for p in inputs}
    freeze(output/'protocol.json',dict(schema='qor-local-smoke-009-1',source=str(source),aligned=str(aligned),
        input_sha256=signatures,implementation={n:sha(HERE/n) for n in FILES},
        bootstrap_epochs=2,training_seed=9401,pnr_seed=9501,pnr_arms=['C','T'],pnr_upper=4,
        timeout=timeout,scope='engineering only; no natural-design QoR or generalization result'))
    for record in records:
        graph=torch.load(record['graph'],weights_only=False,map_location='cpu')
        assert verify_teacher(graph,read(record['teacher']))['valid']
    # There is no same-condition measured candidate pool locally. Never invent
    # rewards by pairing the two different architectures.
    model_report=output/'models/bootstrap/bootstrap_report.json'
    prepared=read(model_report) if model_report.exists() else train_bootstrap(
        records,checkpoint,output/'models/bootstrap',[],epochs=2,seed=9401,device='cpu')
    write(model_report,prepared)
    feedback_report=output/'models/feedback/feedback_report.json'
    feedback=read(feedback_report) if feedback_report.exists() else train_feedback(
        records,prepared,output/'models/feedback',[],epochs=2,seed=9401,device='cpu')
    write(feedback_report,feedback)
    baseline=load_008(checkpoint)
    base_xy_hash=state_sha256(baseline.shared_xy)
    model_checks={}
    for arm in ('R','Q','C','X','T'):
        report=prepared['arms'][arm]
        xy_hash=report['shared_xy_sha256']
        expected_frozen=arm in ('R','Q','C')
        check=dict(weights_changed=report['initial_state_sha256'] != report['final_state_sha256'],
            shared_xy_sha256=xy_hash,shared_xy_changed=xy_hash != base_xy_hash,
            xy_expected_frozen=expected_frozen)
        check['passed']=check['weights_changed'] and ((xy_hash == base_xy_hash) if expected_frozen else (xy_hash != base_xy_hash))
        assert check['passed'],(arm,check)
        model_checks[arm]=check
    lowered={}; parity={}; gradients={}
    for name in selected:
        record=next(r for r in records if '/'+name+'/' in r['condition_id'])
        data=torch.load(record['graph'],weights_only=False,map_location='cpu')
        teacher=read(record['teacher'])
        logical=read(source/'model_smoke'/name/'logic.json')
        device=read(source/'model_smoke'/name/'device_graph.json')
        target=dict(assignment=teacher['teacher_region'],layer=teacher['teacher_layer'],xy=teacher['teacher_xy'])
        target_ids=torch.as_tensor(target['assignment'],dtype=torch.long)
        with torch.no_grad():
            encoded=baseline.shared_xy.encode(data)
            original_xy=baseline.shared_xy.deployment_coordinates(data,*encoded,target_ids)
        for arm in ('C','T'):
            model=load_model(prepared['arms'][arm]['checkpoint'])
            model.eval()
            with torch.no_grad():
                loss,detail=model.supervision_loss(data,teacher)
                prediction=generate(model,data,method=arm,diagnostics=True)
                fast=generate(model,data,method=arm,diagnostics=False)
                actual=hard_risk(data,prediction['layer'],prediction['xy'].to(data['demand']))
                fixed_xy=model.conditional_components(data,target)['prediction']
            different_xy=not torch.equal(original_xy,fixed_xy)
            check=dict(training_action_matches_deployment=detail['deployed_action_sha256']==prediction['deployed_action_sha256']==fast['deployed_action_sha256'],
                default_inference_no_risk=not fast['risk_diagnostics_computed'] and 'hard_vertical_risk' not in fast,
                hard_capacity=validate_action(data,prediction),
                same_fixed_assignment_xy_changed=different_xy,
                training_hard_cvar=detail.get('vertical_tail'),deployed_hard_cvar=actual['cvar95'])
            check['risk_forward_matches']=math.isclose(check['training_hard_cvar'],check['deployed_hard_cvar'],rel_tol=0.,abs_tol=1e-7)
            check['passed']=all((check['training_action_matches_deployment'],check['default_inference_no_risk'],
                check['hard_capacity']['valid'],check['risk_forward_matches'],different_xy if arm=='T' else not different_xy))
            assert check['passed'],(name,arm,check)
            parity[name+'/'+arm]=check
            model.zero_grad(set_to_none=True)
            score=model.action_score(data,target)
            score.backward()
            coord_grads=[p.grad for p in model.shared_xy.coordinate_head.parameters() if p.grad is not None]
            gradient=sum(float(g.abs().sum()) for g in coord_grads)
            gradients[name+'/'+arm]=dict(coordinate_score_gradient_l1=gradient,
                source='valid GP teacher action; gradient-path check, not a fabricated QoR pair',
                passed=(gradient>0 if arm=='T' else gradient==0))
            assert gradients[name+'/'+arm]['passed']
            folder=output/'initializers'/name/arm
            generated=lower_initialization(logical,device,fast,folder,method='qor_'+arm)
            action=read(generated['target'])
            action.update(schema=ACTION_SCHEMA,decoder_kind=DECODER_SCHEMA,collection_arm=arm,
                collection_checkpoint_sha256=sha(prepared['arms'][arm]['checkpoint']),
                shared_xy_sha256=state_sha256(model.shared_xy),constraints_sha256=constraints_digest(data))
            write(generated['target'],action); generated['target_sha256']=sha(generated['target'])
            validate_collected_action(data,action)
            write(folder/'generation.json',generated)
            write(folder/'qor_contract_report.json',dict(parity=check,gradient=gradients[name+'/'+arm]))
            lowered[name,arm]=generated
    trials=[]
    for name in selected:
        for arm in ('C','T'):
            case=cases[name]; generated=lowered[name,arm]; target=output/'deployments'/name/arm
            target.mkdir(parents=True,exist_ok=True)
            durable=target/'qor_pnr_report.json'
            if durable.exists():
                saved=read(durable)
                if saved['initializer_sha256'] != generated['ap_init_sha256']: raise ValueError('resume initializer changed')
                trials.append(saved); continue
            command=vpr_command(binary,case['arch'],case['blif'],case['width'],9501)+case['options']
            command += ['--write_flat_place','post_route.fplace','--write_legalized_flat_place','post_legalizer.fplace','--flat_place_verbosity','0']
            dependencies=[binary,case['arch'],case['blif'],generated['ap_init']]
            if case.get('constraints'): dependencies.append(case['constraints'])
            result=run_trial(target/'vpr',command,inputs=dependencies,
                env=dict(TC_GP_INIT=generated['ap_init'],TC_GP_LAYER_PRIOR='0',TC_GP_EXPORT=target/'post_gp',TC_GP_Z_USAGE=target/'vertical_usage'),
                timeout=timeout,prune=False,metadata=dict(stage='qor_local_engineering',case=name,method=arm,seed=9501))
            work=Path(result['workdir']); log=Path(result['log']).read_text(errors='replace')
            routes=list(work.glob('*.route'))
            usage=summarize_route(routes[0]) if routes else dict(available=False)
            write(target/'route_usage_report.json',usage)
            fixed=native_fixed_check(case['structure'],target/'post_gp') if Path(str(target/'post_gp')+'.blocks.tsv').exists() else dict(passed=False)
            observations=collect_layers(generated['ap_init'],target/'post_gp',final_fplace=work/'post_route.fplace',legalized_fplace=work/'post_legalizer.fplace')
            row=dict(case=name,arm=arm,status=result['status'],metrics=result['metrics'],
                initializer_sha256=generated['ap_init_sha256'],init_consumed='TAILCOMPILE_V7_GP_INIT_CONSUMED' in log,
                native_gp_continued='TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' in log,fixed=fixed,
                layer_observations=observations,route_usage=usage,log=result['log'],vpr_manifest=str(target/'vpr/manifest.json'))
            row['passed']=row['status']=='success' and row['init_consumed'] and row['native_gp_continued'] and fixed['passed']
            write(durable,row); trials.append(row)
            if result['status'] in ('success','unroutable','complete'):
                prune_derived(work,protected=dependencies)
                removed=[]
                for p in list(target.glob('post_gp.*'))+[work/'post_route.fplace',work/'post_legalizer.fplace']:
                    if p.is_file() and not p.is_symlink() and target.resolve() in p.resolve().parents:
                        removed.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)));p.unlink()
                write(target/'trace_pruned_report.json',dict(files=removed,observations_and_route_usage_durable=True))
    unchanged=signatures=={str(p):sha(p) for p in inputs}
    report=dict(schema='qor-local-smoke-009-1',complete=True,
        passed=all(t['passed'] for t in trials) and unchanged and all(v['passed'] for v in model_checks.values()) and all(v['passed'] for v in parity.values()),
        source_read_only_verified=unchanged,binary_sha256=sha(binary),model_checks=model_checks,
        training_deployment_parity=parity,coordinate_score_gradients=gradients,
        bootstrap_epochs=2,bootstrap_teacher_only=True,feedback_training_performed=False,
        feedback_reason='no same-condition distinct-action measured QoR pool collected locally; do not pair different architectures',
        feedback_report=feedback,pnr_count=len(trials),pnr_success_count=sum(t['status']=='success' for t in trials),trials=trials,
        new_backend_builds=0,new_rrg_extractions=0,natural_design_training_completed=False,
        main_512_completed=False,performance_claim=False,Wmin_measured=False)
    write(output/'qor_smoke_report.json',report)
    print(dict(passed=report['passed'],pnr=report['pnr_count'],success=report['pnr_success_count'],report=str(output/'qor_smoke_report.json')),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=HERE/'runs/development_007_layer_local')
    p.add_argument('--aligned',type=Path,default=HERE/'runs/development_008_aligned_local')
    p.add_argument('--out',type=Path,default=HERE/'runs/development_009_qor_local')
    p.add_argument('--binary',type=Path,default=REPO/'.deps/tailcompile_v7_3d_layer/vtr/build_tailcompile_layer/vpr')
    p.add_argument('--timeout',type=int,default=180)
    args=p.parse_args()
    if not run(args.source,args.aligned,args.out,args.binary,timeout=args.timeout)['passed']: raise SystemExit(1)


if __name__=='__main__': main()
