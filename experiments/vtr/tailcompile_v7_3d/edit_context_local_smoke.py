"""012 offline engineering replay using two old tiny-architecture measurements.

No VPR runs. One design with balanced/asymmetric device fixtures, partial
measured candidate banks, is not natural-design generalization evidence.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from io_utils import read, write, freeze, sha
from repository_guard import validate_output_roots

HERE=Path(__file__).resolve().parent


def run(output):
    import torch
    from train_edit_context import prepare_cache,save_cache,fit_cache,load_model,evaluate_cache
    from train_residual import from_source
    from residual_actions import validate_residual_action
    from aligned_model import action_digest
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    output=Path(output).resolve()
    if output.parent!=HERE/'runs' or not output.name.startswith('development_012_edit_context_local'):
        raise ValueError('012 local output requires a separate ignored local cohort')
    source=HERE/'runs/development_011_residual_local'
    checkpoint=HERE/'runs/development_010_structured_local/models/decoder.pt'
    records=read(HERE/'runs/development_008_aligned_local/teacher_records.json')['records']
    records=[dict(r,architecture=r['condition_id'].split('/')[1]) for r in records]
    feedback=read(source/'feedback.json')
    deps=[checkpoint,source/'feedback.json']+[Path(r['graph']) for r in records]+[Path(e['action']) for e in feedback]
    pins={str(p):sha(p) for p in deps}
    validate_output_roots((output,),dependency_roots=(source,checkpoint))
    freeze(output/'protocol.json',dict(schema='edit-context-local-012-1',source_inputs=pins,
        new_pnr=0,natural_design_claim=False,measured_bank_partial=True,epochs=80,training_seeds=[12301,12302]))
    cache=prepare_cache(records,checkpoint,feedback,expected_sha256=sha(checkpoint))
    save_cache(cache,output/'feature_cache.pt')
    ids=sorted(r['condition_id'] for r in records)
    splits=[dict(id='full_fit',train=ids,evaluate=ids,heldout=False)]
    splits += [dict(id='hold_'+str(i),train=[c for c in ids if c!=cid],evaluate=[cid],heldout=True)
               for i,cid in enumerate(ids)]
    fits=[]
    for split in splits:
        for seed in (12301,12302):
            paired=[]
            for mode in ('G','E'):
                folder=output/'fits'/split['id']/f's{seed}'/mode
                if (folder/'training_report.json').exists():
                    fit=read(folder/'training_report.json')
                else:
                    fit=fit_cache(cache,split['train'],folder,mode=mode,seed=seed,epochs=80,
                                  snapshots=(5,20,80),device='cpu')
                model=load_model(fit['checkpoint'])
                evaluation=evaluate_cache(model,cache,split['evaluate'])
                paired.append(fit)
                fits.append(dict(split=split['id'],heldout=split['heldout'],mode=mode,seed=seed,
                    training_rank=fit['training_evaluation']['ranking'],evaluation=evaluation,
                    checkpoint=fit['checkpoint'],checkpoint_sha256=sha(fit['checkpoint']),
                    initial_state_sha256=fit['initial_state_sha256'],parameter_count=fit['parameter_count']))
                print(dict(split=split['id'],mode=mode,seed=seed,
                    train_pairs=fit['train_pairs'],identity=evaluation['identity_selections']),flush=True)
            assert paired[0]['initial_state_sha256']==paired[1]['initial_state_sha256']
            assert paired[0]['parameter_count']==paired[1]['parameter_count']
    deployment=[]
    for mode in ('G','E'):
        fit=next(f for f in fits if f['split']=='full_fit' and f['mode']==mode and f['seed']==12301)
        model=load_model(fit['checkpoint'])
        backbone=from_source(checkpoint)
        for record in records:
            graph=torch.load(record['graph'],weights_only=False,map_location='cpu')
            base=None
            for arm in ('L','Y','J'):
                base,bank=backbone.candidate_bank(graph,arm,base_action=base)
                action=model.select_action(graph,arm,checkpoint)
                assert action_digest(action) in {action_digest(a) for a in bank}
                validate_residual_action(graph,action,base_action=base,arm=arm)
                deployment.append(dict(condition_id=record['condition_id'],mode=mode,arm=arm,
                    candidate_id=action['candidate_id'],exact_original_candidate=True))
    unchanged=pins=={str(p):sha(p) for p in deps}
    report=dict(schema='edit-context-local-smoke-012-1',passed=unchanged,
        offline_fits=len(fits),new_pnr=0,source_unchanged=unchanged,
        full_matrix_server_fits_completed=False,natural_design_claim=False,performance_claim=False,
        measured_candidate_counts=[b['measured_candidate_count'] for b in cache['banks']],
        global_dim=cache['global_dim'],local_dim=cache['local_dim'],cache_bytes=(output/'feature_cache.pt').stat().st_size,
        paired_initial_weights_and_parameter_counts_identical=True,fits=fits,deployment=deployment)
    write(output/'edit_context_smoke_report.json',report)
    print(dict(passed=unchanged,offline_fits=len(fits),deployment_checks=len(deployment),new_pnr=0),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=HERE/'runs/development_012_edit_context_local')
    args=parser.parse_args()
    raise SystemExit(0 if run(args.out)['passed'] else 1)
