"""Actual 3D graph -> trained GNN -> XYZ -> native GP engineering smoke.

This intentionally overfits a small training fixture. It does not select a
research checkpoint or claim runtime/generalization benefits.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from io_utils import emit, read, write, freeze, sha, run_trial, vpr_command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--architecture', type=Path, required=True)
    parser.add_argument('--blif', type=Path, required=True)
    parser.add_argument('--device-graph', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--timeout', type=int, default=180)
    args = parser.parse_args()
    for name in ('binary','architecture','blif','device_graph','out'):
        setattr(args, name, getattr(args,name).resolve())
    from repository_guard import validate_output_roots
    validate_output_roots((args.out,))
    args.out.mkdir(parents=True, exist_ok=True)
    import torch
    torch.set_num_threads(1)
    from data import build_cached_graph, raw_teacher
    from generation import lower_initialization
    from model import Hierarchical3DInitializer, capacity_only
    from train import train
    from supervision import project_teacher
    device = read(args.device_graph)
    width = int(device['channel_width'])
    identity = dict(schema='tailcompile-v7-local-trained-smoke-1',
        purpose='one-fixture teacher overfit and real backend interface, not independent QoR evidence',
        binary=sha(args.binary), arch=sha(args.architecture), blif=sha(args.blif),
        device_graph=sha(args.device_graph), epochs=args.epochs, training_seed=601,
        teacher_seed=511, deployment_seed=531, graph_data=sha(Path(__file__).with_name('data.py')))
    freeze(args.out/'protocol.json', identity)
    exports = args.out/'exports'
    exports.mkdir(exist_ok=True)
    prefix, teacher = exports/'structure', exports/'teacher'
    command = vpr_command(args.binary,args.architecture,args.blif,width,511)
    row = run_trial(args.out/'teacher',command,inputs=[args.architecture,args.blif],
        env=dict(TC_GP_STRUCTURE=prefix, TC_GP_EXPORT=teacher),timeout=args.timeout,
        metadata=dict(stage='train_fixture_teacher',backend=identity['binary']))
    if row['status'] != 'success':
        raise RuntimeError('native teacher failed; see durable teacher log')
    paths = build_cached_graph(prefix,'qualification_fixture',args.blif,args.architecture,
        args.device_graph,args.out/'graph',family='engineering_fixture',split='train')
    logical = read(paths['logic'])
    target = raw_teacher(teacher, logical)
    write(args.out/'teacher_raw.json',target)
    graph = torch.load(paths['graph'],map_location='cpu',weights_only=False)
    derived, supervision_audit = project_teacher(graph,target)
    write(args.out/'teacher_projected.json',derived)
    write(args.out/'supervision_audit.json',supervision_audit)
    records = [dict(graph=paths['graph'],teacher=str(args.out/'teacher_projected.json'),
                    split='train',family='engineering_fixture',condition_id='local_H1Z1_fixture')]
    write(args.out/'records.json',records)
    if not (args.out/'model'/'training_report.json').exists():
        started = time.perf_counter()
        trained = train(records,args.out/'model',epochs=args.epochs,hidden=16,seed=601,device='cpu')
        write(args.out/'training_wall.json',dict(seconds=time.perf_counter()-started))
    else:
        trained = read(args.out/'model'/'training_report.json')
    checkpoint = trained['checkpoints'][-1]
    if isinstance(checkpoint,dict):
        checkpoint = checkpoint.get('checkpoint',checkpoint.get('path'))
    payload = torch.load(checkpoint,map_location='cpu',weights_only=False)
    model = Hierarchical3DInitializer(**payload['config'])
    model.load_state_dict(payload['state_dict'])
    model.eval()
    rows = []
    for method in ('default','capacity_only','gnn'):
        env, inputs = {}, [args.architecture,args.blif]
        init_manifest = None
        if method != 'default':
            started = time.perf_counter()
            with torch.no_grad():
                prediction = capacity_only(graph) if method=='capacity_only' else model.predict(graph)
            prediction_seconds = time.perf_counter()-started
            init_manifest = lower_initialization(logical,device,prediction,args.out/'initializers'/method,method=method)
            env = dict(TC_GP_INIT=init_manifest['ap_init'], TC_GP_TRACE=exports/(method+'_injected'),
                       TC_GP_EXPORT=exports/(method+'_post_gp'))
            inputs.append(init_manifest['ap_init'])
            init_manifest['prediction_wall_seconds'] = prediction_seconds
            write(args.out/'initializers'/method/'generation.json',init_manifest)
        row = run_trial(args.out/'deployment'/method,
            vpr_command(args.binary,args.architecture,args.blif,width,531),inputs=inputs,env=env,
            timeout=args.timeout,metadata=dict(stage='local_trained_smoke',method=method,
                checkpoint=sha(checkpoint) if method=='gnn' else None,backend=identity['binary']))
        text = Path(row['log']).read_text(errors='replace')
        rows.append(dict(method=method,status=row['status'],metrics=row['metrics'],
                         initializer=init_manifest,
                         init_consumed=method=='default' or 'TAILCOMPILE_V7_GP_INIT_CONSUMED' in text,
                         native_gp_continued=method=='default' or
                             'TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1' in text))
    report = dict(schema='tailcompile-v7-local-trained-smoke-1',
        purpose=identity['purpose'], complete=True,
        interface_passed=all(row['status']=='success' and row['init_consumed'] and row['native_gp_continued']
                             for row in rows),
        graph=dict(nodes=paths['node_count'],regions=paths['region_count']),
        weights_changed=trained.get('weights_changed'), training_report=str(args.out/'model'/'training_report.json'),
        checkpoint=str(checkpoint), checkpoint_sha256=sha(checkpoint),
        checkpoint_selection='fixed final epoch for engineering smoke; pending real research validation',
        runtime_acceptance_evaluated=False, deployment_rows=rows)
    write(args.out/'smoke_report.json',report)
    emit('learned_3d_smoke',passed=report['interface_passed'],graph=report['graph'],
         weights_changed=report['weights_changed'])
    raise SystemExit(0 if report['interface_passed'] else 1)


if __name__=='__main__':
    main()
