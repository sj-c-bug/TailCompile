"""007 real tiny graph/teacher training smoke; adds no complete P&R runs.

Reuses the balanced qualification's GP export. Pack-only RR graph exports are
separate preparation, then a trained Tail initializer is lowered for asymmetric
qualification. This is engineering evidence, not research performance evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from io_utils import read, write, sha, freeze, run_trial, prune_derived
from repository_guard import validate_output_roots

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=HERE/'runs/development_007_layer_local')
    parser.add_argument('--binary', type=Path, default=REPO/'.deps/tailcompile_v7_3d_layer/vtr/build_tailcompile_layer/vpr')
    args = parser.parse_args()
    args.out = args.out.resolve(); args.binary = args.binary.resolve()
    validate_output_roots((args.out,))
    import torch
    from data import build_logic, raw_teacher
    from device_graph import build_device_graph
    from fast_graph import make_graph
    from layer_features import augment_graph
    from vertical_risk import prepare_device
    from supervision import project_teacher
    from train_layers import train_shared, prepare_arms, train_feedback, load_model
    from layer_model import generate
    from generation import lower_initialization
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    inputs = {r['name']: r for r in read(args.out/'qualification_inputs.json')}
    root = args.out/'model_smoke'; root.mkdir(parents=True, exist_ok=True)
    freeze(root/'protocol.json', dict(schema='layer-local-model-smoke-007', binary_sha256=sha(args.binary),
        shared_epochs=20, candidate_epochs=5, hidden=16, training_seed=2801,
        teacher='existing balanced seed2701 I post_gp', complete_pnr_added=0,
        purpose='real graph/teacher/GNN/initialization engineering, no research acceptance'))
    graphs, logicals, devices, graph_files, preparation, cleanup = {}, {}, {}, {}, [], []
    for name in ('balanced', 'asymmetric'):
        config = inputs[name]; folder = root/name; folder.mkdir(exist_ok=True)
        rr = folder/'rr_graph.xml'
        graph_file = folder/'graph.pt'; graph_files[name] = graph_file
        if not graph_file.exists():
            command = [str(args.binary), config['arch'], config['blif'], '--pack', '--num_workers', '1',
                '--seed', '2701', '--route_chan_width', str(config['width']), '--place_chan_width', str(config['width']),
                '--write_rr_graph', str(rr), *config['options']]
            started = time.perf_counter()
            trial = run_trial(folder/'rr_export', command, inputs=[args.binary, config['arch'], config['blif']], timeout=180)
            if trial['returncode'] != 0 or not rr.is_file():
                raise RuntimeError('pack-only RR export failed: '+trial['log'])
            rr_hash, rr_bytes = sha(rr), rr.stat().st_size
            device = build_device_graph(rr, config['arch'], tile_size=4)
            write(folder/'device_graph.json', device)
            access = prepare_device(rr, device, out=folder/'vertical_summary.json')
            logical = build_logic(config['structure'], name+'_fixture', config['blif'], config['arch'],
                                  family='engineering_fixture', split='train')
            graph = augment_graph(make_graph(logical, device), logical, device, access, config['blif'], config['structure'])
            torch.save(graph, graph_file); write(folder/'logic.json', logical)
            record = dict(name=name, rr_graph_sha256=rr_hash, rr_graph_bytes=rr_bytes,
                rr_export_manifest=str(folder/'rr_export/manifest.json'), graph_sha256=sha(graph_file),
                seconds=time.perf_counter()-started, complete_pnr_executed=False)
            write(folder/'preparation.json', record)
            # No downstream stage needs the raw graph after both summaries exist.
            cleanup.append(dict(path=str(rr), sha256=rr_hash, bytes=rr_bytes)); rr.unlink()
            prune_derived(trial['workdir'], protected=[config['arch'], config['blif']])
        graphs[name] = torch.load(graph_file, weights_only=False, map_location='cpu')
        logicals[name] = read(folder/'logic.json'); devices[name] = read(folder/'device_graph.json')
        preparation.append(read(folder/'preparation.json'))
    raw = raw_teacher(args.out/'deployments/balanced/s2701/I/post_gp', logicals['balanced'])
    projected, projection_audit = project_teacher(graphs['balanced'], raw)
    write(root/'teacher_raw.json', raw); write(root/'teacher_projected.json', projected)
    write(root/'teacher_projection.json', projection_audit)
    records = [dict(condition_id='engineering_fixture/balanced/w320', family='engineering_fixture', split='train',
                    graph=str(graph_files['balanced']), teacher=str(root/'teacher_projected.json'))]
    write(root/'teacher_records.json', dict(records=records))
    training = root/'normalized_training'
    shared_report = training/'shared/training_report.json'
    shared = read(shared_report) if shared_report.exists() else train_shared(records, training/'shared', epochs=20, hidden=16, seed=2801)
    prepare_report = training/'prepared/preparation_report.json'
    prepared = read(prepare_report) if prepare_report.exists() else prepare_arms(shared, records, training/'prepared', epochs=5, seed=2801)
    lowered = {}
    for method in ('D', 'L', 'T'):
        checkpoint = shared['checkpoint'] if method == 'D' else prepared['arms'][method]['checkpoint']
        model = load_model(checkpoint)
        prediction = generate(model, graphs['asymmetric'], method=method)
        lowered[method] = lower_initialization(logicals['asymmetric'], devices['asymmetric'], prediction,
            root/'asymmetric_initializers'/method, method=method)
        write(root/'asymmetric_initializers'/method/'diagnostics.json',
            dict(soft_vertical_risk=prediction['soft_vertical_risk'], hard_vertical_risk=prediction['hard_vertical_risk'],
                 capacity=prediction['capacity_diagnostics'], phase_seconds=prediction['phase_seconds']))
    # Qualification routes do not supply two distinct learned actions on the
    # balanced training condition. Preserve that limitation instead of inventing
    # runtime preferences from different architectures or the same I/S action.
    feedback = train_feedback(records, prepared, training/'feedback', [], epochs=10, seed=2801)
    # Retain cumulative reclamation evidence across repeated read/cache-only
    # invocations; the original large export is intentionally no longer present.
    cumulative_cleanup = []
    for prep in preparation:
        raw_path = root/prep['name']/'rr_graph.xml'
        if not raw_path.exists():
            cumulative_cleanup.append(dict(path=str(raw_path), sha256=prep['rr_graph_sha256'],
                bytes=prep['rr_graph_bytes'], source='persisted pack-only preparation record'))
    superseded = []
    for folder in (root/'shared', root/'prepared', root/'feedback'):
        if folder.is_dir() and root.resolve() in folder.resolve().parents:
            for checkpoint in folder.rglob('*.pt'):
                if checkpoint.is_file() and not checkpoint.is_symlink() and folder.resolve() in checkpoint.resolve().parents:
                    superseded.append(dict(path=str(checkpoint), sha256=sha(checkpoint), bytes=checkpoint.stat().st_size))
                    checkpoint.unlink()
    write(root/'cleanup_ledger.json', dict(raw_rrg=cumulative_cleanup, superseded_checkpoints=superseded,
        raw_rrg_removed_this_invocation_bytes=sum(r['bytes'] for r in cleanup)))
    report = dict(schema='layer-local-model-smoke-007', complete=True, binary_sha256=sha(args.binary),
        research_performance_evaluated=False, complete_pnr_added=0,
        teacher_source=str(args.out/'deployments/balanced/s2701/I/post_gp'),
        graph_nodes={k:len(g['demand']) for k,g in graphs.items()},
        graph_regions={k:len(g['capacity']) for k,g in graphs.items()}, preparation=preparation,
        teacher_projection=projection_audit, shared_training_report=str(shared_report),
        shared_xy_sha256=prepared['shared_xy_sha256'],
        tail_normalization=shared['tail_normalization'],
        shared_training_weights_changed=shared['initial_state_sha256']!=shared['final_state_sha256'],
        layer_training_weights_changed={m:prepared['arms'][m]['initial_state_sha256']!=prepared['arms'][m]['final_state_sha256'] for m in ('L','T')},
        initializers=lowered, feedback_pairs=feedback['feedback_pairs'],
        feedback_training_performed=False, feedback_reason='no measured distinct-action same-condition preference labels',
        research_training_or_independent_validation_claim=False,
        cleanup=cumulative_cleanup, reclaimed_bytes=sum(r['bytes'] for r in cumulative_cleanup),
        superseded_checkpoints_removed=superseded)
    write(args.out/'model_smoke_report.json', report)
    print(json.dumps(dict(complete=True, T_ap_init=lowered['T']['ap_init'],
        graph_nodes=report['graph_nodes'], reclaimed_bytes=report['reclaimed_bytes']), indent=2), flush=True)


if __name__ == '__main__':
    main()
