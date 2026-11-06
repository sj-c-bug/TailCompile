"""Run the v8_014 prospective joint layer/XY initialization comparison."""
import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from campaign import campaign_lock, freeze, identity
from architectures import COMMIT, SOURCE_ARCH
from generalization_campaign import CODE as PREVIOUS_CODE
from io_utils import install_signal_handlers, read, sha, write
from prospective_native import METHODS, SEEDS, model_sources, predict_panel, run_panel

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = HERE/'runs'
STAGES = ('qualify', 'predict', 'compare', 'report')
CODE = tuple(dict.fromkeys((*PREVIOUS_CODE, 'prospective_inputs.py', 'prospective_native.py',
                            'prospective_evaluation.py', 'prospective_campaign.py', 'run_prospective_campaign.sh')))


def seed_ledger(repo=REPO):
    """Inspect tracked trial path components and declared seed fields in prior protocols."""
    reserved = {14001, *SEEDS}
    reserved_strings = {str(n) for n in reserved}
    trial = re.compile(r'(?:^|[/_-])(?:s|seed)[_-]?(14001|14101|14102|14103)(?=[/_.-]|$)', re.I)
    paths = subprocess.check_output(['git', '-C', str(repo), 'ls-files', '-z', '--',
                                     'experiments/vtr', 'experiments/vtr3d']).decode().split('\0')
    protocols = {}
    collisions = []
    def inspect(value, seed_field=False):
        if isinstance(value, dict):
            return any(inspect(v, seed_field or 'seed' in key.lower()) for key,v in value.items())
        if isinstance(value, list):
            return any(inspect(v, seed_field) for v in value)
        return seed_field and str(value) in reserved_strings

    for relative in paths:
        if '/runs/' not in relative or '/runs/development_014' in relative:
            continue
        if trial.search(relative):
            collisions.append(relative)
        tail = relative.split('/runs/', 1)[1].split('/')
        if len(tail) == 2 and (tail[-1].endswith('protocol.json') or tail[-1] == 'queue_plan.json'):
            path = repo/relative
            protocols[relative] = sha(path)
            if inspect(read(path)):
                collisions.append(relative)
    if collisions:
        raise ValueError('014 reserved seeds appear in previous runs: '+str(collisions[:4]))
    return dict(seeds=sorted(reserved), tracked_protocols=protocols,
                scope='tracked 2D/3D run paths and top-level protocol/queue seed declarations')


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=('plan', 'panel', 'check', *STAGES, 'all'))
    p.add_argument('--manifest', type=Path, default=HERE/'prospective_panel_014.json')
    p.add_argument('--out', type=Path, default=RUNS/'development_014_prospective_joint')
    p.add_argument('--mapping-vtr-root', type=Path, default=os.environ.get('VTR_ROOT'))
    p.add_argument('--source012', type=Path, default=RUNS/'development_012_edit_context_parallel')
    p.add_argument('--source013', type=Path, default=RUNS/'development_013_generalization')
    p.add_argument('--backend-manifest', type=Path, default=REPO/'.deps/tailcompile_v7_3d_layer/vtr/backend_manifest.json')
    p.add_argument('--qualification-report', type=Path, default=RUNS/'development_007_layer/qualification/qualification_report.json')
    p.add_argument('--jobs', type=int, default=16)
    p.add_argument('--per-job-mib', type=int, default=8192)
    p.add_argument('--rrg-jobs', type=int, default=2)
    p.add_argument('--rrg-mib', type=int, default=8192)
    p.add_argument('--timeout', type=int, default=3600)
    args = p.parse_args(argv)
    if min(args.jobs, args.per_job_mib, args.rrg_jobs, args.rrg_mib, args.timeout) < 1:
        p.error('worker counts, memory budgets and timeout must be positive')
    args.out = args.out.resolve()
    if args.out.parent != RUNS or not args.out.name.startswith('development_014_'):
        p.error('output must be a direct development_014_ cohort under v7_3d/runs')
    args.scratch = REPO/'vtr_work/tailcompile_v7_3d/v8_014'/args.out.name
    args.workspace = REPO.parent
    args.allow_local = False
    args.threads = 1
    return args


def check_runtime(args):
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError('014 needs the Torch environment used for 013; check TAILCOMPILE_3D_PYTHON') from exc
    args.torch_version = str(torch.__version__)
    root = args.mapping_vtr_root
    if subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip() != COMMIT:
        raise ValueError('014 mapping VTR checkout must use the pinned 40f8 commit')
    required = (root/'vtr_flow/scripts/run_vtr_flow.py', root/SOURCE_ARCH, root/'build/bin/yosys',
                root/'abc/abc', args.backend_manifest, args.qualification_report)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError('014 required server artifact: '+str(path))
    for executable in (root/'build/bin/yosys',root/'abc/abc'):
        if not os.access(executable,os.X_OK):
            raise PermissionError('014 non-executable mapping tool: '+str(executable))
    backend = read(args.backend_manifest)
    args.binary = Path(backend['binary']).resolve()
    args.vtr_root = root
    if backend['vtr_commit'] != COMMIT or sha(args.binary) != backend['binary_sha256']:
        raise ValueError('014 qualified 3D binary does not match its manifest')
    report = read(args.qualification_report)
    if not report.get('passed') or not report.get('complete') or report.get('binary_sha256') != backend['binary_sha256']:
        raise ValueError('014 requires passing qualification for the same 3D backend')
    earlier = read(RUNS/'development_007_layer/layer_protocol.json')
    if (earlier['backend_manifest_sha256'] != sha(args.backend_manifest) or
            earlier['qualification_sha256'] != sha(args.qualification_report)):
        raise ValueError('014 backend and qualification must match the natural-design campaign')
    return model_sources(args.source012,args.source013)


def checked_panel(out):
    value = read(out/'qualified_panel.json')
    conditions = value['conditions']
    if len(conditions) != 8 or len({c['family'] for c in conditions}) != 2:
        raise ValueError('014 requires both families and all eight qualified conditions')
    return conditions


def locked_predictions(out, conditions):
    path = out/'predictions/prediction_report.json'
    lock = read(out/'predictions/selection.json')
    if lock['predictions_sha256'] != sha(path) or lock['conditions_sha256'] != identity(conditions):
        raise ValueError('014 prediction lock or qualified conditions changed')
    return read(path)


def verify_measurements(out, rows):
    """Bind compact trial summaries to the measured native input, seed and QoR."""
    for row in rows:
        method = row.get('measured_method', row['method'])
        if method not in METHODS:
            raise ValueError('014 unknown physical method')
        folder = out/'comparison'/row['condition_id']/f's{row["seed"]}'/method
        if (out/'comparison').resolve() not in folder.resolve().parents:
            raise ValueError('014 invalid trial location')
        path = folder/'vpr/manifest.json'
        if sha(path) != row['vpr_manifest_sha256']:
            raise ValueError('014 native manifest hash differs from the trial row')
        native = read(path)
        measured = read(folder/'metrics.json')
        if any(measured[key] != value for key,value in row.items()
               if key not in ('method','measured_method','identical_input_reuse','deployed_action_sha256')):
            raise ValueError('014 archived row differs from per-trial evidence')
        if measured['method'] != method or (row['method'] == method and
                measured['deployed_action_sha256'] != row['deployed_action_sha256']):
            raise ValueError('014 recorded action differs from physical trial')
        if any(native['metrics'][k] != row[k] for k in ('wirelength','cpd_ns')):
            raise ValueError('014 QoR differs from native manifest')
        sig = native['signature']
        command = sig['command']
        if (int(command[command.index('--seed')+1]) != row['seed'] or
                int(command[command.index('--route_chan_width')+1]) != row['width'] or
                sig['inputs'].get(sig['env'].get('TC_GP_INIT')) != row['ap_init_sha256']):
            raise ValueError('014 seed, width or native initializer hash mismatch')


def execute(stage, args, campaign=None, models=None):
    from prospective_inputs import prepare_inputs, calibrate_inputs, qualified_conditions
    if stage == 'qualify':
        prepare_inputs(campaign)
        calibrate_inputs(campaign)
        conditions = qualified_conditions(campaign)
        freeze(args.out/'qualified_panel.json', dict(conditions=conditions))
        return
    conditions = checked_panel(args.out)
    if stage == 'report':
        from prospective_evaluation import evaluate
        predictions = locked_predictions(args.out, conditions)
        raw = read(args.out/'comparison/trials_report.json')
        if not raw['complete'] or raw['predictions_sha256'] != sha(args.out/'predictions/prediction_report.json'):
            raise ValueError('014 comparison was not bound to the frozen prediction panel')
        verify_measurements(args.out, raw['rows'])
        write(args.out/'prospective_report.json', evaluate(raw['rows'], conditions, predictions))
        return
    if identity(qualified_conditions(campaign)) != identity(conditions):
        raise ValueError('014 qualified input evidence differs from archived panel')
    if stage == 'predict':
        if (args.out/'predictions/selection.json').is_file():
            if locked_predictions(args.out, conditions)['models'] != models:
                raise ValueError('014 frozen prediction checkpoints changed')
            return
        if (args.out/'comparison').exists():
            raise ValueError('014 cannot create predictions after primary measurements exist')
        predict_panel(campaign, conditions, models)
        freeze(args.out/'predictions/selection.json', dict(conditions_sha256=identity(conditions),
               predictions_sha256=sha(args.out/'predictions/prediction_report.json')))
        return
    predictions = locked_predictions(args.out, conditions)
    if predictions['models'] != models:
        raise ValueError('014 scorer bindings changed after prediction lock')
    run_panel(campaign, conditions, predictions)


def main(argv=None):
    args = arguments(argv)
    from prospective_inputs import register_panel, make_campaign
    if args.stage == 'plan':
        print(json.dumps(dict(panel=read(args.manifest), qualification_seed=14001, widths=[128,192,256],
              backend_seeds=SEEDS, physical_methods=METHODS, qualification_limit=24, primary_limit=96),indent=2))
        return
    install_signal_handlers()
    if args.stage == 'report':
        with campaign_lock(args.out,'v8_014_report'):
            execute('report',args)
        return
    if args.mapping_vtr_root is None:
        args.mapping_vtr_root = Path(read(RUNS/'development_004_capacity/protocol.json')['mapping_vtr_root'])
    args.mapping_vtr_root = args.mapping_vtr_root.resolve()
    args.manifest = args.manifest.resolve()
    panel = register_panel(args.manifest,args.mapping_vtr_root)
    ledger = seed_ledger()
    if args.stage == 'panel':
        print(json.dumps(dict(panel=panel, reserved_seeds=ledger['seeds']),indent=2))
        return
    models = check_runtime(args)
    if args.stage == 'check':
        print(json.dumps(dict(ready=True, designs=[d['design'] for d in panel['designs']],
            models={k:v['sha256'] for k,v in models.items()}, binary=str(args.binary), out=str(args.out)),indent=2))
        return
    with campaign_lock(args.out,'v8_014_'+args.stage):
        frozen = dict(schema='v8-014-prospective-joint-1', panel=panel, seed_ledger=ledger, models=models,
            primary_scorer='E012', methods=list(METHODS), measurement_seeds=list(SEEDS),
            qualification_seed=14001, widths=[128,192,256], max_qualification_runs=24, max_primary_runs=96,
            code={n:sha(HERE/n) for n in CODE}, binary_sha256=sha(args.binary),
            qualification_sha256=sha(args.qualification_report), backend_manifest_sha256=sha(args.backend_manifest),
            mapping_root=str(args.mapping_vtr_root), source_arch_sha256=sha(args.mapping_vtr_root/SOURCE_ARCH),
            python=sys.version, torch_version=args.torch_version,
            timeout=args.timeout, jobs=args.jobs, per_job_mib=args.per_job_mib,
            rrg_jobs=args.rrg_jobs, rrg_mib=args.rrg_mib)
        freeze(args.out/'prospective_protocol.json',frozen)
        campaign = make_campaign(args,panel)
        for stage in STAGES if args.stage=='all' else (args.stage,):
            print(f'V8_014_STAGE_START stage={stage}',flush=True)
            execute(stage,args,campaign,models)
            print(f'V8_014_STAGE_COMPLETE stage={stage}',flush=True)
        print(f'V8_014_GROUP_COMPLETE group={args.stage}',flush=True)


if __name__ == '__main__':
    main()
