"""GPU-only runtime-benefit distillation into the actual hierarchical initializer.

There is no VPR subprocess here and no learned runtime scalar substituted for
the generator. Train-only candidates selected using measured total runtime and
frozen QoR guards replace native-GP teacher targets. The existing generator's
coordinate, region, connectivity and soft aggregate-capacity losses are reused.
Actual runtime improvement must still be established by held-out VPR execution.
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path

from bridge import HERE, V68, emit, freeze, read, sha, write


SCHEMA = 'tailcompile-v69-runtime-distillation-1'


def training_inputs(a):
    """Validate provenance before importing Torch or allocating a GPU model."""
    source, out = Path(a.source).resolve(), Path(a.out).resolve()
    for name in ('epochs', 'threads', 'graph_cache', 'chunk_size'):
        if getattr(a, name) < 1:
            raise ValueError(f'{name} must be positive')
    if not math.isfinite(a.lr) or a.lr <= 0:
        raise ValueError('lr must be finite and positive')
    protocol = read(out/'protocol.json')
    if Path(protocol['source']).resolve() != source:
        raise ValueError('Training source differs from frozen v69 source')
    dataset_path = source/'dataset.json'
    if sha(dataset_path) != protocol['source_dataset_sha256']:
        raise ValueError('Frozen source dataset changed')
    checkpoint = Path(protocol['initial_checkpoint']).resolve()
    checkpoint_sha = sha(checkpoint)
    if checkpoint_sha != protocol['initial_checkpoint_sha256']:
        raise ValueError('Initial generator checkpoint changed')
    feedback_path = out/'feedback/feedback_dataset.json'
    feedback = read(feedback_path)
    if not feedback.get('complete'):
        raise ValueError('Runtime feedback is incomplete; finish CPU feedback first')
    if not feedback.get('all_route_outcomes_known', False):
        raise ValueError('Runtime feedback has unresolved route/infrastructure outcomes')
    if feedback.get('generator_checkpoint_sha256') != checkpoint_sha:
        raise ValueError('Runtime labels came from another generator checkpoint')
    selected = feedback.get('selected', [])
    if not selected:
        raise ValueError('No runtime-beneficial training candidates passed the frozen guards; '
                         'refusing an empty-feedback fine-tune. Retain the initial checkpoint '
                         'and diagnose candidate generation instead of claiming learned improvement.')
    dataset = read(dataset_path)
    families, conditions = {}, {}
    for row in dataset['examples']:
        if row['split'] not in ('train', 'validation'):
            raise ValueError('Final-test examples must never enter runtime distillation')
        if families.setdefault(row['family'], row['split']) != row['split']:
            raise ValueError('Training/validation family overlap')
        key = row['design'], row['width']
        if key in conditions:
            raise ValueError('Duplicate source training condition')
        conditions[key] = row
    if 'validation' not in families.values():
        raise ValueError('Validation families are required for runtime distillation')
    seen = set()
    for row in selected:
        key = row['design'], row['width']
        if key in seen:
            raise ValueError('Duplicate runtime feedback condition')
        seen.add(key)
        original = conditions.get(key)
        if (row['split'] != 'train' or original is None or original['split'] != 'train'
                or row['family'] != original['family']):
            raise ValueError('Runtime feedback must match a train-family source condition')
        if sha(row['target']) != row['target_sha256']:
            raise ValueError('Runtime-selected initializer target changed')

    args = argparse.Namespace(dataset=dataset_path, out=out/'models/runtime',
        initialize=checkpoint, feedback=feedback_path, device=a.device,
        epochs=a.epochs, hidden=64, chunk_size=a.chunk_size,
        tile_size=int(dataset['tile_size']), threads=a.threads, lr=a.lr,
        seed=a.seed, graph_cache=a.graph_cache, allow_no_validation=False)
    signature = dict(schema=SCHEMA, source=str(source),
        protocol_sha256=sha(out/'protocol.json'), dataset_sha256=sha(dataset_path),
        feedback_sha256=sha(feedback_path), initial_checkpoint_sha256=checkpoint_sha,
        budgets=protocol['budgets'], decoder_options=protocol['options'],
        epochs=a.epochs, lr=a.lr, seed=a.seed, chunk_size=a.chunk_size,
        selected_conditions=len(selected),
        training_families=sorted(k for k,v in families.items() if v=='train'),
        validation_families=sorted(k for k,v in families.items() if v=='validation'),
        final_test_examples=0,
        objective='Distill measured runtime-beneficial train-only initializers; '
                  'preserve coordinate/region/connectivity/soft-density losses',
        model_selection='Validation teacher reconstruction loss; this is not a held-out runtime guarantee')
    return args, signature


def train_runtime(a):
    args, signature = training_inputs(a)
    signature['implementation_sha256'] = {str(p.relative_to(HERE.parent)): sha(p) for p in
        (Path(__file__), HERE/'fast_model.py', HERE/'generate.py',
         V68/'train.py', V68/'model.py', V68/'supervision.py')}
    freeze(args.out/'runtime_training_protocol.json', signature)
    emit('runtime_distillation_start', selected_conditions=signature['selected_conditions'],
         epochs=args.epochs, device=args.device, output=str(args.out),
         cpu_vpr_tasks=False, final_test_examples=0)
    spec = importlib.util.spec_from_file_location('_tailcompile_v68_runtime_train', V68/'train.py')
    legacy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(legacy)
    legacy.train(args)
    report = read(args.out/'training_report.json')
    if not report.get('complete') or sha(args.out/'generator.pt') != report['checkpoint_sha256']:
        raise ValueError('Legacy generator training did not produce a verified complete checkpoint')
    write(args.out/'runtime_training_report.json', dict(schema=SCHEMA, complete=True,
        checkpoint=str(args.out/'generator.pt'), checkpoint_sha256=report['checkpoint_sha256'],
        runtime_protocol_sha256=sha(args.out/'runtime_training_protocol.json'),
        selected_conditions=signature['selected_conditions'],
        training_families=report['training_families'], validation_families=report['validation_families'],
        final_test_examples=0, epochs_completed=report['epochs_completed'],
        weights_changed=report['weights_changed'],
        heldout_runtime_validated=False,
        next_stage='CPU paired evaluation with inference/decoding/lowering charged to runtime'))
    emit('runtime_distillation_complete', checkpoint=str(args.out/'generator.pt'),
         epochs_completed=report['epochs_completed'], weights_changed=report['weights_changed'])


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True, type=Path, help='Frozen v68 cohort output')
    p.add_argument('--out', required=True, type=Path, help='New v69 cohort output')
    p.add_argument('--device', default='cuda')
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--threads', type=int, default=2)
    p.add_argument('--graph-cache', type=int, default=2)
    p.add_argument('--chunk-size', type=int, default=512)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=6901)
    return p


if __name__ == '__main__':
    train_runtime(parser().parse_args())
