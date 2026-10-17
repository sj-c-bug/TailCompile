# TailCompile v6.3: expanded interventions and Default choice

This experiment follows the v6.2 pilot. It has two independent goals:

1. Re-test three promising swaps on fresh seeds 61/62 against the already recorded
   `gnn_base` runs. This requires six new VPR runs and does not alter the old pilot.
2. Train a placement chooser with considerably more labels. The input remains the
   six v6.1 design families, two widths per family. Each condition has Default
   (unbound VPR placement), `gnn_base`/`noop`, and four distinct interventions at
   each of 1%, 2%, and 5% moved clusters: 14 actions per condition. Seeds 51/52/53
   give 12 × 14 × 3 = 504 label trials, aggregated into 168 training examples.
   The previous 56 v6.2 trials are reused only when candidate hashes and VTR
   manifests match. All six design families have labels; models leave out the
   entire target family during training. Fresh seeds 71/72/73 are reserved for the
   108-run closed-loop evaluation.

The neural model has a learned Default branch. Because Default has no `.fplace`,
its graph uses the `gnn_base` assignment as a reference context together with an
explicit Default indicator; the learned branch estimates the outcome when that
reference is *not* enforced. Swap branches see their actual proposed assignments.
The Default input is a proxy and should be assessed through held-out VPR results.

## Commands

Use the pinned server VTR commit `40f8d19708b0b60e2aa04452deb1cfed68d96b0b`.
On each instance, source the usual environment file and use shared directories:

```bash
export TAILCOMPILE_V63_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v63/runs/expanded_001"
export TAILCOMPILE_V63_JOBS=4
```

CPU instance, candidate probe (six VPR runs):

```bash
bash experiments/vtr/tailcompile_v63/run_pipeline.sh probe
```

CPU instance, full candidate generation, labels and dataset:

```bash
bash experiments/vtr/tailcompile_v63/run_pipeline.sh prepare
```

The probe and prepare phases can run concurrently because their output trees and
seeds differ. Set `TAILCOMPILE_V63_JOBS` according to available memory as well as
CPU cores; one large VPR run can consume several GiB. The prepare phase can be
restarted in the same output directory and reuses completed VPR manifests.

GPU instance, after prepare:

```bash
bash experiments/vtr/tailcompile_v63/run_pipeline.sh train
```

CPU instance, after training:

```bash
bash experiments/vtr/tailcompile_v63/run_pipeline.sh evaluate
```

Check status any time:

```bash
bash experiments/vtr/tailcompile_v63/run_pipeline.sh status
```

Expected counts: probe 6, pool 168, labels 504, examples 168, checkpoints 18,
selections 12, final evaluation 108. The final comparison is against both Default
and `gnn_base`; it reports routability first and QoR on jointly successful runs.
There is no automatic improvement gate that discards an unfavorable result.

## Interpretation boundary

Four swap replicates and three placement seeds improve label stability. They do
not create new independent circuit families: this dataset still contains six
families. Leave-family-out scores and fresh-seed VPR results are evidence for
these six families only. Before a broader claim about GNN generalization, add
independent benchmark families using the same input preparation and frozen
evaluation protocol. In particular, the learned Default branch sees only ten
other design/width contexts in each held-out fold, so uncertainty remains high.

Keep `bundle.pt`, model checkpoints and raw VPR working files out of Git. Sync
the JSON reports, generated final `.fplace`, per-trial manifest/metrics, and logs.
