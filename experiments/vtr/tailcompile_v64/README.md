# TailCompile v6.4: frozen-placement minimum-width audit (2D)

This campaign tests the user's main 2D endpoint: can a learned `.fplace` route
at a **smaller channel width** than both unbound Default and `gnn_base`?
The VTR version is the pinned server commit
`40f8d19708b0b60e2aa04452deb1cfed68d96b0b`.

The v6.3 chooser made a physical change in only two conditions:
`conv_layer` anchored at width 140 (`swap02_r3`) and `lenet` anchored at width
160 (`swap05_r0`). Its other ten choices were exactly the `gnn_base` fplace;
re-running those would not measure an intervention. The two changed placements
are **frozen**: no model training, reselection, or candidate generation occurs
during this experiment. Each condition compares Default, `gnn_base`, and v6.3
using the same packed net, device size, VPR settings, and seed at every width.

Predeclared grids are 80–160 for `conv_layer` and 140–220 for `lenet`, both in
steps of 10. Seeds 81–90 are new, distinct from v6.3 labels and evaluation.
All 540 VPR tasks are independent and may run in parallel. A full grid is
deliberate: VPR heuristics do not guarantee monotone success as width rises,
so binary search alone could report a false minimum. `w_min_grid` is the
smallest successful **tested** width per seed. Success at the grid floor is
left-censored; no success through the ceiling is right-censored. The report
also gives `w80_grid`, the smallest tested width where at least 8/10 seeds
succeed, and paired QoR ratios on jointly successful runs. A width reduction
is evidence of lower required *nominal routing-channel capacity in VTR*; it
does not mean resources were removed from an existing physical FPGA.

## Run on a CPU instance

Use the same mounted workspace and `VTR_ROOT` as v6.3. No GPU is needed.

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export TAILCOMPILE_V64_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v64/runs/wmin_001"
export TAILCOMPILE_V64_JOBS=8

bash experiments/vtr/tailcompile_v64/run_pipeline.sh plan
bash experiments/vtr/tailcompile_v64/run_pipeline.sh status
```

The plan checks that the selected `.fplace`, baseline `.fplace`, BLIF and packed
net exist and match the frozen v6.3 selection hashes. Check `plan.json` before
starting the expensive sweep. Changing widths or seeds after `plan` requires a
new output directory; never overwrite a partially completed cohort.

For a background run with a persistent log:

```bash
mkdir -p "$TAILCOMPILE_REPO/logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
log="$TAILCOMPILE_REPO/logs/v64_wmin_${timestamp}.log"
nohup bash experiments/vtr/tailcompile_v64/run_pipeline.sh run \
  >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
```

Use `status` while the run proceeds. Once all 540 trials finish, `run` writes
`wmin_report.json`. If the shell or instance stops, **run the same command
again**: completed manifests are verified and reused. The script quarantines
only provably inactive, manifest-less partial runs inside this campaign. A
concurrent duplicate sweep is blocked by `.vpr.lock`.

Set `TAILCOMPILE_V64_JOBS` from both CPU cores **and memory**, not CPU cores
alone. Previous large VPR jobs consumed several GiB each; 8 is a starting
point, not a universal safe limit. Increase only after checking memory usage.
`TAILCOMPILE_V64_TIMEOUT` defaults to 1800 seconds; timeout retries up to the
existing 7200-second retry cap. If a VPR run has a non-routing failure, inspect
its `vpr.out`; do not count it as an unroutable result.

## What constitutes useful 2D evidence?

Primary: paired `w_min_grid` reductions and/or lower `w80_grid` versus **both**
Default and `gnn_base`, with censoring and nonmonotonic widths disclosed.
Secondary: region/channel P99, wirelength and CPD on jointly successful runs.
If the grid floor/ceiling censors the boundary, extend the width grid in a new
cohort before drawing a conclusion. A robust width improvement in one frozen
condition is a valid 2D proof of concept and can justify temporarily closing
the 2D engineering stage while prioritizing 3D work; it is not a claim of
cross-design generality. If no improvement appears, adding more seeds to the
same two placements is unlikely to help—candidate generation/selection needs
revision before another costly sweep.

After completion, sync `plan.json`, `wmin_report.json`, and lightweight
`manifest.json`/`trial.json` evidence. The `.gitignore` excludes VPR logs,
packed-net copies, occupancy files, routes, and post-route placements.
