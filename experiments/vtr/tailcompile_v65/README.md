# TailCompile v6.5: guided placement and runtime audit (2D)

The v6.4 frozen placements did **not** reduce grid minimum routing width.
Runtime is a separate engineering endpoint, but a fast routing failure is not
a benefit. The v6.4 paired, jointly successful VPR wall-time ratios were:

| Design | Candidate / Default | Candidate / GNN base | Joint successes |
|---|---:|---:|---:|
| conv_layer | 0.959 | 1.005 | 42 |
| lenet | 1.220 | 1.011 | 43 |

These times exclude generating `.fplace` and were collected under parallel
load. Their log-level placement/route times and routing iteration counts must
be computed on the server, where raw `vpr.out` files remain. The v6.4 compact
report can be created there without rerunning VPR:

```bash
python3 experiments/vtr/analyze_tailcompile_runtime.py \
  --root experiments/vtr/tailcompile_v64/runs/wmin_001 --require-logs
```

`runtime_report.json` distinguishes **joint successes** from both-unroutable
cases. It records paired VPR wall time, analytical placement time, route time,
placement annealer temperatures/swaps, and routing iterations. A shorter wall
time alone does not establish fewer iterations. The current report is VPR-only:
it does not include packing, candidate generation, or file transfer.

## Candidate-generation correction

Previous v6.3 actions were random cluster swaps. Here the existing GNN base
placement is the starting point. The deterministic v4 risk evaluator proposes
hotspot-guided, resource-legal changes, using directional/segment-aware RRG
capacity. Four 512-evaluation restarts are made per boundary condition, capped
at 5% moved non-IO clusters. A candidate must reduce the **predeclared proxy
objective** and keep wire and timing proxy ratios at or below 1.03. This is
only a proposal filter, not a claim that VPR QoR will improve. If no candidate
passes, generation stops rather than quietly benchmarking a no-op.

The two conditions are `conv_layer/w120` and `lenet/w180`, using the existing
v6.1 bundle and packed nets. This CPU-only search requires PyTorch for loading
the saved bundle, but no GPU or retraining. Candidate generation time is saved
in `generation/selection_report.json`. It must be accounted for separately
when claiming an end-to-end speedup; reuse over multiple VPR seeds or widths
amortizes it, whereas one-shot deployment does not.

```bash
cd "$TAILCOMPILE_REPO"
export TAILCOMPILE_V65_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v65/runs/guided_001"
export TAILCOMPILE_V65_JOBS=8

bash experiments/vtr/tailcompile_v65/run_pipeline.sh generate
bash experiments/vtr/tailcompile_v65/run_pipeline.sh plan
bash experiments/vtr/tailcompile_v65/run_pipeline.sh status
```

Inspect the generated selection report and frozen sweep plan before the
expensive phase. The plan uses new seeds 101–110 and exhaustive width grids
80–160 / 140–220, so it can test width and runtime together. For background
execution on a CPU instance:

```bash
mkdir -p "$TAILCOMPILE_REPO/logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
log="$TAILCOMPILE_REPO/logs/v65_${timestamp}.log"
nohup bash experiments/vtr/tailcompile_v65/run_pipeline.sh run \
  >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
```

Once `status` reports 540/540 completed, run:

```bash
bash experiments/vtr/tailcompile_v65/run_pipeline.sh runtime
```

Primary success criterion is an improvement in paired grid `w_min` or W80
without unacceptable QoR loss. Runtime is an **additional** criterion: compare
joint successes at identical design/width/seed, report both VPR wall time and
placement/route stage time, and check actual iteration counters. A result
faster only than Default but not the GNN base is an ordinary prebinding gain,
not proof that the new hotspot search helped. Because VPR jobs run in
parallel, a small wall-time difference should be replicated under controlled
load before making a strong speed claim.

Only sync compact `selection_report.json`, `plan.json`, `wmin_report.json`,
`runtime_report.json`, and `manifest.json`/`trial.json` evidence. Keep raw VPR
artifacts and logs on the server; do not add them to Git.
