# v6.7：ABCD 四路线，分项 runtime

当前 `run_pipeline.sh` 调用 `resume.py`，默认目录 `abcd_001`。
已有服务器实验继续使用 `abcd_002`，执行 `evaluate` 即可复用已有结果。
`four_routes.py`、`refine.py` 和 VPR hook 保持冻结；新入口添加编排与日志，不修改布局动作。
旧的 `direct.py` 双路线入口保留为生成器和历史实验，不要混用旧 `direct_001` 的 VPR 结果。

| 路线 | 初始布局 | 自有局部修正 | 后续流程 |
|---|---|---|---|
| A_default | VPR GP | 无 | appack → annealer → route |
| B_gnn | GNN `.fplace`，原生入口跳过 GP | 无 | 同上 |
| C_default_refine | VPR GP | 有 | 同上 |
| D_gnn_refine | GNN `.fplace`，原生入口跳过 GP | 有 | 同上 |

C/D 在相同的合法化前 hook 执行相同修正：复用 v6.6 完整 AP-block/molecule/RAM-group
交换动作，1%/2%/5% 预算、hotspot/wire 两类、各两次确定性提案，包含 replay。
按归一化 RUDY-tail + HPWL 最小值选择，平分时不移动；沿用 HPWL 增长不超过3%的代理保护。
**这是冻结的无标签启发式选择，不是训练后的 v6.6 selector，不是 VPR oracle，也不保证实际 CPD/拥塞改善。**
修正不读取 route 结果，C/D 同参数，所有搜索候选成本均计入。VPR 的 annealer 仍正常执行。

主报告统一输出 B/A、C/A、D/B、D/C、D/A 配对，以及四组逐 seed Wmin/W80。
局部 worker 在 VPR 进程内同步调用：C 的 GP 成本与 C/D 的导出、启动 worker、搜索、输出和导入
都已在 VPR wall 内，**绝不重复加计 worker_seconds**。B/D 再加下面表格中的 GNN 成本。
每个子进程独立环境变量，避免并行时不同路线互相污染；超时杀死整个 VPR/worker 进程组。
启动时先串行验证第一项 C/D 的真实 hook 标记，再提交批量任务；旧二进制漏装 hook 会立即失败。

这是独立的新对照脚本，不改变 v6.6 已冻结的代码/协议/输出。
优先回答原始问题：**不做局部修改，GNN 生成的预布局交给 VPR，能否改善 Wmin 或耗时？**

## 方法和范围

- Default：VPR analytical placement → appack → annealer → route。
- GNN-only：原始双塔生成器一次前向 → 固定 partition prior → min-cost flow 容量分配 →
  `.fplace` → VPR 原生 `--read_flat_place` → appack → annealer → route。
- B 不进行候选扰动或局部交换；C/D 增加上面明确的修正。四组均不事后按真实 VPR 选最优。
- checkpoint 默认是原有 `model_width_conditioned/ranker.pt`；虽然名字叫 ranker，
  这是输出 cluster→region 分数和相对坐标的原始生成模型。不是 v6.x 只能评分的模型。
- 脚本拒绝 checkpoint 中出现过的训练/验证设计；默认六个 v5 设计都不属于该 checkpoint
  的 sha/attention_layer 训练及 spmv 验证集。但这六个设计曾用于项目开发，**不算全新独立验证**。
- 复用服务器现有 v5 原始 4×4 IR、映射 BLIF、packed net，不误用 tile8 评分器 IR。
  使用同一设计、尺寸、宽度、placement seed 和后续 VPR 参数配对。
- 默认每个设计取旧 Default-only screen 的首个成功宽度 ±30、步长10，七档宽度；
  新 seeds 251–255。每个宽度生成自己的 width-conditioned `.fplace`，每个 seed 复用它。
  Wmin 是同一算法在不同给定宽度下的网格最小成功值，不是固定坐标的纯 route-only Wmin。
- 六个设计共840次主扫描。计时使用共同成功的边界子集，默认最多180个独立任务，
  共享条件去重、已有记录复用后更少；正式 runtime 仍单任务串行测量。
- 输出逐 seed 网格 Wmin、删失/非单调性、W80、共同成功条件下的线长/CPD 和分口径 speedup。

原生 `.fplace` 入口会跳过 GP，并改变 RAM grouping 路径。因此本轮测量的是 GNN→VPR
**整条接入流程**的效果，不能把差异全部归因于坐标质量。v6.6 的 post-GP 回放/修正仍是独立诊断。

## Runtime 口径

每个 design/width 的 `generation.json` 记录实际分项时间。CUDA（可选）前后同步，
避免只测异步调用提交耗时。默认 CPU 推理，VPR 始终 CPU。

| 环节 | 主表 warm_inference_lowering | warm_operational | cold_from_prepared_ir |
|---|---:|---:|---:|
| checkpoint 加载和模型初始化 | 不计，假设驻留 | 不计，假设驻留 | 计 |
| 已有 device IR 加载 | 不计，假设缓存 | 不计，假设缓存 | 计 |
| RRG tower＋region projection | 不计，假设缓存 | 不计，假设缓存 | 计 |
| 当前逻辑 IR 加载、图特征准备 | 不计，假设输入已准备 | 计 | 计 |
| tensor transfer | 计 | 计 | 计 |
| 逻辑 GNN 前向、匹配、坐标 head、prior | 计 | 计 | 计 |
| min-cost flow 容量分配 | 计 | 计 | 计 |
| `.fplace` lowering 和写文件 | 计 | 计 | 计 |
| VPR 全部后续耗时 | 计 | 计 | 计 |

原始 RRG→device IR、原始 RTL→packing、Python 进程启动未在本轮重新测量，记录为 null。
**cold_from_prepared_ir 只是“已有 IR 的缓存未命中”口径，不能称作从原始输入开始的完整冷启动。**
当前 graph_inputs 混合构造逻辑和器件图，无法准确分摊；warm_operational 保守地计入整段，
不会随意扣减其中推测属于 RRG 的部分。

主表遵循用户要求，仅计推理和生成再加 VPR；这是一种服务已有缓存/输入图的条件口径，
不是对首次处理新网表的无条件端到端加速声明。请同时看 warm_operational，防止把设计特有成本隐藏。

缓存键包含 checkpoint hash、device IR hash、channel width（模型有 width conditioning 时）、
计算设备。更换器件尺寸、RRG/容量特征、权重或宽度不能盲目复用 embedding。
默认本轮只做内存内的真实缓存；主表允许服务端缓存已准备好的反事实 warm 口径，
`rrg_cache_hit` 另行记录实际是否命中。没有声称每个首次生成任务都实际命中了缓存。

speedup = Default VPR wall time / (GNN 分口径成本 + GNN 后续 VPR wall time)，大于1为更快。
只统计双方均成功的同宽度、同 seed 条件；快速失败不算加速。
生成成本按每次独立部署加计，不把一次离线生成摊薄到五个评测 seeds。
RRG 一次性成本也不会默默归零；其可测分项单列，未测分项标未知。

## 运行

不需要重新训练，也不需要先跑 v6.6。使用原先可运行 torch 和 VPR 的环境。
VTR 基础 commit 仍为 `40f8d197...`；四组必须使用安装精确 v6.6 hook 后重新编译的同一二进制。
hook 在 A/B 无环境变量时不执行。安装器拒绝覆盖无关源码修改。
每个子进程先清除继承的 TC_GP_*，再为 C/D 设置独立 hook 环境；同一批次冻结二进制 hash。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
python3 experiments/vtr/tailcompile_v66/install_hook.py "$VTR_ROOT"
cmake --build "$VTR_ROOT/build" --target vpr -j 8
python3 experiments/vtr/tailcompile_v67/test_direct.py
python3 experiments/vtr/tailcompile_v67/test_four_routes.py

export TAILCOMPILE_V67_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v67/runs/abcd_001"
export TAILCOMPILE_V67_JOBS=8
export TAILCOMPILE_V67_DEVICE=cpu
mkdir -p logs

nohup bash experiments/vtr/tailcompile_v67/run_pipeline.sh all \
  > "logs/v67_direct_$(date +%Y%m%d_%H%M%S).log" 2>&1 </dev/null &
echo "PID=$!"
```

默认一台 CPU 实例即可完成。GPU 推理另设 DEVICE=cuda，先在 GPU 实例跑 `generate`，
然后到 CPU 实例跑 `evaluate`（不重新生成）。这里不涉及 GPU 训练。
请不要在正式串行 timing 阶段同时运行其他 VPR 重任务。该阶段故意不使用 JOBS。

可选环境变量：`TAILCOMPILE_GENERATOR_CHECKPOINT`、`TAILCOMPILE_V67_INPUTS`、
`TAILCOMPILE_V67_SCREEN`、`TAILCOMPILE_V67_TIMEOUT`。inputs 是现有 holdout_inputs.json。
扩展到其他设计需先提供匹配的 4×4 IR/网表，不会自动把 v6.6 的 20 个设计宣称为已准备完毕。

```bash
bash experiments/vtr/tailcompile_v67/run_pipeline.sh status
```

当前验收：expected_per_phase=840 保留表示原始扫描空间，expected_selected_timing 为真正需要的计时数量。
`timing_selection_progress` 列出 expected、accounted、remaining，不要求计时填满840个条件。
生成 `resume_evaluation_report.json`。complete=true 表示扫描已归档、选中计时已完成且已生成报告；
允许扫描包含 timeout，不表示所有 Wmin 都已确定。
`all_scan_route_outcomes_known` 单独说明扫描中是否仍有未知结果。
完成任务不等于结果有收益；必须查看 W80/Wmin、删失、QoR 和对应计时口径。
完整任务可复用；中断/超时/环境失败保留原始日志并拒绝当成布线失败。不要清空整批有效结果。

## 超时容错和断点重启（适用于已有 abcd_002）

直接重复 `run_pipeline.sh evaluate`。有效的 success/unroutable case 按二进制、架构、网表、
候选、方法代码和 seed/宽度校验后复用，默认不重复运行旧 code=124。
旧 trial.json 不改写；outcome.json 将旧 other_failure/code124 解释为 timeout。
与当前方法不匹配的结果仍会被拒绝，不会靠忽略 hash 混入。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
python3 experiments/vtr/tailcompile_v67/test_resume.py
export TAILCOMPILE_V67_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v67/runs/abcd_002"
export TAILCOMPILE_V67_JOBS=8
mkdir -p logs
log="logs/v67_resume_$(date +%Y%m%d_%H%M%S).log"
nohup bash experiments/vtr/tailcompile_v67/run_pipeline.sh evaluate >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
```

本次升级无需重新编译 VPR，也不需要重新生成 .fplace。
timeout 只终止当前进程组，其他任务和后续阶段继续。精选模式只启动扫描共同成功的条件；
扫描失败/超时和未选中宽度不再重复计时。原有全量计时和跳过记录保留，不伪造计时值。
成功率给出上下界；超时位于首个成功宽度之下时，Wmin 报告为未确定，另给已观察最小成功宽度。
报告列出每个设计/方法的超时数与每项 runtime 比较排除的条件，不能只看共同成功子集的 speedup。

中断后没有 trial.json 的目录会在续跑前移到 `_attempts/` 再启动，原始日志可恢复。
明确重试时同样先归档，绝不清空整批数据。检测到前一进程仍活着时拒绝重复启动；
本入口用 flock 防止同时启动两条 pipeline。重启前确认旧 pipeline 已结束。

可选环境变量（默认均为0）：

- `TAILCOMPILE_V67_RETRY_TIMEOUTS=1`：重跑已超时 case；完成的 case 仍复用。
- `TAILCOMPILE_V67_RETRY_FAILED=1`：重跑其他错误；没有完整结果的中断任务自动归档重试。
- `TAILCOMPILE_V67_TIME_SCAN_TIMEOUTS`：旧编排兼容参数，不扩大当前精选范围。
- `TAILCOMPILE_V67_TIMEOUT`：当前预算，可提高，禁止降低后复用较大预算数据。
- `TAILCOMPILE_V67_HEARTBEAT`：进度日志间隔，默认120秒。

如提高预算，旧成功结果及其原预算保留，重试结果记录新预算；报告属于混合预算探索，
正式等预算比较需查看逐 trial signature，而不能据此声称统一预算收益。
新运行用 `stdbuf` 刷新 VPR 日志，Python worker 逐阶段 flush；不需要更改或重编译 hook。
阶段记录在 `stages.jsonl`、`worker.events.jsonl`，同一 trial 的搜索成本仍计入 VPR wall。
`resume_sessions/*.json` 记录编排/观测代码 hash 与续跑参数。旧成功 case 的旧计时原样保留，
新增 case 的观测开销也包含在时间里；这是复用旧批次的现实限制。

## 先汇总扫描，再做精选计时

当前 `evaluate` 顺序为：复用/补齐扫描 → scan-report → timing-plan → 精选 timing → report。
Wmin 不需要等待任何新的 VPR。`scan_report.json` 给出四路线逐 seed Wmin、W80和五组配对差异，
继续保留超时造成的未知范围。扫描失败与超时也全部参与这份报告。

计时计划 `timing_selection.json` 只读取扫描的布通状态：按原始 seed 顺序固定前三个，
对每个设计/比较/seed 选择最小共同成功宽度。没有共同成功条件时明确列出该 seed，
不会偷偷改用其他 seed，也不看耗时挑选有利结果。同一任务被多个对照使用时去重。
最终 runtime 比较严格限于计划内配对，计划外旧计时保留但不混入本轮平均值。
因此结论仅针对这些共同成功边界条件，不能外推为全部宽度或整个设计集的加速。

服务器若仍在跑旧全量 timing，先更新并停止同一输出目录的控制器，再生成扫描报告与计时计划：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export TAILCOMPILE_V67_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v67/runs/abcd_002"
bash experiments/vtr/tailcompile_v67/run_pipeline.sh stop
# 输出 still_active=[] 后再启动下一阶段；已完成 case 保留，活动 case 会记为 interrupted。
bash experiments/vtr/tailcompile_v67/run_pipeline.sh scan-report
bash experiments/vtr/tailcompile_v67/run_pipeline.sh timing-plan
mkdir -p logs
log="logs/v67_selected_$(date +%Y%m%d_%H%M%S).log"
nohup bash experiments/vtr/tailcompile_v67/run_pipeline.sh evaluate >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
```

`stop` 通过 /proc 只匹配同一输出目录的 resume.py run/timing 控制器，发送 SIGTERM 并最多等待30秒；
不终止其他项目或其他输出目录。活动 case 的旧尝试在下次执行前归档；成功 case 不会被重跑。
`scan-report` 允许在旧计时仍持有锁时只读汇总；`timing-plan` 和运行阶段仍使用锁。

可选 `TAILCOMPILE_V67_TIMING_SEEDS=3`、`TAILCOMPILE_V67_TIMING_WIDTHS=1`。
修改这些参数或更新扫描后，需显式 `TAILCOMPILE_V67_REPLAN_TIMING=1`，原计划将保存到
`timing_selection_history/`；原始扫描/计时文件不删除。计划外 missing 不会阻止精选实验完成。

可先提交 `scan_report.json`、plan.json、abcd_protocol.json 及扫描 trial.json 供 Wmin 验收，
随后再同步精选 runtime；不必等待计时结束才判断布线宽度收益。

## 受控并行计时（4/8/16等并发）

`parallel-plan / parallel-evaluate / parallel-status / parallel-report` 使用同一份精选范围，
不改动 Wmin 扫描。结果独立存放于 `timing_parallel/<run-name>/`，不与串行 timing 混合。
每个 lane 同时最多执行一个 VPR（含其同步 worker），每个任务内部线程数为1。

调度以比较配对为单位：同一 seed/宽度的候选和基线依次使用同一物理核，
按 seed 与比较序号轮换 AB/BA。每批最多 K 组配对，先同步执行各配对的第一项，
等该轮全部完成后同步执行第二项。下一批轮换核分配。两次运行间的等待不计入单任务 runtime。
每组比较独立保留基线测量，因此可能比串行去重任务数更多，但多个配对同时推进。

物理核根据当前进程 sched_getaffinity 与 sysfs package/core ID 选择，避开 SMT 同核线程，
并检查 cgroup CPU 配额。VM/容器只可依据其报告的拓扑，不承诺独占底层宿主机。
可选手动指定 `TAILCOMPILE_V67_TIMING_CPUS="0,2,4,6"`；需恰好 K 个可用、不同物理核的 CPU。
程序在子进程 exec VPR 前绑定；VPR 和局部修正 worker 都记录并校验实际 affinity。

绑定不能消除共享内存、缓存、频率、磁盘及其他租户的干扰。这个指标是明确并发协议下的单任务墙钟时间，
与无干扰串行延迟分别解释。报告记录半批起步偏差、实际峰值/平均活动任务数、CPU/NUMA信息与压力快照。
末尾不足 K 个 lane 的批次单独分层，不混入完整 K 并发的平均值。
需要在专用 CPU 实例运行；不要同时启动另一条扫描或串行计时 pipeline。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export TAILCOMPILE_V67_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v67/runs/abcd_002"
bash experiments/vtr/tailcompile_v67/run_pipeline.sh stop
export TAILCOMPILE_V67_TIMING_JOBS=8
export TAILCOMPILE_V67_PARALLEL_RUN=c8_001
bash experiments/vtr/tailcompile_v67/run_pipeline.sh parallel-plan
# 核对 cpus、waves、trials；配额/物理核不足时会明确拒绝。
mkdir -p logs
log="logs/v67_parallel_c8_$(date +%Y%m%d_%H%M%S).log"
nohup bash experiments/vtr/tailcompile_v67/run_pipeline.sh parallel-evaluate >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
```

默认 K=4；可改为8、16或其他正整数，每个值应使用新的 run-name。
`parallel-plan` 记录内存上限；并发增大时还需实例有足够内存容纳 K 个 VPR。
同名运行冻结并发、核列表、实例/boot、二进制、代码和输入；改变配置应使用新名字，
避免把不同机器或不同负载实验混在一起。切换到并行模式不需要重新生成 .fplace 或重编译。

```bash
bash experiments/vtr/tailcompile_v67/run_pipeline.sh parallel-status
```

输出 remaining_waves；重启同一命令会复用完整批次。中断的批次整体移到该运行的 `_attempts/`
并重跑，保留已完成单项的证据，但不把不同背景负载下的半批拼成公平配对。
旧串行数据完整保留，不能当作新并发测量直接复用。停止命令同时支持串行和并行控制器。

完成后生成 `timing_parallel/c8_001/parallel_report.json`，包含并发条件下的 VPR 加速比、
按原分项生成成本估算的总加速比、QoR、排除条件、吞吐量和环境记录。
GNN 的生成分项复用已有 generation.json 测量，没有在并发 VPR 负载下重测；
报告单独说明这一口径，不把它解释成全链路都在相同负载下重新测量。
有收益后再挑少量关键配对串行复核，不需要等待整个串行范围。

并行结果同步可先提交 protocol.json、schedule.json、parallel_report.json 和各批 wave.json；
另保留 trial.json、launch_affinity.json、worker.events.jsonl 以核验绑定与断点。
原始 VPR 大文件仍留在服务器。

## Git 同步

```bash
git add -- experiments/vtr/tailcompile_v67/runs/abcd_002/signature.json \
  experiments/vtr/tailcompile_v67/runs/abcd_002/plan.json \
  experiments/vtr/tailcompile_v67/runs/abcd_002/abcd_protocol.json \
  experiments/vtr/tailcompile_v67/runs/abcd_002/scan_report.json \
  experiments/vtr/tailcompile_v67/runs/abcd_002/timing_selection.json \
  experiments/vtr/tailcompile_v67/runs/abcd_002/resume_evaluation_report.json \
  ':(glob)experiments/vtr/tailcompile_v67/runs/abcd_002/**/generation.json' \
  ':(glob)experiments/vtr/tailcompile_v67/runs/abcd_002/**/trial.json' \
  ':(glob)experiments/vtr/tailcompile_v67/runs/abcd_002/**/refined.selection.json' \
  ':(glob)experiments/vtr/tailcompile_v67/runs/abcd_002/**/outcome.json' \
  ':(glob)experiments/vtr/tailcompile_v67/runs/abcd_002/**/stages.jsonl' \
  ':(glob)experiments/vtr/tailcompile_v67/runs/abcd_002/**/worker.events.jsonl' \
  ':(glob)experiments/vtr/tailcompile_v67/runs/abcd_002/resume_sessions/*.json'
git commit -m "Add ABCD placement and decomposed runtime results"
git push origin main
```
