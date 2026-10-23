# v6.9：用轻量 GNN 初始化换取真实总耗时收益

目标：允许小幅资源/质量退化，检验 **GNN 初始化 + VPR 原生全局布局、合法化、详细布局、布线** 能否更快完成。不是仅比较神经网络前向时间，也不通过跳过 VPR 全局布局制造加速。

这是新输出目录中的后续实验。复用 v6.8 的细粒度 tile/层次 RRG 图、逻辑图、训练/验证/测试划分和 checkpoint，不改动 v6.8 代码、输入或结果，不重新提取整批 RRG。沿用服务器已审计的 VTR `40f8d19708b0b60e2aa04452deb1cfed68d96b0b` 和 v6.8 GP warm-start hook，无需为本版重新编译 VTR。

## 1. 这次改什么

- **解码变轻**：保留层次 GNN、cluster–region 匹配、区域条件坐标头；用固定次数的软密度价格更新代替逐节点硬容量分配和组合修复。默认 2 次，允许 0–8 次。不退回 4×4 粗图。
- **必要约束仍保留**：单节点资源类型兼容、molecule 原子性、固定节点和坐标范围检查。区域总体容量只作软约束，溢出会记录；最终合法性只能由 VPR 判定，不能把生成成功称为合法放置成功。
- **训练真实生成器**：CPU 测量训练设计上的候选总耗时，选择真正更快且满足 WL/CPD 预算的初始化作为新教师；GPU 微调实际坐标生成器，不是另训练一个评分器。没有通过预算的候选就停止微调，不以空标签训练冒充改进。
- **计时默认并行**：独立物理核、单任务单线程、同核配对、AB/BA 交替、半波次屏障；记录配额、硬件和负载。默认 16，可调 32，实际并发受物理核/cgroup 配额/任务数约束。
- **保留阶段证据**：记录全局布局、完整合法化、详细布局、布线时间及可解析的迭代数。日志没有的字段为 `null`，不猜测；嵌套计时不能直接相加。

## 2. 冻结后的验收口径

每个设计使用同一宽度网格、同一组 **5 个 seeds（371–375）**。先取每个 seed 的最小已观察成功宽度，再计算算术平均：

`mean_Wmin = (Wmin_seed371 + … + Wmin_seed375) / 5`

它是**冻结扫描网格上的平均最小成功宽度**，不是物理最小值，也不是“90% 成功概率宽度”。同时保留各 seed、非单调性和边界截断信息；有 seed 在整个网格都不成功、任务缺失或超时，不丢掉该 seed 后继续求均值。最低扫描宽度已经成功时报告左截断，不声称找到了真实下界。

默认工程预算如下，在 `prepare` 时冻结；可以在首次 `prepare` 前修改，不能看到测试结果后在原目录放宽：

| 指标 | 默认预算 |
|---|---|
| 每设计 5-seed 平均 Wmin | 最多增加 3% |
| 每设计共同成功配对的 CPD 几何平均比 | 最多增加 3% |
| 每设计共同成功配对的 WL 几何平均比 | 最多增加 3% |
| 计入初始化成本的总耗时 | 跨设计等权汇总至少降低 5% |

这里“退化”指 Wmin、WL、CPD 数值增加。报告逐设计和最差配对，不要求每个设计都加速，但所有设计都要通过质量和证据检查。正式相同宽度 runtime 配对不允许把 Default 成功、GNN 失败当成加速；同时报告成功损失/收益和共同成功覆盖率。超时、环境失败是未知结果，不能当作不可布线。

计时宽度仅依据原 Default 校准选择：每设计参考边界附近一个宽度和扫描网格最高宽度。不是事后仅挑 GNN 成功/更快的条件。

### 时间包含什么

主时间从每次独立 worker 进程开始计，包括 Python/Torch 启动、模型/图加载、逻辑 GNN、匹配、软密度解码、坐标与文件生成、完整后续 VPR。每个计时配对都会重新生成初值，不把提前生成的文件假装成零成本。

按约定，单独测得的可复用 RRG GNN 编码时间从主时间扣除，同时保留含该项的 worker 总时间。原始 RRG 解析和图预处理复用已有产物，不计入主时间；因此这是 **prepared-graph 到完成 VPR 的延迟**，不是从 RTL 开始的全部冷启动时间。Default 也计入相同 worker 包装开销。另报纯 VPR 时间，避免口径混淆。

这是受控并发负载下的单任务加速比，不是隔离串行速度；不能和旧串行或扫描时的吞吐计时混合。

## 3. 服务器执行

先激活原来可正常运行 Torch/VPR 的环境。`env.sh` 应只配置环境变量，不自动启动旧实验。所有新输出/临时文件仍位于仓库内。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

export TAILCOMPILE_V68_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v68/runs/hier_init_001"
export TAILCOMPILE_V69_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v69/runs/runtime_001"
export TAILCOMPILE_V69_JOBS=16
export TAILCOMPILE_V69_GENERATION_JOBS=16
export TAILCOMPILE_V69_TIMING_JOBS=16
export TAILCOMPILE_V69_TIMING_RUN=paired_001

# 可选：首次 prepare 前明确预算（省略时就是下列默认值）
export TAILCOMPILE_V69_WIDTH_BUDGET=0.03
export TAILCOMPILE_V69_WIRE_BUDGET=0.03
export TAILCOMPILE_V69_CPD_BUDGET=0.03
export TAILCOMPILE_V69_TIME_GAIN=0.05

python3 -m unittest discover -s experiments/vtr/tailcompile_v69 -p 'test_*.py'
bash experiments/vtr/tailcompile_v69/run_pipeline.sh prepare
```

并发可改 32，但候选生成每进程有一份图和模型，要按实际内存预算设置。无需重做 v6.8 prepare/train/feedback。

为长任务定义统一日志启动函数（每次进入新终端需要重新定义）：

```bash
v69_start() {
  local phase="$1"
  local log="$TAILCOMPILE_REPO/logs/v69_${phase}_$(date +%Y%m%d_%H%M%S).log"
  mkdir -p "$TAILCOMPILE_REPO/logs"
  nohup bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v69/run_pipeline.sh" "$phase" \
    >"$log" 2>&1 </dev/null &
  echo "PID=$! LOG=$log"
}
```

顺序如下，等上一阶段完成再启动下一阶段；不要在同一输出目录并发启动两个 pipeline：

1. **CPU：`v69_start feedback`**。训练设计每个最多两个宽度、默认四个候选、两个新 seeds（351/352），每个候选和 Default 同核配对计时。若有 20 个训练设计且各有两个宽度，最多 640 次 VPR，而不是重新做整批宽度扫描。
2. **GPU：`v69_start finetune`**。默认 50 epochs，仅训练，不运行 VPR；完成后释放 GPU 实例。训练保留原教师的区域/坐标/连通性/软密度损失，并替换、加权有真实 runtime 收益的目标。验证集按教师重建损失选 checkpoint，不能据此声称已经有最终 runtime 收益。
3. **CPU：`v69_start evaluate`**。固定最终 checkpoint，用 5 个新 seeds 比较 Default 和轻量 GNN+VPR，生成平均 Wmin 报告。
4. **CPU：`v69_start timing`**。默认受控并行，不再默认串行。重新生成初值、计入成本，在预先指定宽度做 5-seed 配对计时和 QoR 检查。

最终测试族沿用 v6.8，过去已经检查过其结果；本轮是后续开发验证，不能包装成全新的独立外部测试。

## 4. 进度、断点和失败处理

```bash
bash experiments/vtr/tailcompile_v69/run_pipeline.sh status
```

- `feedback/feedback_dataset.json`：`complete=true`、`all_route_outcomes_known=true`，并检查 `selected` 数量。`selected=[]` 说明当前候选没有通过速度/质量预算，应先分析，而不是继续空反馈训练。
- `models/runtime/runtime_training_report.json`：`complete=true`、训练轮次及 `weights_changed`。
- `evaluation_report.json`：5-seed 宽度及截断/未知结果。
- `timing/paired_001/runtime_report.json`：`complete` 表示运行证据齐全；`evidence_complete` 表示能评价；`passed` 才是收益预算全部通过。重点看 `aggregate` 和 `by_design`。

正常中断后重跑原阶段：已完成扫描任务复用；完整计时波次复用，未完成波次整体归档、整波重测，避免不同负载混算。GPU 从 `latest.pt` 恢复。若有残留 `running.json`，先确认本任务及子进程均停止，再运行 `run_pipeline.sh recover`。不会自动杀死其他实验。

换实例/重启后硬件或 `boot_id` 变化，需要新的 `TAILCOMPILE_V69_TIMING_RUN`，不能拼接两次硬件会话的 runtime。feedback 尚未完成时，新计时名会重测该阶段的配对，而不是混用旧硬件的标签。已完成反馈进入训练后不要改写它。

超时/环境错误默认保留为未知，避免自动无限重跑。修复原因后，可设置 `TAILCOMPILE_V69_RETRY_FAILED=1` 或 `TAILCOMPILE_V69_RETRY_TIMEOUTS=1` 重跑原阶段；计时仍按整个受影响波次重测。不同主机的残留进程状态需要人工确认后才设置 `TAILCOMPILE_V69_CONFIRM_REMOTE_STOPPED=1`。

## 5. 存储与同步

默认自动回收**本版已完成 VPR trial** 的派生 `.net`、post-routing 网表、`.route`、详细报告等大文件；先保存日志、metrics、阶段计时与删除记录，保护 Git 跟踪文件和运行中的文件。不删除 v6.8 原始输入、图、教师、模型或本版初始化文件。删除不进回收站，重分析原始布线需要重跑。若必须保留原始输出，启动前设置 `TAILCOMPILE_V69_PRUNE_COMPLETED=0`。

磁盘自由空间检查和 quota 错误会停止派发，保留已完成证据；`df` 有空间不代表用户 quota 仍有额度。临时目录在本版输出内。

运行后仅同步轻量证据（`.gitignore` 已排除大图、模型、网表和原始布线）：

```bash
cd "$TAILCOMPILE_REPO"
git add -- experiments/vtr/tailcompile_v69/runs/runtime_001
git diff --cached --stat
git commit -m "Record v6.9 runtime-oriented initializer results"
git push origin main
```

本地回归测试覆盖软解码、真实 checkpoint 加载、固定/molecule 约束、均值与预算、未知结果、设计族隔离、阶段解析和完整/中断波次复用；这些是代码正确性证据，**不是新的 VPR 加速结果**。CUDA 和真实大设计闭环仍需在服务器运行。
