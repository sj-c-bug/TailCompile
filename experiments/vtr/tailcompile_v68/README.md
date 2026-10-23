# v6.8：细粒度分层 GNN 初始化器 + 原生 VPR 全局布局

本版直接训练部署时使用的**生成器**，不再用 v6 候选评分器或原 v3 4×4 checkpoint 代替它。
目标：在独立设计族、Default 布线边界附近降低最小成功通道宽度，或在 CPD/线长受控时减少耗时。
代码/烟雾测试通过不等于这个研究目标已实现；正式结论只来自服务器 `evaluation_report.json`。

## 图和粒度

- 逻辑节点：VPR 在全局布局前实际构建的 APBlock，保留 molecule/RAM group 原子性，不是保存的 packed cluster。
- 逻辑边：BLIF 网络收缩后的 source/sink 连接；特征包括类型、规模、局部入射连接权重 P95、原语需求、固定状态，以及明确标注的拓扑 criticality 代理（不是 STA slack）。局部 P95 是 TailRisk 类辅助统计，不是完整的分区 cut-pressure CVaR Profile。
- 器件叶节点：默认每 **4×4 个物理 tile** 为一块，`TAILCOMPILE_V68_TILE_SIZE` 可改。不是把整个 FPGA 固定分为 4×4 个区域。84×84 网格约为 21×21 个底层区域。
- 器件层次：邻近叶区域逐级 2×2 汇聚，再由粗到细下传；支持非方形、非 2 次幂大小。
- 器件边/共享池：实际 RRG 中的方向、segment 和跨边界 wire-pool footprint。跨多个 cut 的同一 wire pool 只编码一次，再向经过的区域提供上下文，不假装是多份独立容量。
- 匹配与坐标：逻辑 GNN 与层次器件 GNN 共同决定区域偏好；坐标 head 同时依赖逻辑节点、选中区域、二者交互和 channel width。
- 解码：LUT/FF/memory/DSP/IO 五维需求受区域容量**上界**约束，不要求每区域按比例精确填满；分块计算匹配、稀疏 top-k 解码，不构造 AP 节点×物理 site 的全连接矩阵。

区域容量来自架构最大模式，是 AP 需求的上界代理，并不证明 packing 可行；容量匹配失败也不能证明器件不可放置。真正的合法化和布线可行性仍由 VPR 决定。
RRG pool 还不是完整的 pin-access/switch reachability 模型，不能当成详细路由的证明。

## 接入方式：不再跳过全局布局

```text
网表 → prepack/AP 图（不运行 GP） + 可复用器件 RRG
     → 层次 GNN → 区域上界解码 → 连续 AP 初值
     → VPR 原生全局布局 → appack → 详细退火 → route/analysis
```

`TC_GP_INIT` 只替换 B2B 求解器的冷初始化。第 0 轮仍运行原生合法化/时序更新，第 1 轮以后执行原生求解器，日志必须出现：

```text
TAILCOMPILE_V68_GP_INIT_CONSUMED
TAILCOMPILE_V68_NATIVE_SOLVER_CONTINUED iteration=1
```

不使用 `--read_flat_place`；该参数会跳过 GP。旧 `TC_GP_WORKER/REPLACE` 也不允许混入本实验。
GP 钩子关闭时，原生 Default 不改变；安装后用旧/新二进制相同 seed 的真实小网表回放检查。

坐标有意分开：`init.ap.tsv` 是无损原生 AP 连续坐标 `[0,W)×[0,H)`，模型按 W/H 归一化。`init.fplace` 是 `floor(AP)` 的辅助导出，未进行 site 合法化，不用于绕过 GP。不要给原生 AP 坐标盲加/减半个 tile。

## 冻结的设计族及标签

`cohort.json` 声明 **27 个候选设计族：20 train、3 validation、4 final_test**。源文件存在已核查，但尚未证明全部能在所选架构完成综合/实现。不能在看到方法收益后静默替换失败设计。
已有开发设计仍在开发侧；最终测试为 `arm_core/spree/mkPktMerge/LU8PEEng`。参数变体、seed 不能冒充独立设计族。

Default 的独立 calibration seeds 301/302 定位边界，宽度步长默认 4；在边界两侧各 3 步和一个较宽对照采样。
教师 seeds 311/312；真实干预反馈 seeds 321/322；最终 seeds 341–345。四组 seed 不重叠。

预训练为同一条件选择一个原生 GP 教师，不平均不相容/对称的坐标。teacher 只来自 train/validation；最终测试只导出无移动坐标的结构图。
测试边界可由 Default 校准，但测试 GP 坐标、布线标签不能进入训练或候选选择。

### 2026-10-21：教师粗坐标与资源叶区域的接口修复

服务器已发现 `bnn/w168` 的 AP 节点 50154：需求为 1 DSP，但原生 GP 坐标归属的 leaf 437 的 DSP 容量为 0，另有 100 个 leaf 能单独容纳该节点。`post_gp_teacher` 在 full legalizer 前导出，不能把连续粗坐标所属 leaf 误当成物理资源合法分配。

训练现在从冻结原始标签派生“单节点资源代理可容纳”的监督：

- 原区域可容纳的节点保持区域标签；仅冲突的 movable 节点选择距原坐标最近的可容纳 leaf，依次按点到矩形距离、中心距离、leaf id 确定性打破平局。距离使用器件宽/高换算为物理 tile 单位。
- 坐标移到目标 leaf 的数值安全内部，避免边界舍入后又归属旧 leaf。固定节点不移动；原始数据、图、RRG 和容量上限不改变。
- 原始 VPR 坐标继续用于 geometry 线长保护；派生坐标只用于区域/坐标监督。投影不是重新运行 VPR，也不保证聚合 packing 合法；区域总量仍由训练 overflow 罚项和部署容量解码器处理。
- `models/pretrain/supervision_audit.json` 记录每设计/宽度/split 的冲突数量、资源、投影位移分位数、聚合 overflow 前后值与 raw/derived target 哈希。只在首次使用条件时按块派生，不在每个 epoch 重复，也不把教师放入图输入。
- 没有任何可容纳 leaf 时明确失败，不扩大容量、不跳过硬资源标签。NaN/Inf 与容量冲突分别报错。大量远距离投影意味着教师与约束存在明显偏差，不能把派生教师重构误差当成原 VPR 布局重现或布线收益。

无需重跑 CPU `prepare` 或重新编译 VTR。但监督定义改变，旧 `training_signature.json/latest.pt` 不允许直接续训。确认旧 GPU 训练已经退出后，在同一个 cohort 中**只归档旧预训练尝试**，保留全部 CPU 数据：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export TAILCOMPILE_V68_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v68/runs/hier_init_001"

flock -n "$TAILCOMPILE_V68_OUT/.lock" bash -c '
  old="$1/models/pretrain"
  if [[ -d "$old" ]]; then
    archived="${old}_before_resource_fix_$(date +%Y%m%d_%H%M%S)"
    [[ ! -e "$archived" ]] || exit 1
    mv -- "$old" "$archived"
    echo "Archived old training attempt: $archived"
  fi
' bash "$TAILCOMPILE_V68_OUT"

# 上一步成功后再启动；若提示锁被占用，不要另开第二份训练。
mkdir -p logs
log="logs/v68_train_resource_fix_$(date +%Y%m%d_%H%M%S).log"
nohup bash experiments/vtr/tailcompile_v68/run_pipeline.sh train >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
```

这是可恢复的同目录归档，不删除旧模型记录；只需在迁移训练定义时执行一次。新定义下正常中断仍原目录运行 `train`，从新的 `latest.pt` 恢复，不要反复归档。

真实反馈阶段在 **train** 设计上采样 8 个实际生成器初值（其中 1 个无噪声），用同一 warm-start VPR 流程和多 seeds 采集结果。
同时要求不损失成功数、CPD/线长通过预先冻结的 3% guard，才将有收益的初始化回灌到**同一个生成器**做改进蒸馏，而不是另训一个选择器。
新布通样本没有同宽 Default QoR 时，用最近较高宽度的成功训练教师作参考，明确记录这种局限，不伪造失败运行 CPD。
若没有任何合格改进，报告 `selected=0`，不得解释为已学到真实改进。

最终只比较 Default 与无候选挑选的 GNN 温启动，检验新设计族泛化。最小宽度报告是冻结扫描网格的最小观测成功宽度；超时/缺失是未知，不当作不可布通。记录左右删失、非单调和逐 seed 结果。

## 服务器准备：仍以 40f8d197 为准

先在 CPU 实例激活原有 vtr 环境、source **仅设置环境变量**的 `env.sh`（其中不要自动启动旧实验）。不要将旧 epoch/checkpoint 直接迁入新目录。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export TAILCOMPILE_V68_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v68/runs/hier_init_001"
export TAILCOMPILE_V68_TILE_SIZE=4

# 必须填实际正在使用的 CMake build 目录，避免再次编译了一个目录却运行另一个。
export TAILCOMPILE_V68_BUILD_DIR="$VTR_ROOT/build_tailcompile_conda"
bash experiments/vtr/tailcompile_v68/install_vtr.sh
```

安装器只接受服务器固定 commit、原始源码或精确审计过的旧 hook；拒绝不明修改。保留旧 VPR 二进制，再编译和检查真实温启动。
这会形成新的二进制实验 cohort，不会把旧 v6.7 的结果混入本次对比。已有多项目共用 VTR 时，建议先在单独 checkout/build 安装。

## 分阶段执行（CPU/GPU 不混跑）

同一个输出目录必须放在 CPU/GPU 都能访问的共享磁盘。所有长 CPU 阶段支持并行和原地续跑，已完成且哈希匹配的成功/不可布通任务复用。

| 阶段 | 实例 | 命令末尾参数 | 工作 |
|---|---|---|---|
| 准备 | CPU | `prepare` | 综合、无 GP 结构、Default 边界、pack-only RRG、细粒度图、教师数据 |
| 预训练 | GPU | `train` | 直接训练层次生成器 |
| 真反馈 | CPU | `feedback` | 多进程生成训练候选，随后并行 warm-start VPR 标签 |
| 微调 | GPU | `finetune` | 将真实收益回灌同一生成器 |
| 最终验证 | CPU | `evaluate` | 未见设计族 Default/GNN 多 seed 宽度扫描与汇总 |
| 推荐计时 | CPU | `parallel-timing` | 最低共同成功条件的受控并行配对计时 |
| 隔离复核 | CPU | `timing` | 保留原有小规模串行配对计时，独立报告 |

每次只运行当前阶段，完成后切换实例：

```bash
# CPU 实例：默认 16；未单独设置的 PREPARE_JOBS/RR_JOBS 继承 JOBS。
# VPR 每个任务内部始终 1 worker。资源足够时，可只把这一项改成 32。
export TAILCOMPILE_V68_JOBS=16
mkdir -p logs
log="logs/v68_prepare_$(date +%Y%m%d_%H%M%S).log"
nohup bash experiments/vtr/tailcompile_v68/run_pipeline.sh prepare >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"

# 完成 prepare 后，到 GPU 实例：
bash experiments/vtr/tailcompile_v68/run_pipeline.sh train

# 回到 CPU：
bash experiments/vtr/tailcompile_v68/run_pipeline.sh feedback

# GPU：
bash experiments/vtr/tailcompile_v68/run_pipeline.sh finetune

# CPU：
bash experiments/vtr/tailcompile_v68/run_pipeline.sh evaluate
```

训练默认 CUDA、150 epochs，保存 optimizer/RNG/latest checkpoint 和逐 epoch 日志，可原地恢复。`TAILCOMPILE_V68_EPOCHS/HIDDEN/CHUNK_SIZE/CANDIDATES` 只能在新 cohort 开始前决定，不能测试后调参仍声称独立验证。
`prepare` 含较长 CPU 工作，不要放 GPU 实例等待；不需要也不使用虚假 GPU 保活。

以下参数分别作用于不同子阶段（顺序执行，并行预算不相加）：

| 参数 | 控制内容 | 未设置时 |
|---|---|---|
| `TAILCOMPILE_V68_JOBS` | Default 边界扫描、教师/反馈标签、最终 VPR 评测 | 16 |
| `TAILCOMPILE_V68_PREPARE_JOBS` | 综合、pre-GP 结构，以及多进程特征构图 | 继承 JOBS |
| `TAILCOMPILE_V68_RR_JOBS` | 独立 RRG 生成与压缩 | 继承 JOBS，不限为 1 |
| `TAILCOMPILE_V68_GENERATION_JOBS` | feedback/evaluate 候选推理、匹配解码及输出 | 继承 JOBS，默认 16 |

例如只设 `export TAILCOMPILE_V68_JOBS=32`，另外三项没有旧值时就全部使用 32。若以前在 env.sh 或当前 shell 写过 `PREPARE_JOBS=2/RR_JOBS=1` 或其他单独并行度，显式旧值仍优先生效；删除旧设置，或先执行：

```bash
unset TAILCOMPILE_V68_PREPARE_JOBS TAILCOMPILE_V68_RR_JOBS TAILCOMPILE_V68_GENERATION_JOBS
export TAILCOMPILE_V68_JOBS=32
```

需要按内存单独限制 RRG 时可以显式 `export TAILCOMPILE_V68_RR_JOBS=4`，不会降低其他阶段的并行度。RRG 只运行 pack 导出，不运行布局/布线；随后流式解析，不在 Python 内存中保留百万 rr_edges。设备相同宽度的 raw RRG 共用并压缩。
16/32 表示任务上限，并非内存保证：若仍是之前的 80 GiB 限额，16 个各占约 6.5 GiB 的任务就可能超限，32 更不能盲设。根据实际 CPU quota 和任务峰值内存选择，脚本不会暗中把请求的 16 降为 1。日志 `parallelism` 会显示实际预算，任务少时实际并发自然低于上限。

2026-10-21 的调度更新只接受精确审计过的旧 v6.8 代码哈希兼容，冻结的 seed、宽度、VPR、数据/目标参数均不改变，可继续使用原输出目录、输入、构图缓存和成功 VPR 结果，不要求清空重跑。已运行进程不会动态改变并行度；请勿同时启动第二份 pipeline。可选隔离计时 `timing` 仍遵守原来的串行测量协议，不使用这里的吞吐并发来冒充单任务加速。受控并行计时使用独立的 `parallel-timing`，详见下节，不能只修改 `JOBS` 来切换串行计时。

### 2026-10-22：候选生成也按单个候选多进程并行

旧 feedback 在生成全部候选后才并行跑 VPR；`JOBS` 之前不控制该串行生成循环。现在 feedback/evaluate 的生成采用独立 CPU 进程池，每个候选是一个任务，同一设计/宽度的 8 个候选也能同时解码。默认 `GENERATION_JOBS` 继承 `JOBS` 的 16/32，不暗中降为 1；如 RAM 预算需要，可单独限制此项。

- 每 worker 的 PyTorch/OMP/MKL/OpenBLAS 线程数为 1，复用自己的模型和最近图编码缓存。不能把同一个有可变缓存的 Generator 对象直接在线程间共享。
- `infer.py`、`model.py`、checkpoint、候选 seed/noise、输出目录与冻结反馈条件均不变。已有有效 `generation.json` 会核对签名与输出哈希后复用，不重生成、不覆盖；只有尚未提交完整 manifest 的候选重新做该项。不需要归档模型或重跑 train/prepare。
- 父进程只派发缺失候选，生成进度和并发配置独立写入 `feedback/generation_progress.json`（评测对应 evaluation 目录）。状态命令也显示生成进度；候选齐全后才进入原有 VPR 阶段。
- 不改变容量解码算法、候选数量或单候选质量。因此单个 BGM 解码仍可能很长，改善的是整批吞吐；内存带宽、进程大图副本与尾部慢任务会限制加速比。不要把并行墙钟时间混进正式串行 runtime 加速结论。`timing` 保留原串行测量；`parallel-timing` 单独报告受控并发下的耗时。
- CPU 使用 spawn 进程，避免 fork 已初始化的 PyTorch；多进程不用于单张 GPU。每个 worker 的大图副本都要计入 RAM，32 并发并非内存可行性保证。

在原 CPU 实例先停止旧 feedback，等 cohort `.lock` 释放后更新，再原目录续跑。旧串行反馈若仍在匹配/解码且没有子 VPR，停止精确匹配的 `experiment.py feedback` 主 Python 即可；不能用泛化 `pkill python/vpr`，也不要杀交互终端的整个进程组。新并行生成器收到 SIGTERM/KeyboardInterrupt 时清理自己启动的 worker；SIGKILL/机器故障无法保证这一点，恢复前仍需确认旧 worker 已停止。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export TAILCOMPILE_V68_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v68/runs/hier_init_001"
export TAILCOMPILE_V68_JOBS=32
export TAILCOMPILE_V68_GENERATION_JOBS=32

# 旧任务已停、锁已释放后启动；保持原 checkpoint/候选数/tile size。
mkdir -p logs
log="logs/v68_feedback_parallel_$(date +%Y%m%d_%H%M%S).log"
nohup bash experiments/vtr/tailcompile_v68/run_pipeline.sh feedback >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
bash experiments/vtr/tailcompile_v68/run_pipeline.sh status
```

若还有未结束 VPR，应先确认它们与父调度均停止；不要用 `recover` 绕过存活 worker 检查。旧候选生成未写 `started.json`，所以 recover 不会统计这些半成品；不需要因此清空 generation 目录。

## 验收、恢复和计时解释

### 磁盘满与自动回收

详见 [STORAGE_MANAGEMENT.md](STORAGE_MANAGEMENT.md)。新版在 VPR 结束且 trial/metrics 已持久化后，默认清理该次运行的派生 `.net/.net.post_routing/.route` 与详细报告，保留输入、教师 TSV、生成候选、图、模型及紧凑证据。通过 `storage.py inventory/plan/apply` 先盘点/预览再手动清理现有积压；清单和审计统一写 `$TAILCOMPILE_REPO/logs/storage/`，默认不在指定 workspace 之外创建文件。遇到配额错误停止继续派发，但保留已完成结果，不清空原实验目录。`TAILCOMPILE_V68_PRUNE_COMPLETED=0` 可临时关闭自动清理。

```bash
bash experiments/vtr/tailcompile_v68/run_pipeline.sh status
python3 experiments/vtr/tailcompile_v68/test_data.py
python3 experiments/vtr/tailcompile_v68/test_model.py
python3 experiments/vtr/tailcompile_v68/test_supervision.py
python3 experiments/vtr/tailcompile_v68/test_pipeline.py
python3 experiments/vtr/tailcompile_v68/test_concurrency.py
python3 experiments/vtr/tailcompile_v68/test_generation_parallel.py
python3 experiments/vtr/tailcompile_v68/test_storage.py
python3 experiments/vtr/tailcompile_v68/test_lifecycle.py
python3 experiments/vtr/tailcompile_v68/test_hook.py
```

- prepare：`dataset.json` 的 missing_conditions 为空，test teacher 数为 0。
- train/finetune：`training_report.json complete=true`、`weights_changed=true`，记录实际生成器及训练/验证族。
- feedback：`feedback_dataset.json complete=true`，同时检查合格收益目标数量，不能只看是否跑完。
- evaluate：`evaluation_report.json complete=true` 表示记录齐全；`all_route_outcomes_known=true` 才表示没有 timeout/基础设施未知结果。是否有收益还要看逐族宽度变化、CPD 和线长。

超时继续记录并运行其余任务，不伪装为 unroutable，也不清空成功结果。修复原因后显式重试：

```bash
export TAILCOMPILE_V68_RETRY_TIMEOUTS=1   # 只重试超时
# 或 TAILCOMPILE_V68_RETRY_FAILED=1，重试基础设施失败
bash experiments/vtr/tailcompile_v68/run_pipeline.sh prepare # 或原来的 feedback/evaluate
```

异常中断留下 started.json 时，先确认所有旧 VPR 已停止，再运行 `recover`。同主机活 PID/异主机状态未知会拒绝搬移；不会擅自杀用户进程。若旧 CPU 实例已关闭且已确认没有远端进程，可显式设置 `TAILCOMPILE_V68_CONFIRM_REMOTE_STOPPED=1` 后运行 `recover`，只隔离未完成目录，成功任务仍保留。
成功记录的 command/hook/input hash 不变才复用，旧 attempt 原地归档。

运行 `timing` 将与并行吞吐扫描分开。报告分别列 VPR-only、已缓存图的推理/解码/生成成本，以及包含新设计结构提取/逻辑特征的首次使用**分项成本估计**；可复用 RRG 预处理单列，不计入这几个加速比。
这个估计在已有 Python 进程内组合测量值，不含未测的 Python 新进程启动，结构/逻辑准备部分来自准备阶段，并非一次隔离进程的完整 stopwatch。也不等于 RTL-to-bitstream（双方相同综合成本未计）。不得把缓存图的 warm speedup 当成新设计端到端 speedup。
计时批次固定 host、CPU affinity 和 CPU quota；换实例或改配额后，设置新的 `TAILCOMPILE_V68_TIMING_RUN=serial_002`，只重做该计时批次，不重跑准备/训练/宽度扫描，也不与旧机器记录混合。

### 2026-10-24：受控并行计时（推荐）

`parallel-timing` 复用已冻结的模型、预布局和评测结果，不重跑训练或宽度扫描。它与串行计时使用相同的最低共同成功宽度、前三个最终评测 seeds；最多 4 个设计 × 3 seeds = 12 个配对条件，因此把上限设成 16/32 不会凭空增加任务。

- 每个配对占用一个物理核，Default/GNN 在同一个核先后运行；不同配对交替 AB/BA 顺序。避开同一物理核的超线程兄弟核，每个 VPR 内部仍为 1 worker。
- 波次内先同步运行每对的第一条路线，全部结束后再运行第二条路线。记录实际并发、核绑定、主机、CPU quota、内存限制与执行顺序；较慢任务造成的波次尾部降载会保留在记录中，不能宣称完全没有干扰。
- 同一半波的 VPR 全部结束后才清理派生输出，避免边测量兄弟任务、边进行大量回收 I/O；每半波之间仍自动回收，不把整轮原始文件堆到最后。
- 默认请求 16 个配对并发，实际受配对数、可用物理核和 CPU quota 限制。此处会明确记录受控计时的有效并发，不等同于前面 `JOBS` 的吞吐调度。
- 结果位于 `timing_parallel/<run>/parallel_report.json`，不覆盖 `timing_report.json`，不把旧串行结果混入并行加速比。报告口径是“受控共享负载下的 runtime”，有明显收益后可用少量串行条件复核。
- 成本仍分别列出 VPR-only、缓存图推理/解码/预布局生成、新设计首次使用分项估计；可复用 RRG 处理成本单列。并行 wall time 不能被当作单次推理成本摊薄。

先停止旧串行调度并确认它启动的 VPR 已结束，再启动新批次。仅停止 nohup shell 或看到 `.lock` 释放并不足以证明其子进程已退出；不要使用泛化 `pkill python/vpr`。已有串行结果原样保留，但不能作为受控并行波次的缓存结果。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
# 如需 source env.sh，确认其中只有环境变量，没有自动启动旧实验的命令。
export TAILCOMPILE_V68_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v68/runs/hier_init_001"
export TAILCOMPILE_V68_TIMING_JOBS=16
export TAILCOMPILE_V68_PARALLEL_TIMING_RUN=paired_001
# 可选：手工指定此实例允许使用的 CPU 编号；不要照抄其他机器的核号。
# export TAILCOMPILE_V68_TIMING_CPUS=0,2,4,6

# 独立停止工具不获取 pipeline 锁，因此可在旧串行计时仍运行时执行。
# 先只读预览，核对显示的旧 timing 路径与 PID，再执行精确停止。
python3 experiments/vtr/tailcompile_v68/stop_timing.py \
  --out "$TAILCOMPILE_V68_OUT" --mode serial \
  --serial-name "${TAILCOMPILE_V68_TIMING_RUN:-serial_001}"
python3 experiments/vtr/tailcompile_v68/stop_timing.py \
  --out "$TAILCOMPILE_V68_OUT" --mode serial \
  --serial-name "${TAILCOMPILE_V68_TIMING_RUN:-serial_001}" --apply

mkdir -p "$TAILCOMPILE_REPO/logs"
log="$TAILCOMPILE_REPO/logs/v68_parallel_timing_$(date +%Y%m%d_%H%M%S).log"
nohup bash experiments/vtr/tailcompile_v68/run_pipeline.sh parallel-timing \
  >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"

bash experiments/vtr/tailcompile_v68/run_pipeline.sh parallel-status
```

停止工具核对 UID、进程启动标识、工作目录、脚本及输出目录，先停止调度，再停止该计时目录中归属明确的 VPR 进程组；不删除结果，审计写入 `timing_stop/`。若需停止新并行批次，单独使用 `--mode parallel --parallel-name paired_001`，同样先预览、再加 `--apply`。不能用串行停止命令代替检查仍存活的并行 worker。

同批次原命令可断点续跑：完整波次复用；被中断、未完整提交的波次必须整体隔离后重做，不能把不同并发背景的两半拼接。恢复前会检查旧 worker：同主机活进程或远端状态未知时拒绝操作。只有人工确认旧远端实例/任务已经停止后，才可设置 `TAILCOMPILE_V68_CONFIRM_REMOTE_STOPPED=1`。改变主机、CPU quota、核列表或计时并发时使用新批次名，例如 `paired_002`，不要覆盖旧协议。日志和 `parallel-status` 可查看波次/任务进度；最终 `parallel_report.json` 的 `complete=true` 才表示该计时批次完成，是否加速及 CPD/线长是否退化需另看报告。

## 同步轻量证据

不要 git add 整个运行目录。`.gitignore` 已排除图缓存、RRG、教师坐标、checkpoint、原始 VPR 文件。

```bash
git add -- experiments/vtr/tailcompile_v68 .gitignore GNN_defination.md
git status --short
```

服务器完整数据留在共享盘。需要检查 checkpoint 时单独选择 best `generator.pt`，不要把 latest/optimizer/全部候选和 raw RRG 提交进 Git。
