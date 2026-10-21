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
| 真反馈 | CPU | `feedback` | 生成训练候选，并行 warm-start VPR 标签 |
| 微调 | GPU | `finetune` | 将真实收益回灌同一生成器 |
| 最终验证 | CPU | `evaluate` | 未见设计族 Default/GNN 多 seed 宽度扫描与汇总 |
| 可选计时 | CPU | `timing` | 最低共同成功条件的小规模串行配对计时 |

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

三个参数分别作用于以下阶段（顺序执行，不是相加成 48 个同时运行）：

| 参数 | 控制内容 | 未设置时 |
|---|---|---|
| `TAILCOMPILE_V68_JOBS` | Default 边界扫描、教师/反馈标签、最终 VPR 评测 | 16 |
| `TAILCOMPILE_V68_PREPARE_JOBS` | 综合、pre-GP 结构，以及多进程特征构图 | 继承 JOBS |
| `TAILCOMPILE_V68_RR_JOBS` | 独立 RRG 生成与压缩 | 继承 JOBS，不限为 1 |

例如只设 `export TAILCOMPILE_V68_JOBS=32`，另外两项没有旧值时就全部使用 32。若以前在 env.sh 或当前 shell 写过 `PREPARE_JOBS=2/RR_JOBS=1`，显式旧值仍优先生效；删除这两行，或先执行：

```bash
unset TAILCOMPILE_V68_PREPARE_JOBS TAILCOMPILE_V68_RR_JOBS
export TAILCOMPILE_V68_JOBS=32
```

需要按内存单独限制 RRG 时可以显式 `export TAILCOMPILE_V68_RR_JOBS=4`，不会降低其他阶段的并行度。RRG 只运行 pack 导出，不运行布局/布线；随后流式解析，不在 Python 内存中保留百万 rr_edges。设备相同宽度的 raw RRG 共用并压缩。
16/32 表示任务上限，并非内存保证：若仍是之前的 80 GiB 限额，16 个各占约 6.5 GiB 的任务就可能超限，32 更不能盲设。根据实际 CPU quota 和任务峰值内存选择，脚本不会暗中把请求的 16 降为 1。日志 `parallelism` 会显示实际预算，任务少时实际并发自然低于上限。

2026-10-21 的调度更新只接受精确审计过的旧 v6.8 代码哈希兼容，冻结的 seed、宽度、VPR、数据/目标参数均不改变，可继续使用原输出目录、输入、构图缓存和成功 VPR 结果，不要求清空重跑。已运行进程不会动态改变并行度；请勿同时启动第二份 pipeline。可选隔离计时 `timing` 仍遵守原来的串行测量协议，不使用这里的吞吐并发来冒充单任务加速。

## 验收、恢复和计时解释

```bash
bash experiments/vtr/tailcompile_v68/run_pipeline.sh status
python3 experiments/vtr/tailcompile_v68/test_data.py
python3 experiments/vtr/tailcompile_v68/test_model.py
python3 experiments/vtr/tailcompile_v68/test_supervision.py
python3 experiments/vtr/tailcompile_v68/test_pipeline.py
python3 experiments/vtr/tailcompile_v68/test_concurrency.py
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

## 同步轻量证据

不要 git add 整个运行目录。`.gitignore` 已排除图缓存、RRG、教师坐标、checkpoint、原始 VPR 文件。

```bash
git add -- experiments/vtr/tailcompile_v68 .gitignore GNN_defination.md
git status --short
```

服务器完整数据留在共享盘。需要检查 checkpoint 时单独选择 best `generator.pt`，不要把 latest/optimizer/全部候选和 raw RRG 提交进 Git。
