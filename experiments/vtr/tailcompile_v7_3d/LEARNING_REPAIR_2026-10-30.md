# 006：修复构图与反馈信号，做一次可解释的短训练复测

## 本次只回答两个问题

1. 保持图内容、GNN、解码器和 VPR 不变，能否显著降低构图开销？
2. 保留候选之间的真实差异后，反馈训练是否比旧截断奖励更有效？

005 已给出明确负面结论：完整冷启动 runtime 不达标，64 个有效动作全部得到 -1，训练排序对为零。006 不重写这一结论，不重新校准宽度、不换设计、不重新采教师、不调退火器、不增加 GNN 容量或层分配启发式。原 2D、005 冻结核心和后端不变。

## 已完成的本地验证

证据：`runs/learning_repair_20261030/local_performance_report.json` 和同目录测试日志。

| 合成 AP 图节点数 | 原构图中位数 | 新构图中位数 | 比值 |
|---:|---:|---:|---:|
| 20,000 | 1.372 s | 0.260 s | 5.27 倍加速 |
| 125,796 | 7.981 s | 1.813 s | 4.40 倍加速 |

单线程 CPU，每种规模交替顺序测量三次，图张量、dtype、特征和元数据完全一致。这里是**合成图的 make_graph 计算时间**，不是 MCML 实测，也不含解析、进程启动、磁盘 I/O 或 VPR。不能直接换算成整体加速。

额外验证：浮点容量阈值、固定/部分固定 XYZ、异构资源兼容、不可放置节点、全局网跳过策略；真实独立构图进程及缓存续跑；真实模型三 epoch 的两奖励对照。合成的 150/200/400 秒候选在 Default=100 秒时，旧奖励全部 -1，新策略保留 3 个排序对；守卫测试保证时序违规或成功率下降不会被速度优势掩盖。

## 实现边界

### 构图

`fast_graph.py` 保持 `data.py` 的输入和图输出语义：对不同 demand 向量批量计算逐层容量兼容性，替代 APBlock×layer 的重复 Torch 调用；用有上限的分块避免构造巨大中间矩阵。逻辑边和 TailRisk 特征定义完全不变。

每次构图由独立 Python 子进程执行，继承单线程设置，正式比较时绑定该方法所在物理核。控制线程不再承担 Python 构图计算。解析、device 读取、图计算、内容哈希、序列化分别记时；完整冷计时包含子进程启动和非复用的准备成本。只把所有方法共同可复用的 RRG 处理列在计时外。

服务器先对 40 个旧条件核对新图与 005 `.pt` 内容哈希，再在五设计的 H0Z0/reference 条件上交替测试原/新构图各两次，分别记录串行及最多 4 进程并发。两引擎均放在独立进程，**该测试隔离计算实现差异，不声称单独测得旧线程争用占比**。并发最后不足一波时记录实际负载，不与满波混合求加速比。

### 反馈奖励

`reward_v2.py` 保留每个配对 seed 的真实状态、总时间 log 比值、CPD/WL 比值。双失败不填造耗时；基础设施错误或缺项不作为训练负例。重复 seed 明确拒绝。

排序优先级冻结为：成功率保护 → CPD/WL 保护 → 成功数 → 违规程度 → 同一成功 seed 集合上的运行时间。时序违规之间可以学习“哪一个违规更轻”，但仍记为违规，signed reward 仍为负；这不等于验收通过。安全候选的 signed reward 使用平滑 tanh，**排序使用独立的原始 log 比值**，因此不依赖 tanh 在极端数值上是否接近 -1。runtime 排序设 2% deadband，不把很小的时间差强行当作优劣。没有有效排序对时停止训练并保留诊断。

### 可归因的训练与比较

两训练臂从 **005 相同的 pretrain 末轮 checkpoint** 出发，使用相同的 24 个 train 条件、教师、候选反馈、seed=601、优化器及学习率，各训练 3 epoch：

| 训练臂 | 唯一差别 |
|---|---|
| clipped | 保持旧 `train.py` 的截断奖励和损失 |
| guarded | 新约束排序奖励与损失 |

每臂保留 epoch 1/3，在 validation 设计的 reference 条件上用同一组真实 VPR 运行选择 checkpoint。teacher loss 不选模。均未通过保护时允许保留明确标注的 exploratory checkpoint，用于诊断比较；报告不能宣称验收通过。

最终五方法同时比较：Default、capacity-only、005 冻结模型＋新构图、clipped 三 epoch、guarded 三 epoch。后四组均用同样的新构图；后端、宽度、架构和解码器一致。

- guarded 对 clipped：回答奖励修改是否有用。
- guarded 对 005 冻结模型：回答此次短训练是否改善已有模型。
- guarded 对 capacity-only：回答学习是否超过确定性初始化。
- guarded 对 Default：回答加入全部准备开销后是否仍有实际收益。
- 原/新构图独立 profiling：回答构图优化是否有效。005 历史总时间只作背景，不能与本轮不同负载下的时间直接作为因果对照。

## 冻结运行量与统计边界

| 阶段 | 工作量 | 目的 |
|---|---:|---|
| prepare | 40 条件内容等价＋40 次图 profiling，无 P&R | 输入与性能资格 |
| feedback | 3 train 设计×4 架构×2 宽度×2 seeds×(4 候选＋Default)=240 P&R | 用新构图成本重新采反馈，不从旧耗时中减掉开销伪造标签 |
| train | 两臂各 3 epoch | 检查排序信号及模型更新 |
| validate | 16 Default＋两臂×两 checkpoint×16=80 P&R | 原验证族、reference 宽度，seeds 541/542 |
| compare | 5 设计×4 架构×reference 宽度×2 seeds×5 方法=200 P&R | 冷启动配对比较，seeds 561/562 |

合计 520 次 P&R，小于 005 原流程上限。CPU 工作主要是必要的新代价标签和比较；训练本身仅 6 epoch，不重新进行 20 epoch 预训练。

反馈/选模阶段使用实测初始化进程＋VPR、记录的冷图构建和 AP 导出准备代价组成开发总时间；其中 AP 导出代价沿用合格输入记录。它们用于训练和筛选，**不作为最终受控计时证据**。最终 compare 每个方法重新从 AP 导出开始计时，真实支付冷准备成本；没有事后扣除构图代价。

最终比较同核、同负载整波运行；五方法顺序轮转并反向，40 个条件×seed 单元中各方法每个位置恰好出现 8 次。完整波才发布、恢复时直接复用；中断波在新 attempt 里完整重跑，不混合半波计时。host/boot/core 变化拒绝混入，需要新输出 cohort。不要与其他大任务同时争用测量 CPU/内存。

所有失败保留分母。train/active-validation 分开报告，reference 配对成功时计算 runtime/CPD/WL 几何均值；每个设计×架构的成功率不得下降且 CPD/WL 不得超过 1.03，才检查 runtime≤0.95。任一未知/基础设施异常禁用改善判定。006 只有两 seed、reference 宽度，没有 stress 全面验证或 Wmin，**不构成论文最终验收**。

## 服务器命令

在原有可用的 PyTorch/VTR 环境中，不需要改 VTR、重新 conda init 或传 ZIP。004/005 的原始图、教师和 checkpoint 需保留在服务器；Git 轻结果不能替代这些输入。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

# 默认4个CPU任务；首次复测保持该设置。当前环境需要可用CUDA供训练。
export TAILCOMPILE_V7_JOBS=4
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_learning_repair.sh" all --background
```

入口输出 PID、LOG、OUT。查看打印出的日志路径：

```bash
tail -f /替换为启动输出中的LOG路径
```

`all` 顺序执行 prepare→feedback→train→validate→compare→report；失败即停，不会带着不完整反馈开始训练。默认新目录 `experiments/vtr/tailcompile_v7_3d/runs/development_006_learning_repair`，所有写入都在仓库内。不修改用户当前 shell 的 2D 环境；不会自动激活 conda。

若 CPU/GPU 分时使用，也可逐个运行下列组，必须等前一组完成：

```bash
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_learning_repair.sh" cpu-prepare --background
# 完成后
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_learning_repair.sh" cpu-feedback --background
# 完成后
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_learning_repair.sh" gpu-train --background
# 完成后
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_learning_repair.sh" cpu-evaluate --background
```

同一 cohort 重跑同一入口可续跑，不需要删除目录。修改代码或测量主机后使用新的 `TAILCOMPILE_V7_REPAIR_OUT`；不能把005作为新输出。代码哈希改变会拒绝旧 cohort 继续写入。

## 输出与存储

重点检查：`graph_equivalence_report.json`、`graph_profile_report.json`、`feedback_signal_report.json`、`models/{clipped,guarded}/attempt_*/training_report.json`、`validation/{clipped,guarded}/selection.json`、`comparison/report.json`、`repair_report.json`。

VPR 每次完成后沿用现有 `.net/.net.post_routing/.route` 派生文件清理规则。profile 报告和完整计时波持久保存后，自动删除对应的重复图张量、logic JSON、临时 AP 结构；先保存哈希与字节数清理账本。训练图、教师、动作、checkpoint、唯一输入、004/005 全部保留。原始图/模型等继续不进 Git。

完成后同步轻结果：

```bash
cd "$TAILCOMPILE_REPO"
git add -- experiments/vtr/tailcompile_v7_3d/runs/development_006_learning_repair
git diff --cached --stat
git commit -m "Record 006 graph and reward repair retest"
git push origin main
```

本轮额外报告/协议已通过精确 `.gitignore` 规则开放；无需强制加入 feedback、图或 checkpoint。`repair_protocol.json` 包含输入哈希；训练报告和选模报告足以复核两臂，完整比较行在 `comparison/report.json`。
