# 3D 已合格设计：学习与三方法开发对照

日期：2026-10-29。新输出：`runs/development_005_qualified_learning`。

## 本轮要回答的问题

**GNN 学到的层分配与层内 XY 初值，能否比原生 VPR 和简单容量约束初始化，更快完成后续布局布线，并保持成功率与质量？**

上一轮 `development_004_capacity` 完成的是输入、器件和 Default 宽度校准。它证明五个设计具备继续实验的条件，还没有比较学习方法。本轮直接进入教师采集、训练、真实 P&R 反馈和三方法对照，不继续围绕临时启发式做调参。

OR1200 的 Default 扫描仍为右删失，单独保留诊断；它不阻塞已合格设计的学习。选择这五个设计只依据已经完成的 Default 资格，发生在查看本轮 GNN 效果之前。开发结果无论正负均保留。

## 固定输入与数据边界

复用 004 的网表、AP 结构、架构、RRG、资格报告与宽度结果；读取前核验来源和哈希。新队列不重新综合、不重新扫描 Default 边界、不生成新 RRG，也不复制大型旧数据。旧 003/004、原 2D 模块与 VTR 后端保持只读。

| 设计 | 家族 | 用途 | 双层 grid | 压力宽度 | 参考宽度 |
| --- | --- | --- | --- | ---: | ---: |
| SHA | sha | 训练 | 16×16 | 96 | 128 |
| SPMV | spmv | 训练 | 64×64 | 96 | 128 |
| MCML | mcml | 训练 | 84×84 | 160 | 224 |
| blob_merge | blob_merge | 活跃验证 | 32×32 | 96 | 128 |
| diffeq1 | diffeq | 活跃验证 | 16×16 | 128 | 160 |
| OR1200 | or1200 | 独立诊断，保留失败记录 | 20×20 | 未确定 | 未确定 |

每个已合格设计仍使用 `H0Z0/H0Z1/H1Z0/H1Z1` 四种架构：层间资源均衡/不均衡 × 垂直通道分散/集中。同一尺寸四架构的供给对照及 Z 容量随宽度不变性已有实际 RRG 审核；不同尺寸之间不宣称总资源或总 Z 容量相同。

五个设计共 40 个“设计×架构×宽度角色”条件。训练集有 24 个条件，验证集有 16 个条件。blob_merge、diffeq1 的教师不进入梯度，实际 VPR 结果用于 checkpoint 选择，因此它们是**活跃验证集，不能称为最终未见测试集**。比较阶段采用新 seeds，但同一家族不会因此变成新家族。

## 方法与归因

三种方法进入相同的原生 3D GP→APPack→详细布局→布线链路；初始化必须被后端消费，并继续执行原生优化。

| 对比 | 直接回答的问题 |
| --- | --- |
| 容量约束初始化 / Default | 简单、合法的层分配与初值本身是否有帮助？ |
| GNN 初始化 / Default | 完整学习初始化是否优于原生流程？ |
| GNN 初始化 / 容量约束初始化 | 学习是否提供超过简单容量分配的额外收益？ |

保持当前 GNN、合法化器、有限密度修正与训练参数，不新增局部搜索或长迭代启发式。本轮的方法差别是初始化来源，不能把后端、器件容量或额外优化步骤一起改变后归因为 GNN。

## 阶段、预算与停止边界

| 阶段 | 操作与固定设置 | 完整 P&R 预算 | 主要输出 |
| --- | --- | ---: | --- |
| 教师采集 | 40 条件×seeds 511/512；固定选择第一个 seed 的完整教师，不按 QoR 挑选 | 80 | `teacher_records.json`、教师与投影审核 |
| 教师预训练 | 仅 3 个训练族；20 epochs、hidden=64、训练 seed=601 | 0 | `models/pretrain/completed.json` |
| 收集反馈 | 24 训练条件×4 候选×seeds 521/522，加同条件 Default 锚点 | 192+48=240 | `feedback.json`，保留成功、失败与负收益 |
| 反馈训练 | 10 epochs；更新真实生成器，保留教师稳定项 | 0 | `models/runtime/completed.json` |
| 验证选模 | 16 验证条件×seeds 531/532×最多 3 checkpoints，加 Default | ≤128 | `selection.json`、真实 VPR 验证报告 |
| 三方法对照 | 5 设计×4 架构×2 宽度×seeds 551–555×3 方法 | 600 | `comparison/report.json` |
| 解释结果 | 固定三个对比，训练族与验证族分开报告 | 0 | `development_report.json` |

预算上限 **1,048 次完整 P&R 任务**；其中比较阶段训练族 360 次、验证族 240 次。预算不含单独的结构导出、图构建和初始化进程。比较阶段对两种初始化方法还会进行冷启动结构导出，其成本计入总时间；这些 `structure-only` 调用不是额外的完整布线实验。生成失败时实际启动的 P&R 次数可能少于任务数，失败仍计入该方法的分母。

教师预训练和反馈训练在服务器 GPU 上运行，本地只做接口与回归检查。阶段之间是明确的 CPU/GPU 交接，不要求 GPU 实例一直等待 CPU 布线。

最多检查三个预登记 checkpoint，按验证集真实成功率、CPD/WL 保护和总成本选模，不根据教师坐标损失选择。沿用已有协议：若所有 checkpoint 都未过保护，但存在可执行的探索候选，允许一次明确标注的开发对照，`selection.status` 必须保留未通过标记；不得改称模型验收成功。基础环境错误、缺失输入和无法选择可执行模型时停止依赖阶段并保留日志，不将其作为普通负收益训练。

## 计时与结论口径

600 次对照在同一受控批次执行，三方法按平衡顺序轮换，配对条件绑定同一物理核，采用相同并行负载。默认请求 8 个并行任务，按物理核、quota 和内存预算降低；**首次启动比较前固定资源，比较中途不切换机器、核配置或预算来拼接计时**。可恢复已完成的完整波次，未完成波次重做且保留原记录。

冷启动总时间包括该方法需要的 AP 导出、图构建、初始化和全部后续 VPR 时间。可复用的 RRG 构建与校验单列，不作为某个方法独占的加速收益；原生 Default 没有初始化工作，因此不会人为补上同样开销。不要与 v6.10 正式计时或其他重任务同时运行本轮受控对照。

结果分为以下几类，不能仅看“程序正常结束”：

1. **成功率**：压力、参考两个宽度均按计划任务计数。每个设计×架构×宽度保留全部五 seeds；初始化失败、超时和不布通均不可删掉。
2. **参考宽度上的开发目标**：每个参考条件要求五个配对 seed 均成功。训练集和验证集分别检查总 runtime 至少降低 5%，每个参考条件的 CPD、WL 几何平均比不超过 1.03，同时两个宽度上的成功率均不下降。三个方法对比独立报告。
3. **压力宽度上的现象**：重点看是否增加成功次数；CPD/WL 与耗时比只在共同成功样本上描述，并报告样本数。双方都失败时不存在可解释的 QoR 比，不能填零或认作改善；失败更早也不等于加速成功。
4. **未知与缺测**：基础环境错误、缺失指标或不完整面板单列，不能自动计为开发目标通过。

本轮可以得出“GNN 在哪些开发条件有/没有额外收益”的结论。只有 GNN 同时优于 Default 和容量约束基线，才能支持学习提供额外价值；若容量约束已足够而 GNN 无额外改善，就如实记录这一负面结论。报告 GP、合法化、详细布局、布线等已有阶段指标，帮助判断变化来源，不据此临时更改本轮方法。

这 **600 次五设计开发对照不替代原来的 480 次四训练族矩阵**，也不测量 Wmin，不证明最终未见族泛化。完整性能验收仍需要补齐原面板、冻结独立五 seed 宽度网格和最终未见族实验。两个固定宽度不能推出 Wmin；当前报告固定标记 `Wmin_measured=false`、`formal_480_completed=false`、`full_acceptance_passed=false`。

## 服务器运行命令

先激活已有 Torch 环境，使用当前 TailCompile 仓库，不重新创建源码副本。以下命令不依赖旧 `env_3d_v7.sh`，不自动激活 Conda，也不修改原 2D 环境。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
entry="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_qualified_learning.sh"

# 第一步，CPU：核验并引用 004 的已合格输入，然后采集教师。
bash "$entry" cpu-teachers --background
```

等待当前组日志出现 `V7_LEARNING_GROUP_COMPLETE`，再启动下一组。不要一次性提交下面所有后台命令，它们存在数据依赖。后续每组在对应 CPU/GPU 实例的同一共享仓库运行：

```bash
# 第二步，GPU：教师预训练；完成后再执行下一条。
bash "$entry" gpu-pretrain --background

# 第三步，CPU：真实候选反馈；完成后再执行下一条。
bash "$entry" cpu-feedback --background

# 第四步，GPU：反馈训练；完成后再执行下一条。
bash "$entry" gpu-finetune --background

# 第五步，CPU：验证选模 → 600 次受控对照 → 汇总。
bash "$entry" cpu-evaluate --background
```

GPU 组默认 `cuda`，CPU 组默认 `cpu`。若之前显式导出过 `TAILCOMPILE_3D_TRAIN_DEVICE`，它会优先使用；可在对应命令前显式设置 `TAILCOMPILE_3D_TRAIN_DEVICE=cuda` 或 `cpu`。选择具体 GPU 用 `CUDA_VISIBLE_DEVICES`。默认 Python 为当前 `python3`，必要时通过 `TAILCOMPILE_3D_PYTHON` 指定环境解释器。

每次后台启动打印 `PID`、`LOG`、`PIDFILE` 和 `OUT`，日志位于仓库 `logs/`。例如使用打印出的日志路径：

```bash
tail -f /打印出的完整LOG路径
```

前台作业或平台持久作业可以把 `--background` 改成 `--foreground`。`nohup` 仅保证终端断开不发送挂断信号，不能延长平台资源到期时间。某阶段中断后用相同命令恢复；已有签名和完整波次支持续跑，不清空旧目录。如果报配置或计时宿主不一致，应保留证据后处理，不能绕过签名检查。

有明确资源余量时，在首次启动比较前设置 `TAILCOMPILE_V7_JOBS`；每任务内存预算用 `TAILCOMPILE_V7_JOB_MIB`，P&R 超时用 `TAILCOMPILE_V7_TIMEOUT`（默认 3600 秒）。超时与协议不能在看到方法胜负后随意更改。

## OR1200 并行诊断

以下命令只读扫描既有 Default 结果和服务器原始日志，不运行新的 P&R，不扩大宽度或修改架构，可以与上述教师/训练阶段并行：

```bash
python3 "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/diagnose_censored_design.py" \
  --source "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_004_capacity" \
  --output "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_005_qualified_learning/or1200_diagnostic_report.json"
```

报告列出逐架构/seed 的宽度结果、正常失败与环境错误、可用日志片段和非单调现象。OR1200 在部分条件已成功；右删失表示尚未找到同一宽度下八个校准条件全成功的参考点，不表示完全不能布通。Z 容量不随 XY 通道宽度增加，不能假定继续加宽一定解决。缺少原始日志时明确标记证据不足，不凭失败次数编造根因。

## 保存与同步

新 cohort 保存自己的协议、来源引用、训练记录、比较计划、指标与失败状态。checkpoint、教师张量、初始化动作、原始 RRG 和网表留在服务器，不进入 Git。现有 VPR 清理逻辑在指标落盘后回收可再生 `.net`、`.net.post_routing`、`.route` 等派生文件；作为原始输入、教师或回放依赖的文件保留。不要为了腾空间删除 003/004 的共享输入和缓存，它们仍被新 cohort 引用。

查看主要结果：`development_005_qualified_learning/development_report.json`，并联读 `selection.json`、`comparison/report.json` 和 `or1200_diagnostic_report.json`。按仓库工作流只同步轻量报告与协议；在任何提交前查看暂存文件清单，避免强制加入整个运行目录。
