# 007 分层实验：本地结果与服务器执行

日期：2026-11-01。对应 [分层实验设计](../../vtr3d/LAYER_INITIALIZATION_PLAN_2026-10-31.md)。新实现独立于冻结的 2D 和 003–006。

**服务器完成后的更新：**007已完成460次P&R及完整224比较，结果上传提交`8b5d4e319`。T-S参考总时间相对Default+13.70%，没有平均加速；相对L-S的时间/CPD有1.29%/2.04%增量改善。下一步需要对齐教师、软风险和硬容量解码，见 [结果评估与证据](runs/development_007_layer/REVIEW_2026-11-01.md)。下方缺数据、待执行及命令说明是启动时记录，不代表当前实验仍未完成。

## 已完成及当前结论

- 显式层 GNN、冻结的条件 leaf/XY、确定性分层、容量检查及真实风险接口已实现。
- 垂直资源按独立 RR 成员计数；接入使用真实定向 RR 边的区域收缩上界。它不是具体 pin/site 的详细布线证明。
- Tail 尺度只从 train 教师的正 CVaR 中位数确定，并写入 checkpoint；L/T 共用。容量、cut、XY 项保持固定物理归一化。真实反馈按同条件后端时间排序，没有旧的逐例 3% 过滤或统一 -1 截断。
- 新后端完成隔离编译，原 VTR 哈希保持不变。12/12 小 P&R 通过；其中异构双层的 4 次实际消费了新 GNN 训练产生的初值。单层 I/S 输出一致，双层 prior 按 1、0.5、0 衰减后继续原生优化，固定约束通过。
- 真实 tiny 图为 141 APBlock、18 leaf。完成 20 epoch 共享预训练和 L/T 各 5 epoch 准备；本地没有同条件不同动作的实测反馈，明确跳过反馈微调。
- 风险计算合成基准：10 万节点/10 万网/32 池，单线程首次约 2.93 s，缓存计算与反向约 0.030–0.038 s。这只是计算规模证据。
- 已回收约 103.2 MB 原始 RRG 和约 7.5 MB P&R 中间文件；大型模型、图和 backend 不进 Git。

**这些是实现与正确性结果。自然设计的 160 次反馈、80 次选模、224 次开发比较尚未执行。** 当前本地缺少服务器上的教师/图/自然设计原始输入，缺项见 `runs/development_007_layer_local/server_payload_report.json`。本轮没有远程服务器执行通道。

小例不能支持加速结论：异构 fixture 的早期垂直超用有所下降，但 S 的 wall 约 3.71/3.94 s，I 约 2.59/2.54 s；完整值保留在资格报告。后端的 `last.tsv` 指最后完成的路由迭代，可能早于恢复 best routing，不能将它误读为最终合法路由占用。

## 服务器一条命令执行完整批次

在已有 Torch 的环境中执行；脚本继承当前环境，不自动 `conda activate`。沿用已设置的 `TAILCOMPILE_REPO`。`LD_LIBRARY_PATH` 沿用已验证的 VPR 配置。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_3D_TRAIN_DEVICE=cuda
export TAILCOMPILE_V7_JOBS=4

bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_layer_campaign.sh" \
  all --background
```

会打印 `PID`、`LOG`、`OUT`。用打印的日志路径查看进度：

```bash
tail -f /实际打印的LOG路径
```

默认结果：`$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_007_layer/`。默认新后端：`$TAILCOMPILE_REPO/.deps/tailcompile_v7_3d_layer/vtr/`。

顺序为：构建独立后端并做其资格 → 一次列全缺失输入 → 新图/垂直接入 → 共享预训练和 L/T 准备 → 共同反馈采集 → 配对反馈微调 → VPR 选共同 epoch → 224 次七臂比较 → `layer_report.json`。首个失败会停止该组后续阶段，日志保留具体底层错误。

构建器从 004 的 `binary_identity.json` 定位已验证的旧 3D backend，只读复用它的构建依赖。支持现有 normal CMake/Make 后端及本地 exact-object 后端；Ninja 仅支持明确可解析的纯命令。新依赖、日志、缓存和结果均写在主仓库内部。

如果旧 3D backend 被移动，可在运行前显式设置：

```bash
export TAILCOMPILE_V7_LAYER_BASE=/原有合格3D后端根目录
```

它应是含 `backend_manifest.json` 的根目录。不能指向 2D 安装。若新构建中途失败，保留失败目录；入口拒绝覆盖未知的半成品，应先根据底层日志修复，再指定新的 `TAILCOMPILE_V7_LAYER_BACKEND` 路径。

## 分开使用 CPU/GPU 实例

同一套源码和持久化输入可逐阶段执行。每一步完成后再启动下一步：

```bash
script="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_layer_campaign.sh"
bash "$script" bootstrap --background
bash "$script" prepare --background
bash "$script" pretrain --background
bash "$script" feedback --background
bash "$script" train --background
bash "$script" evaluate --background
```

上面的多行是执行次序，**不要一次全部粘贴启动**。`pretrain/train` 使用 `TAILCOMPILE_3D_TRAIN_DEVICE=cuda`；其他阶段设置 `cpu`。逐阶段协议分别冻结资源设置，允许在 GPU 训练与 CPU 执行之间切换；正式计时批次固定 host、boot、cores、并行 lanes。

`plan` 只打印预登记预算；`check` 只检查 payload。直接重跑相同阶段会复用完全匹配的结果；半个计时波会在保留旧代价后最多整体重试一次，不将缓存方法和新计时方法混在不同负载中。旧 cohort 不参与新时间对照。

## 预算、数据与指标

- 服务器批次：12 次新 binary 资格＋最多 160 反馈＋80 选模＋224 比较＝最多 476 次 P&R。
- 本地工程资格另有已完成 12 次。因此跨两台机器最多 488 次；GPU 训练、结构/RRG 导出及已证实基础设施故障重试另记。这一执行说明补清了原计划未单列的服务器重新编译资格成本，研究矩阵没有扩大。
- SPMV/MCML 是 train 开发族，blob_merge/diffeq 是已暴露 active validation；本批不宣称未见族泛化。
- 主比较 T-S/B，参考宽度的冷启动总 runtime 为主指标。族等权平均收益大于零即记录平均正收益；压力宽度、CPD、WL、成功率及失败另列。失败早退不算加速，缺测不填零。
- 共同成功子集的统计明确标为条件结果；缺失或失败覆盖不能被隐藏成全队列收益。两宽度不产生 Wmin 结论。
- 训练、候选和选模预算固定，无排序对时记录跳过反馈训练，使用准备后的权重；不无限补候选。
- 器件 RRG 的首次提取/校验成本单列并保留；正式单次计时不反复 hash 多 GB RRG。计时包含每次逻辑构图、加载模型、生成/解码、文件及完整 VPR。

GP、APPack 合法化后及最终 atom 层分布按实际导出汇总，保留覆盖和缺失状态。诊断导出开销保留在当前有观测的执行时间中，各臂采用相同输出设置。最终路由 RR 使用与最后一轮路由迭代快照分开解释。

每次指标、层迁移与路由摘要持久化后，清理该 attempt 的 `.net/.net.post_routing/.route` 和重复图/诊断导出。训练输入、反馈动作、权重及父 RRG 缓存按依赖保留。同步结果时只添加新 cohort 中 Git 已允许的轻量证据，勿 force-add 图或权重。

## 结果入口

- 本地：`runs/development_007_layer_local/qualification_report.json`、`model_smoke_report.json`、`risk_performance_report.json`、`targeted_unit_tests.log`。
- 服务器：`runs/development_007_layer/layer_report.json` 为完整七臂结论；`selection.json` 为共同 epoch，`comparison/report.json` 为全部任务。未生成完整报告前，不称自然设计实验已完成。
