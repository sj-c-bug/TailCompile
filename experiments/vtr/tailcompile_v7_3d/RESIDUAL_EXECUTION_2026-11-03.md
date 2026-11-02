# 011：从 D 出发，用真实 QoR 学习有限的布局修改

## 目标与边界

本轮检验：保留 010 D 的 GNN 与硬容量解码后，能否学习小幅修改，改善最终 routed WL 和 CPD。010 的新增训练 N 不作为本轮起点；固定使用 010 `prepared_report.json` 记录的 D checkpoint 及其 SHA。

学习对象是**有限候选的选择器**：冻结基础 GNN，用其网表／器件表示和候选修改特征训练一个小评分网络，选择实际可输出的动作。它没有重新训练整套坐标生成网络，也不声称可以生成候选集之外的任意布局。候选的连接启发式只负责提出修改；实际 VPR 结果负责判断好坏。候选无收益时，不把评分网络训练成功当作方法有效。

原 2D、007–010 源码、结果和模型保持冻结。复用已有器件、RRG 和合格 3D 后端，继续原生 GP → APPack → 详细退火 → route；`TC_GP_LAYER_PRIOR=0`。本轮不改 GP，不使用 TailRisk，不提前执行完整 packing。

## 五组对照与可执行动作

| 方法 | 初始化 | 对照目的 |
|---|---|---|
| B | 原生 Default | 判断完整方法相对原生后端的价值 |
| D | 冻结的 010 D | 固定学习起点 |
| L | 从 D 选择少量改层动作 | 层分配修改的贡献 |
| Y | 从 D 选择层内 XY 位移 | 层内位置修改的贡献 |
| J | 从 D 选择改层与 XY 联合修改 | 二者组合的贡献 |

每个学习臂的候选集含原始 D 和最多两个非恒等动作，强度固定为 0.15／0.35；每个动作最多涉及 `min(256, max(1, ceil(0.02*N)))` 个 APBlock。候选生成按固定图连接规则和 seed 运行，达到有限预算即停止，不等待收敛，不因负结果继续补候选。相同动作 digest 去重并记录。

L 保持全部 XY 原样，只向包含原 XY 的另一层兼容叶区域移动；Y 保持 layer／leaf 原样，在当前叶内向连接加权邻居中心移动；J 组合两者。固定位置、固定层、不可移动 APBlock 和五类资源约束均受检查。层移动受剩余容量限制，不能靠超容量制造候选。代理可行仍不等于物理 packing 一定合法，最终由 VPR 检验。

011 坐标接口明确允许叶区域内的有限位移。010 的“密度混合坐标 head 可达区间”是旧生成器的输出范围，不是物理合法性规则，因此不用于拒绝新残差坐标。D 原始动作仍单独通过 010 校验；新动作使用独立 011 schema，检查完整覆盖、兼容、固定约束、叶／层硬容量、相对 D 的修改范围、动作 digest 和来源 checkpoint。不能将修改后的动作挂上旧动作的 QoR。

## 标签、训练与验证保护

仅采集 train 条件的新候选反馈；候选的两次原生 VPR 测量均保留，包含失败。比较时先算相同条件、相同成功 seeds 的 WL／CPD 算术均值，再计算等权 log 分数：

`q = 0.5 * log(WL_candidate / WL_control) + 0.5 * log(CPD_candidate / CPD_control)`。

成功次数优先；同成功次数时使用共同成功集合的 QoR，全部失败时不虚构 QoR 排序。只使用动作原样、digest 与实测来源均匹配的标签；不用教师坐标重建、几何代理、密度均衡或初值保持率作为训练奖励。没有有效偏好则明确停止，保留候选覆盖诊断。

冻结基础 GNN，仅训练选择器 5 epochs，保存 epoch 2／5。验证集不反传。每个 L／Y／J 分别在相同共同成功集合比较两个 snapshot 和 D，候选不能降低验证成功数，且必须有负的共同 QoR 分数才启用；否则该臂固定回退 D，并记录“验证不支持新增修改”。不把回退视为学习收益。主比较使用未参与反馈或选模的 seeds，运行后不重新选择 snapshot 或开关。

q 用于选模，WL／CPD 分别报告；一好一坏称为取舍。平均正收益即可记录贡献，不要求每个 case 都改善。成功覆盖和失败必须一起报告。

## 冻结预算与阶段

| 阶段 | 矩阵 | 原生 P&R 上限 |
|---|---|---:|
| 反馈 | SPMV／MCML × 四 H×Z ×（D + L/Y/J 各两个候选）× seeds 11401／11402 | 112 |
| 验证 | 8 个已有 active-validation 条件 ×（D + L/Y/J 各两个 snapshot）× seeds 11201／11202 | 112 |
| 主比较 | SPMV／MCML × 四 H×Z × B/D/L/Y/J × seeds 11101／11102 | 80 |
| 合计 | 重复动作可减少实际运行，不补抽无限候选 | **304** |

参考宽度沿用 SPMV 128、MCML 224。条件和实际执行清单由 `plan` 及冻结 protocol 给出。源 008 图／教师记录仅提供已有条件与图索引，011 不学习教师标签。本轮两个主比较设计族已用于训练，因此属于开发归因；不声称独立族泛化。没有 Wmin 和可配置 Z 容量扫描。runtime 完整计入并报告为成本，不设加速验收门槛。

顺序为 `check → prepare → feedback → train → validate → compare → report`。各阶段具备输入／代码／checkpoint 签名与完成记录；错误停止后续阶段。正式计时固定 host、boot、CPU lanes 和方法轮换，不能把不同负载运行拼成一个计时结果。基础设施重试单列，不算研究收益。

## 后端如何使用修改

保存初值、GP、APPack 合法化和最终布局之间的层变化、覆盖率、GP 迭代及阶段时间；同时保存最终 route 的 XY WL 和实际 CHANZ 使用摘要。GP 改层比例仅用于解释机制；判断价值仍依据最终成功率、WL、CPD。

一次实验结束后区分：候选有真实收益但选择器没有学会；修改经过 GP 大量消失且无最终收益；固定预算内候选本身没有收益。后两种情况都不自动增加同配方训练轮数。若需要改变 GP 使用层偏好的方式，另开带初始化 × 后端对照的新实验。

## 服务器命令

使用保留 008–010 大型输入与 D checkpoint 的原服务器，激活已有 Torch 环境：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_3D_TRAIN_DEVICE=cuda
export TAILCOMPILE_V7_RESIDUAL_JOBS=4
export TAILCOMPILE_V7_RESIDUAL_JOB_MIB=8192

script="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_residual_campaign.sh"
bash "$script" plan
bash "$script" check
bash "$script" all --background
```

入口继承环境，不执行 `conda activate`。后台输出 PID、LOG、PIDFILE，使用打印的日志路径执行 `tail -f`。默认来源 `runs/development_010_structured`，输出 `runs/development_011_residual`；独立变量为 `TAILCOMPILE_V7_RESIDUAL_SOURCE/OUT/JOBS/JOB_MIB/TIMEOUT`。默认 4 并发、每任务 8192 MiB、超时 3600 秒。所有新增写入在当前仓库，原 VTR 安装只读。

分阶段时依次运行 `prepare`、`feedback`、`train`、`evaluate`；每一步确认完成再启动下一步，`evaluate` 执行验证、主比较和报告。不要并发启动同一输出的多个阶段。恢复沿用原参数；改代码、host 或冻结参数需新 cohort。

## 存储与交付

运行后先保存指标、层变化、最终 route 摘要和清理账本，再删除该 attempt 的 `.net`、`.net.post_routing`、`.route` 及重复结构／图导出。共享输入、反馈动作、基础 checkpoint 和选择器模型留在服务器供复核。

本地入口为 `python residual_local_smoke.py`，需要保留原 007／008 小架构及 010 本地 D checkpoint，最多 8 次真实反馈与 10 次五组比较；`python residual_local_smoke.py --verify-workers-only` 从原始结构重放生产 worker，核对实际消费初值的 SHA，不增加 P&R。新增输出位于忽略的 `runs/development_011_residual_local/`。本地小测试只证明代码、约束、训练及 VPR 接口可运行，不能当自然设计 QoR 结论。

评估报告及本地工程输出不提交 Git；源码、测试和本运行说明正常提交推送。服务器完成后按用户要求同步轻量原始 metrics／protocol；不要用 `git add -f` 上传模型或网表。
