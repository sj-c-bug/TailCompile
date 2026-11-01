# 008：让教师、容量解码与部署风险使用同一个方案

**2026-11-02状态更新：**服务器已完成教师派生、训练、160反馈＋80选模＋224比较。一致性修复已落地，但主方案总时间平均+14.24%，未加速，见[008结果评估](runs/development_008_aligned/REVIEW_2026-11-02.md)。下文为启动时配置和本地资格记录，“尚未运行”保留其历史时间点，不要据此再次启动同配方实验。

日期：2026-11-01。当前实现与本地验证进度以本文件后续补充的结果为准；**本页不是服务器实验已运行的证明**。007结果及所有2D版本保持冻结。

本地一致性验证已完成：真实 balanced/asymmetric 小图各141 APBlock、18叶区域，完成共享3 epochs＋L/T各2 epochs，训练/部署硬动作SHA相同、风险前向与硬方案CVaR数值相同，2/2小规模P&R成功。证据在 [aligned_smoke_report.json](runs/development_008_aligned_local/aligned_smoke_report.json)。自然设计的完整研究矩阵尚未运行。

最终38项测试与Bash语法检查通过；完整生成worker也通过构图、生成、lower及反馈动作验真的接口检查，初值SHA与已被VPR消费的文件一致。汇总见[本地结果与边界](runs/development_008_aligned_local/RESULTS_ALIGNMENT_2026-11-01.md)。

## 这次改什么

007发现：有些教师要求某层承载超过代理容量的RAM，训练风险计算又使用软层概率，而部署时硬解码器必须迁移节点。模型优化的对象因此与实际输出不同。

008先修正这个接口：

1. 原始教师只读保留；新教师在与部署相同的fixed、资源兼容和逐层容量代理下投影。新标签保存改层量、资源使用、容量检查及输入哈希；投影失败显式报告，不扩大容量。
2. 学习与部署调用同一套硬容量解码与有限层内密度处理。风险前向根据实际硬层分配和最终XY计算。
3. 硬分配本身不可微。当前训练采用有偏的straight-through层概率梯度近似；风险前向使用实际解码的硬layer和XY，不对分配器或VPR求梯度。前向数值一致不等于梯度精确，也不保证真实拥塞改善。通过容量代理仍不等于真实packing或详细布线合法。

共享条件XY模块使用修正后的训练标签重新进行有界预训练。旧007的共享权重不应默认为已经适应新标签。L/T仍共用冻结XY、共同候选数据、训练预算与选模规则；Tail项是两臂的唯一目标差别。

## 输入与运行边界

- 默认只读来源：`experiments/vtr/tailcompile_v7_3d/runs/development_007_layer/`。
- 原始图：来源cohort `teacher_records.json`指向的graph；以 `graphs/<condition_id>/payload/generation_report.json`记录的实际路径与哈希为准，不能自行拼接`worker`目录。它们包含真实垂直池与方向需求摘要，保留于服务器，不进Git。
- 原教师：同一记录指向的旧005标签。008输出派生标签，不能覆盖005或007文件。
- 后端：从007协议读取已验证的独立3D binary、manifest和qualification，核验身份后直接复用。没有新VTR补丁或RRG重建任务。
- RRG/device：复用007记录的只读缓存。冷启动计时仍包含每次逻辑处理/模型加载/解码/文件和完整VPR；器件准备成本按冻结口径另列。
- 新输出：`runs/development_008_aligned/`；scratch：`$TAILCOMPILE_REPO/vtr_work/tailcompile_v7_3d/aligned/`；日志：`$TAILCOMPILE_REPO/logs/`。

007的教师、旧默认结果和计时只用作来源及历史解释；新的正式计时使用同期Default，不能跨批相减当作方法收益。

## 服务器命令

在已有Torch及VPR依赖的环境执行。脚本继承当前Python和共享库环境，不自动激活conda，也不覆盖2D的`VTR_ROOT`。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_3D_TRAIN_DEVICE=cuda
export TAILCOMPILE_V7_ALIGNED_JOBS=4
export TAILCOMPILE_V7_ALIGNED_JOB_MIB=8192

script="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_aligned_campaign.sh"
bash "$script" check
bash "$script" all --background
```

`check`先检查关键输入是否存在；检查通过后再启动`all`。若来源cohort不完整或基础清单缺失，检查会直接报错，此时不要启动后续阶段。`all`仍会再次检查输入，然后执行 `prepare → pretrain → feedback → train → validate → compare → report`；各写入阶段还会核验来源、后端、资格与冻结哈希。脚本打印PID、日志和输出路径；从打印的日志读取进度。仅从Git拉取轻量结果的本地副本通常没有原始图、教师和RRG，需要在保留这些文件的服务器运行。

`plan`和`check`为前台只读操作，不创建输出目录、日志或锁。若输出/缓存通过symlink跳出所属目录，入口在任何写入前拒绝。

需要分CPU/GPU实例时，按阶段逐个执行，每一步成功后再启动下一步：

```bash
bash "$script" prepare --background
bash "$script" pretrain --background
bash "$script" feedback --background
bash "$script" train --background
bash "$script" evaluate --background
```

这五行表示顺序，不要一次同时启动。`pretrain/train`可使用CUDA；CPU阶段可将`TAILCOMPILE_3D_TRAIN_DEVICE`设为`cpu`。资源配置按阶段冻结，正式计时的host/boot/cores/lanes固定。

新cohort目录只通过`TAILCOMPILE_V7_ALIGNED_OUT`指定；scratch和日志使用上文固定的仓库内目录。入口不会误用旧的`TAILCOMPILE_V7_LAYER_OUT`、通用`TAILCOMPILE_V7_OUT`或`TAILCOMPILE_SCRATCH`。默认每任务8GiB源于007 MCML观测峰值约5.9GiB；调度器还应根据可用内存和物理核限制并行。

如需更换cohort，分别设置`TAILCOMPILE_V7_ALIGNED_SOURCE`和`TAILCOMPILE_V7_ALIGNED_OUT`。来源必须位于本仓库的v7 `runs/`下；新输出必须是其中独立的`development_008...`目录，不能与来源重叠。不要直接复用007输出目录。直接调用Python时参数名是`--source`、`--out`；上述Bash命令已经代传。

## 验收与预算

本次修复的必要验证是：教师通过同一逐层容量检查、训练与部署硬动作相同、风险前向与硬方案诊断一致、固定约束与AP/molecule原子性保留、实际梯度能更新层模型。不能以旧标签`projected`的名称代替检查，也不能把代理通过称为真实物理合法。

自然设计后续对照如执行，预先登记的上限为：反馈160＋选模80＋七组开发224＝**464次研究P&R**，不增加新后端资格P&R。SPMV/MCML仍是train开发族；blob_merge/diffeq仍是已暴露active validation。这不产生独立族泛化或Wmin结论。

每次状态、指标、层迁移与路由摘要落盘后，清理该attempt不再使用的`.net/.net.post_routing/.route`和重复图/导出；教师、训练图、候选动作与权重按后续依赖保留。只提交轻量协议、哈希、指标与中文报告。

## 本地验证的具体边界

这次复用的两个真实GP教师本来都满足逐层容量代理，因此没有把它们描述成“修复了真实RAM溢出”。新投影修正了45/50个AP的XY目标，使其落在部署时密度混合能产生的坐标范围内。另建了明确标记为`controlled=true`、不参与训练的RAM超容量标签：把两个各自兼容的RAM AP都放入仅一个RAM代理单位的层，逐层memory超用从1投影为0，未增大器件容量。

balanced的训练前向/部署CVaR均为0；asymmetric均为1.2003130912780762。关闭推理诊断得到相同动作，且不再额外计算软、硬两套风险。原007图、GP输出、输入XML、网表及binary前后哈希一致。

两次新P&R均使用T-I及seed3701，无新后端编译、无新RRG提取，初值消费、原生GP继续求解与固定原子检查通过。训练/部署一致只针对GP之前输出的初值；GP、APPack和SA仍可以移动节点。此次没有运行性能基线、同条件不同动作的反馈采集或Wmin，反馈微调明确跳过。记录的约2.42/2.62秒不能作为加速结论。

后续等价代码调整仅涉及无梯度路径和统一容差。用最终代码重新加载既有T checkpoint，两个案例生成的初值SHA均与之前VPR实际消费的初值一致，动作及风险值也一致；新增训练和P&R均为0，原始协议与实验报告保持不变。复核证据见 [final_code_compatibility_report.json](runs/development_008_aligned_local/final_code_compatibility_report.json)。这项复核同样不代替尚待服务器执行的自然设计训练与对照。
