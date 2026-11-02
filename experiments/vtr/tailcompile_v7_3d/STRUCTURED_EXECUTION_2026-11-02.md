# 010：资源与连接感知的分层、局部密度和坐标学习

本轮针对009暴露的部署限制改进方法，首版去掉TailRisk。使用已有GNN网表／器件编码、资源需求、连接与逻辑深度时序代理；不把几何或逻辑深度代理称为VPR真实时序。输出仍是packing前APBlock级XYZ初值，后端继续同一原生GP→APPack→详细退火→route。

源码与原2D、007–009隔离；只读复用009选中的X checkpoint、008对齐教师、007图以及已有合格3D后端和RRG。没有新增后端补丁、重编译、提前完整packing或教师P&R。所有新写入位于当前主仓库的010输出、3D scratch和logs；原2D VTR只读。

## 方法边界与四组归因

| 方法 | 初始权重 | 解码器 | 本轮训练 |
|---|---|---|---|
| B | 原生Default | 原生 | 无 |
| X | 已选009X | 原009解码器 | 无 |
| D | 从同一009X迁移，新增上下文输出初始为零 | 新010有界多资源／密度解码 | 无 |
| N | 与D相同迁移起点 | 同一010解码器 | 5 epochs |

主要比较N/B，D/X定位新解码器贡献，N/D定位新学习目标及网表上下文贡献，N/X报告完整改动增量。D/N使用相同容量规则。资源代理可行不等于实际packing合法；新解码器无法在预算内找到可行解时显式报告失败，不保留溢出动作，也不静默回退Default。

层监督按层与资源需求平衡，不强制上下层节点各半。XY训练增加直接可微的连接线长／关键度几何目标和密度目标，不仅依赖旧教师坐标误差。实际效果以最终routed WL与CPD为准。全部方法使用`TC_GP_LAYER_PRIOR=0`，不锁层，不改变GP正常迭代与退出规则，因此本轮不能声称已解决后端改层这一问题。

## 固定研究预算

1. **新反馈最多16次P&R。**8个SPMV/MCML参考条件，每条件用未训练D产生两个候选，score noise为0和0.25、固定生成seed10301；同条件相同SHA跳过，不无限补候选。使用相同反馈seed10401，真实运行后保留成功、失败与QoR。候选按固定CPU lanes并行，每lane顺序执行。
2. **训练5 epochs，保存epoch2/5。**仅16个train条件。旧008/009动作只有在原样通过010完整动作／叶容量检查时才允许使用其旧标签；不投影旧动作后继续挂旧QoR。旧教师若需投影，只提供几何监督，没有虚构的真实QoR。新候选提供010可部署动作的真实排序信号；有效偏好为零时停止，保留诊断，不继续昂贵选模和主比较。
3. **24次真实选模。**8个active-validation条件×seed10201×（B、N_epoch2、N_epoch5）。按成功数优先，再比较相同共同成功集合的WL/CPD等权log分数，平局选早epoch；runtime不参与选择。没有共同成功QoR时停止主比较。
4. **64次主比较。**SPMV/MCML×四架构×seeds10101/10102×B/X/D/N，公共参考W分别128/224。固定host/boot、CPU lanes、方法轮换、同步波次；完整计入冷启动成本。已有同条件的009运行不拼入新计时。

首轮研究上限104次原生P&R，不含另记的本地工程测试。没有Wmin、减配置Z或独立设计族验收，不能从两个新seeds推广到整个benchmark。

主要指标：同条件相同成功seeds的WL、CPD分别算术平均后取方法／基线比值，再对架构、设计族等权几何汇总。允许个案退化；分别报告两指标、失败与覆盖、Pareto取舍及真实最终Z占用。原生WL只含XY路由长度，Z另列。runtime为成本，不再设5%加速门槛。

## 服务器运行

先在保留原008/009大数据的服务器激活已有Torch/VPR依赖环境。入口继承环境，不执行`conda activate`；不用传ZIP，也不会改原2D的`VTR_ROOT`。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_3D_TRAIN_DEVICE=cuda
export TAILCOMPILE_V7_STRUCTURED_JOBS=4
export TAILCOMPILE_V7_STRUCTURED_JOB_MIB=8192

script="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_structured_campaign.sh"
bash "$script" plan
bash "$script" check
bash "$script" all --background
```

完整顺序为`check → prepare → feedback → train → validate → compare → report`。prepare只迁移D控制权重和记录只读输入，不训练、不重新准备RRG。后台启动打印PID、LOG、PIDFILE和OUT，按打印的LOG路径查看`tail -f`。

默认来源`runs/development_009_qor`，默认输出`runs/development_010_structured`。新变量独立于009：`TAILCOMPILE_V7_STRUCTURED_SOURCE/OUT/JOBS/JOB_MIB/TIMEOUT`，默认并发4、每任务8192MiB、超时3600秒。旧的QOR/LAYER/ALIGNED_OUT及`TAILCOMPILE_SCRATCH`不决定010输出。

要分CPU/GPU阶段，顺序运行：

```bash
bash "$script" prepare --background
# 上一步完成后：
bash "$script" feedback --background
# 反馈完成后，在GPU环境：
bash "$script" train --background
# 训练完成后：
bash "$script" evaluate --background
```

`evaluate`依次执行validate、compare、report。每阶段开始前冻结输入与代码签名；失败停止后续阶段。正式计时中更换host、CPU或源码需新cohort，不拼接前后部分波次。

## 本地检查与功能边界

本地小规模入口为`python experiments/vtr/tailcompile_v7_3d/structured_local_smoke.py`，需要保留原007/008小架构与009本地X checkpoint。它最多运行24次原生P&R：8次真实候选标签和16次B/X/D/N比较，使用不同反馈／比较seeds；结果位于忽略的`runs/development_010_structured_local/`。这是两个小架构上的工程检查，不能代替服务器自然设计研究或未见族验收。

新模型、解码、训练、campaign及Bash入口测试覆盖硬叶容量、固定轴梯度、教师投影、真实动作反馈、旧动作支持域、checkpoint防替换和统计聚合；旧009回归一并运行。原固定VPR和2D源码不修改。

硬解码最多两次确定性排序尝试，没有等待收敛的局部搜索。它可能因代理约束或贪心碎片化失败，不能将失败说成芯片物理不可布局。soft区域／层梯度仍是有偏近似，时序仍使用逻辑深度代理；不对VPR直接求导。GNN初值供GP自由优化，本轮没有解决所有后端改层行为，也不将提高初值保留率作为独立收益。

## 存储与结果保留

原生运行先保存指标、最终route摘要、分层变化、动作和清理账本，再删除当前attempt的`.net/.net.post_routing/.route`及重复graph和结构导出；源图、教师、动作和checkpoint继续留存。新协议是`structured_protocol.json`，最终本地报告是`structured_report.json`。

按用户最新约定，**评估报告不提交Git**；算法源码、测试与运行说明提交，结果文件保留本地／服务器。不要对010目录执行`git add -f`。需要协作读取轻量原始metrics时只同步明确需要的原始证据，报告不随之提交。
