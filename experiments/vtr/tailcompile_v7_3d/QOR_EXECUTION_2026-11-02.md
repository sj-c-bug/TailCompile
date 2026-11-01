# 009：WL / CPD 训练与六组开发对照

日期：2026-11-02。研究依据为[009实验计划](../../vtr3d/EXPERIMENT_PLAN_3D_QOR_2026-11-02.md)。本入口只执行该计划前两步：训练改为QoR目标、固定宽度的六组比较；Wmin及减少配置Z容量另行实施。

服务器自然设计阶段包含96次反馈、176次选模、240次开发比较，最多512次真实P&R。本地小模型测试不代替该矩阵，也不构成QoR改善或独立族泛化结论。

实现已通过50项测试、完整生成worker检查和4次真实小网表P&R。[测试记录](runs/development_009_qor_local/unit_tests.json)、[worker记录](runs/development_009_qor_local/worker_contract_report.json)、[真实008指标回放](runs/development_009_qor_local/preference_replay_report.json)均已保存；回放得到42个QoR对和6个成功数对，18个成功对与runtime排序相反。原动作数组仍须在服务器由训练器验真。

固定40f8线长源码与四条真实路线的计数依据见[线长语义报告](runs/development_009_qor_local/wirelength_semantics_report.json)。训练评分使用masked-logits的log_softmax处理极低概率的合法历史动作，避免下溢为零后报错或失去梯度；硬不兼容mask仍拒绝，部署硬解码不变。

## 输入与输出

- 只读来源默认`runs/development_008_aligned/`，以及它记录的原始图、派生教师、候选动作和checkpoint；Git轻量副本没有这些大文件，需要在保留原始数据的服务器运行。
- 复用008已核验的3D后端、资格与RRG，不重编译VTR，不重采已有160次反馈。
- 独立输出默认`runs/development_009_qor/`，scratch为`$TAILCOMPILE_REPO/vtr_work/tailcompile_v7_3d/qor/`，日志为`$TAILCOMPILE_REPO/logs/`。
- 配置只使用`TAILCOMPILE_V7_QOR_*`；旧`TAILCOMPILE_V7_ALIGNED_OUT`、`TAILCOMPILE_V7_OUT`及`TAILCOMPILE_SCRATCH`不决定009输出。入口检查真实路径与symlink，禁止写入旧cohort或仓库外。

## 一次启动完整流程

先激活现有Torch/VPR依赖环境。脚本继承当前环境，不调用`conda activate`，不更改原2D的`VTR_ROOT`。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_3D_TRAIN_DEVICE=cuda
export TAILCOMPILE_V7_QOR_JOBS=4
export TAILCOMPILE_V7_QOR_JOB_MIB=8192

script="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_qor_campaign.sh"
bash "$script" check
bash "$script" all --background
```

`check`为前台只读输入检查；通过后再启动`all`。完整顺序为：

```text
check → prepare → bootstrap → feedback → train → validate → compare → report
```

五个训练臂是R（runtime目标控制）、Q（WL/CPD）、C（Q去掉跨层cut惩罚）、X（C增加条件区域/坐标学习）、T（X增加垂直TailRisk）。加原生Default共六组，使用同一后端、相同参考宽度及仅初值接口。R/Q/C冻结同一XY模块，X/T有可训练坐标路径。训练/选模不再以总runtime收益作为QoR臂成功依据。

`all --background`打印PID、日志和输出目录。使用打印出的绝对路径查看日志，例如`tail -f /实际日志路径.log`。普通断开终端由nohup保护；平台主动结束实例仍需要恢复同一stage。

默认4个CPU并发、每任务至少8GiB；并发会按可用物理核与内存进一步限制。16核且可用内存充分的实例可在首次启动前设`TAILCOMPILE_V7_QOR_JOBS=16`。正式比较开始后固定host/boot、核与并发配置，不把不同会话的时间拼成一个加速比。

## 分CPU/GPU运行和恢复

需要分开实例时，依次执行下列阶段；每个阶段完成后才进入下一阶段，不要一次全部后台启动。

```bash
bash "$script" prepare --background
bash "$script" bootstrap --background
bash "$script" feedback --background
bash "$script" train --background
bash "$script" evaluate --background
```

`bootstrap/train`可设`TAILCOMPILE_3D_TRAIN_DEVICE=cuda`；其他阶段可用`cpu`。`evaluate`依次执行`validate → compare → report`。缺失文件、签名不一致或上游未完成会停止后续阶段，不把未知状态当失败标签。源码或数据语义改变须使用新009cohort。

完整配置变量：

| 变量 | 默认值/含义 |
|---|---|
| `TAILCOMPILE_V7_QOR_SOURCE` | 同仓库`runs/development_008_aligned` |
| `TAILCOMPILE_V7_QOR_OUT` | 同仓库独立`runs/development_009_qor` |
| `TAILCOMPILE_V7_QOR_JOBS` | 4，正整数 |
| `TAILCOMPILE_V7_QOR_JOB_MIB` | 8192，每任务内存预算 |
| `TAILCOMPILE_V7_QOR_TIMEOUT` | 3600秒 |
| `TAILCOMPILE_3D_PYTHON` | 当前`python3` |
| `TAILCOMPILE_3D_TRAIN_DEVICE` | 默认cpu，可设cuda或cuda:N |

`plan/check`不创建输出目录、日志或锁，不支持`--background`。其余阶段只允许同一输出一个主进程，禁止两个任务争写同一cohort。

## 结果解释与清理

主报告分别列WL、CPD、联合分数、Pareto取舍、成功覆盖及全部失败。每条件先五seeds算术平均，再按架构/设计族等权汇总；允许个案退化。两指标平均都改善才称联合改善，一项下降另一项上升明确为取舍。总runtime、内存和训练成本照实报告。

仅16个train条件进入训练/尺度；8个active-validation条件仅用于审计与共同epoch选择。SPMV/MCML本轮都是已训练开发族，不能称未见族验收。没有新增Wmin网格或Z配置扫描，不产生这两项结论。

在VPR完成后先保存路由XY tile-span、去重Z占用、层迁移、状态和QoR摘要，再按当前attempt目录清理`.net/.net.post_routing/.route`及重复导出。训练依赖的图、教师、动作和权重继续保留；Git只同步轻量协议和结果，不能`git add -f`大文件。

## 本地工程验证已完成的范围

证据：[qor_smoke_report.json](runs/development_009_qor_local/qor_smoke_report.json)。复用007的balanced/asymmetric真实小图（每例141 AP、18叶区域）和008对齐教师及L checkpoint，从相同参数开始对五臂各进行2 epochs真实教师监督训练。

- R/Q/C的共享XY权重完全不变，X/T的XY权重及相同指定层/叶分配下的坐标输出均改变。T的坐标action-score梯度L1在两例分别约5.06/2.05；C为0，符合冻结设置。
- 训练前向与部署动作SHA一致，硬方案风险数值一致；默认推理不额外计算风险诊断。本次复用007/008的源图、教师、输入与binary前后hash保持不变。
- C/T各对两个小图、seed9501完成一次真实P&R，共4/4成功；初值被消费、原生GP继续、fixed检查通过。没有新增backend构建或RRG提取。
- 本地没有同条件、不同动作的真实QoR反馈池，所以反馈微调显式跳过；没有用两个不同架构的运行时间或线长伪造偏好。上述训练只验教师/坐标路径，不证明新QoR目标已经有效。
- `.route`在摘要提取后清理。四次VPR `Total wirelength`分别为2489、1987、3038、2899，与每net去重后的XY tile-span逐一相等；CHANZ使用/跃迁分别为21、14、75、74。该指标不包含纵向线长，不能称物理3D总导线长度，也不能据这四次工程样例宣称Tail收益。
- 所有本次不再使用的`.net/.net.post_routing/.route`和重复布局trace已在摘要落盘后清理；训练图、动作和权重按后续依赖保留。

Launcher的8项真实Bash测试通过，覆盖后台日志/PID、阶段失败停止、只读plan/check和009路径隔离。以上是工程证据；自然设计512次尚待服务器执行，不产生Wmin或独立族泛化结论。

后续来源检查加强、route文件读取器的context-manager修正和动作评分的稳定log-softmax实现没有改变本地smoke的生成算法。最终代码重生成四个初值，SHA均与此前VPR实际消费文件相同，动作与硬风险仍一致；四例动作评分与梯度也均为有限值。新增训练/P&R均为0，既有checkpoint未按评分修复重新训练。原始protocol与smoke报告保留，补充证据见[final_code_compatibility_report.json](runs/development_009_qor_local/final_code_compatibility_report.json)。route摘要提取后已清理，因此未声称对最终读取器重新解析原route。
