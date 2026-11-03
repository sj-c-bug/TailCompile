# 012：使用修改局部的 GNN 表示，复用 011 标签做离线比较

## 本轮解决什么

011 的候选由固定代码基于 D 生成，评分网络负责从原始 D、少量改层、层内 XY 和联合修改中作选择。其验证输出全部为原始 D；但已测候选中确实存在最终 QoR 改善。因此本轮检验：让评分网络看清**改动发生在哪些节点和区域**，能否改善候选排序与选择。

新增输入包括被修改 APBlock 及其一跳邻居的冻结 GNN 表示、移动前后所属区域表示、方向性变化、受影响连接和关键度代理、局部资源与坐标变化。局部向量维数为 `10×hidden+77`；关键度仍是原逻辑深度代理，不是免费可用的 post-packing 真实时序。它们均来自布局之前的网表、器件和原候选，不读取教师、真实 route、真实 CPD 或反馈奖励作为输入。最终训练标签仍是原候选的真实 VPR 成功率和 WL／CPD。

本轮仍训练**候选评分头**，不更新 D 的 GNN，不学习产生候选集之外的新坐标。本轮也不改变候选强度、节点选择规则、容量约束或 GP。原 2D、007–011 算法与结果冻结；无 VTR 重编译，无新教师，无新增 P&R。

## 一次安排好归因

| 模式 | 全图 GNN 与原 32 维候选统计 | 新局部表示 | 评分网络 |
|---|---|---|---|
| G | 使用 | 输入置零 | 同一结构、维数、参数数量、起始权重 |
| E | 使用 | 使用真实值 | 同一结构、维数、参数数量、起始权重 |

G 保留 011 的全图／汇总信息，使用和 E 相同的新增输入槽但将其置零。二者使用相同的候选、标签、训练顺序、优化器、学习率、随机种子、更新预算和 identity 平局规则；区别仅在是否提供新局部信息。不将 011 训练 5 epochs 与新模型训练 80 epochs 直接作为特征贡献对照。

原始 D checkpoint 由 `development_011_residual/prepared_report.json` 记录，必须匹配 SHA。特征准备重新生成相同的 011 候选，逐一核对原已测动作的完整 digest、坐标、固定与容量约束、文件字节和模型来源，原样通过才复用标签。不能修改候选后继续使用原 QoR。特征只计算一次并保存签名缓存；后续 28 次拟合复用缓存，不反复构建大网表图。

identity 的新增局部表示为零。G／E 使用同一有限候选域；离线比较只允许选择已有实测标签的动作，不把未测动作假设为成功。自然设计 011 已包含每条件 7 个实测动作；小型工程样例若只有部分动作，覆盖范围明确单列。

## 固定训练与留出设置

已有自然设计数据为 SPMV／MCML × 四 H×Z，共 8 个条件、56 个动作、112 次历史 P&R 测量、72 个有效偏好。112 是已经完成的 011 采集，本轮增加的 P&R 为 **0**。

| 设置 | 训练／评价范围 | 用途 |
|---|---|---|
| 全数据拟合 | 8 条件参与训练；只评价训练拟合 | 判断是否能学会已有候选的得失 |
| 留设计族 | 两折，每次一个设计族的 4 条件训练，另一族评价 | 检查评分标签的跨族转移 |
| 留架构 | 四折，每次 3 种架构的 6 条件训练，剩余架构的 2 条件评价 | 检查评分标签的跨架构转移 |

每种设置使用 G／E 两模式、训练随机 seeds 12301／12302，共 `(1+2+4)×2×2=28` 次离线拟合。训练 seed 与 VPR seed 是不同概念；本轮 VPR 标签继续来自历史 seeds 11401／11402。

28 次拟合相互独立，CPU 默认请求 **4 个进程并行**，实际并发由可用 CPU／内存及待执行任务数量限制。特征准备只执行一次，随后各进程只读共享签名缓存，每个进程一个 Torch／BLAS 线程，使用独立训练随机种子、目录、日志和 checkpoint。并行任务不会在线程间共享模型或全局随机数状态。只有调度和进度写入在父进程中完成；最终清单按固定矩阵顺序排序，任务完成先后不改变 G／E 配对或评价规则。

隐藏宽度固定 32，Adam 学习率固定 0.001，训练 80 epochs。保存 epoch 5／20／80，供全数据训练拟合诊断；**留出结果固定评价 epoch 80**，不根据留出 WL／CPD、排名或收益挑 epoch、学习率或模型。每一折先确定训练 ID，再仅用训练折计算偏好权重尺度，留出标签不影响归一化或优化。

族／架构相同条件下的不同候选、强度与 VPR seeds 一并分组，禁止随机拆成独立行划分训练与测试。若无有效训练偏好，保存诊断并停止，不声称已完成有意义的学习。

基础 GNN D 本身已经接触过 SPMV／MCML，因此这里的“留族”只留出了**评分头的反馈标签**。它不能证明完整工具对未见设计族泛化，也不是论文最终独立设计验收。两个设计族与两个历史 VPR seeds 不足以做强显著性结论。

## 怎样评价

先用特征与模型分数选择候选并冻结选择，再读取该动作已有的测量，不能先看 QoR 再选择。

报告训练／留出排序正确率、identity／非恒等选择比例、选中动作的成功次数、WL／CPD、相对 D 的联合分数，以及相对同一候选池最优动作的 regret。离线最优仅是已知标签的参考上限，不是可部署方法。L／Y／J 分别报告，不将不同动作类型混成一个有利平均值。

成功次数优先；在同条件共同成功 seeds 上先做 WL、CPD 算术平均，再计算方法／D 的比值及等权 log 分数。架构与族等权汇总，WL、CPD 分别展示，取舍明确说明。另按两个历史 seeds 分开核对收益符号是否稳定；不把它们算作额外独立设计。

判断新局部表示有用，需要它在相同预算下相对 G 提升**留出候选选择**，而不只是降低训练损失。全数据拟合好、留出不好应记为转移失败；训练就学不会，应优先检查表示和优化；没有证据时不扩同配方 P&R。

这些离线结果只评价已测有限候选池中的选择能力。没有新 P&R、Wmin 扫描或配置 Z 容量扫描，不能将它们称为新布局布线效果验收。出现值得复验的能力后，另冻结新策略做独立原生 P&R 对照。

## 服务器运行

使用保留原 011 动作与 feedback.json、008 图、010 D checkpoint 的服务器。Git 中的轻量结果不包含这些大型文件，本地只有结果 JSON 时不能完成自然设计特征提取。入口只读借用它们，全部新增写入位于当前仓库的独立 012 输出／scratch／logs。

并行版默认使用新的 `development_012_edit_context_parallel` cohort，原串行 `development_012_edit_context` 的结果保持冻结。若旧串行任务仍在运行，先让它结束或停止该任务，再更新源码；运行中直接 `git pull` 会使旧任务后续阶段遇到代码签名变化。不要把并行版输出指向已有串行 cohort，也不要在同一输出启动两份调度器。

激活已有 Torch 环境后：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_3D_TRAIN_DEVICE=cpu
export TAILCOMPILE_V7_EDIT_CONTEXT_JOBS=4
export TAILCOMPILE_V7_EDIT_CONTEXT_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_012_edit_context_parallel"

script="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_edit_context_campaign.sh"
bash "$script" plan
bash "$script" check
bash "$script" all --background
```

CPU 是默认选择：基础 GNN 表示提取一次后，后续只训练小型评分头。可将 `TAILCOMPILE_V7_EDIT_CONTEXT_JOBS` 改为 `8` 等正整数，按可用 CPU／内存安排进程数；每个进程仍只有一个计算线程。任务运行时间不再用于比较 G／E 的算法性能，完整离线成本另记。

已有 CUDA 环境可将 `TAILCOMPILE_3D_TRAIN_DEVICE` 改为 `cuda`，仅将评分头拟合放到 GPU。CUDA 默认并发为 1，显式设置 jobs 大于 1 会在启动前拒绝，避免多个训练进程同时争用同一设备。切换到 CUDA 时将 `TAILCOMPILE_V7_EDIT_CONTEXT_JOBS=1` 或取消此前的 CPU 并发设置。入口继承环境，不执行 conda activate，不启动 VPR。

默认源 `runs/development_011_residual`，默认输出 `runs/development_012_edit_context_parallel`。独立变量为 `TAILCOMPILE_V7_EDIT_CONTEXT_SOURCE/OUT/JOBS/FIT_TIMEOUT`；每个 worker 默认超时 1800 秒，Python 入口同样支持 `--jobs N` 和 `--fit-timeout SECONDS`。阶段为 `plan/check/prepare/train/report`，`all` 依次检查、准备缓存、并行拟合、汇总。支持后台日志与锁；同输出不能并发启动多个阶段。失败后修复原因并按相同冻结参数恢复，改代码或输入需新 cohort。

若旧串行 cohort 已完成特征准备，可在启动前显式指定：

```bash
export TAILCOMPILE_V7_EDIT_CONTEXT_CACHE_FROM="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_012_edit_context"
```

并行版会核对旧缓存字节／内容签名、特征实现和来源资格，再只读复用；不会覆盖旧结果，也不导入旧评分头或用旧拟合结果替代新对照。此选项不能绕过原 011 数据的核验。未指定时在新 cohort 中正常准备一次缓存；Python 对应参数为 `--cache-from COHORT`。

单个拟合失败时停止继续派发，并终止本次调度器尚未完成的 worker；保留原始失败日志、已完成任务和检查点，后续阶段不会启动。恢复时先核对已完成任务的签名与 checkpoint，仅重跑未完成项；每个拟合最多两个 attempt。阶段日志中的任务完成数表示调度进度，不表示有多少个新布局布线实验。

完成后首先看日志中的阶段完成标记。评估报告保留本地／服务器；如需同步，只提交忽略规则允许的轻量协议、缓存清单与原始拟合 metrics，不使用 `git add -f` 上传缓存张量、动作或模型。

## 本地工程验证与存储

本地入口 `python edit_context_local_smoke.py` 复用已有 011 小架构的真实反馈，完成两个架构上的 12 次小型离线拟合、分组留出、原候选输出与哈希检查；不新增 P&R，不把小样例拟合收益当自然设计证据。本地输出置于忽略的 `runs/development_012_edit_context_local/`。

新缓存只保存紧凑的特征与原测量，不复制完整逻辑图、RRG 或基础 GNN 到每个拟合目录。head checkpoint 与缓存张量不进 Git。原始输入、011 动作与 checkpoint 不删除；本轮不产生 `.net`、`.net.post_routing` 或 `.route`。
