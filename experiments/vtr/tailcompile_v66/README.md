# v6.6：VPR 全局粗布局上的局部修正实验

## 要回答什么

将旧流程的“从独立 GNN 布局开始”改成“从 VPR 自己的全局粗布局开始”。
先验证无修改回放，再分别检查：候选池有没有好解、模型能否选中好解。
主指标为同一宽度网格上的 sampled Wmin 和 W80；端到端 runtime、线长、CPD 为配对次指标。
不把单次失败转成功、代理指标下降或测试标签挑选出的最优候选当作最终验收。

**当前状态：代码与本地接口测试已完成；20 设计族服务器实验尚未运行完成。**
`local_probe_report.json` 是 SHA 上的探索性候选上限检查，不能代替独立验证。

## 为什么需要一个 VTR 补丁

基准 commit 固定为服务器的 `40f8d19708b0b60e2aa04452deb1cfed68d96b0b`。
该版本没有 `--write_post_gp_flat_place` 开关。原生 `--read_flat_place` 会跳过
Global Placement，并改变 RAM 分组路径；直接回放最终 `.place` 也不是粗布局实验。

`install_hook.py` 仅在 `analytical_placement_flow.cpp` 中加入一个调用，另附
`tailcompile_gp_hook.inc`。钩子在真正的 GP 完成之后、完整合法化之前：

1. 导出 VPR 的 AP blocks、原子成员、实际 RAM/molecule 分组、超网和连续坐标。
2. 可选地读入 `.fplace`，检查全部原子覆盖、组内一致、固定块和 layer/subtile 不变。
3. 继续原有 appack、annealer 和 router，不重新推断分组，不跳过 GP。
4. 部署评测可同步调用 CPU worker：候选生成、特征提取、模型加载与推理全部计入运行时间。

未设置 `TC_GP_EXPORT` 时钩子不工作。所有新试验记录基础 commit、补丁、二进制、
架构、网表、候选和代码哈希；这是**独立的 patched-VTR cohort**，不得与旧二进制结果混算。
旧版本 runner 会拒绝这个修改过的 VTR checkout，属于预期保护。
新 `.fplace` 是 **post-GP 接口文件**，不能假设换成原生 `--read_flat_place` 仍有相同效果。

## 数据覆盖与冻结

`cohort.json` 在本轮标签之前固定 20 个设计族：13 train、3 validation、4 test。
以前使用过的 SHA、attention、SP-MV、六个 holdout 以及 bgm/or1200/stereovision/TPU
全部归入开发训练侧。最终 test 为 arm_core、spree、mkPktMerge、LU8PEEng。
同族参数变体不能分到不同集合。数量代表计划；综合/映射失败必须如实报告，不能悄悄补入有利设计。

- Default-only screen：width 40/80/120/160/200/240/280/320，seeds 201/202。
- 每个设计围绕最先成功宽度，冻结 ±30、步长 10 的七档网格。
- train/validation：seeds 211/212/213；test：231–235。
- 每个宽度/seed 的候选都来自它自己的 GP，不从其他 seed 偷用布局。
- 主方法是同一个部署算法随给定宽度生成候选，不是全扫宽度时固定同一张布局。
- 网格下沿成功记左删失；所有宽度失败记右删失；不得宣称这是精确最优 Wmin。
  出现大量删失，应冻结新一轮更宽网格，而不是把端点当作真实边界。

## 动作与模型

每个 GP anchor 产生 replay 加 12 个候选：1%/2%/5% 预算 × hotspot/wire 两种策略 × 两个重复。
热点来自固定 8-tile RUDY 需求估计，不读取真实 routing 标签；只交换模型类型多重集合相同的
可移动 AP blocks，保持 molecule/RAM 原子性及 XY 占用分布。单次交换限制曼哈顿距离 16，
整体 HPWL 增长超过 3% 的候选还原为 no-op。记录真实不同 `.fplace` 数量；重名不同不算多样性。
这不是完整容量合法化，最终合法性仍由 VPR 保证。

选择器是两层 AP 图 GNN：超网用线性规模的星形边展开，不构造二次规模 clique；输入为
坐标、位移、度、固定性、原子数、局部需求、是否移动以及固定架构 model ID 的类别计数。
全局输入包含 channel width、规模、Tail/mean、HPWL。
输出为成功 logit、相对线长/CPD/runtime 和排序分数，使用 VPR 标签监督。
失败运行的短耗时不作为加速标签。seed 数字不是特征。

**范围限制：本轮优先验证 post-GP 动作空间，并未重新训练旧双塔 RRG encoder。**
RUDY/channel width 不是方向/segment-specific RRG cut capacity；不能把这套 selector 称为
已完成的空间容量模型。若候选上限有收益而选择器仍弱，可在同一冻结动作池上接入原层次 RRG
编码器比较，不必再次重跑所有动作标签。候选上限也无收益时，应先改动作/接入位置。

只用 validation 选择 checkpoint。test 选择结果在任何测试候选布线前保存。
允许 Default 回退；固定门槛为 rank margin 0.05、预测成功率下降不超过 0.02，
预测线长/CPD 比率不超过 1.03。门槛不是根据测试结果调整的。
本版没有校准的 ensemble uncertainty，因此 margin 不能解释为概率置信度。

## 服务器运行

使用已装好 numpy、torch 的环境；CPU 阶段也需要 CPU torch 做部署推理。
不要在 GPU 自动空闲关机的实例上执行 prepare/evaluate。
`VTR_ROOT` 必须是上述服务器 commit 且源码没有其他修改；保留旧环境时建议先准备专用 checkout。
现有构建目录可增量编译，工具不会安装/替换 CUDA 或 Conda。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
python3 experiments/vtr/tailcompile_v66/install_hook.py "$VTR_ROOT"
cmake --build "$VTR_ROOT/build" --target vpr -j 8
python3 experiments/vtr/tailcompile_v66/test_v66.py

export TAILCOMPILE_V66_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v66/runs/gp_001"
export TAILCOMPILE_V66_JOBS=8
export TAILCOMPILE_V66_PREPARE_JOBS=2
export TAILCOMPILE_V66_THREADS=1
export TAILCOMPILE_V66_TIMEOUT=7200
mkdir -p "$TAILCOMPILE_REPO/logs"
```

CPU 阶段一：自动综合、Default 扫描、逐 seed GP anchors、候选与特征缓存、训练/验证标签。
没有 GPU 等待型工作；候选生成用独立 Python 进程并行，不依赖 Python 线程突破 GIL。

```bash
nohup bash experiments/vtr/tailcompile_v66/run_pipeline.sh prepare \
  > "$TAILCOMPILE_REPO/logs/v66_prepare.log" 2>&1 </dev/null &
echo "PID=$!"
```

等 `labels_complete.json` 为 true，再在 GPU 实例运行：

```bash
nohup bash experiments/vtr/tailcompile_v66/run_pipeline.sh train \
  > "$TAILCOMPILE_REPO/logs/v66_train.log" 2>&1 </dev/null &
echo "PID=$!"
```

模型完成后回 CPU 实例，保持相同 out、VTR 二进制、每进程线程数：

```bash
nohup bash experiments/vtr/tailcompile_v66/run_pipeline.sh evaluate \
  > "$TAILCOMPILE_REPO/logs/v66_evaluate.log" 2>&1 </dev/null &
echo "PID=$!"
```

evaluate 先冻结选择，再并行评测全部测试候选（用于 oracle 诊断），然后做真实同步 worker 部署，
最后单进程交替顺序运行 Default/selected 计时。计时阶段**刻意不并行 VPR**，避免资源争抢伪造加速；
请使用没有其他重任务的 CPU 实例。它不重复训练。

完整计划约 7,400 次 VPR：不是再用几十条样本判断 GNN；具体数量取决于最终可运行设计数。
实际原子/RAM 组图和特征缓存可能较大，优先高速本地存储；原始输入和结果需长期保留。
第一次建议 JOBS=4 或 8，观察内存后增加，而非仅按 CPU 核数全开。

## 断点、失败与验收

```bash
bash experiments/vtr/tailcompile_v66/run_pipeline.sh status
```

最终需 `labels_complete/trained/selected/evaluation_complete/deployment_complete/timing_complete/report_complete`
全部 true，且 `evaluation_report.json` 中 replay 检查通过、没有基础设施失败。
运行完成与方法有效是两个不同条件。需要同时审查 W80、Wmin 删失、逐族 QoR 和端到端 runtime。

相同协议重新运行可复用完整 trial；代码/二进制/参数改变会拒绝复用。
返回码 127、OOM、timeout 不会混成不可布线标签。排查失败、确认所有旧进程已经停止后：

```bash
bash experiments/vtr/tailcompile_v66/run_pipeline.sh recover
# 然后重启失败的 prepare 或 evaluate 阶段
```

recover 可恢复地移动不完整/失败任务到 `_quarantine`，不会删除有效标签。跨主机确认进程停止是操作者责任。
无需清空整批目录。超时预算本身属于 trial 签名；不能暗中改变已有试验的预算。

## 同步轻量结果

`.gitignore` 排除网表、GP TSV、tensor/cache、原始 route/place 和大日志。

```bash
git add -- experiments/vtr/tailcompile_v66/runs/gp_001/cohort.json \
  experiments/vtr/tailcompile_v66/runs/gp_001/protocol.json \
  experiments/vtr/tailcompile_v66/runs/gp_001/plan.json \
  experiments/vtr/tailcompile_v66/runs/gp_001/selection.json \
  experiments/vtr/tailcompile_v66/runs/gp_001/evaluation_report.json \
  experiments/vtr/tailcompile_v66/runs/gp_001/model/training_report.json \
  ':(glob)experiments/vtr/tailcompile_v66/runs/gp_001/**/*complete.json' \
  ':(glob)experiments/vtr/tailcompile_v66/runs/gp_001/**/trial.json' \
  ':(glob)experiments/vtr/tailcompile_v66/runs/gp_001/**/retention.json' \
  ':(glob)experiments/vtr/tailcompile_v66/runs/gp_001/**/pool.json' \
  ':(glob)experiments/vtr/tailcompile_v66/runs/gp_001/**/deployed.selection.json'
git commit -m "Add v6.6 post-GP experiment results"
git push origin main
```

checkpoint 用共享存储/单独传输，不默认塞入 Git。不要推送整个 VTR 工具链或环境目录。
