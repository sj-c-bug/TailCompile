# v6.7：GNN-only 对 Default，分项 runtime

这是独立的新对照脚本，不改变 v6.6 已冻结的代码/协议/输出。
优先回答原始问题：**不做局部修改，GNN 生成的预布局交给 VPR，能否改善 Wmin 或耗时？**

## 方法和范围

- Default：VPR analytical placement → appack → annealer → route。
- GNN-only：原始双塔生成器一次前向 → 固定 partition prior → min-cost flow 容量分配 →
  `.fplace` → VPR 原生 `--read_flat_place` → appack → annealer → route。
- 不进行候选扰动、局部交换、模拟退火搜索、候选选择或事后选最优。
- checkpoint 默认是原有 `model_width_conditioned/ranker.pt`；虽然名字叫 ranker，
  这是输出 cluster→region 分数和相对坐标的原始生成模型。不是 v6.x 只能评分的模型。
- 脚本拒绝 checkpoint 中出现过的训练/验证设计；默认六个 v5 设计都不属于该 checkpoint
  的 sha/attention_layer 训练及 spmv 验证集。但这六个设计曾用于项目开发，**不算全新独立验证**。
- 复用服务器现有 v5 原始 4×4 IR、映射 BLIF、packed net，不误用 tile8 评分器 IR。
  使用同一设计、尺寸、宽度、placement seed 和后续 VPR 参数配对。
- 默认每个设计取旧 Default-only screen 的首个成功宽度 ±30、步长10，七档宽度；
  新 seeds 251–255。每个宽度生成自己的 width-conditioned `.fplace`，每个 seed 复用它。
  Wmin 是同一算法在不同给定宽度下的网格最小成功值，不是固定坐标的纯 route-only Wmin。
- 六个设计共 420 次主扫描，另做 420 次串行配对计时。主扫描并行；正式 runtime 不用并行扫描的墙钟时间。
- 输出逐 seed 网格 Wmin、删失/非单调性、W80、共同成功条件下的线长/CPD 和分口径 speedup。

原生 `.fplace` 入口会跳过 GP，并改变 RAM grouping 路径。因此本轮测量的是 GNN→VPR
**整条接入流程**的效果，不能把差异全部归因于坐标质量。v6.6 的 post-GP 回放/修正仍是独立诊断。

## Runtime 口径

每个 design/width 的 `generation.json` 记录实际分项时间。CUDA（可选）前后同步，
避免只测异步调用提交耗时。默认 CPU 推理，VPR 始终 CPU。

| 环节 | 主表 warm_inference_lowering | warm_operational | cold_from_prepared_ir |
|---|---:|---:|---:|
| checkpoint 加载和模型初始化 | 不计，假设驻留 | 不计，假设驻留 | 计 |
| 已有 device IR 加载 | 不计，假设缓存 | 不计，假设缓存 | 计 |
| RRG tower＋region projection | 不计，假设缓存 | 不计，假设缓存 | 计 |
| 当前逻辑 IR 加载、图特征准备 | 不计，假设输入已准备 | 计 | 计 |
| tensor transfer | 计 | 计 | 计 |
| 逻辑 GNN 前向、匹配、坐标 head、prior | 计 | 计 | 计 |
| min-cost flow 容量分配 | 计 | 计 | 计 |
| `.fplace` lowering 和写文件 | 计 | 计 | 计 |
| VPR 全部后续耗时 | 计 | 计 | 计 |

原始 RRG→device IR、原始 RTL→packing、Python 进程启动未在本轮重新测量，记录为 null。
**cold_from_prepared_ir 只是“已有 IR 的缓存未命中”口径，不能称作从原始输入开始的完整冷启动。**
当前 graph_inputs 混合构造逻辑和器件图，无法准确分摊；warm_operational 保守地计入整段，
不会随意扣减其中推测属于 RRG 的部分。

主表遵循用户要求，仅计推理和生成再加 VPR；这是一种服务已有缓存/输入图的条件口径，
不是对首次处理新网表的无条件端到端加速声明。请同时看 warm_operational，防止把设计特有成本隐藏。

缓存键包含 checkpoint hash、device IR hash、channel width（模型有 width conditioning 时）、
计算设备。更换器件尺寸、RRG/容量特征、权重或宽度不能盲目复用 embedding。
默认本轮只做内存内的真实缓存；主表允许服务端缓存已准备好的反事实 warm 口径，
`rrg_cache_hit` 另行记录实际是否命中。没有声称每个首次生成任务都实际命中了缓存。

speedup = Default VPR wall time / (GNN 分口径成本 + GNN 后续 VPR wall time)，大于1为更快。
只统计双方均成功的同宽度、同 seed 条件；快速失败不算加速。
生成成本按每次独立部署加计，不把一次离线生成摊薄到五个评测 seeds。
RRG 一次性成本也不会默默归零；其可测分项单列，未测分项标未知。

## 运行

不需要重新训练，也不需要先跑 v6.6。使用原先可运行 torch 和 VPR 的环境。
VTR 基础 commit 仍为 `40f8d197...`；可用干净源码二进制，或安装了精确 v6.6 hook 的二进制。
本脚本会清除 TC_GP_* 环境变量，保证 hook 不参与此次流程；同一批次冻结二进制 hash。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
python3 experiments/vtr/tailcompile_v67/test_direct.py

export TAILCOMPILE_V67_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v67/runs/direct_001"
export TAILCOMPILE_V67_JOBS=8
export TAILCOMPILE_V67_DEVICE=cpu
mkdir -p logs

nohup bash experiments/vtr/tailcompile_v67/run_pipeline.sh all \
  > "logs/v67_direct_$(date +%Y%m%d_%H%M%S).log" 2>&1 </dev/null &
echo "PID=$!"
```

默认一台 CPU 实例即可完成。GPU 推理另设 DEVICE=cuda，先在 GPU 实例跑 `generate`，
然后到 CPU 实例跑 `evaluate`（不重新生成）。这里不涉及 GPU 训练。
请不要在正式串行 timing 阶段同时运行其他 VPR 重任务。该阶段故意不使用 JOBS。

可选环境变量：`TAILCOMPILE_GENERATOR_CHECKPOINT`、`TAILCOMPILE_V67_INPUTS`、
`TAILCOMPILE_V67_SCREEN`、`TAILCOMPILE_V67_TIMEOUT`。inputs 是现有 holdout_inputs.json。
扩展到其他设计需先提供匹配的 4×4 IR/网表，不会自动把 v6.6 的 20 个设计宣称为已准备完毕。

```bash
bash experiments/vtr/tailcompile_v67/run_pipeline.sh status
```

默认验收：trials=420、timing_trials=420、complete=true，且生成 evaluation_report.json。
完成任务不等于结果有收益；必须查看 W80/Wmin、删失、QoR 和对应计时口径。
完整任务可复用；中断/超时/环境失败保留原始日志并拒绝当成布线失败。不要清空整批有效结果。

## Git 同步

```bash
git add -- experiments/vtr/tailcompile_v67/runs/direct_001/signature.json \
  experiments/vtr/tailcompile_v67/runs/direct_001/plan.json \
  experiments/vtr/tailcompile_v67/runs/direct_001/vpr_identity.json \
  experiments/vtr/tailcompile_v67/runs/direct_001/evaluation_report.json \
  ':(glob)experiments/vtr/tailcompile_v67/runs/direct_001/**/generation.json' \
  ':(glob)experiments/vtr/tailcompile_v67/runs/direct_001/**/trial.json' \
  ':(glob)experiments/vtr/tailcompile_v67/runs/direct_001/**/manifest.json'
git commit -m "Add direct GNN placement and decomposed runtime results"
git push origin main
```
