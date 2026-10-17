# v6.1：以最终 .fplace 布线收益为目标

默认只训练一个完整模型，每个设计有一个留出 checkpoint；不运行四套 TailRisk/placement 消融。
在现有六设计上做开发，每折训练排除待生成设计的所有宽度和候选。
这六个设计已被反复分析，因此这是开发验证，不是新的独立设计测试。

## 已实现

1. 固定 tile 尺寸分块（默认 8×8 **tiles/块**，不是总共 8×8 块），节点数随器件面积增加。
   `--tile-size 0` 可读取旧区域。新图从原 IR 的唯一物理根位置重分块，并流式读取实际宽度的 RRG 构造有向共享 wire pools。
2. TailRisk 作为局部特征：均值、最大值、CVaR、X/Y、L4/L16；不是硬编码优化目标。
3. 共享 pool 编码、cluster 与目标区域匹配、局部消息传递、逐层 2×2 聚合和回传；空间状态进入 route、QoR、候选评分头。
4. 同一设计/宽度内 pairwise loss。成功率优先；成功率并列时按 P99、channel P99、wirelength、CPD 的加权 log ratio 排序（0.25/0.15/0.20/0.40）。失败 QoR 不填零。
5. 从固定的 `gnn_base` 出发，同资源 cluster 交换，保持区域资源占用、固定 IO、检查 allowed regions；最多移动 10% 非 IO clusters。
   搜索只看模型分数，预测 wire/CPD 相对初始最多增加 5%。这只是预测保护，最终仍以 VPR 测量为准。
6. 确定性 lowerer 写出 `selected.fplace`。新 seeds 41/42 对比 Default、gnn_base、v61，输出成功率差及共同成功样本的拥塞/线长/CPD 比率。

没有把热点强度叫作移动收益置信度，也没有使用旧 intervention head 控制移动。
本版的改进方向由完整候选的排序分数决定；约束强度由固定移动预算控制。
已有 gnn_base 是固定起点；v6.1 checkpoint 决定局部修改的候选排序。

## 数据与限制

必须在保留完整 v5 档案的服务器运行：需 plan/config、各设计 BLIF/.net/IR、每个宽度 supply_run/device.rr_graph.xml（或 .gz）、候选 lowering/.fplace、每次成功 VPR 的 occupancy 和 manifest/metrics。
训练沿用已有的 72 个候选条件；重新聚合细粒度热点不需要 VPR。闭环对照需 6×2×3×2=72 次新 VPR 运行。

物理位置只计算原 IR 中的 root anchors，不重复计算跨多 tile 的硬块。
目前共享 wire pool 的路径仍是区域跨度近似：没有编码 IPIN/OPIN 的 pin access 和完整开关可达性。方向、L4/L16 和共享池容量保留，但不构成详细可布线证明。
未知 segment 名有独立 pool，模型标记为非 L4/L16，暂未学习任意 segment 类型 embedding。
RiskEvaluator 仍预计算区域最短路，图很大时会成为开销来源；不能据此声称无限扩展。
分块与层次关系只取决于器件几何，不使用 VPR 热点标签。

大图训练按一个 design/width 的六候选累积梯度图。若显存不足，优先调大 tile size（新输出目录），不要修改实验中途已有 bundle。
所有 bundle/checkpoint 都只加载自己生成的可信文件。新增模型 schema 与 v6 不兼容，必须重新训练。

## 服务器分阶段运行

数据准备和 VPR 都是 CPU/内存密集任务，训练才是 GPU 任务。GPU 实例在训练结束后长时间执行 VPR
可能因 GPU 空闲被平台回收，因此正式运行应拆成 CPU → GPU → CPU 三个可恢复阶段。三个实例必须看到
同一个 `TAILCOMPILE_V61_OUT`（共享盘或完成阶段间复制整个输出目录）。不要 source 会自动启动旧任务的脚本。

CPU 实例先准备数据：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export TAILCOMPILE_V61_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v61/runs/tile8_001"
mkdir -p logs
nohup bash experiments/vtr/tailcompile_v61/run_pipeline.sh prepare \
  > logs/v61_prepare.log 2>&1 </dev/null &
echo "PID=$!"
```

GPU 实例只训练。默认明确要求 CUDA；CUDA 不可用会立即失败，不会静默退回 CPU：

```bash
nohup bash experiments/vtr/tailcompile_v61/run_pipeline.sh train \
  > logs/v61_train.log 2>&1 </dev/null &
echo "PID=$!"
```

训练日志出现 `GPU train complete` 后即可释放 GPU 实例。回到 CPU 实例生成候选并续跑 VPR：

```bash
nohup bash experiments/vtr/tailcompile_v61/run_pipeline.sh evaluate \
  > logs/v61_evaluate.log 2>&1 </dev/null &
echo "PID=$!"
```

原来的无参数命令仍可在不会因 GPU 空闲而回收的机器上串行执行全部阶段。各阶段具有已有的
签名和 manifest 复用检查；中断后重跑同一阶段不会重做已完整的 checkpoint、候选或 VPR trial。

随时检查状态：

```bash
python3 experiments/vtr/check_tailcompile_v61.py --out "$TAILCOMPILE_V61_OUT"
```

最终 `accepted=true` 才算全部完成。标准六设计实验还应显示 6 个 checkpoint、12 个
`selected.fplace`、72 个唯一 trial，且状态仅为 `success` 或 `unroutable`；最后结果文件必须有
`complete=true` 和 comparisons。`unroutable` 是有效实验标签，不是程序故障。

完成条件：`candidates/vpr_results.json` 中 complete=true；6 个 leave-design-out checkpoints；12 个 selected.fplace；72 个配对 trial（未指定子集时）。
模型排名提升只是诊断。是否成功以 v61 相对 Default/gnn_base 的实测成功率、拥塞、线长、CPD 为准。
失败/超时保留目录由已有 VPR runner 处理，不伪装为不可布线。

需要同步：data/prepare_report.json、models/*.json、candidates/generation_report.json、candidates/vpr_results.json、各设计/宽度 selection.json、selected.fplace、selected_binding.json、selected_lowering.json，以及运行日志。
无需上传大型 bundle.pt、原始 RRG、训练 checkpoint，除非需要进一步诊断。

## 本地验证记录

`test_tailcompile_v61.py` 的 5 个测试覆盖可变尺寸层次、空间梯度与匹配影响、排序目标、tile 根位置守恒、训练/checkpoint/report 复用、同资源交换；v6 原有 7 个测试也通过。
真实 SHA 烟雾记录在 `local_smoke/smoke_report.json`：185 clusters、9 区域、52 shared pools，9→4→1 层次，9 区域真实 occupancy 标签，生成与恢复通过。
这个烟雾测试使用随机模型，不能当成布线收益证据。本地缺少六设计的完整物理输入，正式训练和 72 次新 VPR 评测需在服务器完成。
