# TailCompile v6.2：与部署动作对齐的置信候选选择

v6.1 的 leave-design-out 候选排序准确率只有 33.9%，训练时看到的是旧的粗候选，部署时却连续执行
cluster swap。v6.2 保留固定 tile、层次 RRG GNN、TailRisk 和 cluster-region matcher，只修正这个
训练—部署错位。

每个设计/宽度生成 `noop + 2×{1%,2%,5%}` 共 7 个候选。四个边界条件
`conv_layer/w120`、`eltwise_layer/w120`、`lenet/w180`、`lstm/w140` 使用 seeds 51/52 采集 56 个
VPR 标签。三个独立初始化模型组成 ensemble；目标设计的全部标签都从其训练集排除。

部署时只有候选相对 no-op 同时满足以下条件才允许移动：排序分数的保守置信下界至少 0.01、预测
可布线概率保守下界不下降、预测线长及 CPD 的上界增幅不超过 1%。否则输出原 `gnn_base.fplace`。
pilot 用全新 seeds 61/62；只有 v6.2 相对 Default 不减少成功数且相对 gnn_base 至少净增 1 个成功，
共同成功样本的四项 QoR 均不恶化超过 3%，并且 Region/Channel P99 至少一项改善，才允许启动全部
12 个条件的 full evaluation。
full 阶段直接合并 pilot 的 24 个 trial，只补跑其余 48 个，不会用同一 seeds 重复计算四个边界条件。

## 分阶段运行

所有实例需访问同一个共享输出目录：

```bash
export TAILCOMPILE_V62_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v62/runs/intervention_001"
export TAILCOMPILE_V62_JOBS=2
```

CPU 候选、标签和训练集：

```bash
nohup bash experiments/vtr/tailcompile_v62/run_pipeline.sh prepare > logs/v62_prepare.log 2>&1 </dev/null &
```

GPU 只训练：

```bash
nohup bash experiments/vtr/tailcompile_v62/run_pipeline.sh train > logs/v62_train.log 2>&1 </dev/null &
```

CPU pilot：

```bash
nohup bash experiments/vtr/tailcompile_v62/run_pipeline.sh pilot > logs/v62_pilot.log 2>&1 </dev/null &
```

检查 `pilot_gate_passed=true` 后，运行全 12 条件：

```bash
nohup bash experiments/vtr/tailcompile_v62/run_pipeline.sh full > logs/v62_full.log 2>&1 </dev/null &
```

任意时刻查看：

```bash
bash experiments/vtr/tailcompile_v62/run_pipeline.sh status
```

标准计数依次为：84 个 pool candidate、56 个标签 trial、28 个训练 example、18 个 checkpoint、
12 个选择结果、24 个 pilot trial、72 个 full trial。大 tensor bundle、checkpoint 和原始 VPR 中间文件
不应提交 Git，只同步 JSON、最终 `.fplace`、manifest、`v62_metrics.json` 和日志。
