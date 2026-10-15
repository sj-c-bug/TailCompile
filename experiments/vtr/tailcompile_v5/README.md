# TailCompile v5：按布线异质性分层的 2D 独立验证

超时恢复：v5 的筛选、supply 和确认任务首次使用原来的超时预算；只有返回码 124 的任务自动归档至同级 `_timeout_history/`，然后以 `TAILCOMPILE_VPR_RETRY_TIMEOUT`（默认 7200 秒）从头重试。已完成的 success/unroutable 任务按原签名验证复用。延长预算后再次超时会停止，不能将超时作为不可布线标签。归档中的 manifest 属于历史尝试，统计当前结果时应排除 `_timeout_history`。

## 目标

这一阶段不检验“TailRisk 是否对所有设计都更好”，而检验一个更准确、可证伪的条件性假设：

> 在未参与方法开发的设计族中，预布线连接压力越不均匀，TailRisk 相对平均压力预绑定的收益是否越明显；低异质性设计作为负对照，其收益可以很小或为零。

`sha`、`attention_layer` 和 `spmv` 已用于开发，因此被输入准备程序硬编码排除。默认留出集是七个 Koios 设计族：`bnn`、`conv_layer`、`eltwise_layer`、`lenet`、`lstm`、`robot_rl` 和 `tiny_darknet_like`。

## 冻结的实验定义

- VTR commit：`40f8d19708b0b60e2aa04452deb1cfed68d96b0b`
- 架构：`k6FracN10LB_mem20K_complexDSP_customSB_22nm.xml`
- 物理区域：4×4
- 筛选 seeds：11、12，仅用于 Default 的通道宽度边界扫描
- 确认 seeds：21–28，与筛选 seeds 不重叠
- 每个设计：选择“最大的非全成功宽度”和“最小的全成功宽度”
- 方法：Default、det-base、det-mean、det-tail、gnn-base、gnn-mean、gnn-tail
- AP full legalizer：所有方法统一使用 `appack`；`flat-recon` 在 conv_layer 上会重建出超过器件 DSP 容量的 clusters
- 搜索预算：512 次有效代理目标计算，搜索 seed 固定为 2027
- 公平性：Mean 使用 `mean / initial_mean`，Tail 使用 `CVaR / initial_CVaR`；两者在同一初始解上的风险项幅度都为 1
- GNN checkpoint、候选、RRG supply、配置和输入均记录 SHA-256；确认阶段不再训练或调参

如果七个设计都找到边界，确认阶段共有 `7 × 2 × 7 × 8 = 784` 次 VPR 运行。部分设计若在预设宽度范围内没有失败—成功边界，会被标记为 censored；少于三个未删失设计时流程停止，而不会临时挑选“好看”的样本。

## 防止标签泄漏

流程严格按以下顺序执行：

1. 对未见设计生成并锁定 BLIF、packed net 和 4×4 IR。
2. 只运行 Default，并用 seeds 11/12 找到两个确认宽度。
3. 在这些宽度上根据实际 RRG 构造候选；用于导出 RRG 的运行仍使用已排除的筛选 seed。
4. 在任何确认方法运行前，根据确定性 base 候选的 `CVaR / mean` 锁定 low/medium/high 三档。
5. 使用 seeds 21–28 运行冻结的七种方法。
6. 比较 `gnn-tail` 对 `gnn-mean` 和 `gnn-base` 的配对收益。

异质性分层只使用布线前代理，不使用任何确认阶段的 VPR 成败、拥塞、线长或 CPD。

## 一键运行

先拉取代码并加载只包含环境变量的 `env.sh`：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
source env.sh

export TAILCOMPILE_V5_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001"
export TAILCOMPILE_VPR_JOBS=2
mkdir -p "$TAILCOMPILE_REPO/logs"
timestamp="$(date +%Y%m%d_%H%M%S)"
log="$TAILCOMPILE_REPO/logs/tailcompile_v5_${timestamp}.log"

nohup bash experiments/vtr/tailcompile_v5/run_pipeline.sh \
  >"$log" 2>&1 </dev/null &
echo "PID=$!"
echo "LOG=$log"
```

可用 `tail -f "$log"` 查看进度。脚本具有阶段级可恢复性；如果出现 `other_failure` 或输入签名变化，会停止并保留现场，不会静默覆盖。

`TAILCOMPILE_VPR_JOBS` 控制边界扫描和确认阶段同时运行的 VPR 进程数，默认值为 1。建议在 80 GiB 限额的平台上从 2 开始；观察 `memory.current`、`memory.events` 和 CPU 利用率后再尝试 4。RRG XML 解析和候选准备仍保持串行，避免大型设计再次触发 OOM。并行度不进入算法定义，也不改变 seed、候选或 VPR 参数。

## 验收

运行结束后应存在：

```text
<campaign>/inputs/holdout_inputs.json
<campaign>/screen/screen_summary.json
<campaign>/screen/confirmatory_config.json
<campaign>/confirm/heterogeneity_lock.json
<campaign>/confirm/plan.json
<campaign>/confirm/summary.json
<campaign>/confirm/confirmatory_analysis.json
<campaign>/confirm/confirmatory_analysis.csv
<campaign>/confirm/confirmatory_analysis_paired_rows.json
```

快速检查：

```bash
python3 - <<'PY'
import json, os
from pathlib import Path

root = Path(os.environ['TAILCOMPILE_V5_OUT']) / 'confirm'
s = json.loads((root / 'summary.json').read_text())
a = json.loads((root / 'confirmatory_analysis.json').read_text())
print('expected/observed:', s['expected'], s['observed'])
print('missing:', len(s['missing']), 'metric_errors:', s['metric_errors'])
print('status_counts:', s['status_counts'])
print('design_count:', a['design_count'])
print('verdict:', a['verdict'])
PY
```

完整性的硬要求是 `expected == observed`、`missing == []`、`metric_errors == 0` 且没有 `other_failure`。`unroutable` 是通道边界实验中的有效标签，不是程序错误。

## 统计判定与结论边界

主比较是 `gnn-tail` 对 `gnn-mean`，`det-tail` 对 `det-mean` 是更换初始化方式后的复现。两种 Tail 还分别与其 Base 和 Default 配对，以区分风险聚合、局部搜索和初始化器的贡献。报告：

- 每个设计的配对可布线成功率增益；
- 两者都成功时，区域 P99、通道 P99、线长和 CPD 的百分比改善；
- 各异质性档位的设计级均值和 bootstrap 95% 区间；
- 异质性与设计级收益之间的 Spearman 相关。

预设的最小支持条件是：高异质性档的 Tail 不降低可布线率、降低区域 P99，且“异质性—区域 P99 收益”的 Spearman 系数为正。即使满足，也只能声称 TailRisk 在未见的高异质性设计中具有条件性收益，不能声称它跨所有设计普遍优于简单预绑定。
