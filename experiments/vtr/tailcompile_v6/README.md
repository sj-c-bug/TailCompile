# TailCompile v6：早预测与粗放置增益的严格对照

## 这一步回答什么

v6 不预设“粗放置一定必要”，也不再要求 TailRisk 单独完成拥塞预测。它在完全相同的样本、标签、留设计划分、参数量和训练预算下比较四组模型：

| 模型 | 网表 + RRG | TailRisk 特征 | 候选粗放置 |
|---|---:|---:|---:|
| `early_no_tail` | 是 | 否 | 否 |
| `early_with_tail` | 是 | 是 | 否 |
| `placement_no_tail` | 是 | 否 | 是 |
| `placement_with_tail` | 是 | 是 | 是 |

这同时分离两个增量：TailRisk 是否提供额外信息；候选粗放置是否值得牺牲“布局前”时机。

## 输入、监督与空间风险

- 逻辑端：packed cluster 图、资源类型、规模、度、关键度和局部连接尾部。
- 物理端：4×4 区域 RRG 图、资源容量、方向 escape supply、segment supply 和实际 channel width。
- placement 模式额外输入：cluster→region 分配。逻辑超边在 region RRG 上进行确定性粗路由，得到每个有向、segment-aware supply pool 的 `demand/capacity`，再投影为区域风险场。
- 标签：多个 placement seed 聚合后的成功概率、成功运行 QoR 中位数，以及 VPR `chanx/chany_occupancy` 对齐到 4×4 区域后的真实拥塞图。
- placement 模型还输出 cluster 级“值得干预”置信度，以该 cluster 所在区域的真实热点强度作弱监督。生成约束时先过 `release_mask`：默认置信度不足 0.70 的 cluster 释放，并且最多约束 25%，防止再次演变成全局强制预绑定。

`TailRisk off` 并非删掉全部负载信息：平均需求/容量、线长和几何仍保留，只屏蔽 P95/CVaR/max/Gini/top-share 等尾部字段，因此它是“尾部统计的增量消融”。

目前 VPR occupancy 直接提供区域和 X/Y 方向标签，但不直接标注 segment type；segment 信息存在于预测侧的 RRG supply pool。若 v6 显示空间定位有效，下一步再用 route trace + RR node 映射增加真实 segment 标签，避免本轮一次引入两个不确定因素。

## 防止数据泄漏

- seed 只作为重复试验聚合，不进入模型。
- 按设计族做 leave-one-design-out；一个设计的任何 width、method、seed 都不会同时出现在训练和该折测试中。
- 四组模型使用同一组候选与标签。
- 空间代理只使用网表、RRG 和候选粗放置，不读取 VPR occupancy 或 QoR。

## 判定规则（运行前固定）

满足任一条件，就认为粗放置值得增加预测延迟：

1. 留设计 Brier score 相对改善至少 10%；
2. 候选排序 Spearman 增益至少 0.15，且 placement 模型达到至少 0.30。

若 early 模型 `Brier >= 0.20` 或 `AUC <= 0.60`，标记为“纯布局前预测不足”。这些阈值写在代码中，不能看完结果后修改。

## 运行

```bash
cd "$TAILCOMPILE_REPO"
source env.sh

export TAILCOMPILE_V6_SOURCE="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v5/runs/holdout_001/confirm"
export TAILCOMPILE_V6_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v6/runs/information_ablation_001"
export TAILCOMPILE_V6_DEVICE=auto
export TAILCOMPILE_V6_EPOCHS=60

mkdir -p "$TAILCOMPILE_REPO/logs"
stamp="$(date +%Y%m%d_%H%M%S)"
log="$TAILCOMPILE_REPO/logs/tailcompile_v6_${stamp}.log"

nohup bash experiments/vtr/tailcompile_v6/run_pipeline.sh \
  >"$log" 2>&1 </dev/null &

echo "PID=$!"
echo "LOG=$log"
```

## 验收

```bash
test -s "$TAILCOMPILE_V6_OUT/dataset/dataset.json"
test -s "$TAILCOMPILE_V6_OUT/dataset/dataset_summary.json"
test -s "$TAILCOMPILE_V6_OUT/models/evaluation_report.json"

python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['TAILCOMPILE_V6_OUT']) / 'models/evaluation_report.json'
r = json.loads(p.read_text())
print(json.dumps(r['decision'], indent=2))
print('variants:', sorted(r['variants']))
print('folds:', {k: len(v['folds']) for k, v in r['variants'].items()})
PY
```

六个设计时，每个 variant 应有 6 个 held-out folds；最终结论以 `decision` 为准，而不是训练 loss。
