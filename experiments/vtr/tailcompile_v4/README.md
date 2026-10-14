# TailCompile v4：2D 候选的供需风险反馈修正

本轮仅修改 2D，不依赖也不改动 `experiments/vtr3d`。固定 VTR
`40f8d19708b0b60e2aa04452deb1cfed68d96b0b`（服务器版本）。已有 checkpoint
保持冻结；本轮不重训、不使用 VPR 标签选择或接受局部搜索动作。

2026-10-14：本地已完成 SHA + attention_layer 的 88 次配对运行，见
[RESULTS_2026-10-14.md](RESULTS_2026-10-14.md)。spmv 的服务器原始 packed net
未同步，本地重建不等价，因此尚未完成其 44 次验证。

## 本轮问题与预先固定的实验

问题：同一起点经 cluster-level 风险修正后，是否改善真实 P&R？
其次判断尾部目标、初始化和计算预算各自的贡献。

- 设计：SHA、attention_layer、spmv。三者均属于开发集；spmv 已参与方法分析，
  不再称为未触碰的最终测试集。
- 宽度：SHA 90/100，attention_layer 110/120，spmv 90/100。
- placement seeds：11、12；扩展稳定性实验再使用 13、14。
- 每个起点、目标和宽度固定最多 512 次合法候选的代理评估，搜索 seed=2027。
- 两个起点：既有 v2 确定性绑定；冻结 width-conditioned checkpoint 的单个候选，
  seed=2027，无噪声、无 VPR best-of-N 筛选。
- 每个起点比较 `base`、`tail`、`mean`、`none`、`random`：未修正、CVaR 风险、
  平均风险、只保留线长/时序代理、等代理评估预算的随机扰动候选筛选。
- 加 Default analytical placement，共 11 方法 × 2 seeds × 2 widths × 3 designs = 132 次配对运行。
  此外每个设计/宽度运行一次独立的供给准备流程，输出实际目标宽度的 RRG。

`random` 从原始起点随机进行 1–8 次移动/交换后评价，不通过累计局部改进形成搜索轨迹。
它与局部搜索使用同一个 tail 代理选最好候选，报告实际评估次数与耗时。
这不是 VPR 等预算 best-of-N 对照；若以后引入多候选 VPR 筛选，必须另加这种对照。

`tail/mean/none` 共用基于当前负载热点的动作优先级，所以 `none` 仅移除目标中的
风险项，不能解读为完全移除 RRG/TailRisk 信息。mean/tail 共用 CVaR 归一化分母，
初始目标幅度并不相同，不能单凭两者差异认定尾部分布的独立因果贡献。

## 新风险的语义

旧网表 TailRisk Profile 和旧 checkpoint 不改定义。这里新增的是
**映射后的 wire-pool 供需风险**，不应混称为原始 TailRisk。

1. 用实际目标 channel width 的原始 RRG，读取 CHANX/CHANY 的方向、segment
   和容量。每条 wire node 只贡献一次容量，多个开关边不会增加供给。
2. 将具有相同有向区域边界 footprint 和 segment 的导线聚成共享 pool。
   长导线跨过多个区域边界时仍是一个 pool，双向导线的两个方向共享容量。
3. 每个网从驱动区域到各接收区域走固定、可复现的粗路径；路径长度优先，
   容量作为微小 tie-break。多接收端取路径并集，在同一 pool 上取一次最大占用份额。
   每个 net 是一单位需求，不把高扇出再任意乘成独立网络数。
4. 一个边界有多个 pool 时按容量比例分摊需求。计算 pool 的 rho=D/C，以及
   所有 pool（包含未使用者）的均值、最大值、上尾 25% CVaR。
5. 初始确定性候选提供所有对照共用的归一化尺度。目标固定为
   `risk/ref_cvar + 0.15*wire/ref_wire + 0.25*timing/ref_timing`。
   `mean` 替换 risk，`none` 将其置零；没有逐方法重新归一化。

时钟和常量沿用 `tailrisk_core` 的剔除规则；普通高扇出网保留并合并共享路径。
时序项沿用布局前单位延迟深度代理，不是实际 STA 或 CPD。
不引入 router occupancy、最终坐标或标签到搜索目标。

这只是供需代理：未完整建模区域内 pin access、switch-only 跨区连接、
具体轨道之间的可达性；区域粗路径存在不保证详细 RRG 路径存在。
零供给不加 epsilon：不存在有向粗路径时拒绝该候选，并明确是代理不可行。
供给不按 channel width 线性外推；不同宽度重新提取实际 RRG。

## 合法性与 lowering

- 保留 packed cluster 原子性，分别检查 CLB/BRAM/DSP 的可用具体 tile anchors。
- 搜索允许在真实剩余容量内移动，不继续锁死 v3 的比例填充配额。
- 支持同类型交换及 `allowed_regions` 限制；IO 固定为 lowerer 的真实外围位置，
  不使用旧 binding 中与实际 IO 坐标不一致的抽象分组。
- 显式 macro 输入会报错，尚未声称覆盖隐式 carry-chain、相对 macro 或任意多高度约束。
  当前架构既有 tile-anchor/cluster 模型及 atom-level lowerer 继续由 VPR 检验。
- 所有比较使用相同 v3 lowerer；区域改变后坐标 head 的相对坐标按新区域解释。
  最终 `.fplace` 与评价器的区域逐个核对，并记录 VPR 后 atom 区域保持率。
- 每次搜索结束完整重算，校验增量评价与最终保存分配一致。

## 服务器运行

先激活已有 torch 环境，设置现有 `VTR_ROOT`、`TAILCOMPILE_REPO`。
脚本不自行 conda activate。

```bash
cd "$TAILCOMPILE_REPO"
python3 experiments/vtr/test_tailcompile_v4.py

export TAILCOMPILE_V4_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v4/runs/pilot_2d_001"
bash experiments/vtr/tailcompile_v4/run_2d.sh
```

仅补齐服务器 spmv（使用服务器原有 `.net`，不重新 pack）：

```bash
export TAILCOMPILE_V4_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v4/runs/server_spmv_001"
bash experiments/vtr/tailcompile_v4/run_2d.sh --designs spmv
```

该子实验验收 expected=observed=44。完整三设计默认任务才是 132。

后台运行时，先确保上述环境变量在当前终端已导出：

```bash
mkdir -p "$TAILCOMPILE_REPO/logs"
nohup bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v4/run_2d.sh" \
  >"$TAILCOMPILE_REPO/logs/v4_2d_001.log" 2>&1 </dev/null &
echo $!
```

阶段也可以分别执行 `--stage supply/build/run/summarize`，均使用同一 `--out`。
每个 VPR 子进程直接流式写入 `vpr.out`，主日志打印阶段、方法和耗时。
完整且输入签名一致的运行可恢复跳过；失败/中断目录不会被当成成功，
输入变化也不会静默复用。排查后保留失败批次，使用新输出目录重跑。

原始 `.net` 必须匹配 bundle 的 packed-net SHA-256；服务器已有标签输入应满足。
脚本只把 `.net` 的副本交给 VPR AP，后者可能重打包，不声称 packing 完全冻结。
也验证 VTR checkout、二进制哈希、架构哈希、网表、候选哈希和各运行配置。

## 验收与同步

验收：`summary.json` 中 expected=observed=132、missing=[]、metric_errors=0、
没有 other_failure。unroutable 是有效标签。配对变化只在同设计/宽度/seed 下计算，
共同成功样本的 QoR 与成功率一起报告，不把 seed 当作独立设计。
当前流程每个宽度重新布局，因此不能将最小成功宽度解释为固定布局布线阈值。

运行成本报告初始化、搜索、lowering、VPR；供给准备是器件/宽度级离线开销，
其独立日志和 manifest 保留。Python 解析/建图及 I/O 是共享准备成本，不能把仅有
search+VPR 的数值误写成完整端到端时间。比较时同时给出整个 campaign 的 wall time。

同步 `config.json`、`plan.json`、`summary.json/csv`、每组 `supply.json`、
`candidates/` 中 JSON 和 `.fplace`、`gnn_initial/build_summary.json`、
初始化计时、每次运行的 `manifest.json/metrics.json/vpr.out`。
原始 RRG、完整 route/net/occupancy、模型副本保留服务器，不纳入 Git。

若风险下降而最终 QoR 无改善，应首先审查代理覆盖和 VPR 区域保持率；
若 mean/none 同样有效，则不能把收益归因于尾部风险。
本轮确认真实改善后，才考虑用 VPR 验证过的改进候选进行排序训练或蒸馏。
