# 强制双层部署工程测试（2026-10-25）

本测试复用既有 epoch20 checkpoint，未重新训练、未调参。用途是确认 GNN 的非零层初值被实际消费，并经过原生 GP、APPack、详细布局和真实跨层布线；不构成学习到 3D 性能优势的证据。

使用 legacy `single_site`：10×10、两层，各层物理资源为 62 CLB、1 DSP tile、1 RAM tile，垂直路由保持原 Hecate 定义。设计包含两个全宽乘法器和两个 RAM。固定 width=320、seed=701，共执行 Default、capacity_only、GNN 三次部署 P&R。

| 方法 | P&R | 初值 RAM 层分布 | 初值 DSP 层分布 | 实际 CHANZ 节点数 |
| --- | --- | --- | --- | --- |
| Default | success | 原生 | 原生 | 37 |
| capacity_only | success | L0:1 / L1:1 | L0:1 / L1:1 | 36 |
| GNN | success | L0:1 / L1:1 | L0:2 / L1:0 | 38 |

GNN 初值包含非零 layer，日志确认消费初值后继续执行原生优化。其最终 `gnn.place` 中：RAM `q0_0=(5,3,L0)`、`q1_0=(6,6,L1)`；DSP `p0_0=(2,2,L0)`、`p1_0=(3,7,L1)`。后端将初始同层的两个 DSP 修复为每层一个，随后成功跨层布线。

`report.json` 的 `interface_passed=false` 保留更严格的初值检查结果：GNN 没有在初值中将 DSP 和 RAM **分别都**拆成两层各一个。原因是当前 5 维容量使用 XML 最大 mode 上界，每个 DSP tile 的代理容量为 2，而该例的全宽乘法器实际只能每 tile 容纳一个。这说明该上界不保证 mode 兼容的打包容量；不能把代理容量检查当成最终 packing 合法性保证。本次未为了通过此额外检查修改容量或模型。

因此，本测试确认了非零层初始化、原生后端继续及真实跨层 route 的工程链路，也暴露了模式相关容量的边界。未评估 runtime 验收、泛化或相对性能；三次秒级耗时不能用于性能结论。

轻量归档包括运行协议、资源清单、完整指标报告和三份最终 `.place`。RRG、图、初始化、日志与原始输出保存在 `/home/phoenix/vtr_work/tailcompile_3d/v7_forced_layer_smoke_20261025`。源码哈希在运行前后保持一致。
