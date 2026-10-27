# DSP 模式需求修正后的双层部署（2026-10-25）

`interface_passed=true`。复用原 epoch20 checkpoint，未重新训练；仅以新的 `v7-dsp-mode-equivalent-2` 数据语义重新构图。原 [v1 失败证据](../v7_forced_layer_smoke_20261025/README.md) 保留。

硬资源物理数量未变：每层 1 个 DSP tile、1 个 RAM。每 DSP tile 的小模式单位容量仍为 2。XML 允许 2 个 18×19 乘法器或 1 个 27×27 乘法器，故本例两个全宽乘法器的需求从各 1 修正为各 2；总 DSP demand=4，各层 capacity=2。换算依据是 mapped BLIF 明确端口位和同一 XML 的模式、num_pb，无容量压缩或模型调参。

| 方法 | 初始 DSP | 初始 RAM | 最终 DSP/RAM | P&R | CHANZ 节点数 |
| --- | --- | --- | --- | --- | --- |
| Default | 原生 | 原生 | 各层各 1 | success | 37 |
| capacity_only | 各层各 1 | 各层各 1 | 各层各 1 | success | 36 |
| GNN | 各层各 1 | 各层各 1 | 各层各 1 | success | 38 |

GNN 的非零层初值被消费，随后继续原生 GP，并实际跨层布线。运行前后源码哈希相同。相较 v1，本次不再依赖后端把同层的两个全宽 DSP 初值重新分层。

此测试只验证模式占用与实际部署链路。原 checkpoint 基于旧需求单位，复用只用于工程检查，不构成新训练语义下的性能或泛化证据。正式训练、教师投影与图缓存需采用新语义，不能把旧图混入新实验。其他 packing、宏和模式共享约束仍由 VPR 验证。

`data.py` SHA256：`28698f8024e86e9a6a8bc424ac56d72055ee9beecb94e4303a1a95e29b8ec28c`。相关 DSP 单/双宽、未知模式拒绝与模型/训练回归共 28 项测试通过。未运行性能验收。

原始输出：`/home/phoenix/vtr_work/tailcompile_3d/v7_forced_layer_smoke_20261025_v2`。本目录保存轻量协议、资源清单、完整指标和三份最终 `.place`。
