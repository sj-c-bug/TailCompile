# DSP mode-aware demand（v7-dsp-mode-equivalent-2）

仅修改新的 v7 数据适配层，不修改冻结的 2D 模型、DSP 物理容量或后端 packing。

XML DSP 逻辑根的最大互斥模式原语数量定义 `units_per_root`，与现有 `device_graph.resource_units()` 一致。遍历 `pb_type` 的 `num_pb` 和互斥 `mode`，为每种 primitive model、端口方向和宽度计算该根中最大并行份数。对于 mapped BLIF 实例，从明确列出的端口位下标推导需求宽度；常量、重复输入信号仍占据相应端口位。设可容纳该实例的 XML 模式最大并行数为 `k`，则 DSP 需求为 `units_per_root / k`。

在 Hecate 中，根容量保持 2：27×27→54 的乘法 `k=1`，需求为 2；18×19→37 的乘法 `k=2`，需求为 1。一个 APBlock/molecule 中包含多个 DSP 实例时累加需求，不再将所有 DSP 原语压为布尔值。RAM 的位切片分组规则保持原有定义。

适配器用输出 net 名或明确 `.cname` 精确对应 AP snapshot 的 atom 名，保留每个 DSP 的模型、实际端口宽度、并行数和换算需求。未知模型、未知端口、超出 XML 的宽度、稀疏位下标、重复 pin、歧义输出名、无法对应的 atom，以及 DSP 与其他逻辑根的 equivalent-site 混合会明确拒绝。不会按信号名字或模型名字猜测宽度，也不交换乘法输入端口来假设交换律。

此规则是模式占用约束。它不是一般的 packing、端口共享、跨根异构等价、宏或多高度合法性证明。XML 中非同质模式的共享限制仍由原生 VPR 验证；不能将五维聚合容量视为完整打包证书。

`build_logic()` 新增 `demand_semantics`、`dsp_mode_catalog`，以及 cluster 的 `dsp_mode_audit`。graph audit 和缓存 signature 显式保存版本，signature 还包含当前 `data.py` 源码哈希。旧缓存会拒绝复用，需在新的输出根重新构造图、教师投影、训练与标签；不要混合旧的 DSP 原语计数和新的小模式单位。heavy/light 设计分组继续按物理 hard APBlock 个数计数，不能对 mode-equivalent demand 求和代替物理计数。

原 checkpoint 只能在本次工程 smoke 中复用来检查接口与容量分配，不能把变化后的输入特征与旧训练语义视为一致，或据此声称研究性能收益。后续正式训练需基于新版本图重新开始。
