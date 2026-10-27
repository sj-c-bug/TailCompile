# 有限自然输入资格盘点（2026-10-25）

本轮只执行原有完整 VTR 流程至 ABC，以及新固定 40f8 二进制的 `TC_GP_STRUCTURE_ONLY` 导出。没有执行 GP、packing、placement、routing 或 GNN 效果筛选。两个首选 mcml/ch_intrinsics 与一个预登记备选 stereovision3 全部保留结果。

| 设计族 | 映射／结构状态 | 实际 AP 需求 LUT / FF / RAM / DSP / IO | 重负载门槛 |
|---|---|---|---|
| mcml | 原完整 Yosys/Parmys 在冻结的 120 秒综合预算内超时，未取得 ABC 输出 | 未知 | 未判定，不能填零或称不支持 |
| ch_intrinsics | ABC 与新 AP 结构导出成功 | 153 / 82 / 1 / 0 / 229 | 未达 RAM＋DSP≥16 |
| stereovision3 | ABC 与新 AP 结构导出成功 | 90 / 81 / 0 / 0 / 12 | 未达门槛 |

固定 64×64×2 H0Z0 输入器件每种硬资源容量为 248；ch_intrinsics 的 RAM 需求约 0.40%，属于轻负载。stereovision3 与已有 stereovision0 属同一训练设计族，不能作为新的独立族。

源 RTL 来自固定 40f8 checkout。综合使用已安装的旧 e422 完整工具，具体 commit、Yosys/Parmys/ABC 二进制哈希及原工具未变证据见 `heavy_input_audit.json`。后续解析只使用新 40f8＋v2＋XYZ 后端，未调用旧 VPR。

主语义以 `heavy_input_audit.json` 为准。MCML 的通用 subprocess wrapper 原始分类保留为 `packing_or_placement_failure`，这是 wrapper 对非零退出的默认名称；实际发生的是前端综合超时，未运行 packing/placement。审计已标为 `synthesis_status=timeout`、`structure_compatible=null`，并保留原始 `parmys_timeout.out`。刷新审计时全部命中缓存，没有重复运行。

结论：本次没有找到第二个满足门槛的重硬资源设计。MCML 可在服务器采用事前冻结的较长综合预算完成输入资格；这不是已经准备好的第二重族。保持既定资源门槛，不用低硬资源设计补齐四族 pilot。
