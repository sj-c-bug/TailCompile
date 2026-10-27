# 3D 四格架构与真实 RRG 审计

日期：2026-10-24（Asia/Shanghai）。状态：**架构/图接口资格通过；本报告没有 GNN 性能结论。**

本轮实现了 `H0Z0/H0Z1/H1Z0/H1Z1`：层资源对称/不对称 × 垂直连接分散/集中。固定 VTR 的 `sg_location type="XY_SPECIFIED"` 支持精确 XY 实例化，因此无需为位置对照修改 VTR 的路由器。

源码：[架构生成](../vtr/tailcompile_v7_3d/architectures.py)、[分层器件图](../vtr/tailcompile_v7_3d/device_graph.py)、[有限实测入口](../vtr/tailcompile_v7_3d/probe_architectures.py)。

## 固定输入与构造

- VTR 基础 commit：`40f8d19708b0b60e2aa04452deb1cfed68d96b0b`。
- 本次图导出使用已验收的独立 heterogeneity v2 binary；SHA-256：`05bb1e7090adeb281aea47477b3cc87f5048080a8ccfc7e17bb88c7491b84788`。它只执行 `--pack --write_rr_graph`，没有运行 GP、placement 或 routing；新 XYZ warm-start 后端属于独立资格任务。
- 原始 XML：固定版本 `multi_die/hecate/3d/hecate_3d_sb_10um_bidir_fanin_16_fanout_16_with_esd_7nm.xml`。
- 实际 grid：12×12×2；每层每 4×4 个物理 tile 一叶，边界不拉伸，共 18 个叶区域。器件图具有层次索引、真实跨层边和共享 wire-pool incidence。
- H0：每层 DSP 20、BRAM 20；H1：下层 DSP 30/BRAM 10，上层 DSP 10/BRAM 30。两者硬资源列坐标、总各型资源、CLB/IO、primitive/switch/segment 定义一致。
- Z0：内部奇数坐标规则网格，共 25 个 XY 位置；Z1：同样 25 个按几何规则选择的中心位置。每位置实例化 6 个相同双向 Z 资源，未参考网表热点或路由结果。

这里的 D=100% 是冻结的参考位置预算；不表示使用原始 XML 的全部可用位置。直接替换原 `EVERYWHERE`/硬列增量定义时必须记录此含义，不能与旧 pilot 的 Z 总量混作同一架构。

## 实际图结果

最终资格为四架构×W=64/128，共 8 份真实 RRG。所有结果来自实际 RR node/edge，而非 XML 声明计数。

| 项目 | H0Z0 / H1Z0 | H0Z1 / H1Z1 |
|---|---:|---:|
| 独立、具有非自环入/出边的 CHANZ 容量 | 150 | 150 |
| CHANZ 类型 | `BI_DIR / z_segment` | 同左 |
| 孤立 CHANZ 排除数 | 0 | 0 |
| 实际 Z 坐标的容量加权平方半径 | 16.50 | 4.02 |
| CHANZ 邻接 `L4_driver` 边数 | 4,800 | 4,800 |
| CHANZ 邻接 `bidir_inter_die_switch_10um_with_esd` 边数 | 4,800 | 4,800 |

平方半径以器件中心为原点、单位为 tile²；它核实集中位置确实进入了 RR graph。开关边数作为连接审计单列，不用于增加 wire 容量。双向弧和跨多个区域的同一 RR node 仍只贡献一份共享容量。

四格的全局实际资源均为 CLB 120、DSP 40、BRAM 40、IO subtile 640。IO 原语容量与 root 数量分账，避免将八个 IO subtiles 再乘一次 tile 容量。

宽度从 64 改为 128 时，各图的 CHANX、CHANY 节点数均由 4,284 增至 8,604；CHANZ 容量维持 150、位置和类别不变。因此在本 grid 和这两个已测宽度上，Z 总供给与横向 channel width 解耦。其他 grid/宽度必须重新审计，不能线性外推。

## 数据接口、证据及复现

`build_device_graph(rr_graph, architecture, tile_size=4)` 输出纯 JSON/list：

- `region_x`（16 个有名称的特征）、`region_src/dst/weight`、归一化 `region_bounds`、`region_layer`。
- 每层每 4×4 tile 的叶区域及 `parents`；五维 `capacity` 和 `layer_capacity`，顺序为 LUT、FF、memory、DSP、IO。
- 8 维 `pool_x`、`pool_ids/pool_regions`、共享池及实际 RRG/provenance/audit。

容量是架构最大互斥模式的 AP 需求上界代理，不能证明 packing 合法。旧 streaming parser 在单物理 tile 粒度提取 wire，再收缩为固定 tile 尺寸区域；没有直接复用“全器件均分 4×4 bins”的旧粒度。

轻量证据：[完整审计 JSON](runs/v7_hz_audit_20261024_final/architecture_audit.json)。同目录含四份 XML/构造元数据、8 份图/运行 manifest/日志及源码快照，约 2.1 MB。WSL 原工作目录为 `/home/phoenix/vtr_work/tailcompile_3d/v7_hz_probe_20261024_final`。

2026-10-25 在确认下游只读取图 JSON、架构与 BLIF 后，逐文件校验并回收三套原型的 raw RRG/派生 pack net，共 751,829,700 bytes；输入、图、审计和日志保留。详见 [清理依赖账本](runs/v7_hz_audit_20261024_final/prototype_payload_cleanup.json)。需要重新分析原始 RR 边时按已记录命令重建，不声称 raw RRG 仍保留；已完成的图缓存复用不需要它。

```bash
cd /mnt/c/Users/phoenix/Documents/ChatGPT/DAC2027/TailCompile_latest/experiments/vtr/tailcompile_v7_3d
python3 -m unittest test_device_graph -v
python3 probe_architectures.py \
  --vtr-root /home/phoenix/tools/vtr-40f8d197-tailcompile2d \
  --binary /home/phoenix/vtr_work/tailcompile_3d/heterogeneity_v2_build_20261022/vpr \
  --out /home/phoenix/vtr_work/tailcompile_3d/v7_hz_reproduce_001 \
  --width 12 --height 12 --channels 64 128 --jobs 2
```

三个单测通过：共享/双向容量不重复计数、部分边界叶和 IO root 计数、实际四格预算不匹配拒绝与 width 耦合报告。开发过程中另有两个各 8 图的早期导出，分别补充了位置与实际开关审计，保留在相邻的首轮/r2 目录；最终证据使用本报告链接，不把这 24 次 pack-only 图导出计作完整 P&R。

复现使用新的输出目录。运行后增加了服务器仓库布局的路径兼容逻辑，原已归档源码快照和哈希保留；不能用改变后的实现签名覆盖旧运行目录。

本轮边界：active multi-height tile/macro 仍显式拒绝；pool footprint 不包含完整 pin/switch 端到端可达证明。后续自然设计每个实际 grid/width 按相同审核规则生成数据，再进入学习及三组 P&R 比较。
