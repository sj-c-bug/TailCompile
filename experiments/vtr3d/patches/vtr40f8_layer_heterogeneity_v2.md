# VTR 40f8 layer-heterogeneity v2

## 适用范围

- 固定上游 commit：`40f8d19708b0b60e2aa04452deb1cfed68d96b0b`。
- 各层的 tile 均由同一个架构 XML 定义，沿用 VPR 现有 logical/physical tile 兼容关系。
- 允许各层同类资源的列、行、数量及稀疏布局不同，也允许某类资源只出现在部分层。
- 不引入跨层不同原语的映射，不改架构、RRG、器件资源类型、文件格式、时序代价或布线器。
- 此补丁包含 v1 修复，直接应用于未修改的固定 commit；不要叠加 v1。

## 修复点

1. 源层使用 exact compressed lookup；其他有资源的目标层仅为搜索中心作 approximate lookup。无资源层显式保留 `UNDEFINED`。
2. uniform、median、centroid 向共享搜索器传入真实源层的位置，而不是目标层投影。共享搜索器将源点还原成物理坐标，与候选稀疏网格中的真实 `(x,y,layer)` 比较；不同层具有相同 compressed index 不再被误当同址。
3. 所有搜索均覆盖闭区间的 `xmax-xmin+1` 列。修复 uniform/centroid 及初始化的单列范围完全不搜索、最后一列可能未搜索的问题；median 原本已使用 `+1`。
4. median/centroid 对无相容资源的目标层返回失败，交由正常 move-abort 路径处理；共享搜索器也守卫无效层、空资源层、空或越界范围，避免负索引和非法随机范围。
5. sparse column 的 y 范围扩张只影响当前列，不能污染后续 dense column；固定/受 floorplan 限制的范围仍不扩张。
6. dense 初始放置只遍历该 logical type 实际存在的层；random macro 初始放置仅抽取当前 floorplan region 允许且实际含该 type 的层。原有 macro、type、subtile、floorplan 和移动事务检查保留。

公共函数的参数列表不变。`delta_cx` 和 `is_median` 两个旧参数保留以兼容现有调用，但搜索器改从最终裁剪范围计算统一的 inclusive 列数。

## 兼容性与边界

2D 和资源对齐的多层架构继续使用同样的数据结构、输入和输出格式。inclusive 搜索及按列隔离搜索范围属于行为修复，可能改变同 seed 的随机轨迹；因此不能要求新旧 `.place`/`.route` 逐位相同，应检验合法性、可完成性及性能。

这不是“对任意 3D 架构已完整证明支持”的声明。宏跨层偏移、稀疏资源、高利用率、较大网表，以及其他独立 AP/backend 路径仍需分开回归。补丁不绕过逻辑/物理类型、floorplan 或 macro 约束；不存在合法位置时应正常返回失败，而不是强行放置。

## 复现与检查

补丁文件：`vtr40f8_layer_heterogeneity_v2.patch`。仅在新的隔离源树中应用和构建，原 2D 安装保持只读。

```bash
git -C "$VTR_SOURCE" rev-parse HEAD
git -C "$VTR_SOURCE" apply --check /path/to/vtr40f8_layer_heterogeneity_v2.patch
git -C "$ISOLATED_SOURCE" apply /path/to/vtr40f8_layer_heterogeneity_v2.patch
```

`generate_layer_heterogeneity_v2.py` 是从未修改的固定源码重新输出该补丁的辅助程序：只读源树、向 stdout 输出，不修改源文件。每个文本变换要求恰好匹配一次，避免静默套用到不兼容版本。

已完成隔离构建，22 次 3D P&R、2 次非零 layer `.fplace` round-trip、7 次 2D 回归、定向搜索 probe 和 20 项 Python 单元测试通过。原安装哈希未变。详情与明确未覆盖边界见 [实验验收](../RESULTS_HETEROGENEITY_2026-10-22.md#v2-加固与验收)；不能把无宏 fixture 结果外推为 dense macro fallback 已修复。
