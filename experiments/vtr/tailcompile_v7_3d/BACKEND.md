# v7 原生 XYZ warm-start 后端

基础固定为 `40f8d19708b0b60e2aa04452deb1cfed68d96b0b`，组合只包含层间异构 v2 正确性修复和新 XYZ hook。原二维安装、数据和 checkpoint 不修改。

`install_hook.py` 从固定 pristine 源码生成与 v6.8 相同语义的原生 B2B 初始化扩展，并增加 XYZ 校验和真实层数。iteration 0 跳过冷求解但继续原生 density/timing，iteration 1 以后恢复原生 B2B。warm 初始化禁止与 `--read_flat_place`、旧 replacement worker 混用。

环境接口：

- `TC_GP_STRUCTURE=prefix`：pre-GP APBlock/atom/molecule/net 快照；movable XYZ 未提供教师位置。
- `TC_GP_STRUCTURE_ONLY=1`：结构导出后立即退出，不执行 GP。
- `TC_GP_EXPORT=prefix`：完整原生 GP 解，保留原始连续 Z。
- `TC_GP_INIT=path`：atom-named 完整 movable 初值；每行 `atom x y layer subtile`，XY 为 VPR native AP 坐标，初始 layer 为有效离散层。固定 atom 行可省略；已指定的固定轴不能被移动。
- `TC_GP_TRACE=prefix`：保存真正被注入的初值，用于对照预期输入。

每个快照输出 `.blocks.tsv/.nets.tsv/.pins.tsv/.meta.tsv/.schema.json`。blocks 十列沿用 `ap_block,atom,model_id,fixed,x,y,layer,subtile,model_name,molecule`；meta 为 `width height layer_count`。schema 为 `tailcompile_v7_ap_graph`。

原始 GP Z 不是最终物理层。固定源码 `flat_placement_density_manager.cpp` 在导入 placement 时使用 `std::round(z)`，非负值对应 `floor(z+0.5)`、0.5 映射到 1；监督转换须与之相同。输入初始层本身必须为整数。

资格证据使用：`TAILCOMPILE_V7_GP_INIT_CONSUMED`（非零层计数及 XYZ checksum）、`TAILCOMPILE_V7_NATIVE_SOLVER_CONTINUED iteration=1`、`TAILCOMPILE_V7_POST_APPACK grid=...`。APPack/SA 后实际 atom 位置分别用 VPR 的 `--write_legalized_flat_place` / `--write_flat_place` 保存，最终 `.route` 检查真实跨层资源。

本地可用 `build_backend.py` 从冻结 archive 复制并替换五个精确修改对象。manifest 记录来源库/二进制前后哈希、源码、flags、编译链接命令；此方式是可追溯的增量编译，不称 clean rebuild。完整源码和子模块齐备的服务器也可使用 `install_vtr.sh` 的正常 CMake 构建。

正常 CMake 安装需显式指定 `TAILCOMPILE_HETERO_PATCH`；安装器保留冻结 patch，并在成功后写出 `$VTR_ROOT/backend_manifest.json`，与增量构建共享 `vtr_commit`、`binary_sha256`、源码及补丁哈希契约。

这些脚本和输入契约通过不代表学习性能改善。真实非零层 native GP→APPack→SA→route 资格以及单层兼容回归单独保存。

## 2026-10-24 实际本地资格

隔离二进制位于 `/home/phoenix/tools/vtr-40f8d197-tailcompile-v7-3d-r2/vpr/vpr`，SHA-256 为 `ff5a39c53c7e0be00e68bb5c41afbc521c7d1ff0512dd44047b60396742f0a71`。原有源文件、archive 和二维二进制前后哈希不变。

- 8 项编译 C++ 输入契约测试通过：有效非零层、XYZ checksum、分数/非法层、molecule、fixed XYZ、覆盖和重复 atom。
- 一个真实双层 fixture 的 Default、GP 教师回放、翻层初值三条路线均完成 native GP→APPack→SA→route；warm 两条有实际消费和 iteration 1 继续证据。Default/replay 最终各使用 18 个不同 CHANZ 节点；翻层后后端可以重新优化，不能要求最终层保持原样。
- 既有七项单层回归全部通过；v2 的 inclusive search 会改变 SA RNG 轨迹，最终坐标可不同。
- 旧 v6.8/新 v7 的冷/暖四项真实 native GP 对照全部布通，且 APPack 前原始 GP XYZ 分别完全一致。

轻量证据位于 [backend 结果目录](../../vtr3d/runs/v7_backend_20261024/)。保存日志、初值、教师、坐标、CHANZ ID 和清理账本后回收该 smoke 与 legacy regression 共 18 份派生网表/路由文件、4,017,499 字节；共享输入网表保留。raw route 的进一步重分析需重跑原命令。上述结论仅覆盖工程资格。

2026-10-25 又完成构建副本清理：预先审计实路径、inode、大小、共享链接、运行命令/文件描述符与保留文件哈希后，删除本轮两个失败的独立 checkout（无后缀、`-r1`）及 `-r2` 中已链接完毕的重复静态库。WSL 文件系统释放 1,430,024,192 bytes（约 1.33 GiB）。`-r2` 源码、运行二进制、manifest、冻结 v2 patch、五个小 `.o` 增量缓存保留；原二维环境哈希不变。计划与完成账本均在结果目录；重复静态库可由原只读 archive 加五个对象重建。
