# ABCD 接口烟雾测试（2026-10-18）

本地 WSL、VTR 40f8d197 + 精确 v6.6 hook；SHA，device_width=14，width=100，seed=251。
使用已有 checkpoint_smoke_sha/candidate_00_seed2027.fplace（仅在测试副本转换 CRLF→LF）。
这是四路线接口验证，不是新 v6.7 生成器全流程、独立设计验证或性能实验。

| 路线 | VPR 状态 | VPR wall 秒 | 修正动作 | 移动 AP blocks |
|---|---|---:|---|---:|
| A | success | 3.0442 | 无 | 0 |
| B | success | 2.5788 | 无 | 0 |
| C | success | 25.0280 | wire_05_r1 | 36 |
| D | success | 5.0775 | wire_02_r0 | 24 |

C/D 均检测到真实 TAILCOMPILE_GP_REPLACED 标记，确认在 GP 与 read_flat_place 两条分支后均可修正。
C worker=22.1952秒，D worker=1.8655秒，已经包含在 VPR wall 内；未重复相加。
上述耗时不包含历史 GNN `.fplace` 生成成本，只跑了单次，不据此宣称加速或 Wmin 收益。
原始本地证据：/home/phoenix/vtr_work/v67_abcd_smoke_002/smoke_report.json。

10 项单元回归通过（test_direct.py 4项，test_four_routes.py 6项），Bash 语法检查通过。
修复了原 DirectBackend 在 binary 字段初始化前调用 command 导致 AttributeError 的缺陷。
