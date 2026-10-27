# 3D 计划、共享依赖与轻量证据

本目录属于唯一 TailCompile Git 主仓库，遵循 [REPOSITORY_WORKFLOW.md](../../REPOSITORY_WORKFLOW.md)。新学习实现及入口位于 [tailcompile_v7_3d](../vtr/tailcompile_v7_3d/README.md)；本目录保留共享的3D helper、完整v2补丁、实验方案和本地准备结果。

- [仓库与2D隔离核查](REPOSITORY_ALIGNMENT_2026-10-25.md)：修正ZIP交付方式、源码边界、忽略规则和未发布状态。
- [主实验方案](EXPERIMENT_PLAN_3D_V68_2026-10-23.md)、[准备执行计划](PRE_GNN_EXECUTION_PLAN_2026-10-23.md)、[外部比较计划](EXTERNAL_COMPARISON_PLAN_2026-10-23.md)。
- [本地准备结果](RESULTS_V7_PREPARATION_2026-10-25.md)、[H×Z审核](RESULTS_ARCHITECTURES_2026-10-24.md)。
- [迁入账本](migration_v7_20261025.json)：685份文件、约8.95MB原样迁入；排除原始payload和大文件，保留其来源和哈希。外层旧目录作为历史来源未删除，当前代码不再依赖它。

`runs/v7_*` 是当时工程证据，不等于新代码重跑后的性能结果。原始VPR全文、网表、教师坐标、RRG和图不通过Git传输；旧轻报告可能引用仍留在原WSL／历史归档中的大payload。新服务器数据从固定输入和源码生成，运行日志统一放仓库`logs/`。

新入口只通过主仓库Git同步；旧离线ZIP和旧解压启动命令不再作为日常流程。
