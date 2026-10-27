# v7：学习式 3D 原生 GP 初始化

本目录是与冻结 2D v6.8/v6.9 分离的实现。输入为 prepack APBlock 图和真实 XYZ RRG 区域／共享池图，输出带 layer 的 `init.ap.tsv`，随后继续执行同一 VPR 的原生 GP→APPack→详细退火→布线。第一轮比较 Default、capacity-only、GNN，不使用旧风险局部搜索。

本地已经完成接口资格和真实小模型训练／部署测试；研究规模训练、四族 480 次 pilot、独立 Wmin 扫描及受控总时间验收留给服务器。工程测试通过不代表性能改进。

- [服务器分阶段入口](README_SERVER.md)：环境、双 VTR 根、资格、CPU/GPU 交接、续跑、日志与同步范围。
- [后端构建与接口](BACKEND.md)：固定 `40f8d197`、v2 异构补丁、新 XYZ hook、构建 provenance。
- [开发配置](campaign.example.json)：候选输入、H×Z、seeds、预算和主验收口径；第二 heavy 仍需映射资格。
- `verify.py --out <report.json>`：有界接口／模型／存储测试，保存报告和日志。
- [CPU 准备入口](prepare_server.sh)：从所在 Git 仓库定位源码，只读使用固定 2D 综合环境，独立构建 3D 后端；完成资格和输入准备后停止。

源码及轻量结果统一进入当前 TailCompile 主仓库，遵循 [仓库工作流](../../../REPOSITORY_WORKFLOW.md)。服务器在原 `DAC2027/TailCompile` 目录 `git pull --ff-only`，日常不通过 ZIP 建立另一份源码目录。固定原 VTR 只作为执行和映射依赖，3D 后端使用另一个独立构建目录。

[本地准备结果](../../vtr3d/RESULTS_V7_PREPARATION_2026-10-25.md) 记录工程结果、负结果及未完成事项。发布版本以本仓库 Git commit 为准；服务器需先获取包含本入口的提交。[与运行中v6.10共用实例](CO_RUN_V610_2026-10-27.md) 说明安全更新检查、低并发准备和正式计时边界。
