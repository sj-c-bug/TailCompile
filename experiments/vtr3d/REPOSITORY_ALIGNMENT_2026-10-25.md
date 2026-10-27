# 3D代码管理与2D隔离核查

日期：2026-10-25。结论：之前让服务器传源码ZIP并另建TailCompile解压目录，与仓库既定工作流不符，已纠正。**统一主Git仓库，按模块／运行目录／后端安装隔离2D和3D。**

## 发现与修正

| 发现 | 当前修正 |
|---|---|
| 另建ZIP源码根，服务器代码容易与主仓库漂移 | 日常只在原`DAC2027/TailCompile`中`git pull --ff-only`；旧ZIP入口停用，离线包仅历史留档 |
| 新v7代码在主仓库，helper／补丁／三计划和证据却在外层工作区 | 依赖和轻证据迁入`experiments/vtr3d`，固定从本仓库加载；取消父工作区fallback |
| 前一轮使用了旧服务器VTR路径 | 当前固定只读依赖为工作流中的`vtr_tailcompile/vtr-verilog-to-routing`；不能回到历史e422安装 |
| 输出只检查workspace，误配置可能写进2D结果或依赖安装 | 新guard在创建输出／锁前解析真实路径，拒绝与其他版本及指定VTR根相交，包含symlink和祖先目录 |
| 3D结果、权重、图尚无专属Git规则 | 两个局部`.gitignore`只放行轻量证据；原始图／教师／动作／checkpoint／net／route／日志默认忽略 |
| 同名裸Python模块可能在同一解释器混用 | 正式版本各自用独立CLI进程；v68/v69仅限定名只读依赖，新入口禁用bytecode写入 |

## 3D与2D的边界

- 源码：新功能只写`experiments/vtr/tailcompile_v7_3d`和3D共享目录；没有修改`tailcompile_v68`／`tailcompile_v69`的源码、schema或训练入口。
- 结果：新cohort默认`runs/development_002_git`；scratch只用`repo/vtr_work/tailcompile_v7_3d`，不复用旧2D结果目录。
- 环境：新入口公开`TAILCOMPILE_3D_*`／`TAILCOMPILE_V7_*`，泛变量只在3D子进程内适配；不覆盖调用终端的2D `VTR_ROOT`。`env_3d_v7.sh`为本机生成文件，Git忽略。
- VTR：原40f8安装只读执行综合；新后端默认放`DAC2027/vtr_tailcompile_3d_v7/vtr-verilog-to-routing`，并要求与原安装和源码仓库互不包含。原安装不追加补丁、不重新链接。
- 依赖：只读复用v68纯模型／资源函数、v69阶段解析，记录源码哈希，不复制或覆盖旧checkpoint。代码变化用新protocol／cohort，旧结果不改写。
- 跨系统：新增3D代码／hook／补丁强制LF，避免Windows/Linux checkout字节差异破坏签名。

## 实测检查

统一主仓库内51项测试全部通过、0跳过，含新的四项仓库隔离检查：独立进程导入／一次小训练及lowering期间审计阻止任何冻结2D写入；裸模块污染检查；2D目录／祖先／源代码目录／错误scratch拒绝；VTR和symlink拒绝。见 [测试报告](../vtr/tailcompile_v7_3d/runs/repository_alignment_20261025/unit_tests.json) 和 [日志](../vtr/tailcompile_v7_3d/runs/repository_alignment_20261025/unit_tests.log)。新入口通过`bash -n`和`--help`，未为此次目录管理改动重复大规模P&R或服务器训练。

迁入685文件逐项哈希一致，账本约872KB；685份原始迁入约8.95MB，另排除446份约19.58MB派生文件。随后仅对迁入Markdown的仓库内链接和当前管理说明做显式更新，迁入账本记录复制时的哈希，不冒充修改后文档的永久哈希。

检查原v68/v69的tracked Git diff为空。Git忽略探针确认`model.pt`、`device_graph.json`、`feedback.json`、原始`vpr.out`不入Git，`metrics.json`和`qualification_report.json`可同步。机器可读仓库状态见 [repository_audit.json](../vtr/tailcompile_v7_3d/runs/repository_alignment_20261025/repository_audit.json)。

## 发布状态和下一次服务器运行

2026-10-25整理完成时，本批仍未提交／推送；该日期的审计JSON保留当时状态。2026-10-27用户授权提交本批，发布状态以仓库Git commit及远端同步结果为准。服务器在原TailCompile目录获取对应commit；若v6.10仍在运行，先检查整次更新是否会修改其冻结依赖，见[共用实例说明](../vtr/tailcompile_v7_3d/CO_RUN_V610_2026-10-27.md)。

服务器操作以 [新README_SERVER](../vtr/tailcompile_v7_3d/README_SERVER.md) 为准：先`prepare_server.sh --check`，再后台执行首次CPU准备；它停在`prepare`。现阶段仍需MCML等输入资格，不能把仓库整理或51项工程测试称为480比较／训练／性能验收完成。
