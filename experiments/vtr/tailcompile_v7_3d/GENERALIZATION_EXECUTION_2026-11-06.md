# 013 评分器容量、正则化与归一化对照

本轮检验 012 中训练拟合变好、跨架构排序变差的现象，目标是选出平均 WL、CPD 更好的初始化。输入沿用已核验的 012 特征缓存和 011 真实 VPR 标签，训练对象为候选评分头。

## 固定对照

| 因素 | 原设置 | 新设置 |
|---|---|---|
| 每个评分头的隐藏宽度 | 32 | 8 |
| Adam 的 L2 系数 | 0 | 0.01 |
| 特征变换 | 原始值 | 训练折逐列均值／标准差归一化 |

E 使用局部修改信息，执行三个因素的完整八组组合，可以在其余因素相同时核算各改动的贡献与交互。G 屏蔽局部输入，保留原设置与全部改动后的两个匹配端点，用于检查局部信息在这两个设置下的增量价值。

共 10 个配置 × 7 个固定折 × 两个训练 seeds 12301／12302，合计 **140 次离线评分头拟合**。七折包括全拟合诊断、两折评分标签留族、四折评分标签留架构。每个拟合使用 Adam、学习率 0.001、80 epochs，训练顺序及偏好权重的构造沿用 012。L／Y／J 各自评价。140 次拟合复用同一批 56 个动作、112 次历史 P&R，新增 P&R 为 0。

预先指定的主要配置为 `E_h8_train_std_l2p01`，主要训练 seed 为 12301，12302 用于敏感性核验。其它配置用于归因和探索。留出固定评价 epoch 80，不按留出结果选择 epoch 或超参数。全拟合的 epoch 5／20／80 曲线用于训练诊断。若主要配置没有改善，应记录该结果，探索性较好配置需用新条件重新确认。

## 归一化与训练边界

每个折先划分 train IDs，再按 L／Y／J 分别估计输入列的均值与总体标准差。估计只读取训练折特征，使用 float64 计算，保存 float32 统计量。标准差不大于 `1e-8` 的训练常量列在后续所有推理中置零。这个预设处理可能丢弃新架构的区分信息，其效果由对照判断。

同折 G／E 使用相同统计，G 在变换后将局部输入置零。统计量、常量 mask、训练 IDs 和哈希随 checkpoint 保存，部署使用同一变换。L2 通过 `Adam(weight_decay=0.01)` 施加到评分头参数，基础 GNN 冻结。隐藏宽度 8 的参数量约为宽度 32 的四分之一。

两个 raw32、L2=0 的原设置控制组保留 012 的运算和更新次序。报告会逐折／seed 对比原 012 的初始状态、最终状态、候选选择和排序。同设备复现若不一致，报告明确标记控制组问题，跨设备的数值差异单列。GNN 生成候选、资源约束、原生 GP 和 VPR 流程沿用已冻结实现。

当前基础 D 已接触 SPMV、MCML。这里留出评分头标签，属于开发比较，完整方法的未见设计族验收仍需独立设计。两个训练 seeds 和重复离线折不会增加独立 P&R 样本数。

## 运行

服务器激活已有 Torch 环境后，在 canonical TailCompile checkout 中执行以下命令。CPU 的 16 个独立进程适合本轮小评分头，也便于复验原 012 的 CPU 控制组。有效并发受 CPU quota、可用内存和物理核限制，每个进程一个计算线程。CUDA 可选单任务，脚本拒绝 CUDA jobs 大于 1。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_3D_TRAIN_DEVICE=cpu
export TAILCOMPILE_V7_GENERALIZATION_JOBS=16
unset TAILCOMPILE_V7_GENERALIZATION_OUT
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_generalization_campaign.sh" all --background
```

默认只读输入为 `runs/development_012_edit_context_parallel`，输出为 `runs/development_013_generalization`。脚本按仓库物理路径推导默认输出，支持 `$TAILCOMPILE_REPO` 的等价符号链接入口。临时目录为仓库内 `vtr_work/tailcompile_v7_3d/generalization`，日志为仓库内 `logs/v7_generalization_*.log`。

入口支持 `plan/check/prepare/train/report/all`。`plan/check` 为只读前台检查，`all` 按检查、准备、训练、汇总的顺序执行。独立环境变量为 `TAILCOMPILE_V7_GENERALIZATION_SOURCE/OUT/JOBS/FIT_TIMEOUT`，每任务超时默认 1800 秒。`prepare` 核验旧缓存、来源动作和标签后只读引用原缓存，无需重新构图。

后台启动会打印 `PID`、`LOG`、`SOURCE` 和 `OUT`。完成标记为 `V7_GENERALIZATION_GROUP_COMPLETE group=all`。单个拟合失败后停止新派发、终止本调度器的未完成进程组，保留已完成任务与日志。恢复使用同一命令和冻结参数，每任务最多两个 attempt。运行中保持本 cohort 的源码和输入签名固定，改代码或实验参数使用新的输出 cohort。

## 结果与同步

`trained_report.json` 应显示 `complete=true`、`planned=140`，并包含全部 140 个不同矩阵任务。`prepared_report.json` 记录只读缓存、来源签名和七折。每个任务的 `fit_report.json` 包含真实配置、归一统计摘要、训练状态和固定留出结果。

`generalization_report.json` 报告全部配置的分组 WL／CPD、成功率、排序、选择比例、全因子配对差异、历史 VPR seeds 敏感性和原控制组复现。共同成功 seeds 上先取指标算术均值，再按设计族与架构等权几何平均。WL、CPD 分别报告，联合分数只辅助排序。成功失败和个别退化保留。

该报告保留本地／服务器。Git 同步开放轻量协议、原始拟合记录、manifest 和 metrics，模型、特征张量和评估报告按忽略规则保留在算力磁盘。运行结束后按以下命令同步本轮轻量结果。

```bash
cd "$TAILCOMPILE_REPO"
git add -- experiments/vtr/tailcompile_v7_3d/runs/development_013_generalization
git commit -m "Record 013 scorer generalization metrics"
git pull --rebase origin main
git push origin main
```

本轮得到开发结果后，冻结选定配置和输入，再安排新 seeds、独立设计的小规模 P&R 复验。离线选择使用的是已测候选结果，本轮没有新的相对 Default、Wmin 或纵向容量结论。

## 本地工程检查

`test_generalization_model.py` 核验原控制组精确复现、留出特征／标签扰动隔离、L2 实际更新、常量列、G 屏蔽、checkpoint 往返和原候选域部署。控制器与 Bash 测试核验矩阵、因子归因、签名、恢复和路径边界。

`generalization_local_smoke.py --jobs 4` 使用本地已有小架构 012 缓存，并行完成 10 配置 × 两训练 seeds 的 20 次工程拟合，检查源不变、原控制组复现与模型重载。它输出到忽略的 `development_013_generalization_local`，完成后按已记录的路径／哈希清理本轮诊断 checkpoint。这些小样例用于工程验收，自然设计的 140 次矩阵使用服务器输入运行。
