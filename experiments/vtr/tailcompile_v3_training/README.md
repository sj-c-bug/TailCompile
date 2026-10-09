# TailCompile v3：结构化排序训练包

更新日期：2026-10-09

## 当前状态

训练闭环和 checkpoint 推理烟雾测试已经通过，但目前只完成“工程可运行性”验证，尚未获得
跨设计泛化结论。v2 训练定义不把 placement seed 当作输入，而是在同一 design/width/action
内聚合 seed；channel width 作为具有物理意义的容量条件进入区域图。

- 已生成 SHA 24 个候选和 attention_layer 24 个候选；
- 48 个 `.fplace` 哈希均不同；SHA 有 23 个不同区域分配，attention_layer 有 24 个；
- 标注计划包含 384 次 VPR：每候选 2 个 placement seeds、4 个 channel widths；
- 384 个标签运行完整，旧训练定义完成 200 epochs，但由于 seed 标签冲突只作为 bootstrap；
- checkpoint 已在 SHA 上完成推理，生成 3 个不同区域分配、容量合法的 `.fplace`；
- `campaign_seed_aggregated_v2.json` 增加 attention_layer 边界宽度、更多 seeds 和 spmv 留出族。

上述 100% smoke pair accuracy 不是测试集结果，不能报告为模型性能。

## 为什么采用结构化排序

VPR 和整数 min-cost flow 不可微，因此不能从 VPR 的拥塞值直接穿过合法化器反向传播。
当前训练目标是：在相同 design 和 channel width 下，先跨 placement seeds 聚合候选结果；如果
候选 A 的失败率、拥塞、线长和 CPD 聚合成本优于 B，则 GNN 对 A 的完整
cluster→region assignment 与区域内坐标给出更高结构分数。seed 只是随机重复，原始数值不进入模型。

优化目标默认由以下归一化成本组成：

- route failure：4.0；
- region P99：0.40；
- channel P99：0.20；
- wirelength：0.20；
- CPD：0.20。

权重保存在 checkpoint 内，正式论文实验应同时报告各原始指标，并做权重敏感性分析，不能只报告
一个合成分数。

## 文件入口

- `campaign_bootstrap.json`：48 个候选和 VPR 扫描配置；
- `campaign_seed_aggregated_v2.json`：修正后的增量标注配置；
- `prepare_seed_aggregated_v2.sh`：构建 spmv 4×4 IR、候选池和增量标签脚本；
- `run_labels.sh`：已展开的 384 条 WSL/VTR 命令，可断点式重跑；
- `bundle/dataset.json`：自包含、相对路径的数据集索引；
- `smoke_model/ranker.pt`：仅用于验证接口的 checkpoint；
- `smoke_model/training_report.json`：烟雾训练曲线；
- `requirements-training.txt`：训练依赖；
- `train_platform.sh`：平台侧重建 bundle 和训练入口。

`train_platform.sh` 会实时输出环境、数据准备、训练启动、每 10 个 epoch
和最终验收信息。训练目录还会生成：

- `model_width_conditioned/progress.json`：最近一次已完成 epoch 和逐设计曲线；
- `model_width_conditioned/ranker_latest.pt`：每 20 个 epoch 保存的中间 checkpoint；
- `model_width_conditioned/ranker.pt`：200 个 epoch 全部完成后的最终 checkpoint；
- `model_width_conditioned/training_report.json`：逐设计、逐 width 的最终训练报告。

训练器默认使用 `--device auto`：CUDA 可用时选择 GPU，否则明确记录为 CPU。
日志出现 `stage=training_complete`，并且最终 checkpoint 与报告均非空，才表示
完整成功。只有 bundle 的 `design_count/action_count/trial_count` 汇总表示数据准备
完成，不表示模型训练完成。

## 推荐执行顺序

### 1. 在有 VTR 的机器上采集标签

```bash
bash experiments/vtr/tailcompile_v3_training/run_labels.sh
```

在其他 Linux/WSL 路径上运行时先指定：

```bash
export TAILCOMPILE_REPO=/path/to/DAC2027
export VTR_ROOT=/path/to/vtr-e422b088
export TAILCOMPILE_SCRATCH=/path/to/fast_local_scratch
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v3_training/run_labels.sh"
```

脚本遇到已存在的 archive 时该条命令会失败但继续运行，所以中断后可以重新执行。运行目录中的
`manifest.json` 记录输入哈希、seed、宽度、状态和耗时。数据集构建器只接受 packed-net 哈希与
Logic IR 一致的运行。

### 2. 重建便携数据集

```bash
python3 experiments/vtr/prepare_tailcompile_v3_training.py \
  --out experiments/vtr/tailcompile_v3_training/bundle
```

只有相同 design、channel width 下的不同候选会构成训练 pair。每个候选先跨 seed 计算失败率，
并对成功运行的拥塞、线长和 CPD 取中位数。这样 seed 不会作为伪数值输入，也不会为同一候选对
产生模型无法解释的 seed 条件冲突。

### 3. 烟雾训练

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python3 experiments/vtr/train_tailcompile_v3_ranker.py \
  --dataset experiments/vtr/tailcompile_v3_training/bundle/dataset.json \
  --out experiments/vtr/tailcompile_v3_training/model_width_conditioned \
  --epochs 200 --smoke-train-all
```

### 4. 正式留设计族训练

当前只有两个设计族，不足以划分 train/validation/test。正式训练前应把相同 4×4 IR 和候选生成流程
扩展到至少 20 个独立设计族，然后例如保留若干完整设计族：

```bash
python3 experiments/vtr/train_tailcompile_v3_ranker.py \
  --dataset /path/to/full_bundle/dataset.json \
  --out /path/to/model_leave_family_out \
  --epochs 300 \
  --validation-designs heldout_family_1 heldout_family_2
```

测试设计族不得参与候选成本归一化、超参数选择或 early stopping。

### 5. 从训练 checkpoint 生成候选

```bash
python3 experiments/vtr/tailcompile_v3.py \
  --logic-ir /path/to/logic_ir.json \
  --device-ir /path/to/device_ir.json \
  --blif /path/to/design.blif \
  --checkpoint /path/to/ranker.pt \
  --channel-width 100 \
  --out /path/to/candidates \
  --candidates 20 \
  --score-noise-std 0.05 \
  --coordinate-noise-std 0.02
```

min-cost flow 始终冻结并负责容量合法性；噪声只用于从训练后的代价面附近采样多个候选。
v2 checkpoint 必须提供 `--channel-width`；旧的 width-agnostic bootstrap checkpoint 仍可不提供。

### 6. 增加边界样本和第一个留出设计

```bash
bash experiments/vtr/tailcompile_v3_training/prepare_seed_aggregated_v2.sh
bash experiments/vtr/tailcompile_v3_training/run_labels_seed_aggregated_v2.sh

export TAILCOMPILE_DESIGNS="sha attention_layer spmv"
export TAILCOMPILE_VALIDATION_DESIGNS="spmv"
bash experiments/vtr/tailcompile_v3_training/train_platform.sh "$TAILCOMPILE_REPO"
```

该增量计划为 attention_layer 使用 100/110/114/120，给 SHA 边界增加 seed 13/14，并使用
80/90/100/110 的 spmv 作为首个完整留出设计。三个设计只够验证拆分机制；正式泛化结论仍需
更多独立设计族。

`prepare_seed_aggregated_v2.sh` 在全新 Git 检出中会先调用
`prepare_spmv_pilot_inputs.sh`。后者使用 `$VTR_ROOT` 中的 Koios `spmv.v` 自动生成未纳入 Git 的
mapped BLIF、packed netlist 和压缩 RR graph；已有且非空的文件会被复用。可用
`VTR_PYTHON=/path/to/python` 覆盖运行 `run_vtr_flow.py` 的 Python，VTR 中间文件保留在
`$TAILCOMPILE_SCRATCH/spmv-pilot-inputs.*` 以便审计。
VTR commit 默认从 `$VTR_ROOT` 的 Git `HEAD` 自动读取并写入每个新 manifest；若设置
`VTR_COMMIT`，它只作为严格版本约束，和实际 `HEAD` 不一致时流程会停止。

## 算力判断

双塔 GNN 规模很小，本地 CPU 训练通常不是瓶颈。主要成本是 VPR 标签：当前 384 个任务中，
attention_layer 单次约 1–2 分钟，串行可能需要数小时。平台迁移最有价值的是并行 VPR 标注；
训练本身可在 CPU 或普通单卡 GPU 上完成。
