# TailCompile v3：结构化排序训练包

更新日期：2026-10-03

## 当前状态

训练闭环已经可以执行，但目前只完成“工程可运行性”验证，尚未获得跨设计泛化结论。

- 已生成 SHA 24 个候选和 attention_layer 24 个候选；
- 48 个 `.fplace` 哈希均不同；SHA 有 23 个不同区域分配，attention_layer 有 24 个；
- 标注计划包含 384 次 VPR：每候选 2 个 placement seeds、4 个 channel widths；
- 现有正式数据只有 3 个 actions、3 个 trials，能形成 1 个严格受控 pair；
- 20 epoch 本地烟雾训练把 pair loss 从约 0.692 降至 0.621；
- checkpoint 已在 SHA 上完成推理，生成两个不同、容量合法的 `.fplace`。

上述 100% smoke pair accuracy 不是测试集结果，不能报告为模型性能。

## 为什么采用结构化排序

VPR 和整数 min-cost flow 不可微，因此不能从 VPR 的拥塞值直接穿过合法化器反向传播。
当前训练目标是：在相同 design、placement seed 和 channel width 下，如果 VPR 证明候选 A 优于
候选 B，则 GNN 对 A 的完整 cluster→region assignment 与区域内坐标给出更高结构分数。

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
- `run_labels.sh`：已展开的 384 条 WSL/VTR 命令，可断点式重跑；
- `bundle/dataset.json`：自包含、相对路径的数据集索引；
- `smoke_model/ranker.pt`：仅用于验证接口的 checkpoint；
- `smoke_model/training_report.json`：烟雾训练曲线；
- `requirements-training.txt`：训练依赖；
- `train_platform.sh`：平台侧重建 bundle 和训练入口。

`train_platform.sh` 会实时输出环境、数据准备、训练启动、每 10 个 epoch
和最终验收信息。训练目录还会生成：

- `model_bootstrap/progress.json`：最近一次已完成 epoch 和历史曲线；
- `model_bootstrap/ranker_latest.pt`：每 20 个 epoch 保存的中间 checkpoint；
- `model_bootstrap/ranker.pt`：200 个 epoch 全部完成后的最终 checkpoint；
- `model_bootstrap/training_report.json`：最终训练报告。

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

只有相同 design、placement seed、channel width 下的不同候选会构成训练 pair。失败运行作为明确的
可布线性标签保留；成功运行同时使用拥塞、线长和 CPD。

### 3. 烟雾训练

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
python3 experiments/vtr/train_tailcompile_v3_ranker.py \
  --dataset experiments/vtr/tailcompile_v3_training/bundle/dataset.json \
  --out experiments/vtr/tailcompile_v3_training/model_bootstrap \
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
  --out /path/to/candidates \
  --candidates 20 \
  --score-noise-std 0.05 \
  --coordinate-noise-std 0.02
```

min-cost flow 始终冻结并负责容量合法性；噪声只用于从训练后的代价面附近采样多个候选。

## 算力判断

双塔 GNN 规模很小，本地 CPU 训练通常不是瓶颈。主要成本是 VPR 标签：当前 384 个任务中，
attention_layer 单次约 1–2 分钟，串行可能需要数小时。平台迁移最有价值的是并行 VPR 标注；
训练本身可在 CPU 或普通单卡 GPU 上完成。
