# 014 弱联合修改历史配置记录

这是此前取得有利线长结果的 **014 固定弱联合修改 `j_s1`** 的复现配置。机器可读明细见同目录 `configuration.json`，包含全部8个条件、24次弱修改运行的原始命令、环境变量、输入/模型/器件哈希，以及配对Default/D指标。没有按结果筛掉架构或seed。

本文件是配置档案；详细评估报告继续按用户约定保存在本地。015的新seed复测另有原始记录，不能用这份历史配置替代后续复测结论。

## 原始版本

| 项目 | 固定值 |
| --- | --- |
| 原始cohort | `experiments/vtr/tailcompile_v7_3d/runs/development_014_prospective_joint` |
| 实现commit | `d4638e4a6857efff47aa937df6583d83465749f0` |
| 原始结果commit | `cf0863dabf3994d3b8d0e1c1ecce223fb00ae859` |
| 被记录的方法 | 固定使用 `j_s1`，即D＋弱联合改层/XY |
| 比较基线 | 同一异构兼容VPR的Default，以及不做局部修改的D |
| 学习评分器 | 固定弱修改不使用评分器选候选 |
| 新训练 | 014没有新训练 |

## Benchmark、器件与seeds

| Benchmark | 设计族 | 固定网格 | APBlocks | 正式channel width | 正式后端seeds |
| --- | --- | --- | ---: | ---: | --- |
| `pipelined_fft_64` | `pipelined_fft` | 32×32×2层 | 7190 | 128 | 14101、14102、14103 |
| `cordic` | `cordic` | 24×24×2层 | 510 | 128 | 14101、14102、14103 |

两份RTL均来自固定VTR checkout的 `vtr_flow/benchmarks/arithmetic/open_cores/verilog/processed_files/output/`，文件名分别为 `pipelined_fft_64.v`、`cordic.v`。其别名、原始RTL SHA-256、mapped BLIF SHA-256、AP快照摘要和真实路径均保存在JSON中。

每个benchmark都使用四种架构：

| 架构 | 上下层资源分布 | 垂直通道位置 |
| --- | --- | --- |
| H0Z0 | 均衡 | 分散 |
| H0Z1 | 均衡 | 中央集中 |
| H1Z0 | 不均衡 | 分散 |
| H1Z1 | 不均衡 | 中央集中 |

同一benchmark四架构控制总资源和总垂直容量相同，具体供给和上下层容量以JSON引用的真实RRG审核为准。器件区域tile_size=4，hard_columns=4，每个选定垂直位置6个Z资源。基础架构为固定40f8版本的 `vtr_flow/arch/multi_die/hecate/3d/hecate_3d_sb_10um_bidir_fanin_16_fanout_16_with_esd_7nm.xml`。

资格seed为 **14001**。128/192/256是预先登记的资格候选宽度；本次两个benchmark在128已全部通过，实际资格P&R共8次。正式比较只使用128，未测Wmin。正式弱修改为2×4×3=**24次，全部成功**；连同Default/D/强修改，原实验总计96次正式P&R，全部成功。

## D模型和弱修改参数

D checkpoint：

```text
experiments/vtr/tailcompile_v7_3d/runs/development_010_structured/models/decoder_control/epoch_0000.pt
SHA-256: 13b504e333de66ce00607879d753fca25ffbee8b397da59800942becba12f185
```

D使用已训练009X的权重，配合010严格容量解码器。文件名epoch_0000记录的是解码器迁移起点，不能理解为随机未训练网络。生成D使用CPU、score_noise_std=0；随后确定性编辑候选，不额外pack、不重新训练。

| 参数 | 值 |
| --- | --- |
| arm | `J`，先改层再改XY |
| strengths完整列表 | `[0.15, 0.35]` |
| 本方法选中的候选 | `j_s1`，strength=0.15 |
| max_moves | 256 |
| fraction | 0.02 |
| 确定性编辑seed | **11301** |
| 总候选预算 | `min(256, ceil(APBlock数量 × 0.02))` |
| 弱修改节点预算 | `ceil(总候选预算 × 0.15 / 0.35)` |
| XY步长 | 朝目标叶区域内的加权邻居中心移动15%，每轴位移不超过该叶跨度15% |
| TailRisk权重 | 0，不参与该编辑 |
| 收敛迭代 | 无，有限一次编辑 |

**复现时保留完整strengths列表。**若只传 `[0.15]`，预算公式分母会改变，得到的就不是原014弱修改。FFT实际修改62个APBlocks，CORDIC实际修改5个；四架构均如此。逐条件初值文件和动作摘要在JSON中保存。

## VPR后端和运行参数

| 项目 | 固定值 |
| --- | --- |
| 上游VTR commit | `40f8d19708b0b60e2aa04452deb1cfed68d96b0b` |
| VPR binary SHA-256 | `34cd4e3e97c83d5c75db988f7a3944acd0933f60aac6ac39a2babab8357f644e` |
| 后端修复 | 层间异构v2正确性补丁＋XYZ warm-start/观测hook |
| 流程 | 初始XYZ → 原生GP → APPack → 详细退火 → 布线/分析 |
| full legalizer | `appack` |
| detailed placer | `annealer` |
| flat routing | `off` |
| num_workers | 1 |
| device | `pilot` |
| layer prior | `TC_GP_LAYER_PRIOR=0` |
| 初值接口 | `TC_GP_INIT=对应j_s1/init.ap.tsv` |
| GP导出 | `TC_GP_EXPORT=post_gp` |
| 每任务timeout | 3600秒 |
| 并发请求 | 16个单线程任务，每任务预算8192MiB |
| RRG并发 | 2，每任务预算8192MiB |
| 线程环境 | OMP/MKL/OPENBLAS/NUMEXPR线程数均为1 |
| Python | 3.10.12，conda-forge，GCC12.3.0 |
| PyTorch构建 | 2.11.0+cu128；候选推理实际在CPU |

每次弱修改运行的完整原始VPR命令和环境变量位于JSON的 `weak_runs`；其中 `--seed` 为14101/14102/14103、`--route_chan_width` 为128。其余架构、网表、RRG和初值的具体路径及SHA不能随意替换。

## 当时记录的结果

所有24次弱修改均计入，负数表示改善：

| 比较 | WL变化 | CPD变化 |
| --- | ---: | ---: |
| 弱修改/D | -2.9792% | -1.7584% |
| 弱修改/Default | -3.4128% | +1.4756% |

汇总先取同条件三个seeds的原始指标算术平均，再对四架构和两个设计族等权几何汇总。这里记录的是有利线长结果及其时延代价，不代表两项均优于Default。

## 复现所需文件

服务器需保留固定D checkpoint、mapped BLIF、AP快照、八个架构XML/RRG/device图，以及对应D和j_s1的 `init.ap.tsv`。JSON为这些文件保存路径与哈希，**不会代替实际模型或初值数据**。

此前清理了派生 `.net`、`.net.post_routing`、`.route` 及GP/合法化/最终布局跟踪。重放时将JSON中的原始命令和环境用于新的工作目录，保留历史目录只读；哈希匹配的原始初值可直接使用。若需重新生成，固定模型、源代码和上述完整生成参数，并核对输出初值哈希。

015对同八条件使用15101/15102/15103进行的新seed复测，另见其cohort；该组D/J初值与器件摘要均和014一致。本档保留014历史配置，后续复测不会覆盖或替换它。
