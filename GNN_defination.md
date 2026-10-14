# TailCompile 当前 GNN 定义

> 文档状态：与当前 `tailcompile_v3.py`、`train_tailcompile_v3_ranker.py` 和
> `prepare_tailcompile_v3_training.py` 的实现对齐。v4 没有重新训练 GNN，
> 而是使用这个冻结 v3 checkpoint 生成起始预绑定，再进行确定性供需风险搜索。

## 1. 总体定义

当前模型是一个双塔 GNN：

1. **Logic tower** 对 packed cluster 逻辑网表建图并生成 cluster embedding；
2. **RRG tower** 对 4×4 物理区域级 RRG 粗图建图并生成 region embedding；
3. 两组 embedding 经投影后做点积，生成每个 `cluster → region` 的分数；
4. Logic tower 的 embedding 还通过 coordinate head 输出 cluster 在区域内的相对坐标；
5. 容量约束 min-cost flow 将分数矩阵解码成合法区域分配；
6. 确定性 lowerer 将区域和相对坐标变成 VPR 可读的 `.fplace`。

设 cluster 数为 \(N\)，物理区域数为 \(R=16\)，hidden width 默认为
\(d=32\)。双塔的核心输出为：

\[
S=\frac{(H_LW_L)(H_RW_R)^T}{\sqrt d}\in\mathbb{R}^{N\times16},
\]

\[
U=\tanh(\operatorname{MLP}(H_L))\in[-1,1]^{N\times2}.
\]

\(S_{i,r}\) 是 cluster \(i\) 放到 region \(r\) 的神经网偏好分数，
\(U_i=(u_i,v_i)\) 是 cluster \(i\) 在所选区域内的相对坐标。

## 2. 两个 tower 共用的消息传递层

两端的网络结构相同，但参数不共享。每个 tower 包含：

- 一个 `Linear(input_width, 32) + ReLU`；
- 两层 `GraphLayer(32)`。

对有向边 \(j\rightarrow i\) 及标量权重 \(w_{ji}\)，当前 GraphLayer 先将边权
除以该图所有边权的均值，再做加权邻居平均：

\[
\bar h_i=
\frac{\sum_{j\rightarrow i}\tilde w_{ji}h_j}
{\max(1,\sum_{j\rightarrow i}\tilde w_{ji})},
\qquad
\tilde w_{ji}=\frac{w_{ji}}{\operatorname{mean}(w)}.
\]

输出为：

\[
h_i'=\operatorname{ReLU}〈
\operatorname{LayerNorm}(W_sh_i+W_n\bar h_i)
〉.
\]

当前层没有 attention、edge MLP、residual connection 或高阶超边更新。

## 3. Logic tower：逻辑端 GNN

### 3.1 什么是图的节点

一个节点对应一个 **VPR packed cluster**，不是单个 LUT/FF atom，也不是
v2 划分得到的 16 个逻辑大组。同一 packed cluster 中的 atoms 在后续
lowering 中保持原子性，放在同一个物理 tile/subtile。

### 3.2 节点输入特征

每个 cluster 有 8 维输入：

| 维度 | 定义 |
|---|---|
| 1–4 | 资源类型 one-hot：`clb` / `memory` / `dsp_top` / `io` |
| 5 | `log1p(cluster 内 atom 数)` |
| 6 | `log1p(加权连接度)` |
| 7 | `log1p(关键度加权连接度)` |
| 8 | `log1p(入射网络权重的 P95)` |

加权连接度为所有入射 contracted nets 的 `edge.weight` 之和；关键度加权度为
`edge.weight * edge.criticality` 之和。八个特征列在每个设计内分别做 z-score。

**重要限定：**当前 Logic tower 没有直接输入 v2 `group.tail_score` 或完整
TailRisk Profile。第 8 维只是 cluster 入射边权的局部 P95 代理，不等于我们
后续定义的空间供需 TailRisk。

### 3.3 什么是图的边

1. 先从 mapped BLIF 恢复原子级逻辑网络；
2. 通过 packed netlist 的 `atom_to_cluster` 将每条 net 收缩到 cluster 级；
3. 只剩一个 cluster 的网络被丢弃；
4. 对每条收缩后超边，从 driver 到每个 sink 建立消息边，并同时加入
   sink 到 driver 的反向消息边；
5. 多条 net 产生的相同有向 cluster pair 会被合并，权重相加。

因此，实现中的边是**带标量权重的双向 pairwise message edge**。它不是显式
超图卷积；收缩网络原本的高阶超边身份在转成 pairwise message 后不再单独保留。
`criticality` 只进入 cluster 节点的累积特征，并没有作为每条 message edge
的独立特征。

### 3.4 Logic tower 的输出

Logic tower 输出 \(H_L\in\mathbb{R}^{N\times32}\)，每行是一个 cluster embedding。
它同时被用于：

- 与 region embedding 计算 `cluster → region` 分数；
- 通过 coordinate head 生成两维区域内相对坐标。

## 4. RRG tower：物理端 GNN

### 4.1 什么是图的节点

一个节点对应 FPGA 4×4 划分中的一个**物理区域**，总共 16 个节点。

这不是将 VPR RRG 中每个 `CHANX/CHANY/IPIN/OPIN/SOURCE/SINK` 节点直接送入
GNN。原始 RRG 首先被压缩成区域容量和区域间连接统计，GNN 只看到这个粗图。

### 4.2 节点输入特征

基础版本每个 region 有 12 维特征：

| 维度 | 定义 |
|---|---|
| 1–3 | `log1p(CLB / memory / DSP tile 数)` |
| 4 | `log1p(region 内部 channel supply)` |
| 5–8 | `log1p(left / right / down / up escape supply)` |
| 9–10 | `log1p(L4 / L16 segment supply)` |
| 11–12 | region 中心的归一化 \(x,y\) 坐标 |

这 12 列在当前 device 的 16 个 region 之间分别做 z-score。对当前
width-conditioned checkpoint，还会追加第 13 维：

\[
x_{width}=\frac{\log(1+W)}{\log(301)},
\]

其中 \(W\) 是当前 route channel width。这一列对同一 device 的 16 个 region
相同，并在 z-score 之后追加，因此不会因“区域内是常数”而被标准化掉。

### 4.3 什么是图的边

Device IR 中的 `inter_region_rr_edge_counts_per_channel_width` 为两个物理区域
之间的粗粒度 RRG 连接数。每个无向区域对 \((a,b)\) 被展开为：

- \(a\rightarrow b\)；
- \(b\rightarrow a\)。

两个方向使用同一个标量权重 `count`。因此当前 RRG tower 的 edge feature
只有一个粗粒度连接强度，没有独立的方向类型、segment type、switch type、
wire length 或精确 cut capacity 向量。方向 escape 和 L4/L16 当前只是 region 节点特征。

### 4.4 RRG tower 的输出

RRG tower 输出 \(H_R\in\mathbb{R}^{16\times32}\)，每行是一个 region embedding。
经独立线性投影后，它与 Logic tower 的 cluster embedding 做点积，得到
\(N\times16\) 匹配分数矩阵。

## 5. 模型的完整输入

### 5.1 推理时输入

| 输入 | 用途 |
|---|---|
| `logic_ir.json` | packed clusters、atoms、resource type、v2 group 先验和 packed-net 溯源 |
| `device_ir.json` | 4×4 regions、tile 容量、RRG 供给统计和区域间边 |
| mapped `.blif` | 恢复逻辑 net，并与 packed cluster 关系收缩成逻辑图 |
| channel width | 当前物理容量条件，进入每个 region 的第 13 维特征 |
| checkpoint | 双塔、投影层和 coordinate head 的冻结参数 |

placement seed **不是模型输入**。它只是 VPR 标注时的重复试验条件。

### 5.2 训练时额外输入

每个完整 placement action 包含：

- `assignment`：长度 \(N\) 的 region ID 数组；
- `relative_xy`：\(N\times2\) 的区域内相对坐标；
- 在多个 channel widths 和 placement seeds 下得到的 VPR 状态与 QoR。

## 6. 标签如何定义

### 6.1 没有节点标签或边标签

Logic tower 和 RRG tower **没有各自独立的监督标签**。当前不会标注“某条 net
应该经过哪条 RRG edge”，也不会直接标注“某个 cluster 应该放在哪个 region”。

唯一的监督信号是：**同一 design、同一 channel width 下，两个完整 placement
actions 中哪一个更好。**

### 6.2 先聚合 placement seeds

对同一 `design + channel width + action`：

- 路由失败率 = 失败 seed 数 / 总 seed 数；
- `region_p99`、`channel_p99`、线长和 CPD 只在成功 seeds 中取中位数。

seed 只被视为 nuisance repetition，其数值大小没有物理语义，不会输入 GNN。

### 6.3 候选成本与偏好标签

默认成本为：

\[
C(a)=4.0f_{fail}
+0.40\hat p_{region}
+0.20\hat p_{channel}
+0.20\hat l_{wire}
+0.20\hat t_{CPD}.
\]

四个成功 QoR 指标在当前 `design + channel width` 的所有 actions 之间分别做
min-max 归一化。成本越低越好。如果两个 action 的成本差绝对值不大于
`pair_margin=0.05`，这一对不进入训练；否则低成本 action 是 `preferred`。

因此标签是 pairwise preference，不是绝对拥塞回归值、单个 region 类别或
VPR 最终坐标。

## 7. 训练时如何给一个完整 action 打分

对 action \(a\) 的分配 \(r_i^a\) 和相对坐标 \(u_i^a\)，模型结构分数为：

\[
E(a)=\frac{1}{N}\sum_i S_{i,r_i^a}
-\lambda_{xy}\operatorname{MSE}(U,U^a),
\qquad \lambda_{xy}=0.2.
\]

对一个 `preferred` action \(a^+\) 和较差 action \(a^-\)，使用 softplus pairwise
ranking loss：

\[
\mathcal L=
\operatorname{softplus}
\left(-\frac{E(a^+)-E(a^-)}{T}\right),
\]

当前 temperature \(T=1\)。梯度同时更新两个 tower、两个投影层和 coordinate head。

当前 width-conditioned checkpoint 的训练集设计为 SHA 和 attention_layer，验证设计
为 spmv。训练报告中 train pair accuracy 约为 66.73%，spmv validation pair accuracy
约为 53.95%。spmv 后来又参与了 v4 开发分析，所以不再是最终未见测试集。

## 8. 推理输出与合法化

### 8.1 神经网的原始输出

- `scores`：\(N\times16\) cluster-region 分数矩阵；
- `relative_xy`：\(N\times2\) 区域内相对坐标。

这两个张量本身不是合法 placement，也不是 `.fplace`。

### 8.2 确定性 partition prior

在推理时，实现还会根据 v2 划分组与物理 region 的 Manhattan 距离，向
`scores` 中加入一个默认权重为 0.35 的确定性 prior。因此实际解码的分数
不是纯 GNN 分数。coordinate head 不参与这个 prior 的计算。

### 8.3 容量合法解码

对 `clb`、`memory`、`dsp_top` 三种资源分别构建 min-cost flow：

- 每个 cluster 必须选择一个 region；
- cluster-region 边的 cost 是负的 GNN/prior 分数；
- 每个 region 的目标 cluster 数按该类资源的 tile 容量比例分配；
- `network_simplex` 给出整数、容量合法的区域分配。

IO 的 region ID 按分数与简单均衡项贪心选择，lowerer 则将 IO 放到真实外围
tiles。

### 8.4 `.fplace` lowering

对每个 `region + resource` bucket，lowerer 将 relative coordinate 转成物理目标点，
并选择最近的尚未使用 tile anchor。同一 packed cluster 中的所有 atoms 共享该
tile/subtile。这一步才产生 VPR 可读的 `.fplace`。

## 9. v4 中 GNN 和 TailRisk 的关系

v4 的 `gnn_base` 是冻结 GNN 生成的初始 `.fplace`。`gnn_tail`、`gnn_mean`、
`gnn_none` 不是三个新训练的 GNN，而是从同一 `gnn_base` 区域分配出发，用
v4 确定性局部搜索分别优化 tail/mean/no-risk 目标。

所以当前实验中“GNN + TailRisk”的准确含义是：

\[
\text{frozen GNN initialization}
\rightarrow
\text{capacity-legal assignment}
\rightarrow
\text{deterministic TailRisk refinement},
\]

而不是“GNN 直接以 TailRisk 或最终尾部拥塞为节点标签训练”。

## 10. 当前定义的主要限制

1. Logic tower 将超边展开为 pairwise edges，没有显式保留多端网的高阶结构。
2. RRG tower 只处理 16 个粗区域，没有直接学习原始 RR-node 拓扑。
3. RRG edge 只有一个标量 count，方向和 segment 类型未成为 edge feature。
4. 逻辑端没有显式的完整 TailRisk Profile 输入。
5. 结构化排序标签将失败率、拥塞、线长和 CPD 合成一个人工加权偏好，
   学到的相对偏好依赖这些权重。
6. 当前模型输出必须经过 partition prior、min-cost flow 和 lowerer，实际 `.fplace`
   不是纯神经网直接生成的坐标。
7. v4 中 VPR 后的区域保持率较低，表明后端详细布局会重写大部分粗粒度分配。

## 11. 一句话总结

当前 TailCompile GNN 学习的是：

> 在给定 packed-cluster 逻辑图、4×4 区域级 RRG 粗图和 channel width 时，
> 如何为每个 cluster-region 配对和区域内坐标打分，使 VPR 表现更好的
> 完整 placement action 在结构化排序中获得更高分数。

它当前并不直接预测最终拥塞，不直接输出合法 `.fplace`，也不直接对原始
RRG 全图进行 message passing。
