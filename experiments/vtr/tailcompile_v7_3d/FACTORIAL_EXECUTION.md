# 017：线长／联合时序目标 × 单点／跨层交换

本入口实现完整2×2对照，复用固定D初值、原合格网表/RRG和同一异构兼容VPR。旧2D、007–016源码与结果、GNN checkpoint和后端不修改。没有新增GNN训练。

| 方法 | 优化目标 | 可用移动 |
| --- | --- | --- |
| Default | 原生冷初始化 | 原生GP→APPack→详细退火→布线 |
| D | 冻结010 GNN＋容量解码 | 同一原生后端 |
| W | 线长 | 单点移动 |
| WS | 线长 | 单点移动＋有限跨层交换 |
| T | 线长＋时序代理 | 单点移动 |
| TS | 线长＋时序代理 | 单点移动＋有限跨层交换 |

W是应用共同建模优化后的控制组，需要在新seeds重跑，不能直接把016的E结果充当本轮控制。四组都从同一个D开始；各组搜索状态独立，不能从前一组的最终位置继续修改。

## 两个目标的边界

L沿用016原始net级XY HPWL模型；跨层网络使用实际可达CHANZ入口，不能把消息图展开边当作独立net。每条跨层net共享一个入口仍是当前近似，未模拟完整多端路由树和跨net竞争。

T为独立的器件互连延迟代理：对有明确driver的net，按driver→sink估计连接延迟，取最大sink延迟并按net权重汇总。net权重使用现有逻辑深度代理，明确不是原生STA slack。它不包含完整逻辑单元延迟、时钟、setup约束及负载树，**不是预测CPD**。

延迟系数从真实RRG已解析fanin的switch Tdel和抽样线段R/C提取，区分层内走线、垂直开关方向和末端接入，统一为ns。双向开关同名时可有相同方向系数，不虚构方向差异。大RRG采用有界前缀抽样，必要的层/轴镜像只在XML共享各向同性线段定义通过资格时使用，并明确记入provenance。缺少真实switch Tdel或不满足适配条件时停止，不使用随意默认数值。

固定归一化参考为D的线长最短入口方案：

```text
W、WS： F = L/L0
T、TS： F = 0.5 × L/L0 + 0.5 × T/T0
```

L0/T0只建立一次，四组共享，不随修改变化。联合组可对同一实际入口集合选择联合代价最低入口，因此即使D坐标相同，不同目标使用的预计路由也可能不同；参考归一化保持一致。联合目标下降允许L/T之一退化，不能据此称为最终两指标同时改善。

## 交换、约束与终止

交换是两个APBlock的原子动作。完整检查双方固定约束、资源兼容、叶/层五维容量代理，评价受影响net的并集，公共net只计一次。每个节点最多8个交换伙伴，按资源、连接和距离确定，不无限搜索伙伴。移动保持原D坐标的信赖范围，每轴位移最多为原叶跨度25%。APPack仍负责最终物理合法性。

四组共用节点触碰上限64、候选评价上限8192、3轮、每轮最多256个候选节点和20秒搜索预算。一次单点接受计1次节点触碰，交换计2次，重复移动也累计；不是每组强制改满64个节点。实际接受数、不同节点数、交换数、位移和计算量全部记录。

共同建模将等价可达入口合并、去重，并为规则/不规则入口建立查询索引，对常见二端net批量计算，避免重复扫描所有入口。共同setup与每组搜索使用独立预算；setup默认60秒，必须完整完成，否则在P&R前报错，不能用D回退并冒称MCML已优化。各阶段另有120秒截止，整个预测worker默认600秒外部超时。预算截断与无可改善候选分别记录，不宣称达到全局最优。

## 固定实验面板

六个设计族：SPMV、MCML、blob_merge、diffeq1、pipelined_fft_64、CORDIC；输入资格沿用 `attribution_panel_015.json`。四个H×Z架构仍是同一种Hecate SB跨层拓扑，分别改变资源空间分布和垂直通道位置。**本轮没有CB／CB-O／CB-I对比。**

参考channel width分别为128、224、128、160、128、128。新seeds为17101、17102、17103。六族×四架构×三seeds×六方法，最多 **432次P&R**。所有候选在任何正式实测前生成并冻结；四组共享D、器件、归一化和预算。后端保持GP→APPack→退火→布线，层偏好为0，无提前结束原生优化。

该面板已参与开发，不能称作独立盲测。Wmin和最小纵向供给本轮未测。

## 服务器命令

服务器代码更新到本实现后，在已激活的Torch环境执行：

```bash
cd "$TAILCOMPILE_REPO"
export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_V8_017_JOBS=16
export TAILCOMPILE_V8_017_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_017_factorial_fix1"
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_factorial_campaign.sh" check
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_factorial_campaign.sh" all --background
```

默认输出 `runs/development_017_factorial/`，日志 `logs/v8_017/`，完成标记 `V8_017_GROUP_COMPLETE group=all`。入口不自动激活conda或执行Git。最多16个独立单线程任务，按CPU和每任务8192MiB预算收紧，RRG相关准备最多2个并行。冻结模型推理使用CPU，未设置训练设备。

上面的命令为预测入口修复后的重跑显式选择 `development_017_factorial_fix1`，避免与已冻结的旧代码协议冲突。保留原目录，不删除或覆盖失败证据；此后若代码或参数再次变化，应另选新的 `development_017_` 目录。完整预测入口的回归测试包含真实时序模型校验、四组修正、五组初值lowering与完成标记，不能只测修正核心函数。

阶段支持plan、check、prepare、predict、compare、report、all；plan只读轻量面板，check核对实际服务器payload和既有后端。原015/016垂直摘要校验来源收据后只读复用，017中建立独立缓存。实际RRG时序摘要每条件建立一次并记录签名，缺少大型输入会指明路径并停止。更改参数使用新的development_017_输出目录；同协议恢复只读复用已完成收据。

## 效果归因与结果

默认报告包括：T/W为时序目标贡献、WS/W为交换贡献、TS/T和TS/WS为另一因素的增量、TS/W为组合效果。交互项为 `log(TS)-log(T)-log(WS)+log(W)`，在四组共同成功seeds上计算，避免不同人口混合。与Default、D的比较使用各自配对成功seeds，失败仍保留在完整成功率分母中。

主判断看最终路由WL与CPD的计划六族平均是否分别降低，并报告每设计、每架构的双赢／取舍／双退化比例。允许个别case退化；乘积或加权和变好不等于两项同时改善。先取同条件seed原始指标算术均值，再对四架构、设计族等权几何汇总。若某族缺少有效的四架构结果，主要平均不补零，明确列出有效族数与覆盖；失败仍留在全部运行的成功率分母中。

预测报告、时序模型provenance、逐次修改receipt和原生manifest/metrics是原始证据。`factorial_report.json`是评估报告，忽略入Git。报告不进行事后选组或选seed。完成每个P&R取证后清理派生 `.net`、`.net.post_routing`、`.route`及布局跟踪，保留日志、冻结初值、输入和共享RRG；本地测试同样及时清理。

运行结束后按需同步轻量原始证据：

```bash
cd "$TAILCOMPILE_REPO"
git add -- experiments/vtr/tailcompile_v7_3d/runs/development_017_factorial_fix1
git commit -m "Record 017 objective and swap factorial results"
git push origin main
```
