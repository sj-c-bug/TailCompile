# 016：可解释、有限预算的初值优化

本入口实现四组配对比较：同一异构兼容 VPR 的 Default、冻结 GNN 初值 D、原014弱联合修改 J、D＋新局部优化 E。后台日志为 `logs/v8_016/`，默认新结果目录为 `runs/development_016_explainable/`。007–015、2D、checkpoint 和 VTR 后端只读；本轮不训练网络，不重新综合或编译后端。

## 实际优化的目标

E按原始net的APBlock集合计算XY线长估计，使用实际tile单位，不能把GNN的clique/ring展开边当作独立物理网络。单层网络用HPWL；两层网络对真实RRG垂直成员入口计算两层包围盒长度之和，取允许入口中的最小值。方向与区域接入使用已有RRG摘要，属于粗粒度可达性，并非详细pin可达或实际布线路径证明。

每个跨层网络共享一个垂直入口是当前模型假设。没有合适入口的网络明确记录为模型不支持，并冻结所有关联节点；不把它计成零代价，也不声称完整估计了整张网表。高扇出普通网络保留。目标没有任意跨层次数罚分、TailRisk或真实STA标签，最终CPD由VPR评估。

每次候选包含不动这一选择。只有实际输出精度下的当前目标严格下降、固定约束和资源兼容/叶区域与层容量代理检查通过，才接受移动。移动后更新关联网络和容量，再评价后续候选。每次接受都有前后位置、受影响网络、代价变化记录。容量代理合格不等于物理packing保证，最终由APPack检查。

## 终止与成本边界

默认最多8192次候选评价、64次接受移动、3轮扫描、每轮256个候选节点、2000万工作单元，20秒软时间预算。XY总位移每轴不超过原始D所在叶区域跨度的25%，不能通过重复小步无限漂移。参数随协议冻结，预算是上限，不需要用满。找不到改善会结束；达到计算/时间上限也会返回当前有效解。报告区分无改善与预算截断，不承诺搜索到全局或任意邻域的最优点。

实现为有限坐标下降：按节点关联网络的当前代价安排访问顺序，对访问到的节点比较局部兼容区域内3×3坐标候选，选择该节点的最佳正收益移动。不是每一步重新遍历全网所有可能移动；本版没有双节点交换。候选次序和收益计算确定；时间预算触发时，完成的移动数可能随机器负载变化，复现使用冻结初值及其哈希。

软时间检查之外，优化阶段设60秒SIGALRM截止，整个预测子进程设180秒外部超时，超时会终止进程。硬超时是需要诊断的工程失败，不以D替换并伪装成成功。若在建立目标时达到软预算，原样保留D并明确记录setup预算截断和缺失的目标值，不称为已优化。已有完成初值和P&R收据在同一协议下恢复复用；不同配置必须使用新的016输出目录。

## 输入与实验矩阵

复用 `attribution_panel_015.json` 冻结的六个设计族及已有资格：SPMV、MCML、blob_merge、diffeq1、pipelined_fft_64、CORDIC。四个H×Z架构控制层间资源分布与垂直通道分布，参考宽度分别为128、224、128、160、128、128。

正式后端seeds为 **16101、16102、16103**。六族×四架构×三seeds×四方法，共 **288次P&R**。所有初值在实测前冻结，同一条件三个seed使用相同初值。D使用010已训练模型；J严格保留旧规则 `strengths=(0.15,0.35), max_moves=256, fraction=0.02, seed=11301` 的 `j_s1`，没有根据016结果重新选择修改幅度。

这是已暴露设计族上的开发比较，不能当作最终盲测。没有Wmin扫描。原生GP、APPack、详细退火与布线继续运行，层偏好关闭；Default也使用相同的异构正确性修复版后端。

## 服务器运行

先通过现有Git方式更新代码，在已激活且有Torch的环境中运行：

```bash
cd "$TAILCOMPILE_REPO"
export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_V8_016_JOBS=16
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_explainable_campaign.sh" check
bash "$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/run_explainable_campaign.sh" all --background
```

启动输出LOG、PID和OUT。完成标记为 `V8_016_GROUP_COMPLETE group=all`。支持plan、check、prepare、predict、compare、report、all；plan只读轻证据，check检查服务器大型payload和已合格后端。无需在脚本内再次activate conda。冻结模型前向使用CPU，未进行训练。

默认最多16个单线程任务，每任务8192MiB预算，按实际CPU/内存配额收紧。全部写入主仓库的本轮结果、scratch与日志目录。主入口不自动执行Git操作。具体可配置环境变量与默认值以 `run_explainable_campaign.sh` 的参数为准。

已有015垂直通道摘要会尝试只读复用。复用校验旧selection锁、预测报告、条件request与缓存完整字节哈希，再验证当前代码/RRG/器件签名；命中时不重新解析大型RRG XML，来源记录在 `vertical_cache_report.json`。旧缓存缺失或不合格时，只在016中重建。RRG摘要准备属于器件预处理，其时间单列，不混同于20秒局部优化预算。

## 查看与同步

`predictions/prediction_report.json` 保存D/J/E初值哈希、E的修改依据与终止原因。每条件E目录的 `refinement_receipt.json` 记录接受移动的详细依据，可随轻量原始证据同步。`comparison/trials_report.json` 和各任务metrics/manifest保留全部科学成功或失败；基础设施错误会停止并指出日志。`explainable_report.json`为汇总评估报告，按用户约定忽略入Git。

主要报告E/Default、E/D和E/J的WL、CPD与成功覆盖，另保留独立D/Default比较。每个比较按自身共同成功seeds计算，不因E失败而删掉D/Default结果；同时提供四组共同成功的归因表。先取同条件seed原始指标算术平均，再对四架构、设计族等权几何汇总，保留逐族差异与族级不确定性。代理下降仅代表优化器执行了定义的目标，不等于最终QoR改善。

报告的 `proxy_to_qor_by_condition` 将每个条件的预计收益、完整网络覆盖标记、终止原因，与真实E/D的WL/CPD变化及其有效seeds并列，直接检查预计改善是否转化到后端。失败和缺失保留为缺失，不补成零收益。

成功取证后及时清理派生 `.net`、`.net.post_routing`、`.route` 和GP/合法化/最终布局跟踪，保存清理收据。原始输入、RRG、模型、D/J/E初值和修改记录保留在服务器；大型坐标和网表不入Git。

实验完成后，只同步本轮允许的轻量原始证据：

```bash
cd "$TAILCOMPILE_REPO"
git add -- experiments/vtr/tailcompile_v7_3d/runs/development_016_explainable
git commit -m "Record 016 explainable refinement results"
git push origin main
```
