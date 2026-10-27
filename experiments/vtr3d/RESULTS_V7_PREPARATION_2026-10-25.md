# v7 3D 准备实验：结果、范围与服务器交接

日期：2026-10-25，Asia/Shanghai。承接冻结的 2D v6.9 工程结论。**本轮完成了新的 3D 学习接口和小模型闭环，尚未证明总运行时间或 QoR 改善。** 下一步是服务器自然设计资格、教师和真实收益训练，不再反复调旧启发式修正器。

管理更新：源码和轻结果现统一纳入TailCompile主Git仓库，旧ZIP解压交付已停用；见 [仓库隔离核查](REPOSITORY_ALIGNMENT_2026-10-25.md)。以下工程结果按当时源码解释，新管理入口使用独立新cohort，不改写旧实验。

## 1. 现在已有的可执行流程

```text
RTL／有来源的 mapped BLIF → 原生 prepack APBlock 结构（无 movable 教师坐标）
真实 RRG → 每层固定 4×4 tile 叶区域＋层次图＋去重共享 wire pools
    → 分层双图 GNN 联合生成 layer／leaf／连续 XY
    → 资源兼容、必要层容量检查＋有限 XY 密度处理
    → init.ap.tsv → 原生 3D GP → APPack → 详细退火 → route／analysis
```

同一输入也可运行 Default 和确定性 capacity-only；三组共享后端和后续优化。没有加入旧风险局部移动或测试时候选搜索。AP 初值不等于最终物理合法布局，原生 GP 可以改变其层及 XY。

新代码：[实现入口](../vtr/tailcompile_v7_3d/README.md)、[服务器逐阶段命令](../vtr/tailcompile_v7_3d/README_SERVER.md)、[后端契约](../vtr/tailcompile_v7_3d/BACKEND.md)。原 `tailcompile_v68/`、`tailcompile_v69/`、既有二维安装及 checkpoint 未被替换。

## 2. 实际完成的验证

| 项目 | 实测结果 | 能说明什么 |
|---|---|---|
| 新 XYZ GP 核心资格 | 对齐、错列、固定点／混合资源、SHA 四 cases × Default／教师回放／翻层 × 两 seeds，**24/24 P&R 成功** | 初值确实进入 GP，原生优化继续，经过 APPack／SA／真实跨层 route |
| 单层兼容 | 历史单层七项＋旧 v6.8／新 v7 冷暖 GP 四项，**11/11 成功**；冷暖对照的原始 GP XYZ 分别完全一致 | 新接口在已测范围保留原二维行为；不是所有二维设计的性能等价证明 |
| C++ 输入契约 | 非零层、XYZ checksum、非法／分数层、重复／漏 atom、molecule、fixed XYZ 等八项通过 | 拒绝畸形或违约初值，不静默修正输入 |
| H×Z 场景 | 四架构 × W64/128，共 **8 份真实 RRG**，审核通过 | 资源和独立 Z 供给匹配，垂直位置对照真实存在 |
| 小模型训练／部署 | 141 APBlocks、18 叶区域，真实教师训练 20 epochs，权重发生变化；Default／B1／GNN 都完成 P&R | 图→训练→初值→新后端闭环可执行；不是泛化和性能证据 |
| 编排器冷／常驻路径 | 独立工程 fixture 完成图、教师、冷启动三组部署及常驻重复请求 | 进程启动、图准备、模型载入和请求成本可以分别记录 |
| 最终源码契约回归 | WSL／PyTorch2.6 CPU 下 **47项全部通过，0跳过**，含八项编译C++契约、模式需求、模型训练、门禁、存储、锁和真实TERM／INT恢复 | 验证新源码工程约束，不是47次P&R或性能样本 |

最终编排集成使用本轮冻结后的模式需求和恢复代码，关键九个源码文件运行前后哈希一致，141AP／18叶的真实prepare、两个完整教师、冷启动B1／GNN、三组route与两种常驻重复请求均成功，未做性能验收。见 [最终集成报告](runs/v7_campaign_adapter_final_20261025/integration_report.json)。早期adapter目录保留为历史，不用它代替最终源码证据。

核心资格证据：[qualification_report.json](runs/v7_qualification_20261024/qualification_report.json)。完整报告包含每项检查、输入／binary 哈希、实际命令和状态。最初 harness 错误地要求“全在 layer1 的教师翻层后仍含非零层”；修正为核对请求分布，原 false 报告保留，24 次 VPR 结果未重跑或改写。初值含非零层的覆盖由其他回放／翻层条件保证。

单层证据：[backend 结果](runs/v7_backend_20261024/)。新 binary 的基础 commit 为 `40f8d19708b0b60e2aa04452deb1cfed68d96b0b`，组合为完整层间异构 v2 修复＋新 XYZ hook，SHA-256 为 `ff5a39c53c7e0be00e68bb5c41afbc521c7d1ff0512dd44047b60396742f0a71`。

最终回归保存 [机器可读报告](runs/v7_verification_20261025/unit_tests.json) 和 [逐项日志](runs/v7_verification_20261025/unit_tests.log)，包含被测源码哈希、Python／Torch／平台版本。TERM／INT测试只终止本模块登记的子进程组并保护无关session；SIGKILL无法捕获，仍依赖平台调度器回收。误启动双份stage由同OUT文件锁拒绝，轻量归档拒绝源／目标互相嵌套以防递归膨胀。

本地构建使用受审计的原静态库＋五个修改对象增量链接，manifest 保存来源、源码、flags 和命令，**没有称为 clean rebuild**。服务器提供完整源码下的常规 CMake installer；新编译二进制须重新核验其资格，不能直接借用本地 binary 哈希。

### 2.1 四格器件的控制是否成立

12×12×2 的实际总资源四格均为 CLB120、DSP40、RAM40、IO subtile640。H0 每层 DSP20／RAM20；H1 为 DSP30／10、RAM10／30。Z0 分散、Z1 集中，都是 25 个位置、每位置 6 个相同双向 Z 资源，去重独立容量均150。

实际位置的容量加权平方半径从16.50降至4.02 tile²；方向、segment 及两类邻接 switch 预算一致。W64→128 时 CHANX、CHANY 各从4284增至8604节点，Z容量仍150。此结论只覆盖已测 grid／width；自然设计使用各自实际 RRG 重审。详见 [架构实测报告](RESULTS_ARCHITECTURES_2026-10-24.md)。D100 是冻结的参考位置预算，不表示原 XML 所有位置。

### 2.2 训练测试的证据与负结果

小模型使用一个开发 fixture 的完整原生 GP 教师，不平均多个解；raw XYZ 与投影目标分文件保留。20 epochs 的教师损失从3.156043降至2.019902。checkpoint 固定取 epoch20 用于工程测试，未经 validation 选模；没有自然设计的收益反馈训练。见 [smoke_report.json](runs/v7_learned_smoke_20261025/smoke_report.json)、[training_report.json](runs/v7_learned_smoke_20261025/model/training_report.json)。

最初 W128 的同类 fixture 正常不可布通，保留为 [W128 负结果](runs/v7_w128_negative_20261025/)。随后只用 Default 取得 W320 工程宽裕参考，三组都在该条件执行。这是接口测试范围调整，不是最小宽度、优胜候选挑选或研究收益。

单独的双层单 DSP／RAM site 测试复用旧小模型，三组最终都成功，并证明 GNN 非零层初值和真实 CHANZ 使用。它还暴露旧 DSP demand=1 对最大 mode capacity=2 的模式计数过宽：两个全宽乘法器初始同层，后端才拆开。旧报告的严格检查 **false 保留**：[原始模式边界](runs/v7_forced_layer_smoke_20261025/README.md)。

随后只修正新 v7 的 DSP 需求适配器：读取同一 XML 的互斥 mode 和 mapped BLIF 端口宽度，以小模式等价单位计需求，27×27为2、18×19为1；器件capacity仍2，未扩大资源。未知或复杂未覆盖模式明确拒绝。新图重建后复用原 checkpoint，不调参、不重新训练，**Default／B1／GNN三次P&R全部成功；B1／GNN初始及最终DSP、RAM均两层各一，严格检查通过**，GNN实际使用38个CHANZ节点。见 [模式修正复测](runs/v7_forced_layer_smoke_20261025_v2/README.md) 和 [需求语义说明](../vtr/tailcompile_v7_3d/DSP_MODE_DEMAND.md)。

旧图／旧权重的训练语义不会被自动升级；正式训练按新签名重建图、生成目标和训练。重用旧 checkpoint 的本次试验只验证工程路径。多资源 mode 冲突仍不由五维代理完全刻画，最终 packing 合法性由 APPack 保证。

模式修正前的数据适配器按旧运行中的完整SHA精确恢复并归档；恢复方法和核验依据见 [历史源码说明](source_archive/e66b31da1786f0502796483a1673f11ed0c838dbdc250b43472c399279987f2b/README.md)。旧实验和新版实验没有混用缓存。

## 3. 自然设计输入资格：已知与未知

以下为已有 mapped BLIF 在新原生 AP 结构入口的实际盘点，**不是已完成 Default P&R 或正式教师库**。旧五维需求中 DSP／memory 按 APBlock 计一次；后续模式等价需求与 APBlock 数分账，不据单位变化重新贴重／轻标签。

| 设计族 | 结构盘点 | 当前作用／状态 |
|---|---|---|
| SHA | 2778 APBlocks，RAM0／DSP0 | 轻控制候选；grid16×16，待 Default 校准 |
| OR1200 | 4775 APBlocks，RAM4／DSP4 | 轻控制候选；grid20×20，待 Default 校准 |
| SPMV | 13718 APBlocks，RAM232／DSP32 | 已确认重硬资源候选；grid64×64，待 Default 校准 |
| MCML | 本地综合120秒预算内未完成，硬资源需求未知 | 第二 heavy 候选，服务器优先完成映射和容量资格 |
| attention_layer | 既有网表含 Hecate 不支持的 `addition_fp_clk_16` | 排除当前原语范围，不通过临时添加新原语凑数 |
| TPU small | 既有网表的 primitive 端口不兼容 | 等待合适映射，不记成 GNN 失败 |
| stereovision0／3、ch_intrinsics、BGM | 不含或未达到冻结重型门槛 | 不替换成“第二重族”；BGM 11个 DSP 仍低于16门槛 |

MCML／ch_intrinsics／stereovision3 的额外小规模 RTL 探测使用40f8源 RTL和旧 e422完整前端，工具哈希单列；它只供兼容性诊断，不能冒充正式固定40f8映射资格。见 [有界重型候选探测](runs/v7_heavy_input_probe_20261025/README.md)。正式服务器映射复用同 commit 的完整综合根，和新3D VPR-only根分开。

对已有SHA／OR1200／SPMV／BGM原始结构重新解析的新模式审核均通过：DSP等价需求依次为0／8／64／22，而DSP APBlock数仍为0／4／32／11。原审计不覆写；[模式补充审核](runs/v7_input_probe_20261025/mode_aware_addendum.json) 与 [独立二进制／恢复审计](runs/v7_input_probe_20261025/independent_audit.json) 保留输入／实现哈希及适用范围。

当前四族示例是 **候选配置**，不是已准备完成的480任务。prepare 必须核对完整输入和资源组；calibrate 必须给每个族生成四架构共用的两个有效宽度；门禁缺一即停止。重组仍要求至少16个 hard APBlocks，且一种硬资源利用代理≥10%；轻组至多8个。不降低门槛或依据 GNN 效果换设计。

## 4. 接下来在服务器执行什么

1. 在原TailCompile主Git仓库拉取已发布commit，复用完整固定40f8综合根，构建隔离3D后端并核验资格；按新入口使用独立cohort。
2. CPU `plan → prepare`，首先解决MCML和两个validation输入资格。失败先同步日志，不直接派发480。
3. CPU `calibrate → teachers`：仅 Default 选择压力／参考宽度，四格审核实际供给，保存原始和派生教师。
4. GPU `pretrain`；CPU `feedback`；GPU `finetune`。保留正、负、正常失败候选，实际更新部署生成器；基础设施失败不伪造收益标签。
5. CPU `validate → pilot → scan → timing → resident → report`。最多三个预登记 checkpoint 按 validation 真实 VPR 保护和总时间选模；pilot 三方法480任务，Wmin 独立五seed扫描，总时间用固定并发、同核配对和轮换顺序。

主标准不变：计入初始化／非复用图准备／进程启动的总时间至少下降5%，成功率不下降，五seed平均已测Wmin、CPD、WL各退化≤3%，逐族与预登记分层保护。常驻成本另列；小样例秒级 VPR 时间不能验收研究加速。未通过保护的 validation 可以按预登记做一次明确探索性 pilot，但状态仍是性能未通过。

训练器采用有界图缓存，CPU/GPU分阶段交接；VPR目标16～32并发，按真实CPU配额／物理核／内存决定有效值。有限密度处理固定预算，没有重新加入难收敛的修正搜索。详见 [完整命令](../vtr/tailcompile_v7_3d/README_SERVER.md)。

## 5. 本地存储与恢复

已按用户要求，在指标、日志、输入哈希落盘且无活动依赖后，清理本轮失败构建副本、重复 RRG、完成任务的派生 `.net`／post-routing net／`.route`；保留唯一输入、初值、教师、checkpoint、器件图和轻量证据。没有按扩展名批量清空原实验。

主要账本：

- [构建副本清理](runs/v7_backend_20261024/buildcopy_cleanup_plan_applied.json)：已分配空间1,430,024,192 bytes，实测free-space增量核验。
- [W320参考原始RRG](runs/v7_backend_20261024/w320_reference_rrg_cleanup.json)：已分配空间81,133,568 bytes；图、教师、模型哈希不变。
- [原型重复RRG／pack文件](runs/v7_hz_audit_20261024_final/prototype_payload_cleanup.json)：逻辑文件大小751,829,700 bytes，未将逻辑大小误称为文件系统已分配空间。
- [24次资格归档](runs/v7_qualification_20261024/archive_manifest.json)：72份完成派生文件、逻辑82,348,538 bytes；另有smoke及编排试验清理记录。

合计回收量约2GiB量级，物理空间与逻辑文件大小分账，不作精确混加。原始route／RRG深度重分析需按记录命令重建；现有结果和器件图复用不需要这些raw文件。服务器也默认在完成指标提取后做限定目录清理，并禁用core dump。

## 6. 已完成与未完成的最终边界

**已完成：**工程代码、版本化XYZ接口、24核心资格和11单层回归、小器件H×Z真实性、真实小模型训练／部署、分阶段服务器入口、续跑和清理机制。

**仍待执行：**第二heavy及validation输入资格、自然设计Default校准和教师、真实收益训练和validation选模、480开发比较、Wmin和受控runtime验收、未见族测试、作者工具／Vivado外部比较。没有将这些待执行项填为通过，也没有声称TailRisk、RRG或layer head的独立因果贡献。
