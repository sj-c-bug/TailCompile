# 论文外部对比补充：VTR、Escaping/Beyond Flatland 与 Vivado 2022.1

日期：2026-10-23
状态：**待执行方案；已完成源码/文档核查，尚未运行外部对比或校准。**
关联：[3D 主实验协议](EXPERIMENT_PLAN_3D_V68_2026-10-23.md)。本文补齐论文外部证据，不替换其中的内部归因矩阵、分族划分或固定 VTR 后端。

## 0. 最新执行范围：先做轻量商业参照

按用户最新决定，**当前不接 RapidWright，不开展同 Vivado 后端的 E4 适配**。GNN 验收前的任务安排以 [准备期执行计划](PRE_GNN_EXECUTION_PLAN_2026-10-23.md) 为准。

- 商业首批只选 XC7A35T / `XC7A35T-like` 一档、stock XML、W=190、clustered routing、一个目标周期 10 ns、Vivado 默认策略。
- 三个原语资格例×两工具＝6 次；三个 train 开发设计×（VTR 两个 seeds＋Vivado 一次）＝9 次，共 **15 次准备运行**。GNN 和 7 系列 adapter 验收后，同条件再补 6 次 TailCompile，不能用临时启发式占据其结果列。
- 小例仅做功能、资源及明显时序异常筛查，**不声明完成 C1 精度校准**。完整 part/OOC 实现资格、DSP 时序限制、失败处理和结论边界仍须遵守。
- 第二档器件、多策略/周期、96 次 C1 微基准、7 系列衍生双层 E3-Z、E4/RapidWright 暂不启动。下文对应设计与 §8 的 512/560 次是完整扩展参考，**不是当前派发队列，也不是本轮必须完成的任务**。
- BF 作者流程保留有限资格支线，不因商业缩减而取消学术方法比较；其正式大表仍按主阶段冻结。

以下条款中出现的“双器件档、双时钟、多策略”等，均以本节当前轻量范围为先；共同输入、无测试泄漏和不夸大模型精度的要求不缩减。

## 1. 决定与结论边界

采用用户建议：**同一 benchmark，先用 VTR 自带 Xilinx 7 系列 XML 与真实 7 系列器件建立轻量商业参照。** 总资源匹配的 7 系列衍生 3D 仅保留为后续可选方案。商业工具固定为服务器已有的 Vivado 2022.1，不因当前服务器主要使用 Ultra 系列就改用 UltraScale 的资源模型。

原方案只充分安排了内部消融，外部方法比较不应等全部实验跑完才补。现在把以下面板提前纳入任务图：

| 面板 | 比较 | 地位 | 能支持的结论 |
|---|---|---|---|
| E1：同后端外部初始化 | Native GP、capacity-only、划分型 APBlock prior、TailCompile；相同 XML/VTR | 论文方法对比，须尝试适配 | 初始化方法的增量效果 |
| E2：已有完整工具 | 作者 Beyond Flatland baseline/full flow；我们的 VTR baseline/TailCompile | 论文必须尝试的外部复现 | 共同支持条件下完整流程的效果及成本 |
| E3：商业参照 | Vivado 2022.1 实际 Artix-7；7-series-like VTR Default/TailCompile | 优先执行 | 近似相同逻辑资源预算下的系统性能参照 |
| E3-Z：3D 商业参照 | E3 中的 Vivado结果；总资源匹配的研究双层 VTR Default/TailCompile | 有条件扩展，单独标注 | 假想 3D 架构与商用实现的距离，非同芯片算法排名 |
| E4：同 Vivado 后端 | 同 DCP/part/XDC 下的 Default、capacity-only prior、学习 prior | 可选后续适配，不阻塞 E1–E3 | 我们的方法能否真正改善 Vivado 实现 |

“资源相近的 XML + TailCompile 比 Vivado 快/频率高”可以作为带条件的**完整系统观察**，但不能直接写成“布局算法优于 Vivado”。综合、packing、资源模式、routing、工艺/延迟和时钟模型仍有差异。算法的主要因果证据来自 E1，以及未来通过资格验收的 E4。

不声称支持了所有异构器件，也不以某工具不支持的输入作为其性能失败。论文必须保留支持范围、失败和缺测表。

## 2. E1/E2：已有学术方法如何公平比较

### 2.1 名称、来源及方法边界

正式论文题名是 **Escaping Flatland: A Placement Flow for Enabling 3D FPGAs**，论文指向作者 `IY2002/Beyond-Flatland` 仓库；因此论文引用正式题名，运行记录同时写仓库名。不是从搜索结果随便选一个同名 fork。[论文](https://arxiv.org/html/2604.01078v1)、[作者仓库](https://github.com/IY2002/Beyond-Flatland/tree/vtr-3d)。

原方法包含 post-pack TritonPart 层分配及后续 3D SA 优化，不仅是初始化。作者基础版本也不等同本项目的 `40f8d197`；须固定作者 commit、依赖和 baseline 配置。因此安排两个不同层级：

- **E1 / TP-AP-adapted**：把划分型 prior 适配到 APBlock，接相同需求上界解码、warm hook、GP、APPack 和后端。清楚记录与原论文逻辑单元/容量处理的差异，命名为 adapted/inspired baseline，不能冒称原论文完整复现。
- **E2 / BF-full**：保留作者完整流程，与作者自己的 baseline 配对；另列我们的固定 VTR baseline 与 TailCompile。既看绝对值，也看每套实现相对自身 baseline 的改善。后者降低版本因素的影响，但不自动消除所有系统差异。作者 baseline 必须采用可追溯的 baseline commit，或确认所有新增算法均已禁用；仅关闭 TritonPart、不关闭其他移动/代价修改，不能称为原始 baseline。

E1 所有 prior 均为同样的软初始层，不允许一方固定层、另一方可跨层。外部分区和所有准备开销计入运行时间。移植不应成为另一个方法优化项目：两轮 correctness 适配后仍无法保持清晰语义，则报告 E1 未实现，E2 仍继续。

### 2.2 共同比较条件

1. 同 RTL/参数/功能；共同映射 BLIF 若确实兼容则优先复用，并校验 primitive 接口。不能因为文件后缀一样就认为同一网表。
2. 同 XML、固定 device layout、channel width、时钟约束和硬件预算；审计不同 backend 产生的 RRG/实际资源，XML 相同本身不保证所有底层实现一致。
3. BF 使用 post-pack 对象，我们使用 pre-GP APBlock/APPack。E2 的 packing 差异属于完整系统，不宣称“固定同一个 packed netlist”。若强制 frozen packing，那是另一个后端实验，不能代替 v6.8 路线。
4. 首选公开固定参数；必要的适配只在已见开发族上进行，参数预算提前登记，不在测试族上调到胜出。
5. 运行 SHA/OR1200 等开发族和 H0/H1 的有限支持性测试；没有实际测试，不预判 BF 必然支持或不支持异构层。
6. 共同支持面板包括所有正常方法失败；`unsupported` 单列，不填成极差的 CPD。不得只留下大家都成功的结果表。

E2 优先复用主协议四架构×两宽度×三 seeds 的条件。若作者流程不能在这些架构工作，另用作者公开 CB/SB 两种架构，在方法效果揭盲前冻结共同支持清单，四个流程一起运行；不得拿不同架构的结果拼成同条件排行榜。

资格阶段最多两个开发族×两架构×baseline/full 的 8 次首轮 P&R，以及一轮修复复验。不重新优化作者的 SA 参数。时序、线长与时间按同一采集口径重测，不将作者历史 CSV 的共享服务器时间与我们的本机时间直接比较。

## 3. 7 系列参照：本地已经核实的具体对象

### 3.1 固定 XML，不另造一个“看起来像 Xilinx”的器件

固定 VTR `40f8d19708b0b60e2aa04452deb1cfed68d96b0b` 内已存在：

```text
vtr_flow/arch/xilinx/7series_BRAM_DSP_carry.xml
vtr_flow/arch/xilinx/7series_BRAM_DSP_carry-unmodified_length.xml
vtr_flow/arch/xilinx/simple-7series.xml
vtr_flow/arch/xilinx/simple-7series_diagonal.xml
```

**首选 `7series_BRAM_DSP_carry.xml`。** 本地核验的 SHA-256 为：

```text
f3884f74332a7e98804809a66076205d5b395319729fcee0b6086ca004e4883b
```

它包含完整度更高的 CLB/carry/BRAM/DSP 和按 Artix-7 资源预算构造的 fixed layouts；`simple-*` 用于特定 routing 能力测试，不用作混合 DSP/BRAM 基准的主要商业参照。`unmodified_length` 涉及另一套 routing 建模前提，首轮不混用。

源码明确说明该文件是近似模型：资源预算接近，但物理排布、部分 switchblock 连接和 timing 并非商业器件精确重建。公开文件入口：[VTR Xilinx architectures](https://github.com/verilog-to-routing/vtr-verilog-to-routing/tree/master/vtr_flow/arch/xilinx)；执行以**上面固定 commit 和本地哈希**为准，不自动跟随 master。

### 3.2 候选器件档：当前先用 35T，100T 留作扩展

| 商业候选 | LUT6 / FF | DSP48E1 | BRAM36 | 对应 XML fixed layout |
|---|---:|---:|---:|---|
| XC7A35T | 20,800 / 41,600 | 90 | 50 | `XC7A35T-like`，58×58 |
| XC7A100T | 63,400 / 126,800 | 240 | 135 | `XC7A100T-like`，100×100 |

商业资源口径依据 [DS180](https://docs.amd.com/v/u/en-US/ds180_7Series_Overview) 和 [UG474](https://docs.amd.com/r/en-US/ug474_7Series_CLB/7-Series-FPGA-CLB-Resources)。XML 注释对应为 2,576 / 7,840 个 CLB tile、45 / 120 个 DSP tile、50 / 135 个 BRAM tile；一个 DSP tile 含两枚乘法器，不能按 tile 个数与商业 DSP slice 一比一。

上述 XML 数量目前是源文件声明，仍需实际 device grid/RRG census 核对；不把注释当作运行验收。每 CLB 的物理 LUT6/FF 能力、可分裂模式和可用 IO 分开报告。尤其 package 限制下的 IO 数可能小于器件家族最大值，完整 part 需要冻结 package、speed grade 和 temperature grade。

Vivado 2022.1 官方支持表包含这两款 Artix-7，且列入当时的 ML Standard 支持范围；因此不需要因为版本是 2022.1 而换工具。但服务器是否安装对应 device files、选定完整 part 是否能实际实现，仍要检查。[UG973 2022.1](https://docs.amd.com/r/2022.1-English/ug973-vivado-release-notes-install-license/Supported-Devices)

服务器第一步只盘点，不默认安装或更改许可证；在 Vivado Tcl 中执行：

```tcl
puts "VIVADO_VERSION=[version -short]"
foreach pattern {xc7a35t* xc7a100t*} {
    set candidates [get_parts $pattern]
    puts "PATTERN=$pattern COUNT=[llength $candidates]"
    foreach part $candidates { report_property $part }
}
```

当前从结果中选一个 35T 完整 part；100T 仅记录可用性，扩展时优先同一 speed grade。再用 LUT/FF、BRAM、DSP 小例验收。软件支持、器件包已安装、实际运行通过是三个独立状态。**只做综合/P&R/STA 比较不要求手头有对应开发板；没有上板测量时一律称工具预测结果。**

### 3.3 不可跳过的四项模型边界

1. **固定 channel width = 190。** 源文件明确要求 190 或其倍数以维护部分 diagonal wire 连通性；本轮固定 190。主 3D 协议的 32/48/… 宽度扫描不能照搬过来，也不增加宽度去追平 Vivado 的性能。
2. **DSP 时序需要审计。** 头部写没有可信 DSP timing；正文实际上有 25×18 乘法的 max=1.523 ns、min=0.776 ns 常数，但旁边明确注明不准确。因此不是“没有数字”，而是“现有数字不能直接当作已校准的 DSP48E1 延迟”。
3. **功能与面积不是完整复刻。** DSP 主要为乘法模式，不代表完整预加器/ALU/寄存/级联；BRAM 模式和 CLB mux/carry 也有范围限制。area 数值借用其他架构，不能用于对商业芯片的 mm² 或面积效率结论。
4. **多高度与链结构要验收。** BRAM/DSP 高度为 4，carry direct 跨相邻 CLB。必须验证 root-anchor 计数、APBlock/molecule 与合法化；不预设可运行，也不预设一定不支持。

这些限制从固定源码直接核查，不以 XML 名称含“7series”代替模型验证。

**路由模式也在校准前冻结。** 首轮明确使用 `--flat_routing off`，与现有 v6.8 流程和固定 VTR 的默认 clustered routing 一致。固定源码的 `vtr_xilinx_qor` 参考回归采用 W=190、`--flat_routing on`，因此不能直接拿其 golden 结果替本流程背书。C0 必须验收选定模式；微基准、Default、TailCompile 和正式实验使用同一模式，按该模式生成/记录 RRG。若资格失败，先定位是否确为模式问题；任何改为 flat routing 的决定都必须在 C1 和正式测试前整体修订协议、验证图接口，不能只给某一方法切模式或事后挑选较好结果。

## 4. “校准”具体做什么：有限校核，不重建整个芯片

### C0：物理资源和功能校核

分别导出 physical-site census 与 mode-aware capacity。资源对标使用：

`LUT6物理位置、FF、DSP运算模式/位宽/内部寄存配置、BRAM数据位/端口/宽深/读延迟、IO及clock/carry限制`。

不能直接用 TailCompile 的上界代理当芯片规格：当前 v6.8 对该 XML 的 CLB 可能得到 16 个可分裂 LUT 模式容量，而物理上是 8 个 LUT6 位置；BRAM 的两个小块模式也不等于两个独立 BRAM36。BRAM18/36 互斥，parity 位和可用数据位要分别记录。[7-series CLB](https://docs.amd.com/r/en-US/ug474_7Series_CLB/CLB-Overview)、[7-series memory](https://docs.amd.com/v/u/en-US/ug473_7Series_Memory_Resources)

此外，现有 `tailcompile_v68/data.py` 的硬资源识别偏向 `memory/dsp_top/io` 名称，尚未把该 XML 顶层 `DSP` 正确转换为 `dsp_top` 供给。**需要在隔离的 7 系列 adapter 中修复类型映射并单测，不改原 2D release。**同时审计两个 25×18 DSP 子块、RAM 模式、carry 原子性与 root anchors。只给 parser 增加别名不能替代全链验收。

资格要求：资源计数可追溯、支持模式功能一致、三个原语小例能正常综合/布局布线、TailCompile 初值实际被 GP 消费。Default 可以先校核；TailCompile adapter 未通过时该行标未实现，不能用旧贪心初值冒充。

### C1：完整精度校准参考（当前暂不启动）

预登记六类结构：LUT/FF 组合路径、carry 加法、BRAM、25×18 乘法、局部互连、高扇出/长距离互连。每类四个参数实例：两个用于诊断/允许的校准，两个严格留出；共 24 个小例。两个器件档、两套工具各一次，原始模型共 **96 次 P&R**，分为先诊断 48 次、后留出 48 次，另有少量资格运行。先只查看 12 个诊断实例；决定是否修订并冻结 XML 后，才运行/揭盲另外 12 个留出实例。留出失败只能降低模型可信度，不能再用于拟合第二版。

- 固定功能、位宽、signedness、latency 和端口模式。对原语 arc 校核可使用工具专属、功能等价的 wrapper；对自然 benchmark 不据此宣称双方映射网表相同。
- DSP 校核采用共同支持的乘法模式，明确输入/输出寄存配置。Vivado 若把寄存器吸收到 DSP，必须记录，不能用不同 pipeline 比时延。
- 保存 max/min 路径、cell/net delay 分解、实际放置/路由及 endpoint 类别。冻结商业 part/speed grade、温度/电压和 timing corner；max/min 与不同 corner 分开比较，原语和寄存配置必须一致。XML 无可靠 PVT 来源或无对应 min delay 时标为不可比，不能冒称完成 hold 校准。跨工具的长连线只有经过长度/扇出/逻辑深度分层后的统计可比性，不能强行认作同一物理路径。
- 首先运行原始 XML。最多一次有物理来源的修订，优先修正明确错误/占位的 DSP arc；所有参数来源、脚本和差异留档。**不拟合一组任意系数使最终 benchmark CPD 接近 Vivado**，不据 TailCompile 是否获益选择 XML。
- 同 primitive、相同寄存模式且有可对应 corner 的 cell arc，分别对 max/min 预设检查目标为留出中位相对误差≤20%、P90≤35%；这是本研究的资格阈值，不是官方精度保证。小延迟分母使用冻结的 0.05 ns 下限。互连单独报告误差和分层趋势，不能用 cell arc 合格宣称 routing 精确。
- 发现未建模功能、DSP 时序仍不可信或 route 结构偏差大，停止精确时序校准主张；保留其为资源匹配、定性系统参照。DSP 相关数据不能悄悄混入“校准通过”的 Fmax 总表。

**预算出口：**资源适配两轮 correctness 修复；微基准一轮原始运行、最多一次可解释修订及留出检验，工程检查点约两个工作日。仍不满足时降级证据，不转为无限期商业 FPGA 建模研究。纯 LUT/FF/carry 等已通过子集可以继续，但要预先列明子集规则及覆盖率，不按性能胜负筛样本。

若确需一次修订，复用原 Vivado 结果，仅增加修订 XML 的诊断 VTR 运行至多 12×2=24 次及留出 VTR 运行 12×2=24 次，即最多额外 48 次。stock/calibrated 的留出结果都在修订冻结后揭盲；不把留出集偷偷改名为第二轮诊断集。

输出 `stock.xml` 引用、必要时的独立 `calibrated.xml`、`calibration_manifest.json`、逐类误差表与未建模模式清单。冻结后才揭盲正式 benchmark；同一 XML 必须同时供 VTR Default 与 TailCompile 使用。

## 5. E3：同 benchmark 的商业参照矩阵

### 5.1 输入、映射与规模

- 共用同一 RTL 源码、top、参数、位宽、复位语义、memory 初始化及功能测试；保存哈希。
- 不因一边资源紧张而单独缩小设计。装不下记 `capacity_infeasible`；完整表保留共同资格范围与规模覆盖，不能只留下两边成功的情况。
- 初轮用主协议四个留出族中可共同表达的 RTL，随后按封存清单扩到至少八族。校准微基准与已见开发族不进入未见测试统计。
- 保存两边综合后的 LUT/FF/BRAM/DSP/状态位、latency/II 和推断模式；同 RTL 不等于同 technology-mapped netlist。使用功能仿真，能形式等价验证的范围再加等价检查；只做仿真不能称为形式证明。
- 保留 vendor-native 映射为主表，不为了“公平”只关闭 Vivado 的正常优化。若需要共同原语子集/关闭某类优化，另立对称诊断表，不拿它冒充商业工具最佳正常流程。
- 小设计在 100T 上利用率很低时如实报告，不靠复制 benchmark 到不同规模追平利用率。参数化负载扫描若另做，同一实例必须同时送给全部流程，并归入同一设计族。

### 5.2 时钟、IO 和综合优化

首轮采用 core/OOC 口径，主要比较相同 register-to-register 功能路径：避免包引脚数量差异先决定输赢。Vivado 的 OOC 综合可避免自动 IO buffer 插入；它不是完整板级系统实现。[UG835 synth_design](https://docs.amd.com/r/2022.1-English/ug835-vivado-tcl-commands/synth_design)

在 C0 中先完成一个实际的 Vivado 2022.1 OOC 综合→place→route→STA 资格例，不能只凭 OOC 综合成功就开始批跑。冻结实现入口、时钟源/BUFG 的处理、理想或传播时钟假设、输入输出例外及 register-to-register 评价范围；记录 OOC 必要且合理的 DRC 例外与原因，不允许批量豁免 DRC 来冒充完整实现闭合。

必须显式处理 clock、input/output delay、false/multicycle path、uncertainty；检查没有意外 unconstrained 内部端点。IO 路径若不纳入主时序指标，说明其 scope，不能把未约束路径当作闭合。不能把 VPR 的周边 IO 模型说成与 OOC 端口物理位置相同；IO 影响属于剩余混淆。完整带 pin assignment 的实现另表，不与 OOC 混在一起。

当前轻量版仅用 **10 ns**。完整扩展可预登记 **10 ns、5 ns** 两个共同目标；若开发集显示不适用，只允许在正式测试前按同一规则修订并冻结。两边均使用相同目标，不为某工具挑容易满足的时钟。

每个 tool×design×device 的基准综合配置固定，在 10 ns 参考约束下生成一次输入，再在各 implementation 周期/策略间复用该工具自己的综合 checkpoint/BLIF。明确这是固定综合输入的 implementation 扫描；若另做每周期重新综合的完整 flow，也须对全部工具对称执行并单列成本。

Retiming、寄存复制、DSP 吸收、RAM 推断等选项与实际变化都记录。需要严格保持 latency 的比较必须核验周期级功能；保持共同输入不代表 physical optimization 后网表还完全相同。

### 5.3 必须同时有的三列

| 流程列 | 固定对象 | 主用途 |
|---|---|---|
| Vivado 2022.1 | 真实 XC7A35T / XC7A100T 的完整 part | 商业实现参照 |
| VTR-7S-2D | 对应 `*-like` fixed layout、W=190、同一时序模型 | 验证近似模型的系统差距；Default→TailCompile 的算法增量 |
| VTR-7S-derived-3D（条件扩展） | 保持全局资源预算的假想双层、独立 Z 供给假设 | 架构探索；不是实际存在的 3D Artix-7 |

2D 列先行，不能直接拿另一种自定义双层架构与 Artix-7 做一个无中间参照的算法排名。7 系列模型也**不替换**自定义 3D 主实验（当前匹配构造基于 Hecate）：主架构负责受控 3D 因子结论，这一面板负责商业相关性。

TailCompile 在 7 系列首先用冻结 checkpoint + 经过资格验收的语义 adapter，标明 architecture transfer。若需 7 系列 train-only 教师微调，在运行正式测试前另冻结模型版本、训练预算和数据；不能用本面板 test 结果微调后还称零样本迁移。

### 5.4 策略与重复

- 当前 Vivado 只用 2022.1 默认 implementation 策略。完整扩展再另列预登记的 `Performance_Explore`，实际名称与展开 Tcl 以安装版本可用列表核验；两列分别展示，不逐 benchmark 偷挑较好者。
- 当前开发 pilot 的 VTR 用 501/502；正式扩展的 Default 和 TailCompile 才使用相同 541/542/543 三 seeds，报告分布，不把开发结果冒充正式重复。按 §3.3 冻结 clustered routing 及所有 flags。
- 不为 Vivado 编造与 VPR 同名的随机 seed 参数。官方 `place_design` 公开接口以 directive 等控制行为；同配置重复主要用于计时稳定性，不当成独立 QoR seed。[place_design 文档](https://docs.amd.com/r/2022.1-English/ug835-vivado-tcl-commands/place_design)
- 若另报告 best-of-budget，各工具的候选/策略数量、总 CPU 时间和选择规则须对称、预登记；不把 Vivado 多策略最佳值与我们单次或反过来做主表。

## 6. 指标：可以比较什么，不能比较什么

### 必报

1. 在同目标周期下，route 是否完整、setup 是否满足；Vivado 同时报 hold/pulse-width/DRC 和 unconstrained 检查。若 VPR 不能提供等价 signoff 检查，字段标 unavailable，不合成一个看似同标准的“全部时序通过率”。
2. routed setup timing、关键路径类型及 cell/net delay 分解；**VPR 模型预测**与**Vivado 器件数据库 STA**明确标列，均不是硬件实测。
3. 物理容量和已用资源向量、模式、利用率；资源量匹配不代表映射后的利用率相同。
4. P&R 阶段时间、RTL→route 总时间、峰值 RSS；图生成/推理/修正/额外 prepass 全计入，训练成本另列。
5. 每个 VTR 架构内部的 Default→TailCompile 相对改善；E2 作者工具也相对作者 baseline 单独计算。

“最高已测通过频率”必须来自统一、预登记的目标周期扫描，并满足所声明的检查集合。当前只有 10 ns 一个目标，或扩展只有 10/5 ns 两点时，只报告对应目标下的结果，不声称精确 Fmax。需要更细频率曲线时在揭盲前登记 `[20,10,6.667,5] ns` 等共同网格，所有方法一起执行；禁止把任意 WNS 代入 `1/(T-WNS)` 当作通用硬件 Fmax。多时钟/多周期路径单列；II 不同时同时报告 `F/II` 与 latency。

Vivado 至少留存 `report_timing_summary`、`check_timing`、`report_utilization`、`report_route_status`、`report_drc` 和 post-route DCP；每个命令参数按 2022.1 `help` 校验。route 命令退出成功或 setup WNS≥0，均不能独自代表整个实现正确。[timing summary](https://docs.amd.com/r/2022.1-English/ug835-vivado-tcl-commands/report_timing_summary?contentId=OhAfwSPrk8sIZi0~UELO8Q)、[route status](https://docs.amd.com/r/2022.1-English/ug835-vivado-tcl-commands/report_route_status?contentId=Xdoq2Kq~mJGhx1~BvzSaGw)

### 不作为跨工具等价指标

- 不把 VPR track 长度、Vivado wire/node 数、粗需求 P99 画成单位相同的拥塞/线长排行榜；分别报告工具内定义。
- 不用此 XML 的 area 参数比较 mm²；不做线性工艺节点缩放来消除 40/28/22nm 来源差异。
- 没有同样 activity、corner 和可靠模型时，不做跨工具能耗优越性主张。
- 不把一个裁剪 Pblock 当成资源等量的独立小 FPGA：放置范围不自动限制所有外部路由。若使用 Pblock，须有同 Pblock 的 Default 对照并单独标注，不替代 whole-device 主表。[CONTAIN_ROUTING](https://docs.amd.com/r/2022.1-English/ug912-vivado-properties/CONTAIN_ROUTING)

## 7. 3D、商业 SLR 与同 Vivado 适配

### E3-Z：假想双层的有限扩展

只在 2D 模型与 adapter 验收后登记一组总 CLB/FF/DSP/BRAM 数量匹配的双层衍生架构；不要简单复制整片两次、把资源翻倍还称相同预算。每层 geometry、硬资源位置、Z 资源/延迟和 channel width 明确列出，并对 Default/TailCompile 共用。

2D 微基准校准只能支持复用的层内原语/路由近似，**不能校准现实不存在的 3D Artix 垂直互连**。Z 参数以独立架构假设和预登记敏感性分析处理；不能从 Vivado 2D 的 timing 反推出 CHANZ。若七系列 carry/多高度资源的双层支持导致范围明显扩大，就暂停这一衍生面板，继续自定义 3D 主实验，不再开一个芯片重建项目。

### 商业 SLR 不是该 XML 的上下物理层

UltraScale/部分其他商用多 die 产品的 SSI/SLR 通过 interposer/SLL 连接，可以作为跨分区稀缺互连的另一个外部场景，但不能把 GUI 上下排列的 SLR 当作任意 CHANZ 的垂直堆叠。此轮先做已选的 Artix-7 单器件参照，不顺带扩大为 UltraScale 实验。[USER_SLR_ASSIGNMENT](https://docs.amd.com/r/2022.1-English/ug912-vivado-properties/USER_SLR_ASSIGNMENT)

### E4：同 Vivado 后端的扩展参考（当前不做，含 RapidWright 适配）

如果后续要声称“TailCompile 改善 Vivado”，需要同一个 post-synth/opt DCP、part、XDC 和 strategy 下比较：

`无引导 / capacity-only 引导 / 学习式引导`。

候选接口为受支持的 soft Pblock；多 SLR 器件再考虑 soft SLR assignment。它们不是 VTR `.fplace` 的直接导入，也不保证与 GP warm-start 同一语义。Vivado cells、carry/DSP/RAM macro、site/clock 区域需要新映射；不能假装已有完整器件 RRG 与时延数据库，缺失供给信息要明确建模。硬 LOC/BEL 或锁层若采用，必须另做约束强度对照。

同一输入 DCP、backend 策略、soft/hard 语义和开销完全配对，保存约束接受情况和实际 placement 变化。先做不超过两个开发例、两个器件、三臂的 12 次资格 P&R；两轮接口定位后仍不通即记录阻塞，不影响 E1–E3。soft 约束可以被工具放松，不能仅凭写出了 XDC 就宣布初值生效。[IS_SOFT](https://docs.amd.com/r/2022.1-English/ug912-vivado-properties/IS_SOFT?contentId=kIrdYCwi3xLfE0JAq09fbg)

## 8. 完整扩展预算参考：当前不启动此大表

以下按四个测试族、两个商业资源档计算；只统计实际资格通过的面板，但所有排除/未实现保留在覆盖表。

| 面板 | 公式 | 次数 |
|---|---|---:|
| E1 新增一个 adapted prior | 4族×4主架构×2宽度×3 VPR seeds | 96 P&R |
| E2 作者 baseline/full，共同主架构均支持时 | 4×4×2×3×2新增流程 | 192 P&R |
| E2 若改用两种作者架构替代上述面板 | 4族×2作者架构×2宽度×3 seeds×4完整流程 | 192 P&R；替代，非自动叠加 |
| C1 初轮独立微基准 | 24实例×2器件档×2工具 | 96 P&R |
| E3 单层商业参照 | 4族×2档×2时钟目标×(VTR Default 3 seeds + Tail 3 seeds + Vivado 2策略) | 128 P&R，其中 Vivado 32 |
| E3-Z 条件扩展 | 4族×2档×2目标×2 VTR流程×3 seeds | 新增96；复用 Vivado 行 |
| E4 有限接口资格 | 2开发例×2器件×3引导臂 | 12 P&R，不计为论文有效性结果 |

E1+E2+C1+E3 共 **512 次新增 P&R 上限口径**，不含资格修复、必要的校准复验、综合、串行计时重复和条件扩展；不是已经运行的数量。完全相同签名才能复用原实验结果。微基准属于 calibration，不算增加 24 个独立测试族。

若使用 C1 唯一一次校准修订，增加至多 48 次 VTR 复验，上述口径变为 560 次；其他排除项仍另记。不得用“512 次”掩盖实际资格与准备成本。

从四族扩到八族时，正式 E1/E2/E3 数量按族数扩展；资格与微基准不因此翻倍。Vivado 重复相同策略只用于计时，不伪造成三个随机 seeds。最多两个预登记策略，禁止测完多策略只报最好一个。

## 9. 执行顺序、产物及论文呈现

### 并行安排

1. **现在就可准备**：Vivado 2022.1 服务器 inventory Tcl、固定 XML 资源 census、支持范围、微基准、E1/E2 输入/版本清单和报告解析器；不需要等待 3D 模型训练完成。
2. 与主协议 P1 并行做 7 系列 adapter 的有限资格、BF 安装/支持性资格；不修改原 2D release，也不切换主后端。
3. 先完成 2D 模型校核；E3 的测试族 Vivado/Default 可提前计算，但 QoR 封存，只检查任务执行及解析正确性。冻结 checkpoint、adapter 和实验配置后才揭盲，并补入完全同条件的 TailCompile 行；不得借提前看到的测试 QoR 选择模型，原始测试结果也不回流训练。
4. 主协议 P4 同时提交 E1/E2/E3 正式任务；不要等方法结果出来后再挑对手、器件或容易取胜的 benchmark。
5. E3-Z/E4 按各自资格与预算单独决定，未完成就诚实限制结论，不阻止已有实验形成清晰论文。

### 新增记录

- `external_protocol.yaml`：面板、软件/模型版本、part/strategy、clock、路由模式、预算和停止规则。
- `tool_support_matrix.*`：按原语、architecture、逻辑单元、backend 和失败类别记录支持情况。
- `resource_match.json`：真实 part、XML fixed layout、物理资源/互斥模式、容量残差、IO/clock/carry 限制。
- `calibration_manifest.json` / `calibration_report.md`：拟合与留出分组、参数来源/修改、误差、未建模模式、是否允许绝对时序解释。
- `workload_manifest.json`：RTL、参数、wrapper、功能检查、时序约束语义、两边综合资源与 latency/II。
- 每次运行的日志、Tcl、XDC/SDC、checkpoint/网表哈希、失败状态、报告、CPU/RSS、QoR 和计时边界；商业 DCP/器件数据按许可与共享权限保存，不默认上传公开仓库。

### 论文主表提前确定

1. **方法效果表**：同 VTR 后端 Default、capacity-only、划分 prior、TailCompile；沿用主协议内部消融解释各组件。
2. **外部完整工具表**：BF baseline/full、VTR baseline/TailCompile；给共同支持范围及系统差异。
3. **商业参照表**：每个 RTL 的 Vivado 2022.1 / 7-series-like VTR Default / TailCompile，列明 part、资源残差、目标周期、时序模型、成功/失败及总耗时。
4. **模型可信度表**：stock/calibrated XML 的微基准误差与未建模功能，防止把近似器件模型误差当算法优势。

允许的结论示例：“在资源接近 XC7A35T 的 7-series-like 模型上，本方法相对同后端 Default 改善 X；与 Vivado 2022.1 的对应实际器件实现仍有 Y 差距，差异包含已列出的映射与模型因素。”

不允许的结论示例：“两个器件 LUT/DSP 数相近，所以 CPD 比值证明我们的布局器比 Vivado 好 X%”；“校准 2D 后已验证真实 3D Artix 性能”；“BF 不支持某输入，所以其可布线性不如我们”。

本方案的实际第一步是服务器 7 系列 part 盘点 + 固定 XML 的资源/时序核验，而不是马上跑大量自然设计。此门槛一次通过后就进入并行对比，不反复把校准或临时适配当成研究终点。
