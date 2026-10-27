# v7 3D 服务器实验入口

初版：2026-10-25；更新：2026-10-27。本目录提供 APBlock→原生 3D GP→APPack 的工程实现，以及分 CPU/GPU 阶段运行的三方法开发实验。遵循 [REPOSITORY_WORKFLOW.md](../../../REPOSITORY_WORKFLOW.md)：源码和轻量结果只在原 TailCompile Git 主仓库同步，VTR 是固定依赖。服务器长训练、480 次 pilot、宽度扫描及受控计时尚未执行；本地小测试不代替这些结果。

服务器需先获取包含本入口的 Git 提交，发布版本以本仓库 commit 和远端同步结果为准。**若v6.10仍在运行，先按[共用实例说明](CO_RUN_V610_2026-10-27.md)检查待更新文件，再更新工作区；避免把其他尚未同步的2D修复一起带入活跃实验。** 日常不再传 ZIP 或创建另一份 TailCompile 源码目录。

## 输入和目录

需要 Python 3、PyTorch、NumPy、VTR 构建依赖、一个完整的同 commit VTR 映射环境，以及本目录依赖的 v6.8/v6.9 模块和 `experiments/vtr3d` helper/patch。无需 RapidWright。

**两个 VTR 根有不同用途：**固定原 2D VTR 只读用于 RTL→BLIF 映射（ABC/Yosys）；新隔离 3D VTR 用于 XYZ GP/APPack/P&R。3D installer 只构建 VPR，不重新编译 ABC/Yosys，也不修改原 2D VTR。

正式 CPU 映射只接受**完整且固定为 `40f8d19708b0b60e2aa04452deb1cfed68d96b0b` 的综合根**。历史本地 `e422` 映射曾用于输入兼容性探测，不是正式教师或方法对比输入的资格证据；不要把该旧根填入下面的变量。若已有 2D 安装不是此 commit，应另建完整 40f8 映射环境，或提供来源、映射版本和哈希已审计的正式 `mapped_blif` 记录；不修改冻结的 2D 安装。

```bash
export TAILCOMPILE_REPO=/inspire/hdd/project/project-public/changshunjie-253114010247/changshunjie-253114010247/workspace/DAC2027/TailCompile
cd "$TAILCOMPILE_REPO"
git pull --ff-only
```

发生 Git 冲突或非 fast-forward 时先处理仓库状态，不能重置或覆盖已有实验。`prepare_server.sh` 从自身路径定位这个 Git 根，不改到 ZIP 目录，也不通过 VTR Git 同步本项目。

默认目录如下：

| 用途 | 目录 |
| --- | --- |
| 唯一源码仓库 | `DAC2027/TailCompile` |
| 固定只读综合依赖 | `DAC2027/vtr_tailcompile/vtr-verilog-to-routing` |
| 独立 3D 后端 | `DAC2027/vtr_tailcompile_3d_v7/vtr-verilog-to-routing` |
| 新版开发输出 | `TailCompile/experiments/vtr/tailcompile_v7_3d/runs/development_002_git` |
| 大型临时产物 | `TailCompile/vtr_work/tailcompile_v7_3d` |
| 统一日志 | `TailCompile/logs` |

固定原 VTR 下必须有完整的 `abc/abc`、`parmys/yosys/yosys` 和 `vtr_flow/`。新 3D 根必须与原 VTR、源码仓库互不包含；不能把它设成原 2D VTR。新 cohort 使用 `development_002_git`，避免与旧交付方式留下的 protocol 混用。

先激活你已有的 Torch 环境并确认 `python3 -c "import torch; print(torch.__version__)"`。脚本继承当前环境，不自动执行 `conda activate`，也不要求 `conda init`。必要时按已有环境设置动态库路径；CPU 节点也需 PyTorch，因为图构建和初始化会用 CPU tensors。

## 首次 CPU 准备

激活已有 Torch 环境后，先检查前提，再后台运行：

```bash
export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_V7_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_002_git"
export TAILCOMPILE_V7_TIMEOUT=3600
entry="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/prepare_server.sh"
bash "$entry" --check
mkdir -p "$TAILCOMPILE_REPO/logs"
log="$TAILCOMPILE_REPO/logs/v7_prepare_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$entry" >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
```

默认入口只做：检查固定依赖→建立独立 3D checkout/build→接口回归→从固定 40f8 综合根映射 SHA→运行资格→`plan`→`prepare`，然后停止。它不启动宽度校准、教师收集、训练或480 pilot。`3600` 是整个设计映射的墙钟预算，不能保证 MCML 在此预算内完成。查看日志以区分映射超时、原语不兼容和容量不符。

入口继承激活环境，无 `conda activate/init`。对原 2D shell 只读，不更改调用者的 `VTR_ROOT`、`TAILCOMPILE_SCRATCH` 或旧入口。兼容旧编排器所需的泛变量仅通过 `env` 传给 3D 子进程；公开配置使用 `TAILCOMPILE_3D_*` / `TAILCOMPILE_V7_*`。入口禁用 core dump 和 bytecode 写入，每次 CLI 用独立 Python 进程。scratch基目录传 `TailCompile/vtr_work`，`run_pipeline.sh` 只追加一次 `tailcompile_v7_3d`。

接口资格最多24次、工程 seeds701/702，需要 `qualification_report.json` 的 `complete=true, passed=true` 及当前 binary hash。已有构建只能在 hook/源码/patch/manifest 均匹配时复用，见 [BACKEND.md](BACKEND.md)。固定 2D 映射依赖的 executable 哈希保存并在成功结束时复核；不依赖旧 ZIP 的资格 BLIF。

## 冻结开发配置

入口首次复制 [campaign.example.json](campaign.example.json) 到输出目录的 `campaign.json`，不会覆盖已有配置。在任何方法效果可见前核对 RTL、固定 grid、家族与资源组。准备完成后可恢复独立的3D环境配置：

```bash
source "$TAILCOMPILE_REPO/env_3d_v7.sh"
```

示例是 4 个 train 开发候选（MCML/SPMV 重负载候选，SHA/OR1200 轻控制）＋2 个 active validation 候选（blob_merge、diffeq1）。**第二个重负载族尚未资格通过，当前不能启动 480 pilot。** 本地真实 AP 导出已确认 SHA 的 memory/DSP 为 0/0、OR1200 为 4/4、SPMV 为 232/32；MCML 的前端在本地 120 秒预算内未完成，其 hard-resource 需求未知。服务器第一步应完成 MCML 映射与容量资格，保留实际日志；不能把 pending 标签当作通过证据。

固定 grid 是待资格检验的输入，不代表已证明资源可行。prepare 用实际 hard APBlock 数检查重/轻分组：重组至少 16 个 memory/DSP APBlocks，且 DSP/BRAM 中至少一项相对该 grid 的容量上界利用率达到 10%；轻组至多 8 个 hard APBlocks。这里每个 hard APBlock 计一次，独立于大 DSP 消耗两个模式容量单位、小 DSP 消耗一个单位的需求表达；容量利用率采用修正后的模式单位。历史候选账本的 boolean hard-demand 单位保留注明，不用于替代当前需求。BGM 的 11 个大 DSP 不会因模式单位翻倍而突然变成合格重族。MCML 的 64×64 是候选 grid，可在看到方法效果之前按实际输入容量修订；不能降低 heavy 门槛来凑齐两重两轻。映射或基础容量不支持时先处理资格，进入正式方法比较后的失败则保留为方法结果。

`candidate_status_ledger` 保留已查候选及排除原因：attention_layer 的现成网表含当前架构不支持的 `addition_fp_clk_16`，TPU 现成网表端口不兼容；stereovision0/3、ch_intrinsics、BGM 未达到重负载门槛。softmax 暂列原语资格待查，不是 active validation 输入。此处不通过添加新 FP 原语扩大既定 scope，也不按 GNN 胜负换设计。

其余候选 train 族可在输入资格阶段按明确固定 grid 添加；不默认一次派发 20 族。20 族对应的 1280 次候选反馈只是公式，不是本示例已经运行的数量。原 2D 四个已查看测试族不能混入新的未见最终测试。此开发 campaign 拒绝 final/inherited holdout 记录。

每个 record 可用 `source`（相对于完整映射 VTR 的 `vtr_flow/benchmarks/verilog`），或用 `mapped_blif` 的绝对路径输入已有网表。所有路径均需位于注册 workspace 内；BLIF 进入 Hecate 前做有记录的原语适配，原文件保留。

首次运行生成不可变 `protocol.json`，包含配置、binary 和代码/传递依赖哈希。协议变化用新的版本输出目录，不清空旧结果。

同一个 `TAILCOMPILE_V7_OUT` 同时只允许一个主阶段进程。直接调用 Python CLI 和 shell 入口都会持有 Linux 非阻塞 `flock`；重复启动会立即报错，不争写 manifest 或重复派发。该锁不限制阶段内部线程并行，初始化子 worker 不抢主阶段锁。阶段退出后可正常续跑，`.campaign-stage.lock` 文件会保留，不能仅凭文件存在判断进程仍运行。

依赖门禁是可执行检查：`prepare` 必须完成全部配置输入才会生成 `prepared.json`；后续阶段核验该文件及输入哈希。`calibrate` 必须为每个开发族提供完整四架构、两个有效宽度，`pilot` 才能生成队列。MCML 映射/容量未过、缺少一个家族或缺少宽度面板时会报错停止派发；仅有三族的现成结果不能被包装为 480 ready。

## CPU/GPU 阶段命令

```bash
entry="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/prepare_server.sh"

# CPU：先看数量，再单独检查输入资格。当前MCML仍pending。
bash "$entry" --stage plan
bash "$entry" --stage prepare
# prepare完全通过后，依次执行以下两个阶段。
bash "$entry" --stage calibrate
bash "$entry" --stage teachers

# GPU：教师锚点预训练，输出最多三个预登记epoch快照；不按教师loss选赢家。
export TAILCOMPILE_3D_TRAIN_DEVICE=cuda
bash "$entry" --stage pretrain

# CPU：每个train条件4候选×521/522，另有同条件Default锚点。
bash "$entry" --stage feedback

# GPU：用正/负/失败账本更新实际生成器，保留教师稳定项。
bash "$entry" --stage finetune

# CPU：最多三个ckpt，实际validation总成本＋成功/QoR保护选模。
bash "$entry" --stage validate

# CPU：严格四开发族×四架构×两宽度×五seeds×三方法，480个方法任务。
bash "$entry" --stage pilot

# CPU：五seed已测Wmin独立扫描；开发仍用531–535，不改名为最终测试。
bash "$entry" --stage scan

# CPU：同核三方法平衡次序、受控并行、真正冷准备成本。
bash "$entry" --stage timing

# 常驻成本面板：一个真实持续进程，多次生成并复用模型/器件embedding。
bash "$entry" --stage resident

# 汇总门槛；缺scan或独立timing时不会判完整通过。
bash "$entry" --stage report
```

CPU/GPU 可以是不同实例，只要同一 Git 仓库、数据和依赖路径在共享存储可见。每一步完成后再派发依赖它的下一步；教师完成后可以释放 CPU 实例、切到 GPU，feedback/validate/pilot 再回 CPU。不要求在 GPU 实例空等长时间 VPR。CPU 阶段将 `TAILCOMPILE_3D_TRAIN_DEVICE` 设为 `cpu`，GPU 训练设为 `cuda`；不要复制旧2D的环境脚本。

当前示例的资格后预算为 6 个 active 输入，最多 480 次 Default 宽度校准、96 次教师、256 次候选反馈＋64 次 Default 锚点、最多 128 次 validation P&R；随后是 480 个 pilot 方法任务。扫描和受控计时单列。它们是排队预算，尚非已经完成的实验。输入映射在每个 trial 独立 attempt 下生成 `flow/`，续跑不会借用另一 campaign 的综合中间目录。

后台运行单阶段的例子：

```bash
mkdir -p "$TAILCOMPILE_REPO/logs"
log="$TAILCOMPILE_REPO/logs/v7_feedback_$(date +%Y%m%d_%H%M%S).log"
nohup bash "$entry" --stage feedback >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
```

`nohup` 只解决终端断开，不延长平台作业资源期限；长阶段放到平台提供的持久 CPU/GPU 作业中。默认请求 32 并发，按物理核、CPU quota、声明的每任务 RSS 和实际内存余量降低并输出原因；RRG 有独立限额。默认每任务先保留 4096 MiB。首批 `resources.txt` 明确实测峰值后，可在下一阶段通过 `TAILCOMPILE_V7_JOB_MIB` / `TAILCOMPILE_V7_RRG_MIB` 设置有依据的预算，正式计时开始前冻结实际并发。

## 关键输出与解释

- `prepared.json`、`boundaries.json`、`conditions.json`：明确源、grid、资源组、删失及同一设计四架构共用宽度。没有两个有效宽度时不伪造 480 队列。
- `device_cache`、`architectures/*/actual_audit.json`：实际 RRG、独立 Z 容量、位置、switch 类别与 width 耦合；共享池不按开关边重复计容量。
- `teacher_records.json`：raw XYZ 与 projected targets 分文件；固定 seed 选择完整解，不平均坐标。validation 记录不能进入梯度。
- `feedback.json`：全部四候选、两个 backend seeds、Default anchors、失败状态和动作保留；缺库/未知OOM不当负收益标签；没有完整可执行动作时保留结果但不伪造动作概率梯度。
- `models/*/completed.json`：训练的实际 payload 哈希与最多三个 checkpoint；`selection.json` 记录真实 VPR 选模，teacher loss 不决定最终 checkpoint。
- `pilot/report.json`：480个方法条件的状态全表。若生成器失败，实际 VPR 启动次数可能少于480，解码失败仍留在方法分母。
- `scan/report.json`：五 seeds 各自最小已测成功宽度及算术均值，保留无成功、非单调、timeout和删失；不声称精确物理最小宽度。
- `timing/report.json`：冷启动主口径。额外 AP 导出、逻辑图、生成和后续全部 VPR 成本计入，RRG 可复用成本单列。整波按相同条件三方法、同一物理核和平衡顺序执行；不拼接旧串行吞吐时间。常驻生成报告只给真实生成成本及摊销，不能据它单独宣布 P&R 加速。
- `acceptance_report.json`：目标为总时间比≤0.95、成功率不降、CPD/WL和五seed平均已测Wmin比≤1.03；逐设计/架构/宽度角色保护和缺测检查均保留。开发通过不等于全历史未见族泛化。

若所有 validation checkpoint 均未通过保护，示例预登记允许一次**明确标注的探索性 pilot**，选择仍仅依据 validation，`selection.status` 会标失败；这不把方法改写为性能通过。可在运行前把该选项关闭。

每次任务保留 manifest、实际命令、输入哈希、stderr/stdout、CPU/RSS、GP/APPack/SA/route阶段指标与跨层使用量。原子结果和签名支持续跑；空间/配额预检失败即停止新派发，已运行兄弟任务收尾。完成运行的派生 `.net/.route` 由受限清理函数在指标落盘后回收，原始输入、教师、初值和共享RRG保护；清理记录说明字节数与恢复需要。

完整未见族测试需在开发结束后另冻结全历史暴露清单与新任务协议，并使用541–545；本入口不把开发族换几个seed就称为最终验证。

阶段完成后通过同一 TailCompile Git 仓库同步 `.gitignore` 允许的轻量证据：协议/配置、资格报告、`prepare_progress.json` 或完整 `prepared.json`、资源账本、阶段 report/manifest 和汇总报告；保留失败与删失记录。checkpoint、图张量、原始 RRG、临时综合目录、`.net/.route` 留在服务器，不进入 Git。日志统一保存在 `repo/logs`，需要定位故障时只同步选定的小日志或报告引用，不能打包整个日志树。第二 heavy 资格失败时先同步输入资格日志并明确阻塞，不能上传一个标称已准备完成的 480 结论。
