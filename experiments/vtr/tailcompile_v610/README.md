# v6.10：冻结权重的 benchmark 遍历

这轮回答：在更广的设计中，现有分层 GNN 的初始化是否有收益，以及收益出现在哪里。**没有训练阶段，不修改现有 checkpoint**。Default 和 GNN 初始化后都继续运行相同的原生 VPR 全局布局、APPack、详细布局与布线。

## 范围与解码

- 默认 `all`：现有 v6.8 benchmark 清单与服务器固定 VTR checkout 中全部顶层 Koios RTL 的并集。排除共享 include，记录缺失、失败及排除原因，不把参数变体算成独立设计族。`inventory_preview.json` 给出实际数量，不预设有多少个可运行设计。
- 本地固定版本清单检查发现30个 Koios 条目，合并旧清单后44个设计、31个按当前规则归类的分组；排除了两个仅含宏定义的配置 include。`test.v` 属于 DSP 回归测试，保留遍历，但不能作为独立应用设计证据。这些数量不表示已全部通过新流程，最终以服务器资格与执行报告为准。
- 每个设计标注原 split、family 和 checkpoint 的训练/验证暴露关系。这是探索性覆盖实验，已见过的设计不用于声称独立泛化；事后发现的有益设计需另行确认。
- 继续使用固定 tile 的分层 GNN。默认 `balanced` 解码在原快速坐标输出上增加有限轮次的离散需求修正：按真实分配后的各类资源溢出，把 molecule 移向兼容且溢出更少的区域，再用冻结坐标 head 输出坐标。保留固定节点、原子性及资源类型兼容。
- 修正限制轮数、候选区域数和移动预算，报告前后溢出与占用分布。它改善的是**区域容量代理**，不能保证最终打包合法、时序或布线收益；最终由 VPR 判定。无法完全消除的溢出保留记录，不无限重试。
- 合成压力测试中，10k molecule/169 区域的单点集中被分散至169区域，50k/400区域分散至393区域；这是实现验证，不能替代真实 benchmark 结果。修正额外时间全部计费。
- `TAILCOMPILE_V610_DECODER=legacy_soft` 可在**另一输出目录**做原解码诊断；本轮主扫描不为每个设计事后挑选更好的解码。

## 指标与公平性

1. **Wmin**：固定宽度网格上，分别求 seeds 411–415 的最小成功宽度，再取五个值的算术平均。完整扫描网格，不假定布线结果严格单调。全部失败为右删失，最低宽度就成功为左删失；timeout/环境错误为未知，不能当作不可布线或从平均值中删除。
2. 新设计用 Default seeds 401/402 先寻找参考边界，再冻结双方相同的扫描网格；已有 v6.8 网格与输入只有通过哈希审核后才能复用。默认步长4、边界两侧各3点，另保留较宽点。结果称为**观测网格 Wmin**，不声称找到了精确物理下界。若网格截断边界，报告限制，下一批再对称扩展，不能只补某一方法。
3. **CPD/WL/runtime**：每个设计预先选择 Default 参考宽度与冻结网格最高宽度，运行相同五个 seeds。选择不看 GNN 是否成功；不把失败条件悄悄丢掉。runtime/QoR 比值只在双方成功的配对上有意义，同时报告完整成功、失败、未知和覆盖率。
4. 主 runtime 为 `新鲜 GNN 推理 + 匹配/修正/坐标生成 + 初始化文件生成 + VPR`。**不计 Python/PyTorch 启动、模型构造/加载、逻辑图与器件数据加载及搬运**。双端 GNN 编码都从头执行并计费；RRG解析/图构建为可复用准备工作，单列。`excluded_setup_seconds` 和原始 worker wall time仍保留。不能把这个预加载口径与旧版包含加载的 runtime 直接混算。
5. 计时按物理核绑定，每任务内部1线程，避开超线程兄弟核，自动受 CPU 配额限制。配对的 A/B 在同一个核先后运行，跨配对 AB/BA 交替；每半波所有任务完成加载后统一放行，等待不计费。不同配对并行，轮末再清理。仍可能存在共享内存/缓存干扰，所以称为**受控并发负载下**的速度比较。
6. 沿用暂定验收参考：含推理的总时间至少改善5%，成功率不下降，平均观测 Wmin、CPD、WL 退化不超过3%。输出逐设计结果和设计族/暴露分组，不要求所有 benchmark 都有效，也不把少数正例写成普遍有效。失败或未知不会被计为验收通过。

## 服务器运行

全部阶段可以在 **CPU 实例**完成。使用已能运行的 v6.8/v6.9 Conda 环境与带审计 GP warm-start hook 的服务器 VTR `40f8d19708b0b60e2aa04452deb1cfed68d96b0b`。脚本不重新激活 Conda，不训练、不改权重。

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

export TAILCOMPILE_V610_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v610/runs/koios_sweep_001"
export TAILCOMPILE_V610_CHECKPOINT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v69/runs/runtime_001/models/runtime/generator.pt"
export TAILCOMPILE_V610_JOBS=32
# 未设置的 PREPARE_JOBS/RR_JOBS/GENERATION_JOBS/TIMING_JOBS 均继承 JOBS。
# 可独立提高/降低；RRG并行不是只能1，需按峰值内存预算设置。
export TAILCOMPILE_V610_MIN_FREE_GIB=30

bash experiments/vtr/tailcompile_v610/run_pipeline.sh inventory
mkdir -p "$TAILCOMPILE_REPO/logs"
log="$TAILCOMPILE_REPO/logs/v610_sweep_$(date +%Y%m%d_%H%M%S).log"
nohup bash experiments/vtr/tailcompile_v610/run_pipeline.sh all \
  >"$log" 2>&1 </dev/null &
echo "PID=$!"
echo "LOG=$log"
```

`all` 顺序为：`prepare → evaluate（含生成）→ timing → 汇总`。可分别执行这三个阶段；改变并行度不改变科学条件。正式 timing 更改核数/主机时更换 `TAILCOMPILE_V610_TIMING_RUN`，不能把不同负载结果合并。32为请求并行度，实际计时受物理核、CPU quota及剩余配对数量约束。大量 RRG 任务可能先耗尽内存，应依据内存峰值调低 RR 并行或提高实例内存，而不是自动抹掉失败设计。

```bash
bash experiments/vtr/tailcompile_v610/run_pipeline.sh status
```

## 断点与硬盘

- 所有新增数据、临时目录和日志都在 DAC2027 工作区内；旧 v6.8 输入只读复用，不复制旧全套实验，不递归清理旧版本。
- 一个器件/宽度的 RRG 尽量共享；该组所有 graph/IR 哈希验证完成后删除新建原始 RRG，保留哈希及回收记录。旧共享输入不删除。
- 默认运行结束、`trial.json` 与 `metrics.json` 落盘后立即回收逐任务派生 `.route`、重复 `.net`/post-routing netlist和详细报告。保留命令引用的输入、初始化文件、图/IR、权重、JSON/CSV、日志及 Git 跟踪文件。计时任务等半波全部完成后再回收，避免清理I/O干扰配对。
- 磁盘低于阈值或发现 quota 错误时停止新提交；阈值是可见文件系统余量，不能保证检测到所有账户配额，30GiB也不是32个重任务的预留上界。降低并行度可降低同时在写的文件量。
- 成功/不可布线/超时结果与输入签名均保存；正常续跑不重做已有完整任务。失败重试需显式 `TAILCOMPILE_V610_RETRY_FAILED=1`，超时重试需 `TAILCOMPILE_V610_RETRY_TIMEOUTS=1`。未知结果仍会阻止完整证据验收。
- 异常中断后，先确认旧进程及子进程停止，再执行 `recover`，随后同目录继续 `all`。如果任务来自另一主机，确认停止后才设 `TAILCOMPILE_V610_CONFIRM_REMOTE_STOPPED=1`。不能两处同时写同一输出目录。
- 完整 timing 波次复用；中断的半波/波次重测以保持并发条件，旧证据进入归档。输出目录、checkpoint或实验参数变化会报错，防止混入旧结果。

最终重点查看：

| 文件 | 用途 |
|---|---|
| `inventory.json` / `preparation_report.json` | 全清单、族、已见关系、缺失/失败、实际覆盖 |
| `generation/generation_report.json` | 解码前后集中/溢出、生成失败 |
| `evaluation_report.json` | 五seed观测 Wmin、删失与非单调性 |
| `benchmark_summary.csv` | 逐benchmark四指标及通过情况 |
| `benchmark_summary.json` | 完整配对、覆盖、设计族/暴露统计与成本口径 |
| `timing/preloaded_001/runtime_report.json` | 受控并行计时及硬件协议 |

`complete` 表示证据齐全，`passed` 表示达到收益门槛，两者不能混淆。准备阶段出现不支持/环境/超时会保留已完成同伴并报告，修复后续跑；需要排除无法运行设计时另建明确子集，不静默改变冻结清单。

## 2026-10-25：旧清单误收两个配置 include 的修复

初版自动发现误收 `koios/hard_mem_include.v` 和 `koios/complex_dsp_include.v`。它们只定义架构宏，没有独立顶层模块，现已按源文件角色排除，保留源哈希与排除原因。已运行的 `koios_sweep_001` 无需删除重建：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

# 先预览；确认旧 pipeline 已退出。
python3 experiments/vtr/repair_tailcompile_v610_inventory.py \
  --out "$TAILCOMPILE_V610_OUT"

# 应用同一修复，保存完整旧/新记录及原失败报告。
python3 experiments/vtr/repair_tailcompile_v610_inventory.py \
  --out "$TAILCOMPILE_V610_OUT" --apply
```

迁移脚本持有同一 cohort 锁，只允许在准备尚未完成、没有新图或正式评测时运行；核对原始 inventory 源码哈希、所有其他冻结代码、模型哈希，以及每个保留设计的完整记录。仅更新 inventory.py 的代码哈希和对应辅助文件条目，不改宽度、seeds、超时、权重或 VTR。旧运行目录全部保留；审计和备份在 `_repairs/helper_inventory/repair.json`。两个 JSON 替换之间中断时可以重复同一命令恢复。

其余超时或 VPR 错误需要分别查日志。`RETRY_TIMEOUTS` 保持同一预算重试超时；`RETRY_FAILED` 只重试其他失败。**不要在同一输出目录直接提高 `TAILCOMPILE_V610_TIMEOUT`**：它同时冻结在主协议、校准协议和综合命令中。准备并发可调整，例如改为16，完整成功任务仍复用。仅修复辅助条目不会把其余失败自动算成成功或不可布线。

轻量同步：`.gitignore` 默认仅放行审计文件、报告、成本和指标，不提交 RRG、网表、图张量、模型或逐任务巨大输出。

```bash
git add -- experiments/vtr/tailcompile_v610/runs/koios_sweep_001
git diff --cached --stat
git commit -m "Record frozen-weight benchmark sweep"
git push origin main
```
