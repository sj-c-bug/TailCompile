# v6.8 磁盘配额恢复与产物保留策略

## 本次报错的含义

`OSError: [Errno 122] Disk quota exceeded` 是用户/项目配额耗尽，不一定是整个文件系统已满。`df -h` 即使显示 TB 级空闲，也不能排除个人 quota。用平台配额页面或 `quota -s` 核对；`df -i` 可另查 inode。

用户日志在 `feedback_dataset.json` 汇总写入处失败。该写入发生在本轮并发 VPR 返回、质量比较之后：已完成候选及成功试验可以继续复用，但其中 `other_failure` 仍需诊断重试，不能当作不可布线标签，也不能仅补写一个 complete 标志进入微调。

本地审计确认 VPR 的实际文件名包括 `sha.net`、`sha.net.post_routing`。后者是路由后的 packed-netlist XML 文件，不是函数。几千个候选/宽度/seed 反复生成两份大 XML，加上 `.route`，会累计数百 GiB；实际占比必须以服务器盘点为准。

## 保留什么，清理什么

| 产物 | 本轮 v6.8 策略 | 原因 |
|---|---|---|
| `inputs/` 中的 BLIF、canonical `.net`、structure TSV | 保留 | 输入、结构及后续重建依据 |
| `graphs/`、logic/device/supply IR、`device_cache/` 压缩 RRG | 保留 | 生成、训练、设备缓存复用；不是逐次运行副本 |
| checkpoint、optimizer、模型报告 | 保留 | 已训练模型与断点，不为腾空间重训 |
| `generation/` 的 `init.ap.tsv`、`.fplace`、`target.json`、`generation.json` | 保留 | VPR 输入及反馈微调目标 |
| 教师 `gp.blocks.tsv/gp.nets.tsv/gp.meta.tsv` | 保留 | 后续仍要解析教师；不得按 TSV 或 gp 前缀删除 |
| `trial.json`、`started.json`、`metrics.json`、`vpr.out`、JSON/CSV 结果 | 保留 | 断点身份、结果、诊断、CPD/线长/阶段时间解析 |
| 已结束 route 目录中 VPR 新生成的 `.net`、`.net.post_routing`、兼容拼写 | 指标持久化后清理 | v6.8 route 从 BLIF 启动，不通过 `--net_file` 读这些输出 |
| 同目录 `.route/.route.gz`、`.rpt/.echo` | 指标持久化后清理 | 当前闭环只从日志/指标读取；不再反复保存全量原始输出 |
| `.place/.fplace`、未知大文件、未结束目录 | 默认保留，先盘点 | 不凭扩展名扩大删除范围 |

尤其不能运行 `find . -name '*.net*' -delete`：旧 v6.7 等路线会通过 `--net_file input.net` 读取网表，其他版本的打包输入也不能照本轮策略删除。新工具只处理明确的 v6.8 cohort，不碰旧版本、代码或目录本身；Git 跟踪、输入依赖、符号链接/硬链接、正在使用的文件均受保护。

清理是永久 unlink，不进回收站。保留紧凑证据能复用结果，但不能还原原始 `.route/.net`；如以后需要离线分析新的空间拥塞指标，须重跑相应 VPR。外部 journal 和 `payload_pruned.json` 记录删除内容。

## 1. 停止任务、更新、盘点

先确认该 cohort 的控制器及所有子 VPR 已停止。不要只关闭终端，也不要 `pkill python`。手动清理会获取与 pipeline 相同的 `.lock` 并检查活动文件，锁仍被占用时拒绝删除。不要手工删除 `.lock` 绕过保护。

保留原 Conda/VTR/输出路径，勿 source 会自动启动任务的旧 env.sh：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

export TAILCOMPILE_V68_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v68/runs/hier_init_001"
storage_report="/tmp/v68_storage_$(date +%Y%m%d_%H%M%S).json"
cleanup_plan="/tmp/v68_cleanup_$(date +%Y%m%d_%H%M%S).json"

df -h "$TAILCOMPILE_V68_OUT"
df -i "$TAILCOMPILE_V68_OUT"
quota -s 2>/dev/null || true

python3 experiments/vtr/tailcompile_v68/storage.py inventory \
  --repo "$TAILCOMPILE_REPO" --out "$TAILCOMPILE_V68_OUT" \
  --report "$storage_report"

python3 experiments/vtr/tailcompile_v68/storage.py plan \
  --repo "$TAILCOMPILE_REPO" --out "$TAILCOMPILE_V68_OUT" \
  --plan "$cleanup_plan"
```

`inventory` 仅读取元数据，不解压 RRG，不读取大网表正文。报告按真实文件类型统计文件数、逻辑大小、唯一 inode 的实际占用及大文件排行。`plan` 不删除文件，输出 `candidate_files`、`allocated_GiB`、按类型汇总和跳过原因；完整逐文件清单在 `$cleanup_plan`。默认允许清理刚结束的运行，不必等待旧脚本的 24 小时。

`/tmp` 通常是实例本地磁盘，不受共享项目的同一配额约束，但应先确认它还有空间。清理后将报告、plan、receipt/journal 复制到有空间的 logs 目录留档；换实例前不要遗失 `/tmp` 审计记录。

若 `git pull` 本身也因配额失败，可把本次独立、只依赖 Python 标准库的工具下载到实例 `/tmp`，无需先向项目盘写入代码。将下文 inventory/plan/apply 命令中的脚本路径改成 `"$storage_tool"`，清理后再 `git pull --ff-only` 启用自动回收：

```bash
storage_tool="/tmp/tailcompile_v68_storage_$(date +%Y%m%d_%H%M%S).py"
curl --fail --location \
  https://raw.githubusercontent.com/sj-c-bug/TailCompile/main/experiments/vtr/tailcompile_v68/storage.py \
  --output "$storage_tool"
```

也可以先使用仓库原有的旧版清理工具释放空间（它不删除 `.net`，不动 v6.8）：

```bash
bootstrap_plan="/tmp/old_vpr_cleanup_$(date +%Y%m%d_%H%M%S).json"
python3 experiments/vtr/storage_cleanup.py plan \
  --repo "$TAILCOMPILE_REPO" \
  --scratch "${TAILCOMPILE_SCRATCH:-$TAILCOMPILE_REPO/vtr_work}" \
  --plan "$bootstrap_plan"
# 看过清单后再执行：永久删除旧版已结束任务的可复现 route/RRG/详细报告。
python3 experiments/vtr/storage_cleanup.py apply \
  --repo "$TAILCOMPILE_REPO" --plan "$bootstrap_plan"
```

## 2. 审核后执行清理

确认计划只包含本轮已结束 route 目录的可复现输出，再执行：

```bash
python3 experiments/vtr/tailcompile_v68/storage.py apply \
  --repo "$TAILCOMPILE_REPO" --out "$TAILCOMPILE_V68_OUT" \
  --plan "$cleanup_plan"
```

apply 会重新校验路径、文件 stat、trial/metrics 哈希、输入依赖和活动状态。变动文件跳过，不追随符号链接，不删除目录。输出 `deleted_files`、`released_GiB`、receipt 和 journal 路径；实际释放量可能因底层快照、其他硬链接或平台统计延迟不同于预计值。

盘点发现大文件却没有进入计划时，先查看原因，不能据此扩大 glob。特别是未写完 `trial.json` 或 `metrics.json` 的目录不会被自动判定为可清理。

## 3. 原目录恢复 feedback

```bash
# 只在 status 报告存在 interrupted_tasks、且确认所有旧 worker 已停止时执行：
bash experiments/vtr/tailcompile_v68/run_pipeline.sh recover

# 成功/unroutable 不重跑；只重试 other_failure/killed 等基础设施失败。
export TAILCOMPILE_V68_RETRY_FAILED=1
export TAILCOMPILE_V68_PRUNE_COMPLETED=1
export TAILCOMPILE_V68_JOBS=32
export TAILCOMPILE_V68_GENERATION_JOBS=32

mkdir -p logs
log="logs/v68_feedback_storage_resume_$(date +%Y%m%d_%H%M%S).log"
nohup bash experiments/vtr/tailcompile_v68/run_pipeline.sh feedback \
  >"$log" 2>&1 </dev/null &
echo "PID=$! LOG=$log"
```

不要清空 feedback/generation、graphs 或模型，不需要重新 prepare/train。若换实例恢复，且 recover 无法验证旧主机，必须先确认旧实例已关闭、没有远端 worker，再显式使用已有 `TAILCOMPILE_V68_CONFIRM_REMOTE_STOPPED=1`；不要为了跳过检查直接设置。

## 4. 防止再次累计数百 GiB

新版默认 `TAILCOMPILE_V68_PRUNE_COMPLETED=1`：子 VPR 结束、trial 与 metrics 成功写入之后立即清理该次运行的可复现大输出。成功、不可布线和已记录的异常均保留紧凑证据，复用时不依赖被删的文件。需要暂时保留原始输出做诊断时显式设为 `0`，但要预留空间，结束后手动 plan/apply。

调度器只保留至多 JOBS 个在途任务；一旦检测到 ENOSPC/EDQUOT 或 VPR 日志明确报告配额耗尽，不再派发新任务，等待已运行任务结束并保存结果，不强杀有价值的并发结果。普通不可布线仍是正常标签，不触发该保护。

`TAILCOMPILE_V68_MIN_FREE_GIB` 默认 10：启动新命令前检查文件系统剩余空间。**它不是个人 quota 检测器**；共享盘余量再大，也可能先撞个人配额。查看平台 quota 并据库存留足并发峰值：

`预算 ≳ 固定输入/图/模型缓存 + JOBS × 最大单次运行瞬时输出 + 安全余量`

每次完成即回收后，raw 输出从“全部试验累计”变为主要受“在途并发数 × 单次峰值”约束，但不能保证 32 个超大设计一定容得下。空间紧张可只降低 `TAILCOMPILE_V68_JOBS`，不必降低独立的候选生成并行度；这两个阶段顺序执行。不要默默降到单核，也不要以删除运行中的文件代替容量规划。

`storage_policy.json` 记录实际保留策略；删除在 VPR 计时结束后执行，不能把清理本身当作 VPR 加速。本次不改变 VPR 命令、模型/训练/候选定义或冻结 seed，不因存储改动重做成功试验。

后续增加空间定位标签时，应先在原始 `.route` 仍存在时提取紧凑的区域/cut/segment 统计，存好派生指标后再回收，不能先删原始数据再声称仍能离线补标签。
