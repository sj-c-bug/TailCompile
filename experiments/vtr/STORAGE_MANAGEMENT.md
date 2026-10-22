# 全版本实验数据清理与管理

## 为什么旧版清过一次仍有约 800 GB

旧 `storage_cleanup.py` 仅清理 v0–v6.5 的部分 `.route`、RRG 和详细报告，保留所有 `.net`，且排除 v6.6/v6.7。它不是完整的旧版归档清理工具。

源码审计找到实际的重复来源：

| 位置 | 大文件来源 | 本轮处理 |
|---|---|---|
| v0–v3 的 scratch | `<design>.locked-input.net`，命令实际引用 | 默认保留引用文件；封存后可显式去重 |
| v0–v3 的 archive | scratch 全量复制后，又额外复制 `<design>.input-packed.net` | 无其他引用的归档复制品可删除 |
| v4–v6.7 每个运行目录 | canonical packed net 复制为 `input.net` | 被 `--net_file` 引用则保留；封存后可显式去重 |
| 各版 VPR 输出 | `.net.post_routing`、无引用的输出 `.net`、`.route`、RRG、详细报告 | 完成证据、依赖、活跃状态检查后清理 |
| v5 的 `inputs/<design>/full/*.net` 等 | canonical 打包输入 | 保留，不与运行副本硬链接 |
| `vtr_work` 中临时综合/布局目录 | 旧流程未回收的 work 目录 | 全量盘点；没有可验证完成证据的目录不自动删除 |

800 GB 是用户报告的旧版总量，不等于可安全释放量。需要服务器 inventory 输出按版本/文件类型统计，再据 plan 的预计释放量执行；本地无法代替服务器确认配额或实际用量。

## 先定位占用，再决定清理范围

若目前连空间究竟在哪都不清楚，先运行下面的全账户只读盘点，不必先猜测或删除版本目录。

## 一次性全账户只读盘点（包括项目以外的占用）

`disk_audit.py` 从指定账户挂载目录扫描，而不是只扫描 TailCompile。隐藏缓存、Conda、其他工作区、VTR 构建均会进入统计；仓库/scratch 若在主根之外，作为额外扫描根加入。重复和相互包含的路径合并，不跟随目录内部的符号链接，不跨越其他文件系统。每个文件只读取元数据，不计算内容哈希、不读取模型/网表正文，不做删除或权限修改。

在 CPU 实例执行，一次后台完成：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

audit_root="/inspire/hdd/global_user/changshunjie-253114010247"
audit_dir="$(mktemp -d /tmp/tailcompile_disk_audit.XXXXXX)"

nohup python3 -u experiments/vtr/disk_audit.py \
  --root "$audit_root" \
  --extra-root "$TAILCOMPILE_REPO" \
  --extra-root "${TAILCOMPILE_SCRATCH:-$TAILCOMPILE_REPO/vtr_work}" \
  --extra-root "${CONDA_PREFIX:-$audit_root}" \
  --out "$audit_dir" \
  >"$audit_dir/run.log" 2>&1 </dev/null &

echo "$!" > "$audit_dir/audit.pid"
echo "PID=$!"
echo "报告目录=$audit_dir"
echo "进度日志=$audit_dir/run.log"
```

只在终端显示上述几行；不会把完整清单刷到终端。检查进度用 `tail -n 10 "$audit_dir/run.log"`，不需要 `cat` 全量 CSV。最终以 `summary.json` 的 `complete` 和日志完成标记判断是否跑完；同时查看权限错误、跳过挂载点等覆盖限制，不把“扫描结束”等同于“整台 GPFS 的所有文件均可见”。

结果包括：

- `report.log`：一次性可读汇总，含 `df`/inode/可用 quota 等环境信息、各扫描根和大目录、类型分布、最大文件、错误提示。
- `directories.csv`：全量目录统计，便于继续下钻具体 workspace、缓存、版本和子实验目录；目录大小是包含后代的累计值，不能把所有行相加。
- `file_types.csv`：按真实后缀区分 `.net.post_routing`、RRG、BLIF、模型、压缩包、JSON 等。
- `largest_files.csv`：最大的文件及大小；不用复制长终端输出。
- `deleted_open.csv`：仅本机 `/proc` 可见、位于扫描根的已删除但仍被打开的文件，不列出无关进程命令参数。其他 CPU/GPU 实例中的打开句柄不在本机可见范围。
- `summary.json`、`progress.json`、`errors.log`、`run.log`：机器可读统计、进度和完整诊断。

逻辑大小与 `st_blocks*512` 估算的已分配大小分别记录；硬链接和目录别名尽量按 inode 去重，归属按首次遇到的路径统计。`files` 与逻辑总量也是去重后的普通文件统计，不是硬链接路径数。GPFS 配额、快照、其他节点的打开文件、权限边界及扫描期间的变化，均可能造成 `df`、目录统计和配额的差异；这是诊断，不是原子快照。`quota` 等命令不存在或无权限时会记录限制，不为查询配额执行提权。

先发送 **`report.log` 和 `summary.json`** 即可；需要下钻时再补对应 CSV。全账户文件名和目录可能包含隐私，不要把这些报告提交到公共 Git 仓库。`/tmp` 为实例本地临时目录，关闭实例前把需要的报告复制到有空间的持久目录或下载保存。

## 旧版本清理命令

`storage_reclaim.py` 的旧版范围包含 v0/v1/v2/v3/v4/v5/v6/v6.1–v6.7，以及对应的 scratch；**不操作 v6.8、3D 实验或未知版本**。已结束旧实验可以清理，但控制器和子 VPR 必须先停止。对于当前还在运行的 v6.8，建议先正常停止后统一盘点，避免磁盘仍在增长。

在服务器更新后执行。所有清单写 `/tmp`，避免项目配额满时连清理记录都写不出：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only

stamp="$(date +%Y%m%d_%H%M%S)"
old_report="/tmp/tailcompile_old_inventory_${stamp}.json"
old_plan="/tmp/tailcompile_old_cleanup_${stamp}.json"

python3 experiments/vtr/storage_reclaim.py inventory \
  --repo "$TAILCOMPILE_REPO" \
  --scratch "${TAILCOMPILE_SCRATCH:-$TAILCOMPILE_REPO/vtr_work}" \
  --report "$old_report"

python3 experiments/vtr/storage_reclaim.py plan \
  --repo "$TAILCOMPILE_REPO" \
  --scratch "${TAILCOMPILE_SCRATCH:-$TAILCOMPILE_REPO/vtr_work}" \
  --include-netlists --plan "$old_plan"
```

默认至少保留最近 24 小时的文件。若所有旧任务确定已结束、且确实要清理刚结束的旧运行，可在 plan 显式加 `--min-age-hours 0`；不能靠这个参数绕过活动进程保护。

此阶段不会把 `.net` 扩展名当作删除许可：还要检查 Git 跟踪、真实路径依赖、完成记录、活动进程、符号/硬链接、文件年龄。依赖查询覆盖仓库当前与旧版的轻量配置/运行元数据，不只查询该版本自己的命令；`/global_user` 与 `/project-public` 的实际路径通过 resolve 统一。

大 IR JSON 的路径引用采用流式扫描，避免一次加载巨型数组，也避免对每个 atom 名查询 NAS。若 `unreadable_dependency_records` 非空，网表清理/去重会保守跳过；先检查相应记录，不要绕过依赖校验。

看到按版本/类型的释放量并审核清单后，才执行：

```bash
python3 experiments/vtr/storage_reclaim.py apply \
  --repo "$TAILCOMPILE_REPO" \
  --scratch "${TAILCOMPILE_SCRATCH:-$TAILCOMPILE_REPO/vtr_work}" \
  --plan "$old_plan"
```

这是永久删除，不进回收站。工具不删目录、源代码、BLIF、IR、候选 `.fplace/.ap.tsv`、GP/coarse TSV、occupancy TXT、模型、训练 bundle、JSON/CSV 或日志。每个删除文件均留下审计。清理后保留紧凑结果，但重做新的原始路由空间分析可能需要重跑 VPR。

清单生成后文件或证据变化会被跳过；更新了清理工具后旧 plan 也失效，应重新生成。不要为了释放更多空间直接改 JSON 清单或批量 `find ... -delete`。

## 第二轮（可选）：封存的逐次输入副本去重

如果第一轮后剩余的大头确实是成千份同内容的 `input.net/locked-input.net`，不能一边要求保留旧运行的输入路径，一边直接把这些路径删除。可对**已完成且不再原位修改的旧运行副本**规划 SHA256 去重：

```bash
dedup_plan="/tmp/tailcompile_old_dedup_$(date +%Y%m%d_%H%M%S).json"

python3 experiments/vtr/storage_reclaim.py plan \
  --repo "$TAILCOMPILE_REPO" \
  --scratch "${TAILCOMPILE_SCRATCH:-$TAILCOMPILE_REPO/vtr_work}" \
  --include-netlists --deduplicate-netlists --plan "$dedup_plan"
```

去重需要读取候选网表计算哈希，元数据盘点快不代表去重扫描也快。文件名相同或大小相同都不是重复证据。

硬链接会保留各路径和字节，但让多个归档副本共享 inode。它**不是写入隔离的 copy-on-write**：以后原位修改其中一个路径，其他链接也会改变。只对封存的旧运行副本使用，canonical inputs、当前实验、训练 bundle 和 Git 跟踪文件不参与。保持至少一份 canonical 源输入独立，不能让整个项目的输入都指向一个可写 inode。普通只读权限也不能阻止 root 绕过。

确认这些旧归档不再原位重启/修改后，显式承认该约束才能 apply：

```bash
python3 experiments/vtr/storage_reclaim.py apply \
  --repo "$TAILCOMPILE_REPO" \
  --scratch "${TAILCOMPILE_SCRATCH:-$TAILCOMPILE_REPO/vtr_work}" \
  --plan "$dedup_plan" --ack-immutable-archives
```

若仍计划经常修改/复跑旧版目录，不执行这一步，保留独立副本。新实验使用新目录，勿复制后原位覆盖这些已封存的链接文件。

## v6.8 单独清理和后续增长控制

全版本旧数据清理不代替 v6.8 的当前反馈恢复。另按 [v6.8 磁盘管理方案](tailcompile_v68/STORAGE_MANAGEMENT.md) 执行当前 cohort 的 inventory/plan/apply，然后 `RETRY_FAILED=1` 原目录续跑。不要清空当前候选、模型或图。

v6.8 默认在 trial/metrics 保存后回收本次派生大输出；配额错误停止进一步派发。共享设备 RRG、逻辑特征、模型与输入作为独立缓存保留，不再按每个 seed 重复归档。

长期执行三层保留策略：

1. 输入/模型/共享缓存：保留，带哈希；相同逻辑/器件不按 trial 重复复制。
2. 完成结果：长期保留 JSON/CSV、日志和必要的紧凑空间统计；可抽样留少量代表性原始路线。
3. 临时工作集：完成后及时回收；旧版定期通过已审核 plan 清理，未知大文件先盘点再决定。

`df` 反映整个文件系统余量，不反映个人/项目 quota；应另看平台配额或 `quota -s`。并发上限还要为 `JOBS × 最大单次瞬时输出` 留余量，不能认为有自动回收就一定能装下 32 个大任务。
