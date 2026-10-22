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

## 第一轮：覆盖全部旧版，先删除不再被引用的输出

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
