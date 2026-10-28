# MCML BRAM容量不足：修正输入并继续校准

## 已确认的原因

服务器 `development_003_paths` 的MCML在 `w32`、四种H×Z、seeds501/502下均以code1结束。用户提供的前两个底层日志明确报告：

| 物理block类型 | APPack实际需求 | 64×64双层器件可用量 |
|---|---:|---:|
| CLB | 5502 | 7192 |
| DSP | 102 | 248 |
| BRAM（memory） | **294** | **248** |
| IO | 392 | 3968 |

因此失败在打包，缺少46个BRAM，未进入布线。增大channel width、激活其他Torch环境或提高并发不会增加BRAM。这些记录作为输入容量资格失败保留，不改成普通不可布通标签，也不作为GNN的负收益训练标签。

同一批SPMV在 `w128` 四架构×两seeds均成功。新恢复流程对条件完全相同且证据完整的其他设计复用原校准，保留各trial原manifest／日志和来源哈希。没有完成完整边界的设计继续派发必要的校准；不声称所有旧任务都能直接复用。

## 冻结容量修正规则

仅对MCML采用：在不改变原语、硬列数量及比例的前提下，选择**边长按4对齐、所有已观测打包资源利用率不超过90%的最小正方形**。根据当前实际需求，得到 **84×84×2**。

恢复工具会读取八条失败日志，取各类实际需求的最大值重新计算；若尚未展示的其他case需求更高，仍严格按同一规则选尺寸，以冻结catalog中的结果为准，不硬编码修改日志或需求。

- BRAM容量：`4×(84−2)=328`，利用率 `294/328≈89.63%`。
- 80×80只有312个BRAM，利用率94.23%，未达到该规则。
- 新容量为CLB12792、DSP328、BRAM328、IO5248。H0层间BRAM164/164，H1为82/246，DSP反向；四架构总量一致。
- 这是在GNN方法对比前修正不可容纳的输入，不根据GNN胜负选器件。不对已能容纳的其他设计新增90%筛选条件。
- 器件面积增加也增加了声明Z预算。不能声称64到84之间Z容量不变；只比较84尺寸内总资源及总Z容量匹配的四种H×Z。

本地已生成64/80/84三个尺寸×四架构的12个XML，核对资源匹配、Z位置及原语定义，随后清理临时XML。此证据不代替84尺寸的实际RRG审计或Default P&R成功。

## 服务器执行命令

保持已激活的Torch环境，确认v6.10不在同实例正式timing。通过同一主Git仓库获取本次修改后执行：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
bash experiments/vtr/tailcompile_v7_3d/recover_capacity.sh
```

入口默认后台运行，打印PID、日志路径和新输出目录 `runs/development_004_capacity`。按打印的绝对路径查看日志，例如 `tail -f <日志路径>`。无需清空003，无需重编VTR，无需重跑24次后端资格。

入口继承已设置的并发；未设置时使用单并发。若有已确认的16个空闲物理核及足够内存，可在启动前设置 `TAILCOMPILE_V7_JOBS=16`、`TAILCOMPILE_V7_RRG_JOBS=2`。容量错误与CPU并发无关，正式timing仍需独立控制。

脚本只读核验原3D后端。新写入全部位于主仓库：004输出、v7 scratch、logs。旧后端和003不搬移、不重编、不删除。若当前终端未保留 `TAILCOMPILE_3D_VTR_ROOT`，默认读取已建的同级 `vtr_tailcompile_3d_v7/vtr-verilog-to-routing`；若你的旧后端路径不同，应在启动前显式设置该变量。原2D VTR继续只读。

## 自动恢复内容及检查

恢复工具先核验旧protocol、prepared、后端、源码、原始输入和打包错误证据，生成冻结的 `capacity_recovery_catalog.json`，再启动新cohort：

1. 复用已通过的24次后端资格，保留原文件及哈希。
2. 其他设计复用已核验的BLIF／结构；MCML复用同一固定前端的BLIF，在84器件重新导出结构。
3. 完整且条件不变的其他设计校准通过显式来源账本导入；缺失／不完整校准不伪造为完成。
4. 同binary、同XML哈希及同width的旧RRG只读共享；MCML的新84 RRG重新生成并审计。大型缓存不复制。
5. 重新运行MCML Default校准，不继承64器件的容量失败作为84器件结果。

整个过程不重写旧trial命令或旧protocol。新protocol记录修复代码和输入修正，四种架构、两个宽度及后续三方法都以新MCML尺寸为准。prepare另加明显资源上界不足检查，可提前拒绝已能证明无法容纳的网表；通过上界检查仍不代表实际packing可行。

完成后应有六个输入的 `prepared.json` 及 `boundaries.json`；六个输入均为 `paired` 后再采教师。看到 `prepared_input_reused`、`calibration_reused` 表示相应旧证据已复用，不是重新执行。日志末尾 `V7_CAPACITY_BASELINE_FINISHED_REVIEW_BOUNDARIES` 仅表示阶段完成，仍需核对删失项。

本入口使用阶段运行，不更新旧 `env_3d_v7.sh`。后续阶段先显式设置OUT/CONFIG/QUALIFICATION到004，并保持原后端只读模式；不要source仍指向002／003的旧环境快照。
