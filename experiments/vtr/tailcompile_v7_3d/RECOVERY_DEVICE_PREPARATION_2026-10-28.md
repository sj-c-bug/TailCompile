# 大型RRG特征提取超时：原004继续运行

## 本次事实

MCML在84×84双层器件完成了打包并进入布线；服务器日志中的 `w64 status=unroutable` 属于正常的宽度扫描结果，必须保留。随后四份新RRG生成成功，各耗时约132秒；在将完整RRG转换为器件区域／共享池图时，原解析器抛出 `complete device graph extraction exceeded its preparation budget`。

原 `build_device_graph` 有300秒默认准备预算，包含完整供给提取、区域收缩和额外switch审计。四份大型XML同时在Python线程内处理时可能有GIL和I/O竞争；当前日志能确认超时，不能单独证明各因素的贡献。这不是MCML再次容量不足，也不是GNN生成失败。

## 处理范围

保留当前 **development_004_capacity**，不创建005，不修改 `campaign.py`、`device_graph.py`、模型或冻结协议。新增的离线准备工具调用同一个既有解析API，只把每份完整图的准备预算显式设置为3600秒，串行完成。

- 只处理“VPR生成已成功、RRG已存在、器件图cache manifest尚未完成”的条目。
- 核验冻结源码、输入、binary、producer命令、架构和缓存key；前后检查RRG哈希。
- 完整图通过校验后才原子写入器件图及cache manifest。超时、错误或中断不将部分特征视为有效结果。
- 原RRG生成、成功／正常不可布通的P&R、已有有效cache不重跑或改写；不复制大型RRG，不更换FPGA尺寸、width、seeds、搜索预算或标签。
- 继续校准时将RRG准备并发固定为1，P&R并发独立保留。若更高width仍触发原300秒准备门限，协调器只对这一明确的准备超时补齐cache再续跑，并要求每次至少新增一个完整cache；其他错误正常停止。
- 准备耗时和预算单独记入恢复报告。它们属于可复用器件处理，不能作为GNN的P&R加速收益。

全部新增写入仍只在主仓库的004、v7 scratch和logs内；原2D与3D后端只读。既有解析函数部分阶段只在末尾检查预算，新增工具在Linux主线程为每图设置硬截止；超时或中断保留原RRG及错误记录，不发布部分特征或近似图。

## 服务器命令

在原Torch环境中获取本次新增工具后执行：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
bash experiments/vtr/tailcompile_v7_3d/resume_device_preparation.sh
```

入口自动后台运行并打印PID、日志和OUT；默认选择原004，覆盖旧终端遗留的OUT／CONFIG／QUALIFICATION，后端使用现成的只读3D安装。不要清空004，不要重新运行完整bootstrap。

默认每图3600秒；若需明确调整，在启动前设置 `TAILCOMPILE_V7_DEVICE_PREP_SECONDS`。这是器件准备预算，不更改 `TAILCOMPILE_V7_TIMEOUT` 中的VPR预算。已有P&R必须按原签名复用，不能通过改VPR timeout强行重贴标签。

自定义原cohort可设置 `TAILCOMPILE_V7_PREPARATION_OUT`；标准流程无需设置。此入口只负责器件准备与Default校准，完成后仍需检查 `boundaries.json` 的六个输入均为 `paired`，再采教师和训练。
