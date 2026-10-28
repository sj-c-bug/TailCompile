# 大型RRG准备预算恢复验证

服务器004日志显示MCML已在84器件上进入布线，w64完成一波8个条件，尾部两项为正常 `unroutable`。随后四份新RRG均生成成功，约132秒；异常由原 `build_device_graph` 的300秒准备预算触发，不是VPR生成RRG失败，也不是又一次BRAM不足。

新增离线准备工具保持原11个冻结实现／模型／网表／架构／seed／width／VPR预算不变，原004继续使用。补齐程序调用同一已有解析API，以3600秒每图预算串行完成缺失器件图，Linux SIGALRM约束晚检查阶段；完整性核验后发布与原Campaign相同的cache manifest。原RRG、P&R和已完成cache只读复用，不返回部分图，不伪造旧执行签名。

协调器只对退出码1且精确匹配既有准备超时的异常恢复，每次必须新增至少一个完整cache，重试总数受冻结宽度／器件面板约束。其他异常、无进展或更长预算仍不足时停止并保留原始状态。INT／TERM只清理自身进程组。恢复成本和源码哈希独立记录，不能当成GNN的加速收益。

完整WSL Python3.12／Torch2.6CPU回归 **93/93通过，0跳过**，约117.24秒；见 [unit_tests.json](unit_tests.json) 与 [unit_tests.log](unit_tests.log)。真实小型RRG在5秒与30秒预算下所有供给特征、容量池和审计结果完全一致，仅提取耗时不同。额外检查覆盖producer签名、原RRG不变、超时不发布manifest、阶段锁及软链接边界、原004变量选择、精确重试条件和真实子进程清理。Bash语法与 `git diff --check` 通过。

本次没有在服务器执行大图补齐或继续P&R；该成功仍待服务器实际运行验证。冻结的 `campaign.py`、`device_graph.py`、`recover_capacity.py` 及2D目录没有修改，因此不要求重新创建cohort或重做既有P&R。命令见 [器件准备恢复说明](../../RECOVERY_DEVICE_PREPARATION_2026-10-28.md)。
