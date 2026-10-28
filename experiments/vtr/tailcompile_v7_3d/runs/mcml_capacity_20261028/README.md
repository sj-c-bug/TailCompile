# MCML容量失败、器件修正与恢复回归

服务器事实：MCML在64×64双层器件、w32、四H×Z×两seed下8项均code1，约250秒。用户提供的两个H0Z0底层日志表明实际APPack需求为CLB5502、DSP102、BRAM294、IO392，器件仅BRAM248，缺46个；打包失败后未进入布线。SPMV在w128四H×Z×两seed全部成功。

对不可容纳的MCML采用容量修正规则：各类型已观测打包需求占容量不超过90%，边长按4对齐的最小正方形。当前观测推导84×84，BRAM328，利用率89.63%。恢复工具从8份实际错误日志取各类型最大需求重算，不以模型收益选器件；新尺寸实际packing和RRG尚待服务器核验。

[sizing_audit_report.json](sizing_audit_report.json) 记录本地64/80/84三尺寸×四H×Z的12个XML审计。新84四架构物理总量均为CLB12792、DSP328、BRAM328、IO5248，原语定义不变。Z声明预算随器件尺寸增长；只对同尺寸四格匹配，不将64到84视作Z容量固定。没有运行新的P&R；临时XML已清理，原VTR只读。

新增 `recover_capacity.py`、一键后台入口及签名明确的只读复用接口。新004从003导入经核验的输入和完整Default校准，MCML使用同一固定前端BLIF在新尺寸重新导出结构；未变XML/width/binary的RRG只读共享，不拷贝大型文件，不伪造旧trial执行命令。原003、后端和失败日志保留。旧实现的兼容性由固定Git blob及未变模块／传递依赖哈希验证；新protocol纳入恢复器源码哈希。

完整WSL Python3.12／Torch2.6CPU契约回归 **76/76通过，0跳过**，见 [unit_tests.json](unit_tests.json) 和 [unit_tests.log](unit_tests.log)。新增检查覆盖资源上界不足提前拒绝、实际八条容量错误解析、尺寸选择、完整校准导入、旧数据改动／超时不同／实现未知拒绝、符号链接边界，以及恢复入口覆盖旧cohort变量后按顺序执行。Bash语法检查和 `git diff --check` 通过，冻结2D源码差异为空。

本次没有在服务器代为运行校准，没有训练或性能收益新结论。恢复命令见 [容量恢复说明](../../RECOVERY_CAPACITY_2026-10-28.md)。
