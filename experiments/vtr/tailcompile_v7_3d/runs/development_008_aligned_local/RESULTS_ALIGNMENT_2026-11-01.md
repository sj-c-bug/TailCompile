# 008 一致性修复：本地结果及边界

本轮完成了“训练评价的初始布局，与部署实际输出的初始布局一致”的代码与本地资格验证。没有重写007结论，没有修改冻结2D代码；自然SPMV/MCML输入只在服务器，尚未运行新一轮自然设计训练或性能比较。

## 已实现

- 原教师只读，派生008教师使用部署同一逐层资源容量代理、资源兼容和固定约束。投影后独立重算使用量；失败显式停止，不增大容量、不静默用旧标签。
- 教师XY也限制在最终密度混合实际可生成的范围。原本已经满足约束的目标保持不变；新共享XY模块按派生教师重新预训练，随后L/T共用并冻结。
- 训练风险前向使用实际硬层分配及有限密度处理后的XY。硬分配不求导，层概率使用有偏straight-through近似；该梯度不是精确的分配器或VPR梯度。
- 层监督混合节点平均与资源均衡损失，避免少量RAM/DSP节点被大量LUT节点淹没；逐资源容量惩罚不再被无关资源和层数稀释。
- 服务默认不额外计算软/硬风险诊断。模型、教师和反馈动作带独立008 schema及约束/共享XY身份，拒绝混用旧007标签和checkpoint。
- 新入口只读复用007后端、图、RRG和原教师，所有派生产物放入独立008目录。

## 实测证据

| 检查 | 结果 | 证据 |
|---|---|---|
| 教师、模型、训练、launcher、runner测试 | 38项通过；包含非零层模型梯度、冻结XY、旧标签拒绝、恢复一致性 | [unit_tests.json](unit_tests.json)、[targeted_unit_tests.log](targeted_unit_tests.log) |
| 真实小网表训练/部署动作一致 | balanced/asymmetric各141 AP、18叶区域，动作SHA相同；训练风险与硬方案CVaR分别同为0和1.2003130912780762 | [aligned_smoke_report.json](aligned_smoke_report.json) |
| 实际P&R | 2/2成功；T-I、seed3701，初值被消费、原生GP继续、固定原子检查通过 | 同上及各trial的manifest/metrics |
| 受控RAM超容量标签 | 未训练的明确控制例：memory代理超用1→0，容量不变 | [controlled_memory_overflow_report.json](controlled_memory_overflow_report.json) |
| 大规模投影性能 | 合成125796 AP/882叶，一线程约6.17秒，层代理溢出消除 | [teacher_projection_performance_report.json](teacher_projection_performance_report.json) |
| 最终模型代码复核 | 两例重生成初值SHA与VPR实际消费文件相同；动作及风险一致，未新增P&R | [final_code_compatibility_report.json](final_code_compatibility_report.json) |
| 完整生成worker接口 | 重建图→加载checkpoint→生成→lower→反馈目标检查通过；初值SHA与已消费文件相同，默认不计算风险诊断 | [worker_contract_report.json](worker_contract_report.json) |

真实小图的原生GP教师本来都满足层容量；不能把受控RAM例写成真实教师修复。真实教师的XY分别有45/50个AP被修正到部署可达区间。125796 AP性能测试是合成压力检查，不代表自然MCML完成了新训练。

两次P&R约2.42/2.62秒，没有同期Default或完整候选反馈池，不能据此声称加速或TailRisk有效。最终一致性只指GP之前的初值；后续GP、APPack、SA仍可移动节点，容量代理也不等于真实packing/布线合法。

## 服务器下一步

按[008运行说明](../../ALIGNED_EXECUTION_2026-11-01.md)执行`check → prepare`，派生24个条件的自然教师并检查投影审计。随后新预训练、反馈与同期对照都有独立入口；完整研究上限为160反馈＋80选模＋224比较。沿用平均收益口径，负收益及失败完整保留，不预设修复后必然加速。

未使用的本地`.net/.net.post_routing/.route`与重复worker图已在摘要落盘后按本次目录清理；共享输入、后续依赖和原007数据保留。
