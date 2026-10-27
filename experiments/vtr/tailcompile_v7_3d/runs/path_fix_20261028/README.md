# 2026-10-28 综合路径修复与写入边界回归

服务器提供的事实：`development_002_git` 3D资格报告 `complete=True, passed=True, conditions=8`；四case×两seed×三入口共24次资格运行通过。随后自然SHA输入映射在约0.115秒后失败，Parmys最终错误为 `Failed to load architecture file ... ERROR_PARSE_ARCH`。该失败尚未进入Default宽度校准，不是模型的性能负结果。

固定40f8前端脚本的实际路径构造复现见 [path_report.json](path_report.json)。相对临时目录生成不存在的 `<VTR>/vtr_flow/scripts/flow/arch.xml`，同时配置可能解析为 `attempt/flow/flow/odin_config.xml`；绝对attempt目录下两者均存在。该复现使用捕获runner，没有调用Yosys或执行新的P&R；依赖源文件只读，临时XML/TCL已清理。

v7修复将 `-temp_dir` 在每个独立attempt中解析为绝对路径，记录逻辑和实际命令。恢复工具核验原binary、hook、源码和完整资格后，在新cohort复制配置／资格轻证据，原失败和旧campaign protocol保留。

新增服务器写入边界：新构建仅在主仓库 `.deps/tailcompile_v7_3d/`；旧外部后端需显式只读模式，禁止克隆、打补丁、重编或清理。运行临时目录和缓存均在仓库scratch，输出／依赖／软链接逃逸会在写入前被拒绝。原2D源码、共享冻结依赖和安装未修改。

本地WSL Python3.12／Torch2.6CPU全套契约测试 **66/66通过，0跳过**，约28.32秒；见 [unit_tests.json](unit_tests.json) 和 [unit_tests.log](unit_tests.log)。`bash -n prepare_server.sh` 与 `git diff --check` 通过，v6.8/v6.9/v6.10源码差异为空。新增检查包含真正的子进程路径切换、独立重试与旧失败保留、恢复资格缺失／hash不符拒绝、只读旧后端、符号链接和临时缓存边界。

这些证据不代表服务器修复后已综合成功，也没有研究规模训练或性能收益新结论。下一步按 [恢复命令](../../RECOVERY_MAPPING_2026-10-28.md) 在服务器运行 `prepare`，通过后进行Default校准。
