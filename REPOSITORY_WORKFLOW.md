# TailCompile 代码与实验结果管理

## 唯一主仓库

本项目的源码、实验脚本和需要长期保留的轻量结果统一由 **TailCompile** 仓库管理：

- GitHub：<https://github.com/sj-c-bug/TailCompile.git>
- 本地：`C:\Users\phoenix\Documents\ChatGPT\DAC2027\TailCompile_latest`
- 服务器：`/inspire/hdd/project/project-public/changshunjie-253114010247/changshunjie-253114010247/workspace/DAC2027/TailCompile`

服务器常用环境变量：

```bash
export TAILCOMPILE_REPO=/inspire/hdd/project/project-public/changshunjie-253114010247/changshunjie-253114010247/workspace/DAC2027/TailCompile
export TAILCOMPILE_SCRATCH="$TAILCOMPILE_REPO/vtr_work"
```

各版本代码位于 `experiments/vtr/tailcompile_v*`，运行结果位于对应版本的 `runs/<cohort>`，日志统一放在 `$TAILCOMPILE_REPO/logs`。

## VTR 的定位

VTR 是固定的执行依赖，不用于同步 TailCompile 源码或实验结果：

```bash
export VTR_ROOT=/inspire/hdd/project/project-public/changshunjie-253114010247/changshunjie-253114010247/workspace/DAC2027/vtr_tailcompile/vtr-verilog-to-routing
```

- 固定版本：`40f8d19708b0b60e2aa04452deb1cfed68d96b0b`。
- 当前 checkout 已包含并审计实验所需的 GP warm-start 接口；正常实验期间保持不变。
- 不在 VTR 仓库提交 TailCompile 代码、模型、日志或运行结果，也不通过 VTR Git 来回同步数据。
- 除非任务明确要求修改或重新编译 VTR，否则只使用现有 `$VTR_ROOT/vpr/vpr`。

## 日常同步方式

代码更新通过 TailCompile Git 同步，不需要传源码压缩包：

```bash
cd "$TAILCOMPILE_REPO"
git pull --ff-only
```

服务器实验结束后，只提交 TailCompile 仓库中 `.gitignore` 允许的轻量证据，例如 JSON、CSV、protocol、metrics 和汇总报告：

```bash
cd "$TAILCOMPILE_REPO"
git add -- experiments/vtr/<版本>/runs/<cohort>
git diff --cached --stat
git commit -m "Record <实验名称> results"
git push origin main
```

本地再从同一个 TailCompile 仓库拉取并分析。大型中间产物（RRG、`.net`、`.route`、图张量、checkpoint、临时工作目录等）留在服务器并按存储策略回收，不通过 Git 同步。若某项大型文件确实需要检查，应单独说明，而不是打包整个源码或实验目录。

一句话原则：**TailCompile Git 管代码和轻量结果；固定 VTR checkout 只负责执行；大型派生产物留在服务器。**

## 3D v7：同一源码仓库，独立运行边界

3D 同样通过本仓库 `git pull --ff-only` 同步，不新建 `TailCompile_v7_3d_20261025` 平行源码仓库，不把源码ZIP作为日常交付方式。此前ZIP和单独解压启动命令停止推荐，作为历史离线快照保留。

| 内容 | 位置／约定 |
|---|---|
| 新3D实现 | `experiments/vtr/tailcompile_v7_3d/` |
| 共享3D helper、补丁、计划及旧轻结果 | `experiments/vtr3d/`；不再依赖本地主仓库外的同名目录 |
| 新3D运行结果 | `experiments/vtr/tailcompile_v7_3d/runs/<cohort>/` |
| 新3D scratch | `$TAILCOMPILE_REPO/vtr_work/tailcompile_v7_3d/` |
| 主日志 | `$TAILCOMPILE_REPO/logs/`，忽略入Git |
| 2D执行／综合依赖 | 保持现有 `VTR_ROOT` checkout 和binary不变，只读复用综合工具 |
| 3D执行依赖 | 同一40f8基础版本，独立目录 `.../DAC2027/vtr_tailcompile_3d_v7/vtr-verilog-to-routing`，含v2异构修复和XYZ hook；由3D专用变量指定 |

本次任务明确授权了3D后端开发，所以它可以在独立目录构建；这不是在原2D安装上追加补丁。新入口见 `experiments/vtr/tailcompile_v7_3d/prepare_server.sh`。它从自身位置确定主Git根，不读取旧ZIP清单，不自动激活conda，不自动Git拉取／提交／推送。

原 `tailcompile_v68`、`tailcompile_v69` 的源码、输入schema、checkpoint和运行入口冻结。3D可只读复用其有明确版本的纯函数，但用独立Python进程启动、记录依赖哈希；不将两个版本的裸模块混在同一个解释器中发现测试。3D专用环境变量只在3D子进程中适配，不能覆盖调用终端的2D `VTR_ROOT`。

v6.10运行期间，其cohort签名冻结的v6.10和共享依赖同样不得变动。服务器更新前先fetch，核对整次待拉取差异，而非只看最新3D提交；同实例低并发准备与正式timing的边界见`experiments/vtr/tailcompile_v7_3d/CO_RUN_V610_2026-10-27.md`。代码隔离不等于CPU、内存和I/O隔离。

输出保护解析真实路径和symlink，拒绝与旧版本目录或指定VTR／综合根相交；任何新3D参数／代码签名变化使用新cohort。Git忽略规则只开放轻量协议、manifest、metrics和报告，图、教师、动作数组、原始RRG、mapped网表、模型和scratch仍不入Git。阶段结束同步前需检查 `git diff --cached --stat`，避免将大型payload改名JSON后提交。

本地修改尚未提交／推送时，服务器 `git pull` 无法获得这些文件；不能将“本地已整理”称为“服务器已更新”。提交后冻结Git commit，再在服务器拉取并启动新cohort；不要在运行中的cohort上修改源码。3D旧轻结果的迁入账本见 `experiments/vtr3d/migration_v7_20261025.json`，本轮管理核查见 `experiments/vtr3d/REPOSITORY_ALIGNMENT_2026-10-25.md`。
