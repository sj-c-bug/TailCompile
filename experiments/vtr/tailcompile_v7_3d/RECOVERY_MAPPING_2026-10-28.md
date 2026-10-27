# SHA综合路径修复与服务器恢复

## 原因与已完成证据

服务器 `development_002_git` 的四个case、两个seeds、三种入口共24次3D资格P&R通过。随后SHA输入映射的Parmys报告 `Failed to load architecture file ... ERROR_PARSE_ARCH`。这是v7批处理入口的相对路径错误：固定VTR收到 `-temp_dir flow` 时，把架构写为 `<VTR>/vtr_flow/scripts/flow/arch.xml`，并可能在已进入 `flow` 后读取 `flow/odin_config.xml`。

修复在v7自己的runner中，将此参数解析成当前attempt内的绝对目录，同时保存逻辑命令、实际执行命令和路径策略。重试使用独立attempt，不借用失败的中间文件。前端失败会直接附带Parmys日志尾部。原2D代码及VTR源码、binary不变。本地固定前端脚本的路径构造复现和子进程回归不能代替服务器实际综合成功；恢复后仍需查看输入准备结果。

## 写入边界

未经进一步要求，服务器仅向 `$TAILCOMPILE_REPO` 和用户指定的原 `DAC2027/vtr_tailcompile/vtr-verilog-to-routing` 根写入。本流程将全部新写入放在前者，原2D VTR继续只读。

- 新建3D构建默认位于 `$TAILCOMPILE_REPO/.deps/tailcompile_v7_3d/vtr-verilog-to-routing`，不进入Git。
- 旧的仓库外3D后端可显式只读复用。下面命令读取其源码、manifest和binary，不修改、搬移、清理或重编译它。
- `project-public` 与 `global_user` 可能是同一挂载的路径别名，边界按真实路径判定。不能以“同在DAC2027”作为可写依据。

## 复用服务器已通过的24次资格

以下命令仅在**本次修复源码已通过同一个TailCompile Git同步到服务器后**执行。仍使用已激活的Torch环境，不安装或升级共用环境。

```bash
cd "$TAILCOMPILE_REPO"
export PYTHONDONTWRITEBYTECODE=1
export GIT_OPTIONAL_LOCKS=0
export PYTHONUNBUFFERED=1
export TAILCOMPILE_3D_PYTHON="$(command -v python3)"
export TAILCOMPILE_3D_TRAIN_DEVICE=cpu

previous="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_002_git"
export TAILCOMPILE_V7_OUT="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/runs/development_003_paths"
export TAILCOMPILE_V7_CONFIG="$TAILCOMPILE_V7_OUT/campaign.json"
export TAILCOMPILE_3D_QUALIFICATION="$TAILCOMPILE_V7_OUT/qualification"
export TAILCOMPILE_3D_VTR_ROOT="$(dirname "$(realpath "$TAILCOMPILE_REPO")")/vtr_tailcompile_3d_v7/vtr-verilog-to-routing"
export TAILCOMPILE_3D_BACKEND_READ_ONLY=1
export TAILCOMPILE_V7_TIMEOUT=3600

# 保留先前按空闲CPU／内存设置的并发；未设置时从单并发开始。
export TAILCOMPILE_V7_JOBS="${TAILCOMPILE_V7_JOBS:-1}"
export TAILCOMPILE_V7_RRG_JOBS="${TAILCOMPILE_V7_RRG_JOBS:-1}"

# 核验binary、hook、源码及八个完整资格条件，再复制配置／资格轻证据。
# 旧失败attempt、旧protocol和外部后端原样保留；不复用旧campaign protocol。
"$TAILCOMPILE_3D_PYTHON" experiments/vtr/tailcompile_v7_3d/recover_prepare.py \
  --previous-out "$previous" --out "$TAILCOMPILE_V7_OUT" \
  --backend "$TAILCOMPILE_3D_VTR_ROOT"
```

只有上条输出 `recovery=ready` 后，才启动新准备与Default校准：

```bash
entry="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/prepare_server.sh"
bash "$entry" --check && {
    mkdir -p "$TAILCOMPILE_REPO/logs"
    log="$TAILCOMPILE_REPO/logs/v7_recover_$(date +%Y%m%d_%H%M%S).log"
    nohup bash -c '
        set -euo pipefail
        entry="$TAILCOMPILE_REPO/experiments/vtr/tailcompile_v7_3d/prepare_server.sh"
        bash "$entry" --stage plan
        bash "$entry" --stage prepare
        bash "$entry" --stage calibrate
        echo V7_BASELINE_FINISHED_REVIEW_BOUNDARIES
    ' >"$log" 2>&1 </dev/null &
    echo "PID=$! LOG=$log"
}
```

不会重复构建或重跑24次资格。若旧binary、源码、hook或资格不匹配，恢复工具报错且不创建新输出目录；不能跳过检查。新源码签名在新cohort冻结。若目标目录已经属于不同protocol，另选新cohort，不覆盖旧结果。此路线直接调用stage，不更新bootstrap生成的根目录 `env_3d_v7.sh`；换终端时重执行本节exports，不要source仍指向002的旧环境快照。

准备成功应有包含六个输入、`complete=true` 的 `prepared.json`。校准完成仍需检查 `boundaries.json` 中六个输入均为 `paired`；校准命令返回0不意味着没有删失。满足后才执行 `--stage teachers`，然后训练。v6.10正式timing期间不要在同实例新增高负载。

本次修复保持三方法、H×Z、设计族、宽度策略和验收指标不变。没有将前端错误当作不可布通标签，也没有取得模型性能收益的新结论。
