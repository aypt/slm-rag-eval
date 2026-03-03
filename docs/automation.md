# Automation guide — 在你自己的云服务器上无人值守开发

目标：你的 dev container 跑在一台常开的云服务器上 → 笔记本关机也不影响；
所有 agent 调用都是"本地 CLI"形态（`codex exec`），不依赖 Codex Cloud。

## 一次性准备（在 dev container 里）

```bash
# 1. 安装并登录 Codex CLI（headless 服务器用 device-auth：终端给一个码，任意设备浏览器确认）
npm install -g @openai/codex
codex login --device-auth        # 若版本不支持该 flag，用 codex login 的提示流程

# 2. 项目依赖（run_next_task.sh 里的 make check 需要）
cd /path/to/slm-rag-eval
make setup

# 3. git 初始化（骨架交付时不含 .git）
git init && git add -A && git commit -m "chore: scaffold (M00)"
git remote add origin <your-github-repo>   # 可选但强烈建议
```

登录凭据存放在 `~/.codex`。如果你的 dev container 会重建，把 `~/.codex` 挂载成
volume（devcontainer.json 的 `mounts`），否则每次重建都要重新登录。

## 沙箱与权限（为什么脚本默认 danger-full-access）

Codex 在 Linux 上的内置沙箱依赖内核特性（Landlock/seccomp 等），在 Docker 容器里经常
不可用或表现异常。业界通行做法：**容器本身就是隔离边界**——在非特权容器内给 Codex
完全权限，同时把挂载和凭据保持最小化。所以：

- 在 dev container 里跑：保持默认 `CODEX_ARGS="--sandbox danger-full-access"`。
- 万一你在裸机上跑：务必 `export CODEX_ARGS="--sandbox workspace-write"`。

`codex exec` 是官方非交互模式：不弹任何审批、进度打到 stderr、最终消息打到 stdout、
任务完成即退出，天生适合脚本和 cron。

## 运行方式

```bash
# 手动触发一个任务（第一次建议手动跑，盯着 log 看一遍全流程）
make next-task

# 一晚清空队列（队列空或首个任务失败即停，失败会挡住后续依赖任务）
make all-tasks

# 定时触发：把 scripts/crontab.example 里的一行装进 crontab -e
```

runner 的行为（`scripts/run_next_task.sh`）：

1. 有 `.agent-stop` 文件就不跑（急停开关：`touch .agent-stop`）；flock 防重入。
2. 取 `tasks/todo/` 里按文件名排序的第一个任务，整个文件作为 prompt 喂给
   `codex exec`。
3. **独立验证**：不信 agent 的自我汇报，runner 自己再跑一次 `make check`。
4. 双绿 → 任务文件移入 `tasks/done/`；任一失败 → 移入 `tasks/needs-review/`
   （这样 cron 不会对同一个失败任务无限重试烧 token），并提交一条记录 commit。
5. 一切都发生在 `agent/auto` 分支上，**绝不触碰 main**。`PUSH=1` 可自动推送。

## 每日循环

- **早晨（15–30 分钟）**：`git log agent/auto` + 看 diff；抽查测试是不是真测试；
  满意就把 `agent/auto` 合并进 main（或在 GitHub 上开 PR 自审后合并）。
  发现问题 → 把修改意见写成一个小任务文件（如 `M02a-fix-xxx.md`）放进
  `tasks/todo/` 队首，当晚自动修。
- **白天**：交互式处理不适合无人值守的部分——prompt 调优、指标质量判断、实验解读。
- **睡前 30 秒**：确认队首任务的前置依赖已合并、`.agent-stop` 不存在，收工。

## 换用其他 agent（骨架是解耦的）

任务文件是纯文本、AGENTS.md 是开放规范，所以执行引擎可插拔。例如换 Claude Code：

```bash
# runner 里等价的一行（Claude Code 的 headless/print 模式）
claude -p "$(cat tasks/todo/M01-llm-client.md)"
```

仓库里的 CLAUDE.md 已把 Claude Code 指向 AGENTS.md，规则只维护一份。
验证层（make check）、任务层（tasks/）、记录层（PROGRESS.md、logs/）全部不变。

## 进阶：并行任务（可选）

严格串行最安全。确要并行互不相关的任务（见 docs/plan.md 的可并行组合），用 git
worktree 给每个任务开独立工作树，各自跑各自的 `codex exec`，避免两个 agent 写同一
个 checkout：

```bash
git worktree add .worktrees/m06 -b agent/m06 agent/auto
(cd .worktrees/m06 && codex exec --sandbox danger-full-access "$(cat ../../tasks/todo/M06-docker.md)")
```

注意每个 worktree 需要自己装依赖才能跑 `make check`。不熟悉前先别用。

## 交互式长跑的替代：/goal

如果某晚你想盯前半程、后半程挂机，可以不用 cron，改在 tmux 里开交互式 Codex 并用
`/goal` 下达持久目标（`/goal pause` / `resume` / `clear` 管理生命周期，注意它有
token 预算，预算耗尽会进入 budget-limited 状态，第二天 resume 即可）。cron + `codex
exec` 与 tmux + `/goal` 二选一即可，不要同时对同一个 checkout 跑两个 agent。

---

## v2 加固（冒烟测试 L1–L7 之后新增）

L7 测试证明：给 agent 一个无解任务，它宁可用 `__eq__` 语义把戏让检查变绿，也不承认失败
——因为它以为失败不被允许。v2 的所有改动都围绕这一发现。

### 判决门（run_next_task.sh v2）

| 门 | 检查什么 | 失败去向 |
|---|---|---|
| 1 | `codex exec` 退出码 | needs-review |
| 2 | runner 自己重跑 `make check`（不信 agent 汇报） | needs-review |
| 3 | holdout 测试（agent 永远看不到，见下） | needs-review |
| 4 | diff 体检（削弱测试/特判/dunder 把戏，见下） | needs-review |
| 5 | BLOCKED 报告 或 空 diff（静默无操作 ≠ 成功） | needs-review |

### BLOCKED 协议（先授权失败）

AGENTS.md 现在开篇即声明"诚实的失败是成功"。agent 遇到矛盾/无解任务的正确动作：
往 PROGRESS.md 追加 `## <TASK> — BLOCKED` 段落说明冲突，不改任何代码，退出。
runner 检测到该段落会把任务转入 needs-review 并停止 run_all_tasks.sh 的后续链条
（后续任务通常依赖它，继续跑只会烧 token）。

### diff 体检的覆盖范围与已知误报

只标记高信号行为：**删改既有测试的行**（新增测试完全合法，不标）、新增
skip/xfail/type-ignore/noqa、src/ 里新增 `__eq__`/`__hash__`/`__bool__`、
碰 scripts/ 或 tasks/、凭空出现的 conftest.py。已知误报：测试里为"故意喂非法输入"
新增的 `# type: ignore` 会被标记 —— 代价是一次早晨复核，可接受；漏报的代价是基准
数字被污染，不可接受。

### holdout 测试（对付"拟合测试"的最后一道墙）

在**仓库之外**建目录（agent 的工作区里看不到），放只有 runner 会跑的 pytest 文件：

```bash
mkdir -p ~/holdout-slm-rag-eval
# cron 行里加：HOLDOUT_DIR=$HOME/holdout-slm-rag-eval
```

何时填充：M02 落地后放几条 faithfulness 行为测试（换一组输入验证同样逻辑）；
M07 落地后放一小撮 RAGTruth 样本的端到端断言——这份样本永不进入工作区，
agent 拟合不了看不见的东西。

### 跨模型评审钩子（CLAUDE_REVIEW=1）

`claude -p` 以只读方式复审 Codex 的 diff：task spec + diff 全部从 stdin 管道喂入，
不需要任何权限放行、无副作用；输出存到 `logs/<task>-<stamp>-review.md`，
只作早晨评审的参考，**不参与判决**（避免双 agent 互相卡死）。前提：容器里
`claude` 已登录。这利用了不同模型盲区不相关的特性，专门抓 L7 那类语义把戏。

### 环境定稿清单（来自冒烟测试待办）

```jsonc
// devcontainer.json 增补
"mounts": [
  "source=codex-home,target=/root/.codex,type=volume",   // 登录态在容器重建后保留
  "source=claude-home,target=/root/.claude,type=volume"  // 若启用评审钩子
],
"postStartCommand": "service cron start || sudo service cron start || true"
```

```toml
# ~/.codex/config.toml —— 裸默认推理档太低，务必钉高（字段名以 codex --help 校对）
model = "gpt-5.6-sol"
model_reasoning_effort = "high"
cli_auth_credentials_store = "file"

[profiles.night]           # 夜间难任务用：codex exec --profile night ...
model_reasoning_effort = "xhigh"
```

### L5 / L6 验证命令（纯环境检查，几分钟做完）

```bash
# L5 脱离 TTY：确认没有任何交互阻塞
setsid ./scripts/run_next_task.sh </dev/null >/tmp/l5.log 2>&1; tail /tmp/l5.log

# L6 真 cron：装一条一分钟触发的临时条目，验证 PATH/环境无误后立刻删掉
# * * * * * cd /path/to/slm-rag-eval && ./scripts/run_next_task.sh >> logs/cron.log 2>&1
```

### 额度守门（gate 0，v3 新增）

设计前提：以**周额度**为主（按你环境里的窗口为准），阈值默认 90%（`QUOTA_MAX_PCT` 可调）。
官方没有可脚本化的用量子命令（`/status` 只在交互会话里可用），所以守门是两层的：

- **兜底层（永远生效，零依赖）**：`codex exec` 失败且日志尾部出现限流特征
  （usage/rate limit reached、429 等）时，写入 `.quota-cooldown-until`
  （默认 8 小时，`COOLDOWN_HOURS` 可调）；冷却期内 runner 直接以"跳过"退出，
  cron 不会反复撞墙烧重试。误报的代价只是多歇 8 小时，可接受。
- **预防层（可选，尽力而为）**：`scripts/check_quota.sh` 若能打印出周用量百分比
  整数，runner 会在 ≥ 阈值时不开新任务。脚本里预置了两个社区方案的接线说明
  （codex-cli-usage / codex-ratelimit），打不出数字就**放行**（fail open），
  由兜底层保证安全。第一次真实撞限后，把日志里的报错原文补进兜底层的 grep
  模式里，识别会更准。

退出码语义随之统一：0=任务完成、1=needs-review、3=跳过（停止/锁/空队列/额度）。
`run_all_tasks.sh` 见 0 继续、见 3 正常收工、见其他停链。Claude 评审钩子的用量
可以忽略不计（每次一条只读消息）；若未来让 claude 当 builder，同样的两层模式
照搬即可（它有 5 小时 + 周双窗口）。
