# 从零部署手册（SETUP）

目标：从"手里只有一个 zip"到"cron 每晚自动开发、你每天早晨审 15 分钟"。
前提：你的 dev container 跑在常开云服务器上；Codex CLI 已在里面登录过（冒烟测试时完成）。
人工操作总计约 40 分钟；第 5 步的首个任务需要挂机 1–3 小时（建议在场旁观）。

每一步都给出：要执行的命令、它在做什么、以及**预期看到什么**。任何一步的实际输出
和预期不符，先停下解决，不要往下走——后面所有判决都建立在前面的基线上。

---

## 第 0 步 · 把包放上服务器（约 5 分钟）

从对话里下载 `slm-rag-eval-complete.zip` 到笔记本，然后二选一：

**方式 A：scp**
```bash
scp slm-rag-eval-complete.zip <user>@<server>:~/
ssh <user>@<server>
unzip slm-rag-eval-complete.zip && cd slm-rag-eval
```

**方式 B：VS Code Remote** — 连上服务器后，把 zip 直接拖进文件管理器面板，
右键解压（或终端里 `unzip`），`cd slm-rag-eval`。

**预期**：`ls` 能看到
`AGENTS.md CLAUDE.md Makefile README.md PROGRESS.md docs scripts src tasks tests`。

---

## 第 1 步 · git 初始化（约 3 分钟）

```bash
git init
git add -A          # 初始导入用 -A 没问题：.gitignore 已把缓存/密钥类挡在外面
git commit -m "chore: scaffold + hardening v3 (M00)"
git branch -M main
```

为什么：runner 的一切记账（BASE 快照、diff 体检、任务归档 commit）都依赖 git。
没有这一步，第 5 步会直接失败。

**强烈建议**再推 GitHub（备份 + 以后可以用 PR 界面做早晨评审）：
```bash
git remote add origin git@github.com:<你>/slm-rag-eval.git
git push -u origin main
```

**预期**：`git log --oneline` 显示一条 commit；`git status` 显示 working tree clean。

---

## 第 2 步 · Python 环境与基线验证（约 5 分钟）

```bash
python3 --version                 # 需要 ≥ 3.11
python3 -m venv .venv
source .venv/bin/activate
make setup                        # pip 安装本包 + 开发工具（editable 模式）
make check                        # 裁判本人：ruff + mypy + pytest
```

**预期**（逐行核对）：
```
ruff check src tests
All checks passed!
mypy src
Success: no issues found in 21 source files
pytest -q
..................... 21 passed
```

为什么这步重要：`make check` 是之后每个夜间任务的判决依据。**现在**绿，才能证明
以后变红一定是 agent 的问题而不是环境的问题。

注意：cron 不会帮你 `source .venv/bin/activate`——所以第 6 步的 crontab 模板把
`/path/to/slm-rag-eval/.venv/bin` 直接写进了 PATH，记得替换成你的真实路径。

---

## 第 3 步 · Codex 定稿（约 10 分钟）

**3.1 确认登录还在**
```bash
ls ~/.codex/auth.json    # 预期：文件存在。不存在则重新 codex login --device-auth
```

**3.2 写配置**（裸默认推理档太低；字段名以你版本的 `codex --help` 为准）
```bash
cat >> ~/.codex/config.toml <<'CFG'
model = "gpt-5.6-sol"
model_reasoning_effort = "high"
cli_auth_credentials_store = "file"

[profiles.night]
model_reasoning_effort = "xhigh"
CFG
```

**3.3 无害冒烟一发**（验证 exec 路径通，不改任何东西）
```bash
codex exec --sandbox danger-full-access \
  "Run 'git status' and 'ls'. Summarize what you see. Change nothing."
```
**预期**：它输出一段目录/仓库状态的总结后正常退出，`git status` 依然 clean。

**3.4 防容器重建丢失**（devcontainer.json 增补，然后 Rebuild 一次验证登录仍在）
```jsonc
"mounts": [
  "source=codex-home,target=/root/.codex,type=volume",
  "source=claude-home,target=/root/.claude,type=volume"   // 若用评审钩子
],
"postStartCommand": "service cron start || sudo service cron start || true"
```

---

## 第 4 步 · holdout 目录（1 分钟）

```bash
mkdir -p ~/holdout-slm-rag-eval
```
现在是空的，这是正常的。它的作用（agent 永远看不到的测试）从 M02 落地后开始，
往里放什么见 `docs/automation.md` 的 holdout 一节。

---

## 第 5 步 · 首发实弹：手动跑 M01（在场旁观 1–3 小时）

开**两个终端**（都 `cd` 到仓库、`source .venv/bin/activate`）：

```bash
# 终端 A：发射
make next-task

# 终端 B：实时看它在干什么
tail -f logs/M01-*.log
```

**结束时读判决**（终端 A 的最后一行）：

- `M01-llm-client: done — all gates green` → 进入审查三件套：
  ```bash
  git log --oneline agent/auto        # 它提交了什么
  git diff main..agent/auto           # 逐行读 diff —— 这一步不能省
  cat PROGRESS.md                     # 验收项是否逐条附了证据
  ```
  满意就合并：
  ```bash
  git checkout main && git merge agent/auto && git push
  ```
- `needs-review — <REASON>` → 先读 REASON（五道门哪道红了），再看
  `tail -50 logs/M01-*.log` 和 `tasks/needs-review/` 里的任务文件。
  处理完把任务文件移回 `tasks/todo/` 重跑，或写个修正版工单。

这一步同时是对五道门本身的实弹检验。**不要跳过人工旁观直接上 cron。**

---

## 第 6 步 · 交给 cron（约 10 分钟）

**6.1 L5：脱离 TTY 验证**（确认无交互阻塞）
```bash
setsid ./scripts/run_next_task.sh </dev/null >/tmp/l5.log 2>&1; sleep 5; cat /tmp/l5.log
```
预期：正常滚动（若队列里是 M02 且你还没想跑它，先 `touch .agent-stop` 再测，
预期输出 ".agent-stop present"，测完 `rm .agent-stop`）。

**6.2 L6：真 cron 验证（安全版，不真发任务）**
```bash
touch .agent-stop
crontab -e     # 临时加一行： * * * * * cd /path/to/slm-rag-eval && ./scripts/run_next_task.sh >> logs/cron.log 2>&1
sleep 90 && cat logs/cron.log
```
**预期**：出现 ".agent-stop present — not running"。这证明 cron 触发、PATH、工作目录
全部正确，而没有消耗任何额度。然后删掉临时行、`rm .agent-stop`。

**6.3 装正式条目**：照 `scripts/crontab.example` 改路径后粘进 `crontab -e`，
按需取消注释 `HOLDOUT_DIR` / `CLAUDE_REVIEW=1` / `QUOTA_MAX_PCT=90`。

---

## 第 7 步 · 日常循环（每天早晨 15–30 分钟）

```bash
ls tasks/needs-review/                  # 1. 有没有卡住的
git log --oneline main..agent/auto      # 2. 昨晚干了什么
git diff main..agent/auto               # 3. 逐行读 diff（这是你不可替代的工作）
cat PROGRESS.md                         # 4. 证据核对
ls logs/*-review.md 2>/dev/null         # 5. 若开了评审钩子，读 Claude 的意见
git checkout main && git merge agent/auto && git push    # 6. 满意就合并
```
发现问题 → 照 `tasks/TEMPLATE.md` 写修正工单（如 `M02a-fix-xxx.md`）放进
`tasks/todo/` 队首，当晚自动修。

---

## 额度守门是怎么工作的

以周额度为主，阈值默认 90%（cron 里 `QUOTA_MAX_PCT` 可改）。两层：

1. **兜底层（零依赖，保证生效）**：任务失败且日志尾部出现限流特征时，自动写
   8 小时冷却（`COOLDOWN_HOURS` 可调），冷却期内 cron 触发直接跳过。
2. **预防层（可选）**：`scripts/check_quota.sh` 若能打印周用量百分比，≥阈值就
   不开新任务；打不出数字则放行、靠兜底层。想要精确百分比，照脚本里的注释
   接入 codex-cli-usage 或 codex-ratelimit 之一。

## 故障排查速查

| 症状 | 最可能原因 → 动作 |
|---|---|
| cron 没任何反应 | cron 服务没起 / PATH 缺 venv 或 codex → `service cron start`；核对 crontab 首行 PATH |
| `make check` 在第 2 步就红 | 环境问题不是代码问题 → 确认 python ≥3.11、venv 已激活、`make setup` 无报错 |
| codex exec 提示未登录 | token 失效或 ~/.codex 没挂载 → 重新 `codex login --device-auth`，补第 3.4 步 |
| 同一任务反复 needs-review | 颗粒度太大或规格含糊 → 按 TEMPLATE 拆成设计说明 + 实现两张工单 |
| 日志出现限流报错 | 正常，兜底层已写冷却 → 等待即可；顺手把报错原文补进 runner 的 grep 模式 |
| 想立刻停下一切 | `touch .agent-stop`（恢复：`rm .agent-stop`） |
