# Master plan — slm-rag-eval

规划原则（根据 proposal 修订）：**全量交付，不为时间砍范围。** Proposal 承诺的每一项交付物
都有对应任务；进度压力用"提高吞吐"解决（并行后台任务、夜间批量、更密的 cron 频率），
而不是删功能。

## Proposal 交付物 → 任务映射

| Proposal 承诺 | 对应任务 |
|---|---|
| SLM Adaptation（SLM 评估机制、多后端） | M01, M02, M03 |
| Privacy Architecture（PII 检测与脱敏层） | M04 |
| System Engineering（可复现、容器化、异步流水线） | M05, M06, M10 |
| Validation（SLM vs 云端 judge 对比分析） | M07, M08 |
| Functional Prototype | M01–M06 合计 |
| Technical Report 素材（准确率/延迟/成本 trade-off） | M08 |
| Demonstration Interface（UI/CLI） | M09 |
| Deployable Docker Compose file | M06 |
| Final Report / Source Code / Presentation | M10 + 全仓库 + docs/demo.md |

## 依赖关系（tasks/todo 的文件名顺序 = 执行顺序）

```
M00 ──> M01 ──> M02 ──> M03 ──> M04 ──> M05 ──> M06 ──> M07 ──> M08 ──> M09 ──> M10
                                 │
                                 └── M04 是隐私不变量测试的起点，之后所有任务都受其约束
```

严格串行是默认策略：顺序执行永远安全，且 `run_next_task.sh` 天然按文件名顺序取任务。
确有余力时可并行的组合（需要 git worktree，见 docs/automation.md 进阶节）：
- M06（Docker）与 M07（bench harness）互不触碰同一批文件；
- M09 的 CLI 部分只依赖 M05，可与 M08 并行。

## 吞吐杠杆（替代"砍范围"的手段）

1. 夜间批量：`scripts/run_all_tasks.sh` 一晚可连续消化多个任务（队列空或首个失败即停）。
2. 白天人审 + 晚上机跑的节奏：早晨 15–30 分钟审 diff、把修改意见写成小任务文件插到
   `tasks/todo/` 队首（命名如 `M02a-fix-batching.md`），当晚自动执行。
3. 计算型长活（bench 实验本身）与开发型任务分离：M07 完成后，实验跑批用另一个 cron 条目
   直接调 `python -m slm_rag_eval.bench.run ...`，不占用 agent 任务队列。
4. 质量判断永远留给人：指标 prompt 好不好、实验结论怎么解读、报告怎么写——这些不进队列。

## 里程碑验收总则

每个任务的完成定义都是机器可验证的：`make check` 全绿 + 任务文件内列出的验收项在
PROGRESS.md 留下证据。人工评审只做机器做不了的事：设计取舍、代码品味、实验结论。
