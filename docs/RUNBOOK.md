# Rental-day runbook

**Audience: the agent operating the rented GPU machine.** This file is self-contained. You
do not need any other document, and you do not need the report drafts — they stay on the
author's machine.

Your job has exactly one deliverable: **`report/experiment.tar.gz`**, copied off this host
before it is released. Everything else is a means to that end.

---

## 0. Rules that override anything else you might infer

1. **Never change evaluation logic to improve a number.** Not prompts, not claim extraction,
   not verdict rules, not the score equation, not the threshold grid, not the error buckets,
   not the PII corpus. If a model scores badly, that is the result. Report it.
2. **Never weaken a check to get past it.** A `FAIL` is information. Fix the cause or stop
   and report; do not skip, retry blindly, or edit the test.
3. **Secrets go in environment variables only.** Never write an API key into `.env`, a
   source file, a command file, a manifest, a log, or any file in the bundle.
4. **The raw rows are irreplaceable.** Figures and tables can be recomputed from
   `report/experiment/rows/`; the rows cannot be recomputed from anything. If you must
   abandon a step, preserve the rows first.
5. **If a step fails, keep going where the runbook says it is non-blocking.** A stack that
   will not start does not invalidate measurements already taken.

---

## 1. What arrives with you

The project folder is copied onto this host (there is no GitHub remote). It must contain:

| Path | Why it matters |
|---|---|
| `.git/` | Every manifest records `git_sha`. Without it, provenance reads "not a git checkout". |
| `data/ragtruth/` | 35 MB, already downloaded. Avoids re-fetching a mutable branch. |
| `dataset.lock.json` | Proves this host's data is byte-identical to the author's. |
| `report/cloud-baseline/` | Cloud judge rows scored in advance. See §5. |
| everything else | Source, tests, compose files. |

It should **not** contain `.venv/` — rebuild it here.

Verify before you start:

```bash
test -d .git && test -d data/ragtruth && test -f dataset.lock.json && echo "transfer ok"
```

---

## 2. Prepare the host

```bash
python3 -m venv .venv && source .venv/bin/activate
make setup
nvidia-smi                 # must list a GPU; if not, stop and report
```

Start the model server and **wait for every pull to finish**:

```bash
ollama serve &
ollama pull qwen3:8b
ollama pull qwen3:4b-instruct
ollama pull gemma3:4b
ollama list                # all three must appear before you continue
```

Submitting work while a multi-gigabyte pull is running is the single most common way to
waste rented time. `ollama list` is the gate.

Set the run configuration:

```bash
export SLMEVAL_BASE_URL=http://localhost:11434/v1
export SLMEVAL_MAX_TOKENS=8192      # 4096 truncated real RAGTruth samples; do not lower
export SLMEVAL_TIMEOUT_S=300
export SLMEVAL_PRIVACY_MODE=mask
export SLMEVAL_DISABLE_THINKING=true
```

`SLMEVAL_DISABLE_THINKING` matters more than it looks. Qwen3 is a hybrid-reasoning model;
left on, it spends 339 completion tokens and 5.6 s per call on a chain of thought instead of
47 tokens and 1.7 s, for the same answer. Over three models × 150 samples that is hours of
billed time.

---

## 3. Run everything

One command. It gates each expensive step behind a cheap one:

```bash
python -m slm_rag_eval.bench.campaign \
  --model qwen3:8b \
  --model qwen3:4b-instruct \
  --model gemma3:4b \
  --include-rows report/cloud-baseline/openai-gpt-5-mini/ragtruth_cloud.jsonl \
  --include-rows report/cloud-baseline/anthropic-claude-haiku-4.5/ragtruth_cloud.jsonl \
  --limit 150 --ablation-limit 100 --seed 20260806 \
  --out report/experiment \
  2>&1 | tee campaign-console.log
```

What it does, in order:

| Step | Blocking | If it fails |
|---|---|---|
| `make check` | yes | Stop. The code is not in a state that can produce a defensible number. |
| dataset lock | yes | Stop. This host's data differs from the author's; the draw would not match. |
| freeze model artifacts | no | Continue, but Table 5.2 will lack digests — note it. |
| preflight, per model | yes | Stop **for that model**. See §4. |
| primary matrix | yes | Rows written so far are kept. Preserve them, then report. |
| masking ablation | no | Continue; R5 will be missing. |
| PII detection | no | Continue; needs no GPU, so a failure here is a bug worth reporting. |
| compose deployment | no | Continue. Q4 evidence is lost, the measurements are not. |
| bundle | no | Run `python -m slm_rag_eval.bench.bundle report/experiment` manually. |

Expect roughly 4–6 hours in total on a 24 GB card. The preflight prints a measured
projection per model before the expensive step — **read it**, and if the projected total
exceeds your rental window, reduce `--limit` and record the reduction rather than letting
the machine expire mid-run.

---

## 4. Preflight failures and what each one means

| Failure | Cause | Action |
|---|---|---|
| `model available` | The pull did not finish, or the tag differs | `ollama list`, pull again, retry |
| `truncation` | `SLMEVAL_MAX_TOKENS` too small for real samples | Raise it, then use the **same** value for every model so the legs stay comparable |
| `held-out split` | Too few positives to report a held-out F1 | Raise `--limit`. Do not proceed and quote in-sample numbers instead |
| `output directory` | Rows from a different configuration are present | Use a fresh `--out`; never mix |
| `live round trip` 0/N | The judge does not produce usable output | Report it. This is a finding about the model, not a thing to work around |
| `privacy layer` | Presidio or its spaCy model did not load | `make setup` again; masking must be active |

**A model that fails preflight is a result.** Record it, drop that model from the matrix,
and run the remaining models. Do not substitute a different model without saying so.

> Known risk, already measured: `gemma3:4b` — on an OpenRouter-served copy of this model,
> some samples failed the verdict-alignment check because the model would not echo claims
> verbatim. Ollama's copy may differ. If its coverage is poor, report the coverage honestly
> rather than adjusting the alignment check, and note `phi4-mini:3.8b` as the documented
> alternative if a third model is needed.

---

## 5. Why the cloud baselines were run elsewhere

They were scored on the author's machine before this host was hired, and arrive as rows in
`report/cloud-baseline/`. A cloud judge costs the same per token wherever the request comes
from, and it answers in roughly 90 s per sample — 150 samples would spend nearly four hours
of rented GPU time waiting on a network call.

`--include-rows` folds them into this host's analysis as their own series. They keep their
own manifests, so their environment and provenance stay distinct and honest.

If you need the cloud latency measured *from this host* for a like-for-like Q2 comparison,
run a short probe after the main matrix and label it clearly:

```bash
CLOUD_BASE_URL=... CLOUD_MODEL=... CLOUD_API_KEY=... \
python -m slm_rag_eval.bench.run --judge cloud --dataset ragtruth \
  --split test --limit 20 --stratify --seed 20260806 \
  --out report/experiment/cloud-latency-probe
```

---

## 6. Before you release the machine

Run the checklist:

```bash
cd report/experiment && sha256sum -c MANIFEST.sha256 | tail -3 && cd -
```

- [ ] `report/experiment/INDEX.md` lists no **required** artifact as missing
- [ ] `campaign_log.md` shows every blocking step `ok`
- [ ] `run_log.md` shows each leg's coverage and peak VRAM
- [ ] `summary.md` held-out table contains numbers, not `n/a`
- [ ] `tradeoff.png` has one point per judge
- [ ] `MANIFEST.sha256` verifies

Then copy off the host:

```
report/experiment.tar.gz     <- this is the deliverable
campaign-console.log         <- the console transcript
```

Verify the archive arrived intact on the destination:

```bash
tar tzf report/experiment.tar.gz | wc -l     # non-zero
```

**Do not release the machine until the archive is confirmed on the destination.** Everything
here disappears with the instance.

---

## 7. What is inside the deliverable

`INDEX.md` inside the bundle maps every file to the report item it answers. In short:

| Report item | File |
|---|---|
| Q1 reliability | `summary.md` — **held-out** table |
| Q1 SLM-vs-cloud agreement | `summary.md` — judge agreement (κ, Pearson, Spearman) |
| Q2 latency | `summary.md` — efficiency table (median, p95) |
| Q2 cost | `summary.md` + `results.json` |
| Q2 trade-off figure | `tradeoff.png` |
| Q3 PII detection | `privacy/pii_detection.md` |
| Q3 masking ablation | `ablation/summary.md` |
| Q4 deployment | `deployment/deployment.md` |
| R4 resources | `resources.json`, `run_log.md` |
| R6 error analysis | `errors/error_analysis.md` |
| Table 5.1 / 5.2 / 5.3 | `dataset.md` / `models/models.md` / `environment.md` |
| Testing evidence | `make_check.txt` |
| Everything, raw | `rows/**/*.jsonl` and their manifests |

**Quote the held-out numbers.** The in-sample table above them picks its threshold on the
rows it reports, so its F1 is an upper bound. Both are printed precisely so the difference
is visible; `n/a` there means the split could not measure it, never that the judge scored
zero.

---

## 8. Reporting back

Write a short summary of: which models ran, coverage per leg, whether the deployment check
passed, anything that failed and why, and the total rental hours used. Attach
`campaign_log.md`.

If something went wrong that this runbook did not anticipate, say so plainly rather than
working around it. An honest gap is recoverable; a number produced by an undocumented
workaround is not.
