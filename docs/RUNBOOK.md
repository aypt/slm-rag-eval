# Rental-day runbook

The experiment is meant to be execution, not improvisation. This is the exact sequence, in
order, with the checkpoint that must pass before each paid step.

The design assumption throughout: **every command here can be rehearsed for free before the
machine is rented.** Nothing below discovers a problem for the first time on the clock.

---

## Before you rent (free, on any machine)

### 1. Cache the dataset

```bash
python scripts/download_ragtruth.py          # writes data/ragtruth/
python -m slm_rag_eval.bench.datasets --dataset ragtruth --split test
```

The second command prints report **Table 5.1**. Read the label balance now: the experiment
plan depends on it, and it is the number the Limitations section has to be honest about.

### 2. Rehearse the whole pipeline with no judge

```bash
python -m slm_rag_eval.bench.experiment --model rehearsal --no-cloud \
  --dataset ragtruth --split test --limit 20 --out /tmp/rehearsal --dry-run
```

`--dry-run` prints the matrix and exits. Drop it and point `SLMEVAL_BASE_URL` at any
OpenAI-compatible endpoint to rehearse for real.

### 3. Produce the PII numbers — these need no GPU at all

```bash
python -m slm_rag_eval.privacy.evaluation --out report/privacy
```

This is the whole of **Q3's detection half** and it runs on a laptop in seconds. There is no
reason to spend rented time on it.

---

## On the rented machine

### 4. Start the judge and wait for the pull to finish

```bash
ollama serve &
ollama pull qwen2.5:7b-instruct
ollama pull llama3.2:3b
ollama pull phi3.5:3.8b-mini-instruct-q4_0
ollama list                                   # all three must appear before continuing
```

The single most common way to waste rented time is submitting work while a multi-gigabyte
pull is still running. `ollama list` is the gate.

### 5. Preflight — **do not skip this**

Run it once per model:

```bash
for m in qwen2.5:7b-instruct llama3.2:3b phi3.5:3.8b-mini-instruct-q4_0; do
  SLMEVAL_MODEL=$m python -m slm_rag_eval.bench.preflight \
    --dataset ragtruth --split test --limit 150 --stratify --sample-size 3
done
```

Every line must be `PASS` or an understood `WARN`. Any `FAIL` exits non-zero, and each names
its own fix:

| Failure | What it means |
|---|---|
| `model available` | The pull did not finish, or the tag is spelled differently |
| `truncation` | Raise `SLMEVAL_MAX_TOKENS`; long samples are being cut off |
| `held-out split` | Too few positives to report a held-out F1 — raise `--limit` |
| `output directory` | Rows from a different configuration are already there |
| `live round trip` | The judge is reachable but is not producing usable output |

**Read the `projected 200-sample run` line before going further.** It is a measured
extrapolation from real samples on this machine, and it is how you decide how many hours to
rent and whether `--limit` needs to come down.

### 6. Run the experiment

```bash
export CLOUD_BASE_URL=https://api.openai.com/v1
export CLOUD_MODEL=gpt-4o-mini
export CLOUD_API_KEY=...

python -m slm_rag_eval.bench.experiment \
  --model qwen2.5:7b-instruct \
  --model llama3.2:3b \
  --model phi3.5:3.8b-mini-instruct-q4_0 \
  --ablation-model qwen2.5:7b-instruct \
  --dataset ragtruth --split test --limit 150 --stratify --seed 20260806 \
  --cloud --cloud-input-cost-per-1m 0.15 --cloud-output-cost-per-1m 0.60 \
  --out report/experiment
```

One judge failing does not abort the rest — the run log records what happened to each leg.

> **The cloud leg sends benchmark text to a third party.** RAGTruth is public, which is why
> this is acceptable here. Never point the cloud judge at private or production data.

### 7. Collect

```
report/experiment/
  environment.md        Table 5.3   (the machine, captured automatically)
  dataset.md            Table 5.1
  summary.md            Tables 6.1–6.3, in-sample and held-out
  results.json          every number above, machine-readable
  reliability.png       Figure 6.1
  tradeoff.png          Figure 6.3  ← the accuracy/latency/cost figure
  roc_curves.png, score_distributions.png, latency_box.png
  errors/               Table 6.4 with worked examples
  privacy/              PII precision and recall
  rows/                 raw JSONL + manifests behind all of it
  run_log.md            what each leg did, including failures
```

Also capture, by hand, while the machine is still alive:

```bash
nvidia-smi --query-gpu=name,memory.used --format=csv   # R4 peak VRAM, during a run
ollama ps                                              # resident model size
```

### 8. Before releasing the machine

- [ ] `run_log.md` shows every leg `ok` (or a `partial` you can explain)
- [ ] `summary.md` held-out table has numbers, not `n/a`
- [ ] `tradeoff.png` has one point per judge
- [ ] `report/experiment/rows/` is copied off the machine — it is the only irreplaceable part

---

## Which numbers go in the report

| Report item | Where it comes from |
|---|---|
| Q1 reliability | `summary.md` **held-out** table — not the in-sample one |
| Q1 SLM-vs-cloud agreement | `summary.md` judge-agreement table (κ, Pearson, Spearman) |
| Q2 latency | `summary.md` efficiency table (median, p95) |
| Q2 cost | same table; state the per-token rates you passed in |
| Q2 trade-off | `tradeoff.png` |
| Q3 PII detection | `privacy/pii_detection.md` |
| Q3 ablation | `summary.md` privacy-ablation table (ΔF1, ΔAcc) |
| Q4 deployment | `docker compose up` transcript + `environment.md` |
| R4 resources | `nvidia-smi` / `ollama ps` captured in step 7 |
| R6 error analysis | `errors/error_analysis.md` |

**Quote the held-out numbers.** The in-sample table picks its threshold on the rows it
reports, so its F1 is an upper bound; both are printed precisely so the difference is
visible rather than hidden.

`n/a` in the held-out table never means "the judge scored zero" — it means one half of the
split held a single class, so no threshold was distinguishable. Fix it with a larger
`--limit`, not by quoting the in-sample number instead.

---

## Still to do by hand

These are not automated, and the report needs them:

- **Docker Compose evidence (Q4).** `docker compose up` with a transcript. Wait for
  `ollama-init` to exit successfully before submitting a job — the API and worker do not
  currently wait for it.
- **Peak VRAM/RAM (R4).** Step 7's manual capture.
- **The candidate-SLM table (Table 5.2).** Parameter counts, quantization and context
  lengths come from the model cards, not from this code.
