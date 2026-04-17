# Demo walkthrough

A copy-paste script for recording the project demo. Every command below runs from a clean
checkout; the only prerequisite that takes real time is pulling the judge model.

## 0 · Setup (once)

```bash
source .venv/bin/activate
make setup                      # package + dev + analysis extras + the spaCy model
pip install -e ".[demo]"        # Streamlit, only needed for step 4
make check                      # expect: ruff clean, mypy clean, all tests passed
```

Start a judge. Either a local Ollama:

```bash
ollama serve &
ollama pull qwen2.5:7b-instruct
export SLMEVAL_BASE_URL=http://localhost:11434/v1
export SLMEVAL_MODEL=qwen2.5:7b-instruct
```

…or the whole stack in containers (`cp .env.example .env && make docker-build && make docker-up`).

## 1 · Score one answer from the terminal

```bash
rageval eval \
  --question "What did the Aurora probe carry?" \
  --answer   "The Aurora probe carried 4 instruments and launched in 2019." \
  --context  "Aurora launched in 2019 carrying 3 instruments."
```

Expected output shape — this transcript is captured by the test runner
(`tests/test_cli.py::test_demo_transcript_matches_the_documented_output`) with a scripted
judge, so the wording is exact and only the two timings vary:

```text
CLAIM                                    VERDICT      REASON
------------------------------------------------------------
The Aurora probe carried 4 instruments.  unsupported  The context says 3 instruments.
The Aurora probe launched in 2019.       supported    The context gives the same year.

faithfulness: 0.500
judge: qwen2.5:7b-instruct   privacy: off
judge time: <ms> ms   wall: <ms> ms
```

One claim contradicts the context and one is supported, so faithfulness is 0.5. Add
`--fail-under 0.8` to make the command exit 1 — that is how it drops into CI.

Show the privacy layer by feeding it invented PII with masking on (the default):

```bash
SLMEVAL_PRIVACY_MODE=mask rageval eval \
  --question "Who filed ticket 42?" \
  --answer   "Rowan Quill filed it from rowan@example.com." \
  --context  "Ticket 42 was filed by Rowan Quill (rowan@example.com)."
```

The verdicts come back with the real names restored, but nothing left the process unmasked —
step 4 shows the masked text the judge actually saw.

## 2 · Score a file

```bash
printf '%s\n' \
  '{"id":"a","question":"What did Aurora carry?","answer":"Three instruments.","contexts":["Aurora carried 3 instruments."]}' \
  > /tmp/demo.jsonl

rageval batch /tmp/demo.jsonl --out /tmp/demo-results.jsonl
cat /tmp/demo-results.jsonl
```

## 3 · Benchmark and report

```bash
python scripts/download_ragtruth.py
python -m slm_rag_eval.bench.run --dataset ragtruth --judge slm --limit 20 --out results/
python -m slm_rag_eval.bench.analyze results/ragtruth_slm.jsonl --out report/
open report/summary.md            # tables + three PNG figures
```

`docs/sample_report/` holds the same output built from synthetic fixtures, if you want to show
the report shape without waiting for a real run.

## 4 · Dashboard

```bash
make demo                          # streamlit run apps/dashboard.py
```

- **Evaluate** tab: paste the PII example from step 1. The "What the judge will see" panel shows
  the masked text with every placeholder in bold — that is the screenshot worth recording.
  Press *Evaluate* for the claim table, the score metrics, and the latency.
- **Results** tab: upload `results/ragtruth_slm.jsonl` (or type its path) to render the summary
  tables and figures inline.
- Sidebar: change base URL, model, privacy mode, and metrics without restarting.

## 5 · Service

```bash
rageval serve &                    # or: make run
curl -s localhost:8000/healthz
```

Then the submit/poll round trip from the README's "Run the service" section.
