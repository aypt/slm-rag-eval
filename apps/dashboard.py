"""Streamlit demo for slm-rag-eval.

Deliberately thin: every computation lives in `slm_rag_eval.demo` / `slm_rag_eval.bench`,
so this file only collects input and renders output. Run it with `make demo`.
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

import streamlit as st

from slm_rag_eval.bench.analyze import analyze
from slm_rag_eval.core.config import Settings, get_settings
from slm_rag_eval.core.schemas import EvalRequest
from slm_rag_eval.demo import (
    highlight_placeholders,
    sanitized_preview,
    score_summary,
    token_totals,
    total_latency_ms,
    verdict_rows,
)
from slm_rag_eval.llm import build_client
from slm_rag_eval.metrics.registry import evaluate

st.set_page_config(page_title="slm-rag-eval", layout="wide")


def sidebar_settings() -> Settings:
    """Judge configuration, editable at runtime."""
    defaults = get_settings()
    with st.sidebar:
        st.header("Judge")
        base_url = st.text_input("Base URL", value=defaults.base_url)
        model = st.text_input("Model", value=defaults.model)
        privacy_mode = st.selectbox(
            "Privacy mode",
            options=["mask", "off"],
            index=0 if defaults.privacy_mode == "mask" else 1,
        )
        metrics = st.multiselect(
            "Metrics",
            options=["faithfulness", "relevance"],
            default=list(defaults.enabled_metrics),
        )
    return defaults.model_copy(
        update={
            "base_url": base_url,
            "model": model,
            "privacy_mode": privacy_mode,
            "enabled_metrics": metrics or ["faithfulness"],
        }
    )


def evaluate_tab(settings: Settings) -> None:
    """Paste an interaction, see what the judge is shown, then see the verdicts."""
    question = st.text_input("Question", value="Which material shields the module?")
    answer = st.text_area("Answer", value="The module uses a ceramic shield.")
    contexts_text = st.text_area(
        "Contexts (one per line)", value="The module is protected by a ceramic shield."
    )
    contexts = [line.strip() for line in contexts_text.splitlines() if line.strip()]
    request = EvalRequest(question=question, answer=answer, contexts=contexts)

    judge_request, mapping = sanitized_preview(request, settings=settings)
    st.subheader("What the judge will see")
    if settings.privacy_mode == "mask":
        st.caption(f"{len(mapping)} value(s) masked before anything leaves this process.")
    else:
        st.warning("Privacy mode is off: the text below is sent to the judge unchanged.")
    st.markdown(f"**Question:** {highlight_placeholders(judge_request.question, mapping)}")
    st.markdown(f"**Answer:** {highlight_placeholders(judge_request.answer, mapping)}")
    for index, context in enumerate(judge_request.contexts, start=1):
        st.markdown(f"**Context {index}:** {highlight_placeholders(context, mapping)}")

    if not st.button("Evaluate"):
        return

    with st.spinner("Asking the judge…"):
        result = asyncio.run(
            evaluate(
                request,
                judge=build_client(settings),
                metrics=list(settings.enabled_metrics),
                k=settings.k,
                strict=settings.strict,
                settings=settings,
            )
        )

    scores = score_summary(result)
    columns = st.columns(3)
    columns[0].metric("Faithfulness", _format_score(scores["faithfulness"]))
    columns[1].metric("Relevance", _format_score(scores["relevance"]))
    columns[2].metric("Judge latency", f"{total_latency_ms(result):.0f} ms")

    st.subheader("Claims")
    rows = verdict_rows(result)
    if rows:
        st.dataframe(rows, width="stretch")
    else:
        st.info("No claims were extracted from this answer.")
    st.caption(f"Tokens: {token_totals(result) or 'not reported by the backend'}")


def results_tab() -> None:
    """Render an M08 report from a benchmark JSONL file."""
    uploaded = st.file_uploader("Benchmark results (.jsonl)", type=["jsonl"])
    path_input = st.text_input("…or a path to a results file", value="")

    source: Path | None = None
    if uploaded is not None:
        temporary = Path(tempfile.gettempdir()) / uploaded.name
        temporary.write_bytes(uploaded.getvalue())
        source = temporary
    elif path_input.strip():
        source = Path(path_input.strip())

    if source is None:
        st.info("Upload a file written by `python -m slm_rag_eval.bench.run`.")
        return
    if not source.exists():
        st.error(f"No such file: {source}")
        return

    out_dir = Path(tempfile.mkdtemp(prefix="slmeval-report-"))
    outcome = analyze([source], out_dir)
    st.markdown(Path(outcome["summary_path"]).read_text(encoding="utf-8"))
    for figure in outcome["figures"]:
        st.image(str(figure), caption=Path(figure).stem)


def _format_score(score: float | None) -> str:
    return "n/a" if score is None else f"{score:.3f}"


def render() -> None:
    """Entry point: sidebar plus the two tabs."""
    st.title("slm-rag-eval")
    settings = sidebar_settings()
    evaluate_view, results_view = st.tabs(["Evaluate", "Results"])
    with evaluate_view:
        evaluate_tab(settings)
    with results_view:
        results_tab()


render()
