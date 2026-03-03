"""Smoke test: every module (including stubs) must at least import cleanly."""

from __future__ import annotations

import importlib

import pytest

MODULES = [
    "slm_rag_eval",
    "slm_rag_eval.cli",
    "slm_rag_eval.core.config",
    "slm_rag_eval.core.schemas",
    "slm_rag_eval.llm.client",
    "slm_rag_eval.metrics.faithfulness",
    "slm_rag_eval.metrics.relevance",
    "slm_rag_eval.metrics.registry",
    "slm_rag_eval.privacy.sanitizer",
    "slm_rag_eval.service.api",
    "slm_rag_eval.service.db",
    "slm_rag_eval.service.worker",
    "slm_rag_eval.bench.datasets",
    "slm_rag_eval.bench.run",
    "slm_rag_eval.bench.analyze",
]


@pytest.mark.parametrize("name", MODULES)
def test_module_imports(name: str) -> None:
    importlib.import_module(name)
