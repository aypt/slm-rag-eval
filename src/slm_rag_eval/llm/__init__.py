"""LLM inference clients and structured-output helpers."""

from __future__ import annotations

from slm_rag_eval.core.config import Settings
from slm_rag_eval.llm.client import LLMClient, LLMResponse, OpenAICompatClient
from slm_rag_eval.llm.errors import JSONGenerationError
from slm_rag_eval.llm.structured import generate_json

__all__ = [
    "JSONGenerationError",
    "LLMClient",
    "LLMResponse",
    "OpenAICompatClient",
    "build_client",
    "generate_json",
]


def build_client(settings: Settings) -> LLMClient:
    """Build the configured LLM backend."""
    return OpenAICompatClient(settings)
