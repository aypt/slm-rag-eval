# M00 · Project scaffold (completed at repo creation)

Scaffolded: pyproject (fastapi, pydantic v2, sqlalchemy, httpx, typer, tenacity; dev:
pytest, pytest-asyncio, ruff, mypy), src/slm_rag_eval package (core schemas implemented,
LLMClient Protocol pinned, all other modules stubbed), tests with FakeLLMClient fixture,
Makefile (setup/check/test/run), GitHub Actions CI running `make check`, AGENTS.md,
task-runner scripts. Verified: `make check` green.
