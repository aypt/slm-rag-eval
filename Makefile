.PHONY: setup lint type test check run next-task all-tasks

setup:
	pip install -e ".[dev]"

lint:
	ruff check src tests

type:
	mypy src

test:
	pytest -q

check: lint type test

run:
	uvicorn slm_rag_eval.service.api:app --reload --host 0.0.0.0 --port 8000

next-task:
	bash scripts/run_next_task.sh

all-tasks:
	bash scripts/run_all_tasks.sh
