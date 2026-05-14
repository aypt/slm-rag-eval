.PHONY: setup lint type test check run demo reproduce docker-build docker-up docker-down docker-logs \
	next-task all-tasks

setup:
	pip install -e ".[dev,analysis]" -c constraints.txt
	python -m spacy download en_core_web_lg

lint:
	ruff check src tests

type:
	mypy src

test:
	pytest -q

check: lint type test

run:
	uvicorn slm_rag_eval.service.api:app --reload --host 0.0.0.0 --port 8000

demo:
	streamlit run apps/dashboard.py

reproduce:
	python -m slm_rag_eval.bench.reproduce

docker-build:
	docker compose build

docker-up:
	docker compose up -d

docker-down:
	docker compose down

docker-logs:
	docker compose logs -f api worker

next-task:
	bash scripts/run_next_task.sh

all-tasks:
	bash scripts/run_all_tasks.sh
