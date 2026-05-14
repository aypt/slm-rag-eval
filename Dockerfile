# Multi-stage build: the spaCy model is ~400 MB and the compile toolchain is only needed to
# build wheels, so both are confined to the builder and only site-packages is carried over.
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /src

RUN apt-get update \
    && apt-get install --no-install-recommends -y build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md constraints.txt ./
COPY src ./src

# -c pins the same versions the test suite and the benchmark report were produced with.
RUN pip install ".[postgres]" -c constraints.txt \
    && python -m spacy download en_core_web_lg


FROM python:3.11-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    SLMEVAL_BASE_URL=http://ollama:11434/v1 \
    DATABASE_URL=sqlite+aiosqlite:///data/jobs.db

COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Non-root runtime user; /app/data is where the SQLite fallback writes.
RUN useradd --create-home --uid 10001 slmeval \
    && mkdir -p /app/data \
    && chown -R slmeval:slmeval /app

WORKDIR /app
USER slmeval

EXPOSE 8000

# urllib rather than curl: no extra package, and a non-2xx raises like `curl --fail`.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz', timeout=4)"]

CMD ["uvicorn", "slm_rag_eval.service.api:app", "--host", "0.0.0.0", "--port", "8000"]
