"""Structural checks on the deployment files.

The docker CLI is not available in every dev environment, so these tests parse the compose
file and the Dockerfile directly instead of shelling out; CI runs the real `docker build` and
`docker compose config` on top of them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices
from pydantic.fields import FieldInfo

from slm_rag_eval.core.config import Settings

REPO_ROOT = Path(__file__).resolve().parents[2]
COMPOSE = yaml.safe_load((REPO_ROOT / "docker-compose.yml").read_text())
DOCKERFILE = (REPO_ROOT / "Dockerfile").read_text()
ENV_EXAMPLE = (REPO_ROOT / ".env.example").read_text()


def _service(name: str) -> dict[str, Any]:
    services: dict[str, Any] = COMPOSE["services"]
    assert name in services, f"compose is missing the {name} service"
    return dict(services[name])


def _documented_env_keys() -> set[str]:
    """Every KEY= in .env.example, including the commented-out optional ones."""
    keys: set[str] = set()
    for raw_line in ENV_EXAMPLE.splitlines():
        line = raw_line.lstrip("# ").strip()
        if "=" in line and not line.startswith("-"):
            keys.add(line.split("=", 1)[0].strip())
    return keys


def _env_var_name(field_name: str, field: FieldInfo) -> str:
    alias = field.validation_alias
    if isinstance(alias, AliasChoices):
        return str(alias.choices[0])
    if isinstance(alias, str):
        return alias
    return f"SLMEVAL_{field_name.upper()}"


def test_compose_defines_the_full_stack() -> None:
    assert set(COMPOSE["services"]) == {"api", "worker", "db", "ollama", "ollama-init"}
    assert set(COMPOSE["volumes"]) == {"pgdata", "ollama"}


def test_api_serves_http_while_the_worker_runs_separately() -> None:
    api = _service("api")
    worker = _service("worker")

    assert api["environment"]["WORKER_EMBEDDED"] == "0"
    assert "8000:8000" in api["ports"]
    assert api["depends_on"]["db"]["condition"] == "service_healthy"
    assert worker["command"] == ["python", "-m", "slm_rag_eval.service.worker"]
    assert worker["depends_on"]["db"]["condition"] == "service_healthy"
    assert api["env_file"] == worker["env_file"] == ".env"


def test_worker_does_not_inherit_the_api_healthcheck() -> None:
    """The image probes /healthz; only the api serves it, so the worker must opt out."""
    worker = _service("worker")
    api = _service("api")

    assert worker["healthcheck"] == {"disable": True}
    assert worker["image"] == api["image"]  # same image, hence the inherited HEALTHCHECK
    assert "ports" not in worker
    assert "HEALTHCHECK" in DOCKERFILE


def test_database_is_postgres_with_a_readiness_check_and_named_volume() -> None:
    db = _service("db")

    assert db["image"] == "postgres:16"
    assert "pg_isready" in " ".join(db["healthcheck"]["test"])
    assert db["volumes"] == ["pgdata:/var/lib/postgresql/data"]


def test_ollama_init_pulls_the_configured_judge_model() -> None:
    init = _service("ollama-init")
    command = " ".join(init["command"])

    assert "ollama pull" in command
    assert "JUDGE_MODEL" in command
    assert init["restart"] == "no"
    assert _service("ollama")["volumes"] == ["ollama:/root/.ollama"]


def test_the_base_stack_requests_no_gpu_so_it_starts_on_any_host() -> None:
    """A reservation for an nvidia driver stops the whole stack on a host without one."""
    assert _service("ollama").get("deploy") is None


def test_the_gpu_overlay_grants_the_judge_a_gpu_and_nothing_else() -> None:
    """A commented-out block only works if someone remembers to uncomment it.

    The overlay is applied with `-f docker-compose.yml -f docker-compose.gpu.yml`, which is
    a flag the runbook can carry and a file this test can check. Without a GPU the judge
    still answers, just far slower — a silent failure that is expensive on rented hardware,
    so it is worth making declarative.
    """
    import yaml

    overlay = yaml.safe_load((REPO_ROOT / "docker-compose.gpu.yml").read_text())

    assert set(overlay["services"]) == {"ollama"}, "the overlay must not touch other services"
    devices = overlay["services"]["ollama"]["deploy"]["resources"]["reservations"]["devices"]
    assert devices[0]["driver"] == "nvidia"
    assert "gpu" in devices[0]["capabilities"]


def test_the_api_and_worker_wait_for_the_model_pull_to_finish() -> None:
    """Depending on `ollama` alone let a job be submitted mid-pull, which fails obscurely."""
    for name in ("api", "worker"):
        depends = _service(name)["depends_on"]
        assert depends["ollama-init"]["condition"] == "service_completed_successfully", name


def test_dockerfile_is_multi_stage_slim_non_root_and_healthchecked() -> None:
    assert DOCKERFILE.count("FROM python:3.11-slim") == 2
    assert "AS builder" in DOCKERFILE
    assert "python -m spacy download en_core_web_lg" in DOCKERFILE
    assert "USER slmeval" in DOCKERFILE
    assert "HEALTHCHECK" in DOCKERFILE
    assert "/healthz" in DOCKERFILE
    # The runtime stage must not carry the build toolchain.
    runtime_stage = DOCKERFILE.split("AS runtime", 1)[1]
    assert "build-essential" not in runtime_stage


def test_env_example_documents_every_setting() -> None:
    documented = _documented_env_keys()
    missing = {
        _env_var_name(name, field)
        for name, field in Settings.model_fields.items()
        if _env_var_name(name, field) not in documented
    }

    assert not missing, f".env.example does not document: {sorted(missing)}"
    assert "JUDGE_MODEL" in documented, "ollama-init needs JUDGE_MODEL documented"
