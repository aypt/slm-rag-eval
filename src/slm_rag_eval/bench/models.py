"""Freeze the exact model artifacts an experiment used (report Table 5.2).

"qwen3:8b" is a moving tag, not an artifact. Ollama republishes tags, quantizations differ
between them, and a report that names only the tag cannot be re-run a year later with any
confidence that the same weights answered. The digest can.

Everything here is read from the running Ollama, so the table describes what actually served
the requests rather than what someone intended to pull.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

app = typer.Typer(help=__doc__, add_completion=False)


@dataclass
class ModelArtifact:
    """One judge model, identified precisely enough to obtain it again."""

    tag: str
    digest: str | None = None
    parameter_size: str | None = None
    quantization: str | None = None
    family: str | None = None
    context_length: int | None = None
    size_bytes: int | None = None
    license_name: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


def _ollama_root(base_url: str) -> str:
    """Ollama's native API sits beside the OpenAI-compatible `/v1` path, not under it."""
    trimmed = base_url.rstrip("/")
    return trimmed[: -len("/v1")] if trimmed.endswith("/v1") else trimmed


def fetch_artifact(tag: str, base_url: str, timeout_s: float = 30.0) -> ModelArtifact:
    """Ask Ollama what this tag actually is. Missing fields stay None rather than guessed."""
    root = _ollama_root(base_url)
    artifact = ModelArtifact(tag=tag)
    try:
        response = httpx.post(f"{root}/api/show", json={"model": tag}, timeout=timeout_s)
        response.raise_for_status()
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return artifact
    if not isinstance(body, dict):
        return artifact

    raw_details = body.get("details")
    details: dict[str, Any] = raw_details if isinstance(raw_details, dict) else {}
    artifact.details = details
    artifact.parameter_size = details.get("parameter_size")
    artifact.quantization = details.get("quantization_level")
    artifact.family = details.get("family")
    licence = body.get("license")
    if isinstance(licence, str) and licence.strip():
        artifact.license_name = licence.strip().splitlines()[0][:120]

    raw_info = body.get("model_info")
    info: dict[str, Any] = raw_info if isinstance(raw_info, dict) else {}
    for key, value in info.items():
        if key.endswith(".context_length") and isinstance(value, int):
            artifact.context_length = value
            break

    # /api/tags carries the digest and size; /api/show does not.
    try:
        listing = httpx.get(f"{root}/api/tags", timeout=timeout_s).json()
        for entry in listing.get("models", []) if isinstance(listing, dict) else []:
            if isinstance(entry, dict) and entry.get("name") == tag:
                digest = entry.get("digest")
                artifact.digest = digest if isinstance(digest, str) else None
                size = entry.get("size")
                artifact.size_bytes = size if isinstance(size, int) else None
                break
    except (httpx.HTTPError, ValueError):
        pass
    return artifact


def render(artifacts: list[ModelArtifact]) -> str:
    """Report Table 5.2 as Markdown."""
    lines = [
        "### Judge model artifacts",
        "",
        "| Tag | Parameters | Quantization | Context | Size | Digest |",
        "|---|---|---|---|---|---|",
    ]
    for item in artifacts:
        size = f"{item.size_bytes / 1e9:.1f} GB" if item.size_bytes else "unknown"
        context = f"{item.context_length:,}" if item.context_length else "unknown"
        digest = f"`{item.digest[:16]}`" if item.digest else "unavailable"
        lines.append(
            f"| `{item.tag}` | {item.parameter_size or 'unknown'} | "
            f"{item.quantization or 'unknown'} | {context} | {size} | {digest} |"
        )
    lines += [
        "",
        "Digests identify the artifact that actually answered. A tag alone is republishable "
        "and does not pin the weights, so it cannot support a reproducibility claim on its own.",
    ]
    return "\n".join(lines)


def freeze(tags: list[str], base_url: str, out: Path) -> list[ModelArtifact]:
    """Record every tag's artifact identity into the report directory."""
    artifacts = [fetch_artifact(tag, base_url) for tag in tags]
    out.mkdir(parents=True, exist_ok=True)
    (out / "models.md").write_text(render(artifacts) + "\n", encoding="utf-8")
    (out / "models.json").write_text(
        json.dumps([asdict(item) for item in artifacts], indent=2), encoding="utf-8"
    )
    return artifacts


@app.command()
def main(
    model: Annotated[list[str] | None, typer.Option("--model", help="Tag; repeat.")] = None,
    base_url: Annotated[
        str, typer.Option(help="Judge endpoint; the Ollama API is read beside it.")
    ] = "http://localhost:11434/v1",
    out: Annotated[Path, typer.Option(help="Where to write the table.")] = Path("report/models"),
) -> None:
    """Freeze the exact artifacts behind the given tags."""
    tags = list(model or [])
    if not tags:
        raise typer.BadParameter("Pass at least one --model.")
    artifacts = freeze(tags, base_url, out)
    typer.echo(render(artifacts))
    missing = [item.tag for item in artifacts if item.digest is None]
    if missing:
        typer.echo("")
        typer.echo(
            f"WARNING: no digest for {', '.join(missing)} — is Ollama running at {base_url}, "
            "and is the model pulled?",
            err=True,
        )
    typer.echo("")
    typer.echo(f"wrote {out / 'models.md'}")


if __name__ == "__main__":  # pragma: no cover
    app()
