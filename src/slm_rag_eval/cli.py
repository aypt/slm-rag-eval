"""Command line interface. Extended across tasks M07–M09."""

from __future__ import annotations

import typer

from slm_rag_eval import __version__

app = typer.Typer(help="slm-rag-eval — privacy-preserving RAG evaluation with SLM judges.")


@app.callback()
def main() -> None:
    """Privacy-preserving RAG evaluation with SLM judges."""


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


if __name__ == "__main__":  # pragma: no cover
    app()
