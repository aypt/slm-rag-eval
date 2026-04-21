#!/usr/bin/env python
"""Emit the README configuration table straight from the Settings model.

Run `python scripts/emit_config_table.py` and paste the output into README.md so the
documented defaults can never drift from the code.
"""

from __future__ import annotations

from pydantic import AliasChoices
from pydantic.fields import FieldInfo

from slm_rag_eval.core.config import Settings


def env_var_name(field_name: str, field: FieldInfo) -> str:
    """The environment variable that sets this field (alias wins over the SLMEVAL_ prefix)."""
    alias = field.validation_alias
    if isinstance(alias, AliasChoices):
        return str(alias.choices[0])
    if isinstance(alias, str):
        return alias
    return f"SLMEVAL_{field_name.upper()}"


def default_text(field_name: str) -> str:
    """The default as a reader would type it into a shell."""
    value = getattr(Settings(_env_file=None), field_name)
    if value is None:
        return "(unset)"
    if isinstance(value, bool):
        return f"`{str(value).lower()}`"
    if isinstance(value, list):
        return "`" + str(value).replace("'", '"') + "`"
    return f"`{value}`"


def main() -> int:
    """Print the Markdown table."""
    print("| Environment variable | Default | Description |")
    print("|---|---|---|")
    for name, field in Settings.model_fields.items():
        description = field.description or ""
        print(f"| `{env_var_name(name, field)}` | {default_text(name)} | {description} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
