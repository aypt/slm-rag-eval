#!/usr/bin/env bash
# Best-effort weekly quota probe for the runner's gate 0 (Layer 2).
# Contract: print the weekly USED percentage as a bare integer (e.g. "42") and exit 0.
# If usage cannot be determined, print NOTHING — the runner then FAILS OPEN and
# relies on the reactive cooldown (Layer 1), which needs no tooling at all.
#
# There is no official scriptable `codex usage` subcommand today; /status is
# TUI-only. Two community options can feed this script — pick one and wire it in:
#
#   Option A: codex-cli-usage (PyPI) — talks to codex's own app-server for limits.
#     pip install codex-cli-usage
#     Run `codex-cli-usage` once, look at its output/JSON, then edit the OPTION A
#     block below to extract the weekly used-percent field.
#
#   Option B: codex-ratelimit (github.com/xiangz19/codex-ratelimit) — parses the
#     freshest rate-limit snapshot out of ~/.codex/sessions/*.jsonl, has --json.
#
# Until you wire one in, this script prints nothing (fail open) — that is safe,
# just not preemptive.
set -euo pipefail

# --- OPTION A (uncomment and adapt after inspecting the tool's JSON) ---------
# command -v codex-cli-usage >/dev/null 2>&1 || exit 0
# codex-cli-usage --json 2>/dev/null \
#   | python3 -c 'import sys,json;d=json.load(sys.stdin);print(int(d["weekly"]["used_percent"]))' \
#   2>/dev/null || true

# --- OPTION B (uncomment and adapt) ------------------------------------------
# python3 /path/to/codex-ratelimit/ratelimit_checker.py --json 2>/dev/null \
#   | python3 -c 'import sys,json;d=json.load(sys.stdin);print(int(d["weekly"]["used_percent"]))' \
#   2>/dev/null || true

exit 0
