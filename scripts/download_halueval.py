#!/usr/bin/env python
"""Download the HaluEval QA subset into data/halueval/ (gitignored).

Source: https://github.com/RUCAIBox/HaluEval — see README for license and citation.
"""

from __future__ import annotations

import sys

from slm_rag_eval.bench.download import download_dataset


def main() -> int:
    """Fetch the HaluEval QA file; already-cached files are left alone unless --force."""
    force = "--force" in sys.argv[1:]
    for path in download_dataset("halueval", force=force):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
