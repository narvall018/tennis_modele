#!/usr/bin/env python3
"""Download and validate the Football-Data odds tables for 22 European divisions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.football_pipeline import run_football_update


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    progress = None if args.quiet else (lambda message: print(message, flush=True))
    report = run_football_update(args.project_root, progress=progress)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
