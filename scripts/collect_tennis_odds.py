#!/usr/bin/env python3
"""Collecte prospective append-only des moneylines ATP."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.data.tennis_odds_collector import collect_current_atp_odds  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--maximum-sports",
        type=int,
        default=None,
        help="Limite optionnelle de tournois pour maîtriser le quota API.",
    )
    args = parser.parse_args()
    result = collect_current_atp_odds(BASE_DIR, maximum_sports=args.maximum_sports)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
