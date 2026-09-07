#!/usr/bin/env python3
"""Scan live totals markets for middles, with the same guards as arbitrage.

A middle is the one structure where both legs can win: back Over at a low line
with one operator and Under at a higher line with another, and any result
between the two pays twice. Unlike everything else in this project it needs no
forecast — only two lines, two prices, and the discipline to check that the gap
can actually contain a result.

Reports every guard as its own column rather than filtering silently, so the
count that survives is visible next to the count a naive scan would claim.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.odds_api import (
    MMA_SPORT_KEY,
    TENNIS_SPORT_PREFIX,
    active_sports,
    fetch_market_odds,
)
from src.backtesting.middles import classify, scan

TOTALS_MARKET = "totals"


def _default_sports(root: Path) -> list[str]:
    catalogue = active_sports(root)
    if not catalogue.ok:
        return []
    return [
        sport["key"] for sport in catalogue.events
        if sport.get("active") and (
            str(sport["key"]).startswith(TENNIS_SPORT_PREFIX)
            or sport["key"] == MMA_SPORT_KEY
        )
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--regions", default="eu,uk,us,au")
    parser.add_argument("--sports", nargs="*", default=None)
    parser.add_argument("--market", default=TOTALS_MARKET)
    args = parser.parse_args()
    root = args.project_root.resolve()

    sports = args.sports or _default_sports(root)
    if not sports:
        print("aucun sport actif; catalogue indisponible ou vide")
        return 1

    frames = []
    for sport in sports:
        response = fetch_market_odds(root, sport, args.market, regions=args.regions)
        if not response.ok:
            print(f"{sport}: {response.error}")
            continue
        found = scan(response.events, sport, args.market)
        print(f"{sport}: {len(response.events)} événements, {len(found)} middle(s)")
        if found:
            frames.append(classify(found))

    if not frames:
        print("\naucun middle sur les marchés interrogés")
        return 0

    table = pd.concat(frames, ignore_index=True)
    table = table.sort_values(["exploitable", "largeur"], ascending=[False, False])
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print()
        print(table.to_string(index=False))

    usable = table[table["exploitable"]]
    print(f"\n{len(usable)} middle(s) exploitable(s) sur {len(table)} détecté(s)")
    if not usable.empty:
        # A middle is only worth taking if the result lands inside it more often
        # than the pair's margin costs. That probability is a property of the
        # sport, not of these prices: see scripts/run_middle_study.py.
        print("P(pile) requise par middle exploitable:")
        for row in usable.itertuples(index=False):
            print(f"  {row.événement:44s} {row.middle:>14s}  "
                  f"P > {row.P_requise:.2%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
