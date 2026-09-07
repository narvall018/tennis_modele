#!/usr/bin/env python3
"""Scan live markets for cross-bookmaker arbitrage, with the guards applied.

Reports every opportunity it finds *and* which guard each one fails, so the
count that survives is visible next to the count a naive scan would claim. Each
sport costs API quota, so the sport list is explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.odds_api import MMA_SPORT_KEY, TENNIS_SPORT_PREFIX, active_sports, fetch_h2h_odds
from src.backtesting.arbitrage import classify, scan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--regions", default="eu,uk,us,au")
    parser.add_argument("--sports", nargs="*", default=None,
                        help="clés de sport; par défaut tennis actifs + MMA")
    args = parser.parse_args()
    root = args.project_root.resolve()

    sports = args.sports
    if not sports:
        catalogue = active_sports(root)
        if not catalogue.ok:
            print(f"catalogue indisponible: {catalogue.error}")
            return 1
        sports = [
            sport["key"] for sport in catalogue.events
            if sport.get("active") and (
                str(sport["key"]).startswith(TENNIS_SPORT_PREFIX)
                or sport["key"] == MMA_SPORT_KEY
            )
        ]
    print(f"sports interrogés: {', '.join(sports)}\n")

    opportunities = []
    remaining = None
    for sport in sports:
        response = fetch_h2h_odds(root, sport, regions=args.regions)
        remaining = response.remaining if response.remaining is not None else remaining
        if not response.ok:
            print(f"  {sport}: {response.error}")
            continue
        found = scan(response.events, sport)
        opportunities += found
        print(f"  {sport}: {len(response.events)} marchés, {len(found)} arbitrage(s)")

    frame = classify(opportunities)
    if frame.empty:
        print("\nAucun arbitrage détecté.")
        return 0

    pd.set_option("display.width", 220)
    print("\n" + frame[[
        "sport", "événement", "gain_garanti", "fraîcheur_pire_min",
        "assez_frais", "sans_exchange", "books_accessibles", "exploitable", "books",
    ]].to_string(index=False))

    usable = int(frame["exploitable"].sum())
    print(f"\nbruts: {len(frame)} · exploitables après garde-fous: {usable}")
    if usable == 0:
        failed = {
            "fraîcheur": int((~frame["assez_frais"]).sum()),
            "exchange": int((~frame["sans_exchange"]).sum()),
            "accessibilité": int((~frame["books_accessibles"]).sum()),
        }
        print("écartés par: " + ", ".join(f"{k} ({v})" for k, v in failed.items()))
        print(
            "\nUn arbitrage écarté pour accessibilité n'est pas une erreur de mesure: "
            "il existe, mais chez des opérateurs qu'un compte unique ne peut pas "
            "atteindre. Voir RAPPORT_ARBITRAGE.md."
        )

    output = root / "models" / "arbitrage_scan.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({
            "scanned_at_utc": pd.Timestamp.now("UTC").isoformat(),
            "regions": args.regions,
            "sports": sports,
            "remaining_requests": remaining,
            "opportunities": frame.to_dict("records"),
        }, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    print(f"\nquota restant: {remaining} · rapport: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
