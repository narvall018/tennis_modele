#!/usr/bin/env python3
"""Record prices now, close them just before kick-off, report the CLV.

Two modes, because closing line value needs two moments:

* ``record`` stores the best accessible price for matches not yet started;
* ``close`` attaches the last pre-match price to entries whose match is about to
  begin, and only to those — once a match is in play the quote follows the score
  and is no longer a closing price.

``report`` prints what the log says so far and how far it is from a verdict.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.app.odds_api import MMA_SPORT_KEY, TENNIS_SPORT_PREFIX, active_sports, fetch_h2h_odds
from src.backtesting.arbitrage import EXCHANGES, REACHABLE_FROM_FRANCE
from src.backtesting.closing_line import (
    TakenPrice,
    clv_frame,
    clv_summary,
    load_log,
    record_taken,
    settle_with_closing,
)

MAX_STALENESS_MINUTES = 10.0


def _live_sports(root: Path) -> list[str]:
    catalogue = active_sports(root)
    if not catalogue.ok:
        return []
    return [
        sport["key"] for sport in catalogue.events
        if sport.get("active") and (
            str(sport["key"]).startswith(TENNIS_SPORT_PREFIX) or sport["key"] == MMA_SPORT_KEY
        )
    ]


def _quotes(event: dict, now: pd.Timestamp, accessible_only: bool) -> pd.DataFrame:
    rows = []
    for book in event.get("bookmakers") or []:
        key = str(book.get("key") or "")
        if key in EXCHANGES:
            continue
        if accessible_only and key not in REACHABLE_FROM_FRANCE:
            continue
        updated = pd.to_datetime(book.get("last_update"), utc=True, errors="coerce")
        stale = (now - updated).total_seconds() / 60.0 if pd.notna(updated) else 1e9
        if stale > MAX_STALENESS_MINUTES:
            continue
        for market in book.get("markets") or []:
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes") or []:
                name = str(outcome.get("name") or "").strip()
                price = outcome.get("price")
                if name.lower() in {"draw", "tie"}:
                    continue
                if isinstance(price, (int, float)) and price > 1.0:
                    rows.append({"book": key, "outcome": name, "price": float(price)})
    return pd.DataFrame(rows)


def _event_key(sport: str, event: dict) -> str:
    return f"{sport}|{event.get('home_team')}|{event.get('away_team')}"


def record(root: Path, accessible_only: bool) -> int:
    now = pd.Timestamp.now("UTC")
    stored = 0
    for sport in _live_sports(root):
        response = fetch_h2h_odds(root, sport, regions="eu,uk")
        if not response.ok:
            continue
        for event in response.events:
            starts = pd.to_datetime(event.get("commence_time"), utc=True, errors="coerce")
            if pd.isna(starts) or starts <= now:
                continue
            frame = _quotes(event, now, accessible_only)
            if frame.empty:
                continue
            best = frame.loc[frame.groupby("outcome")["price"].idxmax()]
            reference = 1.0 / best.set_index("outcome")["price"]
            reference = reference / reference.sum()
            for row in best.itertuples(index=False):
                ok, message = record_taken(root, TakenPrice(
                    key=_event_key(sport, event), sport=sport,
                    event=f"{event.get('home_team')} vs {event.get('away_team')}",
                    outcome=row.outcome, bookmaker=row.book, price=row.price,
                    reference_probability=float(reference[row.outcome]),
                    taken_at_utc=now.isoformat(),
                    commence_time=str(event.get("commence_time", "")),
                ))
                stored += int(ok)
                print(f"  {'+' if ok else '·'} {row.outcome:26s} {row.price:6.2f} "
                      f"{row.book:14s} {message if not ok else ''}")
    return stored


# The closing reference is one sharp book, not the best of a pool. Comparing a
# price taken among accessible operators against the best of every operator
# would make the drift positive by construction: the closing pool would simply
# be larger. Pinnacle is the reference this project has treated as truth
# throughout.
REFERENCE_BOOK = "pinnacle"


def _reference_quote(event: dict, now: pd.Timestamp) -> pd.DataFrame:
    """Pinnacle's two-way quote, or the market median when it is absent."""
    frame = _quotes(event, now, accessible_only=False)
    if frame.empty:
        return frame
    sharp = frame[frame["book"] == REFERENCE_BOOK]
    if len(sharp) == 2:
        return sharp
    median = frame.groupby("outcome", as_index=False)["price"].median()
    median["book"] = "consensus"
    return median if len(median) == 2 else frame.iloc[0:0]


def close(root: Path) -> int:
    """Refresh the pre-match reference price for everything still pending."""
    now = pd.Timestamp.now("UTC")
    pending = [entry for entry in load_log(root) if not entry.get("closing_price")]
    if not pending:
        print("aucune ligne en attente")
        return 0
    sports = sorted({entry["sport"] for entry in pending})
    closing: dict[str, tuple[float, float]] = {}
    for sport in sports:
        response = fetch_h2h_odds(root, sport, regions="eu,uk")
        if not response.ok:
            continue
        for event in response.events:
            reference = _reference_quote(event, now)
            if len(reference) != 2:
                continue
            inverse = 1.0 / reference.set_index("outcome")["price"]
            probability = inverse / inverse.sum()
            key = _event_key(sport, event)
            for row in reference.itertuples(index=False):
                closing[f"{key}|{row.outcome}"] = (
                    float(row.price), float(probability[row.outcome])
                )
    updated, message = settle_with_closing(root, closing)
    print(message)
    return updated


def report(root: Path) -> None:
    summary = clv_summary(root)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))
    frame = clv_frame(root)
    if not frame.empty:
        print()
        print(frame[[
            "event", "outcome", "bookmaker", "taken_price", "closing_price",
            "clv_prix", "a_battu_la_cloture",
        ]].to_string(index=False))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["record", "close", "report"])
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--all-books", action="store_true",
                        help="ne pas restreindre aux opérateurs accessibles")
    args = parser.parse_args()
    root = args.project_root.resolve()

    if args.action == "record":
        print(f"enregistrement à {datetime.now(timezone.utc):%H:%M:%S} UTC")
        print(f"{record(root, not args.all_books)} nouvelle(s) ligne(s)")
    elif args.action == "close":
        close(root)
    report(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
