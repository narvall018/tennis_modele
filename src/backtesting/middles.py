"""Middles: the one structure where both sides of a bet can win.

Arbitrage exploits two books disagreeing on a *price*. A middle exploits two
books disagreeing on the *line*. Back Over 1.5 rounds at one operator and Under
2.5 at another, and a fight ending in round two wins both.

That makes it a different animal from everything else here:

* the downside is bounded — outside the middle, one side wins and one loses, so
  the loss is only the pair's combined margin;
* the upside is a double win, which no single-line market offers;
* like arbitrage, it needs no forecast at all. Only two lines and two prices.

The same guards as the arbitrage scanner apply, for the same reasons: a stale
quote is not a simultaneous one, an exchange price is an offer of unknown depth,
and a line at an unreachable operator is not a line. A middle also has one guard
of its own — the gap must be wide enough to contain an actual outcome. Over 2.5
against Under 2.75 can never both win, whatever the prices say.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import numpy as np
import pandas as pd

from src.backtesting.arbitrage import EXCHANGES, REACHABLE_FROM_FRANCE


MAX_STALENESS_MINUTES = 5.0


@dataclass
class Middle:
    sport: str
    event: str
    commence_time: str
    market: str
    over_line: float
    over_price: float
    over_book: str
    over_staleness: float
    under_line: float
    under_price: float
    under_book: str
    under_staleness: float
    seen_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def width(self) -> float:
        """How much room the middle leaves for a winning outcome."""
        return self.under_line - self.over_line

    @property
    def cost_if_missed(self) -> float:
        """Loss per unit staked when the result falls outside the middle.

        Stakes are balanced so either single win returns the same. Outside the
        middle one leg pays and the other does not, so the loss is whatever the
        pair's combined margin costs — usually a fraction of a percent, not the
        whole stake.
        """
        total = 1.0 / self.over_price + 1.0 / self.under_price
        return (1.0 / total) - 1.0

    @property
    def gain_if_hit(self) -> float:
        """Profit per unit staked when the result lands inside the middle.

        With stakes balanced so either single win returns the same amount R,
        both legs paying returns 2R — the whole point of the structure.
        """
        single = 1.0 / (1.0 / self.over_price + 1.0 / self.under_price)
        return 2.0 * single - 1.0

    @property
    def break_even_probability(self) -> float:
        """How often the middle must land for the pair to be worth taking."""
        cost = -self.cost_if_missed
        total = self.gain_if_hit + cost
        return cost / total if total > 0 else 1.0

    @property
    def worst_staleness(self) -> float:
        return max(self.over_staleness, self.under_staleness)

    @property
    def uses_exchange(self) -> bool:
        return self.over_book in EXCHANGES or self.under_book in EXCHANGES

    @property
    def all_reachable(self) -> bool:
        return (
            self.over_book in REACHABLE_FROM_FRANCE
            and self.under_book in REACHABLE_FROM_FRANCE
        )

    @property
    def contains_an_outcome(self) -> bool:
        """A middle only pays if a whole result can land strictly inside it.

        Totals are quoted on half-point lines, so the interval must span at
        least one integer. Over 2.5 / Under 2.75 is arithmetically a gap and
        practically nothing.
        """
        lower = np.floor(self.over_line) + 1
        return bool(self.width > 0 and lower < self.under_line and lower > self.over_line)


def _quotes(event: dict[str, Any], market_key: str, now: pd.Timestamp) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for book in event.get("bookmakers") or []:
        updated = pd.to_datetime(book.get("last_update"), utc=True, errors="coerce")
        staleness = (
            (now - updated).total_seconds() / 60.0 if pd.notna(updated) else float("inf")
        )
        for market in book.get("markets") or []:
            if market.get("key") != market_key:
                continue
            for outcome in market.get("outcomes") or []:
                name = str(outcome.get("name") or "").strip().lower()
                point = outcome.get("point")
                price = outcome.get("price")
                if name not in {"over", "under"}:
                    continue
                if not isinstance(point, (int, float)) or not isinstance(price, (int, float)):
                    continue
                if price <= 1.0:
                    continue
                rows.append({
                    "book": str(book.get("key") or "?"), "side": name,
                    "line": float(point), "price": float(price),
                    "staleness": staleness,
                })
    return rows


def scan_event(event: dict[str, Any], sport: str, market_key: str,
               now: pd.Timestamp) -> list[Middle]:
    """Every over/under pair whose lines leave room for both to win."""
    rows = _quotes(event, market_key, now)
    overs = [row for row in rows if row["side"] == "over"]
    unders = [row for row in rows if row["side"] == "under"]
    found: list[Middle] = []
    for over in overs:
        for under in unders:
            if under["line"] <= over["line"]:
                continue
            if over["book"] == under["book"]:
                continue
            middle = Middle(
                sport=sport,
                event=f"{event.get('home_team')} vs {event.get('away_team')}",
                commence_time=str(event.get("commence_time", "")),
                market=market_key,
                over_line=over["line"], over_price=over["price"],
                over_book=over["book"], over_staleness=over["staleness"],
                under_line=under["line"], under_price=under["price"],
                under_book=under["book"], under_staleness=under["staleness"],
            )
            if middle.contains_an_outcome:
                found.append(middle)
    return found


def scan(events: Iterable[dict[str, Any]], sport: str, market_key: str,
         now: pd.Timestamp | None = None) -> list[Middle]:
    moment = now or pd.Timestamp.now("UTC")
    found: list[Middle] = []
    for event in events:
        found += scan_event(event, sport, market_key, moment)
    # One middle per event: the widest, then the cheapest to miss.
    best: dict[str, Middle] = {}
    for middle in found:
        current = best.get(middle.event)
        if current is None or (middle.width, middle.cost_if_missed) > (
            current.width, current.cost_if_missed
        ):
            best[middle.event] = middle
    return sorted(best.values(), key=lambda item: (-item.width, -item.cost_if_missed))


def classify(middles: list[Middle],
             max_staleness: float = MAX_STALENESS_MINUTES) -> pd.DataFrame:
    """Report each guard as its own column rather than filtering silently."""
    rows = []
    for middle in middles:
        rows.append({
            "sport": middle.sport,
            "événement": middle.event,
            "marché": middle.market,
            "middle": f"{middle.over_line:g} – {middle.under_line:g}",
            "largeur": middle.width,
            "coût_si_raté": middle.cost_if_missed,
            "gain_si_touché": middle.gain_if_hit,
            "P_requise": middle.break_even_probability,
            "over": f"{middle.over_price:.2f} @ {middle.over_book}",
            "under": f"{middle.under_price:.2f} @ {middle.under_book}",
            "fraîcheur_pire_min": middle.worst_staleness,
            "assez_frais": middle.worst_staleness <= max_staleness,
            "sans_exchange": not middle.uses_exchange,
            "books_accessibles": middle.all_reachable,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["exploitable"] = (
        frame["assez_frais"] & frame["sans_exchange"] & frame["books_accessibles"]
    )
    return frame.sort_values("largeur", ascending=False).reset_index(drop=True)
