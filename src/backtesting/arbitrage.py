"""Simultaneous cross-bookmaker arbitrage, with the guards that decide if it is real.

Every other avenue in this repository tried to forecast better than a price and
failed. Arbitrage asks nothing of a forecast: if two books disagree enough,
backing both sides returns a profit whatever happens. That is why it deserves a
look the models did not earn.

It is also the classic trap, and a naive scan finds "opportunities" that cannot
be taken. Four guards separate the two:

* **Staleness.** The API returns each book's own ``last_update``. A quote thirteen
  minutes old against one refreshed thirty seconds ago is not a simultaneous
  pair; the apparent gap is simply one book not having moved yet.
* **Exchanges.** A price on Betfair, Smarkets or Matchbook is an *offer* of
  unknown size. A 10.00 that exists for £5 is not a market you can arbitrage into
  at any meaningful stake, and exchanges also charge commission on winnings.
* **Reachability.** A margin at a book one cannot open an account with is not a
  margin. The operator list is explicit rather than assumed.
* **Persistence.** An arbitrage that vanishes before both legs are placed never
  existed. Only a scan repeated over time can measure that, so the scanner
  records what it saw and when.

Nothing here places a bet. It measures whether the opportunity survives its own
constraints, which is the question the historical data could never answer because
its prices were not simultaneous.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable

import pandas as pd


# Exchanges quote offers of unknown depth and take commission on winnings; a
# book quotes a price it must honour to its posted limit. They are not
# interchangeable and are separated rather than filtered by default.
EXCHANGES = frozenset({
    "betfair_ex_uk", "betfair_ex_eu", "betfair_ex_au", "smarkets", "matchbook",
    "betdaq",
})

# Operators a French-resident account can realistically use. Everything else is
# reported but flagged, because a margin at an unreachable book is not a margin.
REACHABLE_FROM_FRANCE = frozenset({
    "onexbet", "unibet", "unibet_eu", "unibet_fr", "winamax", "winamax_fr",
    "betclic", "pmu", "parionssport", "zebet", "netbet", "bwin", "pinnacle",
    "marathonbet", "nordicbet", "betsson", "coolbet", "everygame", "betanysports",
})

MAX_STALENESS_MINUTES = 5.0


@dataclass
class ArbitrageOpportunity:
    sport: str
    event: str
    commence_time: str
    legs: list[dict[str, Any]] = field(default_factory=list)
    overround: float = 1.0
    seen_at_utc: str = ""

    @property
    def guaranteed_return(self) -> float:
        """Profit per unit staked when both legs are placed proportionally."""
        return (1.0 / self.overround) - 1.0 if self.overround > 0 else 0.0

    @property
    def worst_staleness(self) -> float:
        return max((leg["staleness_minutes"] for leg in self.legs), default=float("nan"))

    @property
    def uses_exchange(self) -> bool:
        return any(leg["bookmaker"] in EXCHANGES for leg in self.legs)

    @property
    def all_reachable(self) -> bool:
        return all(leg["bookmaker"] in REACHABLE_FROM_FRANCE for leg in self.legs)

    def stakes(self, bankroll: float) -> dict[str, float]:
        """How much on each leg so every outcome returns the same amount."""
        return {
            leg["outcome"]: round(bankroll / leg["price"] / self.overround, 2)
            for leg in self.legs
        }

    def key(self) -> str:
        return f"{self.sport}|{self.event}"


def _best_quotes(event: dict[str, Any], now: pd.Timestamp) -> dict[str, dict[str, Any]]:
    """Best price per outcome, remembering which book and how fresh."""
    best: dict[str, dict[str, Any]] = {}
    for book in event.get("bookmakers") or []:
        updated = pd.to_datetime(book.get("last_update"), utc=True, errors="coerce")
        staleness = (
            (now - updated).total_seconds() / 60.0 if pd.notna(updated) else float("inf")
        )
        for market in book.get("markets") or []:
            if market.get("key") != "h2h":
                continue
            for outcome in market.get("outcomes") or []:
                name = str(outcome.get("name") or "").strip()
                price = outcome.get("price")
                if not name or name.lower() in {"draw", "tie"}:
                    continue
                if not isinstance(price, (int, float)) or price <= 1.0:
                    continue
                current = best.get(name)
                if current is None or price > current["price"]:
                    best[name] = {
                        "outcome": name,
                        "price": float(price),
                        "bookmaker": str(book.get("key") or "?"),
                        "staleness_minutes": staleness,
                    }
    return best


def scan_event(event: dict[str, Any], sport: str, now: pd.Timestamp
               ) -> ArbitrageOpportunity | None:
    """Return the opportunity if backing every outcome guarantees a profit."""
    best = _best_quotes(event, now)
    if len(best) != 2:
        return None
    overround = sum(1.0 / leg["price"] for leg in best.values())
    if overround >= 1.0:
        return None
    return ArbitrageOpportunity(
        sport=sport,
        event=f"{event.get('home_team')} vs {event.get('away_team')}",
        commence_time=str(event.get("commence_time", "")),
        legs=list(best.values()),
        overround=overround,
        seen_at_utc=datetime.now(timezone.utc).isoformat(),
    )


def scan(events: Iterable[dict[str, Any]], sport: str,
         now: pd.Timestamp | None = None) -> list[ArbitrageOpportunity]:
    moment = now or pd.Timestamp.now("UTC")
    found = [scan_event(event, sport, moment) for event in events]
    return [item for item in found if item is not None]


def classify(opportunities: list[ArbitrageOpportunity],
             max_staleness: float = MAX_STALENESS_MINUTES) -> pd.DataFrame:
    """Table of what was found, with each guard as its own column.

    The guards are reported rather than applied, so the count that survives all
    of them is visible next to the count a naive scan would have claimed.
    """
    rows = []
    for item in opportunities:
        rows.append({
            "sport": item.sport,
            "événement": item.event,
            "gain_garanti": item.guaranteed_return,
            "overround": item.overround,
            "fraîcheur_pire_min": item.worst_staleness,
            "assez_frais": item.worst_staleness <= max_staleness,
            "sans_exchange": not item.uses_exchange,
            "books_accessibles": item.all_reachable,
            "books": " / ".join(leg["bookmaker"] for leg in item.legs),
            "cotes": " / ".join(f"{leg['price']:.2f}" for leg in item.legs),
            "vu_a": item.seen_at_utc,
        })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["exploitable"] = (
        frame["assez_frais"] & frame["sans_exchange"] & frame["books_accessibles"]
    )
    return frame.sort_values("gain_garanti", ascending=False).reset_index(drop=True)
