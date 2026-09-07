"""Closing line value — the only verdict that arrives before you are old.

Everything in this project ran into the same wall: `RAPPORT_RENTABILITE.md`
computes that demonstrating a +0.74% edge needs 35 250 settled bets, roughly
thirty-four years. A strategy cannot be steered on feedback that slow.

Closing line value is the professional answer. It does not measure profit; it
measures whether the price you took was better than the price the market settled
on. That has two properties the profit series does not:

* **It converges in hundreds of bets, not tens of thousands**, because the
  closing price is a far less noisy target than a win or a loss. A bet either
  beat the close or it did not — no coin flip in between.
* **It is the accepted leading indicator.** Consistently beating the close is what
  a genuine edge looks like from the inside; consistently losing to it is what
  every strategy in this repository has done, measured after the fact.

This module records what was taken and, once the market settles, what it settled
at. It does not claim CLV proves profit — a bettor can beat the close and still
lose to commission and limits. It claims something narrower and more useful:
**negative CLV rules an edge out quickly**, and that is the test worth running
before spending a year on a paper trail.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


LEDGER_NAME = "closing_line_log.json"


@dataclass(frozen=True)
class TakenPrice:
    """A price recorded at the moment of the decision."""

    key: str
    sport: str
    event: str
    outcome: str
    bookmaker: str
    price: float
    reference_probability: float
    taken_at_utc: str
    commence_time: str


def log_path(root: Path) -> Path:
    return root / "models" / LEDGER_NAME


def load_log(root: Path) -> list[dict[str, Any]]:
    path = log_path(root)
    if not path.exists():
        return []
    try:
        content = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return content if isinstance(content, list) else []


def save_log(root: Path, entries: list[dict[str, Any]]) -> None:
    path = log_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(entries, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def record_taken(root: Path, taken: TakenPrice) -> tuple[bool, str]:
    """Store a decision. The same outcome is never logged twice."""
    entries = load_log(root)
    identifier = f"{taken.key}|{taken.outcome}"
    if any(entry.get("id") == identifier for entry in entries):
        return False, "déjà enregistré pour cette issue"
    entries.append({
        "id": identifier,
        "sport": taken.sport,
        "event": taken.event,
        "outcome": taken.outcome,
        "bookmaker": taken.bookmaker,
        "taken_price": taken.price,
        "reference_probability": taken.reference_probability,
        "taken_at_utc": taken.taken_at_utc,
        "commence_time": taken.commence_time,
        "closing_price": None,
        "closing_probability": None,
        "closed_at_utc": None,
    })
    save_log(root, entries)
    return True, f"{taken.outcome} à {taken.price:.2f} chez {taken.bookmaker} enregistré"


def settle_with_closing(
    root: Path, observed: dict[str, tuple[float, float]]
) -> tuple[int, str]:
    """Track the last pre-match price, and promote it once the match begins.

    The closing line is the last price available *before* play starts. Reading a
    quote after kick-off would capture an in-play price, which follows the score
    and answers a different question entirely. So each call refreshes a rolling
    pre-match quote, and only the transition to "started" turns it into a close.

    ``observed`` maps an entry id to (price, no-vig probability) seen now.
    """
    entries = load_log(root)
    now = datetime.now(timezone.utc)
    refreshed = 0
    closed = 0
    for entry in entries:
        if entry.get("closing_price") is not None:
            continue
        started = pd.to_datetime(entry.get("commence_time"), utc=True, errors="coerce")
        in_play = pd.notna(started) and started <= now
        pair = observed.get(entry["id"])
        if not in_play:
            if pair is not None:
                entry["latest_prematch_price"] = float(pair[0])
                entry["latest_prematch_probability"] = float(pair[1])
                entry["latest_prematch_at_utc"] = now.isoformat()
                refreshed += 1
            continue
        # The match has begun: whatever was last seen before it is the close.
        last = entry.get("latest_prematch_price")
        if last is None:
            continue
        entry["closing_price"] = float(last)
        entry["closing_probability"] = float(
            entry.get("latest_prematch_probability") or 0.0
        )
        entry["closed_at_utc"] = entry.get("latest_prematch_at_utc") or now.isoformat()
        closed += 1
    if refreshed or closed:
        save_log(root, entries)
    return closed, f"{refreshed} prix rafraîchi(s), {closed} ligne(s) close(s)"


def clv_frame(root: Path) -> pd.DataFrame:
    """Per-bet closing line value, in price and in probability terms."""
    entries = [entry for entry in load_log(root) if entry.get("closing_price")]
    if not entries:
        return pd.DataFrame()
    frame = pd.DataFrame(entries)
    taken = frame["taken_price"].astype(float)
    closing = frame["closing_price"].astype(float)
    # Beating the close means having taken a longer price than it settled at.
    frame["clv_prix"] = taken / closing - 1.0
    frame["a_battu_la_cloture"] = taken > closing
    reference = frame["closing_probability"].astype(float)
    frame["ev_au_prix_de_cloture"] = reference * taken - 1.0
    return frame


def clv_summary(root: Path) -> dict[str, Any]:
    """What the log says so far, and how far it is from a verdict."""
    frame = clv_frame(root)
    pending = len([entry for entry in load_log(root) if not entry.get("closing_price")])
    if frame.empty:
        return {
            "settled": 0,
            "pending": pending,
            "verdict": "pas encore de ligne close",
        }
    values = frame["clv_prix"].to_numpy(dtype=float)
    mean = float(values.mean())
    error = float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else float("nan")
    beat_rate = float(frame["a_battu_la_cloture"].mean())
    # A t of 2 is the usual bar; the sample needed scales with the square of it.
    needed = (
        int(np.ceil((2.0 * values.std(ddof=1) / mean) ** 2))
        if len(values) > 1 and mean > 0 else None
    )
    if len(values) < 30:
        verdict = "échantillon trop petit pour conclure"
    elif mean <= 0:
        verdict = (
            "CLV négatif: les prix pris sont plus courts que la clôture. "
            "Un avantage est écarté, sans attendre des milliers de paris."
        )
    elif error and mean / error > 2.0:
        verdict = "CLV positif et distinguable de zéro"
    else:
        verdict = "CLV positif mais encore indistinguable de zéro"
    return {
        "settled": int(len(values)),
        "pending": pending,
        "clv_moyen": mean,
        "erreur_standard": error,
        "t": mean / error if error else None,
        "taux_battu": beat_rate,
        "paris_requis_estimes": needed,
        "verdict": verdict,
        "note": (
            "Le CLV mesure si le prix pris valait mieux que celui de clôture, pas "
            "un profit. Un CLV négatif écarte un avantage; un CLV positif ne le "
            "prouve pas, la commission et les plafonds restant à payer."
        ),
    }
