"""Bet the soft book when it disagrees with the sharp one. No sports model.

Every study in this repository has tried to forecast better than the price, and
every one has failed at the same step: the price already knows. This module
stops fighting the market and uses it instead. Pinnacle runs on low margin and
high limits and is treated here as the best available estimate of the true
probability; a bet is taken only where a softer book prices the same match
noticeably away from it.

There is no model to overfit. The only free parameter is the edge threshold, and
every threshold tried is published rather than the best one.

Two things must be said before any number is read:

* **The prices are not proven simultaneous.** Tennis-Data publishes one row per
  match with each book's closing price, without timestamps. If the soft price is
  captured earlier than Pinnacle's, part of any apparent edge is a stale quote
  rather than a real disagreement, and it would not have been available to bet.
* **``maximum`` is not a book.** It is the best price found anywhere, so betting
  it whenever it beats Pinnacle is close to circular; it is reported for
  contrast, never as an executable result.

A falsification arm is computed alongside: betting the soft book on the side
where it is *worse* than Pinnacle. If the mechanism is real, that arm must lose
clearly more than the main one. If both behave alike, the signal is noise.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


SHARP_BOOK = ("Pinnacle_1", "Pinnacle_2")
SOFT_BOOKS = {
    "bet365": ("B365_1", "B365_2"),
    "market_average": ("Avg_1", "Avg_2"),
    "market_maximum": ("Max_1", "Max_2"),
}
EDGE_THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.05)
HAIRCUT = 0.02


def _valid_pair(frame: pd.DataFrame, columns: tuple[str, str], sharp: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left = pd.to_numeric(frame[columns[0]], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(frame[columns[1]], errors="coerce").to_numpy(dtype=float)
    overround = 1.0 / left + 1.0 / right
    # The sharp book must look like a coherent two-sided market; a soft price is
    # only required to be a usable quote, since its overround is the point.
    if sharp:
        valid = (overround >= 0.98) & (overround <= 1.12)
    else:
        valid = (overround >= 0.85) & (overround <= 1.30)
    valid &= np.isfinite(left) & np.isfinite(right) & (left > 1.0) & (right > 1.0)
    return left, right, valid


def _month_block_interval(returns: np.ndarray, months: np.ndarray, samples: int = 5000,
                          seed: int = 0) -> tuple[float | None, float | None]:
    if not len(returns):
        return None, None
    codes, _ = pd.factorize(months)
    order = np.argsort(codes, kind="stable")
    sorted_returns, sorted_codes = returns[order], codes[order]
    block_count = int(codes.max()) + 1
    starts = np.searchsorted(sorted_codes, np.arange(block_count))
    ends = np.append(starts[1:], len(sorted_codes))
    sums = np.add.reduceat(sorted_returns, starts)
    counts = ends - starts
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, block_count, size=(samples, block_count))
    resampled = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(resampled, [5, 95])
    return float(low), float(high)


def evaluate_book(
    frame: pd.DataFrame,
    soft_columns: tuple[str, str],
    threshold: float,
    haircut: float = HAIRCUT,
    contrarian: bool = False,
) -> dict[str, Any]:
    """One book, one threshold. ``contrarian`` inverts the signal as a control."""
    sharp_left, sharp_right, sharp_ok = _valid_pair(frame, SHARP_BOOK, sharp=True)
    soft_left, soft_right, soft_ok = _valid_pair(frame, soft_columns, sharp=False)
    overround = 1.0 / sharp_left + 1.0 / sharp_right
    truth_p1 = (1.0 / sharp_left) / overround
    completed = frame["Status"].astype(str).str.lower().eq("completed").to_numpy()
    usable = sharp_ok & soft_ok & completed

    edge_p1 = truth_p1 * soft_left - 1.0
    edge_p2 = (1.0 - truth_p1) * soft_right - 1.0
    if contrarian:
        edge_p1, edge_p2 = -edge_p1, -edge_p2
    on_p1 = edge_p1 >= edge_p2
    best_edge = np.where(on_p1, edge_p1, edge_p2)
    odds = np.where(on_p1, soft_left, soft_right)

    take = usable & np.isfinite(best_edge) & (best_edge > threshold)
    labels = frame["_label"].to_numpy(dtype=int)
    won = np.where(on_p1[take], labels[take] == 1, labels[take] == 0)
    priced = 1.0 + (odds[take] - 1.0) * (1.0 - haircut)
    unit = np.where(won, priced - 1.0, -1.0)

    months = frame["_month"].to_numpy()[take]
    low, high = _month_block_interval(unit, months)
    years = frame["_year"].to_numpy()[take]
    per_year = {
        str(int(year)): float(unit[years == year].mean()) for year in np.unique(years)
    } if len(unit) else {}
    return {
        "threshold": threshold,
        "eligible_matches": int(usable.sum()),
        "n_bets": int(len(unit)),
        "bet_rate": float(len(unit) / usable.sum()) if usable.sum() else 0.0,
        "roi": float(unit.mean()) if len(unit) else None,
        "profit_units": float(unit.sum()) if len(unit) else None,
        "hit_rate": float(won.mean()) if len(unit) else None,
        "average_odds": float(odds[take].mean()) if len(unit) else None,
        "mean_claimed_edge": float(best_edge[take].mean()) if len(unit) else None,
        "roi_ci_90": [low, high],
        "interval_excludes_zero": bool(low is not None and low > 0.0),
        "positive_years": int(sum(value > 0 for value in per_year.values())),
        "total_years": len(per_year),
        "roi_by_year": per_year,
    }


def prepare(path: str, tour: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    frame["_date"] = pd.to_datetime(frame["Date"], errors="coerce")
    frame = frame.dropna(subset=["_date"]).copy()
    frame["_year"] = frame["_date"].dt.year
    frame["_month"] = frame["_date"].dt.to_period("M").astype(str)
    frame["_label"] = (frame["Winner"].astype(str) == frame["Player_1"].astype(str)).astype(int)
    frame["_tour"] = tour
    return frame


def simultaneity_audit(frame: pd.DataFrame) -> dict[str, Any]:
    """How often each price source implies a risk-free arbitrage.

    A two-sided quote with an overround below 1.0 is free money, so it cannot
    persist in a liquid market. A source showing them in bulk is not quoting one
    moment in time — it is the best price seen across books at different moments,
    and it could never have been bet as a pair. This is the guard that decides
    whether a source is executable at all.
    """
    audit: dict[str, Any] = {}
    sources = {"pinnacle": SHARP_BOOK, **SOFT_BOOKS}
    for name, columns in sources.items():
        left = pd.to_numeric(frame[columns[0]], errors="coerce")
        right = pd.to_numeric(frame[columns[1]], errors="coerce")
        quoted = (left > 1.0) & (right > 1.0)
        overround = (1.0 / left + 1.0 / right)[quoted]
        arbitrage_rate = float((overround < 1.0).mean()) if len(overround) else 0.0
        audit[name] = {
            "quoted_matches": int(quoted.sum()),
            "median_overround": float(overround.median()) if len(overround) else None,
            "implied_arbitrage_rate": arbitrage_rate,
            "deep_arbitrage_rate": float((overround < 0.98).mean()) if len(overround) else 0.0,
            # A real book sits far below 1%. Anything above is a composite.
            "executable_as_a_pair": bool(arbitrage_rate < 0.01),
        }
    return audit


def analyse(frame: pd.DataFrame) -> dict[str, Any]:
    report: dict[str, Any] = {
        "matches": int(len(frame)),
        "years": [int(frame["_year"].min()), int(frame["_year"].max())],
        "sharp_book": "Pinnacle",
        "simultaneity_audit": simultaneity_audit(frame),
        "books": {},
    }
    for name, columns in SOFT_BOOKS.items():
        book: dict[str, Any] = {
            "executable_as_a_pair": report["simultaneity_audit"][name]["executable_as_a_pair"],
            "thresholds": {},
            "falsification": {},
        }
        for threshold in EDGE_THRESHOLDS:
            book["thresholds"][f"{threshold:.2f}"] = evaluate_book(frame, columns, threshold)
            book["falsification"][f"{threshold:.2f}"] = evaluate_book(
                frame, columns, threshold, contrarian=True
            )
        report["books"][name] = book
    return report
