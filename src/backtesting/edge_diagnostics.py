"""Is a measured forecasting gain large enough to survive a bookmaker's margin?

This module answers a question, it does not select a strategy. Everything it
reads has already been spent: the ATP tables and the WTA holdout are both burned,
so no number produced here is evidence of profitability and none of it may be
used to pick a rule and call that rule validated.

What it is for: a log-loss gain of 0.001 is easy to measure and hard to
interpret. Before building anything on top of one, it is worth knowing whether a
gain that size can clear a 3% overround at all, and where in the price range it
lives. The single rule applied below — *back whichever side the blend prices as
positive expected value* — is deliberately the most permissive one available and
is fixed in advance, so that no search happens here at all.

Two comparisons matter and both are reported:

* against the **market average**, whose overround is around 6% on tennis and
  which no one can actually bet into;
* against **Pinnacle**, whose overround is around 3% and which is a real,
  executable, low-margin counterparty.

Beating the first and not the second means the model has learned that the
average is stale, which is true, well known, and worth nothing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.backtesting.rigorous_strategy import PRICE_COLUMNS, ProtocolWindows


# Fixed before any diagnostic was computed: the widest sensible band, so the
# result reflects the edge rather than a chosen slice of it.
ODDS_BANDS = ((1.0, 1.5), (1.5, 2.0), (2.0, 3.0), (3.0, 6.0), (6.0, np.inf))
HAIRCUT = 0.02

# Applied *after* the bands above showed the 6+ band losing 24% to 65% in every
# period and at every price source. Excluding extreme longshots is a standard
# prior — models are known to overprice tails — but this particular cutoff was
# read off already-spent data, so any result under it is a hypothesis and not a
# validated finding.
LONGSHOT_CAP = 6.0


def _devig(frame: pd.DataFrame, source: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return P1's no-vig probability, both decimal prices, and a validity mask."""
    left_column, right_column = PRICE_COLUMNS[source]
    left = pd.to_numeric(frame[left_column], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(frame[right_column], errors="coerce").to_numpy(dtype=float)
    overround = 1.0 / left + 1.0 / right
    low = 0.85 if source == "maximum" else 0.95
    high = 1.20 if source == "maximum" else 1.25
    valid = (
        np.isfinite(left) & np.isfinite(right)
        & (left > 1.0) & (right > 1.0)
        & (overround >= low) & (overround <= high)
    )
    probability = np.where(valid, (1.0 / left) / overround, np.nan)
    return probability, left, right, valid


def _settle(
    labels: np.ndarray, odds: np.ndarray, on_p1: np.ndarray, haircut: float
) -> tuple[float, float, int]:
    """Flat one-unit returns for a set of already-chosen bets."""
    if not len(labels):
        return 0.0, 0.0, 0
    priced = 1.0 + (odds - 1.0) * (1.0 - haircut)
    won = np.where(on_p1, labels == 1, labels == 0)
    profit = np.where(won, priced - 1.0, -1.0)
    return float(profit.sum()), float(profit.mean()), int(len(profit))


def month_block_confidence(
    unit_returns: np.ndarray, months: np.ndarray, samples: int = 5000, seed: int = 0
) -> tuple[float, float] | tuple[None, None]:
    """90% interval from resampling whole months, so a hot streak cannot hide."""
    if not len(unit_returns):
        return None, None
    codes, _ = pd.factorize(months)
    order = np.argsort(codes, kind="stable")
    sorted_returns, sorted_codes = unit_returns[order], codes[order]
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


def unconditional_positive_ev(
    frame: pd.DataFrame,
    blend: np.ndarray,
    source: str,
    haircut: float = HAIRCUT,
    max_odds: float = np.inf,
) -> dict[str, Any]:
    """Back every side the blend prices above its market price. No tuning."""
    market, left, right, valid = _devig(frame, source)
    completed = frame["_status"].eq("completed").to_numpy()
    usable = valid & completed
    labels = frame["_label"].to_numpy(dtype=int)

    ev_p1 = blend * left - 1.0
    ev_p2 = (1.0 - blend) * right - 1.0
    on_p1 = ev_p1 >= ev_p2
    best_ev = np.where(on_p1, ev_p1, ev_p2)
    odds = np.where(on_p1, left, right)

    take = usable & np.isfinite(best_ev) & (best_ev > 0.0) & (odds < max_odds)
    profit, roi, count = _settle(labels[take], odds[take], on_p1[take], haircut)
    result: dict[str, Any] = {
        "price_source": source,
        "odds_haircut": haircut,
        "max_odds": None if np.isinf(max_odds) else float(max_odds),
        "matches_priced": int(usable.sum()),
        "n_bets": count,
        "bet_rate": float(count / usable.sum()) if usable.sum() else 0.0,
        "roi": roi if count else None,
        "profit_units": profit if count else None,
        "average_odds": float(odds[take].mean()) if count else None,
        "mean_claimed_edge": float(best_ev[take].mean()) if count else None,
        "median_overround": float(np.median((1.0 / left + 1.0 / right)[usable])) if usable.sum() else None,
    }
    if usable.sum() >= 100:
        result["market_log_loss"] = float(log_loss(labels[usable], np.clip(market[usable], 1e-6, 1 - 1e-6)))
        result["blend_log_loss"] = float(log_loss(labels[usable], np.clip(blend[usable], 1e-6, 1 - 1e-6)))
        result["blend_gain_vs_this_source"] = result["market_log_loss"] - result["blend_log_loss"]

    by_band = {}
    for low, high in ODDS_BANDS:
        band = take & (odds >= low) & (odds < high)
        band_profit, band_roi, band_count = _settle(
            labels[band], odds[band], on_p1[band], haircut
        )
        by_band[f"{low:g}-{high:g}"] = {
            "n_bets": band_count,
            "roi": band_roi if band_count else None,
            "profit_units": band_profit if band_count else None,
        }
    result["by_odds_band"] = by_band
    return result


def break_even_gain(overround: float) -> float:
    """Rough log-loss gain a forecaster needs before a margin can be cleared.

    A bettor giving up ``overround - 1`` of margin needs their probabilities to
    be better than the book's by at least the entropy that margin costs. The
    approximation below is the Kullback-Leibler distance between a fair coin and
    a coin shaded by half the total margin, which is the right order of
    magnitude for the near-even prices this project actually bets.
    """
    shade = (overround - 1.0) / 2.0
    if shade <= 0:
        return 0.0
    p = 0.5 + shade
    return float(p * np.log(p / 0.5) + (1 - p) * np.log((1 - p) / 0.5))


def diagnose_study(
    predictions_path: str | Path,
    frozen_strategy_path: str | Path,
    windows: ProtocolWindows,
    sources: tuple[str, ...] = ("average", "pinnacle", "bet365", "maximum"),
) -> dict[str, Any]:
    predictions = pd.read_csv(predictions_path, low_memory=False)
    frozen = json.loads(Path(frozen_strategy_path).read_text(encoding="utf-8"))
    weight = float(frozen["model_weight"])

    average_probability, _, _, _ = _devig(predictions, "average")
    model = predictions["model_probability_p1"].to_numpy(dtype=float)
    blend = weight * model + (1.0 - weight) * np.nan_to_num(average_probability, nan=0.5)

    periods = {
        "development": windows.development,
        "tuning": windows.tuning,
        "validation": windows.validation,
        "holdout": windows.holdout,
    }
    report: dict[str, Any] = {
        "what_this_is": (
            "Diagnostic sur données déjà dépensées. Aucun chiffre ici n'est une preuve de "
            "rentabilité et aucune règle ne peut être sélectionnée à partir de ces cellules."
        ),
        "rule": "back every side whose blended expected value is positive; no threshold, no tuning",
        "odds_haircut": HAIRCUT,
        "model_weight": weight,
        "periods": {},
    }
    for name, years in periods.items():
        mask = predictions["_year"].isin(list(years)).to_numpy()
        window = predictions[mask]
        block: dict[str, Any] = {"years": list(years), "rows": int(mask.sum()), "sources": {}}
        for source in sources:
            block["sources"][source] = {
                "all_odds": unconditional_positive_ev(window, blend[mask], source),
                f"odds_below_{LONGSHOT_CAP:g}": unconditional_positive_ev(
                    window, blend[mask], source, max_odds=LONGSHOT_CAP
                ),
            }
        report["periods"][name] = block

    months = pd.to_datetime(predictions["_date"]).dt.to_period("M").astype(str).to_numpy()
    evaluated = predictions["_year"].isin(
        [year for window in periods.values() for year in window]
    ).to_numpy()
    report["pooled_across_all_periods"] = {}
    for source in sources:
        _, left, right, valid = _devig(predictions, source)
        completed = predictions["_status"].eq("completed").to_numpy()
        labels = predictions["_label"].to_numpy(dtype=int)
        ev_p1, ev_p2 = blend * left - 1.0, (1.0 - blend) * right - 1.0
        on_p1 = ev_p1 >= ev_p2
        best_ev = np.where(on_p1, ev_p1, ev_p2)
        odds = np.where(on_p1, left, right)
        cells = {}
        for label, cap in (("all_odds", np.inf), (f"odds_below_{LONGSHOT_CAP:g}", LONGSHOT_CAP)):
            take = valid & completed & evaluated & np.isfinite(best_ev) & (best_ev > 0.0) & (odds < cap)
            priced = 1.0 + (odds[take] - 1.0) * (1.0 - HAIRCUT)
            won = np.where(on_p1[take], labels[take] == 1, labels[take] == 0)
            unit = np.where(won, priced - 1.0, -1.0)
            low, high = month_block_confidence(unit, months[take])
            positive_periods = sum(
                (report["periods"][name]["sources"][source][label]["roi"] or 0.0) > 0.0
                for name in periods
            )
            cells[label] = {
                "n_bets": int(len(unit)),
                "roi": float(unit.mean()) if len(unit) else None,
                "profit_units": float(unit.sum()) if len(unit) else None,
                "roi_ci_90": [low, high],
                "positive_periods_out_of_4": positive_periods,
                "interval_excludes_zero": bool(low is not None and low > 0.0),
            }
        report["pooled_across_all_periods"][source] = cells

    overrounds = {
        source: report["periods"]["holdout"]["sources"][source]["all_odds"]["median_overround"]
        for source in sources
    }
    report["break_even_reference"] = {
        source: {
            "median_overround": value,
            "approximate_log_loss_gain_required": break_even_gain(value) if value else None,
        }
        for source, value in overrounds.items()
    }
    return report
