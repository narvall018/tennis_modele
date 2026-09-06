"""Opening against closing, and three markets of very different margin.

Two questions this project could never ask before:

1. **Does the price move, and does the move carry information?** With an opening
   and a closing quote for the same match, the drift between them is measurable.
   If closing forecasts better than opening, information arrives during the
   window — which is the premise of every "bet early" argument.
2. **Is the opening price soft enough to bet?** That the closing price is sharper
   does not mean the opening price is wrong in a way anyone can use. The test is
   whether the same model-free selection returns more at opening than at closing.

Alongside, the same calibration audit used on tennis runs on three markets whose
margins differ by a factor of three — 1X2, over/under 2.5 goals, Asian handicap.
Tennis died because a real 1.65-point bias could not clear a 6.7% margin. An
Asian handicap costs about 2%, so a bias of that size would survive there. That
is the entire reason football is worth a look.

Every rule here is model-free and fixed in advance. All cells are reported, never
the best, and a bias must appear in several countries before it counts —
the cross-league check plays the role the ATP/WTA check played in tennis.
"""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.data.football_pipeline import HANDICAP_LINES, PRICE_GROUPS


HAIRCUT = 0.02
PROBABILITY_EDGES = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75, 1.0)
MIN_CELL = 300


def devig(frame: pd.DataFrame, columns: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    """No-vig probabilities for an n-way market, plus a validity mask.

    A market is only usable when every leg is quoted, so a partial row cannot
    silently produce a probability that does not sum to one.
    """
    columns = list(columns)
    prices = frame[columns].to_numpy(dtype=float)
    valid = np.isfinite(prices).all(axis=1) & (prices > 1.0).all(axis=1)
    inverse = np.where(valid[:, None], 1.0 / np.where(prices > 0, prices, np.nan), np.nan)
    total = inverse.sum(axis=1)
    # A quoted market pays a margin; anything below 1 is a composite of moments.
    valid &= np.isfinite(total) & (total >= 0.98) & (total <= 1.35)
    probabilities = np.where(valid[:, None], inverse / total[:, None], np.nan)
    return probabilities, valid


def outcome_matrix(frame: pd.DataFrame, market: str) -> np.ndarray | None:
    """Which leg won, as a boolean column per leg, in the market's own order."""
    if market == "1x2":
        result = frame["result"].to_numpy()
        return np.column_stack([result == "H", result == "D", result == "A"])
    if market == "over_under_25":
        total = frame["total_goals"].to_numpy(dtype=float)
        return np.column_stack([total > 2.5, total < 2.5])
    return None


def _settle(won: np.ndarray, odds: np.ndarray, haircut: float = HAIRCUT) -> np.ndarray:
    priced = 1.0 + (odds - 1.0) * (1.0 - haircut)
    return np.where(won, priced - 1.0, -1.0)


def _interval(returns: np.ndarray, blocks: np.ndarray, samples: int = 3000,
              seed: int = 0) -> tuple[float | None, float | None]:
    """Bootstrap over whole months, so a good run cannot masquerade as an edge."""
    if not len(returns):
        return None, None
    codes, _ = pd.factorize(blocks)
    order = np.argsort(codes, kind="stable")
    sorted_returns, sorted_codes = returns[order], codes[order]
    count = int(codes.max()) + 1
    starts = np.searchsorted(sorted_codes, np.arange(count))
    ends = np.append(starts[1:], len(sorted_codes))
    sums = np.add.reduceat(sorted_returns, starts)
    sizes = ends - starts
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, count, size=(samples, count))
    resampled = sums[picks].sum(axis=1) / sizes[picks].sum(axis=1)
    low, high = np.percentile(resampled, [5, 95])
    return float(low), float(high)


def _cell(won: np.ndarray, odds: np.ndarray, blocks: np.ndarray,
          probability: np.ndarray | None = None,
          haircut: float = HAIRCUT) -> dict[str, Any]:
    if not len(won):
        return {"n": 0}
    unit = _settle(won, odds, haircut)
    low, high = _interval(unit, blocks)
    cell = {
        "n": int(len(unit)),
        "roi": float(unit.mean()),
        "roi_ci_90": [low, high],
        "hit_rate": float(won.mean()),
        "average_odds": float(odds.mean()),
        "profitable": bool(low is not None and low > 0.0),
        "significant": bool(low is not None and (low > 0.0 or high < 0.0)),
    }
    if probability is not None:
        cell["market_probability"] = float(np.nanmean(probability))
        cell["calibration_gap"] = float(won.mean() - np.nanmean(probability))
    return cell


def timing_comparison(frame: pd.DataFrame, market: str, book: str) -> dict[str, Any]:
    """Opening against closing for the same matches and the same selection.

    Restricting to matches that carry both quotes is what makes the comparison
    fair; anything else would compare two different populations.
    """
    open_columns = PRICE_GROUPS[market].get(f"{book}_open")
    close_columns = PRICE_GROUPS[market].get(f"{book}_close")
    outcomes = outcome_matrix(frame, market)
    if not open_columns or not close_columns or outcomes is None:
        return {"available": False}

    open_probability, open_ok = devig(frame, open_columns)
    close_probability, close_ok = devig(frame, close_columns)
    usable = open_ok & close_ok
    if usable.sum() < MIN_CELL:
        return {"available": False, "n": int(usable.sum())}

    blocks = pd.to_datetime(frame["match_date"]).dt.to_period("M").astype(str).to_numpy()[usable]
    open_probability = open_probability[usable]
    close_probability = close_probability[usable]
    outcomes = outcomes[usable]
    open_odds = frame[list(open_columns)].to_numpy(dtype=float)[usable]
    close_odds = frame[list(close_columns)].to_numpy(dtype=float)[usable]
    winner = outcomes.argmax(axis=1)

    report: dict[str, Any] = {
        "available": True,
        "matches": int(usable.sum()),
        "open_log_loss": float(log_loss(winner, open_probability, labels=list(range(outcomes.shape[1])))),
        "close_log_loss": float(log_loss(winner, close_probability, labels=list(range(outcomes.shape[1])))),
        "open_overround": float(np.nanmedian((1.0 / open_odds).sum(axis=1))),
        "close_overround": float(np.nanmedian((1.0 / close_odds).sum(axis=1))),
        "mean_absolute_drift": float(np.nanmean(np.abs(close_probability - open_probability))),
        "legs": {},
    }
    report["closing_beats_opening_by"] = report["open_log_loss"] - report["close_log_loss"]

    # Model-free selection: back one leg every time, first at opening then at
    # closing. Any difference is the value of the timing alone.
    for leg in range(outcomes.shape[1]):
        won = outcomes[:, leg]
        report["legs"][f"leg_{leg}"] = {
            "at_open": _cell(won, open_odds[:, leg], blocks, open_probability[:, leg]),
            "at_close": _cell(won, close_odds[:, leg], blocks, close_probability[:, leg]),
        }
        report["legs"][f"leg_{leg}"]["open_minus_close_roi"] = (
            report["legs"][f"leg_{leg}"]["at_open"]["roi"]
            - report["legs"][f"leg_{leg}"]["at_close"]["roi"]
        )
    return report


def calibration_audit(frame: pd.DataFrame, market: str, book_timing: str,
                      haircut: float = HAIRCUT) -> dict[str, Any]:
    """Devigged price against realised frequency, by probability band.

    ``haircut`` is a bookmaker's execution slippage but an exchange's commission,
    and the distinction matters: commission is charged on winnings only, which is
    exactly what the settlement formula does. Backing a short-priced favourite on
    an exchange therefore costs a small fraction of the stake, while the same 5%
    against a long shot costs far more.
    """
    columns = PRICE_GROUPS[market].get(book_timing)
    outcomes = outcome_matrix(frame, market)
    if not columns or outcomes is None:
        return {"available": False}
    probability, valid = devig(frame, columns)
    if valid.sum() < MIN_CELL:
        return {"available": False, "n": int(valid.sum())}

    odds = frame[list(columns)].to_numpy(dtype=float)[valid]
    probability = probability[valid]
    outcomes = outcomes[valid]
    blocks = pd.to_datetime(frame["match_date"]).dt.to_period("M").astype(str).to_numpy()[valid]

    flat_probability = probability.reshape(-1)
    flat_odds = odds.reshape(-1)
    flat_won = outcomes.reshape(-1)
    flat_blocks = np.repeat(blocks, outcomes.shape[1])

    report: dict[str, Any] = {
        "available": True,
        "matches": int(valid.sum()),
        "overround_median": float(np.nanmedian((1.0 / odds).sum(axis=1))),
        "bands": {},
    }
    for low_edge, high_edge in zip(PROBABILITY_EDGES[:-1], PROBABILITY_EDGES[1:]):
        band = (flat_probability >= low_edge) & (flat_probability < high_edge)
        if band.sum() < MIN_CELL:
            continue
        report["bands"][f"{low_edge:.2f}-{high_edge:.2f}"] = _cell(
            flat_won[band], flat_odds[band], flat_blocks[band], flat_probability[band],
            haircut=haircut,
        )
    report["haircut"] = haircut
    return report


def asian_handicap_returns(
    goal_difference: np.ndarray, line: np.ndarray, odds: np.ndarray, side: str,
    haircut: float = HAIRCUT,
) -> np.ndarray:
    """Unit return of one Asian handicap bet, pushes and quarter lines included.

    The published line is applied to the home team. A quarter line splits the
    stake across the two neighbouring half-lines, which is why a bet can win or
    lose exactly half. Treating those as wins or losses — the usual shortcut —
    would misstate the return of the market this project cares most about,
    because the quarter lines are where the tightest prices sit.
    """
    if side not in {"home", "away"}:
        raise ValueError(f"Unknown side: {side}")
    margin = goal_difference if side == "home" else -goal_difference
    handicap = line if side == "home" else -line
    adjusted = margin + handicap
    payout = 1.0 + (odds - 1.0) * (1.0 - haircut)
    win = payout - 1.0
    return np.select(
        [adjusted > 0.25, adjusted == 0.25, adjusted == 0.0, adjusted == -0.25],
        [win, win / 2.0, 0.0, -0.5],
        default=-1.0,
    )


def asian_handicap_audit(frame: pd.DataFrame, book: str, timing: str) -> dict[str, Any]:
    """Back every home line, then every away line, at the published handicap."""
    columns = PRICE_GROUPS["asian_handicap"].get(f"{book}_{timing}")
    line_column = HANDICAP_LINES["open" if timing == "open" else "close"]
    if not columns or line_column not in frame.columns:
        return {"available": False}

    prices = frame[list(columns)].to_numpy(dtype=float)
    line = pd.to_numeric(frame[line_column], errors="coerce").to_numpy(dtype=float)
    goal_difference = frame["goal_difference"].to_numpy(dtype=float)
    overround = (1.0 / prices).sum(axis=1)
    usable = (
        np.isfinite(prices).all(axis=1) & (prices > 1.0).all(axis=1)
        & np.isfinite(line) & np.isfinite(goal_difference)
        & (overround >= 0.98) & (overround <= 1.15)
        # Football-Data quotes quarter lines; anything off the 0.25 grid is a
        # transcription error rather than a real handicap.
        & (np.abs(line * 4 - np.round(line * 4)) < 1e-9)
    )
    if usable.sum() < MIN_CELL:
        return {"available": False, "n": int(usable.sum())}

    blocks = pd.to_datetime(frame["match_date"]).dt.to_period("M").astype(str).to_numpy()[usable]
    report: dict[str, Any] = {
        "available": True,
        "matches": int(usable.sum()),
        "overround_median": float(np.median(overround[usable])),
        "line_median": float(np.median(line[usable])),
        "sides": {},
    }
    for index, side in enumerate(("home", "away")):
        unit = asian_handicap_returns(
            goal_difference[usable], line[usable], prices[usable][:, index], side
        )
        low, high = _interval(unit, blocks)
        report["sides"][side] = {
            "n": int(len(unit)),
            "roi": float(unit.mean()),
            "roi_ci_90": [low, high],
            "push_rate": float((unit == 0.0).mean()),
            "profitable": bool(low is not None and low > 0.0),
            "significant": bool(low is not None and (low > 0.0 or high < 0.0)),
        }
    return report


def asian_handicap_by_country(frame: pd.DataFrame, book: str, timing: str,
                              side: str) -> dict[str, Any]:
    """The cross-league guard, applied to the handicap market."""
    per_country: dict[str, Any] = {}
    for country in sorted(frame["country"].dropna().unique()):
        block = asian_handicap_audit(frame[frame["country"] == country], book, timing)
        if block.get("available"):
            per_country[country] = block["sides"][side]
    profitable = [name for name, cell in per_country.items() if cell["profitable"]]
    return {
        "side": side,
        "countries": per_country,
        "profitable_countries": profitable,
        "countries_examined": len(per_country),
        "confirmed_in_majority": bool(len(profitable) > len(per_country) / 2),
    }


def cross_league_consistency(frame: pd.DataFrame, market: str, book_timing: str,
                             leg: int) -> dict[str, Any]:
    """A bias that pays in one country and not the others is a lucky country."""
    columns = PRICE_GROUPS[market].get(book_timing)
    outcomes = outcome_matrix(frame, market)
    if not columns or outcomes is None:
        return {"available": False}
    probability, valid = devig(frame, columns)
    odds = frame[list(columns)].to_numpy(dtype=float)
    blocks = pd.to_datetime(frame["match_date"]).dt.to_period("M").astype(str).to_numpy()
    countries = frame["country"].to_numpy()

    per_country: dict[str, Any] = {}
    for country in sorted(set(countries)):
        mask = valid & (countries == country)
        if mask.sum() < MIN_CELL:
            continue
        per_country[country] = _cell(
            outcomes[mask][:, leg], odds[mask][:, leg], blocks[mask], probability[mask][:, leg]
        )
    profitable = [name for name, cell in per_country.items() if cell.get("profitable")]
    return {
        "available": True,
        "leg": leg,
        "countries": per_country,
        "profitable_countries": profitable,
        "countries_examined": len(per_country),
        "confirmed_in_majority": bool(len(profitable) > len(per_country) / 2),
    }
