"""Where, if anywhere, is the tennis moneyline systematically mispriced?

Every previous study asked whether *our model* beats the price. This one asks a
different and simpler question that needs no model at all: taken as it is, does
the market's own price match what happens? The same audit on the UFC method
market found a real +2.92-point bias on decisions, so it is a test that does find
things when they exist.

Two readings are produced for every slice:

* **Calibration** — the devigged Pinnacle probability against the realised win
  rate. Pinnacle is used as the reference because its margin is the smallest, so
  its devigged price is the least distorted estimate available.
* **Return** — what backing that side would actually have paid at a bettable
  price, margin included. A slice can be mispriced and still unprofitable, which
  is what a margin is for.

Multiplicity is the danger here, not leakage: many slices are examined, so some
will look significant by luck. Every slice is therefore declared in this file
before the run, all of them are reported, and a falsification arm splits the same
data on a deterministic hash of the match — a partition that cannot carry any
real signal. If the hash slices look like the real ones, nothing has been found.
"""

from __future__ import annotations

import hashlib
from typing import Any, Callable

import numpy as np
import pandas as pd


PROBABILITY_EDGES = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0)
HAIRCUT = 0.02
SHARP = ("Pinnacle_1", "Pinnacle_2")
BETTABLE = ("B365_1", "B365_2")


def _devig(frame: pd.DataFrame, columns: tuple[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    left = pd.to_numeric(frame[columns[0]], errors="coerce").to_numpy(dtype=float)
    right = pd.to_numeric(frame[columns[1]], errors="coerce").to_numpy(dtype=float)
    overround = 1.0 / left + 1.0 / right
    valid = (
        np.isfinite(overround) & (left > 1.0) & (right > 1.0)
        & (overround >= 0.98) & (overround <= 1.15)
    )
    probability = np.where(valid, (1.0 / left) / overround, np.nan)
    return probability, left, right, valid


def _hash_bucket(frame: pd.DataFrame, buckets: int = 4) -> np.ndarray:
    """A partition that cannot correlate with anything real."""
    keys = (
        frame["Date"].astype(str) + "|" + frame["Player_1"].astype(str)
        + "|" + frame["Player_2"].astype(str)
    )
    return np.array([
        int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16) % buckets for key in keys
    ])


# Declared before the run. Each entry maps a frame to a label per row.
SEGMENTS: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    "tier": lambda f: f["Series"].astype(str).str.strip().replace("", "unknown"),
    "surface": lambda f: f["Surface"].astype(str).str.strip().replace("", "unknown"),
    "best_of": lambda f: "Bo" + pd.to_numeric(f["Best of"], errors="coerce").fillna(3).astype(int).astype(str),
    "round_stage": lambda f: np.where(
        f["Round"].astype(str).str.contains("1st|2nd", regex=True, na=False), "early", "late"
    ),
    "rank_gap": lambda f: pd.cut(
        (pd.to_numeric(f["Rank_1"], errors="coerce") - pd.to_numeric(f["Rank_2"], errors="coerce")).abs(),
        [-0.1, 10, 30, 80, 1e9],
        labels=["gap_0_10", "gap_10_30", "gap_30_80", "gap_80_plus"],
    ).astype(object),
    "era": lambda f: np.where(pd.to_datetime(f["Date"]).dt.year < 2015, "2000_2014", "2015_2026"),
    "hash_control": lambda f: pd.Series(
        ["hash_" + str(value) for value in _hash_bucket(f)], index=f.index
    ),
}


def _cell(
    probability: np.ndarray,
    odds: np.ndarray,
    won: np.ndarray,
    months: np.ndarray,
    haircut: float = HAIRCUT,
    samples: int = 3000,
) -> dict[str, Any]:
    if not len(won):
        return {"n": 0}
    priced = 1.0 + (odds - 1.0) * (1.0 - haircut)
    unit = np.where(won, priced - 1.0, -1.0)
    codes, _ = pd.factorize(months)
    order = np.argsort(codes, kind="stable")
    sorted_unit, sorted_codes = unit[order], codes[order]
    block_count = int(codes.max()) + 1
    starts = np.searchsorted(sorted_codes, np.arange(block_count))
    ends = np.append(starts[1:], len(sorted_codes))
    sums = np.add.reduceat(sorted_unit, starts)
    counts = ends - starts
    rng = np.random.default_rng(0)
    picks = rng.integers(0, block_count, size=(samples, block_count))
    resampled = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    low, high = np.percentile(resampled, [5, 95])
    return {
        "n": int(len(won)),
        "market_probability": float(np.nanmean(probability)),
        "realised_rate": float(won.mean()),
        "calibration_gap": float(won.mean() - np.nanmean(probability)),
        "roi": float(unit.mean()),
        "roi_ci_90": [float(low), float(high)],
        "significant": bool(low > 0.0 or high < 0.0),
        "profitable": bool(low > 0.0),
    }


def audit(frame: pd.DataFrame, bettable: tuple[str, str] = BETTABLE) -> dict[str, Any]:
    """``bettable`` is the price actually staked; the reference stays Pinnacle.

    Passing Pinnacle itself is legitimate and is the sharpest test available — a
    small bias has its best chance against the smallest margin — but then the
    calibration columns become self-referential and only the ROI columns mean
    anything.
    """
    sharp_probability, _, _, sharp_ok = _devig(frame, SHARP)
    _, bet_left, bet_right, bet_ok = _devig(frame, bettable)
    completed = frame["Status"].astype(str).str.lower().eq("completed").to_numpy()
    usable = sharp_ok & bet_ok & completed
    label = frame["_label"].to_numpy(dtype=int)
    months = frame["_month"].to_numpy()

    # Orient every row onto the side the sharp book calls the favourite, so a
    # "favourite" cell means the same thing in every slice.
    favourite_is_p1 = sharp_probability >= 0.5
    favourite_probability = np.where(favourite_is_p1, sharp_probability, 1.0 - sharp_probability)
    favourite_odds = np.where(favourite_is_p1, bet_left, bet_right)
    favourite_won = np.where(favourite_is_p1, label == 1, label == 0)
    underdog_odds = np.where(favourite_is_p1, bet_right, bet_left)

    report: dict[str, Any] = {
        "matches_usable": int(usable.sum()),
        "reference_book": "Pinnacle (devigged)",
        "bettable_book": bettable[0].rsplit("_", 1)[0],
        "calibration_is_self_referential": bettable == SHARP,
        "odds_haircut": HAIRCUT,
        "by_probability_decile": {},
        "segments": {},
    }
    for low_edge, high_edge in zip(PROBABILITY_EDGES[:-1], PROBABILITY_EDGES[1:]):
        band = usable & (favourite_probability >= low_edge) & (favourite_probability < high_edge)
        if band.sum() < 200:
            continue
        report["by_probability_decile"][f"{low_edge:.1f}-{high_edge:.1f}"] = {
            "favourite": _cell(
                favourite_probability[band], favourite_odds[band], favourite_won[band], months[band]
            ),
            "underdog": _cell(
                1.0 - favourite_probability[band], underdog_odds[band], ~favourite_won[band], months[band]
            ),
        }

    for name, mapper in SEGMENTS.items():
        values = pd.Series(mapper(frame), index=frame.index).astype(str)
        block: dict[str, Any] = {}
        for value in sorted(values.dropna().unique()):
            slice_mask = usable & values.eq(value).to_numpy()
            if slice_mask.sum() < 300:
                continue
            block[value] = {
                "favourite": _cell(
                    favourite_probability[slice_mask], favourite_odds[slice_mask],
                    favourite_won[slice_mask], months[slice_mask],
                ),
                "underdog": _cell(
                    1.0 - favourite_probability[slice_mask], underdog_odds[slice_mask],
                    ~favourite_won[slice_mask], months[slice_mask],
                ),
            }
        report["segments"][name] = block

    cells = [
        cell
        for group in list(report["by_probability_decile"].values()) + [
            value for block in report["segments"].values() for value in block.values()
        ]
        for cell in group.values()
        if cell.get("n")
    ]
    report["multiplicity"] = {
        "cells_examined": len(cells),
        "cells_with_interval_excluding_zero": sum(cell["significant"] for cell in cells),
        "cells_profitable": sum(cell["profitable"] for cell in cells),
        "expected_by_chance_at_90pct": round(0.10 * len(cells), 1),
    }
    return report


def cross_tour_consistency(reports: dict[str, dict[str, Any]], book: str) -> dict[str, Any]:
    """A profitable cell must survive the other tour, or it is a lucky cell.

    With this many slices some will look profitable by chance, and the only cheap
    defence is that ATP and WTA are independent samples of the same market
    behaviour. A bias that is structural appears in both; a bias that appears in
    one and reverses in the other is noise, whatever its interval says.
    """
    tours = [key for key in reports if key.endswith(f"_{book}")]
    if len(tours) < 2:
        return {"checked": False}

    verdicts: dict[str, Any] = {"checked": True, "book": book, "cells": {}}
    left_report, right_report = (reports[tour] for tour in tours[:2])
    bands = set(left_report["by_probability_decile"]) & set(right_report["by_probability_decile"])
    for band in sorted(bands):
        for side in ("favourite", "underdog"):
            left = left_report["by_probability_decile"][band][side]
            right = right_report["by_probability_decile"][band][side]
            if not (left.get("profitable") or right.get("profitable")):
                continue
            total = left["n"] + right["n"]
            pooled_roi = (left["roi"] * left["n"] + right["roi"] * right["n"]) / total
            verdicts["cells"][f"{band}/{side}"] = {
                tours[0]: {"n": left["n"], "roi": left["roi"], "profitable": left["profitable"]},
                tours[1]: {"n": right["n"], "roi": right["roi"], "profitable": right["profitable"]},
                "pooled_roi": pooled_roi,
                "confirmed_by_both_tours": bool(left["profitable"] and right["profitable"]),
                "verdict": (
                    "confirmé par les deux circuits"
                    if left["profitable"] and right["profitable"]
                    else "cellule chanceuse: l'autre circuit ne la confirme pas"
                ),
            }
    verdicts["any_confirmed"] = any(
        cell["confirmed_by_both_tours"] for cell in verdicts["cells"].values()
    )
    return verdicts
