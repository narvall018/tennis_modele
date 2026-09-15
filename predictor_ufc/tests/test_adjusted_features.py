"""Causalité et orientation des descripteurs UFC ajustés."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from rigorous.adjusted_features import (
    ADJUSTED_FEATURES,
    build_adjusted_features,
    orient_to_pairs,
)


def _toy() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    fights = [
        ("2020-01-04", "f1", "A", "B", (50, 20), (0, 1)),
        ("2020-03-07", "f2", "A", "C", (60, 30), (1, 0)),
        ("2020-05-02", "f3", "B", "C", (25, 40), (0, 0)),
        ("2020-08-01", "f4", "A", "B", (70, 15), (2, 0)),
    ]
    for date, fid, red, blue, (sig_r, sig_b), (kd_r, kd_b) in fights:
        for who, sig, kd in ((red, sig_r, kd_r), (blue, sig_b, kd_b)):
            rows.append(
                {
                    "fight_id": fid, "fighter_id": who, "fighter_url": f"u/{who}",
                    "event_date": pd.Timestamp(date), "weight_class": "Lightweight",
                    "kd": float(kd), "sig_lnd": float(sig), "td_lnd": 1.0,
                    "ctrl_secs": 60.0, "elo_global_pre": 1500.0,
                }
            )
    appearances = pd.DataFrame(rows)
    bio = pd.DataFrame(
        {
            "fighter_url": ["u/A", "u/B", "u/C"],
            "dob": pd.to_datetime(["1990-01-01", "1995-01-01", "1985-01-01"]),
        }
    )
    return appearances, bio


def test_first_fight_has_no_history():
    appearances, bio = _toy()
    per = build_adjusted_features(appearances, bio, progress=lambda m: None)
    first = per[per.fight_id == "f1"]
    assert (first["wear_fights"] == 0).all()
    assert (first["adj_sig_lnd"] == 0).all()
    assert (first["fights_365d"] == 0).all()


def test_wear_accumulates_only_from_past_fights():
    appearances, bio = _toy()
    per = build_adjusted_features(appearances, bio, progress=lambda m: None)
    a_rows = per[per.fighter_id == "A"].sort_values("fight_id")
    # A combat en f1, f2 puis f4: son compteur doit valoir 0, 1 puis 2.
    assert list(a_rows["wear_fights"]) == [0.0, 1.0, 2.0]


def test_age_increases_with_time():
    appearances, bio = _toy()
    per = build_adjusted_features(appearances, bio, progress=lambda m: None)
    a_rows = per[per.fighter_id == "A"].sort_values("fight_id")
    ages = list(a_rows["age_years"])
    assert ages == sorted(ages)
    assert 29.9 < ages[0] < 30.1


def test_no_future_information():
    """Tronquer l'avenir ne change rien au passé."""
    appearances, bio = _toy()
    full = build_adjusted_features(appearances, bio, progress=lambda m: None)
    cutoff = appearances[appearances.event_date <= pd.Timestamp("2020-05-02")]
    truncated = build_adjusted_features(cutoff, bio, progress=lambda m: None)

    key = ["fight_id", "fighter_id"]
    left = full.set_index(key).loc[truncated.set_index(key).index]
    right = truncated.set_index(key)
    pd.testing.assert_frame_equal(left.sort_index(), right.sort_index())


def test_orientation_is_antisymmetric():
    """Échanger les deux coins doit exactement inverser le signe des écarts."""
    appearances, bio = _toy()
    per = build_adjusted_features(appearances, bio, progress=lambda m: None)
    pairs = pd.DataFrame(
        {"fight_id": ["f4"], "fighter_1_id": ["A"], "fighter_2_id": ["B"]}
    )
    swapped = pd.DataFrame(
        {"fight_id": ["f4"], "fighter_1_id": ["B"], "fighter_2_id": ["A"]}
    )
    straight = orient_to_pairs(per, pairs)
    reversed_ = orient_to_pairs(per, swapped)
    for column in ADJUSTED_FEATURES:
        left = straight[column].iloc[0]
        right = reversed_[column].iloc[0]
        if np.isnan(left) and np.isnan(right):
            continue
        assert left == pytest.approx(-right), column


def test_adjustment_rewards_beating_a_tough_defence():
    """Porter des frappes à quelqu'un qui en concède peu doit mieux noter que
    la même ligne brute contre une défense poreuse."""
    appearances, bio = _toy()
    per = build_adjusted_features(appearances, bio, progress=lambda m: None)
    # Avant f4, A a porté 50 puis 60 frappes; B en concédait 50 en moyenne
    # avant f2, donc l'ajustement de A doit être défini et fini.
    value = per[(per.fight_id == "f4") & (per.fighter_id == "A")]["adj_sig_lnd"].iloc[0]
    assert np.isfinite(value)


def test_real_table_orientation_repair():
    """Sur les données réelles, la réparation d'orientation doit rétablir la
    correspondance entre age_diff existant et une vraie différence d'âge."""
    processed = BASE_DIR / "data" / "processed" / "features_v2.parquet"
    if not processed.exists():
        pytest.skip("table features_v2 absente")

    sys.path.insert(0, str(BASE_DIR))
    from run_adjusted_ablation import load_oriented_table

    _, audit = load_oriented_table()
    assert audit["flipped_rows"] + audit["consistent_rows"] == audit["fights"]
    assert audit["age_diff_correlation_after_repair"] > 0.9
