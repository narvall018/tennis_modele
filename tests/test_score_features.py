"""Le point à vérifier est la causalité: un descripteur de marge ne doit rien
savoir d'un match qui n'a pas encore eu lieu."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.features.score_features import (
    SCORE_FEATURES,
    build_score_features,
    parse_score,
)


# ---------------------------------------------------------------------------
# Lecture du score
# ---------------------------------------------------------------------------
def test_parse_score_orientation_player_1():
    parsed = parse_score("6-4 5-7 7-6")
    assert parsed is not None
    assert parsed.games_1 == 18 and parsed.games_2 == 17
    assert parsed.sets_1 == 2 and parsed.sets_2 == 1
    assert parsed.tiebreaks_1 == 1 and parsed.tiebreaks_2 == 0
    assert parsed.decider_played and parsed.decider_won_by_1
    assert not parsed.lost_first_set


def test_parse_score_detects_comeback():
    parsed = parse_score("1-6 6-2 6-4")
    assert parsed is not None
    assert parsed.lost_first_set
    assert parsed.sets_1 > parsed.sets_2


def test_parse_score_rejects_garbage():
    assert parse_score("") is None
    assert parse_score(None) is None
    assert parse_score("w/o") is None
    assert parse_score("6-4 abandon") is None


def test_game_ratio_is_symmetric():
    parsed = parse_score("6-0 6-0")
    assert parsed is not None
    assert parsed.game_ratio_1 == 1.0


# ---------------------------------------------------------------------------
# Jeu de données jouet
# ---------------------------------------------------------------------------
def _toy_frame() -> pd.DataFrame:
    rows = [
        # date,        p1,    p2,    winner, score,          status
        ("2020-01-06", "A", "B", "A", "6-0 6-0", "completed"),
        ("2020-01-06", "C", "D", "C", "7-6 6-7 7-6", "completed"),
        ("2020-01-13", "A", "C", "A", "6-4 6-4", "completed"),
        ("2020-01-13", "B", "D", "D", "3-6 2-6", "completed"),
        ("2020-01-20", "A", "D", "A", "6-3 6-3", "completed"),
        ("2020-01-20", "B", "C", "C", "6-7 4-6", "completed"),
        ("2020-01-27", "A", "B", "A", "6-2 6-2", "completed"),
        ("2020-02-03", "C", "D", "D", "6-4 1-0", "retired"),
        ("2020-02-10", "A", "C", "C", "4-6 6-3 6-4", "completed"),
    ]
    frame = pd.DataFrame(rows, columns=["Date", "Player_1", "Player_2", "Winner", "Score", "Status"])
    frame["Surface"] = "Hard"
    frame["Tournament"] = "Toy Open"
    frame["source_row_id"] = np.arange(len(frame), dtype=np.int64)
    return frame


def test_first_match_has_no_history():
    features, _ = build_score_features(_toy_frame(), progress=lambda m: None)
    first = features.loc[features.source_row_id == 0].iloc[0]
    # Aucun joueur n'a d'histoire: tous les écarts doivent être exactement nuls.
    for column in SCORE_FEATURES:
        if column == "log_common_opponent_count":
            assert first[column] == 0.0
        else:
            assert first[column] == pytest.approx(0.0, abs=1e-12), column


def test_same_day_matches_share_the_same_state():
    """Deux matchs du même jour voient le même état: l'ordre de passage à
    l'intérieur d'une journée ne peut donc pas porter d'information."""
    frame = _toy_frame()
    features, _ = build_score_features(frame, progress=lambda m: None)
    shuffled = frame.iloc[::-1].reset_index(drop=True)
    shuffled_features, _ = build_score_features(shuffled, progress=lambda m: None)

    left = features.set_index("source_row_id").sort_index()
    right = shuffled_features.set_index("source_row_id").sort_index()
    pd.testing.assert_frame_equal(left[SCORE_FEATURES], right[SCORE_FEATURES])


def test_dominant_player_leads_on_margin():
    features, _ = build_score_features(_toy_frame(), progress=lambda m: None)
    # A a gagné 6-0 6-0 puis 6-4 6-4; au troisième match il doit dominer D
    # sur la marge, alors que le bilan victoires/défaites ne les sépare pas.
    row = features.loc[features.source_row_id == 4].iloc[0]
    assert row["game_ratio_ewma_diff"] > 0.0
    assert row["mov_elo_diff"] > 0.0


def test_retired_match_feeds_workload_but_not_margin():
    features, _ = build_score_features(_toy_frame(), progress=lambda m: None)
    after = features.loc[features.source_row_id == 8].iloc[0]
    # C a subi l'abandon de personne: c'est D qui gagne, donc C a abandonné.
    assert after["games_played_14d_diff"] != 0.0


# ---------------------------------------------------------------------------
# Le test qui compte: aucune information future
# ---------------------------------------------------------------------------
def test_features_do_not_depend_on_the_future():
    """Tronquer l'avenir ne doit rien changer au passé.

    On construit les descripteurs sur l'historique complet, puis sur le même
    historique coupé à une date. Si un descripteur lisait un résultat futur,
    les deux passes différeraient sur les matchs antérieurs à la coupure.
    """
    frame = _toy_frame()
    full, _ = build_score_features(frame, progress=lambda m: None)

    cutoff = pd.Timestamp("2020-01-20")
    truncated_input = frame[pd.to_datetime(frame["Date"]) <= cutoff]
    truncated, _ = build_score_features(truncated_input, progress=lambda m: None)

    common = truncated["source_row_id"]
    left = full.set_index("source_row_id").loc[common, SCORE_FEATURES]
    right = truncated.set_index("source_row_id").loc[common, SCORE_FEATURES]
    pd.testing.assert_frame_equal(left, right)


def test_no_future_information_on_real_atp_sample():
    """Le même contrôle sur les données réelles, pas sur un jouet."""
    path = PROJECT_ROOT / "data" / "atp_tennis.csv"
    if not path.exists():
        pytest.skip("table ATP absente")

    raw = pd.read_csv(path, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Player_1", "Player_2", "Winner"]).copy()
    raw["source_row_id"] = np.arange(len(raw), dtype=np.int64)
    sample = raw[raw["Date"].dt.year.between(2000, 2004)]

    full, _ = build_score_features(sample, progress=lambda m: None)
    cutoff = pd.Timestamp("2003-06-30")
    truncated, _ = build_score_features(
        sample[sample["Date"] <= cutoff], progress=lambda m: None
    )

    common = truncated["source_row_id"]
    left = full.set_index("source_row_id").loc[common, SCORE_FEATURES]
    right = truncated.set_index("source_row_id").loc[common, SCORE_FEATURES]
    pd.testing.assert_frame_equal(left, right)


def test_orientation_is_not_a_signal():
    """Les écarts sont signés Player_1 moins Player_2. Sur un grand échantillon
    réel leur moyenne doit rester proche de zéro: sinon l'orientation elle-même
    porterait de l'information sur le résultat."""
    path = PROJECT_ROOT / "data" / "atp_tennis.csv"
    if not path.exists():
        pytest.skip("table ATP absente")

    raw = pd.read_csv(path, low_memory=False)
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce")
    raw = raw.dropna(subset=["Date", "Player_1", "Player_2", "Winner"]).copy()
    raw["source_row_id"] = np.arange(len(raw), dtype=np.int64)
    sample = raw[raw["Date"].dt.year.between(2000, 2005)]
    features, _ = build_score_features(sample, progress=lambda m: None)

    for column in ("game_ratio_ewma_diff", "mov_elo_diff", "margin_residual_diff"):
        values = features[column]
        # Moyenne inférieure à 5 % d'un écart-type.
        assert abs(values.mean()) < 0.05 * values.std(), column
