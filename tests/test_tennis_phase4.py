from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.backtesting.tennis_phase4 import build_phase4_features, sha256_file
from src.data.tennis_odds_collector import normalise_tennis_odds_payload


BASE = Path(__file__).resolve().parents[1]


def _row(match_id: str, match_date: str, p1: str, p2: str, label: int, minutes: int = 80) -> dict:
    winner, loser = (p1, p2) if label == 1 else (p2, p1)
    return {
        "match_id": match_id,
        "match_date": match_date,
        "match_status": "completed",
        "surface": "Hard",
        "tourney_name": "Test Open",
        "tourney_level": "250",
        "round": "R32",
        "best_of": 3,
        "indoor": "O",
        "player_1_id": p1,
        "player_2_id": p2,
        "player_1_name": p1,
        "player_2_name": p2,
        "player_1_won": label,
        "player_1_odds": 1.90,
        "player_2_odds": 2.00,
        "player_1_rank": 10,
        "player_2_rank": 20,
        "player_1_rank_points": 2000,
        "player_2_rank_points": 1200,
        "player_1_age": 25,
        "player_2_age": 27,
        "player_1_ht": 185,
        "player_2_ht": 182,
        "player_1_hand": "R",
        "player_2_hand": "L",
        "minutes": minutes,
        "winner_name": winner,
        "loser_name": loser,
        "postmatch_player_1_svpt": 60,
        "postmatch_player_2_svpt": 62,
        "postmatch_player_1_1stIn": 38,
        "postmatch_player_2_1stIn": 39,
        "postmatch_player_1_1stWon": 28,
        "postmatch_player_2_1stWon": 27,
        "postmatch_player_1_2ndWon": 11,
        "postmatch_player_2_2ndWon": 10,
        "postmatch_player_1_ace": 5,
        "postmatch_player_2_ace": 4,
        "postmatch_player_1_df": 2,
        "postmatch_player_2_df": 3,
        "postmatch_player_1_bpSaved": 4,
        "postmatch_player_2_bpSaved": 3,
        "postmatch_player_1_bpFaced": 6,
        "postmatch_player_2_bpFaced": 7,
    }


def _build(rows: list[dict]) -> pd.DataFrame:
    with TemporaryDirectory() as directory:
        path = Path(directory) / "matches.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        features, _ = build_phase4_features(path, progress=lambda _: None)
    return features


def test_future_match_does_not_change_existing_phase4_features():
    base = [
        _row("m1", "2020-01-01", "A", "B", 1),
        _row("m2", "2020-01-02", "A", "C", 0),
    ]
    past = _build(base)
    extended = _build([*base, _row("m3", "2021-01-01", "A", "D", 1)])
    pd.testing.assert_frame_equal(
        past.reset_index(drop=True),
        extended.iloc[: len(past)].reset_index(drop=True),
        check_dtype=False,
    )


def test_same_day_matches_share_start_of_day_snapshot():
    features = _build(
        [
            _row("m1", "2020-01-01", "A", "B", 1, 120),
            _row("m2", "2020-01-01", "A", "C", 0, 180),
            _row("m3", "2020-01-02", "A", "D", 1, 90),
        ]
    ).set_index("_match_id")
    assert features.loc["m1", "global_elo_diff"] == 0.0
    assert features.loc["m2", "global_elo_diff"] == 0.0
    assert features.loc["m2", "minutes_3d_diff"] == 0.0
    assert features.loc["m3", "minutes_3d_diff"] == 1.0


def test_phase4_protocol_hash_and_fail_closed_lock():
    output = BASE / "models" / "rigorous_strategy"
    protocol = output / "phase4_protocol.json"
    lock = json.loads((output / "phase4_lock.json").read_text())
    report = json.loads((output / "phase4_report.json").read_text())
    assert lock["protocol_sha256"] == sha256_file(protocol)
    assert lock["data_sha256"] == json.loads(protocol.read_text())["data_sha256"]
    assert lock["real_money_approved"] is False
    assert report["deep_learning_diagnostic"]["strategy_eligible"] is False
    assert report["decision"]["real_money_approved"] is False


def test_phase4_bets_and_predictions_are_unique():
    output = BASE / "models" / "rigorous_strategy"
    bets = pd.read_parquet(output / "phase4_development_bets.parquet")
    predictions = pd.read_parquet(output / "phase4_oos_predictions.parquet")
    protocol = json.loads((output / "phase4_protocol.json").read_text())
    rule = protocol["fixed_economic_diagnostic"]
    assert bets["_match_id"].is_unique
    assert predictions["_match_id"].is_unique
    assert bets["edge"].ge(rule["minimum_edge"] - 1e-12).all()
    assert bets["expected_roi"].ge(rule["minimum_expected_roi"] - 1e-12).all()
    assert bets["bet_odds"].between(rule["minimum_decimal_odds"], rule["maximum_decimal_odds"]).all()


def test_prospective_tennis_collector_rejects_post_start_rows():
    payload = [
        {
            "id": "event-1",
            "commence_time": "2026-09-04T12:00:00Z",
            "home_team": "Player B",
            "away_team": "Player A",
            "bookmakers": [
                {
                    "key": "book",
                    "title": "Book",
                    "last_update": "2026-09-03T10:00:00Z",
                    "markets": [
                        {
                            "key": "h2h",
                            "last_update": "2026-09-03T10:00:00Z",
                            "outcomes": [
                                {"name": "Player B", "price": 2.1},
                                {"name": "Player A", "price": 1.8}
                            ]
                        }
                    ]
                }
            ]
        }
    ]
    before = normalise_tennis_odds_payload(
        payload, "tennis_atp_test", "ATP Test", pd.Timestamp("2026-09-03T12:00:00Z")
    )
    after = normalise_tennis_odds_payload(
        payload, "tennis_atp_test", "ATP Test", pd.Timestamp("2026-09-04T13:00:00Z")
    )
    assert before.loc[0, "player_1"] == "Player A"
    assert before.loc[0, "temporal_status"] == "PRE_MATCH"
    assert after.loc[0, "temporal_status"] == "POST_START_EXCLUDED"
