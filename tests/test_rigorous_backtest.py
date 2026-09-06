from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtesting.rigorous_strategy import (
    BetRule,
    StakePlan,
    apply_rule,
    prepare_bet_candidates,
    simulate_bankroll,
)
from src.features.elo_system import TennisEloEngine
from src.features.feature_builder import FeatureBuilder


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Date": "2020-01-01", "Tournament": "A", "Round": "1st Round",
                "Player_1": "Alpha", "Player_2": "Beta", "Winner": "Alpha",
                "Surface": "Hard", "Series": "ATP250", "Status": "completed",
                "Rank_1": 10, "Rank_2": 20, "Pts_1": 1000, "Pts_2": 500,
                "Odd_1": 1.7, "Odd_2": 2.2,
            },
            {
                "Date": "2020-01-02", "Tournament": "A", "Round": "2nd Round",
                "Player_1": "Alpha", "Player_2": "Gamma", "Winner": "Gamma",
                "Surface": "Hard", "Series": "ATP250", "Status": "completed",
                "Rank_1": 10, "Rank_2": 30, "Pts_1": 1000, "Pts_2": 300,
                "Odd_1": 1.5, "Odd_2": 2.8,
            },
            {
                "Date": "2021-01-01", "Tournament": "B", "Round": "1st Round",
                "Player_1": "Alpha", "Player_2": "Delta", "Winner": "Delta",
                "Surface": "Clay", "Series": "ATP250", "Status": "completed",
                "Rank_1": 12, "Rank_2": 25, "Pts_1": 900, "Pts_2": 400,
                "Odd_1": 1.8, "Odd_2": 2.1,
            },
        ]
    )


class LeakageGuardsTests(unittest.TestCase):
    def test_appending_a_future_match_does_not_change_past_features(self):
        base = _matches().iloc[:2].copy()
        extended = _matches().copy()
        base_features = FeatureBuilder().build_dataset(TennisEloEngine().fit(base).get_history())
        extended_features = FeatureBuilder().build_dataset(TennisEloEngine().fit(extended).get_history())
        pd.testing.assert_frame_equal(
            base_features.reset_index(drop=True),
            extended_features.iloc[:2].reset_index(drop=True),
            check_dtype=False,
        )

    def test_fit_resets_elo_state(self):
        engine = TennisEloEngine().fit(_matches().iloc[:2])
        first = engine.get_history().copy()
        engine.fit(_matches().iloc[:2])
        pd.testing.assert_frame_equal(first, engine.get_history())

    def test_missing_odds_never_create_a_bet(self):
        frame = pd.DataFrame(
            {
                "Avg_1": [np.nan], "Avg_2": [2.0],
                "model_probability_p1": [0.9], "_p1": ["A"], "_p2": ["B"],
                "_label": [1], "_status": ["completed"],
            }
        )
        candidates = prepare_bet_candidates(frame, model_weight=1.0)
        permissive = BetRule(0.0, -1.0, 0.0, 1.01, 100.0)
        self.assertTrue(apply_rule(candidates, permissive).empty)

    def test_same_day_stakes_use_start_of_day_bankroll(self):
        bets = pd.DataFrame(
            {
                "_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
                "_tournament": ["A", "B"], "_source_row_id": [1, 2],
                "bet_player": ["A", "B"], "_status": ["completed", "completed"],
                "bet_odds": [2.0, 2.0], "bet_probability": [0.6, 0.6],
                "edge": [0.1, 0.1], "expected_roi": [0.2, 0.2], "won": [True, True],
            }
        )
        plan = StakePlan("test", "flat", flat_fraction=0.01, max_bet_fraction=0.01, max_daily_fraction=0.05)
        _, ledger = simulate_bankroll(bets, plan, initial_bankroll=1000.0)
        self.assertEqual(ledger["stake"].tolist(), [10.0, 10.0])


if __name__ == "__main__":
    unittest.main()

