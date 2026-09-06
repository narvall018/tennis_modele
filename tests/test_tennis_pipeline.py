from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data.tennis_pipeline import (
    LEGACY_COLUMNS,
    _atomic_csv,
    add_stable_player_orientation,
    attach_odds,
    deterministic_orientation,
    normalize_rich_matches,
    transform_tennis_data_raw,
)


class TennisPipelineTests(unittest.TestCase):
    def test_gzip_publication_is_byte_reproducible(self):
        frame = pd.DataFrame({"value": [1, 2], "text": ["a", "b"]})
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv.gz"
            second = Path(directory) / "second.csv.gz"
            _atomic_csv(frame, first, gzip=True)
            _atomic_csv(frame, second, gzip=True)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_raw_transform_falls_back_when_pinnacle_is_nan(self):
        raw = pd.DataFrame(
            [
                {
                    "Tournament": "Test Open",
                    "Date": "2026-08-20",
                    "Series": "ATP250",
                    "Court": "Outdoor",
                    "Surface": "Hard",
                    "Round": "The Final",
                    "Best of": 3,
                    "Winner": "Winner A.",
                    "Loser": "Loser B.",
                    "Comment": "Completed",
                    "WRank": 10,
                    "LRank": 20,
                    "WPts": 3000,
                    "LPts": 1500,
                    "PSW": np.nan,
                    "PSL": np.nan,
                    "B365W": 1.70,
                    "B365L": 2.20,
                    "W1": 6,
                    "L1": 4,
                    "W2": 6,
                    "L2": 3,
                }
            ]
        )
        transformed = transform_tennis_data_raw(raw)
        row = transformed.iloc[0]
        winner_odd = row["Odd_1"] if row["Player_1"] == row["Winner"] else row["Odd_2"]
        loser_odd = row["Odd_2"] if row["Player_1"] == row["Winner"] else row["Odd_1"]
        self.assertEqual(winner_odd, 1.70)
        self.assertEqual(loser_odd, 2.20)

    def test_orientation_is_stable_when_source_order_changes(self):
        base = {
            "Tournament": "Test Open",
            "Date": "2026-08-20",
            "Series": "ATP250",
            "Court": "Outdoor",
            "Surface": "Hard",
            "Round": "The Final",
            "Best of": 3,
            "Winner": "Alpha A.",
        }
        first = pd.DataFrame(
            [{**base, "Player_1": "Alpha A.", "Player_2": "Beta B.", "Rank_1": 1, "Rank_2": 2, "Pts_1": 10, "Pts_2": 5, "Odd_1": 1.5, "Odd_2": 2.8, "Score": "6-4 6-3"}],
            columns=LEGACY_COLUMNS,
        )
        second = pd.DataFrame(
            [{**base, "Player_1": "Beta B.", "Player_2": "Alpha A.", "Rank_1": 2, "Rank_2": 1, "Pts_1": 5, "Pts_2": 10, "Odd_1": 2.8, "Odd_2": 1.5, "Score": "4-6 3-6"}],
            columns=LEGACY_COLUMNS,
        )
        left = deterministic_orientation(first).iloc[0]
        right = deterministic_orientation(second).iloc[0]
        pd.testing.assert_series_equal(left, right)

    def test_exact_match_attaches_odds_to_the_correct_players(self):
        rich_raw = pd.DataFrame(
            [
                {
                    "tourney_id": "2026-1",
                    "tourney_name": "Test Open",
                    "surface": "Hard",
                    "draw_size": 32,
                    "tourney_level": "A",
                    "indoor": "O",
                    "tourney_date": 20260820,
                    "match_num": 1,
                    "winner_id": "A1",
                    "winner_name": "Alice Alpha",
                    "winner_hand": "R",
                    "winner_ht": 180,
                    "winner_ioc": "FRA",
                    "winner_age": 25,
                    "winner_rank": 10,
                    "winner_rank_points": 3000,
                    "loser_id": "B1",
                    "loser_name": "Bob Beta",
                    "loser_hand": "L",
                    "loser_ht": 185,
                    "loser_ioc": "USA",
                    "loser_age": 26,
                    "loser_rank": 20,
                    "loser_rank_points": 1500,
                    "score": "6-4 6-3",
                    "best_of": 3,
                    "round": "F",
                    "minutes": 80,
                    "_source_file": "2026.csv",
                    "_source_updated_at": "2026-08-21T00:00:00Z",
                }
            ]
        )
        odds = pd.DataFrame(
            [
                {
                    "Tournament": "Test Open",
                    "Date": "2026-08-20",
                    "Series": "ATP250",
                    "Court": "Outdoor",
                    "Surface": "Hard",
                    "Round": "The Final",
                    "Best of": 3,
                    "Player_1": "Beta B.",
                    "Player_2": "Alpha A.",
                    "Winner": "Alpha A.",
                    "Rank_1": 20,
                    "Rank_2": 10,
                    "Pts_1": 1500,
                    "Pts_2": 3000,
                    "Odd_1": 2.20,
                    "Odd_2": 1.70,
                    "Score": "4-6 3-6",
                }
            ]
        )
        rich = normalize_rich_matches(rich_raw, today=date(2026, 9, 1))
        result = add_stable_player_orientation(attach_odds(rich, odds)).iloc[0]
        self.assertEqual(result["winner_odds"], 1.70)
        self.assertEqual(result["loser_odds"], 2.20)
        self.assertEqual(result["odds_match_confidence"], 0.99)


if __name__ == "__main__":
    unittest.main()
