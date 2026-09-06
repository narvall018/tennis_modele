from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from src.data.tennis_expansion import (
    _add_empty_odds_columns,
    _segment_summary,
    validate_expansion,
)
from src.data.tennis_pipeline import (
    DataQualityError,
    add_stable_player_orientation,
    normalize_rich_matches,
    transform_tennis_data_raw,
)


def _rich_row(**overrides: object) -> dict[str, object]:
    row = {
        "tourney_id": "2026-W1",
        "tourney_name": "Test Cup",
        "surface": "Hard",
        "draw_size": 32,
        "tourney_level": "W",
        "indoor": "O",
        "tourney_date": 20260820,
        "match_num": 1,
        "winner_id": "W1",
        "winner_name": "Alice Alpha",
        "winner_hand": "R",
        "winner_ht": 175,
        "winner_ioc": "FRA",
        "winner_age": 24,
        "winner_rank": 8,
        "winner_rank_points": 3200,
        "loser_id": "L1",
        "loser_name": "Bea Beta",
        "loser_hand": "L",
        "loser_ht": 180,
        "loser_ioc": "USA",
        "loser_age": 27,
        "loser_rank": 30,
        "loser_rank_points": 1200,
        "score": "6-4 6-3",
        "best_of": 3,
        "round": "F",
        "minutes": 78,
        "_source_file": "2026_wta.csv",
        "_source_updated_at": "2026-08-21T00:00:00Z",
    }
    row.update(overrides)
    return row


class TennisExpansionTests(unittest.TestCase):
    def test_wta_tier_column_is_read_as_the_event_grade(self):
        raw = pd.DataFrame(
            [
                {
                    "Tournament": "Test Cup",
                    "Date": "2026-08-20",
                    "Tier": "WTA1000",
                    "Court": "Outdoor",
                    "Surface": "Hard",
                    "Round": "The Final",
                    "Best of": 3,
                    "Winner": "Alpha A.",
                    "Loser": "Beta B.",
                    "Comment": "Completed",
                    "WRank": 8,
                    "LRank": 30,
                    "AvgW": 1.55,
                    "AvgL": 2.45,
                    "W1": 6,
                    "L1": 4,
                }
            ]
        )
        # fetch_odds_snapshot renames Tier before transforming; emulate that step.
        raw["Series"] = raw["Tier"]
        transformed = transform_tennis_data_raw(raw)
        self.assertEqual(transformed.iloc[0]["Series"], "WTA1000")

    def test_identifier_prefix_keeps_tours_and_tiers_apart(self):
        wta = normalize_rich_matches(pd.DataFrame([_rich_row()]), today=date(2026, 9, 1), id_prefix="wta:")
        challenger = normalize_rich_matches(
            pd.DataFrame([_rich_row()]), today=date(2026, 9, 1), id_prefix="chal:"
        )
        self.assertTrue(wta.iloc[0]["match_id"].startswith("wta:"))
        self.assertNotEqual(wta.iloc[0]["match_id"], challenger.iloc[0]["match_id"])

    def test_unpriced_matches_carry_no_price(self):
        normalized = normalize_rich_matches(
            pd.DataFrame([_rich_row()]), today=date(2026, 9, 1), id_prefix="chal:"
        )
        oriented = add_stable_player_orientation(_add_empty_odds_columns(normalized))
        for column in ["player_1_odds", "player_2_odds", "market_overround"]:
            self.assertTrue(np.isnan(float(oriented.iloc[0][column])))

    def test_validation_rejects_a_stale_snapshot(self):
        frame = normalize_rich_matches(
            pd.DataFrame([_rich_row()]), today=date(2026, 9, 1), id_prefix="wta:"
        )
        report = {
            "wta": {
                "matches": _segment_summary(frame, "wta main"),
                "odds": {"priced_rows": 40_000, "one_sided_pairs": 0},
            },
            "unpriced": {},
        }
        validate_expansion(report, today=date(2026, 9, 1))
        with self.assertRaises(DataQualityError):
            validate_expansion(report, today=date(2026, 12, 1))

    def test_validation_rejects_a_one_sided_price_pair(self):
        frame = normalize_rich_matches(
            pd.DataFrame([_rich_row()]), today=date(2026, 9, 1), id_prefix="wta:"
        )
        report = {
            "wta": {
                "matches": _segment_summary(frame, "wta main"),
                "odds": {"priced_rows": 40_000, "one_sided_pairs": 3},
            },
            "unpriced": {},
        }
        with self.assertRaises(DataQualityError):
            validate_expansion(report, today=date(2026, 9, 1))


if __name__ == "__main__":
    unittest.main()
