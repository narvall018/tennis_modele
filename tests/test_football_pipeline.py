from __future__ import annotations

import unittest
from datetime import date

import numpy as np
import pandas as pd

from src.backtesting.football_audit import devig, outcome_matrix, timing_comparison
from src.data.football_pipeline import (
    MISSING_FILE_STATUSES,
    PRICE_GROUPS,
    build_quality_report,
    normalize,
    validate,
)
from src.data.tennis_pipeline import DataQualityError


def _raw(**overrides) -> dict:
    row = {
        "Div": "E0",
        "Date": "12/08/2023",
        "HomeTeam": "Arsenal",
        "AwayTeam": "Chelsea",
        "FTHG": 2,
        "FTAG": 1,
        "FTR": "H",
        "PSH": 1.90, "PSD": 3.60, "PSA": 4.20,
        "PSCH": 1.80, "PSCD": 3.70, "PSCA": 4.60,
        "_league": "E0",
        "_season_start": 2023,
    }
    row.update(overrides)
    return row


class FootballPipelineTests(unittest.TestCase):
    def test_a_missing_season_is_not_a_failure(self):
        """The site answers 300, not 404, for a division that did not play."""
        self.assertIn(300, MISSING_FILE_STATUSES)
        self.assertIn(404, MISSING_FILE_STATUSES)

    def test_both_year_formats_are_parsed(self):
        frame = normalize(
            pd.DataFrame([_raw(Date="12/08/2023"), _raw(Date="13/08/23", HomeTeam="Spurs")]),
            today=date(2026, 9, 6),
        )
        self.assertEqual(len(frame), 2)
        self.assertEqual(sorted(frame["match_date"]), ["2023-08-12", "2023-08-13"])

    def test_a_result_contradicting_its_score_is_dropped(self):
        """A published result that disagrees with the goals is a corrupt row."""
        frame = normalize(
            pd.DataFrame([_raw(), _raw(HomeTeam="Spurs", FTHG=0, FTAG=3, FTR="H")]),
            today=date(2026, 9, 6),
        )
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["home_team"], "Arsenal")

    def test_future_matches_are_excluded(self):
        frame = normalize(
            pd.DataFrame([_raw(Date="12/08/2030")]), today=date(2026, 9, 6)
        )
        self.assertEqual(len(frame), 0)

    def test_opening_and_closing_prices_stay_in_separate_columns(self):
        frame = normalize(pd.DataFrame([_raw()]), today=date(2026, 9, 6))
        row = frame.iloc[0]
        self.assertEqual(row["PSH"], 1.90)
        self.assertEqual(row["PSCH"], 1.80)
        self.assertNotEqual(row["PSH"], row["PSCH"])

    def test_validation_rejects_an_implausible_home_rate(self):
        report = {
            "duplicate_match_ids": 0,
            "matches": 200_000,
            "matches_with_pinnacle_open_and_close": 50_000,
            "result_distribution": {"H": 0.80, "D": 0.10, "A": 0.10},
            "date_max": str(date(2026, 9, 1)),
        }
        with self.assertRaises(DataQualityError):
            validate(report, today=date(2026, 9, 6))

    def test_quality_report_counts_matches_with_both_timings(self):
        frame = normalize(
            pd.DataFrame([_raw(), _raw(HomeTeam="Spurs", PSCH=np.nan)]), today=date(2026, 9, 6)
        )
        report = build_quality_report(frame, today=date(2026, 9, 6))
        self.assertEqual(report["matches_with_pinnacle_open_and_close"], 1)


class FootballAuditTests(unittest.TestCase):
    def test_devig_sums_to_one_and_rejects_partial_markets(self):
        frame = pd.DataFrame({
            "PSH": [2.0, 2.0, np.nan],
            "PSD": [4.0, 4.0, 4.0],
            "PSA": [4.0, np.nan, 4.0],
        })
        probabilities, valid = devig(frame, ("PSH", "PSD", "PSA"))
        self.assertTrue(valid[0])
        self.assertFalse(valid[1])
        self.assertFalse(valid[2])
        self.assertAlmostEqual(probabilities[0].sum(), 1.0)

    def test_outcome_matrix_matches_the_market_leg_order(self):
        frame = pd.DataFrame({
            "result": ["H", "D", "A"],
            "total_goals": [4.0, 1.0, 3.0],
        })
        one_x_two = outcome_matrix(frame, "1x2")
        np.testing.assert_array_equal(one_x_two.argmax(axis=1), [0, 1, 2])
        over_under = outcome_matrix(frame, "over_under_25")
        np.testing.assert_array_equal(over_under[:, 0], [True, False, True])

    def test_an_asian_handicap_has_no_derivable_outcome(self):
        """The handicap line varies per match, so a naive leg map would be wrong."""
        frame = pd.DataFrame({"result": ["H"], "total_goals": [3.0]})
        self.assertIsNone(outcome_matrix(frame, "asian_handicap"))

    def test_timing_comparison_uses_only_matches_carrying_both_quotes(self):
        size = 800
        rng = np.random.default_rng(0)
        frame = pd.DataFrame({
            "match_date": pd.date_range("2020-01-01", periods=size, freq="D").astype(str),
            "result": rng.choice(["H", "D", "A"], size),
            "total_goals": rng.integers(0, 6, size).astype(float),
            "PSH": 2.0, "PSD": 3.6, "PSA": 4.0,
            "PSCH": 2.1, "PSCD": 3.5, "PSCA": 3.9,
        })
        frame.loc[frame.index[:200], "PSCH"] = np.nan
        block = timing_comparison(frame, "1x2", "pinnacle")
        self.assertTrue(block["available"])
        self.assertEqual(block["matches"], size - 200)

    def test_price_groups_never_mix_two_books(self):
        for market, books in PRICE_GROUPS.items():
            for book, columns in books.items():
                prefix = "C" in book
                self.assertGreaterEqual(len(columns), 2, f"{market}/{book}")
                self.assertEqual(len(set(columns)), len(columns), f"{market}/{book}")


if __name__ == "__main__":
    unittest.main()
