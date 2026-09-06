from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from rigorous.data_pipeline import fight_key
from rigorous.secondary_odds import (
    american_to_decimal,
    canonicalise_secondary_odds,
    cross_check,
    match_secondary_to_fights,
)


def _source_row(**overrides: object) -> dict[str, object]:
    row = {
        "R_fighter": "Alpha One",
        "B_fighter": "Beta Two",
        "date": "2024-05-04",
        "R_odds": -200.0,
        "B_odds": 170.0,
        "Winner": "Red",
        "r_ko_odds": 300.0,
        "b_ko_odds": 400.0,
        "r_sub_odds": 500.0,
        "b_sub_odds": 600.0,
        "r_dec_odds": 150.0,
        "b_dec_odds": 250.0,
    }
    row.update(overrides)
    return row


class SecondaryOddsTests(unittest.TestCase):
    def test_american_conversion_rejects_the_impossible_range(self):
        self.assertAlmostEqual(american_to_decimal(-200), 1.5)
        self.assertAlmostEqual(american_to_decimal(150), 2.5)
        self.assertTrue(math.isnan(american_to_decimal(50)))
        self.assertTrue(math.isnan(american_to_decimal("")))

    def test_ambiguous_repeated_pairings_are_dropped(self):
        source = pd.DataFrame([_source_row(), _source_row(R_odds=-150.0, B_odds=130.0)])
        self.assertEqual(len(canonicalise_secondary_odds(source)), 0)

    def test_price_follows_the_fighter_when_corners_are_swapped(self):
        source = pd.DataFrame([_source_row(R_fighter="Beta Two", B_fighter="Alpha One")])
        secondary = canonicalise_secondary_odds(source)
        fights = pd.DataFrame(
            [
                {
                    "fight_id": "f1",
                    "fight_key": fight_key("2024-05-04", "Alpha One", "Beta Two"),
                    "event_date": pd.Timestamp("2024-05-04"),
                    "fighter_1": "Alpha One",
                    "fighter_2": "Beta Two",
                }
            ]
        )
        matched, report = match_secondary_to_fights(secondary, fights)
        self.assertEqual(report["matched_fights"], 1)
        # Beta Two was the red corner here and carried the -200 favourite price,
        # so fighter_1 (Alpha One) must receive the +170 underdog price.
        self.assertAlmostEqual(matched.iloc[0]["odds_fighter_1"], 2.7)
        self.assertAlmostEqual(matched.iloc[0]["odds_fighter_2"], 1.5)

    def test_identical_prices_are_reported_as_a_shared_origin(self):
        primary = pd.DataFrame(
            [
                {
                    "fight_id": f"f{index}",
                    "event_date": pd.Timestamp("2020-01-01"),
                    "market_p1": 0.6,
                    "overround": 1.05,
                    "temporal_quality": "legacy_unverified",
                    "source": "zewnetrzne",
                }
                for index in range(10)
            ]
        )
        secondary = primary[["fight_id", "market_p1", "overround"]].copy()
        _, report = cross_check(primary, secondary)
        self.assertEqual(report["legacy_bit_identical_rate"], 1.0)
        self.assertTrue(report["independence_verdict"].startswith("sources_are_not_independent"))

    def test_genuinely_different_prices_are_reported_as_independent(self):
        rng = np.random.default_rng(0)
        primary = pd.DataFrame(
            [
                {
                    "fight_id": f"f{index}",
                    "event_date": pd.Timestamp("2020-01-01"),
                    "market_p1": 0.6,
                    "overround": 1.05,
                    "temporal_quality": "legacy_unverified",
                    "source": "zewnetrzne",
                }
                for index in range(50)
            ]
        )
        secondary = primary[["fight_id", "market_p1", "overround"]].copy()
        secondary["market_p1"] += rng.normal(0, 0.02, len(secondary))
        _, report = cross_check(primary, secondary)
        self.assertLess(report["legacy_bit_identical_rate"], 0.5)
        self.assertEqual(report["independence_verdict"], "sources_are_independent")


if __name__ == "__main__":
    unittest.main()
