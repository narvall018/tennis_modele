from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from rigorous.method_market import PROP_COLUMNS, market_calibration


BASE = Path(__file__).resolve().parents[1]


def _props(rows: int, dec_odds: float) -> pd.DataFrame:
    frame = pd.DataFrame({
        "fight_id": [f"fight{index}" for index in range(rows)],
        "event_date": pd.Timestamp("2020-01-01"),
        "f1_ko_odds": 6.0, "f2_ko_odds": 6.0,
        "f1_sub_odds": 9.0, "f2_sub_odds": 9.0,
        "f1_dec_odds": dec_odds, "f2_dec_odds": dec_odds,
    })
    inverse = 1.0 / frame[PROP_COLUMNS]
    frame["overround"] = inverse.sum(axis=1)
    for column in PROP_COLUMNS:
        frame[f"p_{column}"] = inverse[column] / frame["overround"]
    return frame


class MethodMarketTests(unittest.TestCase):
    def test_a_fairly_priced_market_shows_no_bias(self):
        rows = 1000
        props = _props(rows, dec_odds=4.0)
        share = float(props["p_f1_dec_odds"].iloc[0])
        # Make f1 win by decision at exactly the devigged rate the market implies.
        wins = int(round(share * rows))
        outcomes = pd.DataFrame({
            "fight_id": props["fight_id"],
            "y": 1.0,
            "method_category": ["dec"] * wins + ["ko"] * (rows - wins),
        })
        report = market_calibration(props, outcomes)
        self.assertAlmostEqual(report["by_outcome"]["f1_dec"]["bias_points"], 0.0, places=3)

    def test_an_underpriced_outcome_is_reported_as_positive_bias(self):
        rows = 1000
        props = _props(rows, dec_odds=4.0)
        outcomes = pd.DataFrame({
            "fight_id": props["fight_id"],
            "y": 1.0,
            "method_category": ["dec"] * 600 + ["ko"] * 400,
        })
        report = market_calibration(props, outcomes)
        self.assertGreater(report["by_outcome"]["f1_dec"]["bias_points"], 0.2)
        self.assertGreater(report["by_outcome"]["f1_dec"]["roi_backing_every_one"], 0.0)

    def test_the_published_analysis_records_a_verdict_and_its_limits(self):
        path = BASE / "data/rigorous/quality/method_market_analysis.json"
        report = json.loads(path.read_text(encoding="utf-8"))
        self.assertIn(report["verdict"], {"AVENUE_FERMEE", "A_APPROFONDIR"})
        self.assertGreater(report["overround"]["median_six_way"], 1.15)
        # The untimestamped-price caveat must never be dropped from the report.
        self.assertTrue(
            any("horodat" in limit for limit in report["limitations"]),
            "le rapport doit conserver l'avertissement sur les prix non horodatés",
        )


if __name__ == "__main__":
    unittest.main()
