from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtesting.market_audit import SEGMENTS, audit, cross_tour_consistency


def _frame(pin_1, pin_2, soft_1, soft_2, labels) -> pd.DataFrame:
    size = len(labels)
    return pd.DataFrame({
        "Date": pd.date_range("2015-01-01", periods=size, freq="D").astype(str),
        "Player_1": [f"A{index}" for index in range(size)],
        "Player_2": [f"B{index}" for index in range(size)],
        "Pinnacle_1": np.full(size, pin_1, dtype=float),
        "Pinnacle_2": np.full(size, pin_2, dtype=float),
        "B365_1": np.full(size, soft_1, dtype=float),
        "B365_2": np.full(size, soft_2, dtype=float),
        "Series": "ATP250",
        "Surface": "Hard",
        "Best of": 3,
        "Round": "1st Round",
        "Rank_1": 20,
        "Rank_2": 40,
        "Status": "completed",
        "_label": labels,
        "_year": pd.date_range("2015-01-01", periods=size, freq="D").year,
        "_month": pd.date_range("2015-01-01", periods=size, freq="D").to_period("M").astype(str),
    })


class MarketAuditTests(unittest.TestCase):
    def test_a_fairly_priced_market_shows_no_calibration_gap(self):
        rng = np.random.default_rng(0)
        labels = (rng.random(6000) < 0.65).astype(int)
        # Pinnacle 1.50/2.79 devigs to almost exactly 0.65 for P1.
        frame = _frame(1.50, 2.79, 1.50, 2.79, labels)
        report = audit(frame)
        cell = report["by_probability_decile"]["0.6-0.7"]["favourite"]
        self.assertLess(abs(cell["calibration_gap"]), 0.02)

    def test_an_underpriced_favourite_is_reported_as_profitable(self):
        rng = np.random.default_rng(1)
        # The favourite really wins 75% of the time but is priced near 65%.
        labels = (rng.random(6000) < 0.75).astype(int)
        frame = _frame(1.50, 2.79, 1.50, 2.79, labels)
        report = audit(frame)
        cell = report["by_probability_decile"]["0.6-0.7"]["favourite"]
        self.assertGreater(cell["calibration_gap"], 0.05)
        self.assertTrue(cell["profitable"])

    def test_the_hash_control_segment_is_always_present(self):
        """A partition that cannot carry signal, kept as a sanity baseline."""
        self.assertIn("hash_control", SEGMENTS)
        labels = np.array([1, 0] * 2000)
        frame = _frame(1.60, 2.50, 1.55, 2.40, labels)
        report = audit(frame)
        self.assertGreaterEqual(len(report["segments"]["hash_control"]), 2)

    def test_multiplicity_reports_how_many_cells_chance_would_flag(self):
        labels = np.array([1, 0] * 2000)
        frame = _frame(1.60, 2.50, 1.55, 2.40, labels)
        report = audit(frame)
        multiplicity = report["multiplicity"]
        self.assertGreater(multiplicity["cells_examined"], 0)
        self.assertAlmostEqual(
            multiplicity["expected_by_chance_at_90pct"],
            round(0.10 * multiplicity["cells_examined"], 1),
        )

    def test_a_cell_profitable_on_one_tour_only_is_called_lucky(self):
        """The guard that killed the WTA heavy-favourite cell."""
        reports = {
            "ATP_Pinnacle": {
                "by_probability_decile": {
                    "0.9-1.0": {
                        "favourite": {"n": 3383, "roi": -0.0033, "profitable": False},
                        "underdog": {"n": 3383, "roi": -0.20, "profitable": False},
                    }
                }
            },
            "WTA_Pinnacle": {
                "by_probability_decile": {
                    "0.9-1.0": {
                        "favourite": {"n": 1936, "roi": +0.0083, "profitable": True},
                        "underdog": {"n": 1936, "roi": -0.18, "profitable": False},
                    }
                }
            },
        }
        result = cross_tour_consistency(reports, "Pinnacle")
        cell = result["cells"]["0.9-1.0/favourite"]
        self.assertFalse(cell["confirmed_by_both_tours"])
        self.assertFalse(result["any_confirmed"])
        # Pooling the two tours must sit between them, near zero here.
        self.assertLess(abs(cell["pooled_roi"]), 0.002)

    def test_a_cell_profitable_on_both_tours_is_confirmed(self):
        reports = {
            "ATP_Pinnacle": {
                "by_probability_decile": {
                    "0.9-1.0": {
                        "favourite": {"n": 3000, "roi": 0.02, "profitable": True},
                        "underdog": {"n": 3000, "roi": -0.2, "profitable": False},
                    }
                }
            },
            "WTA_Pinnacle": {
                "by_probability_decile": {
                    "0.9-1.0": {
                        "favourite": {"n": 2000, "roi": 0.03, "profitable": True},
                        "underdog": {"n": 2000, "roi": -0.2, "profitable": False},
                    }
                }
            },
        }
        result = cross_tour_consistency(reports, "Pinnacle")
        self.assertTrue(result["any_confirmed"])


if __name__ == "__main__":
    unittest.main()
