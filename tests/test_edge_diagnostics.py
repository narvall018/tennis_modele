from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtesting.edge_diagnostics import (
    LONGSHOT_CAP,
    break_even_gain,
    month_block_confidence,
    unconditional_positive_ev,
)


def _frame(odds_1: np.ndarray, odds_2: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "_status": "completed",
            "_label": labels,
            "Pinnacle_1": odds_1,
            "Pinnacle_2": odds_2,
        }
    )


class EdgeDiagnosticsTests(unittest.TestCase):
    def test_a_perfect_forecaster_is_paid_and_a_blind_one_is_not(self):
        rng = np.random.default_rng(3)
        labels = rng.integers(0, 2, 2000)
        frame = _frame(np.full(2000, 2.0), np.full(2000, 2.0), labels)

        omniscient = np.where(labels == 1, 0.99, 0.01)
        paid = unconditional_positive_ev(frame, omniscient, "pinnacle", haircut=0.0)
        self.assertGreater(paid["roi"], 0.9)

        blind = np.full(2000, 0.5)
        # A flat 0.5 against a fair 2.0/2.0 book gives zero expected value, so
        # nothing qualifies and no bet is invented.
        self.assertEqual(unconditional_positive_ev(frame, blind, "pinnacle")["n_bets"], 0)

    def test_the_longshot_cap_only_removes_long_prices(self):
        labels = np.array([1, 0] * 500)
        odds_1 = np.where(np.arange(1000) % 2 == 0, 2.0, 9.0)
        odds_2 = np.where(np.arange(1000) % 2 == 0, 2.0, 1.11)
        frame = _frame(odds_1, odds_2, labels)
        confident = np.full(1000, 0.9)
        uncapped = unconditional_positive_ev(frame, confident, "pinnacle")
        capped = unconditional_positive_ev(frame, confident, "pinnacle", max_odds=LONGSHOT_CAP)
        self.assertLess(capped["n_bets"], uncapped["n_bets"])
        self.assertIsNone(uncapped["max_odds"])
        self.assertEqual(capped["max_odds"], LONGSHOT_CAP)

    def test_the_haircut_can_only_reduce_a_return(self):
        rng = np.random.default_rng(11)
        labels = rng.integers(0, 2, 1500)
        # 1.90 both sides is a ~5% overround, inside the accepted validity band.
        frame = _frame(np.full(1500, 1.90), np.full(1500, 1.90), labels)
        blend = np.where(labels == 1, 0.7, 0.3)
        gross = unconditional_positive_ev(frame, blend, "pinnacle", haircut=0.0)["roi"]
        net = unconditional_positive_ev(frame, blend, "pinnacle", haircut=0.05)["roi"]
        self.assertLess(net, gross)

    def test_the_confidence_interval_brackets_a_known_return(self):
        rng = np.random.default_rng(5)
        returns = rng.normal(0.05, 1.0, 4000)
        months = np.repeat([f"2020-{month:02d}" for month in range(1, 41)], 100)
        low, high = month_block_confidence(returns, months)
        self.assertLess(low, returns.mean())
        self.assertGreater(high, returns.mean())

    def test_a_bigger_margin_demands_a_bigger_edge(self):
        self.assertEqual(break_even_gain(1.0), 0.0)
        self.assertLess(break_even_gain(1.02), break_even_gain(1.06))

    def test_diagnostics_never_return_a_chosen_best_cell(self):
        """The report is a table, not a recommendation.

        Guarding this in a test because the whole value of the diagnostic
        collapses the moment it starts naming a winner.
        """
        rng = np.random.default_rng(9)
        labels = rng.integers(0, 2, 1200)
        frame = _frame(np.full(1200, 2.1), np.full(1200, 2.1), labels)
        result = unconditional_positive_ev(frame, np.where(labels == 1, 0.6, 0.4), "pinnacle")
        for forbidden in ("best_source", "recommended", "selected_rule", "chosen"):
            self.assertNotIn(forbidden, result)


if __name__ == "__main__":
    unittest.main()
