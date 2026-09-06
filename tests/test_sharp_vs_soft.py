from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtesting.sharp_vs_soft import (
    analyse,
    evaluate_book,
    simultaneity_audit,
)


def _frame(pin_1, pin_2, soft_1, soft_2, labels, year=2020) -> pd.DataFrame:
    size = len(labels)
    return pd.DataFrame({
        "Pinnacle_1": np.full(size, pin_1, dtype=float),
        "Pinnacle_2": np.full(size, pin_2, dtype=float),
        "B365_1": np.full(size, soft_1, dtype=float),
        "B365_2": np.full(size, soft_2, dtype=float),
        "Avg_1": np.full(size, soft_1, dtype=float),
        "Avg_2": np.full(size, soft_2, dtype=float),
        "Max_1": np.full(size, soft_1, dtype=float),
        "Max_2": np.full(size, soft_2, dtype=float),
        "Status": "completed",
        "_label": labels,
        "_year": year,
        "_month": [f"{year}-{1 + index % 12:02d}" for index in range(size)],
    })


class SharpVsSoftTests(unittest.TestCase):
    def test_a_soft_book_generously_priced_on_the_true_side_pays(self):
        rng = np.random.default_rng(0)
        # Pinnacle says P1 wins 60% of the time, and it is right.
        labels = (rng.random(4000) < 0.6).astype(int)
        # The soft book offers 1.90 on P1, well above the fair 1/0.6 = 1.667.
        frame = _frame(1.63, 2.45, 1.90, 2.00, labels)
        result = evaluate_book(frame, ("B365_1", "B365_2"), threshold=0.0, haircut=0.0)
        self.assertGreater(result["n_bets"], 3000)
        self.assertGreater(result["roi"], 0.05)

    def test_the_inverted_control_loses_when_the_signal_is_real(self):
        rng = np.random.default_rng(1)
        labels = (rng.random(4000) < 0.6).astype(int)
        frame = _frame(1.63, 2.45, 1.90, 2.00, labels)
        main = evaluate_book(frame, ("B365_1", "B365_2"), threshold=0.0, haircut=0.0)
        control = evaluate_book(
            frame, ("B365_1", "B365_2"), threshold=0.0, haircut=0.0, contrarian=True
        )
        self.assertGreater(main["roi"], control["roi"])

    def test_the_haircut_can_only_reduce_a_return(self):
        rng = np.random.default_rng(2)
        labels = (rng.random(3000) < 0.6).astype(int)
        frame = _frame(1.63, 2.45, 1.90, 2.00, labels)
        gross = evaluate_book(frame, ("B365_1", "B365_2"), 0.0, haircut=0.0)["roi"]
        net = evaluate_book(frame, ("B365_1", "B365_2"), 0.0, haircut=0.05)["roi"]
        self.assertLess(net, gross)

    def test_a_composite_price_is_flagged_as_not_executable(self):
        """The guard that disqualified the `maximum` column.

        A pair of prices implying a risk-free arbitrage cannot have been quoted
        at the same instant, so a source showing them in bulk is a composite of
        different moments and must never be presented as a bettable result.
        """
        labels = np.array([1, 0] * 500)
        frame = _frame(1.90, 2.10, 2.20, 2.20, labels)  # 1/2.2 + 1/2.2 = 0.909
        audit = simultaneity_audit(frame)
        self.assertFalse(audit["market_maximum"]["executable_as_a_pair"])
        self.assertGreater(audit["market_maximum"]["implied_arbitrage_rate"], 0.9)
        self.assertTrue(audit["pinnacle"]["executable_as_a_pair"])

    def test_the_report_carries_the_executability_flag_next_to_every_book(self):
        labels = np.array([1, 0] * 500)
        frame = _frame(1.90, 2.10, 2.20, 2.20, labels)
        report = analyse(frame)
        for book in report["books"].values():
            self.assertIn("executable_as_a_pair", book)
        self.assertFalse(report["books"]["market_maximum"]["executable_as_a_pair"])

    def test_raising_the_threshold_never_increases_the_bet_count(self):
        rng = np.random.default_rng(3)
        labels = (rng.random(2000) < 0.55).astype(int)
        frame = _frame(1.80, 2.20, 2.00, 2.05, labels)
        counts = [
            evaluate_book(frame, ("B365_1", "B365_2"), threshold)["n_bets"]
            for threshold in (0.0, 0.02, 0.05)
        ]
        self.assertEqual(counts, sorted(counts, reverse=True))


if __name__ == "__main__":
    unittest.main()
