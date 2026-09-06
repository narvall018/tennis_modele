from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.backtesting.football_audit import asian_handicap_audit, asian_handicap_returns


def _returns(goal_difference, line, odds=2.0, side="home"):
    return asian_handicap_returns(
        np.array([goal_difference], dtype=float),
        np.array([line], dtype=float),
        np.array([odds], dtype=float),
        side,
        haircut=0.0,
    )[0]


class AsianHandicapSettlementTests(unittest.TestCase):
    def test_a_whole_line_can_be_refunded(self):
        """Home -1.0, home wins by exactly one: the stake comes back."""
        self.assertEqual(_returns(1, -1.0), 0.0)

    def test_a_half_line_never_refunds(self):
        self.assertEqual(_returns(1, -0.5), 1.0)
        self.assertEqual(_returns(0, -0.5), -1.0)

    def test_a_quarter_line_splits_the_stake(self):
        """Home -0.25 with a draw loses half; home +0.25 with a draw wins half."""
        self.assertEqual(_returns(0, -0.25), -0.5)
        self.assertEqual(_returns(0, +0.25), 0.5)

    def test_a_three_quarter_line_splits_on_the_other_side(self):
        # Home -0.75, home wins by one: half the stake pushes at -1, half wins at -0.5.
        self.assertEqual(_returns(1, -0.75), 0.5)
        # Winning by two clears both halves.
        self.assertEqual(_returns(2, -0.75), 1.0)

    def test_the_away_side_is_the_exact_mirror(self):
        for goal_difference in (-2, -1, 0, 1, 2):
            for line in (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5):
                home = _returns(goal_difference, line, side="home")
                away = _returns(goal_difference, line, side="away")
                # At a fair 2.0 both sides cannot win, and a push is shared.
                if home == 0.0:
                    self.assertEqual(away, 0.0)
                else:
                    self.assertAlmostEqual(home, -away * 1.0 if abs(home) == 1.0 else home)
                    self.assertNotEqual(np.sign(home), np.sign(away))

    def test_the_haircut_reduces_only_the_winning_part(self):
        gross = asian_handicap_returns(
            np.array([2.0]), np.array([-0.5]), np.array([2.0]), "home", haircut=0.0
        )[0]
        net = asian_handicap_returns(
            np.array([2.0]), np.array([-0.5]), np.array([2.0]), "home", haircut=0.10
        )[0]
        self.assertEqual(gross, 1.0)
        self.assertAlmostEqual(net, 0.9)
        # A loss is a full stake whatever the price.
        loss = asian_handicap_returns(
            np.array([-2.0]), np.array([-0.5]), np.array([2.0]), "home", haircut=0.10
        )[0]
        self.assertEqual(loss, -1.0)

    def test_off_grid_lines_are_rejected_as_transcription_errors(self):
        size = 800
        frame = pd.DataFrame({
            "match_date": pd.date_range("2020-01-01", periods=size, freq="D").astype(str),
            "goal_difference": np.tile([1.0, -1.0], size // 2),
            "PAHH": 1.95, "PAHA": 1.95,
            "AHh": np.where(np.arange(size) < 400, -0.5, -0.37),
        })
        block = asian_handicap_audit(frame, "pinnacle", "open")
        self.assertTrue(block["available"])
        self.assertEqual(block["matches"], 400)

    def test_a_fair_handicap_market_returns_about_minus_the_margin(self):
        rng = np.random.default_rng(0)
        size = 4000
        # A symmetric line on a symmetric outcome: the only drag is the margin.
        frame = pd.DataFrame({
            "match_date": pd.date_range("2020-01-01", periods=size, freq="D").astype(str),
            "goal_difference": rng.choice([-2.0, -1.0, 1.0, 2.0], size),
            "PAHH": 1.95, "PAHA": 1.95,
            "AHh": 0.0,
        })
        block = asian_handicap_audit(frame, "pinnacle", "open")
        for side in ("home", "away"):
            self.assertLess(block["sides"][side]["roi"], 0.0)
            self.assertGreater(block["sides"][side]["roi"], -0.10)


if __name__ == "__main__":
    unittest.main()
