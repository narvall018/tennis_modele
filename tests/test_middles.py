from __future__ import annotations

import unittest

import pandas as pd

from src.backtesting.middles import Middle, classify, scan_event

NOW = pd.Timestamp("2026-09-07T12:00:00Z")


def _event(quotes, home="A", away="B"):
    """quotes: list of (book, side, line, price, minutes_stale)."""
    books: dict[str, list] = {}
    for book, side, line, price, stale in quotes:
        books.setdefault(book, []).append((side, line, price, stale))
    return {
        "home_team": home, "away_team": away, "commence_time": "2026-09-08T12:00:00Z",
        "bookmakers": [
            {
                "key": book,
                "last_update": (NOW - pd.Timedelta(minutes=entries[0][3])).isoformat(),
                "markets": [{
                    "key": "totals",
                    "outcomes": [
                        {"name": side.capitalize(), "point": line, "price": price}
                        for side, line, price, _ in entries
                    ],
                }],
            }
            for book, entries in books.items()
        ],
    }


class DetectionTests(unittest.TestCase):
    def test_a_real_gap_between_lines_is_found(self):
        event = _event([
            ("pinnacle", "over", 38.5, 1.92, 0.5),
            ("onexbet", "under", 39.5, 1.92, 0.5),
        ])
        found = scan_event(event, "tennis", "totals", NOW)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].width, 1.0)

    def test_an_inverted_pair_is_not_a_middle(self):
        """Over 39.5 against Under 38.5 is a gap the wrong way: both can lose."""
        event = _event([
            ("pinnacle", "over", 39.5, 1.92, 0.5),
            ("onexbet", "under", 38.5, 1.92, 0.5),
        ])
        self.assertEqual(scan_event(event, "tennis", "totals", NOW), [])

    def test_a_gap_containing_no_whole_result_is_rejected(self):
        """Over 2.5 / Under 2.75 is arithmetically a gap and practically nothing."""
        event = _event([
            ("pinnacle", "over", 2.5, 1.92, 0.5),
            ("onexbet", "under", 2.75, 1.92, 0.5),
        ])
        self.assertEqual(scan_event(event, "mma", "totals", NOW), [])

    def test_the_two_legs_must_come_from_different_books(self):
        event = _event([
            ("pinnacle", "over", 38.5, 1.92, 0.5),
            ("pinnacle", "under", 39.5, 1.92, 0.5),
        ])
        self.assertEqual(scan_event(event, "tennis", "totals", NOW), [])


class EconomicsTests(unittest.TestCase):
    def _middle(self, over_price=1.92, under_price=1.92) -> Middle:
        return Middle(
            sport="tennis", event="A vs B", commence_time="", market="totals",
            over_line=38.5, over_price=over_price, over_book="pinnacle",
            over_staleness=0.5,
            under_line=39.5, under_price=under_price, under_book="onexbet",
            under_staleness=0.5,
        )

    def test_missing_the_middle_costs_only_the_pair_margin(self):
        """The downside is bounded: one leg still pays when the middle misses."""
        cost = self._middle().cost_if_missed
        self.assertAlmostEqual(cost, -0.04, places=3)
        self.assertGreater(cost, -0.10)

    def test_a_worse_priced_pair_costs_more_to_miss(self):
        tight = self._middle(1.92, 1.92).cost_if_missed
        wide = self._middle(1.60, 1.60).cost_if_missed
        self.assertLess(wide, tight)

    def test_hitting_the_middle_pays_both_legs(self):
        middle = self._middle()
        self.assertAlmostEqual(middle.gain_if_hit, 0.92, places=3)

    def test_the_break_even_probability_matches_the_reported_threshold(self):
        """A 1.92/1.92 middle needs the exact result 4.17% of the time."""
        self.assertAlmostEqual(self._middle().break_even_probability, 0.0417, places=3)

    def test_a_wider_priced_pair_demands_a_likelier_middle(self):
        tight = self._middle(1.92, 1.92).break_even_probability
        wide = self._middle(1.60, 1.60).break_even_probability
        self.assertGreater(wide, tight)


class GuardTests(unittest.TestCase):
    def test_each_guard_is_reported_separately(self):
        event = _event([
            ("tab", "over", 38.5, 1.92, 0.5),
            ("smarkets", "under", 39.5, 1.92, 20.0),
        ])
        frame = classify(scan_event(event, "tennis", "totals", NOW))
        row = frame.iloc[0]
        self.assertFalse(bool(row["assez_frais"]))
        self.assertFalse(bool(row["sans_exchange"]))
        self.assertFalse(bool(row["books_accessibles"]))
        self.assertFalse(bool(row["exploitable"]))

    def test_a_clean_middle_at_reachable_books_is_exploitable(self):
        event = _event([
            ("pinnacle", "over", 38.5, 1.92, 0.5),
            ("onexbet", "under", 39.5, 1.92, 1.0),
        ])
        frame = classify(scan_event(event, "tennis", "totals", NOW))
        self.assertTrue(bool(frame.iloc[0]["exploitable"]))

    def test_the_widest_middle_wins_when_several_exist(self):
        from src.backtesting.middles import scan

        event = _event([
            ("pinnacle", "over", 38.5, 1.92, 0.5),
            ("onexbet", "under", 39.5, 1.92, 0.5),
            ("coolbet", "under", 41.5, 1.85, 0.5),
        ])
        found = scan([event], "tennis", "totals", NOW)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].width, 3.0)


if __name__ == "__main__":
    unittest.main()
