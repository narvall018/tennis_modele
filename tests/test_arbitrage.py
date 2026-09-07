from __future__ import annotations

import unittest

import pandas as pd

from src.backtesting.arbitrage import (
    EXCHANGES,
    REACHABLE_FROM_FRANCE,
    classify,
    scan_event,
)

NOW = pd.Timestamp("2026-09-07T12:00:00Z")


def _event(quotes, home="A", away="B"):
    """quotes: list of (bookmaker, outcome, price, minutes_stale)."""
    books: dict[str, list] = {}
    for bookmaker, outcome, price, stale in quotes:
        books.setdefault(bookmaker, []).append((outcome, price, stale))
    return {
        "home_team": home, "away_team": away, "commence_time": "2026-09-08T18:00:00Z",
        "bookmakers": [
            {
                "key": bookmaker,
                "last_update": (NOW - pd.Timedelta(minutes=entries[0][2])).isoformat(),
                "markets": [{
                    "key": "h2h",
                    "outcomes": [
                        {"name": outcome, "price": price} for outcome, price, _ in entries
                    ],
                }],
            }
            for bookmaker, entries in books.items()
        ],
    }


class DetectionTests(unittest.TestCase):
    def test_a_genuine_price_gap_is_detected(self):
        event = _event([
            ("unibet", "A", 2.10, 0.5),
            ("bwin", "B", 2.10, 0.5),
        ])
        opportunity = scan_event(event, "test", NOW)
        self.assertIsNotNone(opportunity)
        self.assertAlmostEqual(opportunity.overround, 1 / 2.10 * 2)
        self.assertGreater(opportunity.guaranteed_return, 0.0)

    def test_a_normal_market_yields_nothing(self):
        event = _event([("unibet", "A", 1.90, 0.5), ("bwin", "B", 1.90, 0.5)])
        self.assertIsNone(scan_event(event, "test", NOW))

    def test_the_best_price_per_outcome_is_taken_across_books(self):
        event = _event([
            ("unibet", "A", 1.80, 0.5), ("unibet", "B", 2.00, 0.5),
            ("bwin", "A", 2.10, 0.5), ("bwin", "B", 1.95, 0.5),
        ])
        opportunity = scan_event(event, "test", NOW)
        self.assertIsNotNone(opportunity)
        prices = {leg["outcome"]: leg["price"] for leg in opportunity.legs}
        self.assertEqual(prices, {"A": 2.10, "B": 2.00})

    def test_a_draw_leg_never_enters_a_two_way_market(self):
        event = _event([
            ("unibet", "A", 2.10, 0.5), ("bwin", "B", 2.10, 0.5),
            ("bwin", "Draw", 33.0, 0.5),
        ])
        opportunity = scan_event(event, "test", NOW)
        self.assertIsNotNone(opportunity)
        self.assertEqual(len(opportunity.legs), 2)


class StakeTests(unittest.TestCase):
    def test_stakes_return_the_same_amount_whichever_side_wins(self):
        event = _event([("unibet", "A", 2.10, 0.5), ("bwin", "B", 2.10, 0.5)])
        opportunity = scan_event(event, "test", NOW)
        stakes = opportunity.stakes(1000.0)
        payouts = [
            stakes[leg["outcome"]] * leg["price"] for leg in opportunity.legs
        ]
        self.assertAlmostEqual(payouts[0], payouts[1], places=1)
        self.assertGreater(payouts[0], sum(stakes.values()))


class GuardTests(unittest.TestCase):
    def test_a_stale_quote_is_flagged_not_silently_kept(self):
        """A 13-minute-old price against a fresh one is not a simultaneous pair."""
        event = _event([("unibet", "A", 2.10, 13.0), ("bwin", "B", 2.10, 0.5)])
        frame = classify([scan_event(event, "test", NOW)])
        self.assertFalse(bool(frame.iloc[0]["assez_frais"]))
        self.assertFalse(bool(frame.iloc[0]["exploitable"]))

    def test_an_exchange_leg_is_flagged(self):
        """An exchange price is an offer of unknown depth, not a quotable book."""
        self.assertIn("smarkets", EXCHANGES)
        event = _event([("smarkets", "A", 2.10, 0.5), ("bwin", "B", 2.10, 0.5)])
        frame = classify([scan_event(event, "test", NOW)])
        self.assertFalse(bool(frame.iloc[0]["sans_exchange"]))
        self.assertFalse(bool(frame.iloc[0]["exploitable"]))

    def test_an_unreachable_bookmaker_is_flagged(self):
        self.assertNotIn("tab", REACHABLE_FROM_FRANCE)
        event = _event([("tab", "A", 2.10, 0.5), ("bwin", "B", 2.10, 0.5)])
        frame = classify([scan_event(event, "test", NOW)])
        self.assertFalse(bool(frame.iloc[0]["books_accessibles"]))
        self.assertFalse(bool(frame.iloc[0]["exploitable"]))

    def test_only_a_clean_opportunity_is_called_exploitable(self):
        event = _event([("unibet", "A", 2.10, 0.5), ("bwin", "B", 2.10, 1.0)])
        frame = classify([scan_event(event, "test", NOW)])
        row = frame.iloc[0]
        self.assertTrue(bool(row["assez_frais"]))
        self.assertTrue(bool(row["sans_exchange"]))
        self.assertTrue(bool(row["books_accessibles"]))
        self.assertTrue(bool(row["exploitable"]))

    def test_the_guards_are_reported_alongside_the_naive_count(self):
        """A scan that only showed survivors would hide how many it discarded."""
        events = [
            _event([("unibet", "A", 2.10, 0.5), ("bwin", "B", 2.10, 0.5)], "X", "Y"),
            _event([("tab", "A", 2.10, 0.5), ("smarkets", "B", 2.10, 20.0)], "W", "Z"),
        ]
        frame = classify([scan_event(event, "test", NOW) for event in events])
        self.assertEqual(len(frame), 2)
        self.assertEqual(int(frame["exploitable"].sum()), 1)


if __name__ == "__main__":
    unittest.main()
