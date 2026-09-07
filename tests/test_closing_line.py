from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

from src.backtesting.closing_line import (
    TakenPrice,
    clv_frame,
    clv_summary,
    load_log,
    record_taken,
    settle_with_closing,
)


def _taken(key="m1", outcome="A", price=2.10, commence=None) -> TakenPrice:
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    return TakenPrice(
        key=key, sport="tennis", event="A vs B", outcome=outcome,
        bookmaker="unibet", price=price, reference_probability=0.5,
        taken_at_utc=past, commence_time=commence or past,
    )


class RecordingTests(unittest.TestCase):
    def test_a_decision_is_stored_with_its_price(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            ok, _ = record_taken(root, _taken())
            self.assertTrue(ok)
            entries = load_log(root)
            self.assertEqual(len(entries), 1)
            self.assertEqual(entries[0]["taken_price"], 2.10)
            self.assertIsNone(entries[0]["closing_price"])

    def test_the_same_outcome_is_never_logged_twice(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            record_taken(root, _taken())
            ok, message = record_taken(root, _taken())
            self.assertFalse(ok)
            self.assertIn("déjà", message)
            self.assertEqual(len(load_log(root)), 1)

    def test_a_corrupt_log_reads_as_empty_rather_than_raising(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "models" / "closing_line_log.json"
            path.parent.mkdir(parents=True)
            path.write_text("{pas du json", encoding="utf-8")
            self.assertEqual(load_log(root), [])


class SettlementTests(unittest.TestCase):
    def test_a_match_not_yet_started_cannot_have_a_closing_price(self):
        """A price taken before the close is not a closing price."""
        future = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            record_taken(root, _taken(commence=future))
            updated, _ = settle_with_closing(root, {"m1|A": (2.00, 0.5)})
            self.assertEqual(updated, 0)

    def test_a_started_match_is_closed_once(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            record_taken(root, _taken())
            self.assertEqual(settle_with_closing(root, {"m1|A": (2.00, 0.5)})[0], 1)
            self.assertEqual(settle_with_closing(root, {"m1|A": (1.90, 0.5)})[0], 0)
            self.assertEqual(load_log(root)[0]["closing_price"], 2.00)


class SummaryTests(unittest.TestCase):
    def _log(self, root: Path, pairs):
        for index, (taken, closing) in enumerate(pairs):
            record_taken(root, _taken(key=f"m{index}", price=taken))
            settle_with_closing(root, {f"m{index}|A": (closing, 1 / closing)})

    def test_beating_the_close_shows_positive_clv(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._log(root, [(2.10, 2.00)] * 40)
            frame = clv_frame(root)
            self.assertTrue(bool(frame["a_battu_la_cloture"].all()))
            summary = clv_summary(root)
            self.assertGreater(summary["clv_moyen"], 0.0)
            self.assertEqual(summary["taux_battu"], 1.0)

    def test_losing_to_the_close_rules_an_edge_out(self):
        """The point of CLV: a negative verdict arrives without 35 000 bets."""
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._log(root, [(1.95, 2.05)] * 40)
            summary = clv_summary(root)
            self.assertLess(summary["clv_moyen"], 0.0)
            self.assertIn("écarté", summary["verdict"])
            self.assertEqual(summary["settled"], 40)

    def test_a_small_sample_refuses_to_conclude(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            self._log(root, [(2.10, 2.00)] * 5)
            self.assertIn("trop petit", clv_summary(root)["verdict"])

    def test_an_empty_log_reports_no_verdict(self):
        with TemporaryDirectory() as directory:
            summary = clv_summary(Path(directory))
            self.assertEqual(summary["settled"], 0)
            self.assertIn("pas encore", summary["verdict"])

    def test_pending_entries_are_counted_separately(self):
        future = (datetime.now(timezone.utc) + timedelta(hours=3)).isoformat()
        with TemporaryDirectory() as directory:
            root = Path(directory)
            record_taken(root, _taken(key="m9", commence=future))
            self.assertEqual(clv_summary(root)["pending"], 1)


if __name__ == "__main__":
    unittest.main()
