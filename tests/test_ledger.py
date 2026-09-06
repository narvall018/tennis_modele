from __future__ import annotations

import sqlite3
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.app.ledger import (
    migrate_sports,
    parse_matchup,
    record_recommendation,
)

LEGACY_EVENTS = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sport TEXT NOT NULL CHECK (sport IN ('tennis', 'ufc')),
    title TEXT, participant_a TEXT NOT NULL, participant_b TEXT NOT NULL,
    event_datetime TEXT, odds_a REAL, odds_b REAL,
    predicted_prob_a REAL, predicted_prob_b REAL, stats_a TEXT, stats_b TEXT,
    status TEXT NOT NULL DEFAULT 'upcoming'
        CHECK (status IN ('upcoming', 'completed')),
    winner_side TEXT, created_by INTEGER,
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL)
"""
LEGACY_BETS = """
CREATE TABLE IF NOT EXISTS bets (
    id INTEGER PRIMARY KEY AUTOINCREMENT, user_id INTEGER NOT NULL,
    sport TEXT NOT NULL CHECK (sport IN ('tennis', 'ufc')),
    event_id INTEGER NOT NULL,
    pick_side TEXT NOT NULL CHECK (pick_side IN ('a', 'b')),
    odds REAL NOT NULL, stake REAL NOT NULL CHECK (stake > 0),
    model_probability REAL, edge REAL, ev REAL,
    status TEXT NOT NULL DEFAULT 'open' CHECK (status IN ('open', 'resolved')),
    result TEXT, profit REAL, source TEXT NOT NULL DEFAULT 'manual',
    notes TEXT, created_at TEXT NOT NULL, resolved_at TEXT)
"""


def _legacy_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, username TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'julien')")
    conn.execute(LEGACY_EVENTS)
    conn.execute(LEGACY_BETS)
    conn.execute(
        "INSERT INTO events (sport, participant_a, participant_b, created_at, updated_at)"
        " VALUES ('tennis', 'Ancien A', 'Ancien B', 'x', 'x')"
    )
    conn.commit()
    conn.close()


def _row(**overrides):
    row = {
        "sport": "UFC", "quand": "2026-09-19",
        "rencontre": "Giga Chikadze – Joanderson Brito",
        "pari": "Giga Chikadze", "cote_pari": 3.6, "p_pari": 0.382,
        "espérance": 0.374, "score": 0.374, "écart": 0.118,
    }
    row.update(overrides)
    return row


class MatchupParsingTests(unittest.TestCase):
    def test_the_football_division_suffix_is_stripped(self):
        self.assertEqual(
            parse_matchup("Valencia – Barcelona  (SP1)"), ("Valencia", "Barcelona")
        )

    def test_several_separators_are_accepted(self):
        for label in ("A – B", "A - B", "A vs B", "A — B"):
            self.assertEqual(parse_matchup(label), ("A", "B"), label)

    def test_an_unsplittable_label_yields_no_opponent(self):
        self.assertEqual(parse_matchup("Match mystère"), ("Match mystère", ""))


class MigrationTests(unittest.TestCase):
    def test_football_is_allowed_after_migration_and_data_survives(self):
        """A CHECK rebuild must not silently drop the existing ledger."""
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.db"
            _legacy_database(path)
            conn = sqlite3.connect(path)
            self.assertTrue(migrate_sports(conn))
            stored = conn.execute(
                "SELECT sql FROM sqlite_master WHERE name='events'"
            ).fetchone()[0]
            self.assertIn("football", stored)
            self.assertEqual(
                conn.execute("SELECT count(*) FROM events").fetchone()[0], 1
            )
            conn.close()

    def test_migrating_twice_is_a_no_op(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "ledger.db"
            _legacy_database(path)
            conn = sqlite3.connect(path)
            self.assertTrue(migrate_sports(conn))
            self.assertFalse(migrate_sports(conn))
            conn.close()


class RecordingTests(unittest.TestCase):
    def _prepared(self, directory: str) -> Path:
        path = Path(directory) / "ledger.db"
        _legacy_database(path)
        conn = sqlite3.connect(path)
        migrate_sports(conn)
        conn.close()
        return path

    def test_a_bet_is_written_with_the_price_it_was_taken_at(self):
        with TemporaryDirectory() as directory:
            path = self._prepared(directory)
            ok, message = record_recommendation(path, 1, _row(), 3.0)
            self.assertTrue(ok, message)
            conn = sqlite3.connect(path)
            sport, side, odds, stake, source = conn.execute(
                "SELECT sport, pick_side, odds, stake, source FROM bets"
            ).fetchone()
            conn.close()
            self.assertEqual((sport, side, odds, stake, source),
                             ("ufc", "a", 3.6, 3.0, "recommandation"))

    def test_recording_the_same_match_twice_reuses_the_event(self):
        """Two clicks must not create two fixtures for one match."""
        with TemporaryDirectory() as directory:
            path = self._prepared(directory)
            record_recommendation(path, 1, _row(), 3.0)
            record_recommendation(path, 1, _row(), 2.0)
            conn = sqlite3.connect(path)
            events = conn.execute(
                "SELECT count(*) FROM events WHERE sport='ufc'"
            ).fetchone()[0]
            bets = conn.execute("SELECT count(*) FROM bets").fetchone()[0]
            conn.close()
            self.assertEqual(events, 1)
            self.assertEqual(bets, 2)

    def test_a_draw_is_refused_rather_than_mangled(self):
        """The ledger holds two sides; a 1X2 draw has no honest home there."""
        with TemporaryDirectory() as directory:
            path = self._prepared(directory)
            ok, message = record_recommendation(
                path, 1, _row(sport="Football", rencontre="A – B", pari="Nul"), 2.0
            )
            self.assertFalse(ok)
            self.assertIn("nul", message.lower())

    def test_a_pick_matching_neither_participant_is_refused(self):
        with TemporaryDirectory() as directory:
            path = self._prepared(directory)
            ok, message = record_recommendation(path, 1, _row(pari="Quelqu'un"), 2.0)
            self.assertFalse(ok)
            self.assertIn("correspond", message)

    def test_a_zero_stake_writes_nothing(self):
        with TemporaryDirectory() as directory:
            path = self._prepared(directory)
            ok, _ = record_recommendation(path, 1, _row(), 0.0)
            self.assertFalse(ok)
            conn = sqlite3.connect(path)
            self.assertEqual(conn.execute("SELECT count(*) FROM bets").fetchone()[0], 0)
            conn.close()


if __name__ == "__main__":
    unittest.main()
