from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.data.tennis_pipeline import (
    LEGACY_COLUMNS,
    _atomic_csv,
    add_stable_player_orientation,
    attach_odds,
    deterministic_orientation,
    normalize_rich_matches,
    transform_tennis_data_raw,
)


class TennisPipelineTests(unittest.TestCase):
    def test_gzip_publication_is_byte_reproducible(self):
        frame = pd.DataFrame({"value": [1, 2], "text": ["a", "b"]})
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv.gz"
            second = Path(directory) / "second.csv.gz"
            _atomic_csv(frame, first, gzip=True)
            _atomic_csv(frame, second, gzip=True)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_raw_transform_falls_back_when_pinnacle_is_nan(self):
        raw = pd.DataFrame(
            [
                {
                    "Tournament": "Test Open",
                    "Date": "2026-08-20",
                    "Series": "ATP250",
                    "Court": "Outdoor",
                    "Surface": "Hard",
                    "Round": "The Final",
                    "Best of": 3,
                    "Winner": "Winner A.",
                    "Loser": "Loser B.",
                    "Comment": "Completed",
                    "WRank": 10,
                    "LRank": 20,
                    "WPts": 3000,
                    "LPts": 1500,
                    "PSW": np.nan,
                    "PSL": np.nan,
                    "B365W": 1.70,
                    "B365L": 2.20,
                    "W1": 6,
                    "L1": 4,
                    "W2": 6,
                    "L2": 3,
                }
            ]
        )
        transformed = transform_tennis_data_raw(raw)
        row = transformed.iloc[0]
        winner_odd = row["Odd_1"] if row["Player_1"] == row["Winner"] else row["Odd_2"]
        loser_odd = row["Odd_2"] if row["Player_1"] == row["Winner"] else row["Odd_1"]
        self.assertEqual(winner_odd, 1.70)
        self.assertEqual(loser_odd, 2.20)

    def test_orientation_is_stable_when_source_order_changes(self):
        base = {
            "Tournament": "Test Open",
            "Date": "2026-08-20",
            "Series": "ATP250",
            "Court": "Outdoor",
            "Surface": "Hard",
            "Round": "The Final",
            "Best of": 3,
            "Winner": "Alpha A.",
        }
        first = pd.DataFrame(
            [{**base, "Player_1": "Alpha A.", "Player_2": "Beta B.", "Rank_1": 1, "Rank_2": 2, "Pts_1": 10, "Pts_2": 5, "Odd_1": 1.5, "Odd_2": 2.8, "Score": "6-4 6-3"}],
            columns=LEGACY_COLUMNS,
        )
        second = pd.DataFrame(
            [{**base, "Player_1": "Beta B.", "Player_2": "Alpha A.", "Rank_1": 2, "Rank_2": 1, "Pts_1": 5, "Pts_2": 10, "Odd_1": 2.8, "Odd_2": 1.5, "Score": "4-6 3-6"}],
            columns=LEGACY_COLUMNS,
        )
        left = deterministic_orientation(first).iloc[0]
        right = deterministic_orientation(second).iloc[0]
        pd.testing.assert_series_equal(left, right)

    def test_exact_match_attaches_odds_to_the_correct_players(self):
        rich_raw = pd.DataFrame(
            [
                {
                    "tourney_id": "2026-1",
                    "tourney_name": "Test Open",
                    "surface": "Hard",
                    "draw_size": 32,
                    "tourney_level": "A",
                    "indoor": "O",
                    "tourney_date": 20260820,
                    "match_num": 1,
                    "winner_id": "A1",
                    "winner_name": "Alice Alpha",
                    "winner_hand": "R",
                    "winner_ht": 180,
                    "winner_ioc": "FRA",
                    "winner_age": 25,
                    "winner_rank": 10,
                    "winner_rank_points": 3000,
                    "loser_id": "B1",
                    "loser_name": "Bob Beta",
                    "loser_hand": "L",
                    "loser_ht": 185,
                    "loser_ioc": "USA",
                    "loser_age": 26,
                    "loser_rank": 20,
                    "loser_rank_points": 1500,
                    "score": "6-4 6-3",
                    "best_of": 3,
                    "round": "F",
                    "minutes": 80,
                    "_source_file": "2026.csv",
                    "_source_updated_at": "2026-08-21T00:00:00Z",
                }
            ]
        )
        odds = pd.DataFrame(
            [
                {
                    "Tournament": "Test Open",
                    "Date": "2026-08-20",
                    "Series": "ATP250",
                    "Court": "Outdoor",
                    "Surface": "Hard",
                    "Round": "The Final",
                    "Best of": 3,
                    "Player_1": "Beta B.",
                    "Player_2": "Alpha A.",
                    "Winner": "Alpha A.",
                    "Rank_1": 20,
                    "Rank_2": 10,
                    "Pts_1": 1500,
                    "Pts_2": 3000,
                    "Odd_1": 2.20,
                    "Odd_2": 1.70,
                    "Score": "4-6 3-6",
                }
            ]
        )
        rich = normalize_rich_matches(rich_raw, today=date(2026, 9, 1))
        result = add_stable_player_orientation(attach_odds(rich, odds)).iloc[0]
        self.assertEqual(result["winner_odds"], 1.70)
        self.assertEqual(result["loser_odds"], 2.20)
        self.assertEqual(result["odds_match_confidence"], 0.99)


if __name__ == "__main__":
    unittest.main()


class DownloadMessageTests(unittest.TestCase):
    """HTTP statuses describe requests, not whether a season has been published."""

    def test_an_outage_says_to_retry_and_that_data_is_safe(self):
        from src.data.tennis_pipeline import _download_message
        message = _download_message("http://x/2026.xlsx", {503}, None)
        self.assertIn("indisponible", message)
        self.assertIn("conservées intactes", message)
        self.assertIn("relancer plus tard", message)

    def test_a_404_does_not_claim_the_season_is_unpublished(self):
        from src.data.tennis_pipeline import _download_message
        message = _download_message("http://x/2027.xlsx", {404}, None)
        self.assertIn("déplacée", message)
        self.assertIn("index officiel", message)
        self.assertNotIn("pas encore publiée", message)

    def test_access_errors_are_not_reported_as_missing_files(self):
        from src.data.tennis_pipeline import _download_message
        for status in [401, 403, 429]:
            message = _download_message('https://stats.example/data.csv', {status}, None)
            self.assertIn('Accès refusé ou limité', message)
            self.assertIn('stats.example', message)
            self.assertNotIn('tennis-data.co.uk', message)

    def test_a_mixed_or_unknown_failure_keeps_the_raw_error(self):
        from src.data.tennis_pipeline import _download_message
        message = _download_message("http://x/a.xlsx", {404, 503}, RuntimeError("boum"))
        self.assertIn("boum", message)


class WorkbookLinksTests(unittest.TestCase):
    def test_moved_links_distinguish_tours_and_excel_extensions(self):
        from src.data.tennis_pipeline import _tennis_data_workbook_links
        html = b'''<a href="new-prefix/2026/2026.xlsx">2026</a>
                   <a href="new-prefix/2026w/2026.xlsx">2026 WTA</a>
                   <a href="new-prefix/2000/2000.xls">2000</a>
                   <a href="https://other.example/2025/2025.xlsx">outside</a>
                   <a href="javascript:/2024/2024.xlsx">invalid</a>'''
        atp = _tennis_data_workbook_links(html, 'atp')
        wta = _tennis_data_workbook_links(html, 'wta')
        self.assertEqual(set(atp), {2000, 2026})
        self.assertEqual(set(wta), {2026})
        self.assertTrue(atp[2026].endswith('/new-prefix/2026/2026.xlsx'))
        self.assertTrue(wta[2026].endswith('/new-prefix/2026w/2026.xlsx'))
        self.assertTrue(atp[2000].endswith('.xls'))

    def test_conflicting_links_fail_closed(self):
        from src.data.tennis_pipeline import _tennis_data_workbook_links, DataQualityError
        with self.assertRaises(DataQualityError):
            _tennis_data_workbook_links(b'<a href="a/2026/2026.xlsx">a</a><a href="b/2026/2026.xlsx">b</a>', 'atp')

    def test_fetch_uses_advertised_link_and_keeps_provenance(self):
        import io
        from unittest.mock import patch
        from src.data.tennis_pipeline import fetch_odds_snapshot, TENNIS_DATA_INDEX_URL
        excel = io.BytesIO()
        pd.DataFrame({'Date': ['2026-09-13']}).to_excel(excel, index=False)
        official = 'http://www.tennis-data.co.uk/moved/2026/2026.xlsx'
        def response(url):
            if url == TENNIS_DATA_INDEX_URL:
                return b'<a href="moved/2026/2026.xlsx">2026</a>'
            self.assertEqual(url, official)
            return excel.getvalue()
        with patch('src.data.tennis_pipeline._http_bytes', side_effect=response):
            frame, _ = fetch_odds_snapshot(2026, 2026)
        self.assertEqual(frame.iloc[0]['_source_url'], official)

    def test_missing_link_does_not_try_a_guessed_url(self):
        from unittest.mock import patch
        from src.data.tennis_pipeline import fetch_odds_snapshot, DataQualityError
        with patch('src.data.tennis_pipeline._http_bytes', return_value=b'<html></html>') as fetch:
            with self.assertRaises(DataQualityError):
                fetch_odds_snapshot(2026, 2026)
            self.assertEqual(fetch.call_count, 1)


class DownloadDeadlineTests(unittest.TestCase):
    """Attendre douze minutes pour apprendre qu'un site est en panne est un défaut.

    Une reprise n'a d'intérêt que si elle reste moins coûteuse que relancer la
    commande soi-même; sans borne globale, un hôte muet consomme le délai
    d'attente à chaque essai.
    """

    def test_the_total_budget_is_bounded(self):
        from src.data import tennis_pipeline as pipeline
        worst = pipeline.REQUEST_TIMEOUT * 3 * 2
        self.assertGreater(worst, pipeline.DOWNLOAD_DEADLINE,
                           "sans borne le pire cas dépasserait la limite annoncée")
        self.assertLessEqual(pipeline.DOWNLOAD_DEADLINE, 120,
                             "au-delà de deux minutes, mieux vaut rendre la main")

    def test_a_silent_host_gives_up_within_the_deadline(self):
        import time
        from unittest import mock
        from src.data import tennis_pipeline as pipeline

        def never_answers(*_args, **kwargs):
            time.sleep(0.2)
            raise TimeoutError("pas de réponse")

        started = time.monotonic()
        with mock.patch.object(pipeline.urllib.request, "urlopen", never_answers):
            with self.assertRaises(pipeline.DataQualityError) as raised:
                pipeline._http_bytes("http://www.tennis-data.co.uk/x.xlsx",
                                     timeout=1, attempts=3, deadline=0.5)
        self.assertLess(time.monotonic() - started, 5.0)
        self.assertIn("Abandon après", str(raised.exception))

    def test_the_deadline_is_actually_respected(self):
        """Un hôte muet ne doit pas faire dépasser la borne d'un délai entier."""
        import time
        from unittest import mock
        from src.data import tennis_pipeline as pipeline

        def never_answers(request, timeout=None, **kwargs):
            time.sleep(timeout if timeout else 1.0)
            raise TimeoutError("timed out")

        started = time.monotonic()
        with mock.patch.object(pipeline.urllib.request, "urlopen", never_answers):
            with self.assertRaises(pipeline.DataQualityError):
                pipeline._http_bytes("http://www.tennis-data.co.uk/x.xlsx",
                                     timeout=10, attempts=3, deadline=1.0)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 2.5,
                        f"borne de 1 s dépassée: {elapsed:.1f} s écoulées")

    def test_the_message_reports_how_long_it_waited(self):
        from src.data.tennis_pipeline import _download_message
        message = _download_message("http://x/a.xlsx", {503}, None, elapsed=87.4)
        self.assertIn("Abandon après 87 s", message)
        self.assertIn("conservées intactes", message)
