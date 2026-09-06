from __future__ import annotations

import base64
import json
import unittest
from pathlib import Path
from unittest.mock import patch

from src.app.github_ledger import (
    GitHubConfig,
    append_bet,
    ensure_branch,
    load_config,
    read_ledger,
)

CONFIG = GitHubConfig(token="tok", repository="user/repo")


class FakeResponse:
    def __init__(self, status_code: int, payload=None, text: str = ""):
        self.status_code = status_code
        self._payload = payload
        self.text = text

    @property
    def ok(self) -> bool:
        return 200 <= self.status_code < 300

    def json(self):
        return self._payload


def _contents(bets) -> FakeResponse:
    encoded = base64.b64encode(
        json.dumps(bets).encode("utf-8")
    ).decode("ascii")
    return FakeResponse(200, {"content": encoded, "sha": "abc123"})


class ReadTests(unittest.TestCase):
    def test_a_missing_file_is_an_empty_ledger_not_an_error(self):
        with patch("src.app.github_ledger.requests.get", return_value=FakeResponse(404)):
            bets, sha, message = read_ledger(CONFIG)
        self.assertEqual(bets, [])
        self.assertIsNone(sha)
        self.assertIn("vide", message)

    def test_existing_bets_are_decoded(self):
        with patch("src.app.github_ledger.requests.get",
                   return_value=_contents([{"pari": "X"}])):
            bets, sha, _ = read_ledger(CONFIG)
        self.assertEqual(bets, [{"pari": "X"}])
        self.assertEqual(sha, "abc123")

    def test_unexpected_content_is_refused_rather_than_coerced(self):
        encoded = base64.b64encode(json.dumps({"pas": "une liste"}).encode()).decode()
        response = FakeResponse(200, {"content": encoded, "sha": "s"})
        with patch("src.app.github_ledger.requests.get", return_value=response):
            bets, _, message = read_ledger(CONFIG)
        self.assertEqual(bets, [])
        self.assertIn("liste", message)


class WriteTests(unittest.TestCase):
    def test_a_bet_is_appended_to_what_is_already_there(self):
        captured = {}

        def fake_put(url, headers=None, json=None, timeout=None):
            captured.update(json)
            return FakeResponse(200)

        with patch("src.app.github_ledger.requests.get",
                   return_value=_contents([{"pari": "ancien"}])), \
             patch("src.app.github_ledger.requests.put", side_effect=fake_put):
            ok, message = append_bet(CONFIG, {"sport": "UFC", "pari": "nouveau"})

        self.assertTrue(ok, message)
        written = json.loads(base64.b64decode(captured["content"]).decode())
        self.assertEqual(len(written), 2)
        self.assertEqual(written[0]["pari"], "ancien")
        self.assertEqual(written[1]["pari"], "nouveau")
        # The blob SHA must travel with the write, or a concurrent bet is lost.
        self.assertEqual(captured["sha"], "abc123")
        self.assertEqual(captured["branch"], "ledger")

    def test_a_conflict_is_retried_against_fresh_content(self):
        """Two writes racing must not silently drop one."""
        attempts = {"n": 0}

        def fake_put(url, headers=None, json=None, timeout=None):
            attempts["n"] += 1
            return FakeResponse(409) if attempts["n"] == 1 else FakeResponse(200)

        with patch("src.app.github_ledger.requests.get", return_value=_contents([])), \
             patch("src.app.github_ledger.requests.put", side_effect=fake_put):
            ok, _ = append_bet(CONFIG, {"pari": "X"})
        self.assertTrue(ok)
        self.assertEqual(attempts["n"], 2)

    def test_a_permission_failure_names_the_missing_scope(self):
        with patch("src.app.github_ledger.requests.get", return_value=_contents([])), \
             patch("src.app.github_ledger.requests.put", return_value=FakeResponse(403)):
            ok, message = append_bet(CONFIG, {"pari": "X"})
        self.assertFalse(ok)
        self.assertIn("Contents", message)

    def test_nothing_is_attempted_without_configuration(self):
        with patch("src.app.github_ledger.requests.put") as put:
            ok, message = append_bet(GitHubConfig(token="", repository=""), {"pari": "X"})
        self.assertFalse(ok)
        put.assert_not_called()
        self.assertIn("GITHUB_TOKEN", message)


class BranchTests(unittest.TestCase):
    def test_an_existing_branch_is_left_alone(self):
        with patch("src.app.github_ledger.requests.get", return_value=FakeResponse(200)), \
             patch("src.app.github_ledger.requests.post") as post:
            ok, _ = ensure_branch(CONFIG)
        self.assertTrue(ok)
        post.assert_not_called()

    def test_a_missing_branch_is_created_from_the_default_one(self):
        responses = [
            FakeResponse(404),                                    # ledger ref
            FakeResponse(200, {"default_branch": "main"}),        # repo
            FakeResponse(200, {"object": {"sha": "deadbeef"}}),   # main ref
        ]
        with patch("src.app.github_ledger.requests.get", side_effect=responses), \
             patch("src.app.github_ledger.requests.post",
                   return_value=FakeResponse(201)) as post:
            ok, message = ensure_branch(CONFIG)
        self.assertTrue(ok, message)
        self.assertEqual(post.call_args.kwargs["json"]["sha"], "deadbeef")


class ConfigTests(unittest.TestCase):
    def test_the_repository_is_detected_from_the_checkout(self):
        config, _ = load_config(Path(__file__).resolve().parents[1])
        self.assertIn("/", config.repository)

    def test_the_ledger_branch_is_not_the_deployed_one(self):
        """Writing to the deployed branch would restart the app on every bet."""
        self.assertEqual(CONFIG.branch, "ledger")
        self.assertNotIn(CONFIG.branch, ("main", "master"))


if __name__ == "__main__":
    unittest.main()
