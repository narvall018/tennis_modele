from __future__ import annotations

import importlib.util
import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from src.backtesting.rigorous_strategy import ProtocolWindows, market_comparison


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# One year per window, matching the four years the fixture below emits.
FIXTURE_WINDOWS = ProtocolWindows(
    development=(2013,), tuning=(2017,), validation=(2020,), holdout=(2023,)
)


def _load_runner():
    """Import the WTA runner by path; scripts/ is not an importable package."""
    spec = importlib.util.spec_from_file_location(
        "run_wta_backtest", PROJECT_ROOT / "scripts" / "run_wta_backtest.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MarketComparisonTests(unittest.TestCase):
    def _predictions(self, model_probability: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
        size = len(labels)
        return pd.DataFrame(
            {
                "_year": np.repeat([2013, 2017, 2020, 2023], size // 4),
                "_status": "completed",
                "_label": labels,
                "model_probability_p1": model_probability,
                "Avg_1": 2.0,
                "Avg_2": 2.0,
            }
        )

    def test_a_better_forecast_shows_a_positive_gain(self):
        rng = np.random.default_rng(7)
        labels = rng.integers(0, 2, 4000)
        # The market is a coin flip here; the model leans the right way.
        model = np.where(labels == 1, 0.62, 0.38)
        report = market_comparison(
            self._predictions(model, labels), model_weight=1.0, windows=FIXTURE_WINDOWS
        )
        for period in ("development", "tuning", "validation", "holdout"):
            self.assertGreater(report[period]["blend_gain_vs_market"], 0.0)

    def test_a_worse_forecast_shows_a_negative_gain(self):
        rng = np.random.default_rng(7)
        labels = rng.integers(0, 2, 4000)
        model = np.where(labels == 1, 0.38, 0.62)
        report = market_comparison(
            self._predictions(model, labels), model_weight=1.0, windows=FIXTURE_WINDOWS
        )
        self.assertLess(report["holdout"]["blend_gain_vs_market"], 0.0)

    def test_small_periods_are_reported_rather_than_scored(self):
        labels = np.array([0, 1] * 20)
        frame = self._predictions(np.full(40, 0.5), labels)
        report = market_comparison(frame, model_weight=1.0, windows=FIXTURE_WINDOWS)
        self.assertTrue(report["holdout"]["insufficient_sample"])


class ProtocolFreezeTests(unittest.TestCase):
    def test_the_protocol_hash_is_stable_and_written_once(self):
        runner = _load_runner()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wta_protocol.json"
            first = runner.freeze(path)
            second = runner.freeze(path)
            self.assertEqual(first["protocol_sha256"], second["protocol_sha256"])
            self.assertEqual(first["frozen_at_utc"], second["frozen_at_utc"])
            self.assertFalse(first["holdout_opened"])

    def test_changing_the_protocol_requires_a_stated_reason(self):
        runner = _load_runner()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wta_protocol.json"
            runner.freeze(path)
            stored = json.loads(path.read_text(encoding="utf-8"))
            stored["protocol_sha256"] = "0" * 64
            path.write_text(json.dumps(stored), encoding="utf-8")
            with self.assertRaises(SystemExit):
                runner.freeze(path)
            amended = runner.freeze(path, amend_reason="fold trop court")
            self.assertEqual(len(amended["superseded"]), 1)
            self.assertEqual(amended["superseded"][0]["reason"], "fold trop court")

    def test_an_opened_holdout_can_never_be_re_frozen(self):
        runner = _load_runner()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "wta_protocol.json"
            runner.freeze(path)
            stored = json.loads(path.read_text(encoding="utf-8"))
            stored["protocol_sha256"] = "0" * 64
            stored["holdout_opened"] = True
            path.write_text(json.dumps(stored), encoding="utf-8")
            with self.assertRaises(SystemExit):
                runner.freeze(path, amend_reason="peu importe la raison")


if __name__ == "__main__":
    unittest.main()
