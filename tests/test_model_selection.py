from __future__ import annotations

import unittest

import numpy as np

from src.app.predictions import RELIABLE_HISTORY, recommendation_score
from src.models.selection import (
    candidate_factories,
    select_best_model,
    walk_forward_scores,
)


# A small, fast subset; breadth is exercised by test_candidates_are_fixed.
FAST = ["logistique_forte", "logistique_douce", "gradient_boosting"]


def _separable(size: int, seed: int = 0):
    """A signal a decent model must find, with periods to walk forward over."""
    rng = np.random.default_rng(seed)
    periods = np.repeat(np.arange(2015, 2015 + 8), size // 8)
    signal = rng.normal(size=len(periods))
    noise = rng.normal(size=len(periods))
    probability = 1.0 / (1.0 + np.exp(-1.5 * signal))
    labels = (rng.random(len(periods)) < probability).astype(int)
    features = np.column_stack([signal, noise])
    return features, labels, periods


class SelectionTests(unittest.TestCase):
    def test_candidates_are_fixed_and_include_calibrated_variants(self):
        factories = candidate_factories(multiclass=False)
        self.assertIn("logistique_forte", factories)
        self.assertTrue(any(name.endswith("_calibre") for name in factories))

    def test_a_fold_is_never_trained_on_its_own_period(self):
        """Walk-forward is the whole point; a leak here would flatter every score."""
        features, labels, periods = _separable(4000)
        seen: list[tuple] = []

        class Spy:
            def fit(self, matrix, target):
                seen.append((matrix.shape[0], target.shape[0]))
                self.classes_ = np.array([0, 1])
                return self

            def predict_proba(self, matrix):
                return np.column_stack([np.full(len(matrix), 0.5)] * 2)

        scores = walk_forward_scores(
            Spy, features, labels, periods, sorted(set(periods))[2:], [0, 1]
        )
        self.assertIsNotNone(scores)
        # Training sets grow monotonically: each fold adds the previous period.
        sizes = [size for size, _ in seen]
        self.assertEqual(sizes, sorted(sizes))
        self.assertLess(max(sizes), len(labels))

    def test_the_winner_beats_a_coin_flip_on_a_learnable_signal(self):
        features, labels, periods = _separable(4000)
        development = sorted(set(periods))[:6]
        evaluation = sorted(set(periods))[6:]
        result = select_best_model(features, labels, periods, development, evaluation,
                                   families=FAST)
        self.assertTrue(result.winner)
        self.assertEqual(len(result.comparison), len(FAST))
        # Ranked best-first, and clearly better than log(2) = 0.693.
        losses = [row["log_loss"] for row in result.comparison]
        self.assertEqual(losses, sorted(losses))
        self.assertLess(losses[0], 0.65)

    def test_the_reported_evaluation_uses_periods_never_ranked_on(self):
        features, labels, periods = _separable(4000)
        development = sorted(set(periods))[:6]
        evaluation = sorted(set(periods))[6:]
        result = select_best_model(features, labels, periods, development, evaluation,
                                   families=FAST)
        reported = result.evaluation["periods"]
        self.assertEqual(reported, [str(period) for period in evaluation])
        # The real guarantee: nothing the winner was ranked on is scored again.
        self.assertFalse(set(reported) & {str(period) for period in development})

    def test_pure_noise_lands_near_the_coin_flip(self):
        rng = np.random.default_rng(1)
        periods = np.repeat(np.arange(2015, 2023), 500)
        features = rng.normal(size=(len(periods), 3))
        labels = rng.integers(0, 2, len(periods))
        result = select_best_model(
            features, labels, periods, sorted(set(periods))[:6], sorted(set(periods))[6:],
            families=FAST,
        )
        self.assertGreater(result.comparison[0]["log_loss"], 0.66)


class RecommendationRankingTests(unittest.TestCase):
    def test_a_negative_expectation_never_ranks(self):
        self.assertEqual(recommendation_score(-0.02, 50, 50, True), 0.0)
        self.assertEqual(recommendation_score(0.0, 50, 50, True), 0.0)

    def test_an_unknown_team_never_ranks(self):
        """A promoted club the model has never seen is where the gap is widest."""
        self.assertEqual(recommendation_score(0.30, 50, 50, False), 0.0)

    def test_thin_history_is_discounted_not_trusted(self):
        full = recommendation_score(0.10, RELIABLE_HISTORY, RELIABLE_HISTORY, True)
        thin = recommendation_score(0.10, 2, RELIABLE_HISTORY, True)
        self.assertAlmostEqual(full, 0.10)
        self.assertLess(thin, full)
        self.assertAlmostEqual(thin, 0.10 * 2 / RELIABLE_HISTORY)

    def test_confidence_saturates_rather_than_rewarding_veterans(self):
        ten = recommendation_score(0.10, 10, 10, True)
        many = recommendation_score(0.10, 500, 500, True)
        self.assertAlmostEqual(ten, many)

    def test_the_weaker_side_sets_the_confidence(self):
        score = recommendation_score(0.10, 100, 5, True)
        self.assertAlmostEqual(score, 0.10 * 5 / RELIABLE_HISTORY)

    def test_missing_history_is_treated_as_no_history(self):
        self.assertEqual(recommendation_score(0.10, np.nan, 50, True), 0.0)


if __name__ == "__main__":
    unittest.main()
