from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.elo_system import TOURNAMENT_WEIGHTS
from src.features.multi_tier_ratings import (
    CHALLENGER_SERIES,
    QUALIFYING_SERIES,
    TIER_WEIGHTS,
    build_rating_input,
    elo_probability,
    fit_ratings,
    register_tier_weights,
)


def _match(match_id: str, date: str, p1: str, p2: str, winner: str, *,
           level: str = "250", round_code: str = "R32", tournament: str = "Test Open") -> dict:
    return {
        "match_id": match_id,
        "match_date": date,
        "player_1_id": p1,
        "player_2_id": p2,
        "winner_id": winner,
        "tourney_name": tournament,
        "tourney_level": level,
        "surface": "Hard",
        "round": round_code,
        "match_status": "completed",
        "best_of": 3,
        "player_1_rank": 50,
        "player_2_rank": 80,
        "player_1_rank_points": 900,
        "player_2_rank_points": 600,
        "player_1_odds": 1.8,
        "player_2_odds": 2.1,
    }


class MultiTierRatingTests(unittest.TestCase):
    def test_lower_tiers_move_ratings_less_than_main_draws(self):
        register_tier_weights()
        self.assertEqual(TOURNAMENT_WEIGHTS[CHALLENGER_SERIES], TIER_WEIGHTS[CHALLENGER_SERIES])
        self.assertLess(TOURNAMENT_WEIGHTS[CHALLENGER_SERIES], TOURNAMENT_WEIGHTS["ATP250"])
        self.assertLess(TOURNAMENT_WEIGHTS[QUALIFYING_SERIES], TOURNAMENT_WEIGHTS[CHALLENGER_SERIES])

    def test_qualifying_rows_are_labelled_qualifying_whatever_their_level(self):
        main = pd.DataFrame([_match("m1", "2020-01-06", "A", "B", "A")])
        unpriced = pd.DataFrame([
            {**_match("q1", "2020-01-01", "A", "C", "A", level="250", round_code="Q1"),
             "segment": "qualifying"},
        ])
        stream = build_rating_input(main, unpriced)
        self.assertEqual(
            stream.loc[stream["match_id"].eq("q1"), "Series"].iloc[0], QUALIFYING_SERIES
        )

    def test_the_stream_is_chronological_and_free_of_duplicates(self):
        main = pd.DataFrame([
            _match("m2", "2020-03-01", "A", "B", "A"),
            _match("m1", "2020-01-01", "A", "C", "C"),
        ])
        unpriced = pd.DataFrame([
            {**_match("c1", "2020-02-01", "B", "C", "B", level="C"), "segment": "challenger"},
            # A duplicate identifier must not be counted twice into a rating.
            {**_match("m1", "2020-01-01", "A", "C", "C"), "segment": "challenger"},
        ])
        stream = build_rating_input(main, unpriced)
        self.assertEqual(list(stream["match_id"]), ["m1", "c1", "m2"])
        self.assertTrue(stream["Date"].is_monotonic_increasing)

    def test_a_challenger_result_changes_a_later_main_tour_rating(self):
        """The whole point: lower-tier form must reach the main draw."""
        main = pd.DataFrame([_match("m1", "2020-06-01", "A", "B", "A")])
        challengers = pd.DataFrame([
            {**_match(f"c{index}", f"2020-0{index+1}-01", "A", "Z", "A", level="C"),
             "segment": "challenger"}
            for index in range(1, 5)
        ])
        without = fit_ratings(build_rating_input(main, None))
        with_tiers = fit_ratings(build_rating_input(main, challengers))
        row_without = without[without["match_id"].eq("m1")].iloc[0]
        row_with = with_tiers[with_tiers["match_id"].eq("m1")].iloc[0]
        self.assertEqual(row_without["p1_matches"], 0)
        self.assertEqual(row_with["p1_matches"], 4)
        self.assertGreater(row_with["elo_p1"], row_without["elo_p1"])

    def test_unpriced_matches_never_enter_the_evaluated_population(self):
        """Challenger and qualifying rows carry no market and must stay ratings.

        Guarded in a test because letting one through would silently create a
        betting population for matches that were never quoted.
        """
        main = pd.DataFrame([_match("m1", "2020-06-01", "A", "B", "A")])
        unpriced = pd.DataFrame([
            {**_match("c1", "2020-01-01", "A", "Z", "A", level="C"), "segment": "challenger"},
            {**_match("q1", "2020-02-01", "A", "Y", "A", round_code="Q1"), "segment": "qualifying"},
        ])
        history = fit_ratings(build_rating_input(main, unpriced))
        evaluated = history[history["segment"].eq("main")]
        self.assertEqual(set(evaluated["match_id"]), {"m1"})
        self.assertEqual(set(history["segment"]), {"main", "challenger", "qualifying"})

    def test_identity_is_the_id_so_names_cannot_split_a_player(self):
        main = pd.DataFrame([_match("m1", "2020-06-01", "104925", "B", "104925")])
        unpriced = pd.DataFrame([
            {**_match("c1", "2020-01-01", "104925", "Z", "104925", level="C"),
             "segment": "challenger"},
        ])
        stream = build_rating_input(main, unpriced)
        self.assertEqual(set(stream["Player_1"]), {"104925"})
        history = fit_ratings(stream)
        self.assertEqual(history[history["match_id"].eq("m1")].iloc[0]["p1_matches"], 1)

    def test_a_stronger_rating_gives_a_higher_probability(self):
        history = pd.DataFrame({
            "elo_p1": [1600.0, 1400.0, 1500.0],
            "elo_p2": [1400.0, 1600.0, 1500.0],
            "surf_elo_p1": [1600.0, 1400.0, 1500.0],
            "surf_elo_p2": [1400.0, 1600.0, 1500.0],
        })
        probability = elo_probability(history)
        self.assertGreater(probability[0], 0.5)
        self.assertLess(probability[1], 0.5)
        self.assertAlmostEqual(probability[2], 0.5)
        self.assertTrue(np.all((probability > 0) & (probability < 1)))


if __name__ == "__main__":
    unittest.main()
