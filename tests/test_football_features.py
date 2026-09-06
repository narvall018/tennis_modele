from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.features.football_features import (
    ELO_HOME_ADVANTAGE,
    ELO_START,
    FEATURE_COLUMNS,
    build_football_features,
)


def _match(match_id: str, date: str, home: str, away: str, home_goals: int, away_goals: int,
           country: str = "England", league: str = "E0", shots_home: float = 12.0,
           shots_away: float = 8.0, target_home: float = 5.0, target_away: float = 3.0) -> dict:
    result = "H" if home_goals > away_goals else "A" if home_goals < away_goals else "D"
    return {
        "match_id": match_id,
        "match_date": date,
        "league": league,
        "country": country,
        "division_rank": 1,
        "season_start": 2020,
        "home_team": home,
        "away_team": away,
        "home_goals": float(home_goals),
        "away_goals": float(away_goals),
        "total_goals": float(home_goals + away_goals),
        "goal_difference": float(home_goals - away_goals),
        "result": result,
        "postmatch_home_shots": shots_home,
        "postmatch_away_shots": shots_away,
        "postmatch_home_shots_on_target": target_home,
        "postmatch_away_shots_on_target": target_away,
        "postmatch_home_corners": 6.0,
        "postmatch_away_corners": 4.0,
        "PSCH": 2.0, "PSCD": 3.5, "PSCA": 4.0,
    }


class FootballFeatureTests(unittest.TestCase):
    def test_a_first_match_carries_no_record(self):
        built = build_football_features(pd.DataFrame([_match("m1", "2020-08-01", "A", "B", 2, 1)]))
        row = built.iloc[0]
        self.assertEqual(row["elo_home"], ELO_START)
        self.assertEqual(row["elo_away"], ELO_START)
        self.assertEqual(row["home_matches_played"], 0.0)
        self.assertTrue(np.isnan(row["target_for_diff"]))
        self.assertAlmostEqual(row["elo_diff"], ELO_HOME_ADVANTAGE)

    def test_the_current_match_never_enters_its_own_features(self):
        """The decisive leakage test.

        Team A wins its first match heavily. Its second row must describe the
        state *after* match one and *before* match two — one match played, and
        an Elo already above the start but not yet updated for match two.
        """
        built = build_football_features(pd.DataFrame([
            # C plays first so that a difference against it is defined at all.
            _match("m0", "2020-07-25", "C", "D", 1, 1),
            _match("m1", "2020-08-01", "A", "B", 3, 0),
            _match("m2", "2020-08-08", "A", "C", 0, 4),
        ])).set_index("match_id")
        second = built.loc["m2"]
        self.assertEqual(second["home_matches_played"], 1.0)
        self.assertEqual(second["away_matches_played"], 1.0)
        # The 0-4 loss in m2 must not have touched the rating m2 is described by.
        self.assertGreater(second["elo_home"], ELO_START)
        # A scored 3 in its only match, C scored 1 in its only match.
        self.assertAlmostEqual(second["goals_for_diff"], 2.0)

    def test_a_difference_is_undefined_while_one_side_has_no_history(self):
        built = build_football_features(pd.DataFrame([
            _match("m1", "2020-08-01", "A", "B", 3, 0),
            _match("m2", "2020-08-08", "A", "C", 0, 4),
        ])).set_index("match_id")
        # C has never played, so no rolling difference can be formed. Returning
        # NaN keeps that honest instead of inventing a league-average opponent.
        self.assertTrue(np.isnan(built.loc["m2"]["goals_for_diff"]))

    def test_matches_on_one_date_share_the_same_pre_date_state(self):
        built = build_football_features(pd.DataFrame([
            _match("m1", "2020-08-01", "A", "B", 3, 0),
            _match("m2", "2020-08-01", "A", "C", 3, 0),
        ])).set_index("match_id")
        self.assertEqual(built.loc["m1"]["elo_home"], built.loc["m2"]["elo_home"])
        self.assertEqual(built.loc["m1"]["home_matches_played"], 0.0)
        self.assertEqual(built.loc["m2"]["home_matches_played"], 0.0)

    def test_winning_raises_a_rating_and_losing_lowers_the_opponent(self):
        built = build_football_features(pd.DataFrame([
            _match("m1", "2020-08-01", "A", "B", 3, 0),
            _match("m2", "2020-08-08", "A", "B", 1, 1),
        ])).set_index("match_id")
        second = built.loc["m2"]
        self.assertGreater(second["elo_home"], ELO_START)
        self.assertLess(second["elo_away"], ELO_START)

    def test_teams_of_the_same_name_in_different_countries_stay_apart(self):
        built = build_football_features(pd.DataFrame([
            _match("m1", "2020-08-01", "Valencia", "B", 3, 0, country="Spain", league="SP1"),
            _match("m2", "2020-08-08", "Valencia", "C", 1, 0, country="England", league="E0"),
        ])).set_index("match_id")
        # The English "Valencia" is a different club and must start fresh.
        self.assertEqual(built.loc["m2"]["home_matches_played"], 0.0)

    def test_no_post_match_column_survives_into_the_feature_table(self):
        """A postmatch_ column reaching a model is the failure this guards."""
        built = build_football_features(pd.DataFrame([
            _match("m1", "2020-08-01", "A", "B", 2, 1),
            _match("m2", "2020-08-08", "A", "C", 1, 1),
        ]))
        leaked = [column for column in built.columns if column.startswith("postmatch_")]
        self.assertEqual(leaked, [])

    def test_every_declared_feature_is_produced(self):
        built = build_football_features(pd.DataFrame([
            _match("m1", "2020-08-01", "A", "B", 2, 1),
            _match("m2", "2020-08-08", "A", "C", 1, 1),
        ]))
        missing = sorted(set(FEATURE_COLUMNS) - set(built.columns))
        self.assertEqual(missing, [])

    def test_rest_days_measure_the_gap_since_the_previous_match(self):
        built = build_football_features(pd.DataFrame([
            _match("m1", "2020-08-01", "A", "B", 2, 1),
            _match("m2", "2020-08-08", "A", "C", 1, 1),
        ])).set_index("match_id")
        self.assertTrue(np.isnan(built.loc["m1"]["home_rest_days"]))
        self.assertEqual(built.loc["m2"]["home_rest_days"], 7.0)


if __name__ == "__main__":
    unittest.main()
