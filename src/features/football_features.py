"""Pre-match descriptors for football, built in one chronological pass.

What a football table offers that tennis never did: for roughly 128 000 matches
the source records shots, shots on target, corners, fouls and cards. Shots on
target in particular are the standard proxy for underlying quality — a team that
out-shoots its opponents while losing tends to win later, and goals alone are too
sparse to say so.

The descriptors below are all *lagged team state*: a team's rating and rolling
averages are read before the current match updates them, and every match on the
same date is scored against the same pre-date state. Nothing about the match
being predicted enters its own row — the ``postmatch_`` prefix in the published
table exists to make that violation hard to commit by accident.

Home and away are kept distinct throughout. Football's home advantage is large
and team-specific, so collapsing the two would throw away real signal and make
the features asymmetric in the wrong way.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
import pandas as pd


ELO_START = 1500.0
ELO_K = 20.0
# Home advantage in Elo points, fixed in advance from the published home win rate
# (44.5%) rather than fitted, so it cannot absorb any of the model's own error.
ELO_HOME_ADVANTAGE = 60.0
ROLLING_WINDOW = 10

TEAM_METRICS = ("goals_for", "goals_against", "shots_for", "shots_against",
                "target_for", "target_against", "corners_for", "points")

FEATURE_COLUMNS = [
    "elo_diff",
    "elo_home", "elo_away",
    "home_matches_played", "away_matches_played",
    "goals_for_diff", "goals_against_diff",
    "shots_for_diff", "shots_against_diff",
    "target_for_diff", "target_against_diff",
    "corners_for_diff",
    "points_diff",
    "home_venue_points", "away_venue_points",
    "home_rest_days", "away_rest_days", "rest_diff",
    "division_rank",
]


class _TeamState:
    """Rolling, pre-match record of one team."""

    __slots__ = ("elo", "played", "last_date", "history", "venue_points")

    def __init__(self) -> None:
        self.elo = ELO_START
        self.played = 0
        self.last_date: pd.Timestamp | None = None
        self.history = {metric: deque(maxlen=ROLLING_WINDOW) for metric in TEAM_METRICS}
        # Points taken at home and away are tracked separately.
        self.venue_points = {"home": deque(maxlen=ROLLING_WINDOW),
                             "away": deque(maxlen=ROLLING_WINDOW)}

    def mean(self, metric: str) -> float:
        values = self.history[metric]
        return float(np.mean(values)) if values else np.nan

    def venue_mean(self, venue: str) -> float:
        values = self.venue_points[venue]
        return float(np.mean(values)) if values else np.nan

    def rest_days(self, current: pd.Timestamp) -> float:
        if self.last_date is None:
            return np.nan
        return float((current - self.last_date).days)

    def update(self, *, date: pd.Timestamp, venue: str, goals_for: float,
               goals_against: float, shots_for: float, shots_against: float,
               target_for: float, target_against: float, corners_for: float,
               points: float) -> None:
        self.played += 1
        self.last_date = date
        for metric, value in (
            ("goals_for", goals_for), ("goals_against", goals_against),
            ("shots_for", shots_for), ("shots_against", shots_against),
            ("target_for", target_for), ("target_against", target_against),
            ("corners_for", corners_for), ("points", points),
        ):
            if pd.notna(value):
                self.history[metric].append(float(value))
        self.venue_points[venue].append(float(points))


def _expected(elo_home: float, elo_away: float) -> float:
    return 1.0 / (1.0 + 10.0 ** ((elo_away - (elo_home + ELO_HOME_ADVANTAGE)) / 400.0))


def team_state_table(states: dict[str, _TeamState]) -> pd.DataFrame:
    """Freeze every team's end-of-history state so fixtures can be scored later.

    Rebuilding 191 000 matches to predict eight fixtures would be absurd, so the
    final state is exported once and reloaded. The stored values are exactly the
    ones the last match left behind — no recomputation, no drift.
    """
    rows = []
    for key, state in states.items():
        country, team = key.split("|", 1)
        row = {
            "country": country,
            "team": team,
            "elo": state.elo,
            "played": float(state.played),
            "last_date": state.last_date,
            "venue_points_home": state.venue_mean("home"),
            "venue_points_away": state.venue_mean("away"),
        }
        for metric in TEAM_METRICS:
            row[f"mean_{metric}"] = state.mean(metric)
        rows.append(row)
    return pd.DataFrame(rows)


def features_for_fixtures(
    fixtures: pd.DataFrame, states: pd.DataFrame
) -> pd.DataFrame:
    """Build the same descriptors for matches that have not been played.

    A fixture whose team is unknown to the state table keeps NaN descriptors
    rather than being silently given league-average ones: a newly promoted club
    the model has never seen is exactly the case where a made-up prior would be
    most misleading.
    """
    indexed = states.set_index(["country", "team"])
    rows: list[dict[str, Any]] = []
    for fixture in fixtures.itertuples(index=False):
        home_key = (fixture.country, fixture.home_team)
        away_key = (fixture.country, fixture.away_team)
        home = indexed.loc[home_key] if home_key in indexed.index else None
        away = indexed.loc[away_key] if away_key in indexed.index else None
        elo_home = float(home["elo"]) if home is not None else np.nan
        elo_away = float(away["elo"]) if away is not None else np.nan
        row: dict[str, Any] = {
            "match_id": fixture.match_id,
            "elo_home": elo_home,
            "elo_away": elo_away,
            "elo_diff": (elo_home + ELO_HOME_ADVANTAGE) - elo_away,
            "home_matches_played": float(home["played"]) if home is not None else np.nan,
            "away_matches_played": float(away["played"]) if away is not None else np.nan,
            "home_venue_points": float(home["venue_points_home"]) if home is not None else np.nan,
            "away_venue_points": float(away["venue_points_away"]) if away is not None else np.nan,
            "division_rank": float(fixture.division_rank),
            "home_known": home is not None,
            "away_known": away is not None,
        }
        match_date = pd.to_datetime(fixture.match_date)
        for side, state in (("home", home), ("away", away)):
            last = pd.to_datetime(state["last_date"]) if state is not None else pd.NaT
            row[f"{side}_rest_days"] = (
                float((match_date - last).days) if pd.notna(last) else np.nan
            )
        row["rest_diff"] = row["home_rest_days"] - row["away_rest_days"]
        for metric in TEAM_METRICS:
            home_value = float(home[f"mean_{metric}"]) if home is not None else np.nan
            away_value = float(away[f"mean_{metric}"]) if away is not None else np.nan
            row[f"{metric}_diff"] = home_value - away_value
        rows.append(row)
    return pd.DataFrame(rows)


def build_football_features(
    frame: pd.DataFrame, progress=None, return_states: bool = False
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """One chronological pass over every match; ratings read before they update.

    Teams are keyed by league *and* name, because promotion and relegation move a
    club between divisions and a name alone is not unique across countries.
    """
    ordered = frame.sort_values(["match_date", "league", "home_team"], kind="mergesort").reset_index(
        drop=True
    )
    ordered["match_date"] = pd.to_datetime(ordered["match_date"])
    states: dict[str, _TeamState] = {}

    def state(country: str, team: str) -> _TeamState:
        return states.setdefault(f"{country}|{team}", _TeamState())

    rows: list[dict[str, Any]] = []
    pending: list[tuple[int, _TeamState, _TeamState, Any]] = []
    current_date: pd.Timestamp | None = None

    def flush() -> None:
        for _, home, away, match in pending:
            home_points = 3.0 if match.result == "H" else 1.0 if match.result == "D" else 0.0
            away_points = 3.0 if match.result == "A" else 1.0 if match.result == "D" else 0.0
            expected = _expected(home.elo, away.elo)
            actual = 1.0 if match.result == "H" else 0.5 if match.result == "D" else 0.0
            delta = ELO_K * (actual - expected)
            home.elo += delta
            away.elo -= delta
            home.update(
                date=match.match_date, venue="home",
                goals_for=match.home_goals, goals_against=match.away_goals,
                shots_for=match.postmatch_home_shots, shots_against=match.postmatch_away_shots,
                target_for=match.postmatch_home_shots_on_target,
                target_against=match.postmatch_away_shots_on_target,
                corners_for=match.postmatch_home_corners, points=home_points,
            )
            away.update(
                date=match.match_date, venue="away",
                goals_for=match.away_goals, goals_against=match.home_goals,
                shots_for=match.postmatch_away_shots, shots_against=match.postmatch_home_shots,
                target_for=match.postmatch_away_shots_on_target,
                target_against=match.postmatch_home_shots_on_target,
                corners_for=match.postmatch_away_corners, points=away_points,
            )
        pending.clear()

    for index, match in enumerate(ordered.itertuples(index=False)):
        if progress and index % 20000 == 0:
            progress(f"Descripteurs: {index}/{len(ordered)} matchs")
        # All matches played on one date read the same pre-date state.
        if current_date is not None and match.match_date != current_date:
            flush()
        current_date = match.match_date

        home = state(match.country, match.home_team)
        away = state(match.country, match.away_team)
        row: dict[str, Any] = {
            "match_id": match.match_id,
            "elo_home": home.elo,
            "elo_away": away.elo,
            "elo_diff": (home.elo + ELO_HOME_ADVANTAGE) - away.elo,
            "home_matches_played": float(home.played),
            "away_matches_played": float(away.played),
            "home_venue_points": home.venue_mean("home"),
            "away_venue_points": away.venue_mean("away"),
            "home_rest_days": home.rest_days(match.match_date),
            "away_rest_days": away.rest_days(match.match_date),
            "division_rank": float(match.division_rank),
        }
        for metric in ("goals_for", "goals_against", "shots_for", "shots_against",
                       "target_for", "target_against", "corners_for", "points"):
            row[f"{metric}_diff"] = home.mean(metric) - away.mean(metric)
        row["rest_diff"] = row["home_rest_days"] - row["away_rest_days"]
        rows.append(row)
        pending.append((index, home, away, match))

    flush()
    features = pd.DataFrame(rows)
    keep = [
        "match_id", "match_date", "league", "country", "division_rank", "season_start",
        "home_team", "away_team", "result", "total_goals", "goal_difference",
    ]
    price_columns = [
        column for column in ordered.columns
        if column not in keep and not column.startswith("postmatch_")
    ]
    merged = ordered[keep + price_columns].merge(
        features.drop(columns="division_rank"), on="match_id", validate="one_to_one"
    )
    if merged["match_id"].duplicated().any():
        raise AssertionError("Un identifiant de match apparaît plusieurs fois")
    if return_states:
        return merged, team_state_table(states)
    return merged
