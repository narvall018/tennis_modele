"""Phase 4 ATP: serve/return, surface partial pooling and fixed diagnostics.

All historical rows are development-only. Player and tournament state is read at
the start of each calendar day, then updated after every row for that day has
been featurized. This deliberately sacrifices same-day information when match
order is uncertain rather than introducing accidental look-ahead.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SURFACES = ("Hard", "Clay", "Grass", "Carpet")
SURFACE_SERVE_PRIOR = {"Hard": 0.635, "Clay": 0.615, "Grass": 0.650, "Carpet": 0.650}
TEST_YEARS = tuple(range(2017, 2027))

MARKET_FEATURES = [
    "market_logit",
    "market_logit_x_hard",
    "market_logit_x_clay",
    "market_logit_x_grass",
]

STRUCTURAL_EXTRA = [
    "global_elo_diff",
    "surface_elo_diff",
    "form_10_logit_diff",
    "surface_form_logit_diff",
    "rank_log_advantage",
    "points_log_advantage",
    "age_diff",
    "peak_age_advantage",
    "height_diff",
    "left_handed_diff",
    "rest_diff",
    "matches_14d_diff",
]
STRUCTURAL_FEATURES = [*MARKET_FEATURES, *STRUCTURAL_EXTRA]

SERVE_EXTRA = [
    "serve_advantage",
    "serve_skill_logit_diff",
    "return_skill_logit_diff",
    "ace_rate_logit_diff",
    "double_fault_rate_logit_diff",
    "first_in_logit_diff",
    "first_won_logit_diff",
    "second_won_logit_diff",
    "break_save_logit_diff",
    "serve_sample_log_diff",
    "minutes_3d_diff",
    "minutes_7d_diff",
    "minutes_14d_diff",
    "retirement_risk_diff",
]
SERVE_FEATURES = [*STRUCTURAL_FEATURES, *SERVE_EXTRA]

SURFACE_INTERACTIONS_BASE = [
    "global_elo_diff",
    "surface_elo_diff",
    "serve_advantage",
    "height_diff",
    "minutes_7d_diff",
]
SURFACE_INTERACTIONS = [
    f"{feature}_x_{surface.lower()}"
    for feature in SURFACE_INTERACTIONS_BASE
    for surface in ("Hard", "Clay", "Grass")
]
PACE_INTERACTIONS = [
    "serve_advantage_x_court_pace",
    "height_diff_x_court_pace",
    "market_logit_x_court_pace",
]
SURFACE_FEATURES = [*SERVE_FEATURES, *SURFACE_INTERACTIONS, *PACE_INTERACTIONS]
MLP_INVARIANT_FEATURES = [
    "is_hard", "is_clay", "is_grass", "is_indoor", "best_of_5",
    "round_progress", "level_strength", "court_pace", "log_surface_samples",
]

CANDIDATE_FEATURES = {
    "market_surface_calibration": MARKET_FEATURES,
    "structural_market_residual": STRUCTURAL_FEATURES,
    "serve_return_residual": SERVE_FEATURES,
    "surface_partial_pooling": SURFACE_FEATURES,
}
CANDIDATE_ORDER = list(CANDIDATE_FEATURES)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def _normalise_text(value: object) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = text.encode("ascii", "ignore").decode("ascii").lower()
    return re.sub(r"[^a-z0-9]", "", text)


def _player_key(identifier: object, name: object) -> str:
    if pd.notna(identifier):
        text = str(identifier).strip()
        if text and text.lower() not in {"nan", "none", "-1"}:
            try:
                return f"id:{int(float(text))}"
            except ValueError:
                return f"id:{text}"
    return f"name:{_normalise_text(name)}"


def _safe_logit(value: float) -> float:
    clipped = float(np.clip(value, 1e-4, 1.0 - 1e-4))
    return math.log(clipped / (1.0 - clipped))


def _shrunk_rate(successes: float, attempts: float, prior: float, weight: float) -> float:
    successes = max(0.0, float(successes))
    attempts = max(0.0, float(attempts))
    return float((successes + prior * weight) / (attempts + weight))


@dataclass
class PlayerHistory:
    global_elo: float = 1500.0
    surface_elo: dict[str, float] = field(
        default_factory=lambda: {surface: 1500.0 for surface in SURFACES}
    )
    outcomes: deque = field(default_factory=lambda: deque(maxlen=80))
    stats: deque = field(default_factory=lambda: deque(maxlen=300))
    workload: deque = field(default_factory=lambda: deque(maxlen=100))
    retirement_exposures: int = 0
    retirements: int = 0

    def win_rate(self, ref_date: date, surface: str | None = None, last_n: int = 10) -> float:
        values = [
            won for played, played_surface, won in self.outcomes
            if played < ref_date and (surface is None or played_surface == surface)
        ][-last_n:]
        return _shrunk_rate(sum(values), len(values), 0.5, 5.0)

    def stat_totals(self, ref_date: date, surface: str, days: int = 730) -> dict[str, float]:
        cutoff = ref_date - timedelta(days=days)
        keys = (
            "serve_won", "serve_total", "return_won", "return_total", "aces", "double_faults",
            "first_in", "first_won", "second_won", "second_total", "bp_saved", "bp_faced",
        )
        totals = {key: 0.0 for key in keys}
        for played, played_surface, values in self.stats:
            if cutoff <= played < ref_date and played_surface == surface:
                for key in keys:
                    totals[key] += float(values.get(key, 0.0))
        return totals

    def workload_totals(self, ref_date: date, days: int) -> tuple[int, float]:
        cutoff = ref_date - timedelta(days=days)
        rows = [(played, minutes) for played, minutes in self.workload if cutoff <= played < ref_date]
        return len(rows), float(sum(minutes for _, minutes in rows))

    def days_rest(self, ref_date: date) -> float:
        past = [played for played, _ in self.workload if played < ref_date]
        return float(min(60, (ref_date - max(past)).days)) if past else 30.0

    def retirement_rate(self) -> float:
        return _shrunk_rate(self.retirements, self.retirement_exposures, 0.02, 50.0)


@dataclass
class CourtHistory:
    observations: deque = field(default_factory=lambda: deque(maxlen=1000))

    def serve_rate(self, ref_date: date, prior: float) -> tuple[float, float]:
        cutoff = ref_date - timedelta(days=1095)
        won = total = 0.0
        for played, successes, attempts in self.observations:
            if cutoff <= played < ref_date:
                won += successes
                total += attempts
        return _shrunk_rate(won, total, prior, 1000.0), total


def _surface_prior(
    surface: str,
    surface_totals: dict[str, list[float]],
) -> float:
    fixed = SURFACE_SERVE_PRIOR.get(surface, 0.625)
    won, total = surface_totals[surface]
    return _shrunk_rate(won, total, fixed, 10_000.0)


def _player_rates(history: PlayerHistory, ref_date: date, surface: str, surface_serve: float) -> dict[str, float]:
    totals = history.stat_totals(ref_date, surface)
    return {
        "serve": _shrunk_rate(totals["serve_won"], totals["serve_total"], surface_serve, 250.0),
        "return": _shrunk_rate(totals["return_won"], totals["return_total"], 1.0 - surface_serve, 250.0),
        "ace": _shrunk_rate(totals["aces"], totals["serve_total"], 0.07, 250.0),
        "df": _shrunk_rate(totals["double_faults"], totals["serve_total"], 0.035, 250.0),
        "first_in": _shrunk_rate(totals["first_in"], totals["serve_total"], 0.62, 250.0),
        "first_won": _shrunk_rate(totals["first_won"], totals["first_in"], 0.72, 180.0),
        "second_won": _shrunk_rate(totals["second_won"], totals["second_total"], 0.52, 180.0),
        "bp_save": _shrunk_rate(totals["bp_saved"], totals["bp_faced"], 0.60, 80.0),
        "serve_total": totals["serve_total"],
    }


def _extract_match_stats(row: pd.Series) -> tuple[dict[str, float], dict[str, float]] | None:
    def numeric(name: str) -> float:
        return float(pd.to_numeric(row.get(name), errors="coerce"))

    try:
        p1_total, p2_total = numeric("postmatch_player_1_svpt"), numeric("postmatch_player_2_svpt")
        p1_first, p2_first = numeric("postmatch_player_1_1stIn"), numeric("postmatch_player_2_1stIn")
        p1_first_won, p2_first_won = numeric("postmatch_player_1_1stWon"), numeric("postmatch_player_2_1stWon")
        p1_second_won, p2_second_won = numeric("postmatch_player_1_2ndWon"), numeric("postmatch_player_2_2ndWon")
    except (TypeError, ValueError):
        return None
    values = [p1_total, p2_total, p1_first, p2_first, p1_first_won, p2_first_won, p1_second_won, p2_second_won]
    if not np.isfinite(values).all() or min(p1_total, p2_total) < 10:
        return None
    p1_serve_won = p1_first_won + p1_second_won
    p2_serve_won = p2_first_won + p2_second_won
    if not (0 <= p1_serve_won <= p1_total and 0 <= p2_serve_won <= p2_total):
        return None

    def record(side: int, own_total: float, own_first: float, own_first_won: float, own_second_won: float, own_serve_won: float, opp_total: float, opp_serve_won: float) -> dict[str, float]:
        def nonnegative(name: str) -> float:
            value = numeric(f"postmatch_player_{side}_{name}")
            return max(0.0, value) if np.isfinite(value) else 0.0

        return {
            "serve_won": own_serve_won,
            "serve_total": own_total,
            "return_won": max(0.0, opp_total - opp_serve_won),
            "return_total": opp_total,
            "aces": nonnegative("ace"),
            "double_faults": nonnegative("df"),
            "first_in": own_first,
            "first_won": own_first_won,
            "second_won": own_second_won,
            "second_total": max(0.0, own_total - own_first),
            "bp_saved": nonnegative("bpSaved"),
            "bp_faced": nonnegative("bpFaced"),
        }

    return (
        record(1, p1_total, p1_first, p1_first_won, p1_second_won, p1_serve_won, p2_total, p2_serve_won),
        record(2, p2_total, p2_first, p2_first_won, p2_second_won, p2_serve_won, p1_total, p1_serve_won),
    )


def _round_progress(value: object) -> float:
    mapping = {
        "R128": 1, "R64": 2, "R32": 3, "R16": 4, "QF": 5, "SF": 6, "F": 7,
        "RR": 4, "BR": 5,
    }
    return float(mapping.get(str(value or "").upper(), 3) / 7.0)


def _level_strength(value: object) -> float:
    return float({"G": 1.0, "M": 0.85, "F": 0.8, "500": 0.65, "250": 0.5, "A": 0.4, "D": 0.35, "O": 0.7}.get(str(value or ""), 0.5))


def build_phase4_features(data_path: str | Path, progress=print) -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = Path(data_path).resolve()
    frame = pd.read_csv(source_path, low_memory=False)
    required = {
        "match_id", "match_date", "match_status", "surface", "tourney_name",
        "player_1_id", "player_2_id", "player_1_name", "player_2_name",
        "player_1_won", "player_1_odds", "player_2_odds",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Colonnes phase 4 absentes: {missing}")
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    frame = frame.dropna(subset=["match_date", "player_1_name", "player_2_name", "player_1_won"])
    frame = frame.sort_values(["match_date", "match_id"], kind="mergesort").reset_index(drop=True)
    if frame["match_id"].duplicated().any():
        raise AssertionError("match_id duplique dans la table enrichie")

    players: defaultdict[str, PlayerHistory] = defaultdict(PlayerHistory)
    courts: defaultdict[tuple[str, str], CourtHistory] = defaultdict(CourtHistory)
    surface_totals: defaultdict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    records: list[dict[str, Any]] = []

    for day_number, (timestamp, daily) in enumerate(frame.groupby("match_date", sort=True)):
        match_day = pd.Timestamp(timestamp).date()
        if day_number % 500 == 0:
            progress(f"Phase 4 features: {timestamp.date()} ({len(records):,}/{len(frame):,})")
        pending_updates: list[tuple[pd.Series, str, str, str, dict[str, float] | None, dict[str, float] | None]] = []
        for _, row in daily.iterrows():
            p1_key = _player_key(row.get("player_1_id"), row.get("player_1_name"))
            p2_key = _player_key(row.get("player_2_id"), row.get("player_2_name"))
            p1, p2 = players[p1_key], players[p2_key]
            surface = str(row.get("surface") or "Unknown")
            if surface not in SURFACES:
                surface = "Unknown"
            surface_serve = _surface_prior(surface, surface_totals)
            tournament_key = (_normalise_text(row.get("tourney_name")), surface)
            court_rate, court_samples = courts[tournament_key].serve_rate(match_day, surface_serve)
            court_pace = _safe_logit(court_rate) - _safe_logit(surface_serve)

            rates1 = _player_rates(p1, match_day, surface, surface_serve)
            rates2 = _player_rates(p2, match_day, surface, surface_serve)
            matchup_serve_1 = float(np.clip(
                surface_serve + (rates1["serve"] - surface_serve)
                - (rates2["return"] - (1.0 - surface_serve)), 0.45, 0.80,
            ))
            matchup_serve_2 = float(np.clip(
                surface_serve + (rates2["serve"] - surface_serve)
                - (rates1["return"] - (1.0 - surface_serve)), 0.45, 0.80,
            ))

            odds1 = float(pd.to_numeric(row.get("player_1_odds"), errors="coerce"))
            odds2 = float(pd.to_numeric(row.get("player_2_odds"), errors="coerce"))
            overround = 1.0 / odds1 + 1.0 / odds2 if odds1 > 1.0 and odds2 > 1.0 else np.nan
            market_valid = bool(np.isfinite(overround) and 0.95 <= overround <= 1.25)
            market_p1 = (1.0 / odds1) / overround if market_valid else np.nan
            market_logit = _safe_logit(market_p1) if market_valid else np.nan

            rank1 = float(pd.to_numeric(row.get("player_1_rank"), errors="coerce"))
            rank2 = float(pd.to_numeric(row.get("player_2_rank"), errors="coerce"))
            rank1 = rank1 if np.isfinite(rank1) and rank1 > 0 else 500.0
            rank2 = rank2 if np.isfinite(rank2) and rank2 > 0 else 500.0
            points1 = float(pd.to_numeric(row.get("player_1_rank_points"), errors="coerce"))
            points2 = float(pd.to_numeric(row.get("player_2_rank_points"), errors="coerce"))
            points1 = max(0.0, points1) if np.isfinite(points1) else 0.0
            points2 = max(0.0, points2) if np.isfinite(points2) else 0.0
            age1 = float(pd.to_numeric(row.get("player_1_age"), errors="coerce"))
            age2 = float(pd.to_numeric(row.get("player_2_age"), errors="coerce"))
            age1 = age1 if np.isfinite(age1) else 27.0
            age2 = age2 if np.isfinite(age2) else 27.0
            height1 = float(pd.to_numeric(row.get("player_1_ht"), errors="coerce"))
            height2 = float(pd.to_numeric(row.get("player_2_ht"), errors="coerce"))
            height1 = height1 if np.isfinite(height1) else 185.0
            height2 = height2 if np.isfinite(height2) else 185.0

            p1_matches_3, p1_minutes_3 = p1.workload_totals(match_day, 3)
            p2_matches_3, p2_minutes_3 = p2.workload_totals(match_day, 3)
            p1_matches_7, p1_minutes_7 = p1.workload_totals(match_day, 7)
            p2_matches_7, p2_minutes_7 = p2.workload_totals(match_day, 7)
            p1_matches_14, p1_minutes_14 = p1.workload_totals(match_day, 14)
            p2_matches_14, p2_minutes_14 = p2.workload_totals(match_day, 14)

            is_hard, is_clay, is_grass = (float(surface == name) for name in ("Hard", "Clay", "Grass"))
            base_signed = {
                "market_logit": market_logit,
                "global_elo_diff": (p1.global_elo - p2.global_elo) / 400.0,
                "surface_elo_diff": (p1.surface_elo.get(surface, 1500.0) - p2.surface_elo.get(surface, 1500.0)) / 400.0,
                "form_10_logit_diff": _safe_logit(p1.win_rate(match_day)) - _safe_logit(p2.win_rate(match_day)),
                "surface_form_logit_diff": _safe_logit(p1.win_rate(match_day, surface, 20)) - _safe_logit(p2.win_rate(match_day, surface, 20)),
                "rank_log_advantage": math.log1p(rank2) - math.log1p(rank1),
                "points_log_advantage": math.log1p(points1) - math.log1p(points2),
                "age_diff": (age1 - age2) / 10.0,
                "peak_age_advantage": (abs(age2 - 27.0) - abs(age1 - 27.0)) / 10.0,
                "height_diff": (height1 - height2) / 20.0,
                "left_handed_diff": float(str(row.get("player_1_hand")) == "L") - float(str(row.get("player_2_hand")) == "L"),
                "rest_diff": (p1.days_rest(match_day) - p2.days_rest(match_day)) / 30.0,
                "matches_14d_diff": float(p1_matches_14 - p2_matches_14) / 5.0,
                "serve_advantage": _safe_logit(matchup_serve_1) - _safe_logit(matchup_serve_2),
                "serve_skill_logit_diff": _safe_logit(rates1["serve"]) - _safe_logit(rates2["serve"]),
                "return_skill_logit_diff": _safe_logit(rates1["return"]) - _safe_logit(rates2["return"]),
                "ace_rate_logit_diff": _safe_logit(rates1["ace"]) - _safe_logit(rates2["ace"]),
                "double_fault_rate_logit_diff": _safe_logit(rates1["df"]) - _safe_logit(rates2["df"]),
                "first_in_logit_diff": _safe_logit(rates1["first_in"]) - _safe_logit(rates2["first_in"]),
                "first_won_logit_diff": _safe_logit(rates1["first_won"]) - _safe_logit(rates2["first_won"]),
                "second_won_logit_diff": _safe_logit(rates1["second_won"]) - _safe_logit(rates2["second_won"]),
                "break_save_logit_diff": _safe_logit(rates1["bp_save"]) - _safe_logit(rates2["bp_save"]),
                "serve_sample_log_diff": math.log1p(rates1["serve_total"]) - math.log1p(rates2["serve_total"]),
                "minutes_3d_diff": (p1_minutes_3 - p2_minutes_3) / 300.0,
                "minutes_7d_diff": (p1_minutes_7 - p2_minutes_7) / 600.0,
                "minutes_14d_diff": (p1_minutes_14 - p2_minutes_14) / 1000.0,
                "retirement_risk_diff": p1.retirement_rate() - p2.retirement_rate(),
            }
            base_signed.update(
                {
                    "market_logit_x_hard": market_logit * is_hard,
                    "market_logit_x_clay": market_logit * is_clay,
                    "market_logit_x_grass": market_logit * is_grass,
                }
            )
            for feature in SURFACE_INTERACTIONS_BASE:
                for name, flag in (("hard", is_hard), ("clay", is_clay), ("grass", is_grass)):
                    base_signed[f"{feature}_x_{name}"] = base_signed[feature] * flag
            base_signed.update(
                {
                    "serve_advantage_x_court_pace": base_signed["serve_advantage"] * court_pace,
                    "height_diff_x_court_pace": base_signed["height_diff"] * court_pace,
                    "market_logit_x_court_pace": market_logit * court_pace,
                }
            )
            stats = _extract_match_stats(row) if str(row.get("match_status")) == "completed" else None
            record = {
                **base_signed,
                "is_hard": is_hard,
                "is_clay": is_clay,
                "is_grass": is_grass,
                "is_indoor": float(str(row.get("indoor")).upper() in {"I", "1", "TRUE"}),
                "best_of_5": float(pd.to_numeric(row.get("best_of"), errors="coerce") == 5),
                "round_progress": _round_progress(row.get("round")),
                "level_strength": _level_strength(row.get("tourney_level")),
                "court_pace": court_pace,
                "log_surface_samples": math.log1p(court_samples),
                "_date": pd.Timestamp(timestamp),
                "_year": int(pd.Timestamp(timestamp).year),
                "_match_id": str(row.get("match_id")),
                "_p1": str(row.get("player_1_name")),
                "_p2": str(row.get("player_2_name")),
                "_surface": surface,
                "_tournament": str(row.get("tourney_name")),
                "_status": str(row.get("match_status")),
                "_label": int(row.get("player_1_won")),
                "_odds1": odds1,
                "_odds2": odds2,
                "_overround": overround,
                "_market_valid": market_valid,
                "_market_p1": market_p1,
                "_feature_snapshot": "start_of_calendar_day",
                "_p1_serve_sample": rates1["serve_total"],
                "_p2_serve_sample": rates2["serve_total"],
            }
            records.append(record)
            pending_updates.append((row, p1_key, p2_key, surface, stats[0] if stats else None, stats[1] if stats else None))

        # All rows above saw exactly the same start-of-day information set.
        for row, p1_key, p2_key, surface, stats1, stats2 in pending_updates:
            p1, p2 = players[p1_key], players[p2_key]
            label = int(row.get("player_1_won"))
            status = str(row.get("match_status"))
            if status == "completed":
                expected = 1.0 / (1.0 + 10.0 ** ((p2.global_elo - p1.global_elo) / 400.0))
                change = 24.0 * (label - expected)
                p1.global_elo += change
                p2.global_elo -= change
                surface_expected = 1.0 / (
                    1.0 + 10.0 ** ((p2.surface_elo.get(surface, 1500.0) - p1.surface_elo.get(surface, 1500.0)) / 400.0)
                )
                surface_change = 32.0 * (label - surface_expected)
                p1.surface_elo[surface] = p1.surface_elo.get(surface, 1500.0) + surface_change
                p2.surface_elo[surface] = p2.surface_elo.get(surface, 1500.0) - surface_change
                p1.outcomes.append((match_day, surface, bool(label)))
                p2.outcomes.append((match_day, surface, not bool(label)))
            minutes = float(pd.to_numeric(row.get("minutes"), errors="coerce"))
            if np.isfinite(minutes) and minutes > 0 and status not in {"walkover", "defaulted"}:
                p1.workload.append((match_day, minutes))
                p2.workload.append((match_day, minutes))
            if status in {"completed", "retired"}:
                p1.retirement_exposures += 1
                p2.retirement_exposures += 1
                if status == "retired":
                    (p2 if label == 1 else p1).retirements += 1
            if stats1 is not None and stats2 is not None:
                p1.stats.append((match_day, surface, stats1))
                p2.stats.append((match_day, surface, stats2))
                serve_won = stats1["serve_won"] + stats2["serve_won"]
                serve_total = stats1["serve_total"] + stats2["serve_total"]
                surface_totals[surface][0] += serve_won
                surface_totals[surface][1] += serve_total
                tournament_key = (_normalise_text(row.get("tourney_name")), surface)
                courts[tournament_key].observations.append((match_day, serve_won, serve_total))

    result = pd.DataFrame(records)
    expected = set(SURFACE_FEATURES + MLP_INVARIANT_FEATURES)
    missing_features = sorted(expected - set(result.columns))
    if missing_features:
        raise AssertionError(f"Features phase 4 manquantes: {missing_features}")
    audit = {
        "source_path": str(source_path),
        "source_sha256": sha256_file(source_path),
        "rows": int(len(result)),
        "completed_rows": int(result["_status"].eq("completed").sum()),
        "valid_market_rows": int(result["_market_valid"].sum()),
        "date_min": result["_date"].min().date().isoformat(),
        "date_max": result["_date"].max().date().isoformat(),
        "label_p1_rate": float(result["_label"].mean()),
        "snapshot_protocol": "start_of_calendar_day_then_batch_update",
        "serve_history_coverage": float(
            ((result["_p1_serve_sample"] > 0) & (result["_p2_serve_sample"] > 0)).mean()
        ),
    }
    return result, audit


def _make_estimator(name: str, protocol: dict[str, Any]) -> Pipeline:
    if name == "small_symmetric_mlp":
        spec = protocol["deep_learning_diagnostic_not_strategy_eligible"]
        estimator = MLPClassifier(
            hidden_layer_sizes=tuple(spec["hidden_layer_sizes"]),
            alpha=float(spec["alpha"]),
            learning_rate_init=float(spec["learning_rate_init"]),
            early_stopping=bool(spec["early_stopping"]),
            validation_fraction=0.10,
            n_iter_no_change=15,
            batch_size=512,
            max_iter=int(spec["max_iter"]),
            random_state=20260902,
        )
    else:
        spec = protocol["primary_candidates"][name]
        estimator = LogisticRegression(
            C=float(spec["C"]), max_iter=3000, random_state=20260902
        )
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", estimator),
        ]
    )


def _candidate_columns(name: str) -> tuple[list[str], int]:
    if name == "small_symmetric_mlp":
        return [*SURFACE_FEATURES, *MLP_INVARIANT_FEATURES], len(SURFACE_FEATURES)
    columns = CANDIDATE_FEATURES[name]
    return columns, len(columns)


def _swap_matrix(matrix: np.ndarray, signed_count: int) -> np.ndarray:
    swapped = np.asarray(matrix, dtype=np.float32).copy()
    swapped[:, :signed_count] *= -1.0
    return swapped


def _fit_symmetric(
    model: Pipeline,
    matrix: np.ndarray,
    labels: np.ndarray,
    signed_count: int,
) -> Pipeline:
    augmented_x = np.vstack([matrix, _swap_matrix(matrix, signed_count)])
    augmented_y = np.concatenate([labels, 1 - labels])
    model.fit(augmented_x, augmented_y)
    return model


def _predict_symmetric(model: Pipeline, matrix: np.ndarray, signed_count: int) -> np.ndarray:
    direct = model.predict_proba(matrix)[:, 1]
    reversed_probability = model.predict_proba(_swap_matrix(matrix, signed_count))[:, 1]
    return np.clip(0.5 * (direct + 1.0 - reversed_probability), 1e-6, 1.0 - 1e-6)


def _fit_temperature(probability: np.ndarray, labels: np.ndarray) -> float:
    z = logit(np.clip(probability, 1e-6, 1 - 1e-6)).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=1000.0, fit_intercept=False, max_iter=1000, random_state=20260902
    )
    calibrator.fit(np.vstack([z, -z]), np.concatenate([labels, 1 - labels]))
    return float(calibrator.coef_[0, 0])


def probability_metrics(labels: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    labels = np.asarray(labels, dtype=int)
    return {
        "n": int(len(labels)),
        "log_loss": float(log_loss(labels, probability)),
        "brier": float(brier_score_loss(labels, probability)),
        "auc": float(roc_auc_score(labels, probability)),
    }


def walk_forward_candidate(
    features: pd.DataFrame,
    name: str,
    protocol: dict[str, Any],
    progress=print,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    columns, signed_count = _candidate_columns(name)
    walk = protocol["development_walk_forward"]
    completed = features["_status"].eq("completed")
    valid = features["_market_valid"].astype(bool)
    outputs: list[pd.DataFrame] = []
    folds: list[dict[str, Any]] = []
    for year in walk["test_years"]:
        train = features[completed & valid & features["_year"].le(int(year) - 2)]
        calibration = features[completed & valid & features["_year"].eq(int(year) - 1)]
        test = features[valid & features["_year"].eq(int(year))]
        if len(train) < int(walk["minimum_train_rows"]) or len(calibration) < int(walk["minimum_calibration_rows"]) or test.empty:
            raise RuntimeError(
                f"Fold {year} insuffisant pour {name}: train={len(train)}, "
                f"calibration={len(calibration)}, test={len(test)}"
            )
        if not (
            train["_date"].max() < calibration["_date"].min()
            <= calibration["_date"].max() < test["_date"].min()
        ):
            raise AssertionError(f"Chevauchement temporel phase 4 dans le fold {year}")
        progress(
            f"{name} | train <= {year - 2}, calibration {year - 1}, test {year}"
        )
        x_train = train[columns].to_numpy(dtype=np.float32)
        y_train = train["_label"].to_numpy(dtype=int)
        x_cal = calibration[columns].to_numpy(dtype=np.float32)
        y_cal = calibration["_label"].to_numpy(dtype=int)
        x_test = test[columns].to_numpy(dtype=np.float32)
        model = _fit_symmetric(_make_estimator(name, protocol), x_train, y_train, signed_count)
        raw_calibration = _predict_symmetric(model, x_cal, signed_count)
        temperature = _fit_temperature(raw_calibration, y_cal)
        raw_test = _predict_symmetric(model, x_test, signed_count)
        probability = expit(temperature * logit(raw_test))
        output = test.copy()
        output["p_model"] = probability
        output["p_raw"] = raw_test
        output["candidate"] = name
        outputs.append(output)
        evaluated = output[output["_status"].eq("completed")]
        metrics = probability_metrics(evaluated["_label"], evaluated["p_model"])
        metrics.update(
            {
                "candidate": name,
                "test_year": int(year),
                "train_rows": int(len(train)),
                "calibration_rows": int(len(calibration)),
                "test_rows": int(len(test)),
                "train_max_date": train["_date"].max().date().isoformat(),
                "calibration_min_date": calibration["_date"].min().date().isoformat(),
                "calibration_max_date": calibration["_date"].max().date().isoformat(),
                "test_min_date": test["_date"].min().date().isoformat(),
                "temperature": temperature,
            }
        )
        folds.append(metrics)
    return pd.concat(outputs, ignore_index=True), folds


def _metrics_with_market(predictions: pd.DataFrame) -> dict[str, Any]:
    played = predictions[predictions["_status"].eq("completed")]
    model = probability_metrics(played["_label"], played["p_model"])
    market = probability_metrics(played["_label"], played["_market_p1"])
    model.update(
        {
            "market_log_loss": market["log_loss"],
            "market_brier": market["brier"],
            "log_loss_improvement_vs_market": market["log_loss"] - model["log_loss"],
        }
    )
    return model


def _segmented_metrics(predictions: pd.DataFrame, column: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    played = predictions[predictions["_status"].eq("completed")]
    for key, group in played.groupby(column):
        if len(group) < 20:
            continue
        model = probability_metrics(group["_label"], group["p_model"])
        market = probability_metrics(group["_label"], group["_market_p1"])
        result[str(key)] = {
            **model,
            "market_log_loss": market["log_loss"],
            "log_loss_improvement_vs_market": market["log_loss"] - model["log_loss"],
        }
    return result


def make_fixed_bets(predictions: pd.DataFrame, protocol: dict[str, Any]) -> pd.DataFrame:
    rule = protocol["fixed_economic_diagnostic"]
    frame = predictions.copy()
    p1 = frame["p_model"].to_numpy(dtype=float)
    p2 = 1.0 - p1
    odds1 = frame["_odds1"].to_numpy(dtype=float)
    odds2 = frame["_odds2"].to_numpy(dtype=float)
    market1 = frame["_market_p1"].to_numpy(dtype=float)
    market2 = 1.0 - market1
    ev1, ev2 = p1 * odds1 - 1.0, p2 * odds2 - 1.0
    choose_p1 = ev1 >= ev2
    frame["bet_side"] = np.where(choose_p1, 1, 2)
    frame["bet_player"] = np.where(choose_p1, frame["_p1"], frame["_p2"])
    frame["bet_probability"] = np.where(choose_p1, p1, p2)
    frame["bet_market_probability"] = np.where(choose_p1, market1, market2)
    frame["bet_odds"] = np.where(choose_p1, odds1, odds2)
    frame["edge"] = frame["bet_probability"] - frame["bet_market_probability"]
    frame["expected_roi"] = np.where(choose_p1, ev1, ev2)
    frame["won"] = np.where(choose_p1, frame["_label"].eq(1), frame["_label"].eq(0))
    mask = (
        frame["edge"].ge(float(rule["minimum_edge"]))
        & frame["expected_roi"].ge(float(rule["minimum_expected_roi"]))
        & frame["bet_odds"].between(
            float(rule["minimum_decimal_odds"]),
            float(rule["maximum_decimal_odds"]),
            inclusive="both",
        )
    )
    return frame.loc[mask].sort_values(["_date", "_match_id"]).reset_index(drop=True)


def _settled_returns(bets: pd.DataFrame, haircut: float) -> tuple[np.ndarray, np.ndarray]:
    settled = bets["_status"].eq("completed").to_numpy()
    effective_odds = 1.0 + (bets["bet_odds"].to_numpy(dtype=float) - 1.0) * (1.0 - haircut)
    unit_return = np.where(bets["won"].to_numpy(), effective_odds - 1.0, -1.0)
    return unit_return, settled


def economic_summary(bets: pd.DataFrame, haircut: float) -> dict[str, Any]:
    returns, settled = _settled_returns(bets, haircut)
    resolved = returns[settled]
    if len(resolved) == 0:
        return {
            "bets": int(len(bets)), "settled": 0, "void": int(len(bets)),
            "roi": None, "profit_units": 0.0, "yearly": {}, "surface": {},
        }
    resolved_bets = bets.loc[settled].copy()
    resolved_bets["unit_return"] = resolved

    def grouped(column: str) -> dict[str, Any]:
        return {
            str(key): {
                "settled": int(len(group)),
                "profit_units": float(group["unit_return"].sum()),
                "roi": float(group["unit_return"].mean()),
            }
            for key, group in resolved_bets.groupby(column)
        }

    return {
        "bets": int(len(bets)),
        "settled": int(settled.sum()),
        "void": int((~settled).sum()),
        "wins": int(resolved_bets["won"].sum()),
        "average_odds": float(resolved_bets["bet_odds"].mean()),
        "average_edge": float(resolved_bets["edge"].mean()),
        "profit_units": float(resolved.sum()),
        "roi": float(resolved.mean()),
        "standard_error": float(np.std(resolved, ddof=1) / math.sqrt(len(resolved))) if len(resolved) > 1 else None,
        "yearly": grouped("_year"),
        "surface": grouped("_surface"),
    }


def month_block_bootstrap(
    bets: pd.DataFrame,
    haircut: float,
    samples: int,
    confidence: float,
) -> dict[str, Any]:
    returns, settled = _settled_returns(bets, haircut)
    resolved = bets.loc[settled, ["_date"]].copy()
    resolved["return"] = returns[settled]
    resolved["month"] = resolved["_date"].dt.to_period("M").astype(str)
    blocks = resolved.groupby("month")["return"].agg(["sum", "count"])
    if blocks.empty:
        return {"samples": 0, "months": 0, "low": None, "median": None, "high": None}
    rng = np.random.default_rng(20260902)
    draw = rng.integers(0, len(blocks), size=(samples, len(blocks)))
    roi = blocks["sum"].to_numpy()[draw].sum(axis=1) / np.maximum(
        blocks["count"].to_numpy()[draw].sum(axis=1), 1
    )
    alpha = (1.0 - confidence) / 2.0
    return {
        "samples": int(samples),
        "months": int(len(blocks)),
        "confidence": float(confidence),
        "low": float(np.quantile(roi, alpha)),
        "median": float(np.median(roi)),
        "high": float(np.quantile(roi, 1.0 - alpha)),
        "probability_roi_positive": float(np.mean(roi > 0.0)),
    }


def simulate_fixed_bankroll(bets: pd.DataFrame, protocol: dict[str, Any]) -> dict[str, Any]:
    spec = protocol["staking_diagnostic"]
    bankroll = float(spec["initial_bankroll"])
    equity = [bankroll]
    total_staked = 0.0
    for _, daily in bets.groupby(bets["_date"].dt.normalize(), sort=True):
        start = bankroll
        fractions = np.full(len(daily), float(spec["fraction_per_bet"]))
        if fractions.sum() > float(spec["maximum_daily_exposure"]):
            fractions *= float(spec["maximum_daily_exposure"]) / fractions.sum()
        stakes = start * fractions
        returns, settled = _settled_returns(
            daily, float(protocol["fixed_economic_diagnostic"]["odds_haircut_primary"])
        )
        profits = stakes * np.where(settled, returns, 0.0)
        bankroll += float(profits.sum())
        total_staked += float(stakes[settled].sum())
        equity.append(bankroll)
    values = np.asarray(equity)
    peaks = np.maximum.accumulate(values)
    drawdowns = (peaks - values) / np.maximum(peaks, 1e-9)
    return {
        "initial_bankroll": float(spec["initial_bankroll"]),
        "final_bankroll": bankroll,
        "return": bankroll / float(spec["initial_bankroll"]) - 1.0,
        "max_drawdown": float(drawdowns.max()),
        "total_staked": total_staked,
    }


def _markdown_report(report: dict[str, Any]) -> str:
    selected = report["selected_primary_candidate"]
    selected_metrics = report["primary_candidate_metrics"][selected]
    economic = report["fixed_economic_diagnostic"]["haircut_2pct"]
    bootstrap = report["fixed_economic_diagnostic"]["bootstrap_99pct"]
    deep = report["deep_learning_diagnostic"]
    decision = report["decision"]
    lines = [
        "# Tennis phase 4 — serve/return et surfaces",
        "",
        f"Décision: **{decision['status']}**",
        "",
        "Toutes les données jusqu'au 30 août 2026 sont du développement déjà exposé. ",
        "Aucune performance ci-dessous n'est une nouvelle validation indépendante.",
        "",
        "## Modèles walk-forward 2017–2026",
        "",
        "| Modèle | Log-loss | Écart au marché | Brier |",
        "|---|---:|---:|---:|",
    ]
    for name in CANDIDATE_ORDER:
        metrics = report["primary_candidate_metrics"][name]
        lines.append(
            f"| `{name}` | {metrics['log_loss']:.5f} | "
            f"{metrics['log_loss_improvement_vs_market']:+.5f} | {metrics['brier']:.5f} |"
        )
    lines.extend(
        [
            f"| Marché dévigé | {selected_metrics['market_log_loss']:.5f} | — | {selected_metrics['market_brier']:.5f} |",
            "",
            f"Candidat primaire retenu par la règle figée: `{selected}`.",
            "",
            "## Deep learning diagnostique",
            "",
            f"Le petit MLP symétrique obtient une log-loss de {deep['log_loss']:.5f}, soit "
            f"{deep['log_loss_improvement_vs_market']:+.5f} contre le marché. Il n'était pas "
            "autorisé à sélectionner une stratégie ou à ouvrir une gate économique.",
            "",
            "## Diagnostic économique fixe, développement uniquement",
            "",
            f"- Paris réglés après décote 2%: {economic['settled']}.",
            f"- ROI: {economic['roi']:.2%}; profit: {economic['profit_units']:.2f} unités.",
            f"- IC bootstrap mensuel 99%: [{bootstrap['low']:.2%}, {bootstrap['high']:.2%}].",
            f"- Bankroll: {report['fixed_economic_diagnostic']['bankroll']['initial_bankroll']:.2f} → "
            f"{report['fixed_economic_diagnostic']['bankroll']['final_bankroll']:.2f}; drawdown "
            f"{report['fixed_economic_diagnostic']['bankroll']['max_drawdown']:.2%}.",
            "",
            "## Gate",
            "",
        ]
    )
    for name, passed in decision["gate_checks"].items():
        lines.append(f"- {'OK' if passed else 'ÉCHEC'} — `{name}`")
    lines.extend(
        [
            "",
            "## Limites",
            "",
            "- Les prix historiques sont des paires pré-match cohérentes, mais sans timestamp exact ni garantie d'exécution.",
            "- Les rapprochements de cotes ont une confiance variable; aucun seuil n'a été modifié après observation des résultats.",
            "- L'effet serve/return est fortement redondant avec le marché, Elo et le classement.",
            "- Les résultats 2017–2026 sont tous du développement déjà exposé; ils ne prouvent aucune rentabilité future.",
            "",
            "Même si la gate de développement passait, elle autoriserait seulement un suivi "
            "prospectif papier à partir du 3 septembre 2026. L'argent réel reste bloqué.",
            "",
        ]
    )
    return "\n".join(lines)


def run_phase4(
    base_dir: str | Path,
    progress=print,
    reuse_features: bool = False,
) -> dict[str, Any]:
    base = Path(base_dir).resolve()
    output = base / "models" / "rigorous_strategy"
    protocol_path = output / "phase4_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    data_path = base / protocol["data_path"]
    current_hash = sha256_file(data_path)
    if current_hash != protocol["data_sha256"]:
        raise RuntimeError(
            "La base a change depuis le gel du protocole phase 4; nouveau protocole requis"
        )
    feature_path = output / "phase4_features.parquet"
    audit_path = output / "phase4_feature_audit.json"
    if reuse_features and feature_path.exists() and audit_path.exists():
        audit = json.loads(audit_path.read_text())
        if audit.get("source_sha256") != current_hash:
            raise RuntimeError("Cache phase 4 lie a une autre version de donnees")
        features = pd.read_parquet(feature_path)
        progress(f"Réutilisation du cache phase 4 vérifié: {len(features):,} lignes")
    else:
        features, audit = build_phase4_features(data_path, progress)
        features.to_parquet(feature_path, index=False)
        audit_path.write_text(
            json.dumps(_json_ready(audit), indent=2, ensure_ascii=False) + "\n"
        )

    predictions_by_name: dict[str, pd.DataFrame] = {}
    candidate_metrics: dict[str, Any] = {}
    fold_metrics: list[dict[str, Any]] = []
    surface_metrics: dict[str, Any] = {}
    yearly_metrics: dict[str, Any] = {}
    for name in CANDIDATE_ORDER:
        predictions, folds = walk_forward_candidate(features, name, protocol, progress)
        predictions_by_name[name] = predictions
        candidate_metrics[name] = _metrics_with_market(predictions)
        surface_metrics[name] = _segmented_metrics(predictions, "_surface")
        yearly_metrics[name] = _segmented_metrics(predictions, "_year")
        fold_metrics.extend(folds)

    selected_name = min(
        CANDIDATE_ORDER,
        key=lambda name: (
            candidate_metrics[name]["log_loss"],
            candidate_metrics[name]["brier"],
            CANDIDATE_ORDER.index(name),
        ),
    )
    selected = predictions_by_name[selected_name]
    progress(f"Candidat primaire phase 4: {selected_name}")

    deep_name = protocol["deep_learning_diagnostic_not_strategy_eligible"]["name"]
    deep_predictions, deep_folds = walk_forward_candidate(features, deep_name, protocol, progress)
    deep_metrics = _metrics_with_market(deep_predictions)
    deep_surface = _segmented_metrics(deep_predictions, "_surface")
    deep_yearly = _segmented_metrics(deep_predictions, "_year")

    bets = make_fixed_bets(selected, protocol)
    haircuts: dict[str, Any] = {}
    for haircut in protocol["fixed_economic_diagnostic"]["odds_haircut_scenarios"]:
        haircuts[f"haircut_{int(float(haircut) * 100)}pct"] = economic_summary(
            bets, float(haircut)
        )
    primary_haircut = float(protocol["fixed_economic_diagnostic"]["odds_haircut_primary"])
    gate = protocol["gate_for_prospective_paper_tracking_only"]
    bootstrap = month_block_bootstrap(
        bets,
        primary_haircut,
        int(gate["bootstrap_samples"]),
        float(gate["cluster_bootstrap_confidence"]),
    )
    bankroll = simulate_fixed_bankroll(bets, protocol)
    primary_economic = haircuts[f"haircut_{int(primary_haircut * 100)}pct"]

    selected_years = yearly_metrics[selected_name]
    years_beating_market = sum(
        metrics["log_loss"] < metrics["market_log_loss"]
        for metrics in selected_years.values()
    )
    surfaces = {
        name: metrics for name, metrics in surface_metrics[selected_name].items()
        if name in {"Hard", "Clay", "Grass"}
    }
    maximum_surface_degradation = max(
        (metrics["log_loss"] - metrics["market_log_loss"] for metrics in surfaces.values()),
        default=float("inf"),
    )
    positive_years = sum(
        metrics["roi"] > 0.0 for metrics in primary_economic["yearly"].values()
    )
    selected_probability = candidate_metrics[selected_name]
    checks = {
        "selected_candidate_is_not_market_only": selected_name != "market_surface_calibration",
        "minimum_log_loss_improvement_vs_market": selected_probability["log_loss_improvement_vs_market"] >= float(gate["minimum_log_loss_improvement_vs_market"]),
        "years_beating_market": years_beating_market >= int(gate["years_beating_market_minimum"]),
        "maximum_surface_log_loss_degradation": maximum_surface_degradation <= float(gate["maximum_surface_log_loss_degradation"]),
        "minimum_settled_bets": primary_economic["settled"] >= int(gate["minimum_settled_bets"]),
        "positive_roi_after_2pct_haircut": primary_economic["roi"] is not None and primary_economic["roi"] > 0.0,
        "positive_years": positive_years >= int(gate["positive_years_minimum"]),
        "positive_99pct_month_bootstrap_lower_bound": bootstrap["low"] is not None and bootstrap["low"] > float(gate["roi_lower_bound_must_exceed"]),
        "maximum_drawdown": bankroll["max_drawdown"] <= float(gate["maximum_drawdown"]),
    }
    paper_tracking_approved = all(checks.values())
    status = (
        "PHASE4_DEVELOPMENT_PASSED_PAPER_TRACKING_ONLY_NO_BET"
        if paper_tracking_approved
        else "PHASE4_REJECTED_NO_BET"
    )

    compact = selected[
        [
            "_match_id", "_date", "_year", "_p1", "_p2", "_surface", "_tournament",
            "_status", "_label", "_odds1", "_odds2", "_market_p1",
        ]
    ].copy()
    for name, predictions in predictions_by_name.items():
        probability = predictions[["_match_id", "p_model"]].rename(
            columns={"p_model": f"p_{name}"}
        )
        compact = compact.merge(probability, on="_match_id", how="left", validate="one_to_one")
    compact = compact.merge(
        deep_predictions[["_match_id", "p_model"]].rename(columns={"p_model": "p_small_symmetric_mlp"}),
        on="_match_id", how="left", validate="one_to_one",
    )
    compact.to_parquet(output / "phase4_oos_predictions.parquet", index=False)
    bets.to_parquet(output / "phase4_development_bets.parquet", index=False)

    report = {
        "protocol": protocol,
        "protocol_sha256": sha256_file(protocol_path),
        "data_audit": audit,
        "primary_candidate_metrics": candidate_metrics,
        "primary_candidate_surface_metrics": surface_metrics,
        "primary_candidate_yearly_metrics": yearly_metrics,
        "selected_primary_candidate": selected_name,
        "fold_metrics": fold_metrics,
        "deep_learning_diagnostic": {
            **deep_metrics,
            "surface": deep_surface,
            "yearly": deep_yearly,
            "folds": deep_folds,
            "strategy_eligible": False,
        },
        "fixed_economic_diagnostic": {
            **haircuts,
            "bootstrap_99pct": bootstrap,
            "bankroll": bankroll,
            "years_positive_after_2pct_haircut": int(positive_years),
        },
        "decision": {
            "status": status,
            "paper_tracking_approved": paper_tracking_approved,
            "real_money_approved": False,
            "gate_checks": checks,
            "years_beating_market": int(years_beating_market),
            "maximum_surface_log_loss_degradation": float(maximum_surface_degradation),
            "reason": "Historical phase 4 is development-only; real money requires 500 new resolved prospective bets after 2026-09-02.",
        },
    }
    ready = _json_ready(report)
    (output / "phase4_report.json").write_text(
        json.dumps(ready, indent=2, ensure_ascii=False) + "\n"
    )
    (output / "PHASE4_REPORT.md").write_text(_markdown_report(ready))
    lock = {
        "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "protocol_sha256": ready["protocol_sha256"],
        "data_sha256": current_hash,
        "selected_primary_candidate": selected_name,
        "paper_tracking_approved": paper_tracking_approved,
        "real_money_approved": False,
        "status": status,
        "new_pristine_period_starts": protocol["new_pristine_period_starts"],
    }
    (output / "phase4_lock.json").write_text(
        json.dumps(lock, indent=2, ensure_ascii=False) + "\n"
    )
    return ready
