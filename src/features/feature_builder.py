"""
Feature Engineering Pipeline — v3
==================================
Construit un vecteur de 48 features pour chaque match.

OPTIMISATION CLÉ : build_dataset utilise une approche ONLINE O(n).
Pour chaque joueur, on maintient un état courant (résultats récents, dates,
H2H par adversaire) qui se met à jour après chaque match.
→ Passage de O(n²) à O(n) sur 67k matchs : ~10s au lieu de plusieurs heures.

Point critique : features calculées avec données ANTÉRIEURES au match
(look-ahead safe). L'état est lu AVANT d'être mis à jour.
"""

from __future__ import annotations

import math
import warnings
from collections import defaultdict, deque
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.features.elo_system import ROUND_MAP, SERIES_MAP, SURFACES

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FEATURE_COLS_V3 = [
    # Elo global
    "elo_diff", "elo_p1", "elo_p2",
    # Elo surface
    "surf_elo_diff", "surf_elo_p1", "surf_elo_p2",
    # Momentum Elo
    "momentum_elo_diff",
    # Forme récente
    "p1_form_3", "p2_form_3",
    "p1_form_5", "p2_form_5",
    "p1_form_10", "p2_form_10",
    "p1_form_20", "p2_form_20",
    # Momentum (tendance)
    "p1_momentum", "p2_momentum",
    # Surface win rate
    "p1_surf_wr_3m", "p2_surf_wr_3m",
    "p1_surf_wr_6m", "p2_surf_wr_6m",
    "p1_surf_wr_12m", "p2_surf_wr_12m",
    # H2H
    "h2h_p1_wr", "h2h_total", "h2h_surf_p1_wr",
    # Fatigue / repos
    "p1_fatigue", "p2_fatigue",
    "p1_days_rest", "p2_days_rest",
    # Ranking
    "rank_diff", "rank_ratio", "pts_diff", "log_rank_diff",
    # Contexte match
    "is_hard", "is_clay", "is_grass", "is_indoor",
    "best_of_5", "round_num", "series_num",
    # Saisonnalité
    "month_sin", "month_cos",
    # Spécialisation surface
    "p1_surf_specialist", "p2_surf_specialist",
    # Pression
    "p1_slam_experience", "p2_slam_experience",
]


# ---------------------------------------------------------------------------
# Online Player State (clé de performance)
# ---------------------------------------------------------------------------

class PlayerState:
    """
    État courant d'un joueur mis à jour après chaque match.
    Permet un calcul O(1) de toutes les features rolling.
    """
    MAX_RESULTS = 20
    MAX_DATES = 30  # Pour fatigue (14j) et repos

    def __init__(self):
        # Résultats récents : liste de (date, won:bool)
        self.results: deque = deque(maxlen=self.MAX_RESULTS)
        # Résultats par surface : {surface: deque[(date, won)]}
        self.surf_results: Dict[str, deque] = defaultdict(lambda: deque(maxlen=50))
        # H2H global et par surface
        self.h2h: Dict[str, list] = defaultdict(list)         # opp -> [(date, won)]
        self.h2h_surf: Dict[str, Dict[str, list]] = defaultdict(lambda: defaultdict(list))
        # Dates récentes pour fatigue/repos
        self.match_dates: deque = deque(maxlen=self.MAX_DATES)
        # Stats Grand Chelem
        self.slam_matches: int = 0

    def _form_n(self, n: int) -> float:
        """Win rate sur les n derniers matchs."""
        recent = list(self.results)[-n:]
        return sum(1 for _, w in recent if w) / len(recent) if recent else 0.5

    def form(self, n: int) -> float:
        return self._form_n(n)

    def surf_wr(self, surface: str, days: int, ref_date: date) -> float:
        """Win rate surface dans la fenêtre [ref_date - days, ref_date)."""
        cutoff = ref_date - timedelta(days=days)
        recent = [(d, w) for d, w in self.surf_results[surface] if d >= cutoff]
        if not recent:
            return 0.5
        return sum(1 for _, w in recent if w) / len(recent)

    def surf_specialist(self, surface: str, ref_date: date) -> float:
        """Ratio surf_wr_12m / global_wr_12m."""
        sw = self.surf_wr(surface, 365, ref_date)
        cutoff = ref_date - timedelta(days=365)
        recent_all = [(d, w) for d, w in self.results if d >= cutoff]
        if not recent_all:
            return 1.0
        gw = sum(1 for _, w in recent_all if w) / len(recent_all)
        return sw / gw if gw > 0 else 1.0

    def h2h_stats(self, opponent: str) -> Tuple[float, int]:
        records = self.h2h.get(opponent, [])
        if not records:
            return 0.5, 0
        wins = sum(1 for _, w in records if w)
        return wins / len(records), len(records)

    def h2h_surf_stats(self, opponent: str, surface: str) -> float:
        records = self.h2h_surf.get(opponent, {}).get(surface, [])
        if not records:
            return 0.5
        wins = sum(1 for _, w in records if w)
        return wins / len(records)

    def fatigue(self, ref_date: date, window_days: int = 14) -> int:
        cutoff = ref_date - timedelta(days=window_days)
        return sum(1 for d in self.match_dates if d >= cutoff)

    def days_rest(self, ref_date: date) -> int:
        dates = [d for d in self.match_dates if d < ref_date]
        if not dates:
            return 30
        return (ref_date - max(dates)).days

    def update(self, match_date: date, won: bool, surface: str, opponent: str, series: str):
        """Met à jour l'état après un match. Appelé APRÈS avoir lu les features."""
        self.results.append((match_date, won))
        self.surf_results[surface].append((match_date, won))
        self.h2h[opponent].append((match_date, won))
        self.h2h_surf[opponent][surface].append((match_date, won))
        self.match_dates.append(match_date)
        if series == "Grand Slam":
            self.slam_matches += 1


# ---------------------------------------------------------------------------
# Context encoders
# ---------------------------------------------------------------------------

def encode_surface(surface: str) -> Tuple[int, int, int]:
    return (
        1 if surface == "Hard" else 0,
        1 if surface == "Clay" else 0,
        1 if surface == "Grass" else 0,
    )


def encode_month_cyclic(month: int) -> Tuple[float, float]:
    angle = 2 * math.pi * (month - 1) / 12
    return math.sin(angle), math.cos(angle)


def is_indoor(surface: str, tournament_name: str = "") -> int:
    if surface == "Carpet":
        return 1
    indoor_keywords = ["indoor", "arena", "salle", "covered", "carpet"]
    for kw in indoor_keywords:
        if kw in tournament_name.lower():
            return 1
    return 0


def safe_rank(val, default=100.0) -> float:
    v = float(val) if val is not None else default
    return default if v <= 0 or math.isnan(v) else v


# ---------------------------------------------------------------------------
# Main Feature Builder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """
    Construit le vecteur de features complet.

    Deux modes :
      1. build_dataset(elo_history, raw_df) → DataFrame complet (O(n), rapide)
      2. build_single(...) → array pour prédiction en production
    """

    def __init__(self, feature_cols: List[str] = None):
        self.feature_cols = feature_cols or FEATURE_COLS_V3

    # ------------------------------------------------------------------
    # MODE ENTRAÎNEMENT — O(n) via PlayerState online
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        elo_history: pd.DataFrame,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        Construit le dataset complet en une passe chronologique.
        elo_history = output de TennisEloEngine.get_history() éventuellement
        enrichi avec Rank_1, Rank_2, Pts_1, Pts_2, Best_of.
        """
        df = elo_history.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        n = len(df)
        player_states: Dict[str, PlayerState] = defaultdict(PlayerState)
        records = []

        for i, row in df.iterrows():
            if progress_callback and i % 5000 == 0:
                progress_callback(f"Features: {i}/{n} matchs")

            match_date: date = row["Date"].date()
            p1 = str(row["Player_1"])
            p2 = str(row["Player_2"])
            surface = str(row.get("Surface", "Hard"))
            series = str(row.get("Series", "ATP250"))
            round_name = str(row.get("Round", "1st Round"))
            label = int(row.get("label", 1))
            winner = p1 if label == 1 else p2
            loser = p2 if label == 1 else p1

            s1 = player_states[p1]
            s2 = player_states[p2]

            # --- Forme ---
            f3_1, f3_2   = s1.form(3),  s2.form(3)
            f5_1, f5_2   = s1.form(5),  s2.form(5)
            f10_1, f10_2 = s1.form(10), s2.form(10)
            f20_1, f20_2 = s1.form(20), s2.form(20)

            # --- Surface WR ---
            sw3_1  = s1.surf_wr(surface, 90,  match_date)
            sw3_2  = s2.surf_wr(surface, 90,  match_date)
            sw6_1  = s1.surf_wr(surface, 180, match_date)
            sw6_2  = s2.surf_wr(surface, 180, match_date)
            sw12_1 = s1.surf_wr(surface, 365, match_date)
            sw12_2 = s2.surf_wr(surface, 365, match_date)

            # --- H2H ---
            h2h_wr, h2h_total   = s1.h2h_stats(p2)
            h2h_surf_wr         = s1.h2h_surf_stats(p2, surface)

            # --- Fatigue / repos ---
            fat1, fat2  = s1.fatigue(match_date), s2.fatigue(match_date)
            rest1, rest2 = s1.days_rest(match_date), s2.days_rest(match_date)

            # --- Surface specialist ---
            spec1 = min(s1.surf_specialist(surface, match_date), 3.0)
            spec2 = min(s2.surf_specialist(surface, match_date), 3.0)

            # --- Ranking ---
            r1 = safe_rank(row.get("Rank_1", 100))
            r2 = safe_rank(row.get("Rank_2", 100))
            pts1 = max(0.0, float(row.get("Pts_1", 0) or 0))
            pts2 = max(0.0, float(row.get("Pts_2", 0) or 0))
            rank_diff = r1 - r2
            rank_ratio = r1 / r2 if r2 > 0 else 1.0
            pts_diff = pts1 - pts2
            log_rank_diff = math.log(r1 + 1) - math.log(r2 + 1)

            # --- Context ---
            is_h, is_c, is_g = encode_surface(surface)
            indoor = is_indoor(surface, str(row.get("Tournament", "")))
            best_of_val = row.get("Best_of", row.get("Best of", 3))
            try:
                best5 = 1 if int(float(best_of_val or 3)) == 5 else 0
            except (ValueError, TypeError):
                best5 = 0
            round_num = ROUND_MAP.get(round_name, 3)
            series_num = SERIES_MAP.get(series, 3)
            m_sin, m_cos = encode_month_cyclic(row["Date"].month)

            # --- Elo (pré-match, depuis elo_history) ---
            elo_p1 = float(row.get("elo_p1", 1500))
            elo_p2 = float(row.get("elo_p2", 1500))
            surf_p1 = float(row.get("surf_elo_p1", 1500))
            surf_p2 = float(row.get("surf_elo_p2", 1500))
            mom_p1 = float(row.get("momentum_elo_p1", 1500))
            mom_p2 = float(row.get("momentum_elo_p2", 1500))

            record = {
                "elo_diff": elo_p1 - elo_p2,
                "elo_p1": elo_p1, "elo_p2": elo_p2,
                "surf_elo_diff": surf_p1 - surf_p2,
                "surf_elo_p1": surf_p1, "surf_elo_p2": surf_p2,
                "momentum_elo_diff": mom_p1 - mom_p2,
                "p1_form_3": f3_1, "p2_form_3": f3_2,
                "p1_form_5": f5_1, "p2_form_5": f5_2,
                "p1_form_10": f10_1, "p2_form_10": f10_2,
                "p1_form_20": f20_1, "p2_form_20": f20_2,
                "p1_momentum": f5_1 - f20_1,
                "p2_momentum": f5_2 - f20_2,
                "p1_surf_wr_3m": sw3_1, "p2_surf_wr_3m": sw3_2,
                "p1_surf_wr_6m": sw6_1, "p2_surf_wr_6m": sw6_2,
                "p1_surf_wr_12m": sw12_1, "p2_surf_wr_12m": sw12_2,
                "h2h_p1_wr": h2h_wr,
                "h2h_total": h2h_total,
                "h2h_surf_p1_wr": h2h_surf_wr,
                "p1_fatigue": fat1, "p2_fatigue": fat2,
                "p1_days_rest": rest1, "p2_days_rest": rest2,
                "rank_diff": rank_diff, "rank_ratio": rank_ratio,
                "pts_diff": pts_diff, "log_rank_diff": log_rank_diff,
                "is_hard": is_h, "is_clay": is_c, "is_grass": is_g,
                "is_indoor": indoor,
                "best_of_5": best5,
                "round_num": round_num, "series_num": series_num,
                "month_sin": m_sin, "month_cos": m_cos,
                "p1_surf_specialist": spec1,
                "p2_surf_specialist": spec2,
                "p1_slam_experience": s1.slam_matches,
                "p2_slam_experience": s2.slam_matches,
                # Metadata
                "_date": row["Date"],
                "_p1": p1, "_p2": p2,
                "_surface": surface, "_series": series, "_round": round_name,
                "_tournament": str(row.get("Tournament", "")),
                "_label": label,
            }
            records.append(record)

            # Mise à jour des états APRÈS avoir lu les features
            s1.update(match_date, label == 1, surface, p2, series)
            s2.update(match_date, label == 0, surface, p1, series)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # MODE PRODUCTION
    # ------------------------------------------------------------------

    def build_single(
        self,
        player1: str,
        player2: str,
        surface: str,
        series: str,
        round_name: str,
        best_of: int,
        match_date: datetime,
        rank1: float,
        rank2: float,
        pts1: float,
        pts2: float,
        elo_features: Dict[str, float],
        recent_history: pd.DataFrame,
        tournament_name: str = "",
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Mode production : construit le vecteur pour un match unique.
        recent_history = DataFrame des matchs récents (12 mois).
        """
        before = pd.Timestamp(match_date)
        ref_date = before.date()

        # Reconstruit des PlayerState depuis recent_history
        s1 = _build_state_from_history(player1, recent_history, before)
        s2 = _build_state_from_history(player2, recent_history, before)

        f3_1, f3_2   = s1.form(3),  s2.form(3)
        f5_1, f5_2   = s1.form(5),  s2.form(5)
        f10_1, f10_2 = s1.form(10), s2.form(10)
        f20_1, f20_2 = s1.form(20), s2.form(20)
        sw3_1  = s1.surf_wr(surface, 90,  ref_date)
        sw3_2  = s2.surf_wr(surface, 90,  ref_date)
        sw6_1  = s1.surf_wr(surface, 180, ref_date)
        sw6_2  = s2.surf_wr(surface, 180, ref_date)
        sw12_1 = s1.surf_wr(surface, 365, ref_date)
        sw12_2 = s2.surf_wr(surface, 365, ref_date)
        h2h_wr, h2h_total = s1.h2h_stats(player2)
        h2h_surf_wr = s1.h2h_surf_stats(player2, surface)
        fat1, fat2 = s1.fatigue(ref_date), s2.fatigue(ref_date)
        rest1, rest2 = s1.days_rest(ref_date), s2.days_rest(ref_date)
        spec1 = min(s1.surf_specialist(surface, ref_date), 3.0)
        spec2 = min(s2.surf_specialist(surface, ref_date), 3.0)

        r1 = safe_rank(rank1)
        r2 = safe_rank(rank2)
        pts1 = max(0.0, float(pts1 or 0))
        pts2 = max(0.0, float(pts2 or 0))

        elo_p1 = elo_features.get("elo_p1", 1500.0)
        elo_p2 = elo_features.get("elo_p2", 1500.0)
        surf_p1 = elo_features.get("surf_elo_p1", 1500.0)
        surf_p2 = elo_features.get("surf_elo_p2", 1500.0)
        mom_p1 = elo_features.get("momentum_elo_p1", 1500.0)
        mom_p2 = elo_features.get("momentum_elo_p2", 1500.0)

        is_h, is_c, is_g = encode_surface(surface)
        indoor = is_indoor(surface, tournament_name)
        best5 = 1 if best_of == 5 else 0
        round_num = ROUND_MAP.get(round_name, 3)
        series_num = SERIES_MAP.get(series, 3)
        m_sin, m_cos = encode_month_cyclic(match_date.month)

        features = {
            "elo_diff": elo_p1 - elo_p2, "elo_p1": elo_p1, "elo_p2": elo_p2,
            "surf_elo_diff": surf_p1 - surf_p2, "surf_elo_p1": surf_p1, "surf_elo_p2": surf_p2,
            "momentum_elo_diff": mom_p1 - mom_p2,
            "p1_form_3": f3_1, "p2_form_3": f3_2,
            "p1_form_5": f5_1, "p2_form_5": f5_2,
            "p1_form_10": f10_1, "p2_form_10": f10_2,
            "p1_form_20": f20_1, "p2_form_20": f20_2,
            "p1_momentum": f5_1 - f20_1, "p2_momentum": f5_2 - f20_2,
            "p1_surf_wr_3m": sw3_1, "p2_surf_wr_3m": sw3_2,
            "p1_surf_wr_6m": sw6_1, "p2_surf_wr_6m": sw6_2,
            "p1_surf_wr_12m": sw12_1, "p2_surf_wr_12m": sw12_2,
            "h2h_p1_wr": h2h_wr, "h2h_total": h2h_total, "h2h_surf_p1_wr": h2h_surf_wr,
            "p1_fatigue": fat1, "p2_fatigue": fat2,
            "p1_days_rest": rest1, "p2_days_rest": rest2,
            "rank_diff": r1 - r2, "rank_ratio": r1 / r2 if r2 > 0 else 1.0,
            "pts_diff": pts1 - pts2,
            "log_rank_diff": math.log(r1 + 1) - math.log(r2 + 1),
            "is_hard": is_h, "is_clay": is_c, "is_grass": is_g,
            "is_indoor": indoor, "best_of_5": best5,
            "round_num": round_num, "series_num": series_num,
            "month_sin": m_sin, "month_cos": m_cos,
            "p1_surf_specialist": spec1, "p2_surf_specialist": spec2,
            "p1_slam_experience": elo_features.get("p1_slam_exp", 0),
            "p2_slam_experience": elo_features.get("p2_slam_exp", 0),
        }

        vector = np.array([features[col] for col in self.feature_cols], dtype=np.float32)
        return vector.reshape(1, -1), features

    def get_feature_names(self) -> List[str]:
        return self.feature_cols


# ---------------------------------------------------------------------------
# Helper pour build_single : reconstruit le PlayerState depuis un DataFrame
# ---------------------------------------------------------------------------

def _build_state_from_history(
    player: str,
    history: pd.DataFrame,
    before: pd.Timestamp,
) -> PlayerState:
    """Reconstruit l'état d'un joueur depuis l'historique récent."""
    state = PlayerState()
    if history.empty:
        return state

    mask = (
        ((history["Player_1"] == player) | (history["Player_2"] == player))
        & (history["Date"] < before)
    )
    player_matches = history[mask].sort_values("Date")

    for _, row in player_matches.iterrows():
        p1 = str(row["Player_1"])
        p2 = str(row["Player_2"])
        label = int(row.get("label", row.get("Winner", p1) == p1))
        won = (p1 == player and label == 1) or (p2 == player and label == 0)
        opponent = p2 if p1 == player else p1
        surface = str(row.get("Surface", "Hard"))
        series = str(row.get("Series", "ATP250"))
        match_date = pd.to_datetime(row["Date"]).date()
        state.update(match_date, won, surface, opponent, series)

    return state
