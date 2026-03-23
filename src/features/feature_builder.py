"""
Feature Engineering Pipeline — v3
==================================
Construit un vecteur de 48 features pour chaque match à partir de :
  - Historique Elo (pré-match, sans data leakage)
  - Statistiques récentes (form, surface win rate, H2H)
  - Contexte du match (surface, tournoi, round, indoor/outdoor)
  - Encodages cycliques (mois de la saison)
  - Features de pression (expérience GS, fatigue)

Point critique : toutes les features sont calculées avec des données
ANTÉRIEURES à la date du match (look-ahead safe).
"""

from __future__ import annotations

import math
import warnings
from datetime import datetime, timedelta
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
    # Momentum (tendance : form récente - form ancienne)
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
# Rolling stats helpers (appliquées sur l'historique des matchs)
# ---------------------------------------------------------------------------

def _matches_for_player(
    player: str,
    history: pd.DataFrame,
    before_date: pd.Timestamp,
    days: Optional[int] = None,
    n_last: Optional[int] = None,
) -> pd.DataFrame:
    """
    Retourne les matchs d'un joueur AVANT before_date.
    Optionnellement filtre sur une fenêtre glissante (days) ou N derniers.
    """
    mask = (
        ((history["Player_1"] == player) | (history["Player_2"] == player))
        & (history["Date"] < before_date)
    )
    df = history[mask].sort_values("Date", ascending=False)

    if days is not None:
        cutoff = before_date - pd.Timedelta(days=days)
        df = df[df["Date"] >= cutoff]

    if n_last is not None:
        df = df.head(n_last)

    return df


def compute_form(player: str, history: pd.DataFrame, before_date: pd.Timestamp, n: int) -> float:
    """Win rate sur les n derniers matchs avant before_date."""
    matches = _matches_for_player(player, history, before_date, n_last=n)
    if matches.empty:
        return 0.5
    wins = ((matches["Player_1"] == player) & (matches["label"] == 1)).sum() + \
           ((matches["Player_2"] == player) & (matches["label"] == 0)).sum()
    return wins / len(matches)


def compute_surface_winrate(
    player: str,
    surface: str,
    history: pd.DataFrame,
    before_date: pd.Timestamp,
    months: int,
) -> float:
    """Win rate sur une surface donnée, dans la fenêtre [before_date - months*30, before_date)."""
    days = months * 30
    matches = _matches_for_player(player, history, before_date, days=days)
    surf_matches = matches[matches["Surface"] == surface]
    if surf_matches.empty:
        return 0.5
    wins = ((surf_matches["Player_1"] == player) & (surf_matches["label"] == 1)).sum() + \
           ((surf_matches["Player_2"] == player) & (surf_matches["label"] == 0)).sum()
    return wins / len(surf_matches)


def compute_h2h(
    player1: str,
    player2: str,
    history: pd.DataFrame,
    before_date: pd.Timestamp,
    surface: Optional[str] = None,
) -> Tuple[float, int]:
    """
    Retourne (win_rate_p1, total_matchs) des confrontations directes.
    Si surface fournie, filtre sur cette surface.
    """
    mask = (
        ((history["Player_1"] == player1) & (history["Player_2"] == player2)) |
        ((history["Player_1"] == player2) & (history["Player_2"] == player1))
    ) & (history["Date"] < before_date)

    h2h = history[mask]
    if surface:
        h2h = h2h[h2h["Surface"] == surface]

    if h2h.empty:
        return 0.5, 0

    p1_wins = (
        ((h2h["Player_1"] == player1) & (h2h["label"] == 1)).sum() +
        ((h2h["Player_2"] == player1) & (h2h["label"] == 0)).sum()
    )
    return p1_wins / len(h2h), len(h2h)


def compute_fatigue(
    player: str,
    history: pd.DataFrame,
    before_date: pd.Timestamp,
    window_days: int = 14,
) -> int:
    """Nombre de matchs joués dans les window_days avant before_date."""
    matches = _matches_for_player(player, history, before_date, days=window_days)
    return len(matches)


def compute_days_rest(
    player: str,
    history: pd.DataFrame,
    before_date: pd.Timestamp,
) -> int:
    """Nombre de jours depuis le dernier match (repos)."""
    matches = _matches_for_player(player, history, before_date)
    if matches.empty:
        return 30  # Valeur neutre si inconnu
    last_date = matches.iloc[0]["Date"]
    return (before_date - last_date).days


def compute_surface_specialist(
    player: str,
    surface: str,
    history: pd.DataFrame,
    before_date: pd.Timestamp,
    months: int = 24,
) -> float:
    """
    Ratio surface win rate (12m) / global win rate (12m).
    > 1.0 → spécialiste de la surface
    < 1.0 → moins à l'aise sur cette surface
    """
    surf_wr = compute_surface_winrate(player, surface, history, before_date, 12)
    matches = _matches_for_player(player, history, before_date, days=months * 30)
    if matches.empty:
        return 1.0
    wins = ((matches["Player_1"] == player) & (matches["label"] == 1)).sum() + \
           ((matches["Player_2"] == player) & (matches["label"] == 0)).sum()
    global_wr = wins / len(matches) if len(matches) > 0 else 0.5
    return surf_wr / global_wr if global_wr > 0 else 1.0


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
    """Encodage cyclique du mois (sin/cos) pour capturer la saisonnalité."""
    angle = 2 * math.pi * (month - 1) / 12
    return math.sin(angle), math.cos(angle)


def is_indoor(surface: str, tournament_name: str = "") -> int:
    """Heuristique : carpet = indoor, certains tournois connus indoor."""
    indoor_keywords = ["indoor", "arena", "salle", "covered", "carpet"]
    name_lower = tournament_name.lower()
    if surface == "Carpet":
        return 1
    for kw in indoor_keywords:
        if kw in name_lower:
            return 1
    return 0


# ---------------------------------------------------------------------------
# Main Feature Builder
# ---------------------------------------------------------------------------

class FeatureBuilder:
    """
    Construit le vecteur de features complet pour chaque match.

    Deux modes d'utilisation :
      1. build_dataset(elo_history, raw_df) → DataFrame avec toutes les features
         (pour entraînement/backtesting)
      2. build_single(player1, player2, match_info, elo_engine, recent_df) → array
         (pour prédiction en production)
    """

    def __init__(self, feature_cols: List[str] = None):
        self.feature_cols = feature_cols or FEATURE_COLS_V3

    # ------------------------------------------------------------------
    # MODE ENTRAÎNEMENT : construit tout le dataset en une passe
    # ------------------------------------------------------------------

    def build_dataset(
        self,
        elo_history: pd.DataFrame,
        progress_callback=None,
    ) -> pd.DataFrame:
        """
        elo_history = output de TennisEloEngine.get_history()
        Contient les Elo pré-match et le label.
        Cette méthode enrichit avec les features rolling (H2H, form, surface wr…).

        IMPORTANT : on utilise UNIQUEMENT les données avant la date du match
        pour chaque feature rolling → pas de data leakage.

        Optimisation : tri chronologique + index pour accélérer les lookups.
        """
        df = elo_history.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        n = len(df)
        records = []

        for i, row in df.iterrows():
            if progress_callback and i % 2000 == 0:
                progress_callback(f"Features: {i}/{n} matchs")

            before = row["Date"]
            p1, p2 = row["Player_1"], row["Player_2"]
            surface = str(row.get("Surface", "Hard"))
            series = str(row.get("Series", "ATP250"))
            round_name = str(row.get("Round", "1st Round"))
            match_date = row["Date"]

            # Historique disponible pour ce match (données antérieures)
            hist_before = df[df["Date"] < before]

            # --- Features rolling ---
            f3_p1 = compute_form(p1, hist_before, before, 3)
            f3_p2 = compute_form(p2, hist_before, before, 3)
            f5_p1 = compute_form(p1, hist_before, before, 5)
            f5_p2 = compute_form(p2, hist_before, before, 5)
            f10_p1 = compute_form(p1, hist_before, before, 10)
            f10_p2 = compute_form(p2, hist_before, before, 10)
            f20_p1 = compute_form(p1, hist_before, before, 20)
            f20_p2 = compute_form(p2, hist_before, before, 20)

            sw3_p1 = compute_surface_winrate(p1, surface, hist_before, before, 3)
            sw3_p2 = compute_surface_winrate(p2, surface, hist_before, before, 3)
            sw6_p1 = compute_surface_winrate(p1, surface, hist_before, before, 6)
            sw6_p2 = compute_surface_winrate(p2, surface, hist_before, before, 6)
            sw12_p1 = compute_surface_winrate(p1, surface, hist_before, before, 12)
            sw12_p2 = compute_surface_winrate(p2, surface, hist_before, before, 12)

            h2h_wr, h2h_total = compute_h2h(p1, p2, hist_before, before)
            h2h_surf_wr, _ = compute_h2h(p1, p2, hist_before, before, surface=surface)

            fat_p1 = compute_fatigue(p1, hist_before, before)
            fat_p2 = compute_fatigue(p2, hist_before, before)
            rest_p1 = compute_days_rest(p1, hist_before, before)
            rest_p2 = compute_days_rest(p2, hist_before, before)

            spec_p1 = compute_surface_specialist(p1, surface, hist_before, before)
            spec_p2 = compute_surface_specialist(p2, surface, hist_before, before)

            # --- Ranking (colonnes optionnelles) ---
            rank1 = float(row.get("Rank_1", 100) or 100)
            rank2 = float(row.get("Rank_2", 100) or 100)
            pts1 = float(row.get("Pts_1", 0) or 0)
            pts2 = float(row.get("Pts_2", 0) or 0)
            rank_diff = rank1 - rank2
            rank_ratio = rank1 / rank2 if rank2 > 0 else 1.0
            pts_diff = pts1 - pts2
            log_rank_diff = math.log(rank1 + 1) - math.log(rank2 + 1)

            # --- Context ---
            is_h, is_c, is_g = encode_surface(surface)
            indoor = is_indoor(surface, str(row.get("Tournament", "")))
            best5 = 1 if int(row.get("Best_of", 3) or 3) == 5 else 0
            round_num = ROUND_MAP.get(round_name, 3)
            series_num = SERIES_MAP.get(series, 3)

            month = match_date.month
            m_sin, m_cos = encode_month_cyclic(month)

            # --- Slam experience ---
            slam_exp_p1 = int(row.get("p1_slam_exp", 0) or 0)
            slam_exp_p2 = int(row.get("p2_slam_exp", 0) or 0)

            # --- Elo features (depuis elo_history) ---
            elo_p1 = float(row.get("elo_p1", 1500))
            elo_p2 = float(row.get("elo_p2", 1500))
            surf_elo_p1 = float(row.get("surf_elo_p1", 1500))
            surf_elo_p2 = float(row.get("surf_elo_p2", 1500))
            mom_elo_p1 = float(row.get("momentum_elo_p1", 1500))
            mom_elo_p2 = float(row.get("momentum_elo_p2", 1500))

            feature_dict = {
                # Elo
                "elo_diff": elo_p1 - elo_p2,
                "elo_p1": elo_p1,
                "elo_p2": elo_p2,
                "surf_elo_diff": surf_elo_p1 - surf_elo_p2,
                "surf_elo_p1": surf_elo_p1,
                "surf_elo_p2": surf_elo_p2,
                "momentum_elo_diff": mom_elo_p1 - mom_elo_p2,
                # Forme
                "p1_form_3": f3_p1,
                "p2_form_3": f3_p2,
                "p1_form_5": f5_p1,
                "p2_form_5": f5_p2,
                "p1_form_10": f10_p1,
                "p2_form_10": f10_p2,
                "p1_form_20": f20_p1,
                "p2_form_20": f20_p2,
                # Momentum (tendance)
                "p1_momentum": f5_p1 - f20_p1,
                "p2_momentum": f5_p2 - f20_p2,
                # Surface WR
                "p1_surf_wr_3m": sw3_p1,
                "p2_surf_wr_3m": sw3_p2,
                "p1_surf_wr_6m": sw6_p1,
                "p2_surf_wr_6m": sw6_p2,
                "p1_surf_wr_12m": sw12_p1,
                "p2_surf_wr_12m": sw12_p2,
                # H2H
                "h2h_p1_wr": h2h_wr,
                "h2h_total": h2h_total,
                "h2h_surf_p1_wr": h2h_surf_wr,
                # Fatigue
                "p1_fatigue": fat_p1,
                "p2_fatigue": fat_p2,
                "p1_days_rest": rest_p1,
                "p2_days_rest": rest_p2,
                # Ranking
                "rank_diff": rank_diff,
                "rank_ratio": rank_ratio,
                "pts_diff": pts_diff,
                "log_rank_diff": log_rank_diff,
                # Context
                "is_hard": is_h,
                "is_clay": is_c,
                "is_grass": is_g,
                "is_indoor": indoor,
                "best_of_5": best5,
                "round_num": round_num,
                "series_num": series_num,
                # Saison
                "month_sin": m_sin,
                "month_cos": m_cos,
                # Spécialiste surface
                "p1_surf_specialist": min(spec_p1, 3.0),  # cap outliers
                "p2_surf_specialist": min(spec_p2, 3.0),
                # Expérience pression
                "p1_slam_experience": slam_exp_p1,
                "p2_slam_experience": slam_exp_p2,
                # Metadata (non utilisé comme feature)
                "_date": match_date,
                "_p1": p1,
                "_p2": p2,
                "_surface": surface,
                "_series": series,
                "_round": round_name,
                "_label": int(row.get("label", 1)),
            }
            records.append(feature_dict)

        return pd.DataFrame(records)

    # ------------------------------------------------------------------
    # MODE PRODUCTION : construit un vecteur unique pour prédiction live
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
        elo_features: Dict[str, float],   # output de TennisEloEngine.get_matchup_features()
        recent_history: pd.DataFrame,      # matchs récents (12 mois)
        tournament_name: str = "",
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Construit le vecteur de features pour un match unique en production.
        Retourne (array, dict_features) pour logging et debug.
        """
        before = pd.Timestamp(match_date)

        f3_p1 = compute_form(player1, recent_history, before, 3)
        f3_p2 = compute_form(player2, recent_history, before, 3)
        f5_p1 = compute_form(player1, recent_history, before, 5)
        f5_p2 = compute_form(player2, recent_history, before, 5)
        f10_p1 = compute_form(player1, recent_history, before, 10)
        f10_p2 = compute_form(player2, recent_history, before, 10)
        f20_p1 = compute_form(player1, recent_history, before, 20)
        f20_p2 = compute_form(player2, recent_history, before, 20)

        sw3_p1 = compute_surface_winrate(player1, surface, recent_history, before, 3)
        sw3_p2 = compute_surface_winrate(player2, surface, recent_history, before, 3)
        sw6_p1 = compute_surface_winrate(player1, surface, recent_history, before, 6)
        sw6_p2 = compute_surface_winrate(player2, surface, recent_history, before, 6)
        sw12_p1 = compute_surface_winrate(player1, surface, recent_history, before, 12)
        sw12_p2 = compute_surface_winrate(player2, surface, recent_history, before, 12)

        h2h_wr, h2h_total = compute_h2h(player1, player2, recent_history, before)
        h2h_surf_wr, _ = compute_h2h(player1, player2, recent_history, before, surface=surface)

        fat_p1 = compute_fatigue(player1, recent_history, before)
        fat_p2 = compute_fatigue(player2, recent_history, before)
        rest_p1 = compute_days_rest(player1, recent_history, before)
        rest_p2 = compute_days_rest(player2, recent_history, before)

        spec_p1 = compute_surface_specialist(player1, surface, recent_history, before)
        spec_p2 = compute_surface_specialist(player2, surface, recent_history, before)

        rank_diff = rank1 - rank2
        rank_ratio = rank1 / rank2 if rank2 > 0 else 1.0
        pts_diff = pts1 - pts2
        log_rank_diff = math.log(rank1 + 1) - math.log(rank2 + 1)

        is_h, is_c, is_g = encode_surface(surface)
        indoor = is_indoor(surface, tournament_name)
        best5 = 1 if best_of == 5 else 0
        round_num = ROUND_MAP.get(round_name, 3)
        series_num = SERIES_MAP.get(series, 3)
        m_sin, m_cos = encode_month_cyclic(match_date.month)

        elo_p1 = elo_features.get("elo_p1", 1500.0)
        elo_p2 = elo_features.get("elo_p2", 1500.0)
        surf_elo_p1 = elo_features.get("surf_elo_p1", 1500.0)
        surf_elo_p2 = elo_features.get("surf_elo_p2", 1500.0)
        mom_p1 = elo_features.get("momentum_elo_p1", 1500.0)
        mom_p2 = elo_features.get("momentum_elo_p2", 1500.0)

        features = {
            "elo_diff": elo_p1 - elo_p2,
            "elo_p1": elo_p1,
            "elo_p2": elo_p2,
            "surf_elo_diff": surf_elo_p1 - surf_elo_p2,
            "surf_elo_p1": surf_elo_p1,
            "surf_elo_p2": surf_elo_p2,
            "momentum_elo_diff": mom_p1 - mom_p2,
            "p1_form_3": f3_p1,
            "p2_form_3": f3_p2,
            "p1_form_5": f5_p1,
            "p2_form_5": f5_p2,
            "p1_form_10": f10_p1,
            "p2_form_10": f10_p2,
            "p1_form_20": f20_p1,
            "p2_form_20": f20_p2,
            "p1_momentum": f5_p1 - f20_p1,
            "p2_momentum": f5_p2 - f20_p2,
            "p1_surf_wr_3m": sw3_p1,
            "p2_surf_wr_3m": sw3_p2,
            "p1_surf_wr_6m": sw6_p1,
            "p2_surf_wr_6m": sw6_p2,
            "p1_surf_wr_12m": sw12_p1,
            "p2_surf_wr_12m": sw12_p2,
            "h2h_p1_wr": h2h_wr,
            "h2h_total": h2h_total,
            "h2h_surf_p1_wr": h2h_surf_wr,
            "p1_fatigue": fat_p1,
            "p2_fatigue": fat_p2,
            "p1_days_rest": rest_p1,
            "p2_days_rest": rest_p2,
            "rank_diff": rank_diff,
            "rank_ratio": rank_ratio,
            "pts_diff": pts_diff,
            "log_rank_diff": log_rank_diff,
            "is_hard": is_h,
            "is_clay": is_c,
            "is_grass": is_g,
            "is_indoor": indoor,
            "best_of_5": best5,
            "round_num": round_num,
            "series_num": series_num,
            "month_sin": m_sin,
            "month_cos": m_cos,
            "p1_surf_specialist": min(spec_p1, 3.0),
            "p2_surf_specialist": min(spec_p2, 3.0),
            "p1_slam_experience": elo_features.get("p1_slam_exp", 0),
            "p2_slam_experience": elo_features.get("p2_slam_exp", 0),
        }

        vector = np.array([features[col] for col in self.feature_cols], dtype=np.float32)
        return vector.reshape(1, -1), features

    def get_feature_names(self) -> List[str]:
        return self.feature_cols

    def get_feature_importance_description(self) -> Dict[str, str]:
        """Documentation des features pour interprétabilité."""
        return {
            "elo_diff": "Différence Elo global (P1 - P2)",
            "surf_elo_diff": "Différence Elo surface-spécifique",
            "momentum_elo_diff": "Différence Elo momentum (K dynamique)",
            "p1_form_5": "Win rate P1 sur 5 derniers matchs",
            "p2_form_5": "Win rate P2 sur 5 derniers matchs",
            "p1_momentum": "Tendance forme P1 (form_5 - form_20)",
            "p2_momentum": "Tendance forme P2 (form_5 - form_20)",
            "p1_surf_wr_12m": "Win rate P1 sur surface (12 mois)",
            "h2h_p1_wr": "Win rate historique P1 contre P2",
            "h2h_surf_p1_wr": "Win rate H2H sur la surface du match",
            "p1_fatigue": "Matchs joués par P1 dans les 14 derniers jours",
            "p1_days_rest": "Jours de repos depuis dernier match P1",
            "log_rank_diff": "Différence log-ranking (non-linéaire)",
            "p1_surf_specialist": "Ratio perf surface / perf globale P1",
            "p1_slam_experience": "Nombre de matchs Grand Chelem joués par P1",
            "month_sin": "Encodage cyclique du mois (saisonnalité)",
            "is_indoor": "Match en salle (terrain couvert)",
        }
