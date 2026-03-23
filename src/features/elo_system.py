"""
Advanced Multi-Variant Elo System for Tennis
============================================
Implémente plusieurs variantes d'Elo pour maximiser la valeur prédictive :

  1. Global Elo          — performance générale
  2. Surface Elo         — Hard / Clay / Grass / Carpet
  3. Momentum Elo        — K-factor dynamique selon forme récente
  4. Decay Elo           — Atténuation temporelle en cas d'inactivité
  5. Tournament-weighted — Poids selon prestige du tournoi

Toutes les variantes sont calculées en une seule passe chronologique.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SURFACES = ("Hard", "Clay", "Grass", "Carpet")

TOURNAMENT_WEIGHTS = {
    "Grand Slam": 1.5,
    "Masters Cup": 1.4,
    "Masters 1000": 1.2,
    "ATP500": 1.1,
    "ATP250": 1.0,
    "International Gold": 1.0,
    "International": 0.9,
    "WTA Premier Mandatory": 1.3,
    "WTA Premier 5": 1.2,
    "WTA Premier": 1.1,
    "WTA International": 0.9,
}

SERIES_MAP = {
    "Grand Slam": 7,
    "Masters Cup": 6,
    "Masters 1000": 5,
    "ATP500": 4,
    "ATP250": 3,
    "International Gold": 2,
    "International": 2,
    "WTA Premier Mandatory": 5,
    "WTA Premier 5": 4,
    "WTA Premier": 3,
    "WTA International": 2,
}

ROUND_MAP = {
    "1st Round": 1, "2nd Round": 2, "3rd Round": 3, "4th Round": 4,
    "Quarterfinals": 5, "Semifinals": 6, "The Final": 7,
    "Round Robin": 4, "Bronze Medal": 5,
}


# ---------------------------------------------------------------------------
# Player Rating Store
# ---------------------------------------------------------------------------
@dataclass
class PlayerRatings:
    """Stocke toutes les variantes d'Elo pour un joueur."""

    global_elo: float = 1500.0
    surface_elo: Dict[str, float] = field(default_factory=lambda: {s: 1500.0 for s in SURFACES})
    momentum_elo: float = 1500.0

    # Suivi historique pour decay et K dynamique
    last_match_date: Optional[date] = None
    matches_played: int = 0
    recent_results: list = field(default_factory=list)  # 1=W, 0=L, derniers 10

    # Stats de carrière pour features
    total_wins: int = 0
    total_losses: int = 0
    slam_wins: int = 0        # Victoires en Grand Chelem (expérience pression)
    slam_matches: int = 0

    def win_rate(self) -> float:
        total = self.total_wins + self.total_losses
        return self.total_wins / total if total > 0 else 0.5

    def recent_form(self, n: int = 5) -> float:
        """Win rate sur les n derniers matchs."""
        recent = self.recent_results[-n:] if len(self.recent_results) >= n else self.recent_results
        return sum(recent) / len(recent) if recent else 0.5

    def add_result(self, won: bool) -> None:
        self.recent_results.append(1 if won else 0)
        if len(self.recent_results) > 20:
            self.recent_results.pop(0)
        if won:
            self.total_wins += 1
        else:
            self.total_losses += 1
        self.matches_played += 1


# ---------------------------------------------------------------------------
# Core Elo math
# ---------------------------------------------------------------------------

def expected_score(rating_a: float, rating_b: float) -> float:
    """Probabilité attendue de victoire de A contre B."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def dynamic_k_factor(
    base_k: float,
    matches_played: int,
    recent_form: float,
    tournament_weight: float = 1.0,
) -> float:
    """
    K-factor dynamique :
    - Réduit après 100 matchs (joueur établi)
    - Boosté si forme récente forte (momentum)
    - Pondéré par prestige tournoi
    """
    # Facteur d'expérience : diminue de 30% après 100 matchs
    experience_factor = 1.0 if matches_played < 30 else (0.85 if matches_played < 100 else 0.70)

    # Facteur momentum : K plus élevé si le joueur est en forme
    momentum_factor = 0.85 + 0.30 * recent_form  # [0.85, 1.15]

    return base_k * experience_factor * momentum_factor * tournament_weight


def decay_rating(
    rating: float,
    last_date: Optional[date],
    current_date: date,
    initial: float = 1500.0,
    half_life_days: int = 180,
    min_factor: float = 0.85,
) -> float:
    """
    Atténue le rating vers 1500 après une longue période d'inactivité.
    decay_factor ∈ [min_factor, 1.0]
    """
    if last_date is None:
        return rating

    days_inactive = (current_date - last_date).days
    if days_inactive <= 30:
        return rating  # Pas de decay si actif dans le dernier mois

    # Decay exponentiel
    factor = 0.5 ** (days_inactive / half_life_days)
    factor = max(factor, min_factor)

    # Rating → initial * (1 - factor) + rating * factor
    decayed = initial * (1.0 - factor) + rating * factor
    return decayed


# ---------------------------------------------------------------------------
# Main Elo Engine
# ---------------------------------------------------------------------------

class TennisEloEngine:
    """
    Moteur Elo complet : calcule toutes les variantes en une passe chronologique.

    Usage :
        engine = TennisEloEngine()
        engine.fit(df)               # df trié chronologiquement
        ratings = engine.get_all_ratings()
        history = engine.get_history()
    """

    def __init__(
        self,
        k_global: float = 32.0,
        k_surface: float = 40.0,
        k_momentum: float = 24.0,
        initial_rating: float = 1500.0,
        half_life_days: int = 180,
        decay_enabled: bool = True,
        min_decay_factor: float = 0.85,
    ):
        self.k_global = k_global
        self.k_surface = k_surface
        self.k_momentum = k_momentum
        self.initial = initial_rating
        self.half_life = half_life_days
        self.decay_enabled = decay_enabled
        self.min_decay = min_decay_factor

        self._players: Dict[str, PlayerRatings] = {}
        # Historique : liste de dicts (une ligne par match, pour features look-ahead safe)
        self._history: list = []

    def _get_or_create(self, name: str) -> PlayerRatings:
        if name not in self._players:
            self._players[name] = PlayerRatings()
        return self._players[name]

    def _apply_decay(self, player: PlayerRatings, current_date: date) -> None:
        if not self.decay_enabled or player.last_match_date is None:
            return
        player.global_elo = decay_rating(
            player.global_elo, player.last_match_date, current_date,
            self.initial, self.half_life, self.min_decay,
        )
        player.momentum_elo = decay_rating(
            player.momentum_elo, player.last_match_date, current_date,
            self.initial, self.half_life, self.min_decay,
        )
        for surf in SURFACES:
            player.surface_elo[surf] = decay_rating(
                player.surface_elo[surf], player.last_match_date, current_date,
                self.initial, self.half_life, self.min_decay,
            )

    def _update_pair(
        self,
        winner: PlayerRatings,
        loser: PlayerRatings,
        surface: str,
        series: str,
        round_name: str,
        match_date: date,
    ) -> Dict[str, float]:
        """Met à jour toutes les variantes d'Elo et retourne les deltas."""

        t_weight = TOURNAMENT_WEIGHTS.get(series, 1.0)
        surf = surface if surface in SURFACES else "Hard"

        # --- Ratings AVANT mise à jour (pour features pré-match) ---
        pre = {
            "w_global": winner.global_elo,
            "l_global": loser.global_elo,
            "w_surf": winner.surface_elo[surf],
            "l_surf": loser.surface_elo[surf],
            "w_momentum": winner.momentum_elo,
            "l_momentum": loser.momentum_elo,
        }

        # --- Global Elo ---
        exp_w = expected_score(winner.global_elo, loser.global_elo)
        exp_l = 1.0 - exp_w
        k_w = dynamic_k_factor(self.k_global, winner.matches_played, winner.recent_form(), t_weight)
        k_l = dynamic_k_factor(self.k_global, loser.matches_played, loser.recent_form(), t_weight)
        delta_w_global = k_w * (1.0 - exp_w)
        delta_l_global = k_l * (0.0 - exp_l)
        winner.global_elo += delta_w_global
        loser.global_elo += delta_l_global

        # --- Surface Elo ---
        exp_w_s = expected_score(winner.surface_elo[surf], loser.surface_elo[surf])
        exp_l_s = 1.0 - exp_w_s
        ks_w = dynamic_k_factor(self.k_surface, winner.matches_played, winner.recent_form(), t_weight)
        ks_l = dynamic_k_factor(self.k_surface, loser.matches_played, loser.recent_form(), t_weight)
        delta_w_surf = ks_w * (1.0 - exp_w_s)
        delta_l_surf = ks_l * (0.0 - exp_l_s)
        winner.surface_elo[surf] += delta_w_surf
        loser.surface_elo[surf] += delta_l_surf

        # --- Momentum Elo (K dynamique plus fort) ---
        exp_w_m = expected_score(winner.momentum_elo, loser.momentum_elo)
        exp_l_m = 1.0 - exp_w_m
        km_w = dynamic_k_factor(self.k_momentum, winner.matches_played, winner.recent_form(3), t_weight)
        km_l = dynamic_k_factor(self.k_momentum, loser.matches_played, loser.recent_form(3), t_weight)
        winner.momentum_elo += km_w * (1.0 - exp_w_m)
        loser.momentum_elo += km_l * (0.0 - exp_l_m)

        # --- Résultats & stats ---
        winner.add_result(True)
        loser.add_result(False)

        # Expérience Grand Chelem
        if series == "Grand Slam":
            winner.slam_matches += 1
            loser.slam_matches += 1
            round_num = ROUND_MAP.get(round_name, 3)
            if round_num >= 5:  # QF, SF, Final
                winner.slam_wins += 1

        winner.last_match_date = match_date
        loser.last_match_date = match_date

        return pre

    def fit(self, df: pd.DataFrame, progress_callback=None) -> "TennisEloEngine":
        """
        Calcule les Elo sur le DataFrame trié chronologiquement.

        Colonnes requises : Date, Player_1, Player_2, Winner, Surface, Series, Round
        Colonnes optionnelles : Best_of, Rank_1, Rank_2, Pts_1, Pts_2

        Pour chaque match, les ratings PRÉ-MATCH sont stockés dans l'historique
        (garantit l'absence de data leakage).
        """
        df = df.copy()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.sort_values("Date").reset_index(drop=True)

        n = len(df)
        self._history = []

        for i, row in df.iterrows():
            if progress_callback and i % 5000 == 0:
                progress_callback(f"Elo: {i}/{n} matchs traités")

            match_date = row["Date"].date()
            p1 = str(row["Player_1"]).strip()
            p2 = str(row["Player_2"]).strip()
            winner_raw = str(row.get("Winner", p1)).strip()
            surface = str(row.get("Surface", "Hard")).strip()
            series = str(row.get("Series", "ATP250")).strip()
            round_name = str(row.get("Round", "1st Round")).strip()

            # Identifier gagnant / perdant
            if winner_raw == p1 or winner_raw == "1":
                winner_name, loser_name = p1, p2
            else:
                winner_name, loser_name = p2, p1

            winner = self._get_or_create(winner_name)
            loser = self._get_or_create(loser_name)

            # Apply temporal decay avant la mise à jour
            self._apply_decay(winner, match_date)
            self._apply_decay(loser, match_date)

            # Stocker les features PRÉ-MATCH (avant update)
            surf = surface if surface in SURFACES else "Hard"
            record = {
                "match_idx": i,
                "Date": row["Date"],
                "Player_1": p1,
                "Player_2": p2,
                "Winner": winner_name,
                "Surface": surface,
                "Series": series,
                "Round": round_name,
                # Elo pré-match
                "elo_p1": (winner.global_elo if winner_name == p1 else loser.global_elo),
                "elo_p2": (loser.global_elo if winner_name == p1 else winner.global_elo),
                "surf_elo_p1": (winner.surface_elo[surf] if winner_name == p1 else loser.surface_elo[surf]),
                "surf_elo_p2": (loser.surface_elo[surf] if winner_name == p1 else winner.surface_elo[surf]),
                "momentum_elo_p1": (winner.momentum_elo if winner_name == p1 else loser.momentum_elo),
                "momentum_elo_p2": (loser.momentum_elo if winner_name == p1 else winner.momentum_elo),
                # Forme récente
                "p1_form_5": (winner.recent_form(5) if winner_name == p1 else loser.recent_form(5)),
                "p2_form_5": (loser.recent_form(5) if winner_name == p1 else winner.recent_form(5)),
                "p1_form_10": (winner.recent_form(10) if winner_name == p1 else loser.recent_form(10)),
                "p2_form_10": (loser.recent_form(10) if winner_name == p1 else winner.recent_form(10)),
                "p1_form_20": (winner.recent_form(20) if winner_name == p1 else loser.recent_form(20)),
                "p2_form_20": (loser.recent_form(20) if winner_name == p1 else winner.recent_form(20)),
                # Matches joués
                "p1_matches": (winner.matches_played if winner_name == p1 else loser.matches_played),
                "p2_matches": (loser.matches_played if winner_name == p1 else winner.matches_played),
                # Expérience Grand Chelem
                "p1_slam_exp": (winner.slam_matches if winner_name == p1 else loser.slam_matches),
                "p2_slam_exp": (loser.slam_matches if winner_name == p1 else winner.slam_matches),
                # Label
                "label": 1 if winner_name == p1 else 0,
            }
            self._history.append(record)

            # Mettre à jour les ratings
            self._update_pair(winner, loser, surface, series, round_name, match_date)

        return self

    def get_history(self) -> pd.DataFrame:
        """Retourne l'historique des matchs avec Elo pré-match (sans data leakage)."""
        return pd.DataFrame(self._history)

    def get_all_ratings(self) -> Dict[str, PlayerRatings]:
        """Ratings courants (après le dernier match connu)."""
        return self._players

    def get_player_snapshot(self, player_name: str) -> Optional[PlayerRatings]:
        return self._players.get(player_name)

    def get_ratings_dataframe(self) -> pd.DataFrame:
        """
        Retourne un DataFrame avec les ratings courants de tous les joueurs.
        Utile pour affichage classements.
        """
        rows = []
        for name, r in self._players.items():
            rows.append({
                "player": name,
                "global_elo": round(r.global_elo, 1),
                "hard_elo": round(r.surface_elo["Hard"], 1),
                "clay_elo": round(r.surface_elo["Clay"], 1),
                "grass_elo": round(r.surface_elo["Grass"], 1),
                "momentum_elo": round(r.momentum_elo, 1),
                "total_wins": r.total_wins,
                "total_losses": r.total_losses,
                "matches_played": r.matches_played,
                "slam_matches": r.slam_matches,
                "last_match": r.last_match_date,
            })
        return pd.DataFrame(rows).sort_values("global_elo", ascending=False).reset_index(drop=True)

    def get_matchup_features(
        self,
        player1: str,
        player2: str,
        surface: str = "Hard",
    ) -> Dict[str, float]:
        """
        Retourne les features Elo pour un matchup en production
        (sans modifier les ratings).
        """
        r1 = self._get_or_create(player1)
        r2 = self._get_or_create(player2)
        surf = surface if surface in SURFACES else "Hard"

        return {
            "elo_p1": r1.global_elo,
            "elo_p2": r2.global_elo,
            "elo_diff": r1.global_elo - r2.global_elo,
            "surf_elo_p1": r1.surface_elo[surf],
            "surf_elo_p2": r2.surface_elo[surf],
            "surf_elo_diff": r1.surface_elo[surf] - r2.surface_elo[surf],
            "momentum_elo_p1": r1.momentum_elo,
            "momentum_elo_p2": r2.momentum_elo,
            "momentum_elo_diff": r1.momentum_elo - r2.momentum_elo,
            "p1_form_5": r1.recent_form(5),
            "p2_form_5": r2.recent_form(5),
            "p1_form_10": r1.recent_form(10),
            "p2_form_10": r2.recent_form(10),
            "p1_slam_exp": r1.slam_matches,
            "p2_slam_exp": r2.slam_matches,
        }

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str) -> "TennisEloEngine":
        import joblib
        return joblib.load(path)
