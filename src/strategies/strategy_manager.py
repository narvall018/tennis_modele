"""
Multi-Strategy Manager — Haute Fréquence
==========================================
Gère un portefeuille de stratégies avec des profils risque/rendement différents.

Philosophie :
  - Chaque stratégie a son propre filtre (seuil, tournoi, surface, round)
  - La mise est calculée via Kelly fractionné ou flat (configurable)
  - Le gestionnaire de portefeuille applique des limites globales
    (exposition journalière, drawdown, concentration)

Stratégies disponibles (volume croissant, ROI décroissant) :
  1. ultra_confidence  → ~80-120 paris/an | ROI cible ≥ 10%
  2. standard_volume   → ~250-350 paris/an | ROI cible ≥ 6%
  3. high_volume       → ~500-800 paris/an | ROI cible ≥ 4%
  4. clay_specialist   → N/A (sous-ensemble clay)
  5. upset_hunter      → désactivé par défaut

Métriques de performance suivies :
  - ROI, win rate, moyenne des cotes
  - CLV (Closing Line Value) si cotes de clôture disponibles
  - Sharpe ratio, max drawdown
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class BetRecommendation:
    """Recommandation de pari issue d'une stratégie."""
    strategy_name: str
    player: str          # Joueur sur lequel parier
    opponent: str
    surface: str
    series: str
    round_name: str
    tournament: str

    model_prob: float    # Probabilité modèle (calibrée)
    market_prob: float   # Probabilité implicite des cotes (1/odds)
    odds: float          # Cote décimale
    edge: float          # model_prob - market_prob
    ev: float            # Expected Value = model_prob * odds - 1
    kelly_raw: float     # Kelly brut

    stake_pct: float     # % du bankroll à miser
    stake_amount: float  # Montant en €
    confidence: str      # "high" | "medium" | "low"

    match_date: Optional[date] = None
    bookmaker: str = ""
    notes: str = ""

    def __post_init__(self):
        if self.ev > 0.08:
            self.confidence = "high"
        elif self.ev > 0.03:
            self.confidence = "medium"
        else:
            self.confidence = "low"


@dataclass
class StrategyPerformance:
    """Suivi des performances d'une stratégie."""
    strategy_name: str
    n_bets: int = 0
    n_wins: int = 0
    total_staked: float = 0.0
    total_profit: float = 0.0
    max_drawdown: float = 0.0
    peak_balance: float = 0.0
    _balance_series: list = field(default_factory=list)
    _clv_values: list = field(default_factory=list)

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_bets if self.n_bets > 0 else 0.0

    @property
    def roi(self) -> float:
        return self.total_profit / self.total_staked if self.total_staked > 0 else 0.0

    @property
    def avg_clv(self) -> float:
        return np.mean(self._clv_values) if self._clv_values else 0.0

    def record_bet(self, stake: float, profit: float, clv: Optional[float] = None):
        self.n_bets += 1
        self.total_staked += stake
        self.total_profit += profit
        if profit > 0:
            self.n_wins += 1

        # Drawdown tracking
        cumulative = sum(self._balance_series) + profit if self._balance_series else profit
        self._balance_series.append(profit)
        peak = max(self._balance_series) if self._balance_series else profit
        dd = (peak - cumulative) / abs(peak) if peak != 0 else 0
        self.max_drawdown = max(self.max_drawdown, dd)

        if clv is not None:
            self._clv_values.append(clv)

    def summary(self) -> dict:
        return {
            "strategy": self.strategy_name,
            "n_bets": self.n_bets,
            "win_rate": f"{self.win_rate:.1%}",
            "roi": f"{self.roi:.2%}",
            "total_profit": f"{self.total_profit:+.2f}€",
            "max_drawdown": f"{self.max_drawdown:.1%}",
            "avg_clv": f"{self.avg_clv:+.3f}",
        }


# ---------------------------------------------------------------------------
# Kelly Criterion
# ---------------------------------------------------------------------------

def kelly_stake(
    prob: float,
    odds: float,
    fraction: float = 4.0,
    min_pct: float = 0.01,
    max_pct: float = 0.05,
) -> Tuple[float, float]:
    """
    Calcule le Kelly fractionné.
    Retourne (stake_pct, kelly_raw).

    kelly_raw = (p * b - q) / b
    où b = odds - 1, p = prob modèle, q = 1 - p
    """
    b = odds - 1.0
    if b <= 0 or prob <= 0 or prob >= 1:
        return min_pct, 0.0

    q = 1.0 - prob
    kelly_raw = (prob * b - q) / b

    if kelly_raw <= 0:
        return 0.0, kelly_raw  # EV négatif

    kelly_adj = kelly_raw / fraction
    stake_pct = max(min_pct, min(kelly_adj, max_pct))
    return stake_pct, kelly_raw


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

class Strategy:
    """
    Définit un filtre de sélection de paris et une règle de mise.
    """

    def __init__(self, name: str, config: dict):
        self.name = name
        self.enabled = config.get("enabled", True)

        # Filtres
        self.model_threshold = config.get("model_threshold", 0.60)
        self.min_edge = config.get("min_edge", 0.02)
        self.min_ev = config.get("min_ev", 0.01)
        self.min_odds = config.get("min_odds", 1.20)
        self.max_odds = config.get("max_odds", 6.00)

        # Filtres optionnels
        self.series_filter: List[str] = config.get("series_filter", [])
        self.surface_filter: List[str] = config.get("surface_filter", [])
        self.round_filter: List[str] = config.get("round_filter", [])

        # Money management
        self.stake_type = config.get("stake_type", "kelly_sixth")
        self.kelly_fraction = config.get("kelly_fraction", 6.0)
        self.flat_pct = config.get("flat_pct", 0.02)
        self.max_stake_pct = config.get("max_stake_pct", 0.04)
        self.min_stake_pct = config.get("min_stake_pct", 0.01)

        self.performance = StrategyPerformance(strategy_name=name)

    def is_eligible(
        self,
        model_prob: float,
        odds: float,
        surface: str,
        series: str,
        round_name: str,
    ) -> Tuple[bool, str]:
        """
        Vérifie si un match satisfait tous les critères de la stratégie.
        Retourne (eligible, reason).
        """
        if not self.enabled:
            return False, "Stratégie désactivée"

        market_prob = 1.0 / odds if odds > 0 else 1.0
        edge = model_prob - market_prob
        ev = model_prob * odds - 1.0

        if model_prob < self.model_threshold:
            return False, f"Prob {model_prob:.1%} < seuil {self.model_threshold:.1%}"

        if edge < self.min_edge:
            return False, f"Edge {edge:.1%} < min {self.min_edge:.1%}"

        if ev < self.min_ev:
            return False, f"EV {ev:.1%} < min {self.min_ev:.1%}"

        if odds < self.min_odds:
            return False, f"Cote {odds:.2f} < min {self.min_odds:.2f}"

        if odds > self.max_odds:
            return False, f"Cote {odds:.2f} > max {self.max_odds:.2f}"

        if self.series_filter and series not in self.series_filter:
            return False, f"Tournoi '{series}' non inclus dans le filtre"

        if self.surface_filter and surface not in self.surface_filter:
            return False, f"Surface '{surface}' non incluse dans le filtre"

        if self.round_filter and round_name not in self.round_filter:
            return False, f"Round '{round_name}' non inclus dans le filtre"

        return True, "OK"

    def compute_stake(self, model_prob: float, odds: float, bankroll: float) -> Tuple[float, float]:
        """
        Calcule la mise selon la règle de la stratégie.
        Retourne (stake_amount, stake_pct).
        """
        if self.stake_type == "flat":
            pct = self.flat_pct
            return bankroll * pct, pct

        # Kelly variants
        pct, kelly_raw = kelly_stake(
            prob=model_prob,
            odds=odds,
            fraction=self.kelly_fraction,
            min_pct=self.min_stake_pct,
            max_pct=self.max_stake_pct,
        )

        if kelly_raw <= 0:
            return 0.0, 0.0

        return bankroll * pct, pct


# ---------------------------------------------------------------------------
# Portfolio Manager
# ---------------------------------------------------------------------------

class PortfolioManager:
    """
    Contrôle l'exposition globale du portefeuille de paris.

    Règles :
      - Exposition journalière max (% bankroll)
      - Nombre max de paris simultanés
      - Concentration max sur un tournoi
      - Protection contre le drawdown
    """

    def __init__(self, config: dict = None):
        cfg = config or {}
        self.max_daily_exposure = cfg.get("max_daily_exposure", 0.10)
        self.max_concurrent_bets = cfg.get("max_concurrent_bets", 8)
        self.max_single_tournament = cfg.get("max_single_tournament", 0.08)
        self.drawdown_limit = cfg.get("drawdown_limit", 0.20)

        self._open_bets: List[dict] = []
        self._daily_exposure: Dict[str, float] = {}  # date → % exposé
        self._peak_bankroll: float = 0.0

    def can_place_bet(
        self,
        bankroll: float,
        stake_pct: float,
        tournament: str,
        match_date: date,
    ) -> Tuple[bool, str]:
        """Vérifie si le pari respecte les limites du portefeuille."""

        # Drawdown protection
        if self._peak_bankroll > 0:
            dd = (self._peak_bankroll - bankroll) / self._peak_bankroll
            if dd >= self.drawdown_limit:
                return False, f"Drawdown {dd:.1%} atteint la limite {self.drawdown_limit:.1%}"

        self._peak_bankroll = max(self._peak_bankroll, bankroll)

        # Exposition journalière
        day_key = str(match_date)
        daily = self._daily_exposure.get(day_key, 0.0)
        if daily + stake_pct > self.max_daily_exposure:
            return False, (
                f"Exposition jour {daily + stake_pct:.1%} > max {self.max_daily_exposure:.1%}"
            )

        # Paris simultanés
        open_count = len(self._open_bets)
        if open_count >= self.max_concurrent_bets:
            return False, f"Trop de paris ouverts ({open_count}/{self.max_concurrent_bets})"

        # Concentration tournoi
        tournament_exposure = sum(
            b["stake_pct"] for b in self._open_bets if b["tournament"] == tournament
        )
        if tournament_exposure + stake_pct > self.max_single_tournament:
            return False, (
                f"Concentration tournoi {tournament_exposure + stake_pct:.1%} > "
                f"max {self.max_single_tournament:.1%}"
            )

        return True, "OK"

    def register_bet(self, stake_pct: float, tournament: str, match_date: date, bet_id: str):
        day_key = str(match_date)
        self._daily_exposure[day_key] = self._daily_exposure.get(day_key, 0.0) + stake_pct
        self._open_bets.append({
            "bet_id": bet_id,
            "stake_pct": stake_pct,
            "tournament": tournament,
            "date": match_date,
        })

    def close_bet(self, bet_id: str):
        self._open_bets = [b for b in self._open_bets if b["bet_id"] != bet_id]

    def get_exposure_summary(self) -> dict:
        return {
            "open_bets": len(self._open_bets),
            "peak_bankroll": self._peak_bankroll,
            "daily_exposure": self._daily_exposure,
        }


# ---------------------------------------------------------------------------
# Strategy Manager (orchestrateur principal)
# ---------------------------------------------------------------------------

class StrategyManager:
    """
    Orchestrateur de toutes les stratégies.

    Usage typique :
        manager = StrategyManager.from_config(config["strategies"])

        # Pour un match donné :
        recommendations = manager.evaluate_match(
            player1="Djokovic N.",
            player2="Nadal R.",
            model_prob_p1=0.68,
            odds_p1=1.75,
            odds_p2=2.10,
            surface="Clay",
            series="Grand Slam",
            round_name="Semifinals",
            tournament="Roland Garros",
            bankroll=1000.0,
        )
    """

    def __init__(self, strategies: Dict[str, Strategy], portfolio: PortfolioManager = None):
        self.strategies = strategies
        self.portfolio = portfolio or PortfolioManager()

    @classmethod
    def from_config(cls, config: dict) -> "StrategyManager":
        """Construit le manager depuis la config YAML."""
        portfolio_cfg = config.get("portfolio", {})
        portfolio = PortfolioManager(portfolio_cfg)

        strategies = {}
        for key, strat_cfg in config.items():
            if key == "portfolio":
                continue
            if isinstance(strat_cfg, dict) and "model_threshold" in strat_cfg:
                strategies[key] = Strategy(
                    name=strat_cfg.get("name", key),
                    config=strat_cfg,
                )

        return cls(strategies=strategies, portfolio=portfolio)

    def evaluate_match(
        self,
        player1: str,
        player2: str,
        model_prob_p1: float,
        odds_p1: float,
        odds_p2: float,
        surface: str,
        series: str,
        round_name: str,
        tournament: str,
        bankroll: float,
        match_date: Optional[date] = None,
        bookmaker: str = "",
    ) -> List[BetRecommendation]:
        """
        Évalue un match contre toutes les stratégies actives.
        Retourne la liste des recommandations (peut être vide).

        Les deux côtés sont évalués (P1 et P2).
        """
        recommendations = []
        match_date = match_date or date.today()

        # Évaluer les deux côtés du pari
        candidates = [
            (player1, player2, model_prob_p1, odds_p1),
            (player2, player1, 1.0 - model_prob_p1, odds_p2),
        ]

        for player, opponent, prob, odds in candidates:
            if odds <= 1.0:
                continue

            market_prob = 1.0 / odds
            edge = prob - market_prob
            ev = prob * odds - 1.0
            kelly_raw_val = max(0.0, (prob * (odds - 1) - (1 - prob)) / (odds - 1)) if odds > 1 else 0.0

            for strat_key, strategy in self.strategies.items():
                eligible, reason = strategy.is_eligible(prob, odds, surface, series, round_name)
                if not eligible:
                    continue

                stake_amount, stake_pct = strategy.compute_stake(prob, odds, bankroll)
                if stake_amount <= 0:
                    continue

                # Vérification portefeuille
                can_bet, port_reason = self.portfolio.can_place_bet(
                    bankroll, stake_pct, tournament, match_date
                )
                if not can_bet:
                    continue

                rec = BetRecommendation(
                    strategy_name=strategy.name,
                    player=player,
                    opponent=opponent,
                    surface=surface,
                    series=series,
                    round_name=round_name,
                    tournament=tournament,
                    model_prob=prob,
                    market_prob=market_prob,
                    odds=odds,
                    edge=edge,
                    ev=ev,
                    kelly_raw=kelly_raw_val,
                    stake_pct=stake_pct,
                    stake_amount=stake_amount,
                    confidence="medium",   # sera recalculé dans __post_init__
                    match_date=match_date,
                    bookmaker=bookmaker,
                )
                recommendations.append(rec)

        # Dédupliquer si plusieurs stratégies choisissent le même joueur
        # (garder celle avec le stake le plus élevé)
        seen: Dict[str, BetRecommendation] = {}
        for rec in recommendations:
            key = f"{rec.player}_{rec.tournament}"
            if key not in seen or rec.stake_amount > seen[key].stake_amount:
                seen[key] = rec

        return list(seen.values())

    def evaluate_batch(
        self,
        matches_df: pd.DataFrame,
        model_probs: np.ndarray,
        bankroll: float,
    ) -> pd.DataFrame:
        """
        Évalue un batch de matchs (pour backtesting).

        matches_df : colonnes attendues = Player_1, Player_2, Surface, Series, Round,
                     Tournament, Date, odds_p1, odds_p2
        model_probs : proba P1 pour chaque ligne, shape (n,)

        Retourne un DataFrame de recommandations.
        """
        all_recs = []
        for i, (row_idx, row) in enumerate(matches_df.iterrows()):
            prob = float(model_probs[i])
            recs = self.evaluate_match(
                player1=str(row.get("Player_1", "")),
                player2=str(row.get("Player_2", "")),
                model_prob_p1=prob,
                odds_p1=float(row.get("odds_p1", 0)),
                odds_p2=float(row.get("odds_p2", 0)),
                surface=str(row.get("Surface", "Hard")),
                series=str(row.get("Series", "ATP250")),
                round_name=str(row.get("Round", "1st Round")),
                tournament=str(row.get("Tournament", "")),
                bankroll=bankroll,
                match_date=pd.to_datetime(row["Date"]).date() if "Date" in row else None,
                bookmaker=str(row.get("bookmaker", "")),
            )
            for rec in recs:
                all_recs.append({
                    "date": rec.match_date,
                    "tournament": rec.tournament,
                    "surface": rec.surface,
                    "series": rec.series,
                    "round": rec.round_name,
                    "strategy": rec.strategy_name,
                    "player": rec.player,
                    "opponent": rec.opponent,
                    "model_prob": rec.model_prob,
                    "market_prob": rec.market_prob,
                    "odds": rec.odds,
                    "edge": rec.edge,
                    "ev": rec.ev,
                    "kelly_raw": rec.kelly_raw,
                    "stake_pct": rec.stake_pct,
                    "stake_amount": rec.stake_amount,
                    "confidence": rec.confidence,
                    "_row_idx": row_idx,
                })

        return pd.DataFrame(all_recs) if all_recs else pd.DataFrame()

    def get_strategy_summary(self) -> pd.DataFrame:
        """Résumé des performances par stratégie."""
        rows = [s.performance.summary() for s in self.strategies.values()]
        return pd.DataFrame(rows)

    def reset_portfolio(self):
        """Remet à zéro le gestionnaire de portefeuille (pour backtesting)."""
        self.portfolio = PortfolioManager(config={
            "max_daily_exposure": self.portfolio.max_daily_exposure,
            "max_concurrent_bets": self.portfolio.max_concurrent_bets,
            "max_single_tournament": self.portfolio.max_single_tournament,
            "drawdown_limit": self.portfolio.drawdown_limit,
        })


# ---------------------------------------------------------------------------
# Stake display helper
# ---------------------------------------------------------------------------

def format_recommendation(rec: BetRecommendation) -> str:
    """Formate une recommandation pour affichage console/log."""
    icon = {"high": "🔥", "medium": "💎", "low": "📊"}.get(rec.confidence, "📊")
    return (
        f"{icon} [{rec.strategy_name}] "
        f"PARIE SUR {rec.player.upper()} @ {rec.odds:.2f} "
        f"| Prob: {rec.model_prob:.1%} | Edge: {rec.edge:+.1%} | EV: {rec.ev:+.1%} "
        f"| Mise: {rec.stake_amount:.1f}€ ({rec.stake_pct:.1%}) "
        f"| {rec.tournament} — {rec.round_name}"
    )
